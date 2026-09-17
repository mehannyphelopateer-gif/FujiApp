"""Bilateral-grid predictor - docs/phase4-scene-adaptive-scope.md SS4.

A small CNN reads a low-resolution (default 256x256) downsample of the
linear input and predicts a low-res grid over space x luma, each cell a
3x4 affine matrix (the same structure Phase 3's hand-fit spatial-tile/
luma-zone model already approximated - see docs/phase3-raw-aware-processor-plan.md
- but predicted by a network conditioned on real image content instead of
a handful of pre-specified scalars). Applying the grid to a full-resolution
image is a single trilinear lookup per pixel (standard HDRNet-style
"bilateral grid slicing", Gharbi et al. 2017) via torch's 5D grid_sample -
cheap, and the same op that has to survive ONNX export for browser
inference later.

Grid default: 4x4 spatial x 7 luma zones, matching SS4's suggested
starting point (Phase 3's own grid resolution) - revisit by experiment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Rec. 709 luma weights - consistent with how this project already treats
# luma elsewhere (not an arbitrary average of R,G,B).
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _conv_block(in_ch, out_ch, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.ReLU(inplace=True),
    )


def local_highpass(luma, kernel_size=9):
    """(B,H,W) -> (B,H,W): how much brighter each pixel is than its local
    neighborhood average, clamped to positive excursions only - a cheap,
    parameter-free "compact bright spot" detector, independent of the
    scene's overall exposure level (unlike a fixed near-white threshold,
    which would fire on nothing for a scene without a truly saturated
    highlight and everything for an overexposed one)."""
    x = luma.unsqueeze(1)  # (B,1,H,W)
    blurred = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)
    return (x - blurred).clamp(min=0).squeeze(1)


class BilateralGridPredictor(nn.Module):
    def __init__(self, grid_spatial=4, grid_luma=7, low_res=256, base_ch=16,
                 use_detail_branch=False, detail_ch=8):
        super().__init__()
        self.grid_spatial = grid_spatial
        self.grid_luma = grid_luma
        self.low_res = low_res
        self.use_detail_branch = use_detail_branch
        self.detail_ch = detail_ch
        coeffs_per_cell = 12  # 3x4 affine matrix (3 output channels, RGB+bias)

        self.encoder = nn.Sequential(
            _conv_block(3, base_ch),          # 256 -> 128
            _conv_block(base_ch, base_ch * 2),  # 128 -> 64
            _conv_block(base_ch * 2, base_ch * 4),  # 64 -> 32
            _conv_block(base_ch * 4, base_ch * 4),  # 32 -> 16
            _conv_block(base_ch * 4, base_ch * 4),  # 16 -> 8
        )

        head_in_ch = base_ch * 4
        if use_detail_branch:
            # 2026-09-16 targeted intervention: the main encoder's 5 stride-2
            # layers (256->8, a 32x reduction) discard the shape of a small
            # compact highlight long before the grid-coefficient head ever
            # sees it - the structural diagnosis found the 4 reproducible
            # regressions differ from matched controls specifically in
            # highlight-blob shape (fewer, larger, more compact/rounder),
            # not overall brightness. This shallow, separate branch keeps
            # only 2 stride-2 layers (256->64, a 4x reduction) on a
            # [luma, local_highpass] input, so far more of that shape
            # survives to the point where it gets pooled and fused into the
            # head - max-pooled (not averaged, unlike the main branch),
            # since a max operation naturally preserves "is there a compact
            # bright feature in this cell" instead of diluting it.
            self.detail_conv1 = _conv_block(2, detail_ch)              # 256 -> 128
            self.detail_conv2 = _conv_block(detail_ch, detail_ch)      # 128 -> 64
            head_in_ch += detail_ch

        self.head = nn.Conv2d(head_in_ch, grid_luma * coeffs_per_cell, kernel_size=1)

    def forward(self, low_res_input):
        """low_res_input: (B,3,low_res,low_res) linear RGB, already resized
        by the caller. Returns the grid as (B, 12, grid_luma, grid_spatial,
        grid_spatial) - a (C,D,H,W) volume ready for apply_bilateral_grid."""
        feat = self.encoder(low_res_input)
        feat = F.adaptive_avg_pool2d(feat, (self.grid_spatial, self.grid_spatial))

        if self.use_detail_branch:
            luma = compute_luma_guide(low_res_input, "rec709")  # (B,H,W)
            highpass = local_highpass(luma)
            detail_input = torch.stack([luma, highpass], dim=1)  # (B,2,H,W)
            detail_feat = self.detail_conv2(self.detail_conv1(detail_input))
            detail_feat = F.adaptive_max_pool2d(detail_feat, (self.grid_spatial, self.grid_spatial))
            feat = torch.cat([feat, detail_feat], dim=1)

        raw = self.head(feat)  # (B, grid_luma*12, gs, gs)
        b, _, gs_h, gs_w = raw.shape
        grid = raw.view(b, self.grid_luma, 12, gs_h, gs_w)
        grid = grid.permute(0, 2, 1, 3, 4).contiguous()  # (B,12,D=luma,H=gs,W=gs)
        return grid

    def predict_and_apply(self, full_res_linear, low_res_size=256, luma_mode="rec709"):
        """Resize -> predict grid -> apply. Returns (output, aux) with aux
        a dict - the uniform interface train.py's run_epoch calls
        regardless of whether the model is grid-only or HybridPredictor
        below (which also implements predict_and_apply with the same
        (output, aux) convention, aux carrying the extra grid_output/delta
        info)."""
        low_res_input = resize_for_network(full_res_linear, low_res_size)
        grid = self(low_res_input)
        output = apply_bilateral_grid(grid, full_res_linear, luma_mode=luma_mode)
        return output, {"grid": grid}


def compute_luma_guide(full_res_linear, luma_mode="rec709"):
    """The coordinate used to index the bilateral grid's luma (depth) axis
    - kept as its own function since 2026-09-15's diagnosis found the 5
    monitor regressions cluster on Auto-WB + bright-highlight scenes, and
    the natural first targeted intervention is this coordinate alone, not
    the model.

    "rec709" (default, matches every checkpoint before this change):
    standard luma from the as-shot-WB linear RGB - shifts with the
    camera's white balance, since as-shot R/B channels are WB-gain-scaled.

    "green_channel" (WB-stable guide): this project's own WB convention
    (src/lib/raw/rawService.ts) normalizes gains relative to green = 1.0 -
    only red/blue gains are ever tracked. That means the G channel of the
    already-cached as-shot-WB .fjlrg data is *already* WB-gain-invariant,
    with no reversal or new per-scene gain extraction needed - the
    "closest rigorously equivalent representation available in the
    current cache" for a pre-WB-gain luminance proxy."""
    r, g, bl = full_res_linear[:, 0], full_res_linear[:, 1], full_res_linear[:, 2]
    if luma_mode == "rec709":
        return LUMA_WEIGHTS[0] * r + LUMA_WEIGHTS[1] * g + LUMA_WEIGHTS[2] * bl
    if luma_mode == "green_channel":
        return g
    raise ValueError(f"unknown luma_mode {luma_mode!r}")


def apply_bilateral_grid(grid, full_res_linear, luma_mode="rec709"):
    """grid: (B,12,D,H,W) from BilateralGridPredictor.forward.
    full_res_linear: (B,3,H_full,W_full) linear RGB at the resolution to
    correct (this pilot's "full resolution" is the .fjlrg's ~1536px-long-
    edge working resolution, matching how Phase 3's own fitting pipeline
    already compared images - see training/target_tiff.py). The as-shot-WB
    color image is always both the input transformed AND the output
    produced - only the luma axis used to INDEX the grid varies with
    luma_mode (see compute_luma_guide).
    Returns the corrected image, same shape as full_res_linear."""
    b, c, h, w = full_res_linear.shape
    assert c == 3, f"expected 3-channel RGB, got {c}"

    luma = compute_luma_guide(full_res_linear, luma_mode)  # (B,H,W)

    device = full_res_linear.device
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
    grid_x = grid_x.unsqueeze(0).expand(b, h, w)
    grid_y = grid_y.unsqueeze(0).expand(b, h, w)
    grid_z = (luma.clamp(0, 1) * 2 - 1)  # luma in [0,1] -> [-1,1]

    # grid_sample 5D expects sampling coords (B, D_out, H_out, W_out, 3) with
    # last dim (x,y,z) mapping to (W,H,D) of the input volume. D_out=1 here -
    # one trilinear lookup per full-res pixel, z varies per-pixel via luma.
    sample_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(1)  # (B,1,H,W,3)

    sampled = F.grid_sample(grid, sample_coords, mode="bilinear", padding_mode="border", align_corners=True)
    coeffs = sampled.squeeze(2)  # (B,12,H,W)
    coeffs = coeffs.view(b, 3, 4, h, w)

    rgb = full_res_linear  # (B,3,H,W)
    out = (
        coeffs[:, :, 0] * rgb[:, 0:1]
        + coeffs[:, :, 1] * rgb[:, 1:2]
        + coeffs[:, :, 2] * rgb[:, 2:3]
        + coeffs[:, :, 3]
    )
    return out


def grid_smoothness_loss(grid):
    """Total-variation penalty across the bilateral grid's three axes
    (luma depth, spatial H, spatial W) - discourages high-frequency,
    per-cell-noisy affine coefficients, which is the standard HDRNet-style
    regularizer for exactly the failure mode a handful of hard-scene
    regressions suggests (the grid overfitting a locally weird cell rather
    than predicting a smooth local transform). grid: (B,12,D,H,W)."""
    d_diff = (grid[:, :, 1:, :, :] - grid[:, :, :-1, :, :]).abs().mean()
    h_diff = (grid[:, :, :, 1:, :] - grid[:, :, :, :-1, :]).abs().mean()
    w_diff = (grid[:, :, :, :, 1:] - grid[:, :, :, :, :-1]).abs().mean()
    return d_diff + h_diff + w_diff


def resize_for_network(full_res_linear, size):
    # mode="area" would be the more standard downsample choice, but MPS's
    # adaptive_avg_pool2d (which "area" delegates to) requires input size
    # evenly divisible by output size - not true for this project's image
    # dimensions (verified 2026-09-14 against torch 2.11.0). Antialiased
    # bilinear is the documented equivalent for arbitrary sizes and works
    # on MPS.
    return F.interpolate(full_res_linear, size=(size, size), mode="bilinear", antialias=True, align_corners=False)


class RefinementNet(nn.Module):
    """2026-09-17 hybrid architecture - round 5 closed the bilateral-grid-
    only tuning branch (WB guide, TV smoothness, encoder detail, grid
    resolution all tested without a reliable fix on the visually-diagnosed
    seam/ring/color-cast artifacts). This is a genuinely different
    component, not another grid tweak: a small multiscale U-Net operating
    at full working resolution (unlike the grid encoder's 256x256 low-res
    path), with a high-resolution skip connection carrying full-detail
    features straight from the input to the output stage - exactly what
    the grid's 8x8-ish spatial cells structurally cannot represent, since
    every pixel within a cell shares one affine transform.

    Bounded and identity-safe by construction, not just by training
    dynamics: the final conv is zero-initialized (so at the start of
    training this is an exact no-op, output == grid_output everywhere -
    training only learns to deviate where the grid's correction is
    measurably wrong), and the residual is hard-clamped to
    +/-max_delta via tanh regardless of what the network ever learns to
    predict, so it can never freely rewrite an already-correct photo.

    Input sizes here are NOT powers of two and not evenly divisible (the
    .fjlrg's long edge is fixed at 1536px but the short edge varies by
    scene aspect ratio - see train.py's batch_size=1 note) - every
    upsample step targets the exact skip tensor's spatial size rather
    than a fixed 2x scale factor, so this is robust to that regardless of
    input shape."""

    def __init__(self, base_ch=8, max_delta=0.08):
        super().__init__()
        self.max_delta = max_delta
        c = base_ch

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(4, out_ch), out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(6, c)                    # full res - input + grid output, concatenated
        self.down1 = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)
        self.enc2 = block(c, c * 2)                 # ~1/2 res
        self.down2 = nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=2, padding=1)
        self.bottleneck = block(c * 2, c * 2)        # ~1/4 res
        self.up2 = block(c * 2 + c * 2, c)
        self.up1 = block(c + c, c)
        self.out_conv = nn.Conv2d(c, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, full_res_linear, grid_output):
        x = torch.cat([full_res_linear, grid_output], dim=1)  # (B,6,H,W)
        e1 = self.enc1(x)
        d1 = F.relu(self.down1(e1), inplace=True)
        e2 = self.enc2(d1)
        d2 = F.relu(self.down2(e2), inplace=True)
        b = self.bottleneck(d2)
        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e1], dim=1))
        raw = self.out_conv(u1)
        delta = self.max_delta * torch.tanh(raw)
        return grid_output + delta, delta


class HybridPredictor(nn.Module):
    """Bilateral grid (global color/tone correction, unchanged mechanism)
    + RefinementNet (bounded, identity-safe local residual correction),
    trained jointly end-to-end. One combined forward pass, low-res input
    -> full-res output - the single graph Codex's ONNX/ONNX Runtime Web
    smoke test needs to validate before any full training run, per
    instruction."""

    def __init__(self, grid_spatial=8, grid_luma=9, low_res=256, base_ch=16,
                 refinement_base_ch=8, refinement_max_delta=0.08):
        super().__init__()
        self.grid = BilateralGridPredictor(
            grid_spatial=grid_spatial, grid_luma=grid_luma, low_res=low_res, base_ch=base_ch,
        )
        self.refinement = RefinementNet(base_ch=refinement_base_ch, max_delta=refinement_max_delta)

    def forward(self, full_res_linear, low_res_size=256, luma_mode="rec709"):
        grid_output, grid_aux = self.grid.predict_and_apply(full_res_linear, low_res_size, luma_mode)
        final_output, delta = self.refinement(full_res_linear, grid_output)
        return final_output, grid_output, delta

    def predict_and_apply(self, full_res_linear, low_res_size=256, luma_mode="rec709"):
        """Same (output, aux) convention as BilateralGridPredictor's
        method of the same name - aux additionally carries grid_output
        (pre-refinement) and delta (the applied residual) for logging."""
        final_output, grid_output, delta = self(full_res_linear, low_res_size, luma_mode)
        return final_output, {"grid_output": grid_output, "delta": delta}
