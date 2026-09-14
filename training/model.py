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


class BilateralGridPredictor(nn.Module):
    def __init__(self, grid_spatial=4, grid_luma=7, low_res=256, base_ch=16):
        super().__init__()
        self.grid_spatial = grid_spatial
        self.grid_luma = grid_luma
        self.low_res = low_res
        coeffs_per_cell = 12  # 3x4 affine matrix (3 output channels, RGB+bias)

        self.encoder = nn.Sequential(
            _conv_block(3, base_ch),          # 256 -> 128
            _conv_block(base_ch, base_ch * 2),  # 128 -> 64
            _conv_block(base_ch * 2, base_ch * 4),  # 64 -> 32
            _conv_block(base_ch * 4, base_ch * 4),  # 32 -> 16
            _conv_block(base_ch * 4, base_ch * 4),  # 16 -> 8
        )
        self.head = nn.Conv2d(base_ch * 4, grid_luma * coeffs_per_cell, kernel_size=1)

    def forward(self, low_res_input):
        """low_res_input: (B,3,low_res,low_res) linear RGB, already resized
        by the caller. Returns the grid as (B, 12, grid_luma, grid_spatial,
        grid_spatial) - a (C,D,H,W) volume ready for apply_bilateral_grid."""
        feat = self.encoder(low_res_input)
        feat = F.adaptive_avg_pool2d(feat, (self.grid_spatial, self.grid_spatial))
        raw = self.head(feat)  # (B, grid_luma*12, gs, gs)
        b, _, gs_h, gs_w = raw.shape
        grid = raw.view(b, self.grid_luma, 12, gs_h, gs_w)
        grid = grid.permute(0, 2, 1, 3, 4).contiguous()  # (B,12,D=luma,H=gs,W=gs)
        return grid


def apply_bilateral_grid(grid, full_res_linear):
    """grid: (B,12,D,H,W) from BilateralGridPredictor.forward.
    full_res_linear: (B,3,H_full,W_full) linear RGB at the resolution to
    correct (this pilot's "full resolution" is the .fjlrg's ~1536px-long-
    edge working resolution, matching how Phase 3's own fitting pipeline
    already compared images - see training/target_tiff.py).
    Returns the corrected image, same shape as full_res_linear."""
    b, c, h, w = full_res_linear.shape
    assert c == 3, f"expected 3-channel RGB, got {c}"

    r, g, bl = full_res_linear[:, 0], full_res_linear[:, 1], full_res_linear[:, 2]
    luma = LUMA_WEIGHTS[0] * r + LUMA_WEIGHTS[1] * g + LUMA_WEIGHTS[2] * bl  # (B,H,W)

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


def resize_for_network(full_res_linear, size):
    # mode="area" would be the more standard downsample choice, but MPS's
    # adaptive_avg_pool2d (which "area" delegates to) requires input size
    # evenly divisible by output size - not true for this project's image
    # dimensions (verified 2026-09-14 against torch 2.11.0). Antialiased
    # bilinear is the documented equivalent for arbitrary sizes and works
    # on MPS.
    return F.interpolate(full_res_linear, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
