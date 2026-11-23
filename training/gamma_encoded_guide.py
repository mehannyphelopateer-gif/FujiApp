"""Research-only hybrid: true-linear RGB, fixed encoded grid-depth guide.

Only the sampling coordinate changes. Encoder, affine RGB and refinement
all receive the unchanged linear tensor. The curve is LibRaw's independently
qualified default table, not a learned mapping or a claim to reconstruct the
legacy area-resized image (encoding and resizing do not commute).
"""
import hashlib

import torch
import torch.nn.functional as F

from eval_provia_gamma_length import libraw_forward_table
from model import HybridPredictor, compute_luma_guide, resize_for_network

TABLE_SHA256 = "1777187e6755daadee56e79d584b361ec03f1fca2c62e71f330a56bda5505f7c"
GUIDE_CONTRACT = "libraw-default-u16-round-gather-rgb-rec709-v1"


def qualified_table():
    table, _ = libraw_forward_table()
    if hashlib.sha256(table.astype('<u2').tobytes()).hexdigest() != TABLE_SHA256:
        raise ValueError("Pinned gamma lookup changed")
    return torch.from_numpy(table.astype('float32') / 65535.)


def encoded_guide(rgb, table):
    # G is decoded as uint16 / 65535. Round restores every possible code
    # exactly; the mapping is fixed, and does not need input gradients.
    # Use exactly the same operation in training and exported inference.
    codes = torch.round(rgb.clamp(0, 1) * 65535.).to(torch.int64)
    encoded = table[codes]
    return compute_luma_guide(encoded, "rec709")


def apply_with_guide(grid, rgb, guide):
    """Original slicing/application math, with an explicit guide tensor."""
    b, _, h, w = rgb.shape
    ys = torch.linspace(-1, 1, h, device=rgb.device)
    xs = torch.linspace(-1, 1, w, device=rgb.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx.unsqueeze(0).expand(b, h, w)
    gy = gy.unsqueeze(0).expand(b, h, w)
    gz = guide.clamp(0, 1) * 2 - 1
    coords = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1)
    sampled = F.grid_sample(grid, coords, mode="bilinear",
                            padding_mode="border", align_corners=True)
    coeffs = sampled.squeeze(2).view(b, 3, 4, h, w)
    return (coeffs[:, :, 0] * rgb[:, 0:1]
            + coeffs[:, :, 1] * rgb[:, 1:2]
            + coeffs[:, :, 2] * rgb[:, 2:3] + coeffs[:, :, 3])


class GammaEncodedGuideHybrid(HybridPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No learned parameters or state-dict keys are added. The immutable
        # table is bound by code + TABLE_SHA256 in the study/export manifests.
        self.register_buffer("gamma_guide_table", qualified_table(), persistent=False)

    def forward(self, full_res_linear, low_res_size=256, luma_mode="rec709", condition=None):
        if luma_mode != "rec709":
            raise ValueError("Encoded-guide assay supports only its frozen rec709 mapping")
        low = resize_for_network(full_res_linear, low_res_size)
        grid = self.grid(low, condition)
        guide = encoded_guide(full_res_linear, self.gamma_guide_table)
        grid_output = apply_with_guide(grid, full_res_linear, guide)
        final, delta = self.refinement(full_res_linear, grid_output)
        return final, grid_output, delta
