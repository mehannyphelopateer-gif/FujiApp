"""Pixel-aligned D65 Lab loss for a separately scoped Provia pilot.

No training or inference defaults change on import. The frozen numpy
scorer supplies its matrix/white point; CIEDE2000 remains the evaluation
metric, not a learned/perceptual-network objective.
"""
import torch

from color_metrics import _M, _WHITE


def linear_rgb_to_lab(rgb):
    """NCHW linear sRGB -> NCHW Lab, matching color_metrics' display clamp.

The inactive cube-root branch is bounded away from zero as well: torch
where evaluates both branches, so pow(0, 1/3) would otherwise introduce
infinite gradients at black even when the linear branch is selected.
"""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("Expected NCHW three-channel RGB")
    if rgb.dtype not in (torch.float32, torch.float64):
        raise ValueError("Color loss requires float32 or float64, without AMP")
    matrix = torch.as_tensor(_M, dtype=rgb.dtype, device=rgb.device)
    white = torch.as_tensor(_WHITE, dtype=rgb.dtype, device=rgb.device)
    xyz = torch.matmul(rgb.clamp(0, 1).permute(0, 2, 3, 1), matrix.T) / white
    delta = 6 / 29
    cube = xyz.clamp_min(delta ** 3).pow(1 / 3)
    f = torch.where(xyz > delta ** 3, cube, xyz / (3 * delta ** 2) + 4 / 29)
    lab = torch.stack((116 * f[..., 1] - 16,
                       500 * (f[..., 0] - f[..., 1]),
                       200 * (f[..., 1] - f[..., 2])), dim=1)
    return lab


def per_scene_terms(pred, target):
    """Return raw linear L1, Lab L1 / 100 and their fixed half/half mean.

Linear L1 stays UNCLAMPED, so out-of-gamut predictions retain a restoring
gradient when the display-clamped Lab term has zero gradient. These are
paired pixel losses; no GAN, LPIPS or synthesis objective is involved.
"""
    if pred.shape != target.shape or pred.dtype != target.dtype or pred.device != target.device:
        raise ValueError("Prediction/target shape, dtype and device must match")
    linear = (pred - target).abs().mean(dim=(1, 2, 3))
    lab = (linear_rgb_to_lab(pred) - linear_rgb_to_lab(target)).abs().mean(dim=(1, 2, 3)) / 100
    return {"linear_l1": linear, "lab_l1_scaled": lab, "combined": (linear + lab) / 2}
