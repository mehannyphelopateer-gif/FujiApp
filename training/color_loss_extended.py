"""Training-only Lab extension; preserves all earlier bound loss/scorer code.

Inside the display RGB gamut this is the identical D65 Lab conversion.
Outside it, retain RGB values and use Lab's linear low-XYZ branch rather
than hard-clamping RGB and losing its colour gradient. Extended values
are an optimisation surrogate, not a physical displayed-colour metric.
Evaluation must continue using the original display-clamped scorer.
"""
import torch

from color_metrics import _M, _WHITE


def linear_rgb_to_lab(rgb):
    if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.dtype not in (torch.float32, torch.float64):
        raise ValueError("Expected float32/64 NCHW RGB")
    matrix = torch.as_tensor(_M, dtype=rgb.dtype, device=rgb.device)
    white = torch.as_tensor(_WHITE, dtype=rgb.dtype, device=rgb.device)
    xyz = torch.matmul(rgb.permute(0, 2, 3, 1), matrix.T) / white
    delta = 6 / 29
    cube = xyz.clamp_min(delta ** 3).pow(1 / 3)
    f = torch.where(xyz > delta ** 3, cube, xyz / (3 * delta ** 2) + 4 / 29)
    return torch.stack((116 * f[..., 1] - 16,
                        500 * (f[..., 0] - f[..., 1]),
                        200 * (f[..., 1] - f[..., 2])), dim=1)


def per_scene_terms(pred, target):
    if pred.shape != target.shape or pred.dtype != target.dtype or pred.device != target.device:
        raise ValueError("Prediction/target shape, dtype and device must match")
    lab = (linear_rgb_to_lab(pred) - linear_rgb_to_lab(target)).abs().mean(dim=(1, 2, 3)) / 100
    linear = (pred - target).abs().mean(dim=(1, 2, 3))
    return {"linear_l1": linear, "lab_l1_scaled": lab, "combined": (linear + lab) / 2}
