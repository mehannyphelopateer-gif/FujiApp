"""Pure NumPy helpers for the fit-only gamma/guide diagnostic. No I/O, no model, no selection.

reference_lab(target_linear)            -> target CIELAB, exactly as the frozen dev metric computes it
error_statistics(pred_linear, target_lab) -> unsigned dE00 / abs dL,da,db,dC / signed Lab, whole and per
                                           reference tone band, with additive dE00 contributions
guide_statistics(guide, target_Lstar)   -> actual trilinear depth-plane occupancy of the bilateral-grid
                                           luma guide (align_corners=True), whole and per tone band

Definitions match provia_color_pilot.dev_metrics / BANDS and model.apply_bilateral_grid:
  * prediction clipped to [0, 1] before Lab; target passed through color_metrics.linear_srgb_to_lab
    unchanged. NOTE: linear_srgb_to_lab itself clips its input to [0, 1], so float32 resize roundoff a
    few ULP above 1 is clipped inside Lab exactly as in the frozen metric (the cached target array is
    never rewritten).
  * bands by reference L*: shadow L<25, mid 25<=L<75, high L>=75 - every pixel in exactly one band.
  * guide clamped to [0, 1]; depth coordinate u = g*(depth-1) (grid_sample align_corners=True,
    z = 2g-1), lower plane floor(u), upper min(lo+1, depth-1), weights 1-frac / frac.
Results are JSON-ready dicts: finite floats or None (never NaN), ints for counts.
These are descriptive statistics only; they establish no causal mechanism and select nothing.
"""
import math

import numpy as np

from color_metrics import ciede2000, linear_srgb_to_lab

BANDS = {"shadow": (None, 25.0), "mid": (25.0, 75.0), "high": (75.0, None)}   # identical to P.BANDS


def _need(cond, msg):
    if not cond:
        raise ValueError(msg)


def _f(x):
    x = float(x)
    _need(math.isfinite(x), "non-finite statistic")
    return x


def _float_array(a, name, ndim, last=None):
    a = np.asarray(a)
    _need(np.issubdtype(a.dtype, np.floating), f"{name} must be a floating array")
    _need(a.ndim == ndim and (last is None or a.shape[-1] == last), f"{name} has wrong shape {a.shape}")
    _need(a.size > 0 and bool(np.isfinite(a).all()), f"{name} must be non-empty and finite")
    return a.astype(np.float64)


def band_masks(L):
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = np.ones(L.shape, bool)
        if lo is not None:
            m &= L >= lo
        if hi is not None:
            m &= L < hi
        out[b] = m
    return out


# ------------------------------------------------------------------ colour error
def reference_lab(target_linear):
    """HxWx3 finite float linear target -> HxWx3 float64 Lab via the frozen metric's conversion."""
    t = _float_array(target_linear, "target_linear", 3, 3)
    return linear_srgb_to_lab(t)


def _error_block(e, d, dC, mask, n_total, e_total):
    n = int(mask.sum())
    if n == 0:
        return {"count": 0, "fraction": 0.0, "mean_dE00": None, "p95_dE00": None,
                "mean_abs_dL": None, "mean_abs_da": None, "mean_abs_db": None, "mean_abs_dC": None,
                "mean_signed_Lab": None, "dE00_contribution": 0.0, "dE00_share": None}
    em, dm = e[mask], d[mask]
    s = float(em.sum())
    return {"count": n, "fraction": _f(n / n_total), "mean_dE00": _f(em.mean()),
            "p95_dE00": _f(np.percentile(em, 95)),
            "mean_abs_dL": _f(np.abs(dm[:, 0]).mean()), "mean_abs_da": _f(np.abs(dm[:, 1]).mean()),
            "mean_abs_db": _f(np.abs(dm[:, 2]).mean()), "mean_abs_dC": _f(np.abs(dC[mask]).mean()),
            "mean_signed_Lab": [_f(v) for v in dm.mean(0)],
            "dE00_contribution": _f(s / n_total), "dE00_share": _f(s / e_total) if e_total > 0 else None}


def error_statistics(pred_linear, target_lab):
    """pred_linear HxWx3 finite float (clipped to [0,1] before Lab); target_lab HxWx3 from reference_lab.

    Errors are signed pred - target. Unsigned error is per-pixel |.|, never |mean signed|.
    Sum of band dE00_contribution == whole mean_dE00; band means weighted by fraction reconstruct it too.
    """
    p = _float_array(pred_linear, "pred_linear", 3, 3)
    R = _float_array(target_lab, "target_lab", 3, 3)
    _need(p.shape == R.shape, f"shape mismatch {p.shape} vs {R.shape}")
    A = linear_srgb_to_lab(np.clip(p, 0.0, 1.0))
    e = ciede2000(A, R)
    _need(bool(np.isfinite(e).all()), "non-finite dE00")
    d = A - R
    dC = np.hypot(A[..., 1], A[..., 2]) - np.hypot(R[..., 1], R[..., 2])
    n_total, e_total = int(e.size), float(e.sum())
    whole = np.ones(e.shape, bool)
    return {"pixels": n_total,
            "whole": _error_block(e, d, dC, whole, n_total, e_total),
            "bands": {b: _error_block(e, d, dC, m, n_total, e_total) for b, m in band_masks(R[..., 0]).items()}}


# ------------------------------------------------------------------ guide occupancy
def _plane_weights(g, depth):
    u = g * (depth - 1)
    lo = np.minimum(np.floor(u).astype(np.int64), depth - 1)
    hi = np.minimum(lo + 1, depth - 1)
    frac = u - lo
    return lo, hi, frac


def _guide_block(g, lo, hi, frac, mask, depth):
    n = int(mask.sum())
    if n == 0:
        return {"count": 0, "lower_index_hist": None, "plane_mass": None, "effective_planes": None,
                "first_interval_fraction": None}
    l, h, f = lo[mask], hi[mask], frac[mask]
    hist = np.bincount(l, minlength=depth) / n
    mass = (np.bincount(l, weights=1.0 - f, minlength=depth) + np.bincount(h, weights=f, minlength=depth)) / n
    return {"count": n, "lower_index_hist": [_f(v) for v in hist], "plane_mass": [_f(v) for v in mass],
            "effective_planes": _f(1.0 / float((mass ** 2).sum())),
            "first_interval_fraction": _f(float((g[mask] < 1.0 / (depth - 1)).mean()))}


def guide_statistics(guide, target_Lstar, depth=9):
    """guide: HxW compute_luma_guide output (before the model's clamp); target_Lstar: HxW reference L*.

    plane_mass is the trilinear interpolation weight mass per depth plane (sums to 1);
    lower_index_hist counts floor(u) including the endpoint plane depth-1;
    first_interval_fraction is the fraction with clamped guide < 1/(depth-1) (g < 1/8 for depth 9).
    """
    _need(type(depth) is int and depth >= 2, "depth must be an int >= 2")
    g0 = _float_array(guide, "guide", 2)
    L = _float_array(target_Lstar, "target_Lstar", 2)
    _need(g0.shape == L.shape, f"shape mismatch {g0.shape} vs {L.shape}")
    out_of_unit = float(((g0 < 0.0) | (g0 > 1.0)).mean())
    g = np.clip(g0, 0.0, 1.0)
    lo, hi, frac = _plane_weights(g, depth)
    whole = _guide_block(g, lo, hi, frac, np.ones(g.shape, bool), depth)
    whole["out_of_unit_fraction_before_clamp"] = _f(out_of_unit)
    return {"depth": depth, "pixels": int(g.size), "whole": whole,
            "bands": {b: _guide_block(g, lo, hi, frac, m, depth) for b, m in band_masks(L).items()}}
