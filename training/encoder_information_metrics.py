"""Pure CPU helpers for the fit-only encoder-information sensitivity assay.

No I/O, no project loaders, no model, no selection, no decision thresholds.
Every value returned is a finite float/int or None (JSON-safe, never NaN).

These helpers measure EMPIRICAL SENSITIVITY only. A small (or large) response to
a gain, DR-condition or sampling-translation perturbation does not by itself show
that information is lost or that a model is defective: GroupNorm with real conv
biases and a nonzero condition plane is not generally invariant, and the bilateral
grid's absolute-luma slicing and affine RGB application still see intensity
downstream of the encoder.

vector_response(base, changed, scale=None)
    L2 / RMS / cosine / relative / scene-normalised difference of two same-shape tensors.
scene_distance_scale(grids)
    Median unique-pair L2 distance between per-scene vectors (None if fewer than 2).
image_summary(rgb, trim=0)
    Rec709 luma statistics of a (1,3,H,W) image after excluding `trim` border pixels.
fixed_box_prefilter(rgb, kernel=7)
    Reflect-padded same-shape box filter. COMPARISON ONLY: models were trained on
    unfiltered thumbnails, so prefiltered inference is out-of-distribution - use for
    stability contrasts, never for accuracy claims or automatic preprocessing changes.
translated_same_shape(rgb, dy, dx, max_abs=2)
    Integer, equal-size, reflect-padded translation. This is a sampling-translation
    perturbation: border content changes (reflection), so content is NOT exactly unchanged.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

REC709_LUMA = (0.2126, 0.7152, 0.0722)   # fixed; equal to model.LUMA_WEIGHTS
LOG_FLOOR = 1.0 / 65535.0                # explicit floor for log2 luma (one 16-bit code)
NORMALIZER_EPS = 1e-12                   # scale at or below this -> normalized_l2 is None


def _need(cond, msg):
    if not cond:
        raise ValueError(msg)


def _f(x):
    x = float(x)
    _need(math.isfinite(x), "non-finite statistic")
    return x


def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _as_f64(a, name):
    """torch tensor or array-like (floating) -> finite, non-empty float64 numpy array."""
    if isinstance(a, torch.Tensor):
        _need(a.is_floating_point(), f"{name} must be a floating tensor")
        a = a.detach().cpu().to(torch.float64).numpy()
    else:
        a = np.asarray(a)
        _need(np.issubdtype(a.dtype, np.floating), f"{name} must be a floating array")
        a = a.astype(np.float64)
    _need(a.size > 0 and bool(np.isfinite(a).all()), f"{name} must be non-empty and finite")
    return a


def _image_tensor(rgb, name="rgb"):
    _need(isinstance(rgb, torch.Tensor), f"{name} must be a torch tensor")
    _need(rgb.is_floating_point() and rgb.dim() == 4 and rgb.shape[0] >= 1 and rgb.shape[2] >= 1
          and rgb.shape[3] >= 1, f"{name} must be a floating (B,C,H,W) tensor")
    _need(bool(torch.isfinite(rgb).all()), f"{name} must be finite")
    return rgb


# ------------------------------------------------------------------ vector responses
def vector_response(base, changed, scale=None):
    """Difference changed - base, flattened.

    l2              ||changed - base||
    rms             l2 / sqrt(element count)
    cosine          <base, changed> / (|base| |changed|), None if either norm is 0
    relative_l2     l2 / |base|, None if |base| == 0
    normalized_l2   l2 / scale, None if scale is None or scale <= 1e-12
    """
    b = _as_f64(base, "base")
    c = _as_f64(changed, "changed")
    _need(b.shape == c.shape, f"base/changed shape mismatch {b.shape} vs {c.shape}")
    b, c = b.ravel(), c.ravel()
    if scale is not None:
        _need(not isinstance(scale, bool), "scale must be a number or None")
        scale = float(scale)
        _need(math.isfinite(scale) and scale >= 0.0, "scale must be finite and non-negative")
    d = c - b
    l2 = float(np.linalg.norm(d))
    nb, nc = float(np.linalg.norm(b)), float(np.linalg.norm(c))
    cosine = None
    if nb > 0.0 and nc > 0.0:
        cosine = _f(min(1.0, max(-1.0, float(np.dot(b, c)) / (nb * nc))))
    return {"elements": int(b.size), "l2": _f(l2), "rms": _f(l2 / math.sqrt(b.size)),
            "base_l2": _f(nb), "changed_l2": _f(nc), "cosine": cosine,
            "relative_l2": _f(l2 / nb) if nb > 0.0 else None,
            "normalized_l2": _f(l2 / scale) if scale is not None and scale > NORMALIZER_EPS else None,
            "scale": scale}


def scene_distance_scale(grids):
    """Median L2 over unique unordered pairs (i<j) of same-shape per-scene vectors.

    Returns None for fewer than 2 grids. np.median: even pair counts average the middle two.
    Identical grids give 0.0, which vector_response then treats as a null normaliser.
    """
    arrs = [_as_f64(g, f"grids[{i}]") for i, g in enumerate(grids)]
    if len(arrs) < 2:
        return None
    _need(all(a.shape == arrs[0].shape for a in arrs), "grids must share one shape")
    flat = [a.ravel() for a in arrs]
    d = [float(np.linalg.norm(flat[i] - flat[j])) for i in range(len(flat)) for j in range(i + 1, len(flat))]
    return _f(np.median(d))


# ------------------------------------------------------------------ image statistics
def image_summary(rgb, trim=0):
    """rgb: (1,3,H,W) torch tensor or array, linear values used as given (no clamp).

    trim excludes `trim` pixels on every border (caller sets it explicitly, e.g. the
    translation range natively and its thumbnail equivalent). Luma is fixed Rec709.
    p99 uses np.percentile linear interpolation. fraction_any_channel_at_or_above_1
    counts stored values >= 1; it is NOT a claim about sensor clipping.
    """
    a = _as_f64(rgb, "rgb")
    _need(a.ndim == 4 and a.shape[0] == 1 and a.shape[1] == 3, f"rgb must be (1,3,H,W), got {a.shape}")
    _need(_is_int(trim) and trim >= 0, "trim must be a non-negative int")
    h, w = a.shape[2], a.shape[3]
    _need(2 * trim < h and 2 * trim < w, "trim leaves no interior")
    x = a[0, :, trim:h - trim, trim:w - trim]
    luma = REC709_LUMA[0] * x[0] + REC709_LUMA[1] * x[1] + REC709_LUMA[2] * x[2]
    floored = np.maximum(luma, LOG_FLOOR)
    return {"pixels": int(luma.size), "trim": int(trim), "luma_weights": list(REC709_LUMA),
            "mean_luma": _f(luma.mean()),
            "log2_floor": LOG_FLOOR,
            "mean_log2_luma_floored": _f(np.log2(floored).mean()),
            "fraction_luma_at_or_below_floor": _f((luma <= LOG_FLOOR).mean()),
            "p99_luma": _f(np.percentile(luma, 99)),
            "fraction_any_channel_at_or_above_1": _f((x >= 1.0).any(axis=0).mean()),
            "fraction_any_channel_below_0": _f((x < 0.0).any(axis=0).mean())}


# ------------------------------------------------------------------ perturbations
def fixed_box_prefilter(rgb, kernel=7):
    """Same-shape box filter: reflect pad by kernel//2, then avg_pool2d(kernel, stride 1).

    Odd kernel only; kernel 1 is the identity. COMPARISON / OOD only (see module doc).
    """
    rgb = _image_tensor(rgb)
    _need(_is_int(kernel) and kernel >= 1 and kernel % 2 == 1, "kernel must be an odd int >= 1")
    p = kernel // 2
    if p == 0:
        return rgb.clone()
    _need(rgb.shape[2] > p and rgb.shape[3] > p, "image too small for reflect padding")
    padded = F.pad(rgb, (p, p, p, p), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1)


def translated_same_shape(rgb, dy, dx, max_abs=2):
    """Integer translation keeping shape: out[..., y, x] = in[..., y - dy, x - dx].

    Positive dy moves content DOWN (toward larger row index); positive dx moves it RIGHT.
    Positions whose source falls outside the image take reflected values (reflect pad,
    edge pixel not repeated). |dy|, |dx| <= max_abs. Border content therefore differs:
    this is a sampling-translation perturbation, not identical content.
    """
    rgb = _image_tensor(rgb)
    _need(_is_int(max_abs) and max_abs >= 0, "max_abs must be a non-negative int")
    _need(_is_int(dy) and _is_int(dx), "dy/dx must be ints")
    _need(abs(dy) <= max_abs and abs(dx) <= max_abs, "translation exceeds max_abs")
    py, px = abs(int(dy)), abs(int(dx))
    h, w = rgb.shape[2], rgb.shape[3]
    _need(h > py and w > px, "image too small for reflect padding")
    padded = F.pad(rgb, (px, px, py, py), mode="reflect") if (py or px) else rgb
    y0, x0 = py - int(dy), px - int(dx)
    return padded[..., y0:y0 + h, x0:x0 + w].clone()
