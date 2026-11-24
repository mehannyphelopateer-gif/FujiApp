"""Pure registration diagnostics for the fit-only geometry check (NumPy + OpenCV, no I/O).

Candidate measurement tools only: nothing here changes a pipeline, picks a warp, crops an input or
target, or touches colour metrics. Metadata windows ASSUME that the decoded G frame spans exactly the
stored RAW cropped size (RawImageCroppedSize); that is a hypothesis to be scored, never a fact, and
cropped size is never asserted to be the visible Fuji area.

normalized_zoom_window(base_hw, top_left_hw, zoom_hw, orientation) -> [x0, y0, x1, y1] (display frame)
zoom_resample(rgb, window)          -> same-size float32 image resampled from the window (cv2.remap, linear)
registration_feature(rgb)           -> float32 gradient magnitude of blurred sRGB-encoded Rec709 luma
feature_ncc(a, b)                   -> Pearson correlation, or None if either std < 1e-4
tile_phase_shifts(reference, moving)-> one record per fixed 3x3 tile (raw values kept, outliers not zeroed)
summarize_shifts(records)           -> usable count and medians (descriptive; no warp selection)
"""
import math

import cv2
import numpy as np

GRID = 3
RIM = 4
MIN_STD = 1e-4
MIN_RESPONSE = 0.2
MAX_SHIFT = 8.0
GAUSS_SIGMA = 1.2
REC709 = (0.2126, 0.7152, 0.0722)
MIN_TILE_INTERIOR = 8


class RegistrationError(ValueError):
    pass


def _need(cond, msg):
    if not cond:
        raise RegistrationError(msg)


def _num(v, name, positive):
    _need(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), f"{name} must be a finite number")
    _need(v > 0 if positive else v >= 0, f"{name} must be {'positive' if positive else 'non-negative'}")
    return float(v)


def _pair(v, name, positive):
    _need(isinstance(v, (list, tuple)) and len(v) == 2, f"{name} must be (height, width)")
    return _num(v[0], f"{name}[0]", positive), _num(v[1], f"{name}[1]", positive)


def normalized_zoom_window(base_hw, top_left_hw, zoom_hw, orientation):
    """Stored (raw, height-width order) ROI -> normalized [x0, y0, x1, y1] in the displayed frame.

    Orientation 1: identity. 8 (rotate CCW to display): (x0,y0,x1,y1) -> (y0, 1-x1, y1, 1-x0).
    6 (rotate CW to display): -> (1-y1, x0, 1-y0, x1). Others are refused.
    """
    H, W = _pair(base_hw, "base_hw", True)
    top, left = _pair(top_left_hw, "top_left_hw", False)
    h, w = _pair(zoom_hw, "zoom_hw", True)
    _need(top + h <= H and left + w <= W, "zoom window extends outside the base frame")
    _need(type(orientation) is int, "orientation must be an int")
    x0, y0, x1, y1 = left / W, top / H, (left + w) / W, (top + h) / H
    if orientation == 1:
        return [x0, y0, x1, y1]
    if orientation == 8:
        return [y0, 1 - x1, y1, 1 - x0]
    if orientation == 6:
        return [1 - y1, x0, 1 - y0, x1]
    raise RegistrationError(f"unsupported orientation {orientation}")


def _image(a, name, channels=None):
    a = np.asarray(a)
    _need(np.issubdtype(a.dtype, np.floating), f"{name} must be floating")
    if channels is None:
        _need(a.ndim == 2, f"{name} must be 2-D")
    else:
        _need(a.ndim == 3 and a.shape[2] == channels, f"{name} must be HxWx{channels}")
    _need(a.shape[0] >= 2 and a.shape[1] >= 2 and bool(np.isfinite(a).all()), f"{name} must be finite and at least 2x2")
    return a


def _window(window):
    _need(isinstance(window, (list, tuple)) and len(window) == 4, "window must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (_num(v, "window", False) for v in window)
    _need(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, "window must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


def zoom_resample(rgb, window):
    """Resample the normalized window to the full HxW (pixel-centre convention, bilinear).

    Destination pixel u maps to source index x0*W + (u + 0.5)*(x1 - x0) - 0.5, so [0,0,1,1] is an exact copy.
    """
    a = _image(rgb, "rgb", 3)
    x0, y0, x1, y1 = _window(window)
    H, W = a.shape[:2]
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    map_x = (x0 * W + (u + 0.5) * (x1 - x0) - 0.5).astype(np.float32)
    map_y = (y0 * H + (v + 0.5) * (y1 - y0) - 0.5).astype(np.float32)
    mx, my = np.meshgrid(map_x, map_y)
    src = np.ascontiguousarray(a, dtype=np.float32)
    return cv2.remap(src, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _srgb_encode(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def registration_feature(rgb):
    """Linear RGB -> clamp [0,1] -> IEC sRGB curve -> Rec709 luma -> Gaussian(1.2) -> Sobel-3 magnitude."""
    a = _image(rgb, "rgb", 3).astype(np.float64)
    enc = _srgb_encode(np.clip(a, 0.0, 1.0))
    luma = (REC709[0] * enc[..., 0] + REC709[1] * enc[..., 1] + REC709[2] * enc[..., 2]).astype(np.float32)
    blur = cv2.GaussianBlur(luma, (0, 0), sigmaX=GAUSS_SIGMA, sigmaY=GAUSS_SIGMA)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def feature_ncc(a, b):
    a = _image(a, "a").astype(np.float64)
    b = _image(b, "b").astype(np.float64)
    _need(a.shape == b.shape, "feature shapes differ")
    a, b = a.ravel(), b.ravel()
    sa, sb = a.std(), b.std()
    if sa < MIN_STD or sb < MIN_STD:
        return None
    r = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    return max(-1.0, min(1.0, r))


def tile_phase_shifts(reference, moving):
    """Fixed 3x3 floor-bounded tiles, 4 px rim discarded, Hann-windowed cv2.phaseCorrelate on COPIES.

    dx, dy are exactly cv2.phaseCorrelate(reference_tile, moving_tile) (OpenCV sign convention).
    A tile is usable iff both stds >= 1e-4, response >= 0.2 and |dx|, |dy| <= 8. Unusable tiles keep
    their raw values (None only where no correlation was computed or OpenCV returned non-finite).
    """
    ref = _image(reference, "reference")
    mov = _image(moving, "moving")
    _need(ref.shape == mov.shape, "reference/moving shapes differ")
    H, W = ref.shape
    ys = [math.floor(i * H / GRID) for i in range(GRID + 1)]
    xs = [math.floor(i * W / GRID) for i in range(GRID + 1)]
    _need(min(b - a for a, b in zip(ys, ys[1:])) - 2 * RIM >= MIN_TILE_INTERIOR
          and min(b - a for a, b in zip(xs, xs[1:])) - 2 * RIM >= MIN_TILE_INTERIOR, "image too small for 3x3 tiles")
    records = []
    for i in range(GRID):
        for j in range(GRID):
            y0, y1, x0, x1 = ys[i] + RIM, ys[i + 1] - RIM, xs[j] + RIM, xs[j + 1] - RIM
            a = np.array(ref[y0:y1, x0:x1], dtype=np.float64, copy=True)
            b = np.array(mov[y0:y1, x0:x1], dtype=np.float64, copy=True)
            sa, sb = float(a.std()), float(b.std())
            rec = {"row": i, "col": j, "bounds_yx": [y0, y1, x0, x1], "std_reference": sa, "std_moving": sb,
                   "dx": None, "dy": None, "response": None, "usable": False, "reasons": []}
            if sa < MIN_STD or sb < MIN_STD:
                rec["reasons"].append("low feature std")
            else:
                window = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_64F)
                (dx, dy), response = cv2.phaseCorrelate(a, b, window)
                if all(math.isfinite(v) for v in (dx, dy, response)):
                    rec.update(dx=float(dx), dy=float(dy), response=float(response))
                    if response < MIN_RESPONSE:
                        rec["reasons"].append("low response")
                    if abs(dx) > MAX_SHIFT or abs(dy) > MAX_SHIFT:
                        rec["reasons"].append("shift beyond 8 px")
                    rec["usable"] = not rec["reasons"]
                else:
                    rec["reasons"].append("non-finite phase correlation")
            records.append(rec)
    return records


def summarize_shifts(records):
    """Descriptive only: usable tile count and median dx/dy (None if no usable tile)."""
    usable = [r for r in records if r["usable"]]
    if not usable:
        return {"usable_tiles": 0, "tiles": len(records), "median_dx": None, "median_dy": None}
    return {"usable_tiles": len(usable), "tiles": len(records),
            "median_dx": float(np.median([r["dx"] for r in usable])), "median_dy": float(np.median([r["dy"] for r in usable]))}
