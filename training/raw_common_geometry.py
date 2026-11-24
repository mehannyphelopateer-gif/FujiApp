"""Pure stdlib geometry for a source-derived common-sensor-rectangle registration CANDIDATE.

No file I/O, no NumPy, no pixels. Inputs: LibRaw's own returned sizes for one source (raw_width/height,
iwidth/iheight as the UNROTATED visible size, left/top_margin as the raw origin, width/height after
rotation, flip) and the RAF header crop descriptor (RawImageCropTopLeft / RawImageCroppedSize, stored
height,width order). The header crop size INCLUDES BORDERS per the reference; treating it as Fuji's
rendered frame is a hypothesis to be measured, never an applied geometry fix. raw_inset_crops and any
extra fields (GPS, target hints) are ignored and cannot drive the result.

common_sensor_windows(...) intersects the LibRaw visible rectangle with the header crop rectangle in raw
sensor coordinates, then expresses that common rectangle as a normalized window inside each frame,
first unrotated and then in the displayed orientation (flip 0 -> orientation 1, 5 -> 8, 6 -> 6).
"""
from fractions import Fraction

MAX_DIM = 65535
FLIP_TO_ORIENTATION = {0: 1, 5: 8, 6: 6}
GEOMETRY_KEYS = ("raw_width", "raw_height", "iwidth", "iheight", "left_margin", "top_margin", "width", "height", "flip")


class CommonGeometryError(ValueError):
    pass


def _need(cond, msg):
    if not cond:
        raise CommonGeometryError(msg)


def _int(v, name, *, zero_ok=False):
    _need(type(v) is int, f"{name} must be an int (bool/float refused)")
    _need((v >= 0 if zero_ok else v > 0) and v <= MAX_DIM, f"{name} must be {'0' if zero_ok else '1'}..{MAX_DIM}, got {v}")
    return v


def _pair(v, name, *, zero_ok=False):
    _need(isinstance(v, (list, tuple)) and len(v) == 2, f"{name} must be [height-axis, width-axis]")
    return _int(v[0], f"{name}[0]", zero_ok=zero_ok), _int(v[1], f"{name}[1]", zero_ok=zero_ok)


def _orient(window, orientation):
    x0, y0, x1, y1 = window
    if orientation == 1:
        return [x0, y0, x1, y1]
    if orientation == 8:
        return [y0, 1 - x1, y1, 1 - x0]
    return [1 - y1, x0, 1 - y0, x1]  # orientation 6


def _window_in(rect, inner):
    x, y, w, h = rect
    ix, iy, iw, ih = inner
    win = [Fraction(ix - x, w), Fraction(iy - y, h), Fraction(ix + iw - x, w), Fraction(iy + ih - y, h)]
    _need(0 <= win[0] < win[2] <= 1 and 0 <= win[1] < win[3] <= 1, "window outside its frame")
    return win


def _exact(win):
    return [[f.numerator, f.denominator] for f in win]


def _floats(win):
    return [float(f) for f in win]


def _crop_margins(rect, inner):
    x, y, w, h = rect
    ix, iy, iw, ih = inner
    return {"left": ix - x, "top": iy - y, "right": x + w - (ix + iw), "bottom": y + h - (iy + ih)}


def common_sensor_windows(geometry, raw_crop_top_left_hw, raw_crop_size_hw, target_orientation):
    """Return rectangles (x, y, w, h in raw sensor pixels) and normalized [x0, y0, x1, y1] windows."""
    _need(isinstance(geometry, dict), "geometry must be a dict of LibRaw sizes")
    missing = [k for k in GEOMETRY_KEYS if k not in geometry]
    _need(not missing, f"geometry missing {missing}")
    raw_w, raw_h = _int(geometry["raw_width"], "raw_width"), _int(geometry["raw_height"], "raw_height")
    iw, ih = _int(geometry["iwidth"], "iwidth"), _int(geometry["iheight"], "iheight")
    left, top = _int(geometry["left_margin"], "left_margin", zero_ok=True), _int(geometry["top_margin"], "top_margin", zero_ok=True)
    width, height = _int(geometry["width"], "width"), _int(geometry["height"], "height")
    flip = geometry["flip"]
    _need(type(flip) is int and flip in FLIP_TO_ORIENTATION, f"unsupported LibRaw flip {flip!r}")
    orientation = FLIP_TO_ORIENTATION[flip]
    _need(type(target_orientation) is int and target_orientation == orientation,
          f"target orientation {target_orientation!r} does not match LibRaw flip {flip} (orientation {orientation})")
    _need((width, height) == ((iw, ih) if flip == 0 else (ih, iw)), "LibRaw width/height do not match the (rotated) visible size")
    _need(left + iw <= raw_w and top + ih <= raw_h, "LibRaw visible rectangle exceeds the raw frame")
    ct, cl = _pair(raw_crop_top_left_hw, "raw_crop_top_left_hw", zero_ok=True)
    ch, cw = _pair(raw_crop_size_hw, "raw_crop_size_hw")
    _need(cl + cw <= raw_w and ct + ch <= raw_h, "header crop exceeds the raw frame")

    source = (left, top, iw, ih)
    target = (cl, ct, cw, ch)
    x0, y0 = max(left, cl), max(top, ct)
    x1, y1 = min(left + iw, cl + cw), min(top + ih, ct + ch)
    _need(x1 > x0 and y1 > y0, "LibRaw visible and header crop rectangles do not intersect")
    common = (x0, y0, x1 - x0, y1 - y0)

    src_raw, tgt_raw = _window_in(source, common), _window_in(target, common)
    src_disp, tgt_disp = _orient(src_raw, orientation), _orient(tgt_raw, orientation)
    for w in (src_disp, tgt_disp):
        _need(0 <= w[0] < w[2] <= 1 and 0 <= w[1] < w[3] <= 1, "oriented window outside [0,1]")
    return {
        "common_rect_xywh": list(common), "source_rect_xywh": list(source), "target_rect_xywh": list(target),
        "source_window": _floats(src_disp), "target_window": _floats(tgt_disp),
        "source_window_exact": _exact(src_disp), "target_window_exact": _exact(tgt_disp),
        "source_window_unrotated": _floats(src_raw), "target_window_unrotated": _floats(tgt_raw),
        "source_cropped_px": _crop_margins(source, common), "target_cropped_px": _crop_margins(target, common),
        "flip": flip, "orientation": orientation,
        "labels": {"source": "LibRaw returned visible area (iwidth x iheight at left/top margin)",
                   "target": "RAF header RawImageCropTopLeft/RawImageCroppedSize (includes borders)",
                   "status": "hypothesis candidate for registration comparison only; not an applied geometry fix"},
    }
