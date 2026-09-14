"""Reader for the X RAW Studio target TIFF (docs/phase3-export-protocol.md).

Loads the full-sensor-resolution 16-bit sRGB TIFF, converts to linear light,
applies the EXIF orientation the raw TIFF storage doesn't bake in, and
area-averages down to match a given .fjlrg's resolution - mirrors
scripts/lib/phase3-linear-rgb.mjs's loadXrawTargetLinear +
resizeLinearRgbTo, but uses tifffile (reads true 16-bit samples directly,
no sips/PNG workaround needed - unlike sharp/libvips, tifffile doesn't
silently truncate this project's TIFFs to 8-bit) and cv2.INTER_AREA for the
downsample (a standard, fast area-average equivalent, not a literal port
of the JS's nested-loop version).
"""
import numpy as np
import tifffile
import cv2

# 8-bit JPEG fallback path (only used if a scene's X RAW Studio export
# genuinely isn't 16-bit TIFF - see docs/phase3-export-protocol.md).
from PIL import Image


def _srgb_gamma_to_linear(c):
    # IEC 61966-2-1, same formula as phase3-linear-rgb.mjs's srgbGammaToLinear.
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _apply_orientation(arr, orientation):
    """arr: HxWx3 array. orientation: standard EXIF tag value 1-8.
    Only the four pure-rotation cases are handled (1,3,6,8) - matches
    phase3-linear-rgb.mjs's applyExifRotation, which also rejects the
    mirrored cases as unexpected for direct camera capture."""
    if orientation == 1:
        return arr
    if orientation == 3:
        return np.rot90(arr, 2)
    if orientation == 6:
        # rotate 90 CW
        return np.rot90(arr, -1)
    if orientation == 8:
        # rotate 90 CCW (270 CW)
        return np.rot90(arr, 1)
    raise ValueError(
        f"Unsupported EXIF orientation {orientation} (mirrored orientations "
        f"2/4/5/7 aren't handled - unusual for a direct camera capture)."
    )


def load_xraw_target_linear(path):
    """Returns (data, width, height) float32 linear-light RGB, full
    (oriented) sensor resolution - caller downsamples to match the input."""
    lower = str(path).lower()
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        arr = tifffile.imread(path)
        if arr.dtype != np.uint16:
            raise ValueError(f"{path}: expected 16-bit TIFF (uint16), got {arr.dtype}")
        with tifffile.TiffFile(path) as tf:
            orientation_tag = tf.pages[0].tags.get("Orientation")
            orientation = int(orientation_tag.value) if orientation_tag is not None else 1
        linear = _srgb_gamma_to_linear(arr.astype(np.float32) / 65535.0)
        oriented = _apply_orientation(linear, orientation)
    else:
        # 8-bit JPEG fallback.
        im = Image.open(path)
        exif = im.getexif()
        orientation = exif.get(274, 1)
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.float32) / 255.0
        linear = _srgb_gamma_to_linear(arr)
        oriented = _apply_orientation(linear, orientation)

    h, w = oriented.shape[:2]
    return np.ascontiguousarray(oriented), w, h


def resize_linear_rgb_to(data, target_width, target_height):
    """Area-average resize of linear-light HxWx3 float32 data - averaging
    in LINEAR light (not gamma-encoded values) is the physically correct
    way to downsample, matching resizeLinearRgbTo's convention."""
    h, w = data.shape[:2]
    if w == target_width and h == target_height:
        return data
    resized = cv2.resize(data, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def load_xraw_target_linear_matching(path, target_width, target_height):
    """Convenience: load + downsample to a given (fjlrg) resolution in one call."""
    data, w, h = load_xraw_target_linear(path)
    return resize_linear_rgb_to(data, target_width, target_height)
