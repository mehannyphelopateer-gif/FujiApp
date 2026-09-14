"""Reader for the .fjlrg linear-RGB container (docs/phase3-linear-rgb-format.md).

Python port of scripts/lib/phase3-linear-rgb.mjs's loadFjlrgLinear — same
format, same convention (uint16 samples normalized to 0..1 linear light,
sRGB primaries, as-shot WB). Keep this in sync with the .mjs version if the
container format ever changes.
"""
import struct
import numpy as np

FJLRGB_MAGIC = b"FJLRGB16"


def load_fjlrg_linear(path):
    """Returns (data, width, height) where data is a float32 HxWx3 array,
    0..1 normalized linear-light RGB."""
    with open(path, "rb") as f:
        buf = f.read()

    magic = buf[0:8]
    if magic != FJLRGB_MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}, expected {FJLRGB_MAGIC!r}")

    version, header_bytes = struct.unpack_from("<HH", buf, 8)
    if version != 1:
        raise ValueError(f"{path}: unsupported format version {version}")
    width, height = struct.unpack_from("<II", buf, 12)
    channels, bits_per_channel = struct.unpack_from("<HH", buf, 20)
    payload_byte_count, flags = struct.unpack_from("<II", buf, 24)
    is_linear = bool(flags & 1)

    if channels != 3:
        raise ValueError(f"{path}: expected 3 channels, got {channels}")
    if bits_per_channel != 16:
        raise ValueError(f"{path}: expected 16 bits/channel, got {bits_per_channel}")
    if not is_linear:
        raise ValueError(f"{path}: flags indicate non-linear transfer - this loader assumes linear light")

    expected_payload = width * height * channels * 2
    actual_payload = len(buf) - header_bytes
    if payload_byte_count != expected_payload or actual_payload != expected_payload:
        raise ValueError(
            f"{path}: payload size mismatch (declared {payload_byte_count}, "
            f"expected {expected_payload}, actual {actual_payload})"
        )

    raw = np.frombuffer(buf, dtype="<u2", count=width * height * channels, offset=header_bytes)
    data = raw.astype(np.float32).reshape(height, width, channels) / 65535.0
    return data, width, height
