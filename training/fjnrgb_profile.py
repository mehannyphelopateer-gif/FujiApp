"""Strict research codec for ONE contract: fit-derived Kelvin 6600 neutral-WB profile inputs.

UNSELECTED RESEARCH CODE. Not a runtime adapter, not a v2 extension, and nothing here selects a model.
It never edits, monkeypatches or relabels the v2 modules (training/fjlrg_v2.py, scripts/lib/fjlrg-v2.mjs);
it only reuses their public pure helpers (strict_json_loads, expected_working_size).

Contract `research-linear-fitprofile-k6600`, sidecar schema `fujiapp.raw-input.research-neutralwb.v1`.
Valid only for X100VI sources with WB mode 4080, Kelvin 6600 and raw fine-tune [20, -60], decoded with
camera WB off and userMul = the frozen fit-derived profile [620, 302, 455, 302] (profile 4080/6600 of
wb-alignment manifest 6f6dac1c...). The profile is a fit-only metadata calibration candidate; the
sensor primitive (sparse + full-native v2 reports bound below) showed only that declared multipliers are
applied to unmixed native photosites on 3 fit sources. Nothing here qualifies demosaiced RGB, Fuji
rendering, other modes/Kelvin values, or any runtime use.

Container (little-endian, 64-byte header, same field geometry as v2, then interleaved uint16 RGB):

    0   8   magic b"FJNRGB16"          32 u16 transfer        1 = linear (six-element gamm)
    8   u16 format version = 1          34 u16 primaries       1 = sRGB (LibRaw outputColor 1)
    10  u16 header bytes = 64           36 u16 wb_policy       2 = explicit userMul
    12  u32 width (long edge <= 1536)   38 u16 wb_coeff_source 3 = frozen fit-derived profile
    16  u32 height                      40 4*u32 declared userMul levels RGBG = 620, 302, 455, 302
    20  u16 channels = 3                56 8   first 8 bytes of SHA256(options_json UTF-8)
    22  u16 bits = 16
    24  u32 payload bytes = w*h*6
    28  u32 flags = 3

Header levels are the DECLARED userMul (parameters_are_declared: true), never an applied-parameter
readback. Source camera metadata is not in the codec; the decoding job must re-read it before decode.
Validation fails closed: missing, unknown, boolean, non-finite or ill-typed fields are refused, and the
caller must supply the expected source RAF and decoder WASM SHA-256.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fjlrg_v2 import FjlrgV2Error, expected_working_size as _v2_expected_working_size, strict_json_loads as _v2_strict_json_loads

# --- Frozen contract constants (exported for cross-language fixtures) ------------------------------

MAGIC = b"FJNRGB16"
FORMAT_VERSION = 1
HEADER_BYTES = 64
CHANNELS = 3
BITS_PER_CHANNEL = 16
FLAGS = 3
TRANSFER_LINEAR = 1
PRIMARIES_SRGB = 1
WB_POLICY_USER_MUL = 2
COEFF_SOURCE_FIT_PROFILE = 3
MAX_WORKING_LONG_EDGE = 1536
HEADER_FORMAT = "<8sHHIIHHIIHHHH4I8s"
assert struct.calcsize(HEADER_FORMAT) == HEADER_BYTES

SCHEMA = "fujiapp.raw-input.research-neutralwb.v1"
CONTRACT = "research-linear-fitprofile-k6600"
WB_SOURCE_KIND = "fit_profile"
PROFILE_LEVELS_RGBG = (620, 302, 455, 302)
PROFILE_MANIFEST_SHA256 = "6f6dac1c4bdd619ae1daacac36581866b1b71b059fc970f23145b89ffbed8ffc"
PROFILE_KEY = "4080/6600"
POLICY_BASIS = "calibrated-kelvin-profile-candidate"
PRIMITIVE_REPORTS = {
    "sparse_primitive_report_sha256": "4fcb31eb72b7e41f3959d360c14c868501e31923142fc025a001d5b3581f213f",
    "fullnative_v2_report_sha256": "d320a1a656a7c16ab2324b4ac1046b469f4d0f57fe03dd3c0aa36cf27a458027",
    "fullnative_v2_verification_sha256": "0900736496e9967d45d936e300946aa6a702427a572a6f91e9b310d40038de32",
}
SOURCE_WB_MODE = 4080
SOURCE_KELVIN = 6600
SOURCE_FINE_TUNE_RAW = (20, -60)
DECODER_PACKAGE = "libraw-wasm"
DECODER_VERSION = "1.6.0"
LIBRAW_BUILD_VERSION = "0.22.1"
CAPTURE_DYNAMIC_RANGES = (100, 200, 400)
CONTAINER_SUFFIX = ".fjnrgb"
SIDECAR_SUFFIX = ".json"

# Exactly the v2 sidecar key set plus four profile keys (asserted against v2 in the tests).
V2_SIDECAR_KEYS = frozenset({
    "schema", "input_contract", "container_sha256", "source_raf_sha256", "wasm_sha256",
    "decoder_package", "decoder_version", "libraw_build_version", "options_json",
    "options_sha256", "declared_options", "wb_source", "source_wb_mode",
    "source_wb_fine_tune_raw", "source_kelvin", "capture_dynamic_range",
    "capture_dynamic_range_source", "decoded_native_size", "parameters_are_declared",
})
PROFILE_SIDECAR_KEYS = frozenset({"profile_manifest_sha256", "profile_key", "policy_basis", "primitive_reports"})
SIDECAR_KEYS = V2_SIDECAR_KEYS | PROFILE_SIDECAR_KEYS
WB_SOURCE_KEYS = frozenset({"kind", "levels_rgbg"})

# Identical to the v2 common options (asserted in tests), then camera WB off + profile userMul.
_COMMON_OPTIONS: tuple[tuple[str, Any], ...] = (
    ("halfSize", False), ("outputBps", 16), ("outputColor", 1), ("gamm", [1, 1, 0, 0, 0, 0]),
    ("noAutoBright", True), ("adjustMaximumThr", 0), ("useCameraMatrix", 1), ("useCameraWb", True),
    ("useAutoWb", False), ("highlight", 0), ("userQual", 3), ("useFujiRotate", -1), ("fbddNoiserd", 0),
)

_HEX64 = re.compile(r"[0-9a-f]{64}")


class FjnrgbProfileError(ValueError):
    """Any container, sidecar or contract violation."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise FjnrgbProfileError(message)


def _is_number(v: Any) -> bool:
    return type(v) is int or (type(v) is float and math.isfinite(v))


def _as_int(v: Any, name: str) -> int:
    _require(_is_number(v) and (type(v) is int or v.is_integer()), f"{name} must be an integer-valued finite number (not bool)")
    return int(v)


def _json_equal(a: Any, b: Any) -> bool:
    if _is_number(a) or _is_number(b):
        return _is_number(a) and _is_number(b) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return False
    return a == b


def _sha(value: Any, name: str) -> str:
    _require(isinstance(value, str) and _HEX64.fullmatch(value) is not None, f"{name} must be 64 lowercase hex")
    return value


def strict_json_loads(text: str) -> Any:
    try:
        return _v2_strict_json_loads(text)
    except FjlrgV2Error as exc:
        raise FjnrgbProfileError(str(exc)) from exc


def expected_working_size(native_width: int, native_height: int) -> tuple[int, int]:
    native_width = _as_int(native_width, "native width")
    native_height = _as_int(native_height, "native height")
    _require(1 <= native_width <= 0xffffffff and 1 <= native_height <= 0xffffffff,
             "Native size must fit positive uint32 (same as JS)")
    try:
        return _v2_expected_working_size(native_width, native_height)
    except FjlrgV2Error as exc:
        raise FjnrgbProfileError(str(exc)) from exc


# --- Options -----------------------------------------------------------------------------------

def canonical_options() -> dict[str, Any]:
    """The only accepted decode options: v2 common options, camera WB off, userMul = profile."""
    options = {key: copy.deepcopy(value) for key, value in _COMMON_OPTIONS}
    options["useCameraWb"] = False
    options["userMul"] = list(PROFILE_LEVELS_RGBG)
    return options


def serialize_options(options: Mapping[str, Any]) -> str:
    """Same writer convention as scripts/lib/fjlrg-v2.mjs serializeOptions (sorted top-level keys)."""
    _require(isinstance(options, dict), "options must be an object")
    return json.dumps({k: options[k] for k in sorted(options)}, separators=(",", ":"), allow_nan=False)


def options_sha256(options_json: str) -> str:
    _require(isinstance(options_json, str), "options_json must be a str")
    return hashlib.sha256(options_json.encode("utf-8")).hexdigest()


# --- Writer ------------------------------------------------------------------------------------

def encode(width: int, height: int, pixels: Any, *, options_sha256: str) -> bytes:
    """Encode a profile container (bytes only; the writer saves the sidecar)."""
    width, height = _as_int(width, "width"), _as_int(height, "height")
    _require(width > 0 and height > 0 and max(width, height) <= MAX_WORKING_LONG_EDGE, "Invalid working dimensions")
    arr = np.asarray(pixels)
    _require(arr.dtype.kind == "u" and arr.dtype.itemsize == 2, "Pixels must be uint16")
    _require(arr.shape == (height, width, CHANNELS), "Pixels must have shape (height, width, 3)")
    prefix = bytes.fromhex(_sha(options_sha256, "options_sha256"))[:8]
    payload = np.ascontiguousarray(arr, dtype="<u2").tobytes()
    header = struct.pack(HEADER_FORMAT, MAGIC, FORMAT_VERSION, HEADER_BYTES, width, height, CHANNELS,
                         BITS_PER_CHANNEL, len(payload), FLAGS, TRANSFER_LINEAR, PRIMARIES_SRGB,
                         WB_POLICY_USER_MUL, COEFF_SOURCE_FIT_PROFILE, *PROFILE_LEVELS_RGBG, prefix)
    return header + payload


def build_provenance(container: bytes, *, source_raf_sha256: str, wasm_sha256: str, capture_dynamic_range: int,
                     capture_dynamic_range_source: str, decoded_native_size: list[int],
                     options_json: str | None = None) -> dict[str, Any]:
    """Canonical sidecar for a container (writer and fixture helper). Validated before return."""
    options_json = serialize_options(canonical_options()) if options_json is None else options_json
    provenance = {
        "schema": SCHEMA, "input_contract": CONTRACT, "container_sha256": hashlib.sha256(bytes(container)).hexdigest(),
        "source_raf_sha256": source_raf_sha256, "wasm_sha256": wasm_sha256,
        "decoder_package": DECODER_PACKAGE, "decoder_version": DECODER_VERSION, "libraw_build_version": LIBRAW_BUILD_VERSION,
        "options_json": options_json, "options_sha256": options_sha256(options_json),
        "declared_options": strict_json_loads(options_json),
        "wb_source": {"kind": WB_SOURCE_KIND, "levels_rgbg": list(PROFILE_LEVELS_RGBG)},
        "source_wb_mode": SOURCE_WB_MODE, "source_wb_fine_tune_raw": list(SOURCE_FINE_TUNE_RAW), "source_kelvin": SOURCE_KELVIN,
        "capture_dynamic_range": capture_dynamic_range, "capture_dynamic_range_source": capture_dynamic_range_source,
        "decoded_native_size": list(decoded_native_size), "parameters_are_declared": True,
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256, "profile_key": PROFILE_KEY, "policy_basis": POLICY_BASIS,
        "primitive_reports": dict(PRIMITIVE_REPORTS),
    }
    decode(container, provenance, expected_source_sha256=source_raf_sha256, expected_wasm_sha256=wasm_sha256)
    return provenance


# --- Reader ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Header:
    width: int
    height: int
    options_sha256_prefix: bytes


@dataclass(frozen=True)
class DecodedFjnrgbProfile:
    """Validated profile input; pixels are read-only linear uint16 RGB (H, W, 3), sRGB primaries."""
    width: int
    height: int
    rgb_uint16: np.ndarray
    provenance: dict
    input_contract: str = CONTRACT

    def to_unit_float32(self) -> np.ndarray:
        return self.rgb_uint16.astype(np.float32) / np.float32(65535.0)


def parse_header(data: bytes) -> Header:
    _require(isinstance(data, (bytes, bytearray, memoryview)), "Container must be bytes")
    data = bytes(data)
    _require(len(data) >= HEADER_BYTES, "Container shorter than 64-byte header")
    (magic, version, header_bytes, width, height, channels, bits, payload_bytes, flags, transfer, primaries,
     policy, source, r, g, b, g2, prefix) = struct.unpack_from(HEADER_FORMAT, data, 0)
    _require(magic == MAGIC, f"Bad magic {magic!r} (v1/v2 FJLRGB16 refused)")
    _require(version == FORMAT_VERSION, f"Unsupported format version {version}")
    _require(header_bytes == HEADER_BYTES, "Header size != 64")
    _require(width > 0 and height > 0 and max(width, height) <= MAX_WORKING_LONG_EDGE, "Invalid working dimensions")
    _require(channels == CHANNELS and bits == BITS_PER_CHANNEL, "Channels/bits differ")
    _require(flags == FLAGS, "Flags differ")
    _require(transfer == TRANSFER_LINEAR and primaries == PRIMARIES_SRGB, "Transfer/primaries differ")
    _require((policy, source) == (WB_POLICY_USER_MUL, COEFF_SOURCE_FIT_PROFILE), f"Invalid WB policy/source ({policy}, {source})")
    _require((r, g, b, g2) == PROFILE_LEVELS_RGBG, "Header levels are not the frozen profile")
    expected_payload = width * height * CHANNELS * 2
    _require(payload_bytes == expected_payload, "Declared payload size mismatch")
    _require(len(data) == HEADER_BYTES + expected_payload, "Container length mismatch (truncated or trailing bytes)")
    return Header(width, height, prefix)


def _validate(p: Any, header: Header, data: bytes, expected_source_sha256: str, expected_wasm_sha256: str) -> None:
    _require(isinstance(p, dict), "Provenance must be a JSON object")
    keys = set(p)
    _require(keys == SIDECAR_KEYS, f"Sidecar keys differ: missing {sorted(SIDECAR_KEYS - keys)}, unknown {sorted(keys - SIDECAR_KEYS)}")
    _require(p["schema"] == SCHEMA, "Wrong sidecar schema")
    _require(p["input_contract"] == CONTRACT, "Wrong input contract")
    _require(_sha(p["container_sha256"], "container_sha256") == hashlib.sha256(data).hexdigest(), "Container SHA256 mismatch")
    _require(_sha(p["source_raf_sha256"], "source_raf_sha256") == _sha(expected_source_sha256, "expected_source_sha256"),
             "Source RAF SHA256 differs from expected")
    _require(_sha(p["wasm_sha256"], "wasm_sha256") == _sha(expected_wasm_sha256, "expected_wasm_sha256"),
             "Decoder wasm SHA256 differs from expected")
    _require((p["decoder_package"], p["decoder_version"], p["libraw_build_version"])
             == (DECODER_PACKAGE, DECODER_VERSION, LIBRAW_BUILD_VERSION), "Wrong decoder provenance")
    _require(p["parameters_are_declared"] is True, "parameters_are_declared must be literal true")

    wb = p["wb_source"]
    _require(isinstance(wb, dict) and set(wb) == WB_SOURCE_KEYS, "wb_source must have exactly kind, levels_rgbg")
    _require(wb["kind"] == WB_SOURCE_KIND, "wb_source.kind must be fit_profile")
    _require(_json_equal(wb["levels_rgbg"], list(PROFILE_LEVELS_RGBG)), "wb_source.levels_rgbg is not the frozen profile")

    _require(isinstance(p["options_json"], str), "options_json must be a string")
    digest = options_sha256(p["options_json"])
    _require(_sha(p["options_sha256"], "options_sha256") == digest, "options_sha256 does not hash options_json")
    _require(bytes.fromhex(digest)[:8] == header.options_sha256_prefix, "Header options prefix mismatch")
    parsed = strict_json_loads(p["options_json"])
    _require(_json_equal(parsed, p["declared_options"]), "declared_options differ from parsed options_json")
    _require(_json_equal(parsed, canonical_options()), "Options are not the canonical profile set")

    _require(_as_int(p["source_wb_mode"], "source_wb_mode") == SOURCE_WB_MODE, "source_wb_mode must be 4080")
    _require(p["source_kelvin"] is not None and _as_int(p["source_kelvin"], "source_kelvin") == SOURCE_KELVIN,
             "source_kelvin must be 6600")
    fine = p["source_wb_fine_tune_raw"]
    _require(isinstance(fine, list) and len(fine) == 2
             and tuple(_as_int(v, "source_wb_fine_tune_raw") for v in fine) == SOURCE_FINE_TUNE_RAW,
             "source_wb_fine_tune_raw must be [20, -60]")
    _require(_as_int(p["capture_dynamic_range"], "capture_dynamic_range") in CAPTURE_DYNAMIC_RANGES, "capture_dynamic_range must be 100/200/400")
    _require(isinstance(p["capture_dynamic_range_source"], str) and p["capture_dynamic_range_source"].strip() != "",
             "capture_dynamic_range_source must be non-empty")
    native = p["decoded_native_size"]
    _require(isinstance(native, list) and len(native) == 2, "decoded_native_size must be 2 values")
    native = [_as_int(v, "decoded_native_size") for v in native]
    _require(all(v > 0 for v in native), "decoded_native_size must be positive")
    _require(expected_working_size(native[0], native[1]) == (header.width, header.height),
             "Working size is not the canonical downsample of decoded_native_size")

    _require(p["profile_manifest_sha256"] == PROFILE_MANIFEST_SHA256, "Wrong profile manifest")
    _require(p["profile_key"] == PROFILE_KEY, "Wrong profile key")
    _require(p["policy_basis"] == POLICY_BASIS, "Wrong policy basis")
    reports = p["primitive_reports"]
    _require(isinstance(reports, dict) and reports == PRIMITIVE_REPORTS, "primitive_reports differ from the frozen hashes")


def decode(data: bytes, provenance: Mapping[str, Any], *, expected_source_sha256: str,
           expected_wasm_sha256: str) -> DecodedFjnrgbProfile:
    """Validate a profile container against its sidecar and the caller's expected source/WASM hashes."""
    data = bytes(data)
    header = parse_header(data)
    _validate(provenance, header, data, expected_source_sha256, expected_wasm_sha256)
    count = header.width * header.height * CHANNELS
    rgb = np.frombuffer(data, dtype="<u2", count=count, offset=HEADER_BYTES).astype(np.uint16)
    rgb = rgb.reshape(header.height, header.width, CHANNELS)
    rgb.setflags(write=False)
    return DecodedFjnrgbProfile(header.width, header.height, rgb, copy.deepcopy(dict(provenance)))


def sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    _require(path.suffix == CONTAINER_SUFFIX, "Profile container path must end in .fjnrgb")
    return path.with_name(path.name + SIDECAR_SUFFIX)


def load(path: str | Path, *, expected_source_sha256: str, expected_wasm_sha256: str) -> DecodedFjnrgbProfile:
    side = sidecar_path(path)
    _require(side.is_file(), f"Missing sidecar {side.name}")
    try:
        text = side.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FjnrgbProfileError("Sidecar is not valid UTF-8") from exc
    return decode(Path(path).read_bytes(), strict_json_loads(text),
                  expected_source_sha256=expected_source_sha256, expected_wasm_sha256=expected_wasm_sha256)
