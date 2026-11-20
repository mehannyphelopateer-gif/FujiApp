"""Strict research codec for the v2 FJLRGB16 raw-input container.

Design: docs/provia-input-contract-v2-plan.md, section 4. This is a standalone
research codec. It does not replace or touch ``training/fjlrg.py`` (the legacy
v1 reader) or any live Round 11 code, and nothing here selects a model.

Container (little-endian, 64-byte header, then interleaved uint16 RGB):

    offset size  field
    0      8     magic b"FJLRGB16"
    8      u16   format version = 2
    10     u16   header bytes = 64
    12     u32   width   (working resolution, long edge <= 1536)
    16     u32   height
    20     u16   channels = 3
    22     u16   bits per channel = 16
    24     u32   payload bytes = width * height * 6
    28     u32   flags = 3 (kept for legacy-geometry compatibility only)
    32     u16   transfer        1 = linear (six-element LibRaw gamm)
    34     u16   primaries       1 = sRGB via LibRaw rgb_cam (camera RGB rejected)
    36     u16   wb_policy       1 = camera WB, 2 = explicit userMul
    38     u16   wb_coeff_source 1 = cam_mul metadata, 2 = Fuji 0xf00d metadata
    40     4*u32 source WB levels RGBG
    56     8     first 8 bytes of SHA256(options_json UTF-8 bytes)

Every file has a sidecar ``<name>.fjlrg.json`` describing provenance. All
validation fails closed: unknown, missing or ill-typed fields are refused,
and the caller must state which input contract it expects. Nothing is
inferred.

Numbers follow JSON semantics so Python and the JS codec agree: JSON has one
number type, so 1 and 1.0 are the same value. Booleans are never numbers.
Fields that must be integers accept any finite integer-valued number and are
coerced to int after validation; NaN, Infinity and non-integral values are
refused.

Header WB levels are *metadata* for the as-shot contract (the decoder used
camera WB without userMul) and the *declared* userMul for the Auto-WB
contract. They are never a readback of parameters LibRaw actually applied.
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
from typing import Any, Mapping, Sequence

import numpy as np

# --- Container constants (exported for cross-language fixtures) -------------

MAGIC = b"FJLRGB16"
FORMAT_VERSION = 2
HEADER_BYTES = 64
CHANNELS = 3
BITS_PER_CHANNEL = 16
FLAGS = 3
TRANSFER_LINEAR = 1
PRIMARIES_SRGB = 1
WB_POLICY_CAMERA = 1
WB_POLICY_USER_MUL = 2
COEFF_SOURCE_CAM_MUL = 1
COEFF_SOURCE_FUJI_F00D = 2
MAX_WORKING_LONG_EDGE = 1536
U32_MAX = 0xFFFFFFFF

HEADER_FORMAT = "<8sHHIIHHIIHHHH4I8s"
assert struct.calcsize(HEADER_FORMAT) == HEADER_BYTES

# --- Contract constants ------------------------------------------------------

CONTRACT_ASSHOT = "v2-linear-asshot"
CONTRACT_AUTOWB = "v2-linear-autowb"

# contract -> (header wb_policy, header wb_coeff_source, sidecar wb_source.kind)
CONTRACT_FIELDS: Mapping[str, tuple[int, int, str]] = {
    CONTRACT_ASSHOT: (WB_POLICY_CAMERA, COEFF_SOURCE_CAM_MUL, "cam_mul"),
    CONTRACT_AUTOWB: (WB_POLICY_USER_MUL, COEFF_SOURCE_FUJI_F00D, "fuji_f00d"),
}

SIDECAR_SCHEMA = "fujiapp.raw-input.v2"
SIDECAR_SUFFIX = ".json"
DECODER_PACKAGE = "libraw-wasm"
DECODER_VERSION = "1.6.0"
LIBRAW_BUILD_VERSION = "0.22.1"

SIDECAR_KEYS = frozenset({
    "schema", "input_contract", "container_sha256", "source_raf_sha256", "wasm_sha256",
    "decoder_package", "decoder_version", "libraw_build_version", "options_json",
    "options_sha256", "declared_options", "wb_source", "source_wb_mode",
    "source_wb_fine_tune_raw", "source_kelvin", "capture_dynamic_range",
    "capture_dynamic_range_source", "decoded_native_size", "parameters_are_declared",
})
WB_SOURCE_KEYS = frozenset({"kind", "levels_rgbg"})
CAPTURE_DYNAMIC_RANGES = (100, 200, 400)
FINE_TUNE_RAW_STEP = 20
FINE_TUNE_RAW_LIMIT = 180

# Decode options shared by both contracts, in the order the exporter declares them.
_COMMON_OPTIONS: tuple[tuple[str, Any], ...] = (
    ("halfSize", False),
    ("outputBps", 16),
    ("outputColor", 1),
    ("gamm", [1, 1, 0, 0, 0, 0]),
    ("noAutoBright", True),
    ("adjustMaximumThr", 0),
    ("useCameraMatrix", 1),
    ("useCameraWb", True),
    ("useAutoWb", False),
    ("highlight", 0),
    ("userQual", 3),
    ("useFujiRotate", -1),
    ("fbddNoiserd", 0),
)

_HEX64 = re.compile(r"[0-9a-f]{64}")


class FjlrgV2Error(ValueError):
    """Raised for any container, sidecar or contract violation."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise FjlrgV2Error(message)


def _is_number(value: Any) -> bool:
    """A finite JSON number. bool is deliberately excluded."""
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def _is_int(value: Any) -> bool:
    """A finite integer-valued JSON number (1 or 1.0). bool is excluded."""
    return _is_number(value) and (type(value) is int or value.is_integer())


def _as_int(value: Any, name: str) -> int:
    """Validate an integer-valued JSON number and return it as int."""
    _require(_is_int(value), f"{name} must be an integer-valued finite number (not bool)")
    return int(value)


def _json_equal(a: Any, b: Any) -> bool:
    """Deep equality under JSON semantics.

    Numbers compare by value (1 == 1.0); booleans, strings, null, lists and
    objects compare only with their own kind, so True never equals 1.
    Non-finite floats never compare equal.
    """
    if _is_number(a) or _is_number(b):
        return _is_number(a) and _is_number(b) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float):  # non-finite on both sides
        return False
    return a == b


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in out, f"Duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise FjlrgV2Error(f"Non-finite JSON constant refused: {name}")


def strict_json_loads(text: str) -> Any:
    """Parse JSON refusing duplicate keys and NaN/Infinity."""
    _require(isinstance(text, str), "JSON input must be a str")
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise FjlrgV2Error(f"Invalid JSON: {exc}") from exc


def _check_sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and _HEX64.fullmatch(value) is not None,
             f"{name} must be 64 lowercase hex characters")
    return value


def _contract_fields(contract: Any) -> tuple[int, int, str]:
    _require(isinstance(contract, str) and contract in CONTRACT_FIELDS,
             f"Unknown input contract: {contract!r}")
    return CONTRACT_FIELDS[contract]


def expected_working_size(native_width: int, native_height: int) -> tuple[int, int]:
    """Working size produced by the canonical phase-3 area average.

    Mirrors the exporter's JS exactly: scale = min(1, 1536 / long edge), then
    Math.round (half-up), with a minimum of 1.
    """
    native_width = _as_int(native_width, "native width")
    native_height = _as_int(native_height, "native height")
    _require(native_width > 0 and native_height > 0, "Native size must be positive")
    scale = min(1.0, MAX_WORKING_LONG_EDGE / max(native_width, native_height))

    def js_round(x: float) -> int:
        return int(np.floor(x + 0.5))

    return max(1, js_round(native_width * scale)), max(1, js_round(native_height * scale))


def check_wb_levels(levels: Any, contract: str) -> tuple[int, int, int, int]:
    """Validate RGBG WB levels for a contract and return them as a tuple.

    R, G and B must be positive uint32. The fourth slot must equal G, except
    that the as-shot contract also allows 0 (LibRaw leaves cam_mul[3] at 0
    when only 0xf00e supplies it).
    """
    _contract_fields(contract)
    _require(isinstance(levels, (list, tuple)) and len(levels) == 4, "WB levels must be 4 values (RGBG)")
    r, g, b, g2 = (_as_int(v, "WB level") for v in levels)
    _require(all(0 <= v <= U32_MAX for v in (r, g, b, g2)), "WB levels must fit uint32")
    _require(r > 0 and g > 0 and b > 0, "WB R/G/B levels must be positive")
    allowed_fourth = {g, 0} if contract == CONTRACT_ASSHOT else {g}
    _require(g2 in allowed_fourth, "WB fourth level must equal G (or 0 for as-shot)")
    return r, g, b, g2


def canonical_options(contract: str, wb_levels_rgbg: Sequence[int]) -> dict[str, Any]:
    """Return the only decode-options object accepted for a contract.

    As-shot: camera WB, no userMul. Auto-WB: camera WB off, userMul equal to
    the header levels. Returned fresh each call so callers cannot mutate a
    shared constant.
    """
    levels = check_wb_levels(wb_levels_rgbg, contract)
    options = {key: copy.deepcopy(value) for key, value in _COMMON_OPTIONS}
    if contract == CONTRACT_AUTOWB:
        options["useCameraWb"] = False
        options["userMul"] = list(levels)
    return options


def options_sha256(options_json: str) -> str:
    """SHA256 of the literal UTF-8 bytes of options_json (never re-serialised)."""
    _require(isinstance(options_json, str), "options_json must be a str")
    return hashlib.sha256(options_json.encode("utf-8")).hexdigest()


# --- Writer -------------------------------------------------------------------

def encode_fjlrg_v2(width: int, height: int, pixels: Any, *, input_contract: str,
                    wb_levels_rgbg: Sequence[int], options_sha256: str) -> bytes:
    """Encode a v2 container. Returns bytes only; the exporter writes the sidecar."""
    policy, source, _ = _contract_fields(input_contract)
    width, height = _as_int(width, "width"), _as_int(height, "height")
    _require(width > 0 and height > 0, "Width/height must be positive")
    _require(max(width, height) <= MAX_WORKING_LONG_EDGE, "Working long edge exceeds 1536")
    arr = np.asarray(pixels)
    _require(arr.dtype.kind == "u" and arr.dtype.itemsize == 2, "Pixels must be uint16")
    _require(arr.shape == (height, width, CHANNELS), "Pixels must have shape (height, width, 3)")
    levels = check_wb_levels(wb_levels_rgbg, input_contract)
    prefix = bytes.fromhex(_check_sha256(options_sha256, "options_sha256"))[:8]
    payload = np.ascontiguousarray(arr, dtype="<u2").tobytes()
    header = struct.pack(HEADER_FORMAT, MAGIC, FORMAT_VERSION, HEADER_BYTES, width, height,
                         CHANNELS, BITS_PER_CHANNEL, len(payload), FLAGS, TRANSFER_LINEAR,
                         PRIMARIES_SRGB, policy, source, *levels, prefix)
    return header + payload


# --- Reader -------------------------------------------------------------------

@dataclass(frozen=True)
class Header:
    width: int
    height: int
    wb_policy: int
    wb_coeff_source: int
    wb_levels_rgbg: tuple[int, int, int, int]
    options_sha256_prefix: bytes


@dataclass(frozen=True)
class DecodedFjlrgV2:
    """A validated v2 input. Pixels are read-only linear uint16 RGB (H, W, 3)."""
    width: int
    height: int
    rgb_uint16: np.ndarray
    provenance: dict
    input_contract: str

    def to_unit_float32(self) -> np.ndarray:
        """Explicit conversion to float32 in 0..1 linear light (sRGB primaries)."""
        return self.rgb_uint16.astype(np.float32) / np.float32(65535.0)


def parse_header(data: bytes) -> Header:
    """Validate the fixed 64-byte header and payload length; return its fields."""
    _require(isinstance(data, (bytes, bytearray, memoryview)), "Container must be bytes")
    data = bytes(data)
    _require(len(data) >= HEADER_BYTES, "Container shorter than 64-byte header")
    (magic, version, header_bytes, width, height, channels, bits, payload_bytes, flags,
     transfer, primaries, policy, source, r, g, b, g2, prefix) = struct.unpack_from(HEADER_FORMAT, data, 0)
    _require(magic == MAGIC, f"Bad magic {magic!r}")
    _require(version == FORMAT_VERSION, f"Unsupported format version {version} (v2 reader)")
    _require(header_bytes == HEADER_BYTES, f"Header size {header_bytes} != 64")
    _require(width > 0 and height > 0, "Zero width or height")
    _require(max(width, height) <= MAX_WORKING_LONG_EDGE, "Working long edge exceeds 1536")
    _require(channels == CHANNELS, f"Channels {channels} != 3")
    _require(bits == BITS_PER_CHANNEL, f"Bits per channel {bits} != 16")
    _require(flags == FLAGS, f"Flags {flags} != 3")
    _require(transfer == TRANSFER_LINEAR, f"Unsupported transfer {transfer}")
    _require(primaries == PRIMARIES_SRGB, f"Unsupported primaries {primaries} (only sRGB=1)")
    _require((policy, source) in {(WB_POLICY_CAMERA, COEFF_SOURCE_CAM_MUL),
                                  (WB_POLICY_USER_MUL, COEFF_SOURCE_FUJI_F00D)},
             f"Invalid WB policy/source pair ({policy}, {source})")
    expected_payload = width * height * CHANNELS * 2
    _require(payload_bytes == expected_payload, "Declared payload size mismatch")
    _require(len(data) == HEADER_BYTES + expected_payload, "Container length mismatch (truncated or trailing bytes)")
    return Header(width, height, policy, source, (r, g, b, g2), prefix)


def _validate_provenance(provenance: Any, header: Header, data: bytes, contract: str,
                         expected_source_sha256: str | None, expected_wasm_sha256: str | None) -> None:
    policy, source, kind = _contract_fields(contract)
    _require(isinstance(provenance, dict), "Provenance must be a JSON object")
    keys = set(provenance)
    _require(keys == SIDECAR_KEYS,
             f"Sidecar keys differ: missing {sorted(SIDECAR_KEYS - keys)}, unknown {sorted(keys - SIDECAR_KEYS)}")
    p = provenance
    _require(p["schema"] == SIDECAR_SCHEMA, "Wrong sidecar schema")
    _require(p["input_contract"] == contract, "Sidecar input_contract differs from expected contract")
    _require((header.wb_policy, header.wb_coeff_source) == (policy, source), "Header WB policy/source differ from contract")
    _require(_check_sha256(p["container_sha256"], "container_sha256") == hashlib.sha256(data).hexdigest(),
             "Container SHA256 mismatch")
    _check_sha256(p["source_raf_sha256"], "source_raf_sha256")
    _check_sha256(p["wasm_sha256"], "wasm_sha256")
    if expected_source_sha256 is not None:
        _require(p["source_raf_sha256"] == _check_sha256(expected_source_sha256, "expected_source_sha256"),
                 "Source RAF SHA256 differs from expected")
    if expected_wasm_sha256 is not None:
        _require(p["wasm_sha256"] == _check_sha256(expected_wasm_sha256, "expected_wasm_sha256"),
                 "Decoder wasm SHA256 differs from expected")
    _require(p["decoder_package"] == DECODER_PACKAGE, "Wrong decoder package")
    _require(p["decoder_version"] == DECODER_VERSION, "Wrong decoder version")
    _require(p["libraw_build_version"] == LIBRAW_BUILD_VERSION, "Wrong LibRaw build version")
    _require(p["parameters_are_declared"] is True, "parameters_are_declared must be literal true")

    # WB source: header levels are the single source of truth; sidecar must match exactly.
    levels = check_wb_levels(list(header.wb_levels_rgbg), contract)
    wb = p["wb_source"]
    _require(isinstance(wb, dict) and set(wb) == WB_SOURCE_KEYS, "wb_source must have exactly kind, levels_rgbg")
    _require(wb["kind"] == kind, "wb_source.kind differs from contract")
    _require(isinstance(wb["levels_rgbg"], list) and len(wb["levels_rgbg"]) == 4, "wb_source.levels_rgbg must be 4 values")
    _require(check_wb_levels(wb["levels_rgbg"], contract) == levels, "wb_source.levels_rgbg differ from header")

    # Options: hash the literal string, parse it strictly, compare to the only allowed object.
    _require(isinstance(p["options_json"], str), "options_json must be a string")
    digest = options_sha256(p["options_json"])
    _require(_check_sha256(p["options_sha256"], "options_sha256") == digest, "options_sha256 does not hash options_json")
    _require(bytes.fromhex(digest)[:8] == header.options_sha256_prefix, "Header options prefix mismatch")
    parsed = strict_json_loads(p["options_json"])
    _require(_json_equal(parsed, p["declared_options"]), "declared_options differ from parsed options_json")
    _require(_json_equal(parsed, canonical_options(contract, levels)), "Options are not the canonical set for this contract")

    # Source metadata (integer-valued JSON numbers; bool, NaN and fractions refused).
    mode = _as_int(p["source_wb_mode"], "source_wb_mode")
    _require(0 <= mode <= 0xFFFF, "source_wb_mode must be 0..65535")
    if contract == CONTRACT_AUTOWB:
        _require(mode == 0, "Auto-WB contract requires source_wb_mode 0 (Auto)")
    fine = p["source_wb_fine_tune_raw"]
    _require(isinstance(fine, list) and len(fine) == 2, "source_wb_fine_tune_raw must be 2 values")
    for v in (_as_int(x, "source_wb_fine_tune_raw") for x in fine):
        _require(v % FINE_TUNE_RAW_STEP == 0 and -FINE_TUNE_RAW_LIMIT <= v <= FINE_TUNE_RAW_LIMIT,
                 "source_wb_fine_tune_raw values must be multiples of 20 in [-180, 180]")
    if p["source_kelvin"] is not None:
        kelvin = _as_int(p["source_kelvin"], "source_kelvin")
        _require(0 < kelvin <= U32_MAX, "source_kelvin must be a positive uint32 or null")
    dr = _as_int(p["capture_dynamic_range"], "capture_dynamic_range")
    _require(dr in CAPTURE_DYNAMIC_RANGES, "capture_dynamic_range must be 100, 200 or 400")
    dr_source = p["capture_dynamic_range_source"]
    _require(isinstance(dr_source, str) and dr_source.strip() != "", "capture_dynamic_range_source must be non-empty")
    native = p["decoded_native_size"]
    _require(isinstance(native, list) and len(native) == 2, "decoded_native_size must be 2 values")
    native = [_as_int(v, "decoded_native_size") for v in native]
    _require(all(v > 0 for v in native), "decoded_native_size must be positive")
    _require(expected_working_size(native[0], native[1]) == (header.width, header.height),
             "Working size is not the canonical downsample of decoded_native_size")


def decode_fjlrg_v2(data: bytes, provenance: Mapping[str, Any], *, expected_contract: str,
                    expected_source_sha256: str | None = None,
                    expected_wasm_sha256: str | None = None) -> DecodedFjlrgV2:
    """Validate a v2 container against its provenance and the expected contract."""
    _contract_fields(expected_contract)
    data = bytes(data)
    header = parse_header(data)
    _validate_provenance(provenance, header, data, expected_contract, expected_source_sha256, expected_wasm_sha256)
    count = header.width * header.height * CHANNELS
    rgb = np.frombuffer(data, dtype="<u2", count=count, offset=HEADER_BYTES).astype(np.uint16)
    rgb = rgb.reshape(header.height, header.width, CHANNELS)
    rgb.setflags(write=False)
    return DecodedFjlrgV2(header.width, header.height, rgb, copy.deepcopy(dict(provenance)), expected_contract)


def sidecar_path(path: str | Path) -> Path:
    """Sidecar location: '<name>.fjlrg' -> '<name>.fjlrg.json'."""
    path = Path(path)
    _require(path.suffix == ".fjlrg", "v2 container path must end in .fjlrg")
    return path.with_name(path.name + SIDECAR_SUFFIX)


def load_fjlrg_v2(path: str | Path, *, expected_contract: str,
                  expected_source_sha256: str | None = None,
                  expected_wasm_sha256: str | None = None) -> DecodedFjlrgV2:
    """Load and fully validate a v2 container and its sidecar from disk."""
    _contract_fields(expected_contract)
    side = sidecar_path(path)
    _require(side.is_file(), f"Missing sidecar {side.name}")
    try:
        text = side.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FjlrgV2Error("Sidecar is not valid UTF-8") from exc
    provenance = strict_json_loads(text)
    return decode_fjlrg_v2(Path(path).read_bytes(), provenance, expected_contract=expected_contract,
                           expected_source_sha256=expected_source_sha256,
                           expected_wasm_sha256=expected_wasm_sha256)


def to_unit_float32(decoded: DecodedFjlrgV2) -> np.ndarray:
    """Module-level form of DecodedFjlrgV2.to_unit_float32."""
    return decoded.to_unit_float32()
