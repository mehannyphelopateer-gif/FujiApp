"""Synthetic tests for the strict v2 research codec. No media, no models.

Expected values are written as literals (offsets, magic, field codes) so the
tests check the agreed format, not just the implementation's own constants.
"""
import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fjlrg
import fjlrg_v2 as v2

H64 = "ab" * 32
WASM = "8947f7e668e488461c3e9defe7007583aa8477b4886aa603b36b407f2f0846ff"
ASSHOT_LEVELS = [628, 302, 445, 0]
AUTO_LEVELS = [518, 302, 570, 302]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pixels(width=4, height=3):
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 65536, size=(height, width, 3), dtype=np.uint16)
    arr[0, 0] = [0x0102, 0, 65535]
    return arr


def make_case(contract="v2-linear-asshot", levels=None, options_json=None, width=4, height=3):
    """Build a valid (container, provenance, pixels) triple for a contract."""
    if levels is None:
        levels = ASSHOT_LEVELS if contract == "v2-linear-asshot" else AUTO_LEVELS
    options = v2.canonical_options(contract, levels)
    if options_json is None:
        options_json = json.dumps(options, separators=(",", ":"))
    opt_sha = sha(options_json.encode("utf-8"))
    px = pixels(width, height)
    data = v2.encode_fjlrg_v2(width, height, px, input_contract=contract,
                              wb_levels_rgbg=levels, options_sha256=opt_sha)
    prov = {
        "schema": "fujiapp.raw-input.v2",
        "input_contract": contract,
        "container_sha256": sha(data),
        "source_raf_sha256": H64,
        "wasm_sha256": WASM,
        "decoder_package": "libraw-wasm",
        "decoder_version": "1.6.0",
        "libraw_build_version": "0.22.1",
        "options_json": options_json,
        "options_sha256": opt_sha,
        "declared_options": json.loads(options_json),
        "wb_source": {"kind": "cam_mul" if contract == "v2-linear-asshot" else "fuji_f00d", "levels_rgbg": list(levels)},
        "source_wb_mode": 4080 if contract == "v2-linear-asshot" else 0,
        "source_wb_fine_tune_raw": [100, -120],
        "source_kelvin": 6600 if contract == "v2-linear-asshot" else None,
        "capture_dynamic_range": 400,
        "capture_dynamic_range_source": "maker-note DevelopmentDynamicRange",
        "decoded_native_size": [width, height],
        "parameters_are_declared": True,
    }
    return data, prov, px


def resign(data, prov):
    """Return provenance whose container hash matches modified bytes, isolating the check under test."""
    out = copy.deepcopy(prov)
    out["container_sha256"] = sha(bytes(data))
    return out


def patch(data, offset, fmt, value):
    buf = bytearray(data)
    struct.pack_into(fmt, buf, offset, value)
    return bytes(buf)


class FormatLayout(unittest.TestCase):
    def test_header_offsets_are_the_agreed_layout(self):
        data, prov, _ = make_case("v2-linear-autowb")
        self.assertEqual(data[0:8], b"FJLRGB16")
        self.assertEqual(struct.unpack_from("<HH", data, 8), (2, 64))
        self.assertEqual(struct.unpack_from("<II", data, 12), (4, 3))
        self.assertEqual(struct.unpack_from("<HH", data, 20), (3, 16))
        self.assertEqual(struct.unpack_from("<II", data, 24), (4 * 3 * 6, 3))
        self.assertEqual(struct.unpack_from("<HHHH", data, 32), (1, 1, 2, 2))
        self.assertEqual(struct.unpack_from("<4I", data, 40), tuple(AUTO_LEVELS))
        self.assertEqual(data[56:64], bytes.fromhex(prov["options_sha256"])[:8])
        self.assertEqual(len(data), 64 + 4 * 3 * 6)

    def test_pixels_are_little_endian_interleaved(self):
        data, _, _ = make_case()
        self.assertEqual(data[64:70], b"\x02\x01\x00\x00\xff\xff")

    def test_levels_are_full_uint32(self):
        levels = [70000, 302, 445, 302]
        data, prov, _ = make_case(levels=levels)
        self.assertEqual(struct.unpack_from("<4I", data, 40), tuple(levels))
        out = v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")
        self.assertEqual(out.provenance["wb_source"]["levels_rgbg"], levels)


class RoundTrip(unittest.TestCase):
    def test_both_contracts_round_trip_exactly(self):
        for contract in ("v2-linear-asshot", "v2-linear-autowb"):
            data, prov, px = make_case(contract)
            out = v2.decode_fjlrg_v2(data, prov, expected_contract=contract,
                                     expected_source_sha256=H64, expected_wasm_sha256=WASM)
            self.assertEqual(out.input_contract, contract)
            np.testing.assert_array_equal(out.rgb_uint16, px)
            self.assertEqual(out.rgb_uint16.dtype, np.uint16)
            self.assertFalse(out.rgb_uint16.flags.writeable)

    def test_float_conversion_is_explicit_and_unit_scaled(self):
        data, prov, px = make_case()
        out = v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")
        f = out.to_unit_float32()
        self.assertEqual(f.dtype, np.float32)
        self.assertEqual(f[0, 0, 2], 1.0)
        self.assertEqual(f[0, 0, 1], 0.0)
        np.testing.assert_allclose(f, px / 65535.0, rtol=0, atol=1e-7)

    def test_file_round_trip_with_sidecar(self):
        data, prov, px = make_case("v2-linear-autowb")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "scene.input-v2-linear-autowb.fjlrg"
            path.write_bytes(data)
            Path(str(path) + ".json").write_text(json.dumps(prov))
            out = v2.load_fjlrg_v2(path, expected_contract="v2-linear-autowb")
            np.testing.assert_array_equal(out.rgb_uint16, px)

    def test_options_hash_uses_literal_bytes_not_reserialisation(self):
        # Same options object, different whitespace: valid when its own hash and prefix are consistent.
        options = v2.canonical_options("v2-linear-asshot", ASSHOT_LEVELS)
        spaced = json.dumps(options, indent=1)
        data, prov, _ = make_case(options_json=spaced)
        v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")
        # ...but a hash of the compact form is refused for the spaced string.
        bad = dict(prov, options_sha256=sha(json.dumps(options, separators=(",", ":")).encode()))
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(data, bad, expected_contract="v2-linear-asshot")

    def test_working_size_matches_canonical_downsample(self):
        self.assertEqual(v2.expected_working_size(7728, 5152), (1536, 1024))
        self.assertEqual(v2.expected_working_size(5152, 7728), (1024, 1536))
        self.assertEqual(v2.expected_working_size(4, 3), (4, 3))


class ContractRefusal(unittest.TestCase):
    def test_expected_contract_is_required(self):
        data, prov, _ = make_case()
        with self.assertRaises(TypeError):
            v2.decode_fjlrg_v2(data, prov)  # pylint: disable=missing-kwoa

    def test_wrong_or_unknown_contract_refused(self):
        data, prov, _ = make_case("v2-linear-asshot")
        for wrong in ("v2-linear-autowb", "v1-legacy-rec709-camwb", ""):
            with self.assertRaises(v2.FjlrgV2Error):
                v2.decode_fjlrg_v2(data, prov, expected_contract=wrong)

    def test_header_policy_must_match_contract(self):
        data, prov, _ = make_case("v2-linear-asshot")
        swapped = patch(patch(data, 36, "<H", 2), 38, "<H", 2)
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(swapped, resign(swapped, prov), expected_contract="v2-linear-asshot")

    def test_external_hash_expectations(self):
        data, prov, _ = make_case()
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot", expected_source_sha256="cd" * 32)
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot", expected_wasm_sha256="cd" * 32)


class HeaderCorruption(unittest.TestCase):
    CASES = [
        ("version 1", 8, "<H", 1), ("version 3", 8, "<H", 3), ("header 32", 10, "<H", 32),
        ("width 0", 12, "<I", 0), ("channels 4", 20, "<H", 4), ("bits 8", 22, "<H", 8),
        ("payload count", 24, "<I", 71), ("flags 1", 28, "<I", 1), ("transfer 0", 32, "<H", 0),
        ("transfer legacy", 32, "<H", 2), ("primaries camera", 34, "<H", 0), ("policy 3", 36, "<H", 3),
        ("source mismatch", 38, "<H", 2), ("levels differ", 40, "<I", 629), ("options prefix", 56, "<Q", 0),
    ]

    def test_each_corrupted_field_is_refused(self):
        data, prov, _ = make_case()
        for label, offset, fmt, value in self.CASES:
            with self.subTest(label):
                bad = patch(data, offset, fmt, value)
                with self.assertRaises(v2.FjlrgV2Error):
                    v2.decode_fjlrg_v2(bad, resign(bad, prov), expected_contract="v2-linear-asshot")

    def test_magic_truncation_and_trailing_bytes(self):
        data, prov, _ = make_case()
        for bad in (b"FJLRGB17" + data[8:], data[:-1], data + b"\x00\x00", data[:40]):
            with self.assertRaises(v2.FjlrgV2Error):
                v2.decode_fjlrg_v2(bad, resign(bad, prov), expected_contract="v2-linear-asshot")

    def test_oversized_working_edge_refused(self):
        data, prov, _ = make_case()
        bad = patch(data, 12, "<I", 1537)
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(bad, resign(bad, prov), expected_contract="v2-linear-asshot")

    def test_pixel_corruption_caught_by_container_hash(self):
        data, prov, _ = make_case()
        bad = bytearray(data)
        bad[100] ^= 1
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(bytes(bad), prov, expected_contract="v2-linear-asshot")


class OptionsValidation(unittest.TestCase):
    def mutate(self, contract, change):
        levels = ASSHOT_LEVELS if contract == "v2-linear-asshot" else AUTO_LEVELS
        options = v2.canonical_options(contract, levels)
        change(options)
        return make_case(contract, options_json=json.dumps(options))

    def assert_refused(self, contract, change):
        data, prov, _ = self.mutate(contract, change)
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(data, prov, expected_contract=contract)

    def test_bool_never_equals_a_number(self):
        self.assert_refused("v2-linear-asshot", lambda o: o.update(noAutoBright=1))
        self.assert_refused("v2-linear-asshot", lambda o: o.update(useCameraMatrix=True))
        self.assert_refused("v2-linear-asshot", lambda o: o.update(highlight=False))

    def test_json_numbers_compare_by_value(self):
        # JSON has one number type: 16 and 16.0 are the same value, as in the JS codec.
        for change in (lambda o: o.update(outputBps=16.0), lambda o: o.update(gamm=[1.0, 1.0, 0.0, 0, 0, 0])):
            data, prov, _ = self.mutate("v2-linear-asshot", change)
            v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")
        data, prov, _ = self.mutate("v2-linear-autowb", lambda o: o.update(userMul=[518.0, 302.0, 570.0, 302.0]))
        v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-autowb")
        self.assert_refused("v2-linear-asshot", lambda o: o.update(outputBps=16.5))

    def test_non_finite_declared_options_refused(self):
        data, prov, _ = make_case()
        prov["declared_options"]["highlight"] = float("inf")
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")

    def test_two_element_gamma_refused(self):
        self.assert_refused("v2-linear-asshot", lambda o: o.update(gamm=[1, 1]))

    def test_extra_and_missing_keys_refused(self):
        self.assert_refused("v2-linear-asshot", lambda o: o.update(extra=0))
        self.assert_refused("v2-linear-asshot", lambda o: o.pop("highlight"))

    def test_wb_option_rules_per_contract(self):
        self.assert_refused("v2-linear-asshot", lambda o: o.update(userMul=ASSHOT_LEVELS))
        self.assert_refused("v2-linear-asshot", lambda o: o.update(useCameraWb=False))
        self.assert_refused("v2-linear-autowb", lambda o: o.update(useCameraWb=True))
        self.assert_refused("v2-linear-autowb", lambda o: o.update(userMul=[518, 302, 571, 302]))
        self.assert_refused("v2-linear-autowb", lambda o: o.pop("userMul"))

    def test_duplicate_keys_and_nan_refused(self):
        options = v2.canonical_options("v2-linear-asshot", ASSHOT_LEVELS)
        text = json.dumps(options)
        for bad_json in (text[:-1] + ', "highlight": 0}', text.replace('"highlight": 0', '"highlight": NaN')):
            data, prov, _ = make_case(options_json=bad_json)
            with self.assertRaises(v2.FjlrgV2Error):
                v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")

    def test_declared_options_must_equal_parsed_options(self):
        data, prov, _ = make_case()
        prov["declared_options"]["highlight"] = 1
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")

    def test_options_sha_must_hash_options_json(self):
        data, prov, _ = make_case()
        prov["options_sha256"] = "cd" * 32
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")


class SidecarValidation(unittest.TestCase):
    def refused(self, contract, change, **case):
        data, prov, _ = make_case(contract, **case)
        change(prov)
        with self.assertRaises(v2.FjlrgV2Error):
            v2.decode_fjlrg_v2(data, prov, expected_contract=contract)

    def test_root_keys_exact(self):
        self.refused("v2-linear-asshot", lambda p: p.update(camera_model="X100VI"))
        self.refused("v2-linear-asshot", lambda p: p.pop("source_kelvin"))

    def test_fixed_identity_fields(self):
        for key, value in (("schema", "fujiapp.raw-input.v1"), ("decoder_package", "libraw"),
                           ("decoder_version", "1.6.1"), ("libraw_build_version", "0.21.0"),
                           ("input_contract", "v2-linear-autowb"), ("parameters_are_declared", 1)):
            with self.subTest(key):
                self.refused("v2-linear-asshot", lambda p, k=key, v=value: p.update({k: v}))

    def test_hashes_must_be_lowercase_hex64(self):
        self.refused("v2-linear-asshot", lambda p: p.update(source_raf_sha256=H64.upper()))
        self.refused("v2-linear-asshot", lambda p: p.update(wasm_sha256="ab" * 31))

    def test_wb_source_rules(self):
        self.refused("v2-linear-asshot", lambda p: p.update(wb_source={"kind": "fuji_f00d", "levels_rgbg": ASSHOT_LEVELS}))
        self.refused("v2-linear-asshot", lambda p: p.update(wb_source={"kind": "cam_mul", "levels_rgbg": [628, 302, 446, 0]}))
        self.refused("v2-linear-asshot", lambda p: p["wb_source"].update(extra=1))
        self.refused("v2-linear-autowb", lambda p: p.update(source_wb_mode=512))

    def test_levels_validated_at_encode(self):
        for contract, levels in (("v2-linear-asshot", [628, 302, 445, 999]), ("v2-linear-autowb", [518, 302, 570, 0]),
                                 ("v2-linear-asshot", [0, 302, 445, 0]), ("v2-linear-asshot", [628.5, 302, 445, 0]),
                                 ("v2-linear-asshot", [True, 302, 445, 0]), ("v2-linear-asshot", [float("nan"), 302, 445, 0]),
                                 ("v2-linear-asshot", [2 ** 32, 302, 445, 0])):
            with self.subTest(levels=levels), self.assertRaises(v2.FjlrgV2Error):
                v2.encode_fjlrg_v2(4, 3, pixels(), input_contract=contract, wb_levels_rgbg=levels, options_sha256=H64)

    def test_integer_valued_float_levels_are_coerced(self):
        data = v2.encode_fjlrg_v2(4.0, 3, pixels(), input_contract="v2-linear-asshot",
                                  wb_levels_rgbg=[628.0, 302.0, 445.0, 0.0], options_sha256=H64)
        self.assertEqual(struct.unpack_from("<4I", data, 40), (628, 302, 445, 0))
        data, prov, _ = make_case()
        prov["wb_source"]["levels_rgbg"] = [628.0, 302.0, 445.0, 0.0]
        v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")

    def test_integer_valued_float_metadata_accepted(self):
        data, prov, _ = make_case()
        prov.update(source_wb_fine_tune_raw=[100.0, -120.0], source_kelvin=6600.0, capture_dynamic_range=400.0,
                    source_wb_mode=4080.0, decoded_native_size=[4.0, 3.0])
        v2.decode_fjlrg_v2(data, prov, expected_contract="v2-linear-asshot")

    def test_fine_tune_kelvin_and_dynamic_range(self):
        for fine in ([10, -120], [200, 0], [100], [110.0, 0], [True, 0], [100.5, 0]):
            with self.subTest(fine=fine):
                self.refused("v2-linear-asshot", lambda p, f=fine: p.update(source_wb_fine_tune_raw=f))
        for kelvin in (0, -6600, 6600.5, True, 2 ** 32, "6600"):
            with self.subTest(kelvin=kelvin):
                self.refused("v2-linear-asshot", lambda p, k=kelvin: p.update(source_kelvin=k))
        self.refused("v2-linear-asshot", lambda p: p.update(capture_dynamic_range=800))
        self.refused("v2-linear-asshot", lambda p: p.update(capture_dynamic_range=True))
        self.refused("v2-linear-asshot", lambda p: p.update(source_wb_mode=False))
        self.refused("v2-linear-asshot", lambda p: p.update(capture_dynamic_range_source="  "))

    def test_native_size_must_downsample_to_working_size(self):
        self.refused("v2-linear-asshot", lambda p: p.update(decoded_native_size=[8, 6]))
        self.refused("v2-linear-asshot", lambda p: p.update(decoded_native_size=[4, 0]))


class LegacyCrossRefusal(unittest.TestCase):
    def test_legacy_loader_refuses_v2(self):
        data, _, _ = make_case()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v2.fjlrg"
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "unsupported format version 2"):
                fjlrg.load_fjlrg_linear(path)

    def test_v2_loader_refuses_v1(self):
        _, prov, px = make_case()
        payload = px.astype("<u2").tobytes()
        legacy = struct.pack("<8sHHIIHHII", b"FJLRGB16", 1, 32, 4, 3, 3, 16, len(payload), 1) + payload
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v1.fjlrg"
            path.write_bytes(legacy)
            fjlrg.load_fjlrg_linear(path)  # the legacy file itself is valid v1
            Path(str(path) + ".json").write_text(json.dumps(resign(legacy, prov)))
            with self.assertRaisesRegex(v2.FjlrgV2Error, "version 1"):
                v2.load_fjlrg_v2(path, expected_contract="v2-linear-asshot")

    def test_load_requires_suffix_and_strict_sidecar(self):
        data, prov, _ = make_case()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.fjlrg"
            path.write_bytes(data)
            with self.assertRaises(v2.FjlrgV2Error):
                v2.load_fjlrg_v2(path, expected_contract="v2-linear-asshot")  # no sidecar
            text = json.dumps(prov)
            Path(str(path) + ".json").write_text(text[:-1] + ', "schema": "fujiapp.raw-input.v2"}')
            with self.assertRaises(v2.FjlrgV2Error):
                v2.load_fjlrg_v2(path, expected_contract="v2-linear-asshot")  # duplicate key
            with self.assertRaises(v2.FjlrgV2Error):
                v2.load_fjlrg_v2(Path(d) / "x.bin", expected_contract="v2-linear-asshot")


if __name__ == "__main__":
    unittest.main()
