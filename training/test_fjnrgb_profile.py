"""Synthetic tests for the research profile codec. No RAF, TIFF, target, model or real container is read."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fjlrg_v2 as V2
import fjnrgb_profile as P

SRC = "a" * 64
WASM = "b" * 64


def fixture(width=40, height=30, native=(40, 30)):
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 65536, size=(height, width, 3), dtype=np.uint16)
    options_json = P.serialize_options(P.canonical_options())
    container = P.encode(width, height, pixels, options_sha256=P.options_sha256(options_json))
    prov = P.build_provenance(container, source_raf_sha256=SRC, wasm_sha256=WASM, capture_dynamic_range=400,
                              capture_dynamic_range_source="frozen fit manifest captureDynamicRange",
                              decoded_native_size=list(native))
    return pixels, container, prov


def decode(container, prov, src=SRC, wasm=WASM):
    return P.decode(container, prov, expected_source_sha256=src, expected_wasm_sha256=wasm)


def rehash(container, prov):
    prov = copy.deepcopy(prov)
    prov["container_sha256"] = hashlib.sha256(bytes(container)).hexdigest()
    return prov


class ContractConstants(unittest.TestCase):
    def test_native_geometry_uint32_matches_javascript_boundary(self):
        for width, height in ((2**32, 3), (3, 2**32), (True, 3), (0, 3)):
            with self.assertRaises(P.FjnrgbProfileError):
                P.expected_working_size(width, height)

    def test_common_options_and_sidecar_keys_match_v2_without_touching_it(self):
        before = (copy.deepcopy(V2._COMMON_OPTIONS), V2.SIDECAR_KEYS, dict(V2.CONTRACT_FIELDS))
        self.assertEqual(P._COMMON_OPTIONS, V2._COMMON_OPTIONS)
        self.assertEqual(P.V2_SIDECAR_KEYS, V2.SIDECAR_KEYS)
        self.assertEqual(P.SIDECAR_KEYS - V2.SIDECAR_KEYS, P.PROFILE_SIDECAR_KEYS)
        self.assertNotIn(P.CONTRACT, V2.CONTRACT_FIELDS)
        self.assertNotEqual(P.MAGIC, V2.MAGIC)
        self.assertEqual((copy.deepcopy(V2._COMMON_OPTIONS), V2.SIDECAR_KEYS, dict(V2.CONTRACT_FIELDS)), before)

    def test_canonical_options_profile_userMul_and_fresh_copies(self):
        o = P.canonical_options()
        self.assertIs(o["useCameraWb"], False)
        self.assertEqual(o["userMul"], [620, 302, 455, 302])
        self.assertEqual(o["gamm"], [1, 1, 0, 0, 0, 0])
        self.assertEqual({k: v for k, v in o.items() if k not in ("useCameraWb", "userMul")},
                         {k: v for k, v in V2._COMMON_OPTIONS if k != "useCameraWb"})
        o["userMul"][0] = 1
        o["gamm"][0] = 2
        self.assertEqual(P.canonical_options()["userMul"][0], 620)
        self.assertEqual(P.canonical_options()["gamm"][0], 1)
        self.assertEqual(json.loads(P.serialize_options(P.canonical_options())), P.canonical_options())
        self.assertEqual(list(json.loads(P.serialize_options(P.canonical_options()))), sorted(P.canonical_options()))


class RoundTrip(unittest.TestCase):
    def test_identity_and_read_only(self):
        pixels, container, prov = fixture()
        d = decode(container, prov)
        self.assertTrue(np.array_equal(d.rgb_uint16, pixels))
        self.assertFalse(d.rgb_uint16.flags.writeable)
        self.assertEqual((d.width, d.height, d.input_contract), (40, 30, P.CONTRACT))
        self.assertEqual(d.provenance, prov)

    def test_downsampled_working_size(self):
        _, container, prov = fixture(width=1536, height=1024, native=(6240, 4160))
        self.assertEqual(decode(container, prov).width, 1536)

    def test_load_from_disk(self):
        pixels, container, prov = fixture()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "synthetic.fjnrgb"
            path.write_bytes(container)
            P.sidecar_path(path).write_text(json.dumps(prov))
            self.assertTrue(np.array_equal(P.load(path, expected_source_sha256=SRC, expected_wasm_sha256=WASM).rgb_uint16, pixels))
            with self.assertRaises(P.FjnrgbProfileError):
                P.sidecar_path(Path(d) / "synthetic.fjlrg")


class CrossRefusal(unittest.TestCase):
    def test_v2_rejects_profile_container(self):
        _, container, prov = fixture()
        for contract in (V2.CONTRACT_ASSHOT, V2.CONTRACT_AUTOWB):
            with self.assertRaises(V2.FjlrgV2Error):
                V2.decode_fjlrg_v2(container, prov, expected_contract=contract)
        with self.assertRaises(V2.FjlrgV2Error):
            V2.parse_header(container)

    def test_profile_rejects_v2_containers_and_sidecars(self):
        levels = [620, 302, 455, 302]
        v2_opts = V2.canonical_options(V2.CONTRACT_AUTOWB, levels)
        opts_json = json.dumps({k: v2_opts[k] for k in sorted(v2_opts)}, separators=(",", ":"))
        pixels = np.zeros((30, 40, 3), np.uint16)
        v2 = V2.encode_fjlrg_v2(40, 30, pixels, input_contract=V2.CONTRACT_AUTOWB, wb_levels_rgbg=levels,
                                options_sha256=V2.options_sha256(opts_json))
        _, _, prov = fixture()
        with self.assertRaises(P.FjnrgbProfileError):
            decode(v2, rehash(v2, prov))
        _, container, prov = fixture()
        v2_style = {k: prov[k] for k in V2.SIDECAR_KEYS}
        v2_style.update(schema=V2.SIDECAR_SCHEMA, input_contract=V2.CONTRACT_AUTOWB,
                        wb_source={"kind": "fuji_f00d", "levels_rgbg": levels})
        with self.assertRaises(P.FjnrgbProfileError):
            decode(container, v2_style)
        relabelled = dict(prov, input_contract=V2.CONTRACT_AUTOWB)
        with self.assertRaises(P.FjnrgbProfileError):
            decode(container, relabelled)


class ProvenanceTamper(unittest.TestCase):
    def setUp(self):
        _, self.container, self.prov = fixture()

    def refused(self, mutate):
        prov = copy.deepcopy(self.prov)
        mutate(prov)
        with self.assertRaises(P.FjnrgbProfileError):
            decode(self.container, prov)

    def test_every_field_tamper_refused(self):
        bad_opts = lambda **chg: (lambda p: _set_options(p, chg))
        cases = [
            lambda p: p.update(schema="fujiapp.raw-input.v2"),
            lambda p: p.update(input_contract="v2-linear-asshot"),
            lambda p: p.update(container_sha256="0" * 64),
            lambda p: p.update(source_raf_sha256="c" * 64),
            lambda p: p.update(wasm_sha256="d" * 64),
            lambda p: p.update(decoder_version="1.6.1"),
            lambda p: p.update(parameters_are_declared=1),
            lambda p: p.update(parameters_are_declared=False),
            lambda p: p.update(wb_source={"kind": "fuji_f00d", "levels_rgbg": [620, 302, 455, 302]}),
            lambda p: p.update(wb_source={"kind": "fit_profile", "levels_rgbg": [636, 302, 414, 302]}),
            lambda p: p.update(wb_source={"kind": "fit_profile", "levels_rgbg": [620, 302, 455, 302], "extra": 1}),
            lambda p: p.update(options_sha256="e" * 64),
            lambda p: p.update(declared_options={**p["declared_options"], "highlight": 1}),
            bad_opts(userMul=[636, 302, 414, 302]),
            bad_opts(useCameraWb=True),
            bad_opts(outputColor=0),
            bad_opts(gamm=[0.45, 4.5, 0, 0, 0, 0]),
            bad_opts(noInterpolation=True),
            lambda p: p.update(source_wb_mode=0),
            lambda p: p.update(source_wb_mode=True),
            lambda p: p.update(source_kelvin=5600),
            lambda p: p.update(source_kelvin=None),
            lambda p: p.update(source_wb_fine_tune_raw=[0, 0]),
            lambda p: p.update(source_wb_fine_tune_raw=[20, -60, 0]),
            lambda p: p.update(capture_dynamic_range=300),
            lambda p: p.update(capture_dynamic_range=True),
            lambda p: p.update(capture_dynamic_range_source=" "),
            lambda p: p.update(decoded_native_size=[41, 30]),
            lambda p: p.update(decoded_native_size=[40]),
            lambda p: p.update(profile_manifest_sha256="f" * 64),
            lambda p: p.update(profile_key="4080/5600"),
            lambda p: p.update(policy_basis="source-auto-estimate"),
            lambda p: p.update(primitive_reports={**P.PRIMITIVE_REPORTS, "fullnative_v2_report_sha256": "0" * 64}),
            lambda p: p.update(primitive_reports={k: v for k, v in P.PRIMITIVE_REPORTS.items() if k != "sparse_primitive_report_sha256"}),
            lambda p: p.pop("profile_key"),
            lambda p: p.update(unexpected=1),
        ]
        for mutate in cases:
            with self.subTest(mutate=mutate):
                self.refused(mutate)

    def test_ambiguous_options_json_refused(self):
        for text in ('{"a":1,"a":1}', '{"x":NaN}', "not json"):
            def mutate(p, text=text):
                p["options_json"] = text
                p["options_sha256"] = P.options_sha256(text)
            self.refused(mutate)

    def test_expected_hashes_are_required(self):
        with self.assertRaises(TypeError):
            P.decode(self.container, self.prov)
        for src, wasm in ((None, WASM), (SRC, None), ("A" * 64, WASM)):
            with self.assertRaises(P.FjnrgbProfileError):
                decode(self.container, self.prov, src=src, wasm=wasm)


def _set_options(p, changes):
    options = dict(p["declared_options"], **changes)
    text = P.serialize_options(options)
    p.update(options_json=text, options_sha256=P.options_sha256(text), declared_options=options)


class HeaderTamper(unittest.TestCase):
    def setUp(self):
        _, self.container, self.prov = fixture()

    def refused(self, offset, raw):
        data = bytearray(self.container)
        data[offset:offset + len(raw)] = raw
        with self.assertRaises(P.FjnrgbProfileError):
            decode(bytes(data), rehash(data, self.prov))  # rehash so the targeted check, not the digest, refuses

    def test_header_fields(self):
        import struct
        cases = [(0, b"FJLRGB16"), (8, struct.pack("<H", 2)), (10, struct.pack("<H", 65)),
                 (12, struct.pack("<I", 1537)), (20, struct.pack("<H", 4)), (22, struct.pack("<H", 8)),
                 (24, struct.pack("<I", 1)), (28, struct.pack("<I", 0)), (32, struct.pack("<H", 2)),
                 (34, struct.pack("<H", 0)), (36, struct.pack("<H", 1)), (38, struct.pack("<H", 2)),
                 (40, struct.pack("<I", 636)), (44, struct.pack("<I", 325)), (52, struct.pack("<I", 0)),
                 (56, b"\x00" * 8)]
        for offset, raw in cases:
            with self.subTest(offset=offset):
                self.refused(offset, raw)

    def test_truncated_and_trailing(self):
        for data in (self.container[:-2], self.container + b"\0\0", self.container[:63]):
            with self.assertRaises(P.FjnrgbProfileError):
                decode(data, rehash(data, self.prov))

    def test_pixel_change_caught_by_digest(self):
        data = bytearray(self.container)
        data[100] ^= 1
        with self.assertRaises(P.FjnrgbProfileError):
            decode(bytes(data), self.prov)


class EncoderRefusals(unittest.TestCase):
    def test_bad_inputs(self):
        sha = P.options_sha256(P.serialize_options(P.canonical_options()))
        for args in ((1537, 1, np.zeros((1, 1537, 3), np.uint16)), (2, 2, np.zeros((2, 2, 3), np.uint8)),
                     (2, 2, np.zeros((2, 3, 3), np.uint16)), (0, 2, np.zeros((2, 0, 3), np.uint16)),
                     (True, 1, np.zeros((1, 1, 3), np.uint16))):
            with self.assertRaises(P.FjnrgbProfileError):
                P.encode(*args, options_sha256=sha)
        with self.assertRaises(P.FjnrgbProfileError):
            P.encode(2, 2, np.zeros((2, 2, 3), np.uint16), options_sha256="xyz")


if __name__ == "__main__":
    unittest.main()
