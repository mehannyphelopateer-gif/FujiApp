"""Synthetic tests for the C8 E2 WB metadata probe. No real TIFF/JPEG/RAF, no inventory reads."""
import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import probe_cuban_auto_wb_metadata as M

SETTINGS = dict(M.E2_SPEC)


def makernote(entries):
    """entries: [(tag, type, count, values-or-raw-bytes)] -> Fuji MakerNote bytes (LE, IFD at 12)."""
    ifd = 12
    data_off = ifd + 2 + 12 * len(entries) + 4
    table, data = b"", b""
    for tag, typ, cnt, vals in entries:
        f, size = M.FMT.get(typ, ("I", 4))
        payload = vals if isinstance(vals, bytes) else struct.pack("<" + f * cnt, *vals)
        if len(payload) <= 4:
            table += struct.pack("<HHI", tag, typ, cnt) + payload.ljust(4, b"\0")
        else:
            table += struct.pack("<HHII", tag, typ, cnt, data_off + len(data))
            data += payload
    return b"FUJIFILM" + struct.pack("<I", ifd) + struct.pack("<H", len(entries)) + table + b"\0\0\0\0" + data


def settings_entries(over=None, wb=(1932, 1208, 1006), bad_wb_type=False):
    s = dict(SETTINGS, **(over or {}))
    inv = {v: k for k, v in M.SETTING_TAGS.items()}
    entries = []
    for name, v in s.items():
        vals = v if isinstance(v, list) else [v]
        typ = 8 if any(x < 0 for x in vals) else 3
        entries.append((inv[name], typ, len(vals), vals))
    entries.append((0x9999, 2, 40, b"x" * 40))   # unknown tag: never unpacked
    for tag, val in zip((0x144a, 0x144b, 0x144c), wb):
        entries.append((tag, 7, 200, b"\xff" * 200) if bad_wb_type else (tag, 3, 1, [val]))
    return entries


class Predictions(unittest.TestCase):
    def test_variants_exact_integers_and_round_boundaries(self):
        x, n = 518, 1159
        self.assertEqual(M.predict(x, n, "H0"), 4 * 518)
        self.assertEqual(M.predict(x, n, "H1"), 4 * ((518 * 1159 + 512) // 1024))
        self.assertEqual(M.predict(x, n, "H2"), (4 * 518 * 1159 + 512) // 1024)
        self.assertEqual(M.predict(x, n, "H3"), (4 * 518 * 1159) // 1024)
        self.assertEqual(M.predict(x, n, "H4"), (4 * 518 * 1159 + 1023) // 1024)
        self.assertEqual(M.predict(1, 128, "H2"), 1)    # 4*128 = 512 -> exactly half rounds up
        self.assertEqual(M.predict(1, 256, "H3"), 1)    # exact multiple: floor == ceil
        self.assertEqual(M.predict(1, 256, "H4"), 1)
        self.assertEqual(M.predict(1, 257, "H4"), 2)
        for bad in (0, -1, 3.0, True):
            with self.assertRaises(ValueError):
                M.predict(bad, 1159, "H1")

    def test_rgb_requires_source_g(self):
        rgb = M.predict_rgb([518, 302, 570, 302], 1159, 852)
        self.assertTrue(all(v[1] == 1208 for v in rgb.values()))
        with self.assertRaises(ValueError):
            M.predict_rgb([518, 300, 570, 300], 1159, 852)


class Makernote(unittest.TestCase):
    def test_settings_parsed_without_wb_values(self):
        s, present = M.parse_makernote(makernote(settings_entries()))
        self.assertEqual({k: s[k] for k in M.E2_SPEC}, M.E2_SPEC)
        self.assertEqual(sorted(present), ["0x144a", "0x144b", "0x144c"])
        self.assertFalse({"R", "G", "B"} & set(s))

    def test_unsupported_wb_type_is_skipped_in_qualify(self):
        s, present = M.parse_makernote(makernote(settings_entries(bad_wb_type=True)))
        self.assertEqual(len(present), 3)
        with self.assertRaises(ValueError):
            M.parse_makernote(makernote(settings_entries(bad_wb_type=True)), want_wb_levels=True)

    def test_wb_levels_read_only_when_requested(self):
        levels, _ = M.parse_makernote(makernote(settings_entries()), want_wb_levels=True)
        self.assertEqual((levels["R"], levels["G"], levels["B"]), (1932, 1208, 1006))

    def test_bounds_and_header_refusals(self):
        with self.assertRaises(ValueError):
            M.parse_makernote(b"NOTFUJI!" + b"\0" * 20)
        mn = bytearray(makernote(settings_entries()))
        struct.pack_into("<H", mn, 12, 5000)
        with self.assertRaises(ValueError):
            M.parse_makernote(bytes(mn))

    def test_wrong_settings_refused(self):
        meta = {"DateTimeOriginal": "2026:07:09 20:45:32", "ColorSpace": 1, "BitsPerSample": [16, 16, 16]}
        ok, _ = M.parse_makernote(makernote(settings_entries()))
        self.assertEqual(M.check_settings(ok, meta, "2026:07:09 20:45:32"), [])
        for over in ({"WhiteBalance": 0xFF0}, {"WBFineTune": [0, 0]}, {"FilmMode": 0}, {"GrainEffectRoughness": 64},
                     {"DevelopmentDynamicRange": 100}):
            s, _ = M.parse_makernote(makernote(settings_entries(over)))
            self.assertTrue(M.check_settings(s, meta, "2026:07:09 20:45:32"), over)
        self.assertTrue(M.check_settings(ok, meta, "2026:07:09 20:45:33"))
        self.assertTrue(M.check_settings(ok, dict(meta, ColorSpace=65535), "2026:07:09 20:45:32"))
        self.assertTrue(M.check_settings(ok, dict(meta, BitsPerSample=[8, 8, 8]), "2026:07:09 20:45:32"))

    def test_metadata_reader_on_synthetic_jpeg(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.jpg"
            exif = Image.Exif()
            sub = exif.get_ifd(0x8769)
            sub[0x9003], sub[0xA001], sub[0x927C] = "2026:07:09 20:45:32", 1, makernote(settings_entries())
            Image.new("RGB", (4, 4)).save(path, exif=exif)
            meta = M.read_exif_metadata(path)
            if meta["_makernote"] is None:
                self.skipTest("This Pillow version does not round-trip Exif sub-IFDs")
            s, present = M.parse_makernote(meta["_makernote"])
            self.assertEqual(M.check_settings(s, meta, "2026:07:09 20:45:32"), [])


def row(name, mode, preds, observed):
    return {"name": name, "capture_mode": mode, "predictions": preds, "observed": observed}


class Compare(unittest.TestCase):
    def preds(self, h0, h1):
        return {"H0": h0, "H1": h1, "H2": h1, "H3": [1, 1208, 1], "H4": [1, 1208, 1]}

    def test_manual_interpreted_only_under_auto_exact_variant(self):
        rows = [row(f"a{i}", 0, self.preds([2000, 1208, 2200], [2264, 1208, 1830]), [2264, 1208, 1830]) for i in range(3)]
        rows += [row(f"m{i}", 4080, self.preds([1500, 1208, 1800], [1700, 1208, 1500]), [1700, 1208, 1500]) for i in range(5)]
        out = M.compare_rows(rows)["per_variant"]
        self.assertTrue(out["H1"]["manual_interpretable"])
        self.assertEqual(out["H1"]["manual_captures"]["rgb_exact"], 5)
        self.assertFalse(out["H0"]["manual_interpretable"])
        self.assertIsNone(out["H0"]["interpretation_if_interpretable"])
        self.assertFalse(out["H3"]["manual_interpretable"])

    def test_h0_exact_means_base_estimate_only(self):
        rows = [row(f"a{i}", 0, self.preds([2000, 1208, 2200], [2264, 1208, 1830]), [2000, 1208, 2200]) for i in range(3)]
        out = M.compare_rows(rows)["per_variant"]["H0"]
        self.assertTrue(out["manual_interpretable"])
        self.assertIn("Q10 transfer unvalidated", out["interpretation_if_interpretable"])

    def test_wrong_scale_is_inconclusive(self):
        rows = [row("a", 0, self.preds([2000, 1208, 2200], [2264, 1208, 1830]), [2264, 302, 1830])]
        self.assertEqual(M.compare_rows(rows)["verdict"], "INCONCLUSIVE_SCALE")


class Receipts(unittest.TestCase):
    cohort1 = {"installed": [{"name": "A.RAF", "variant": "E2-no-grain", "installed": "Shoot 1/" + M.TARGET_NAME, "verified_sha256": "aa" * 32},
                             {"name": "A.RAF", "variant": "E1-full-recipe", "installed": "x", "verified_sha256": "bb" * 32}]}
    pilot = {"passed": True, "entries": [{"name": "P.RAF", "variant": "E2-no-grain", "source_sha256": "cc" * 32, "problems": []}]}

    def test_receipts_collected_from_verified_records_only(self):
        r = M.collect_receipts(self.cohort1, self.pilot, pilot_committed=True)
        self.assertEqual(r["A.RAF"]["sha256"], "aa" * 32)
        self.assertEqual(r["P.RAF"]["receipt"], "pilot-export-check")

    def test_uncommitted_or_failed_pilot_receipt_refused(self):
        with self.assertRaises(ValueError):
            M.collect_receipts(self.cohort1, self.pilot, pilot_committed=False)
        with self.assertRaises(ValueError):
            M.collect_receipts(self.cohort1, dict(self.pilot, passed=False), pilot_committed=True)
        bad = {"passed": True, "entries": [dict(self.pilot["entries"][0], problems=["x"])]}
        with self.assertRaises(ValueError):
            M.collect_receipts(self.cohort1, bad, pilot_committed=True)


class StageGuards(unittest.TestCase):
    def test_compare_refuses_without_matching_passed_qualification(self):
        with tempfile.TemporaryDirectory() as d:
            saved = (M.MANIFEST, M.QUALIFICATION, M.PREDICTIONS, M.load_manifest)
            try:
                M.MANIFEST, M.QUALIFICATION = Path(d) / "m.json", Path(d) / "q.json"
                M.MANIFEST.write_text("{}")
                M.load_manifest = lambda expected: ({"targets": [], "bindings": {}}, {"scenes": []})
                M.QUALIFICATION.write_text('{"passed": false, "manifest_sha256": "x", "rows": []}')
                with self.assertRaises(ValueError):
                    M.compare("x")
                M.QUALIFICATION.write_text('{"passed": true, "manifest_sha256": "other", "rows": [' + ",".join(["{}"] * 8) + "]}")
                with self.assertRaises(ValueError):
                    M.compare("x")
            finally:
                M.MANIFEST, M.QUALIFICATION, M.PREDICTIONS, M.load_manifest = saved

    def test_load_manifest_refuses_drift(self):
        with tempfile.TemporaryDirectory() as d:
            saved = M.MANIFEST
            try:
                M.MANIFEST = Path(d) / "m.json"
                M.MANIFEST.write_text("{}")
                with self.assertRaises(ValueError):
                    M.load_manifest("0" * 64)
            finally:
                M.MANIFEST = saved


if __name__ == "__main__":
    unittest.main()
