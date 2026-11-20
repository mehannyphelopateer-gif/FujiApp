"""Synthetic files only: bounded RAF/TIFF metadata and exact channel ordering."""
import struct
import tempfile
import unittest
from pathlib import Path

from probe_provia_source_wb import MetadataReader, crosscheck, numeric_values, rgb, unique_pairs


def fixture(endian="<", cycle=False, bad_value=False, bad_count=False):
    data = bytearray(1024)
    data[:16] = b"FUJIFILMCCD-RAW "
    struct.pack_into(">IIII", data, 0x5c, 128, 40, 256, 512)
    struct.pack_into(">IHH4HHH", data, 128, 2, 0x2000, 8, 300, 600, 300, 450, 0xc000, 4)
    # C000 has no valid layout: must be skipped rather than read/interpreted.
    data[148:152] = b"xxxx"
    data[256:260] = b"II*\0" if endian == "<" else b"MM\0*"
    struct.pack_into(endian + "I", data, 260, 8)
    struct.pack_into(endian + "H", data, 264, 2)
    struct.pack_into(endian + "HHII", data, 266, 0xf000, 4, 1, 48)
    # Pixel-data pointer is deliberately outside the file: must never be followed.
    struct.pack_into(endian + "HHII", data, 278, 0xf007, 4, 1, 0xfffffff0)
    struct.pack_into(endian + "I", data, 290, 0)
    struct.pack_into(endian + "H", data, 304, 2)
    struct.pack_into(endian + "HHII", data, 306, 0xf00d, 4, 4 if bad_count else 3, 900 if bad_value else 96)
    struct.pack_into(endian + "HHII", data, 318, 0xf00e, 4, 3, 112)
    struct.pack_into(endian + "I", data, 330, 8 if cycle else 0)
    struct.pack_into(endian + "4I", data, 352, 300, 600, 450, 999)
    struct.pack_into(endian + "3I", data, 368, 300, 700, 400)
    return bytes(data)


class ProbeTests(unittest.TestCase):
    def parse(self, data):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "synthetic.RAF"
            path.write_bytes(data)
            r = MetadataReader(path)
            try:
                return r.parse()
            finally:
                r.close()

    def test_exact_macros(self):
        self.assertEqual(rgb([300, 600, 450], "GRB"), [600, 300, 450, None])
        self.assertEqual(rgb([300, 600, 301, 450], "GRGB"), [600, 300, 450, 301])
        with self.assertRaises(ValueError):
            rgb([1, 2], "GRB")

    def test_both_endians_and_skip_pixels(self):
        for endian in ("<", ">"):
            out = self.parse(fixture(endian))
            self.assertEqual(len(out["records"]), 3)
            self.assertEqual(out["records"][1]["rgbg"][:3], [600, 300, 450])
            self.assertEqual(out["records"][2]["rgbg"][:3], [700, 300, 400])
            self.assertLess(out["reader_bytes_read"], 256)
            self.assertEqual(out["skipped_metadata"][0]["tag"], "0xc000")

    def test_cycle_and_bad_offset(self):
        for kwargs in ({"cycle": True}, {"bad_value": True}):
            with self.assertRaises(ValueError):
                self.parse(fixture(**kwargs))

    def test_unsupported_count_not_coerced(self):
        out = self.parse(fixture(bad_count=True))
        rec = out["records"][1]
        self.assertFalse(rec["supported"])
        self.assertEqual(rec["count"], 4)
        self.assertNotIn("rgbg", rec)

    def test_caps_and_section_bounds(self):
        for offset, value, fmt in ((128, 4097, ">I"), (264, 1025, "<H"), (0x60, 2048, ">I")):
            d = bytearray(fixture())
            struct.pack_into(fmt, d, offset, value)
            with self.assertRaises(ValueError):
                self.parse(d)

    def test_independent_disagreement(self):
        rec = [{"name": "WB_GRBLevelsAuto", "values": [300, 600, 450]}]
        self.assertEqual(crosscheck(rec, {"FujiIFD:WB_GRBLevelsAuto": "300 600 450"}), [])
        self.assertTrue(crosscheck(rec, {"FujiIFD:WB_GRBLevelsAuto": "300 700 450"}))
        self.assertTrue(crosscheck(rec, {}))
        self.assertTrue(crosscheck([], {"FujiIFD:WB_GRBLevelsAuto": "300 600 450"}))
        with self.assertRaises(ValueError):
            numeric_values("300 nan 450")

    def test_duplicate_metadata_records(self):
        rec = [{"name": "WB_GRBLevelsAuto", "values": [300, 600, 450]},
               {"name": "WB_GRBLevelsAuto", "values": [300, 610, 460]}]
        ex = {"FujiIFD:WB_GRBLevelsAuto": "300 610 460", "FujiIFD:Copy1:WB_GRBLevelsAuto": "300 600 450"}
        self.assertEqual(crosscheck(rec, ex), [])
        ex.pop("FujiIFD:Copy1:WB_GRBLevelsAuto")
        self.assertTrue(crosscheck(rec, ex))
        with self.assertRaises(ValueError):
            unique_pairs([("same", 1), ("same", 2)])

    def test_16byte_raw_wb(self):
        for second in ([300, 600, 300, 450], [300, 620, 300, 460]):
            data = bytearray(fixture())
            struct.pack_into(">IHH8H", data, 128, 1, 0x2000, 16, *[300, 600, 300, 450], *second)
            out = self.parse(data)
            rec = out["records"][0]
            same = second == [300, 600, 300, 450]
            self.assertEqual(rec["supported"], same)
            self.assertEqual(rec["duplicate_half_identical"], same)
            self.assertEqual(crosscheck([rec], {"RAF:WB_GRGBLevelsAuto": "300 600 300 450"}), [])
            self.assertEqual(rec["libraw_integer_average_rgbg_report_only"][0], (600 + second[1]) // 2)


if __name__ == "__main__":
    unittest.main()
