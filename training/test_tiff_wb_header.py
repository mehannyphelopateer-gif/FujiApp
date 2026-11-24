"""Synthetic TIFF/MakerNote fixtures only; no real files, datasets, media or models."""
import ast
import io
import os
import struct
import tempfile
import unittest
from pathlib import Path

import tiff_wb_header as T

FORMATS = {3: ("H", 2), 4: ("I", 4), 8: ("h", 2), 9: ("i", 4)}
SENTINEL = b"PIXEL-SENTINEL!!"


def makernote(entries):
    """entries: (tag, type, values). Little-endian Fuji MakerNote, offsets relative to its start."""
    n = len(entries)
    data_start = 12 + 2 + 12 * n + 4
    table, extra = b"", b""
    for tag, typ, vals in entries:
        f, _ = FORMATS[typ]
        raw = struct.pack("<" + f * len(vals), *vals)
        field = raw.ljust(4, b"\0") if len(raw) <= 4 else struct.pack("<I", data_start + len(extra))
        if len(raw) > 4:
            extra += raw
        table += struct.pack("<HHI", tag, typ, len(vals)) + field
    return b"FUJIFILM" + struct.pack("<I", 12) + struct.pack("<H", n) + table + b"\0" * 4 + extra


def kelvin_entries(overrides=None):
    """overrides: {tag_int: (type, values) or None to drop the tag}."""
    base = {0x1002: (3, [4080]), 0x1005: (3, [6600]), 0x100a: (9, [20, -60]),
            0x144a: (3, [620]), 0x144b: (3, [302]), 0x144c: (3, [455])}
    base.update(overrides or {})
    return [(tag, item[0], item[1]) for tag, item in base.items() if item is not None]


def build(mn, order="<", magic=42, dup_exif=False, dup_mn=False, exif_to_ifd0=False,
          mn_len=None, mn_off=None, strip_len=4096):
    ent = lambda tag, typ, cnt, val: struct.pack(order + "HHI", tag, typ, cnt) + struct.pack(order + "I", val)
    n0 = 2 + dup_exif
    ifd0 = 8
    exif = ifd0 + 2 + 12 * n0 + 4
    n1 = 1 + dup_mn
    mn_start = exif + 2 + 12 * n1 + 4
    strip_off = mn_start + len(mn)
    ifd0_bytes = (struct.pack(order + "H", n0) + ent(0x0111, 4, 1, strip_off)
                  + ent(T.TAG_EXIF_IFD, 4, 1, ifd0 if exif_to_ifd0 else exif)
                  + (ent(T.TAG_EXIF_IFD, 4, 1, exif) if dup_exif else b"") + struct.pack(order + "I", 0))
    mn_entry = ent(T.TAG_MAKERNOTE, 7, len(mn) if mn_len is None else mn_len, mn_start if mn_off is None else mn_off)
    exif_bytes = struct.pack(order + "H", n1) + mn_entry + (mn_entry if dup_mn else b"") + struct.pack(order + "I", 0)
    header = (b"II" if order == "<" else b"MM") + struct.pack(order + "H", magic) + struct.pack(order + "I", ifd0)
    strip = (SENTINEL * (strip_len // len(SENTINEL) + 1))[:strip_len]
    return header + ifd0_bytes + exif_bytes + mn + strip, strip_off


class Guarded(io.BytesIO):
    """Fails loudly if any read touches the strip/pixel payload."""

    def __init__(self, data, forbidden_from):
        super().__init__(data)
        self.forbidden_from = forbidden_from

    def read(self, n=-1):
        start = self.tell()
        end = len(self.getbuffer()) if n is None or n < 0 else start + n
        if end > self.forbidden_from and start < len(self.getbuffer()):
            raise AssertionError(f"pixel payload read [{start},{end})")
        return super().read(n)


def parse(data, strip_off=None, **kw):
    return T.read_wb_header(Guarded(data, strip_off) if strip_off is not None else io.BytesIO(data), **kw)


class Valid(unittest.TestCase):
    def test_little_and_big_endian_outer_same_values_and_pixels_never_read(self):
        mn = makernote(kelvin_entries())
        results = []
        for order in ("<", ">"):
            data, strip_off = build(mn, order=order)
            r = parse(data, strip_off)
            self.assertTrue(all(x["offset"] + x["length"] <= strip_off for x in r["reads"]))
            self.assertEqual(r["bytes_read"], sum(x["length"] for x in r["reads"]))
            results.append({k: v for k, v in r.items() if k not in ("byte_order", "reads", "bytes_read")})
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], {"WhiteBalance": 4080, "ColorTemperature": 6600, "WBFineTune": [20, -60],
                                      "WBLevelR": 620, "WBLevelG": 302, "WBLevelB": 455})

    def test_signed_shift_sshort_and_non302_green(self):
        mn = makernote(kelvin_entries({0x100a: (8, [-180, 100]), 0x144b: (4, [325]), 0x1005: (4, [2500])}))
        data, s = build(mn)
        r = parse(data, s)
        self.assertEqual((r["WBFineTune"], r["WBLevelG"], r["ColorTemperature"]), ([-180, 100], 325, 2500))

    def test_kelvin_null_outside_kelvin_mode(self):
        for kelvin in ((3, [6600]), None):
            mn = makernote(kelvin_entries({0x1002: (3, [0]), 0x1005: kelvin}))
            data, s = build(mn)
            self.assertIsNone(parse(data, s)["ColorTemperature"])

    def test_path_input(self):
        data, _ = build(makernote(kelvin_entries()))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "synthetic.tif"
            p.write_bytes(data)
            self.assertEqual(T.read_wb_header(p)["WBLevelR"], 620)
            self.assertEqual(T.read_wb_header(str(p))["WBLevelB"], 455)


class Refusals(unittest.TestCase):
    def bad(self, data, **kw):
        with self.assertRaises(T.TiffWBHeaderError):
            parse(data, **kw)

    def test_missing_or_bad_tags(self):
        for over in ({0x1002: None}, {0x100a: None}, {0x144a: None}, {0x144b: None}, {0x144c: None},
                     {0x1005: None}, {0x144b: (3, [0])}, {0x100a: (4, [20, 60])}, {0x100a: (9, [20])},
                     {0x1002: (8, [4080])}, {0x144a: (3, [620, 621])}):
            self.bad(build(makernote(kelvin_entries(over)))[0])

    def test_duplicates(self):
        dup = makernote(kelvin_entries() + [(0x144a, 3, [621])])
        self.bad(build(dup)[0])
        mn = makernote(kelvin_entries())
        self.bad(build(mn, dup_exif=True)[0])
        self.bad(build(mn, dup_mn=True)[0])

    def test_cycle_offsets_truncation_and_budget(self):
        mn = makernote(kelvin_entries())
        self.bad(build(mn, exif_to_ifd0=True)[0])
        data, _ = build(mn)
        self.bad(build(mn, mn_off=len(data) + 10)[0])
        self.bad(build(mn, mn_len=T.MAX_SINGLE_READ + 1)[0])
        self.bad(data[:60])
        self.bad(data, max_total=64)
        self.bad(data, max_read=8)
        with self.assertRaises(T.TiffWBHeaderError):
            T.read_wb_header(io.BytesIO(data), max_total=T.MAX_TOTAL_READ + 1)

    def test_makernote_internal_bounds(self):
        mn = bytearray(makernote(kelvin_entries({0x100a: (9, [20, -60])})))
        good = bytes(mn)
        bad_header = b"NOTFUJI!" + good[8:]
        self.bad(build(bad_header)[0])
        oob_ifd = good[:8] + struct.pack("<I", len(good) + 4) + good[12:]
        self.bad(build(oob_ifd)[0])
        too_many = good[:12] + struct.pack("<H", T.MAX_IFD_ENTRIES + 1) + good[14:]
        self.bad(build(too_many)[0])
        # WBFineTune (8 bytes) lives out of line: point its value offset past the MakerNote end.
        entries = kelvin_entries()
        idx = [t for t, _, _ in entries].index(0x100a)
        pos = 12 + 2 + 12 * idx + 8
        oob_value = good[:pos] + struct.pack("<I", len(good)) + good[pos + 4:]
        self.bad(build(oob_value)[0])

    def test_header_refusals(self):
        mn = makernote(kelvin_entries())
        self.bad(build(mn, magic=43)[0])
        self.bad(build(mn, magic=41)[0])
        data, _ = build(mn)
        self.bad(b"XX" + data[2:])
        zero_ifd = data[:4] + struct.pack("<I", 0) + data[8:]
        self.bad(zero_ifd)
        self.bad(b"")


class Isolation(unittest.TestCase):
    def test_no_image_library_imports(self):
        src = Path(T.__file__).read_text()
        names = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertTrue(names <= {"struct"}, names)
        self.assertFalse(names & {"PIL", "tifffile", "numpy", "cv2", "imageio"})


if __name__ == "__main__":
    unittest.main()
