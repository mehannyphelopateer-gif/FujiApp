"""Synthetic RAF-header fixtures only; no real RAF, media, decoder or dataset."""
import ast
import io
import struct
import tempfile
import unittest
from pathlib import Path

import raf_crop_header as R

SENTINEL = b"FORBIDDEN-PAYLOAD"


def rec(tag, payload):
    return struct.pack(">HH", tag, len(payload)) + payload


def pair(a, b):
    return struct.pack(">HH", a, b)


def standard_records(full=(5152, 7752), top_left=(0, 12), cropped=(5152, 7728), aspect=(2, 3)):
    return [rec(0x100, pair(*full)), rec(0x110, pair(*top_left)), rec(0x111, pair(*cropped)),
            rec(0x130, SENTINEL[:8]), rec(0x115, pair(*aspect)), rec(0x2ff0, SENTINEL * 2)]


def build(dirs, *, header_len=148, compression=b"\0\0\0\x02", jpeg_first=False, mraw=(0, 0), overrides=None, count=None):
    """Sequential layout: header, [jpeg], dirs..., fuji_ifd, [jpeg]. Returns (bytes, forbidden ranges, layout).

    Consistency: a JPEG at or after byte 148 implies the extended header exists, so header_len 112/108
    fixtures must use jpeg_first=True (JPEG directly after the shorter header).
    """
    assert header_len == 148 or jpeg_first, "short-header fixtures must place the JPEG first"
    jpeg = SENTINEL * 8
    fuji = SENTINEL * 4
    body, forbidden, layout = b"", [], {}
    pos = header_len
    if jpeg_first:
        layout["jpeg"] = (pos, len(jpeg)); forbidden.append((pos, pos + len(jpeg))); body += jpeg; pos += len(jpeg)
    for i, records in enumerate(dirs):
        n = len(records) if count is None else count
        blob = struct.pack(">I", n) + b"".join(records)
        layout[f"dir{i}"] = (pos, len(blob))
        p = pos + 4
        for r in records:
            tag, length = struct.unpack(">HH", r[:4])
            if tag not in R.KNOWN_TAGS:
                forbidden.append((p + 4, p + 4 + length))
            p += 4 + length
        body += blob; pos += len(blob)
    layout["fuji"] = (pos, len(fuji)); forbidden.append((pos, pos + len(fuji))); body += fuji; pos += len(fuji)
    if not jpeg_first:
        layout["jpeg"] = (pos, len(jpeg)); forbidden.append((pos, pos + len(jpeg))); body += jpeg; pos += len(jpeg)
    h = bytearray(header_len)
    h[:16] = R.RAF_MAGIC
    h[16:20] = b"0201"
    h[0x3C:0x40] = b"\x01\x02\x00\x00"
    put = lambda off, v: struct.pack_into(">I", h, off, v)
    put(0x48, mraw[0]); put(0x4C, mraw[1])
    put(0x54, layout["jpeg"][0]); put(0x58, layout["jpeg"][1])
    put(0x5C, layout["dir0"][0]); put(0x60, layout["dir0"][1])
    put(0x64, layout["fuji"][0]); put(0x68, layout["fuji"][1])
    if header_len >= 112:
        h[0x6C:0x70] = compression
    if header_len >= 148 and "dir1" in layout:
        put(0x78, layout["dir1"][0]); put(0x7C, layout["dir1"][1])
    for off, v in (overrides or {}).items():
        put(off, v)
    return bytes(h) + body, forbidden, layout


class Guarded(io.BytesIO):
    def __init__(self, data, forbidden):
        super().__init__(data)
        self.forbidden = forbidden

    def read(self, n=-1):
        start = self.tell()
        end = len(self.getbuffer()) if n is None or n < 0 else start + n
        for a, b in self.forbidden:
            if start < b and a < end:
                raise AssertionError(f"forbidden payload read [{start},{end}) overlaps [{a},{b})")
        return super().read(n)


def parse(data, forbidden=(), **kw):
    return R.read_raf_crop_header(Guarded(data, list(forbidden)), **kw)


class Valid(unittest.TestCase):
    def test_single_directory_raw_values_and_reads(self):
        data, forbidden, _ = build([standard_records()])
        r = parse(data, forbidden)
        self.assertEqual(r["reads"][0], {"offset": 0, "length": 108, "purpose": "RAF header prefix"})
        self.assertEqual((r["format_version_hex"], r["firmware_version_hex"], r["compression"]), ("30323031", "01020000", 2))
        self.assertTrue(r["extended_header_read"])
        self.assertIsNone(r["sections"]["raf_dirs"][1])
        self.assertEqual(len(r["directories"]), 1)
        t = r["directories"][0]["tags"]
        self.assertEqual(t["raw_image_full_size"], [5152, 7752])          # stored order, not swapped
        self.assertEqual(t["raw_image_crop_top_left"], [0, 12])
        self.assertEqual(t["raw_image_cropped_size"], [5152, 7728])
        self.assertEqual(t["raw_image_aspect_ratio"], [2, 3])
        for missing in ("raw_zoom_active", "raw_zoom_top_left", "raw_zoom_size"):
            self.assertIsNone(t[missing])
        self.assertEqual([u["tag"] for u in r["directories"][0]["unknown_records"]], [0x130, 0x2FF0])
        self.assertEqual(r["bytes_read"], sum(x["length"] for x in r["reads"]))
        self.assertLessEqual(r["bytes_read"], R.MAX_TOTAL_READ)

    def test_zoom_tags_parsed(self):
        recs = standard_records() + [rec(0x117, struct.pack(">I", 1)), rec(0x118, pair(10, 20)), rec(0x119, pair(100, 200))]
        data, forbidden, _ = build([recs])
        t = parse(data, forbidden)["directories"][0]["tags"]
        self.assertEqual((t["raw_zoom_active"], t["raw_zoom_top_left"], t["raw_zoom_size"]), (1, [10, 20], [100, 200]))

    def test_two_directories_preserved_not_merged(self):
        a = standard_records(cropped=(5152, 7728))
        b = [rec(0x111, pair(5149, 7728)), rec(0x100, pair(5160, 7760))]
        data, forbidden, _ = build([a, b], header_len=148)
        r = parse(data, forbidden)
        self.assertTrue(r["extended_header_read"])
        d0, d1 = r["directories"]
        self.assertEqual((d0["index"], d1["index"]), (0, 1))
        self.assertEqual(d0["tags"]["raw_image_cropped_size"], [5152, 7728])
        self.assertEqual(d1["tags"]["raw_image_cropped_size"], [5149, 7728])
        self.assertIsNone(d1["tags"]["raw_image_crop_top_left"])

    def test_short_header_reads_compression_but_not_extended_pointers(self):
        data, forbidden, _ = build([standard_records()], header_len=112, jpeg_first=True)
        r = parse(data, forbidden)
        self.assertEqual(r["compression"], 2)
        self.assertFalse(r["extended_header_read"])
        self.assertNotIn(R.EXTENDED_OFFSET, [x["offset"] for x in r["reads"]])
        self.assertEqual(len(r["sections"]["raf_dirs"]), 1)

    def test_compression_null_cases(self):
        data, forbidden, layout = build([standard_records()], header_len=108, jpeg_first=True)
        r = parse(data, forbidden)
        self.assertIsNone(r["compression"])
        self.assertNotIn(R.COMPRESSION_OFFSET, [x["offset"] for x in r["reads"]])
        data, forbidden, _ = build([standard_records()], compression=b"\xff\xd8\xff\xe1")
        self.assertIsNone(parse(data, forbidden)["compression"])

    def test_path_and_file_like_types(self):
        data, _, _ = build([standard_records()])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "synthetic.RAF"
            p.write_bytes(data)
            self.assertEqual(R.read_raf_crop_header(p)["compression"], 2)
            self.assertEqual(R.read_raf_crop_header(str(p))["compression"], 2)
        for bad in (data, bytearray(data), 7, None):
            with self.assertRaises(R.RafCropHeaderError):
                R.read_raf_crop_header(bad)


class Refusals(unittest.TestCase):
    def bad(self, data, forbidden=(), **kw):
        with self.assertRaises(R.RafCropHeaderError):
            parse(data, forbidden, **kw)

    def test_header_refusals(self):
        data, f, _ = build([standard_records()])
        self.bad(b"FUJIFILMXXX-RAW " + data[16:], f)
        self.bad(build([standard_records()], mraw=(4096, 64))[0])
        self.bad(data[:100])
        self.bad(build([standard_records()], overrides={0x54: 100})[0])           # JPEG inside header prefix
        self.bad(build([standard_records()], overrides={0x60: 0})[0])             # half-specified primary dir
        self.bad(build([standard_records()], overrides={0x5C: 0, 0x60: 0})[0])    # primary dir missing

    def test_bounds_and_overlap(self):
        data, f, layout = build([standard_records()])
        self.bad(build([standard_records()], overrides={0x60: len(data)})[0])           # dir beyond file
        self.bad(build([standard_records()], overrides={0x5C: layout["jpeg"][0]})[0])   # dir overlaps JPEG
        self.bad(build([standard_records()], overrides={0x5C: layout["fuji"][0]})[0])   # dir overlaps FujiIFD
        self.bad(build([standard_records()], overrides={0x5C: 104})[0])                 # dir overlaps header
        two, _, lay2 = build([standard_records(), standard_records()], header_len=148,
                             overrides={})
        self.bad(build([standard_records(), standard_records()], header_len=148,
                       overrides={0x78: lay2["dir0"][0], 0x7C: lay2["dir0"][1]})[0])   # dirs overlap
        self.bad(data[:-40])                                                            # truncated file

    def test_directory_refusals(self):
        self.bad(build([standard_records()], count=256)[0])
        self.bad(build([standard_records()], count=7)[0])                                # records beyond directory
        self.bad(build([standard_records() + [rec(0x111, pair(1, 2))]])[0])              # duplicate known tag
        self.bad(build([[rec(0x100, b"\0\x01")]])[0])                                    # known tag wrong length
        self.bad(build([[rec(0x117, b"\0\0\0\0\0\0")]])[0])

    def test_budget(self):
        data, f, _ = build([standard_records()])
        self.bad(data, f, max_total=120)
        for bad in (0, 4097, 1.0, True):
            with self.assertRaises(R.RafCropHeaderError):
                R.read_raf_crop_header(io.BytesIO(data), max_total=bad)
        many = [rec(0x4000 + i, b"") for i in range(255)]
        data, f, _ = build([many])
        r = parse(data, f)
        self.assertEqual(r["directories"][0]["record_count"], 255)
        self.assertLessEqual(r["bytes_read"], R.MAX_TOTAL_READ)


class Isolation(unittest.TestCase):
    def test_stdlib_struct_only(self):
        tree = ast.parse(Path(R.__file__).read_text())
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(names, {"os", "struct"})


if __name__ == "__main__":
    unittest.main()
