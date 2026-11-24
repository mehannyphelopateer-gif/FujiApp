"""Bounded, metadata-only reader of RAF header geometry descriptors (standard library only).

Reference: ExifTool lib/Image/ExifTool/FujiFilm.pm (pinned revision in
training/diagnostics/raf-geometry-protocol-source.json): RAFHeader offsets, RAF directory
format (ProcessFujiDir) and RAF tags 0x100/0x110/0x111/0x115/0x117/0x118/0x119.

Reads ONLY: the first 108 header bytes, optionally the 4-byte compression field at 0x6c
(only if the JPEG starts at or after byte 112 and the field's first three bytes are zero),
optionally the extended header pointers 0x78-0x87 (only if the JPEG starts at or after byte
148), and, inside each RAF directory, the 4-byte count, each 4-byte record header and the
4-byte payloads of the known geometry tags. Unknown record payloads are skipped by seek,
never read. JPEG, FujiIFD/TIFF, M-RAW and CFA payloads are never read. No decoder is invoked.

Values are returned RAW, in stored order: documented as (height, width) for 0x100 full size,
0x111 cropped size, 0x115 aspect, 0x118 zoom top-left and 0x119 zoom size, and (top, left) for
0x110 crop top-left. The size tags include borders. Nothing here infers a visible crop, field
of view, orientation, Fuji output size or any transform.
"""
import os
import struct

RAF_MAGIC = b"FUJIFILMCCD-RAW "
MAX_TOTAL_READ = 4096
HEADER_PREFIX = 108          # 0x00..0x6b
COMPRESSION_OFFSET = 0x6C
COMPRESSION_MIN_JPEG = 112   # header must reach 0x70 before the JPEG
EXTENDED_OFFSET = 0x78
EXTENDED_LENGTH = 16         # 0x78 RAF1 off, 0x7c len, 0x80 FujiIFD1 off, 0x84 len
EXTENDED_MIN_JPEG = 148      # header must reach 0x94 before the JPEG
MAX_DIR_RECORDS = 255        # ExifTool: entries < 256
PAIR_TAGS = {0x100: "raw_image_full_size", 0x110: "raw_image_crop_top_left", 0x111: "raw_image_cropped_size",
             0x115: "raw_image_aspect_ratio", 0x118: "raw_zoom_top_left", 0x119: "raw_zoom_size"}
U32_TAGS = {0x117: "raw_zoom_active"}
KNOWN_TAGS = {**PAIR_TAGS, **U32_TAGS}
STORED_ORDER = {"raw_image_full_size": "height,width (includes borders)", "raw_image_crop_top_left": "top,left",
                "raw_image_cropped_size": "height,width (includes borders)", "raw_image_aspect_ratio": "height,width",
                "raw_zoom_top_left": "height-axis,width-axis (relative to cropped size)",
                "raw_zoom_size": "height,width (relative to cropped size)", "raw_zoom_active": "uint32"}


class RafCropHeaderError(ValueError):
    pass


def _need(cond, msg):
    if not cond:
        raise RafCropHeaderError(msg)


class _Reader:
    def __init__(self, f, max_total):
        self.f, self.max_total = f, max_total
        f.seek(0, 2)
        self.size = f.tell()
        self.total = 0
        self.reads = []

    def read(self, offset, length, purpose):
        _need(type(offset) is int and type(length) is int and offset >= 0 and length > 0, f"{purpose}: invalid read")
        _need(offset + length <= self.size, f"{purpose}: read beyond end of file")
        _need(self.total + length <= self.max_total, f"{purpose}: cumulative read budget exceeded")
        self.f.seek(offset)
        data = self.f.read(length)
        _need(isinstance(data, bytes) and len(data) == length, f"{purpose}: truncated or non-binary read")
        self.total += length
        self.reads.append({"offset": offset, "length": length, "purpose": purpose})
        return data


def _u32(buf, at):
    return struct.unpack_from(">I", buf, at)[0]


def _section(name, offset, length, size):
    if offset == 0 and length == 0:
        return None
    _need(offset > 0 and length > 0, f"{name}: half-specified section ({offset}, {length})")
    _need(offset + length <= size, f"{name}: section extends beyond end of file")
    return {"name": name, "offset": offset, "length": length}


def _overlap(a, b):
    return a["offset"] < b["offset"] + b["length"] and b["offset"] < a["offset"] + a["length"]


def _parse_directory(r, section, index):
    start, end = section["offset"], section["offset"] + section["length"]
    _need(section["length"] >= 4, f"RAF dir {index}: shorter than its count field")
    count = _u32(r.read(start, 4, f"RAF dir {index} count"), 0)
    _need(count <= MAX_DIR_RECORDS, f"RAF dir {index}: record count {count} >= 256")
    tags = {name: None for name in KNOWN_TAGS.values()}
    unknown = []
    pos = start + 4
    for i in range(count):
        _need(pos + 4 <= end, f"RAF dir {index}: record {i} header beyond directory")
        tag, length = struct.unpack(">HH", r.read(pos, 4, f"RAF dir {index} record {i} header"))
        payload = pos + 4
        _need(payload + length <= end, f"RAF dir {index}: record 0x{tag:04x} payload beyond directory")
        if tag in KNOWN_TAGS:
            name = KNOWN_TAGS[tag]
            _need(tags[name] is None, f"RAF dir {index}: duplicate tag 0x{tag:04x}")
            _need(length == 4, f"RAF dir {index}: tag 0x{tag:04x} length {length} != 4")
            raw = r.read(payload, 4, f"RAF dir {index} tag 0x{tag:04x}")
            tags[name] = list(struct.unpack(">HH", raw)) if tag in PAIR_TAGS else _u32(raw, 0)
        else:
            unknown.append({"tag": tag, "length": length, "offset": payload})  # payload skipped, never read
        pos = payload + length
    return {"index": index, "offset": start, "length": section["length"], "record_count": count,
            "tags": tags, "stored_order": dict(STORED_ORDER), "unknown_records": unknown}


def read_raf_crop_header(source, *, max_total=MAX_TOTAL_READ):
    """source: path (str/PathLike) or binary file-like with seek/read. Returns raw descriptors."""
    _need(type(max_total) is int and 0 < max_total <= MAX_TOTAL_READ, "max_total may only tighten the 4096-byte budget")
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            return _read(f, max_total)
    _need(hasattr(source, "read") and hasattr(source, "seek") and not isinstance(source, (bytes, bytearray)),
          "source must be a path or a binary file-like object")
    return _read(source, max_total)


def _read(f, max_total):
    r = _Reader(f, max_total)
    head = r.read(0, HEADER_PREFIX, "RAF header prefix")
    _need(head[:16] == RAF_MAGIC, "not a FUJIFILMCCD-RAW file")
    mraw_off, mraw_len = _u32(head, 0x48), _u32(head, 0x4C)
    _need(mraw_off == 0 and mraw_len == 0, "M-RAW (multi-image) RAF is not supported")
    jpeg_off, jpeg_len = _u32(head, 0x54), _u32(head, 0x58)
    _need(jpeg_off >= HEADER_PREFIX, "JPEG offset inside the RAF header prefix")

    compression = None
    if jpeg_off >= COMPRESSION_MIN_JPEG:
        raw = r.read(COMPRESSION_OFFSET, 4, "RAF compression field")
        if raw[:3] == b"\0\0\0":
            compression = _u32(raw, 0)
    header_end = COMPRESSION_MIN_JPEG if jpeg_off >= COMPRESSION_MIN_JPEG else HEADER_PREFIX

    raf_dirs = [_section("raf_dir_0", _u32(head, 0x5C), _u32(head, 0x60), r.size)]
    fuji_ifds = [_section("fuji_ifd_0", _u32(head, 0x64), _u32(head, 0x68), r.size)]
    extended = False
    if jpeg_off >= EXTENDED_MIN_JPEG:
        ext = r.read(EXTENDED_OFFSET, EXTENDED_LENGTH, "RAF extended header pointers")
        raf_dirs.append(_section("raf_dir_1", _u32(ext, 0), _u32(ext, 4), r.size))
        fuji_ifds.append(_section("fuji_ifd_1", _u32(ext, 8), _u32(ext, 12), r.size))
        header_end, extended = EXTENDED_MIN_JPEG, True
    _need(raf_dirs[0] is not None, "primary RAF directory missing")
    jpeg = _section("jpeg", jpeg_off, jpeg_len, r.size)

    header = {"name": "header", "offset": 0, "length": header_end}
    others = [header] + [s for s in [jpeg, *fuji_ifds] if s is not None]
    present_dirs = [d for d in raf_dirs if d is not None]
    for d in present_dirs:
        for o in others:
            _need(not _overlap(d, o), f"{d['name']} overlaps {o['name']}")
    if len(present_dirs) == 2:
        _need(not _overlap(present_dirs[0], present_dirs[1]), "RAF directories overlap")

    directories = [_parse_directory(r, d, int(d["name"][-1])) for d in present_dirs]
    return {"magic": RAF_MAGIC.decode(), "format_version_hex": head[16:20].hex(), "firmware_version_hex": head[0x3C:0x40].hex(),
            "compression": compression, "extended_header_read": extended,
            "sections": {"header": header, "jpeg": jpeg, "raf_dirs": raf_dirs, "fuji_ifds": fuji_ifds,
                         "mraw": None},
            "directories": directories, "bytes_read": r.total, "reads": r.reads,
            "note": "Raw stored-order descriptors only; size tags include borders; no crop/FOV/transform inferred."}
