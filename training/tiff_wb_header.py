"""Bounded, metadata-only reader of Fuji WB tags from a classic TIFF header.

Reads ONLY: the 8-byte TIFF header, IFD0's entry table, the Exif IFD (tag 0x8769) entry table and
the MakerNote blob (Exif tag 0x927C). It never follows StripOffsets/TileOffsets or any other value
offset, never reads ICC profiles or pixel data, and never decodes an image. Standard library only.

MakerNote layout follows training/cuban_exports.read_settings: "FUJIFILM" + little-endian uint32 IFD
offset relative to the MakerNote start; the MakerNote IFD is always little-endian, and value offsets
are relative to the MakerNote start, independent of the outer TIFF byte order.

Returned fields (raw tag values, no interpretation or WB policy):
  WhiteBalance (0x1002, required), ColorTemperature (0x1005, required only when WhiteBalance == 4080,
  otherwise None), WBFineTune (0x100a, required, two signed ints), WBLevelR/G/B (0x144a/b/c, required,
  positive; green is not assumed to be 302).
"""
import struct

MAX_SINGLE_READ = 256 * 1024
MAX_TOTAL_READ = 1024 * 1024
MAX_IFD_ENTRIES = 4096
KELVIN_MODE = 4080
TAG_EXIF_IFD = 0x8769
TAG_MAKERNOTE = 0x927C
TAG_NAMES = {0x1002: "WhiteBalance", 0x1005: "ColorTemperature", 0x100a: "WBFineTune",
             0x144a: "WBLevelR", 0x144b: "WBLevelG", 0x144c: "WBLevelB"}
FMT = {3: ("H", 2), 4: ("I", 4), 8: ("h", 2), 9: ("i", 4)}
UNSIGNED = (3, 4)
SIGNED = (8, 9)


class TiffWBHeaderError(ValueError):
    pass


def _need(cond, msg):
    if not cond:
        raise TiffWBHeaderError(msg)


class _BoundedReader:
    """Seek/read wrapper enforcing in-bounds reads, a per-read cap and a cumulative cap."""

    def __init__(self, f, max_read, max_total):
        self.f, self.max_read, self.max_total = f, max_read, max_total
        f.seek(0, 2)
        self.size = f.tell()
        self.total = 0
        self.reads = []

    def read(self, offset, length, purpose):
        _need(type(offset) is int and type(length) is int and offset >= 0 and length >= 0, "invalid read range")
        _need(offset + length <= self.size, f"{purpose}: read out of bounds")
        _need(length <= self.max_read, f"{purpose}: single read exceeds cap")
        _need(self.total + length <= self.max_total, f"{purpose}: cumulative read budget exceeded")
        self.f.seek(offset)
        data = self.f.read(length)
        _need(len(data) == length, f"{purpose}: truncated read")
        self.total += length
        self.reads.append({"offset": offset, "length": length, "purpose": purpose})
        return data


def _ifd_entries(reader, order, offset, purpose):
    """Entry table only: (tag, type, count, 4-byte value field). The next-IFD pointer is not followed."""
    _need(offset >= 8, f"{purpose}: IFD offset inside header")
    (count,) = struct.unpack(order + "H", reader.read(offset, 2, f"{purpose} count"))
    _need(1 <= count <= MAX_IFD_ENTRIES, f"{purpose}: entry count {count} outside 1..{MAX_IFD_ENTRIES}")
    table = reader.read(offset + 2, 12 * count, f"{purpose} entries")
    return [struct.unpack_from(order + "HHI4s", table, 12 * i) for i in range(count)]


def _single(entries, tag, purpose):
    found = [e for e in entries if e[0] == tag]
    _need(len(found) == 1, f"{purpose}: expected exactly one tag 0x{tag:04x}, found {len(found)}")
    return found[0]


def _makernote_values(mn):
    _need(len(mn) >= 14 and mn[:8] == b"FUJIFILM", "MakerNote is not a FUJIFILM MakerNote")
    (ifd,) = struct.unpack_from("<I", mn, 8)
    _need(12 <= ifd and ifd + 2 <= len(mn), "MakerNote IFD offset out of bounds")
    (count,) = struct.unpack_from("<H", mn, ifd)
    _need(1 <= count <= MAX_IFD_ENTRIES, f"MakerNote entry count {count} outside 1..{MAX_IFD_ENTRIES}")
    _need(ifd + 2 + 12 * count <= len(mn), "MakerNote entry table out of bounds")
    out = {}
    for i in range(count):
        tag, typ, cnt, field = struct.unpack_from("<HHI4s", mn, ifd + 2 + 12 * i)
        if tag not in TAG_NAMES:
            continue
        name = TAG_NAMES[tag]
        _need(name not in out, f"duplicate MakerNote tag {name}")
        _need(typ in FMT, f"{name}: unsupported type {typ}")
        _need(cnt >= 1, f"{name}: zero count")
        f, size = FMT[typ]
        nbytes = size * cnt
        if nbytes <= 4:
            raw = field[:nbytes]
        else:
            (off,) = struct.unpack("<I", field)
            _need(off + nbytes <= len(mn), f"{name}: value offset out of MakerNote bounds")
            raw = mn[off:off + nbytes]
        out[name] = {"type": typ, "values": list(struct.unpack("<" + f * cnt, raw))}
    return out


def _unsigned_scalar(raw, name, positive):
    _need(name in raw, f"missing required tag {name}")
    item = raw[name]
    _need(item["type"] in UNSIGNED and len(item["values"]) == 1, f"{name}: expected one unsigned value")
    value = item["values"][0]
    _need(value > 0 if positive else value >= 0, f"{name}: invalid value {value}")
    return value


def read_wb_header(source, *, max_read=MAX_SINGLE_READ, max_total=MAX_TOTAL_READ):
    """source: path or binary file-like with seek/read. Returns WB fields plus a read descriptor."""
    _need(type(max_read) is int and 0 < max_read <= MAX_SINGLE_READ, "max_read may only tighten the cap")
    _need(type(max_total) is int and 0 < max_total <= MAX_TOTAL_READ, "max_total may only tighten the cap")
    if hasattr(source, "read") and hasattr(source, "seek"):
        return _read(source, max_read, max_total)
    with open(source, "rb") as f:
        return _read(f, max_read, max_total)


def _read(f, max_read, max_total):
    r = _BoundedReader(f, max_read, max_total)
    header = r.read(0, 8, "TIFF header")
    _need(header[:2] in (b"II", b"MM"), "not a TIFF byte-order mark")
    order = "<" if header[:2] == b"II" else ">"
    (magic,) = struct.unpack(order + "H", header[2:4])
    _need(magic != 43, "BigTIFF is not supported")
    _need(magic == 42, f"bad TIFF magic {magic}")
    (ifd0,) = struct.unpack(order + "I", header[4:8])

    _, typ, cnt, field = _single(_ifd_entries(r, order, ifd0, "IFD0"), TAG_EXIF_IFD, "IFD0")
    _need(typ in (4, 13) and cnt == 1, "Exif IFD pointer must be one LONG/IFD")
    (exif,) = struct.unpack(order + "I", field)
    _need(exif != ifd0, "Exif IFD pointer cycles to IFD0")

    _, typ, cnt, field = _single(_ifd_entries(r, order, exif, "Exif IFD"), TAG_MAKERNOTE, "Exif IFD")
    _need(typ in (1, 7), "MakerNote must be BYTE/UNDEFINED")
    _need(14 <= cnt <= max_read, "MakerNote length outside supported range")
    (mn_off,) = struct.unpack(order + "I", field)
    _need(mn_off >= 8, "MakerNote offset inside header")
    raw = _makernote_values(r.read(mn_off, cnt, "MakerNote"))

    wb = _unsigned_scalar(raw, "WhiteBalance", positive=False)
    _need("WBFineTune" in raw, "missing required tag WBFineTune")
    ft = raw["WBFineTune"]
    _need(ft["type"] in SIGNED and len(ft["values"]) == 2, "WBFineTune: expected two signed values")
    kelvin = _unsigned_scalar(raw, "ColorTemperature", positive=True) if wb == KELVIN_MODE else None
    levels = {name: _unsigned_scalar(raw, name, positive=True) for name in ("WBLevelR", "WBLevelG", "WBLevelB")}
    return {"WhiteBalance": wb, "ColorTemperature": kelvin, "WBFineTune": list(ft["values"]), **levels,
            "byte_order": header[:2].decode(), "bytes_read": r.total, "reads": r.reads}
