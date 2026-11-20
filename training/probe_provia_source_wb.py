"""F24-only metadata probe. No image decoding, model inference or fitting.

The independent reader follows only documented RAF metadata and TIFF IFD
links. ExifTool and SHA checks may read image bytes, but no code interprets,
unpacks, demosaics or exports them. Proprietary WB meanings remain inferred.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
MANIFEST = DIAG / "provia-source-wb-probe-manifest.json"
REPORT = DIAG / "provia-source-wb-probe.json"
TOOLCHAIN = DIAG / "provia-source-wb-toolchain.json"
SOURCE_NAMES = ("cuban-cohort-proposal.json", "provia-color-run-manifest.json",
                "cn-step1-wb-reference-audit.json", "provia-wb-metadata-audit.json")
TAGS = {0xf00c: "WB_GRBLevelsStandard", 0xf00d: "WB_GRBLevelsAuto", 0xf00e: "WB_GRBLevels"}
RAF_TAGS = dict(zip((0x2000, 0x2100, 0x2200, 0x2300, 0x2301, 0x2302, 0x2310,
                    0x2311, 0x2400, 0x2410, 0x2ff0),
                   ("WB_GRGBLevelsAuto", "WB_GRGBLevelsDaylight", "WB_GRGBLevelsCloudy",
                    "WB_GRGBLevelsDaylightFluor", "WB_GRGBLevelsDayWhiteFluor",
                    "WB_GRGBLevelsWhiteFluorescent", "WB_GRGBLevelsWarmWhiteFluor",
                    "WB_GRGBLevelsLivingRoomWarmWhiteFluor", "WB_GRGBLevelsTungsten",
                    "WB_GRGBLevelsFlash", "WB_GRGBLevels")))
MAX_IFDS, MAX_ENTRIES, MAX_RECORDS, MAX_VALUE_BYTES = 16, 1024, 4096, 256
LIMIT = .005


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    require(not path.exists(), f"Refusing overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(name):
    return json.loads((DIAG / name).read_text())


def tool_tree_sha():
    """Bind the whole portable Perl library, not just its Fuji tag table."""
    t = json.loads(TOOLCHAIN.read_text())
    exe = ROOT / t["script_relative_path"]
    paths = [exe] + sorted(p for p in (exe.parent / "lib").rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in paths:
        require(not p.is_symlink(), "Unexpected portable-reader symlink")
        h.update(str(p.relative_to(exe.parent)).encode() + b"\0" + sha(p).encode() + b"\n")
    return h.hexdigest()


def rgb(values, order):
    """Exact pinned LibRaw GRBG_2_RGBG / GRGB_2_RGBG, no fitted ordering."""
    require(len(values) == (3 if order == "GRB" else 4), "Unsupported WB count")
    out = [None] * 4
    for i, v in enumerate(values):
        j = (i ^ (i >> 1) ^ 1) if order == "GRB" else i ^ 1
        out[j] = v
    require(all(type(v) is int and v > 0 for v in out[:3]), "Invalid WB levels")
    return out


def ratios(levels):
    return [levels[0] / levels[1], levels[2] / levels[1]]


class MetadataReader:
    def __init__(self, path):
        self.file = open(path, "rb")
        self.size = self.file.seek(0, 2)
        self.bytes_read = 0
        self.records, self.ifds, self.skipped = [], [], []

    def close(self):
        self.file.close()

    def read(self, offset, length, lower=0, upper=None):
        upper = self.size if upper is None else upper
        require(type(offset) is int and type(length) is int and length >= 0 and
                lower <= offset <= upper and offset + length <= upper <= self.size,
                "Metadata offset/length out of bounds")
        self.file.seek(offset)
        data = self.file.read(length)
        require(len(data) == length, "Truncated metadata")
        self.bytes_read += length
        return data

    def parse(self):
        h = self.read(0, 0x70)
        require(h[:16] == b"FUJIFILMCCD-RAW ", "Not a supported RAF header")
        # Pinned ExifTool ProcessRAF: 0x5c RAF directory; 0x64 TIFF directory.
        for off, kind in ((0x5c, "RAF"), (0x64, "TIFF")):
            start, length = struct.unpack_from(">II", h, off)
            if not start:
                continue
            require(length > 0 and start + length <= self.size, "RAF section bounds")
            if kind == "RAF":
                self.raf_directory(start, length)
            else:
                self.tiff(start, length)
        return {"records": self.records, "ifds": self.ifds, "skipped_metadata": self.skipped,
                "reader_bytes_read": self.bytes_read, "file_size": self.size}

    def raf_directory(self, start, length):
        end = start + length
        n = struct.unpack(">I", self.read(start, 4, start, end))[0]
        require(n <= MAX_RECORDS, "RAF record count exceeds cap")
        cursor = start + 4
        for _ in range(n):
            tag, size = struct.unpack(">HH", self.read(cursor, 4, start, end))
            cursor += 4
            require(cursor + size <= end, "RAF record payload out of bounds")
            if tag in RAF_TAGS:
                rec = {"path": "RAF", "tag": hex(tag), "name": RAF_TAGS[tag],
                       "type": "BE uint16", "count": size // 2 if size % 2 == 0 else None,
                       "byte_count": size, "supported": size in (8, 16)}
                if size <= MAX_VALUE_BYTES and size % 2 == 0:
                    rec["values"] = list(struct.unpack(">" + "H" * (size // 2), self.read(cursor, size, start, end)))
                if "values" in rec:
                    # ExifTool explicitly returns Count=4, ignoring the duplicate half.
                    rec["exiftool_comparison_values"] = rec["values"][:4] if size in (8, 16) else rec["values"]
                if size == 16:
                    rec["duplicate_half_identical"] = rec["values"][:4] == rec["values"][4:]
                    rec["supported"] = rec["duplicate_half_identical"]
                    rec["libraw_integer_average_rgbg_report_only"] = rgb([(a + b) // 2 for a, b in zip(rec["values"][:4], rec["values"][4:])], "GRGB")
                if rec["supported"]:
                    rec["rgbg"] = rgb(rec["values"][:4], "GRGB")
                self.records.append(rec)
            elif tag == 0xc000:
                self.skipped.append({"path": "RAF", "tag": "0xc000", "byte_count": size,
                                     "reason": "RAFData payload deliberately not followed"})
            cursor += size

    def tiff(self, start, length):
        end = start + length
        h = self.read(start, 8, start, end)
        require(h[:2] in (b"II", b"MM"), "Unsupported RAF TIFF byte order")
        endian = "<" if h[:2] == b"II" else ">"
        require(struct.unpack_from(endian + "H", h, 2)[0] == 42, "Unsupported TIFF version")
        pending = [struct.unpack_from(endian + "I", h, 4)[0]]
        seen = set()
        while pending:
            rel = pending.pop()
            if rel == 0:
                continue
            require(rel not in seen, "IFD cycle or repeated link")
            seen.add(rel)
            require(len(seen) <= MAX_IFDS, "IFD count exceeds cap")
            pos = start + rel
            n = struct.unpack(endian + "H", self.read(pos, 2, start, end))[0]
            require(n <= MAX_ENTRIES, "IFD entry count exceeds cap")
            entries = self.read(pos + 2, n * 12 + 4, start, end)
            self.ifds.append({"offset_relative": rel, "entries": n, "byte_order": h[:2].decode()})
            for i in range(n):
                entry = entries[i * 12:i * 12 + 12]
                tag, typ, count, value = struct.unpack(endian + "HHII", entry)
                if tag == 0xf000:
                    require(typ in (4, 13) and count == 1, "Unsupported Fuji subIFD pointer")
                    pending.append(value)
                elif tag in TAGS:
                    rec = {"path": "FujiIFD", "ifd_relative": rel, "tag": hex(tag),
                           "name": TAGS[tag], "type": typ, "count": count,
                           "supported": typ in (3, 4) and count == 3 and tag != 0xf00c}
                    size = {3: 2, 4: 4}.get(typ)
                    if size is not None and 0 < count <= MAX_VALUE_BYTES // size:
                        data = entry[8:8 + size * count] if size * count <= 4 else self.read(start + value, size * count, start, end)
                        rec["values"] = list(struct.unpack(endian + ("H" if typ == 3 else "I") * count, data))
                    if rec["supported"]:
                        rec["rgbg"] = rgb(rec["values"], "GRB")
                    self.records.append(rec)
                # All other value/strip/tile offsets are deliberately not followed.
            pending.append(struct.unpack_from(endian + "I", entries, n * 12)[0])


def numeric_values(value):
    if isinstance(value, str):
        vals = [float(v) for v in value.split()]
    elif isinstance(value, list):
        vals = value
    else:
        vals = [value]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in vals), "Non-numeric ExifTool field")
    return vals


def crosscheck(records, external):
    problems, ours, theirs = [], defaultdict(list), defaultdict(list)
    for rec in records:
        if "values" not in rec:
            continue
        ours[rec["name"]].append(tuple(rec.get("exiftool_comparison_values", rec["values"])))
    for k, value in external.items():
        name = k.split(":")[-1]
        if name in set(TAGS.values()) | set(RAF_TAGS.values()):
            theirs[name].append(tuple(numeric_values(value)))
    for name in sorted(set(ours) | set(theirs)):
        if Counter(ours[name]) != Counter(theirs[name]):
            problems.append(f"{name}: reader/ExifTool one-to-one count/value disagreement")
    return problems


def unique_pairs(pairs):
    out = {}
    for k, v in pairs:
        require(k not in out, "Duplicate ExifTool JSON key; family4 grouping insufficient")
        out[k] = v
    return out


def prepare():
    cohort, pool, old, meta = [read_json(n) for n in SOURCE_NAMES]
    fits = [s for s in cohort["scenes"] if s["role"] == "fit"]
    allow = {s["name"]: s for s in pool["scenes"]}
    archive = {s["name"]: s for s in old["scenes"]}
    require(len(fits) == len({s["name"] for s in fits}) == 24, "F24 coverage")
    scenes = []
    for s in fits:
        n = s["name"]
        require(s["tier"] == "train" and n in allow and s["shoot"] == allow[n]["shoot"] == archive[n]["shoot"], "Non-allowlisted source")
        scenes.append({"name": n, "shoot": s["shoot"], "session": s["session"],
                       "source_wb_mode": int(s["asShotWhiteBalance"], 16),
                       "raf_sha256": archive[n]["source_raf_sha256_at_probe"]})
    require(sum(s["source_wb_mode"] == 0 for s in scenes) == 11, "Auto subset differs")
    dump(MANIFEST, {"scope": "F24 source RAF metadata only. SHA/ExifTool may read image bytes; no image interpretation, unpack, demosaic, export, inference or fitting. No K/monitor/finalHeldOut/T8/dr100Eval media.",
        "code_sha256": sha(__file__), "source_hashes": {n: sha(DIAG / n) for n in SOURCE_NAMES},
        "toolchain_sha256": sha(TOOLCHAIN), "portable_reader_tree_sha256": tool_tree_sha(),
        "scenes": sorted(scenes, key=lambda s: s["name"]),
        "reader_limits": {"IFDs": MAX_IFDS, "entries_per_IFD": MAX_ENTRIES, "RAF_records": MAX_RECORDS, "value_bytes": MAX_VALUE_BYTES},
        "parse_scope": "Primary RAF 0x5c directory and 0x64 TIFF; Fuji f000 + nextIFD links only. No strip/tile/f007/f008/C000 payload. Three-value TIFF gains; four-value RAF gains or eight with identical duplicate halves only. Eight-value RAF compares first4 with ExifTool Count4; unequal halves unscored, LibRaw integer-average report-only. Other counts recorded, never coerced. RAFData 0x0c5a/TSNERDTS deferred. Embedded-JPEG fine-tune/mode are ExifTool-only and checked against archived values, not independently parsed here. ExifTool -a -G1:4 preserves duplicates; exact multisets required.",
        "screen": "For each independent candidate tag f00d or 2000: present and supported in all11 source-Auto F24; exact ExifTool cross-check and current/as-shot anchor; normalized R/G and B/G within0.5% of archived zero-fine-tune Provia WB metadata on everyAuto scene; candidate differs >0.5% from as-shot on each nonzero-shift Auto scene; >=2sessions. Report shifts [5,-6] and [4,-5] separately. OtherWB modes inventory only.",
        "limits": "A match is metadata agreement on11 Auto scenes, not proof of fine-tune semantics, applied gain correctness, pixel improvement or parity. No source-zero-shift or AutoWhite/Ambience controls; manual/Kelvin target equivalence untested; no runtime/app/wrapper changes."})
    print("Frozen manifest SHA", sha(MANIFEST))


def run(expected):
    require(not REPORT.exists(), "Refusing existing report")
    require(sha(MANIFEST) == expected, "Manifest mismatch")
    for path in (MANIFEST, Path(__file__).resolve()):
        committed = subprocess.check_output(["git", "show", "HEAD:" + str(path.relative_to(ROOT))], cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == sha(path), "Probe code/manifest not committed")
    m = json.loads(MANIFEST.read_text())
    require(sha(__file__) == m["code_sha256"] and sha(TOOLCHAIN) == m["toolchain_sha256"], "Code/toolchain drift")
    require(tool_tree_sha() == m["portable_reader_tree_sha256"], "Portable Perl library tree drift")
    for n, h in m["source_hashes"].items():
        require(sha(DIAG / n) == h, "Evidence drift")
    t = json.loads(TOOLCHAIN.read_text())
    exe = ROOT / t["script_relative_path"]
    require(sha(exe) == t["script_sha256"] and sha(exe.parent / "lib/Image/ExifTool/FujiFilm.pm") == t["fuji_module_sha256"], "Portable reader drift")
    require(subprocess.check_output(["/usr/bin/perl", str(exe), "-ver"], text=True).strip() == t["exiftool_version"], "ExifTool version drift")
    for rec in t["wb_order_definitions"]:
        require(sha(ROOT / rec["header"]) == rec["sha256"] and rec["line"] in (ROOT / rec["header"]).read_text(), "Channel ordering source drift")
    old = {s["name"]: s for s in read_json(SOURCE_NAMES[2])["scenes"]}
    rows = []
    fields = list(TAGS.values()) + list(RAF_TAGS.values()) + ["FujiFilm:WhiteBalance", "FujiFilm:WhiteBalanceFineTune", "FujiFilm:ColorTemperature"]
    for s in m["scenes"]:
        path = ROOT / "calibration-input" / s["shoot"] / s["name"]
        require(sha(path) == s["raf_sha256"], f"RAF changed: {s['name']}")
        r = MetadataReader(path)
        try:
            parsed = r.parse()
        finally:
            r.close()
        proc = subprocess.run(["/usr/bin/perl", str(exe), "-j", "-n", "-a", "-G1:4", *["-" + f for f in fields], str(path)], capture_output=True, text=True, timeout=120)
        require(proc.returncode == 0, "ExifTool failed: " + proc.stderr[:200])
        ex = json.loads(proc.stdout, object_pairs_hook=unique_pairs)
        require(len(ex) == 1, "ExifTool coverage mismatch")
        external = {k: v for k, v in ex[0].items() if k != "SourceFile"}
        problems = crosscheck(parsed["records"], external)
        current = [v for v in parsed["records"] if v["tag"] in ("0xf00e", "0x2ff0") and v["supported"]]
        anchor = old[s["name"]]
        matched = [v["tag"] for v in current if v["rgbg"][:3] == anchor["libraw_cam_mul"][:3]]
        require(matched, "As-shot/archived LibRaw anchor mismatch")
        modes = [v for k, v in external.items() if k.split(":")[-1] == "WhiteBalance"]
        fine = [numeric_values(v) for k, v in external.items() if k.split(":")[-1] == "WhiteBalanceFineTune"]
        if not modes or any(v != s["source_wb_mode"] for v in modes):
            problems.append("ExifTool WB mode differs from archived source")
        if not fine or any(v != anchor["source"]["WBFineTune"] for v in fine):
            problems.append("ExifTool fine-tune differs from archived source")
        require(sha(path) == s["raf_sha256"], "RAF changed during probe")
        row = dict(s, **parsed, exiftool=external, crosscheck_problems=problems,
                   shift_steps=anchor["source_wb_shift_steps"], candidates={},
                   archived_cam_mul_matching_as_shot_tags=matched,
                   as_shot_records_agree=len({tuple(v["rgbg"][:3]) for v in current}) == 1)
        for tag in ("0xf00d", "0x2000"):
            records = [v for v in parsed["records"] if v["tag"] == tag and v["supported"]]
            if len(records) != 1 or s["source_wb_mode"] != 0:
                row["candidates"][tag] = {"scored": False, "reason": "Not a unique supported Auto record or source WB is not Auto"}
                continue
            gains = ratios(records[0]["rgbg"])
            target = anchor["provia_recorded_level_RB_over_G"]
            source = anchor["libraw_gain_RB_over_G"]
            err = [abs(a / b - 1) for a, b in zip(gains, target)]
            diff = [abs(a / b - 1) for a, b in zip(gains, source)]
            row["candidates"][tag] = {"scored": True, "normalized_RB": gains,
                "target_normalized_RB": target, "relative_target_error_RB": err,
                "relative_as_shot_difference_RB": diff,
                "matches_target": max(err) <= LIMIT, "independent_of_as_shot": max(diff) > LIMIT,
                "passed": not problems and max(err) <= LIMIT and max(diff) > LIMIT}
        rows.append(row)
        print(s["name"], "records", len(parsed["records"]), "crosscheck", "PASS" if not problems else problems, flush=True)
    auto = [r for r in rows if r["source_wb_mode"] == 0]
    screens = {tag: {"passed": len(auto) == 11 and len({r["session"] for r in auto}) >= 2 and all(r["candidates"][tag].get("passed", False) for r in auto),
                     "n_present_supported": sum(r["candidates"][tag]["scored"] for r in auto),
                     "n_matching_target": sum(r["candidates"][tag].get("matches_target", False) for r in auto),
                     "by_shift": {str(shift): {"n": sum(r["shift_steps"] == shift for r in auto),
                                      "n_passed": sum(r["shift_steps"] == shift and r["candidates"][tag].get("passed", False) for r in auto)}
                                  for shift in ([5., -6.], [4., -5.])}}
               for tag in ("0xf00d", "0x2000")}
    ok = all(not r["crosscheck_problems"] for r in rows)
    dump(REPORT, {"scope": m["scope"], "crosschecks_passed": ok, "manifest_sha256": expected, "toolchain_sha256": m["toolchain_sha256"],
                  "rows": rows, "screens": screens, "limits": m["limits"]})
    print(json.dumps(screens, indent=2))
    require(ok, "Independent metadata discrepancy: all candidate conclusions blocked")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--manifest-sha")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        require(bool(args.manifest_sha), "Frozen manifest SHA required")
        run(args.manifest_sha)
