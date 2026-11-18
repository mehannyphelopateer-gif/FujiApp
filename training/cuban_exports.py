"""Classic Cuban Neg X RAW Studio exports - pilot selection, export-folder
prep, verification and installation. Tag/header reads plus a 16-bit
sample-depth check only; no fitting, no scoring, never touches finalHeldOut
or Round 11.

  python3 training/cuban_exports.py prepare-pilot   # pick 2 train RAFs, make cuban-pilot-batch/E1..E5
  python3 training/cuban_exports.py verify pilot    # check every exported TIFF (settings/orientation/colour/16-bit)
  python3 training/cuban_exports.py install pilot   # only after verify passes: copy into Shoot folders
  python3 training/cuban_exports.py prepare-repeat  # pilot RAFs, R1/R2 repeat folders
  python3 training/cuban_exports.py prepare-batch1  # 8 C scenes x C1-C8 + 6 scenes x E1-E5
  (`repeat` / `controls` / `cohort1` / `validation` in place of `pilot` for those batches)

Variants (Codex-approved 2026-11-17). Each is a separate X RAW Studio pass
into its own subfolder; installed names are distinct per stage so the
Provia corpus files (xraw-phase3.tiff etc.) are never overwritten.
"""
import hashlib
import io
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

CAL = Path(__file__).resolve().parent.parent / "calibration-input"
BATCHES = {"pilot": (CAL / "cuban-pilot-batch", CAL / "cuban-pilot-manifest.json"),
           "validation": (CAL / "cuban-validation-batch", CAL / "cuban-validation-manifest.json"),
           "repeat": (CAL / "cuban-repeat-batch", CAL / "cuban-repeat-manifest.json"),
           "controls": (CAL / "cuban-controls-batch", CAL / "cuban-controls-manifest.json"),
           "cohort1": (CAL / "cuban-cohort1-batch", CAL / "cuban-cohort1-manifest.json")}
REPORT = {k: CAL / f"cuban-{k}-export-check.json" for k in BATCHES}

# Expected Fuji MakerNote encodings (verified against Shoot 10/xraw-cuban-neg.jpg
# and xraw-provia-auto-neutral.jpg): HighlightTone -2 -> 32, ShadowTone +1 -> -16,
# Color +4 -> 0xe0, NR -4 -> 0x2e0, Clarity -4 -> -4000, Strong -> 64, Large -> 32,
# Sharpness 0 -> 3, WB shift R+4/B-5 -> [80, -100].
RECIPE_STACK = {"FilmMode": 0x800, "HighlightTone": 32, "ShadowTone": -16, "Saturation": 0xE0,
                "Sharpness": 3, "ColorChromeEffect": 64, "ColorChromeFXBlue": 64,
                "NoiseReduction": 0x2E0, "Clarity": -4000, "WBFineTune": [80, -100]}
NEUTRAL = {"FilmMode": 0x800, "HighlightTone": 0, "ShadowTone": 0, "Saturation": 0, "Sharpness": 3,
           "ColorChromeEffect": 0, "ColorChromeFXBlue": 0, "NoiseReduction": 0, "Clarity": 0,
           "WBFineTune": [0, 0], "GrainEffectRoughness": 0, "GrainEffectSize": 0}
AS_SHOT = "as-shot"  # WhiteBalance must equal the RAF's own as-shot code
VARIANTS = {
    "E1-full-recipe": {**RECIPE_STACK, "DevelopmentDynamicRange": 400, "WhiteBalance": 0x0,
                       "GrainEffectRoughness": 64, "GrainEffectSize": 32},
    "E2-no-grain": {**RECIPE_STACK, "DevelopmentDynamicRange": 400, "WhiteBalance": 0x0,
                    "GrainEffectRoughness": 0, "GrainEffectSize": 0},
    "E3-as-shot-wb": {**RECIPE_STACK, "DevelopmentDynamicRange": 400, "WhiteBalance": AS_SHOT,
                      "GrainEffectRoughness": 0, "GrainEffectSize": 0},
    "E4-classic-neg-only": {**NEUTRAL, "DevelopmentDynamicRange": 100, "WhiteBalance": AS_SHOT},
    "E5-classic-neg-dr400": {**NEUTRAL, "DevelopmentDynamicRange": 400, "WhiteBalance": AS_SHOT},
}
# Repeatability exports (Codex 2026-11-18): E1 and E2 re-exported once more
# for both pilots, same settings, own folders/names; first exports preserved.
REPEAT_OF = {"R1-full-recipe-repeat": "E1-full-recipe", "R2-no-grain-repeat": "E2-no-grain"}
# Isolated controls (proposal a22d354, released as batch 1 2026-11-18): each is
# E5 (Classic Neg, DR400, As-Shot WB, everything else 0/Off) plus exactly one
# recipe setting.
_E5 = VARIANTS["E5-classic-neg-dr400"]
CONTROLS = {
    "C1-highlight-minus2": {**_E5, "HighlightTone": 32},
    "C2-shadow-plus1": {**_E5, "ShadowTone": -16},
    "C3-color-plus4": {**_E5, "Saturation": 0xE0},
    "C4-cce-strong": {**_E5, "ColorChromeEffect": 64},
    "C5-fxblue-strong": {**_E5, "ColorChromeFXBlue": 64},
    "C6-nr-minus4": {**_E5, "NoiseReduction": 0x2E0},
    "C7-clarity-minus4": {**_E5, "Clarity": -4000},
    "C8-wbshift-r4-b5": {**_E5, "WBFineTune": [80, -100]},
}
BATCH_VARIANTS = {"pilot": VARIANTS, "validation": VARIANTS,
                  "repeat": {r: VARIANTS[o] for r, o in REPEAT_OF.items()},
                  "controls": CONTROLS, "cohort1": VARIANTS}
INSTALL_NAME = {v: f"xraw-cuban-{v}.tiff" for v in [*VARIANTS, *REPEAT_OF, *CONTROLS]}
COHORT = CAL / "cuban-cohort-proposal.json"
COHORT_COMMITTED = Path(__file__).resolve().parent / "diagnostics" / "cuban-cohort-proposal.json"

TAGS = {0x1001: "Sharpness", 0x1002: "WhiteBalance", 0x1003: "Saturation", 0x100a: "WBFineTune",
        0x100e: "NoiseReduction", 0x100f: "Clarity", 0x1040: "ShadowTone", 0x1041: "HighlightTone",
        0x1047: "GrainEffectRoughness", 0x104c: "GrainEffectSize", 0x1048: "ColorChromeEffect",
        0x104e: "ColorChromeFXBlue", 0x1401: "FilmMode", 0x1403: "DevelopmentDynamicRange",
        # WB detail, recorded (not all gated): 0x1005 is the Kelvin value (present
        # only for Kelvin WB). 0x144a/b/c read as R/G/B WB levels - interpretation
        # inferred from values (G constant per file, R/B move with the WB shift).
        0x1005: "ColorTemperature", 0x144a: "WBLevelR", 0x144b: "WBLevelG", 0x144c: "WBLevelB"}
WB_DETAIL = ["WhiteBalance", "ColorTemperature", "WBFineTune", "WBLevelR", "WBLevelG", "WBLevelB"]
FMT = {3: ("H", 2), 4: ("I", 4), 8: ("h", 2), 9: ("i", 4)}
MIN_USABLE_16BIT = 10000  # unclipped samples needed for a conclusive 16-bit test


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def variants_sha256(kind):
    return hashlib.sha256(json.dumps(BATCH_VARIANTS[kind], sort_keys=True).encode()).hexdigest()


def tiff_datetime(path):
    with tifffile.TiffFile(path) as tf:
        t = tf.pages[0].tags.get("DateTime")
        return t.value if t is not None else None


def read_settings(path):
    if str(path).upper().endswith(".RAF"):  # settings live in the embedded JPEG's EXIF
        b = Path(path).read_bytes()
        off, ln = struct.unpack_from(">II", b, 84)
        src = io.BytesIO(b[off:off + ln])
    else:
        src = path
    with Image.open(src) as im:
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
    out = {"DateTimeOriginal": sub.get(0x9003), "ColorSpace": sub.get(0xA001)}
    mn = sub.get(0x927C)
    if not isinstance(mn, (bytes, bytearray)) or not mn.startswith(b"FUJIFILM"):
        return out
    ifd = struct.unpack_from("<I", mn, 8)[0]
    for i in range(struct.unpack_from("<H", mn, ifd)[0]):
        t, typ, cnt, val = struct.unpack_from("<HHII", mn, ifd + 2 + 12 * i)
        if t in TAGS and typ in FMT:
            f, size = FMT[typ]
            raw = struct.pack("<I", val)[:size * cnt] if size * cnt <= 4 else mn[val:val + size * cnt]
            vals = list(struct.unpack("<" + f * cnt, raw))
            out[TAGS[t]] = vals[0] if cnt == 1 else vals
    return out


def raf_as_shot_wb(name):
    ex = {e["name"]: e for e in json.loads((CAL / "corpus-exif.json").read_text())}
    return int(ex[name]["fuji"]["WhiteBalance"].split()[0]), ex[name]["dateTimeOriginal"]


def prepare_pilot():
    m = json.loads((CAL / "phase4-corpus-manifest.json").read_text())
    ex = {e["name"]: e for e in json.loads((CAL / "corpus-exif.json").read_text())}
    pool = sorted((s for s in m["tiers"]["train"]["scenes"]
                   if s.get("captureDynamicRange") == 400 and 500 <= s["iso"] <= 6400 and s.get("pairReady")
                   and s["name"] != "DSCF0752.RAF" and s["name"] in ex), key=lambda s: s["name"])
    wb = lambda s: int(ex[s["name"]]["fuji"]["WhiteBalance"].split()[0])
    kelvin = next(s for s in pool if wb(s) == 0xFF0 and s.get("highlightClass") == "mixed-lighting")
    auto = next(s for s in pool if wb(s) == 0x0 and s.get("highlightClass") == "bright-highlight"
                and s["sessionIndex"] != kelvin["sessionIndex"])
    rows = [{"name": s["name"], "shoot": s["shoot"], "tier": "train", "role": "pilot",
             "session": s["sessionIndex"], "iso": s["iso"], "highlightClass": s.get("highlightClass"),
             "asShotWhiteBalance": hex(wb(s))} for s in (kelvin, auto)]
    batch, manifest = BATCHES["pilot"]
    manifest.write_text(json.dumps(rows, indent=1))
    make_folders(batch, rows)
    for r in rows:
        print(r)


def prepare_repeat():
    rows = json.loads(BATCHES["pilot"][1].read_text())
    rows = [{**r, "role": "repeatability"} for r in rows]
    batch, manifest = BATCHES["repeat"]
    if manifest.exists() and json.loads(manifest.read_text()) != rows:
        sys.exit(f"refused: {manifest.name} exists with different content")
    manifest.write_text(json.dumps(rows, indent=1))
    make_folders(batch, rows, BATCH_VARIANTS["repeat"])
    for r in rows:
        print(r["name"], r["shoot"], "->", list(REPEAT_OF))


def prepare_batch1():
    """Batch 1 = isolated controls on the 8 C scenes + E1-E5 for the 6 non-pilot C scenes."""
    if COHORT.read_bytes() != COHORT_COMMITTED.read_bytes():
        sys.exit("refused: calibration-input cohort proposal differs from the committed copy")
    scenes = json.loads(COHORT.read_text())["scenes"]
    c = [r for r in scenes if r["isolatedControls"]]
    if len(c) != 8 or any(r["tier"] != "train" or r["role"] != "fit" for r in c):
        sys.exit("refused: control subset is not 8 train-tier fit scenes")
    plan = {"controls": c, "cohort1": [r for r in c if not r["pilotExportsVerified"]]}
    for kind, rows in plan.items():
        for r in rows:
            for need in (r["name"], "xraw-phase3.tiff"):
                if not (CAL / r["shoot"] / need).is_file():
                    sys.exit(f"refused: {r['shoot']}/{need} missing")
        batch, manifest = BATCHES[kind]
        rows = [{**r, "batch": kind} for r in rows]
        if manifest.exists() and json.loads(manifest.read_text()) != rows:
            sys.exit(f"refused: {manifest.name} exists with different content")
        manifest.write_text(json.dumps(rows, indent=1))
        make_folders(batch, rows, BATCH_VARIANTS[kind])
        print(f"{kind}: {len(rows)} RAFs x {len(BATCH_VARIANTS[kind])} = {len(rows) * len(BATCH_VARIANTS[kind])}",
              [r["name"] for r in rows])


def make_folders(batch, rows, variants=VARIANTS):
    batch.mkdir(exist_ok=True)
    for v in variants:
        (batch / v).mkdir(exist_ok=True)
    for r in rows:
        link = batch / r["name"]
        if not link.exists():
            os.symlink(os.path.relpath((CAL / r["shoot"] / r["name"]).resolve(), batch), link)
        assert link.resolve().exists(), link


def check_one(tif, row, variant, ref_tiff, spec=None):
    spec = spec or VARIANTS[variant]
    problems, info = [], {}
    s = read_settings(tif)
    as_shot, dto = raf_as_shot_wb(row["name"])
    if s.get("DateTimeOriginal") != dto:
        problems.append(f"DateTimeOriginal {s.get('DateTimeOriginal')} != RAF {dto} (wrong source photo?)")
    for k, want in spec.items():
        want = as_shot if want == AS_SHOT else want
        if s.get(k) != want:
            problems.append(f"{k}={s.get(k)} expected {want}")
    # WB mode tag alone does not establish the same Kelvin/gains: keep the detail
    # from both the export and the RAF. Kelvin is gated for as-shot Kelvin
    # captures; gains are report-only (they legitimately change with the WB
    # shift baked into each variant). Pixel comparison (E2 vs E3) stays planned.
    raf = read_settings(CAL / row["shoot"] / row["name"])
    info["wb"] = {"export": {k: s.get(k) for k in WB_DETAIL}, "raf": {k: raf.get(k) for k in WB_DETAIL}}
    if raf.get("WhiteBalance") != as_shot:
        problems.append(f"RAF MakerNote WhiteBalance {raf.get('WhiteBalance')} != corpus-exif {as_shot}")
    if spec["WhiteBalance"] == AS_SHOT and as_shot == 0xFF0 \
            and s.get("ColorTemperature") != raf.get("ColorTemperature"):
        problems.append(f"as-shot Kelvin {s.get('ColorTemperature')} != RAF {raf.get('ColorTemperature')}")
    if spec["WhiteBalance"] == AS_SHOT and as_shot == 0x0:
        info["note"] = ("as-shot WB is Auto: E3's WB mode tag equals E2's; only the recorded "
                        "gains and the planned pixel comparison can show a difference")
    with tifffile.TiffFile(tif) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        orient = int(page.tags["Orientation"].value) if "Orientation" in page.tags else 1
        icc = page.tags["InterColorProfile"].value if "InterColorProfile" in page.tags else None
    if arr.dtype != np.uint16 or arr.ndim != 3 or arr.shape[2] != 3:
        problems.append(f"not 16-bit RGB: dtype {arr.dtype}, shape {arr.shape}")
    else:
        # An 8-bit image stretched to 16-bit is all multiples of 257. 0 and 65535
        # are multiples of 257 in genuine 16-bit data too (black / clipped), so
        # they are excluded; too few remaining samples -> inconclusive (blocks install).
        sub = arr[::7, ::7].ravel()
        usable = sub[(sub > 0) & (sub < 65535)]
        info["sampled"] = int(sub.size)
        info["fraction_black"] = round(float(np.mean(sub == 0)), 4)
        info["fraction_clipped"] = round(float(np.mean(sub == 65535)), 4)
        info["usable_unclipped_samples"] = int(usable.size)
        if usable.size < MIN_USABLE_16BIT:
            info["bitdepth_check"] = "inconclusive"
            problems.append(f"16-bit check inconclusive: {usable.size} unclipped samples < {MIN_USABLE_16BIT}")
        else:
            upscaled8 = float(np.mean(usable % 257 == 0))
            info["fraction_multiple_of_257"] = round(upscaled8, 4)
            info["distinct_levels_unclipped"] = int(np.unique(usable).size)
            ok16 = upscaled8 <= 0.05 and info["distinct_levels_unclipped"] > 256
            info["bitdepth_check"] = "pass" if ok16 else "fail"
            if not ok16:
                problems.append(f"not genuine 16-bit data (unclipped multiples of 257: {upscaled8:.2%}, "
                                f"{info['distinct_levels_unclipped']} levels)")
    with tifffile.TiffFile(ref_tiff) as rf:
        rp = rf.pages[0]
        ref_orient = int(rp.tags["Orientation"].value) if "Orientation" in rp.tags else 1
        ref_icc = rp.tags["InterColorProfile"].value if "InterColorProfile" in rp.tags else None
        ref_shape = rp.shape
    info.update({"orientation": orient, "shape": list(arr.shape), "colorspace_exif": s.get("ColorSpace"),
                 "icc_matches_provia_corpus_tiff": icc == ref_icc and icc is not None})
    if orient != ref_orient or tuple(arr.shape) != tuple(ref_shape):
        problems.append(f"orientation/shape {orient}/{arr.shape} != corpus Provia TIFF {ref_orient}/{ref_shape}")
    if s.get("ColorSpace") != 1 or not info["icc_matches_provia_corpus_tiff"]:
        problems.append(f"colour space: EXIF {s.get('ColorSpace')} (want 1 = sRGB), ICC same as corpus: "
                        f"{info['icc_matches_provia_corpus_tiff']}")
    return problems, {**info, "settings": s}


def src_path(batch, name, variant):
    return batch / variant / (Path(name).stem + ".tif")


def verify(kind):
    batch, manifest = BATCHES[kind]
    manifest_hash = sha256(manifest)
    rows = json.loads(manifest.read_text())
    report, ok = [], True
    for r in rows:
        ref = CAL / r["shoot"] / "xraw-phase3.tiff"
        for v, spec in BATCH_VARIANTS[kind].items():
            tif = src_path(batch, r["name"], v)
            if not tif.exists():
                entry = {"name": r["name"], "variant": v, "problems": ["missing"]}
            else:
                h_before = sha256(tif)
                problems, info = check_one(tif, r, v, ref, spec)
                if v in REPEAT_OF:  # must be a fresh export of the same settings, not a copy
                    orig = CAL / r["shoot"] / INSTALL_NAME[REPEAT_OF[v]]
                    if not orig.is_file():
                        problems.append(f"original {orig.relative_to(CAL)} missing")
                    else:
                        info["original"] = str(orig.relative_to(CAL))
                        info["original_sha256"] = sha256(orig)
                        info["export_datetime"] = tiff_datetime(tif)
                        info["original_export_datetime"] = tiff_datetime(orig)
                        if info["export_datetime"] in (None, info["original_export_datetime"]):
                            problems.append("TIFF DateTime missing or equal to the original export's "
                                            "(a copy, not a fresh export?)")
                if sha256(tif) != h_before:
                    problems.append("TIFF changed while being verified")
                entry = {"name": r["name"], "variant": v, "source": str(tif.relative_to(CAL)),
                         "source_sha256": h_before, "problems": problems, **info}
                if "original_sha256" in info:
                    entry["byte_identical_to_original"] = h_before == info["original_sha256"]
            ok &= not entry["problems"]
            report.append(entry)
            print(f"{'OK  ' if not entry['problems'] else 'FAIL'} {r['name']} {v}: "
                  f"{entry['problems'] or ''} {entry.get('note', '')}")
    REPORT[kind].write_text(json.dumps({
        "passed": ok, "manifest": str(manifest.relative_to(CAL)), "manifest_sha256": manifest_hash,
        "variants": list(BATCH_VARIANTS[kind]), "variants_sha256": variants_sha256(kind), "entries": report},
        indent=1, default=str))
    print("ALL EXPORTS VERIFIED" if ok else "VERIFICATION FAILED - do not install or export the larger batch")
    return 0 if ok else 1


def install_preflight(kind):
    """Every check runs before anything is copied; returns (plan, problems)."""
    batch, manifest = BATCHES[kind]
    try:
        rep = json.loads(REPORT[kind].read_text())
        entries = rep["entries"]
    except Exception as e:
        return [], [f"verification report unusable: {type(e).__name__}: {e}"]
    probs = []
    if rep.get("passed") is not True:
        probs.append("verification report did not pass")
    if rep.get("manifest_sha256") != sha256(manifest):
        probs.append("manifest changed since verify - rerun verify")
    if rep.get("variants") != list(BATCH_VARIANTS[kind]) or rep.get("variants_sha256") != variants_sha256(kind):
        probs.append("variant definitions changed since verify - rerun verify")
    rows = json.loads(manifest.read_text())
    expected = [(r["name"], v) for r in rows for v in BATCH_VARIANTS[kind]]
    got = [(e.get("name"), e.get("variant")) for e in entries]
    if len(got) != len(set(got)):
        probs.append("duplicate scene/variant entries in report")
    if set(got) != set(expected):
        probs.append(f"report coverage mismatch: missing {sorted(set(expected) - set(got))}, "
                     f"unexpected {sorted(set(got) - set(expected))}")
    by_key = {(e.get("name"), e.get("variant")): e for e in entries}
    shoot = {r["name"]: r["shoot"] for r in rows}
    plan, dsts = [], set()
    for name, v in expected:
        e = by_key.get((name, v))
        if e is None:
            continue
        src, dst = src_path(batch, name, v), CAL / shoot[name] / INSTALL_NAME[v]
        if e.get("problems"):
            probs.append(f"{name} {v}: report lists problems {e['problems']}")
        if e.get("source") != str(src.relative_to(CAL)):
            probs.append(f"{name} {v}: report source {e.get('source')} != {src.relative_to(CAL)}")
        if not src.is_file():
            probs.append(f"{name} {v}: source TIFF missing")
        elif sha256(src) != e.get("source_sha256"):
            probs.append(f"{name} {v}: source TIFF changed since verify - rerun verify")
        if not dst.parent.is_dir():
            probs.append(f"{name} {v}: destination folder {dst.parent.relative_to(CAL)} missing")
        if dst.exists() or Path(str(dst) + ".partial").exists():
            probs.append(f"{name} {v}: {dst.relative_to(CAL)} exists - never overwriting")
        if dst in dsts:
            probs.append(f"{name} {v}: destination {dst.relative_to(CAL)} used twice")
        dsts.add(dst)
        plan.append((src, dst, e.get("source_sha256")))
    return plan, probs


def install(kind):
    plan, probs = install_preflight(kind)
    if probs:
        for p in probs:
            print("PREFLIGHT:", p)
        print("INSTALL REFUSED - nothing copied")
        return 1
    for src, dst, h in plan:
        tmp = Path(str(dst) + ".partial")
        shutil.copy2(src, tmp)
        if sha256(tmp) != h:
            tmp.unlink()
            sys.exit(f"copy of {src} did not match its verified hash - stopped, {dst} not written")
        if dst.exists():
            tmp.unlink()
            sys.exit(f"refused: {dst} appeared during install - never overwriting")
        os.rename(tmp, dst)
        print("installed", dst.relative_to(CAL))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1:] or [""]
    if cmd[0] == "prepare-pilot":
        prepare_pilot()
    elif cmd[0] == "prepare-repeat":
        prepare_repeat()
    elif cmd[0] == "prepare-batch1":
        prepare_batch1()
    elif cmd[0] in ("verify", "install") and len(cmd) > 1 and cmd[1] in BATCHES:
        sys.exit((verify if cmd[0] == "verify" else install)(cmd[1]))
    else:
        sys.exit(__doc__)
