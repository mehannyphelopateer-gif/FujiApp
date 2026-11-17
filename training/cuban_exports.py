"""Classic Cuban Neg X RAW Studio exports - pilot selection, export-folder
prep, verification and installation. Tag/header reads plus a 16-bit
sample-depth check only; no fitting, no scoring, never touches finalHeldOut
or Round 11.

  python3 training/cuban_exports.py prepare-pilot   # pick 2 train RAFs, make cuban-pilot-batch/E1..E5
  python3 training/cuban_exports.py verify pilot    # check every exported TIFF (settings/orientation/colour/16-bit)
  python3 training/cuban_exports.py install pilot   # only after verify passes: copy into Shoot folders
  (`validation` in place of `pilot` for the 13-RAF batch, once approved)

Variants (Codex-approved 2026-11-17). Each is a separate X RAW Studio pass
into its own subfolder; installed names are distinct per stage so the
Provia corpus files (xraw-phase3.tiff etc.) are never overwritten.
"""
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
           "validation": (CAL / "cuban-validation-batch", CAL / "cuban-validation-manifest.json")}
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
INSTALL_NAME = {v: f"xraw-cuban-{v}.tiff" for v in VARIANTS}

TAGS = {0x1001: "Sharpness", 0x1002: "WhiteBalance", 0x1003: "Saturation", 0x100a: "WBFineTune",
        0x100e: "NoiseReduction", 0x100f: "Clarity", 0x1040: "ShadowTone", 0x1041: "HighlightTone",
        0x1047: "GrainEffectRoughness", 0x104c: "GrainEffectSize", 0x1048: "ColorChromeEffect",
        0x104e: "ColorChromeFXBlue", 0x1401: "FilmMode", 0x1403: "DevelopmentDynamicRange"}
FMT = {3: ("H", 2), 4: ("I", 4), 8: ("h", 2), 9: ("i", 4)}


def read_settings(path):
    with Image.open(path) as im:
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


def make_folders(batch, rows):
    batch.mkdir(exist_ok=True)
    for v in VARIANTS:
        (batch / v).mkdir(exist_ok=True)
    for r in rows:
        link = batch / r["name"]
        if not link.exists():
            os.symlink(os.path.relpath((CAL / r["shoot"] / r["name"]).resolve(), batch), link)
        assert link.resolve().exists(), link


def check_one(tif, row, variant, ref_tiff):
    problems, info = [], {}
    s = read_settings(tif)
    as_shot, dto = raf_as_shot_wb(row["name"])
    if s.get("DateTimeOriginal") != dto:
        problems.append(f"DateTimeOriginal {s.get('DateTimeOriginal')} != RAF {dto} (wrong source photo?)")
    for k, want in VARIANTS[variant].items():
        want = as_shot if want == AS_SHOT else want
        if s.get(k) != want:
            problems.append(f"{k}={s.get(k)} expected {want}")
    if VARIANTS[variant]["WhiteBalance"] == AS_SHOT and as_shot == 0x0:
        info["note"] = "as-shot WB is Auto: E3's WB tag cannot be told apart from E2's by metadata alone"
    with tifffile.TiffFile(tif) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        orient = int(page.tags["Orientation"].value) if "Orientation" in page.tags else 1
        icc = page.tags["InterColorProfile"].value if "InterColorProfile" in page.tags else None
    if arr.dtype != np.uint16 or arr.ndim != 3 or arr.shape[2] != 3:
        problems.append(f"not 16-bit RGB: dtype {arr.dtype}, shape {arr.shape}")
    else:
        sub = arr[::7, ::7].ravel()
        upscaled8 = float(np.mean(sub % 257 == 0))  # an 8-bit image stretched to 16-bit is all multiples of 257
        info["fraction_multiple_of_257"] = round(upscaled8, 4)
        info["distinct_levels_sampled"] = int(np.unique(sub).size)
        if upscaled8 > 0.05 or info["distinct_levels_sampled"] <= 256:
            problems.append(f"not genuine 16-bit data (multiples of 257: {upscaled8:.2%})")
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


def verify(kind):
    batch, manifest = BATCHES[kind]
    rows = json.loads(manifest.read_text())
    report, ok = [], True
    for r in rows:
        ref = CAL / r["shoot"] / "xraw-phase3.tiff"
        for v in VARIANTS:
            tif = batch / v / (Path(r["name"]).stem + ".tif")
            if not tif.exists():
                entry = {"name": r["name"], "variant": v, "problems": ["missing"]}
            else:
                problems, info = check_one(tif, r, v, ref)
                entry = {"name": r["name"], "variant": v, "problems": problems, **info}
            ok &= not entry["problems"]
            report.append(entry)
            print(f"{'OK  ' if not entry['problems'] else 'FAIL'} {r['name']} {v}: "
                  f"{entry['problems'] or ''} {entry.get('note', '')}")
    REPORT[kind].write_text(json.dumps({"passed": ok, "entries": report}, indent=1, default=str))
    print("ALL EXPORTS VERIFIED" if ok else "VERIFICATION FAILED - do not install or export the larger batch")
    return 0 if ok else 1


def install(kind):
    rep = json.loads(REPORT[kind].read_text()) if REPORT[kind].exists() else {}
    if rep.get("passed") is not True:
        sys.exit("refused: run verify first and it must pass")
    batch, manifest = BATCHES[kind]
    for r in json.loads(manifest.read_text()):
        for v in VARIANTS:
            src, dst = batch / v / (Path(r["name"]).stem + ".tif"), CAL / r["shoot"] / INSTALL_NAME[v]
            if dst.exists():
                sys.exit(f"refused: {dst} exists - never overwriting")
            shutil.copy2(src, dst)
            print("installed", dst.relative_to(CAL))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1:] or [""]
    if cmd[0] == "prepare-pilot":
        prepare_pilot()
    elif cmd[0] in ("verify", "install") and len(cmd) > 1 and cmd[1] in BATCHES:
        sys.exit((verify if cmd[0] == "verify" else install)(cmd[1]))
    else:
        sys.exit(__doc__)
