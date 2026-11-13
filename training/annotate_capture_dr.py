"""Annotate the corpus manifest (train + monitor tiers only) with each
scene's CAPTURE dynamic-range mode, read from the RAF MakerNote dump
(calibration-input/corpus-exif.json, produced by
scripts/dump-raf-exif.mjs), kept strictly separate from the TIFF's
OUTPUT dynamic range.

Resolution rule, per instruction (resolve DR-Auto from the recorded
effective value; flag missing/ambiguous metadata rather than guess):
  - manual DR capture:   0x1403 DevelopmentDynamicRange present, 0x140b absent
                         -> captureDynamicRange = that value, source "DevelopmentDynamicRange"
  - DR-Auto capture:     0x140b AutoDynamicRange present, 0x1403 absent
                         -> captureDynamicRange = that value, source "AutoDynamicRange"
  - both present:        used only if equal; otherwise flagged "ambiguous"
  - neither / unparsable: flagged "missing"
A flagged scene gets captureDynamicRange = null and captureDynamicRangeFlag
set; dataset.py returns NaN for it and the conditioned training path
raises rather than guessing.

tiffOutputDynamicRange is recorded as 100 for every train/monitor scene on
the strength of Codex's independent verification (2026-11-13: all 599
train + 76 monitor TIFFs carry DevelopmentDynamicRange=100 in their own
EXIF); the source string says so, so it is not mistaken for a value this
script measured.

finalHeldOut is not touched: its scenes are neither read nor annotated.
"""
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CAL_DIR = REPO_ROOT / "calibration-input"
MANIFEST_PATH = CAL_DIR / "phase4-corpus-manifest.json"
EXIF_PATH = CAL_DIR / "corpus-exif.json"

TIFF_OUTPUT_DR = 100
TIFF_OUTPUT_DR_SOURCE = "Codex verification 2026-11-13: TIFF EXIF DevelopmentDynamicRange=100 on all 599 train + 76 monitor TIFFs (not measured by this script)"


def tag_int(fuji, key):
    v = fuji.get(key)
    if v is None:
        return None
    m = re.match(r"(-?\d+)", str(v))
    return int(m.group(1)) if m else None


def resolve(fuji):
    dev = tag_int(fuji, "DevelopmentDynamicRange")
    auto = tag_int(fuji, "AutoDynamicRange")
    if dev is not None and auto is None:
        return dev, "DevelopmentDynamicRange", None
    if auto is not None and dev is None:
        return auto, "AutoDynamicRange", None
    if dev is not None and auto is not None:
        if dev == auto:
            return dev, "DevelopmentDynamicRange=AutoDynamicRange", None
        return None, None, f"ambiguous: DevelopmentDynamicRange={dev} AutoDynamicRange={auto}"
    return None, None, "missing: neither DevelopmentDynamicRange nor AutoDynamicRange"


def main():
    exif = {r["name"]: r for r in json.loads(EXIF_PATH.read_text())}
    manifest = json.loads(MANIFEST_PATH.read_text())

    counts = Counter()
    flagged = []
    for tier in ("train", "monitor"):
        for s in manifest["tiers"][tier]["scenes"]:
            r = exif.get(s["name"])
            if r is None or r.get("error"):
                dr, src, flag = None, None, f"missing: no EXIF dump entry ({r.get('error') if r else 'absent'})"
            else:
                dr, src, flag = resolve(r.get("fuji", {}))
            s["captureDynamicRange"] = dr
            s["captureDynamicRangeSource"] = src
            if flag:
                s["captureDynamicRangeFlag"] = flag
                flagged.append((tier, s["name"], flag))
            else:
                s.pop("captureDynamicRangeFlag", None)
            s["tiffOutputDynamicRange"] = TIFF_OUTPUT_DR
            s["tiffOutputDynamicRangeSource"] = TIFF_OUTPUT_DR_SOURCE
            counts[(tier, dr, src)] += 1

    manifest["note"] = (manifest.get("note", "") +
        " 2026-11-13: captureDynamicRange/captureDynamicRangeSource (RAF MakerNote, DR-Auto resolved from the recorded effective value, "
        "missing/ambiguous flagged not guessed) and tiffOutputDynamicRange (=100, Codex-verified) added to train+monitor scenes; "
        "finalHeldOut untouched.")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print("capture DR annotation (tier, captureDynamicRange, source): count")
    for k in sorted(counts, key=lambda k: (k[0], k[1] or 0, str(k[2]))):
        print(f"  {k}: {counts[k]}")
    print(f"flagged (not guessed): {len(flagged)}")
    for f in flagged:
        print(f"  {f}")
    print("finalHeldOut was not read or modified.")


if __name__ == "__main__":
    main()
