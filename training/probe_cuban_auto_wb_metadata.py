"""Cross-pipeline WB metadata test on the 8 locked C-subset Classic Cuban Neg E2 exports.

Question: does Fuji X RAW Studio's requested-Auto WB (E2: Auto, R+4/B-5) record WB levels
predictable from the SOURCE camera's 0xf00d Auto estimate, for Auto captures (controls) and
for manual captures (Kelvin / White Fluorescent)? Metadata only: no pixels, no fitting, no
CN quality metric, no runtime/app/Round 11 change, no protected tiers.

Staged CLI; each stage writes once and the next requires the previous to be committed:
  prepare-predictions            source-only predictions (5 preregistered variants) -> JSON
  prepare-qualification          bind predictions + committed hash receipts for the 8 E2 targets
  qualify  --manifest-sha S      hash targets; check E2 settings WITHOUT reading WB levels
  compare  --manifest-sha S      read 0x144a/b/c once for all 8; report exact counts per variant

Variants (x = source 0xf00d level, n = source-only Q10 factor for that channel's axis):
  H0 4*x                          field-semantics diagnostic: levels may be the PRE-fine-tune
                                  base estimate. H0 exact supports only estimate freshness;
                                  Q10 transfer then stays unvalidated.
  H1 4*((x*n + 512)//1024)        raw half-up, then x4
  H2 (4*x*n + 512)//1024          x4, then half-up
  H3 (4*x*n)//1024                x4, then floor
  H4 (4*x*n + 1023)//1024         x4, then ceil
G is 4*302 = 1208 in every variant. A variant's manual-capture results are interpretable only
if all Auto controls are RGB-exact under that same variant. No per-scene best variant, no new
variants, no adaptive exclusion. A failing shift variant is not evidence of a decoder defect.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from PIL import Image

import probe_provia_wb_shift_intervals as I

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
CAL = ROOT / "calibration-input"
Q10_REPORT = DIAG / "provia-wb-q10-report.json"
COHORT = DIAG / "cuban-cohort-proposal.json"
AUDIT = DIAG / "provia-color-data-audit.json"
CUBAN_CODE = ROOT / "training/cuban_exports.py"
SCALE_EVIDENCE = DIAG / "cn-step1-wb-reference-audit.json"
COHORT1_RECEIPT = DIAG / "cuban-cohort1-verification.json"
PILOT_RECEIPT = DIAG / "cuban-pilot-e2-hash-receipts.json"
PREDICTIONS = DIAG / "cuban-auto-wb-predictions.json"
MANIFEST = DIAG / "cuban-auto-wb-metadata-manifest.json"
QUALIFICATION = DIAG / "cuban-auto-wb-qualification.json"
REPORT = DIAG / "cuban-auto-wb-metadata-report.json"
OWN_CODE = ["training/probe_cuban_auto_wb_metadata.py", "training/test_cuban_auto_wb_metadata.py"]
TARGET_NAME = "xraw-cuban-E2-no-grain.tiff"
SCALE, SOURCE_G = 4, 302
VARIANTS = ("H0", "H1", "H2", "H3", "H4")
# Expected E2 settings (copied from training/cuban_exports.py VARIANTS["E2-no-grain"]; the file is hash-bound).
E2_SPEC = {"FilmMode": 0x800, "HighlightTone": 32, "ShadowTone": -16, "Saturation": 0xE0, "Sharpness": 3,
           "ColorChromeEffect": 64, "ColorChromeFXBlue": 64, "NoiseReduction": 0x2E0, "Clarity": -4000,
           "WBFineTune": [80, -100], "DevelopmentDynamicRange": 400, "WhiteBalance": 0,
           "GrainEffectRoughness": 0, "GrainEffectSize": 0}
SETTING_TAGS = {0x1001: "Sharpness", 0x1002: "WhiteBalance", 0x1003: "Saturation", 0x100a: "WBFineTune",
                0x100e: "NoiseReduction", 0x100f: "Clarity", 0x1040: "ShadowTone", 0x1041: "HighlightTone",
                0x1047: "GrainEffectRoughness", 0x104c: "GrainEffectSize", 0x1048: "ColorChromeEffect",
                0x104e: "ColorChromeFXBlue", 0x1401: "FilmMode", 0x1403: "DevelopmentDynamicRange", 0x1005: "ColorTemperature"}
WB_LEVEL_TAGS = {0x144a: "R", 0x144b: "G", 0x144c: "B"}
FMT = {3: ("H", 2), 4: ("I", 4), 8: ("h", 2), 9: ("i", 4)}
MAX_COUNT = 8

require, sha, dump, committed = I.require, I.sha, I.dump, I.committed


# --- Pure helpers ---------------------------------------------------------------

def predict(x, n, variant):
    """One channel's predicted 4x-scale level."""
    I.check_level(x, "x")
    if variant == "H0":
        return SCALE * x
    require(type(n) is int and n > 0, "Factor n must be a positive int")
    return {"H1": SCALE * ((x * n + 512) // 1024), "H2": (SCALE * x * n + 512) // 1024,
            "H3": (SCALE * x * n) // 1024, "H4": (SCALE * x * n + 1023) // 1024}[variant]


def predict_rgb(auto_rgbg, n_r, n_b):
    require(auto_rgbg[1] == auto_rgbg[3] == SOURCE_G, "Source G must be 302")
    return {v: [predict(auto_rgbg[0], n_r, v), SCALE * SOURCE_G, predict(auto_rgbg[2], n_b, v)] for v in VARIANTS}


def parse_makernote(mn, want_wb_levels=False):
    """Strict bounded Fuji MakerNote parse. Settings tags only; WB-level tags are only
    recorded as present unless want_wb_levels (compare stage). Unknown tags are never unpacked."""
    require(isinstance(mn, (bytes, bytearray)) and mn[:8] == b"FUJIFILM" and len(mn) >= 14, "Not a Fuji MakerNote")
    ifd = struct.unpack_from("<I", mn, 8)[0]
    require(12 <= ifd and ifd + 2 <= len(mn), "MakerNote IFD out of bounds")
    count = struct.unpack_from("<H", mn, ifd)[0]
    require(count <= 512 and ifd + 2 + 12 * count <= len(mn), "MakerNote entry table out of bounds")
    out, present_wb = {}, []
    for i in range(count):
        offset = ifd + 2 + 12 * i
        tag = struct.unpack_from("<H", mn, offset)[0]
        if tag in WB_LEVEL_TAGS and not want_wb_levels:
            present_wb.append(hex(tag))      # whitelisted skip: never unpacked here
            continue
        name = SETTING_TAGS.get(tag) or (WB_LEVEL_TAGS.get(tag) if want_wb_levels else None)
        if name is None:
            continue
        _, typ, cnt, val = struct.unpack_from("<HHII", mn, offset)
        require(typ in FMT and 1 <= cnt <= MAX_COUNT, f"Unsupported type/count for tag {hex(tag)}")
        f, size = FMT[typ]
        nbytes = size * cnt
        if nbytes <= 4:
            raw = struct.pack("<I", val)[:nbytes]
        else:
            require(val + nbytes <= len(mn), f"Value of tag {hex(tag)} out of bounds")
            raw = mn[val:val + nbytes]
        vals = list(struct.unpack("<" + f * cnt, raw))
        value = vals[0] if cnt == 1 else vals
        require(name not in out or out[name] == value, "Conflicting metadata tag " + hex(tag))
        out[name] = value
    return out, present_wb


def read_exif_metadata(path):
    """Metadata only: Pillow header/EXIF access, never Image.load or pixel arrays."""
    with Image.open(path) as im:
        sub = im.getexif().get_ifd(0x8769)
        bits = getattr(im, "tag_v2", {}).get(258) if hasattr(im, "tag_v2") else None
    return {"DateTimeOriginal": sub.get(0x9003), "ColorSpace": sub.get(0xA001),
            "BitsPerSample": list(bits) if isinstance(bits, (tuple, list)) else bits, "_makernote": sub.get(0x927C)}


def check_settings(settings, meta, expected_dto):
    problems = [f"{k}={settings.get(k)} expected {v}" for k, v in E2_SPEC.items() if settings.get(k) != v]
    if meta["DateTimeOriginal"] != expected_dto:
        problems.append(f"DateTimeOriginal {meta['DateTimeOriginal']} != source audit {expected_dto}")
    if meta["ColorSpace"] != 1:
        problems.append(f"ColorSpace {meta['ColorSpace']} != 1")
    if meta["BitsPerSample"] is not None and meta["BitsPerSample"] not in ([16, 16, 16], 16):
        problems.append(f"BitsPerSample {meta['BitsPerSample']} != 16")
    return problems


def compare_rows(rows):
    """rows: [{name, capture_mode, predictions{variant:[R,G,B]}, observed:[R,G,B]}] -> report."""
    if any(r["observed"][1] != SCALE * SOURCE_G for r in rows):
        return {"verdict": "INCONCLUSIVE_SCALE", "note": "Observed G != 1208 for at least one target; no variant interpreted."}
    auto = [r for r in rows if r["capture_mode"] == 0]
    manual = [r for r in rows if r["capture_mode"] != 0]
    per_variant = {}
    for v in VARIANTS:
        def stats(subset):
            errs = [[r["observed"][c] - r["predictions"][v][c] for c in range(3)] for r in subset]
            return {"n": len(subset), "rgb_exact": sum(all(e == 0 for e in es) for es in errs),
                    "channel_exact": [sum(es[c] == 0 for es in errs) for c in range(3)],
                    "code_errors": {r["name"]: es for r, es in zip(subset, errs)}}
        a, m = stats(auto), stats(manual)
        interpretable = len(auto) > 0 and a["rgb_exact"] == len(auto)
        meaning = None
        if interpretable:
            meaning = ("pre-fine-tune base-estimate metadata only; Q10 transfer unvalidated" if v == "H0"
                       else "supports Q10-shifted metadata under this rounding variant")
        per_variant[v] = {"auto_controls": a, "manual_captures": m, "manual_interpretable": interpretable,
                          "interpretation_if_interpretable": meaning}
    return {"verdict": "REPORTED", "per_variant": per_variant,
            "note": "No best-variant selection. Manual results are read only under variants whose Auto controls are all RGB-exact."}


def collect_receipts(cohort1, pilot, pilot_committed):
    """E2 hash receipts: {name: {installed, sha256, receipt}} from verified export records only."""
    out = {}

    def walk(o):
        if isinstance(o, dict):
            yield o
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
    require(cohort1.get("passed") is not False, "Cohort receipt failed")
    for e in walk(cohort1):
        if e.get("variant") == "E2-no-grain" and "verified_sha256" in e and "installed" in e:
            require(not e.get("problems"), "Cohort E2 receipt has problems")
            require(e["name"] not in out or out[e["name"]]["sha256"] == e["verified_sha256"], "Conflicting cohort receipt")
            out[e["name"]] = {"installed": e["installed"], "sha256": e["verified_sha256"], "receipt": "cohort1-verification"}
    if pilot is not None:
        require(pilot_committed, "Pilot export-check receipt is not committed; refusing to use it")
        require(pilot.get("passed") is True, "Pilot export check did not pass")
        for e in pilot["entries"]:
            if e.get("variant") == "E2-no-grain":
                require(e.get("problems") == [], f"Pilot E2 receipt has problems: {e['name']}")
                require(e["name"] not in out or out[e["name"]]["sha256"] == e["source_sha256"], "Conflicting pilot receipt")
                out.setdefault(e["name"], {"installed": None, "sha256": e["source_sha256"], "receipt": "pilot-export-check"})
    return out


# --- Stages ---------------------------------------------------------------------

def locked_c8():
    cohort = json.loads(COHORT.read_text())["scenes"]
    c8 = [r for r in cohort if r.get("isolatedControls")]
    require(len(c8) == 8 and all(r["tier"] == "train" and r["role"] == "fit" for r in c8), "Locked C8 subset is not 8 train/fit scenes")
    require(len({r["name"]for r in c8}) == len({r["shoot"]for r in c8}) == 8, "Duplicate C8 identity")
    return sorted(c8, key=lambda r: r["name"])


def prepare_predictions():
    import cuban_exports as C  # spec only; C.read_settings is never called (it parses WB levels)
    require(C.VARIANTS["E2-no-grain"] == E2_SPEC, "E2 spec differs from training/cuban_exports.py")
    q10 = json.loads(Q10_REPORT.read_text())
    steps = [v // I.FINE_TUNE_STEP for v in E2_SPEC["WBFineTune"]]
    keys = {"R": f"R{steps[0]}", "B": f"B{steps[1]}"}
    proposal = {a["axis"]: a for a in q10["future_transfer_proposal"]["axes"]}
    factors = {}
    for c, key in keys.items():
        require(key in proposal, f"Q10 report has no qualified source factor for {key}")
        loso = q10["loso"]["summary"]["per_axis"][key]
        require(loso["held_rows"] == loso["measurable_rows"] == loso["exact_rows"] > 0, f"{key} LOSO not fully covered and exact")
        factors[c] = proposal[key]["n"]
    require(sha(I.POOL) == I.POOL_SHA, "Frozen train allowlist changed")
    pool = {(s["name"], s["shoot"]) for s in json.loads(I.POOL.read_text())["scenes"]}
    inventory = {r["name"]: r for r in json.loads(I.INVENTORY.read_text())["rows"]}
    dto = {r["name"]: r["settings"]["DateTimeOriginal"] for r in json.loads(AUDIT.read_text())["per_scene"]}
    scenes = []
    for r in locked_c8():
        require((r["name"], r["shoot"]) in pool and r["name"] in inventory, f"{r['name']} outside the train allowlist/inventory")
        src = inventory[r["name"]]
        scenes.append({"name": r["name"], "shoot": r["shoot"], "capture_mode": src["source_wb_mode"],
                       "source_auto_rgbg": src["auto_rgbg"], "expected_dto": dto[r["name"]],
                       "predictions": predict_rgb(src["auto_rgbg"], factors["R"], factors["B"])})
    require(sum(s["capture_mode"] == 0 for s in scenes) == 3, "Expected 3 Auto controls in the locked C8")
    files = OWN_CODE + [str(p.relative_to(ROOT)) for p in (Q10_REPORT, I.INVENTORY, I.POOL, COHORT, AUDIT, CUBAN_CODE, SCALE_EVIDENCE)]
    dump(PREDICTIONS, {"scope": "Source-only predictions frozen before E2 WB values are interpreted in this probe; earlier export verification recorded target metadata, so this is a development diagnostic, not untouched acceptance.",
                       "axes": keys, "factors": factors, "scale": SCALE, "scale_evidence": "Audited Provia WB level G1208 versus source G302; reporting-unit convention only, not target-derived runtime gains", "variants": list(VARIANTS),
                       "bindings": {f: sha(ROOT / f) for f in files}, "scenes": scenes})
    print("Predictions SHA", sha(PREDICTIONS))


def prepare_qualification():
    for p in [PREDICTIONS] + [ROOT / f for f in OWN_CODE]:
        require(committed(p), f"{p} must be committed before qualification is prepared")
    preds = json.loads(PREDICTIONS.read_text())
    for f, h in preds["bindings"].items():
        require(sha(ROOT / f) == h, "Prediction input changed: " + f)
    require(committed(COHORT1_RECEIPT), "Cohort receipt must be committed")
    pilot = json.loads(PILOT_RECEIPT.read_text()) if PILOT_RECEIPT.exists() else None
    receipts = collect_receipts(json.loads(COHORT1_RECEIPT.read_text()), pilot, pilot is not None and committed(PILOT_RECEIPT))
    targets = []
    for s in preds["scenes"]:
        require(s["name"] in receipts, f"GAP: no verified E2 hash receipt for {s['name']}; stopping")
        path = f"calibration-input/{s['shoot']}/{TARGET_NAME}"
        rec = receipts[s["name"]]
        require(rec["installed"] in (None, f"{s['shoot']}/{TARGET_NAME}"), f"Receipt install path differs for {s['name']}")
        targets.append({"name": s["name"], "path": path, "receipt_sha256": rec["sha256"], "receipt": rec["receipt"]})
    files = [str(PREDICTIONS.relative_to(ROOT)), str(COHORT1_RECEIPT.relative_to(ROOT)), *OWN_CODE]
    if pilot is not None:
        files.append(str(PILOT_RECEIPT.relative_to(ROOT)))
    dump(MANIFEST, {"scope": "C8 E2 WB metadata test; metadata only; CN target metadata access for WB fields only, no pixels or scoring.",
                    "bindings": {f: sha(ROOT / f) for f in files}, "targets": targets})
    print("Metadata manifest SHA", sha(MANIFEST))


def load_manifest(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    for p in [MANIFEST, PREDICTIONS] + [ROOT / f for f in OWN_CODE]:
        require(committed(p), f"{p} must be committed")
    m = json.loads(MANIFEST.read_text())
    for f, h in m["bindings"].items():
        require(sha(ROOT / f) == h, "Bound file changed: " + f)
    preds = json.loads(PREDICTIONS.read_text())
    for f,h in preds["bindings"].items():
        require(sha(ROOT / f) == h, "Prediction source changed: " + f)
    require(len(m["targets"]) == len({t["name"] for t in m["targets"]}) == 8 and {t["name"]for t in m["targets"]} == {s["name"]for s in preds["scenes"]}, "Target/prediction coverage mismatch")
    return m, preds


def qualify(expected):
    m, preds = load_manifest(expected)
    by_name = {s["name"]: s for s in preds["scenes"]}
    rows, ok = [], True
    for t in m["targets"]:
        path = ROOT / t["path"]
        digest = sha(path)
        require(digest == t["receipt_sha256"], "Target SHA differs from receipt; metadata access refused")
        problems = []
        meta = read_exif_metadata(path)
        settings, present_wb = parse_makernote(meta["_makernote"])
        problems += check_settings(settings, meta, by_name[t["name"]]["expected_dto"])
        if sha(path) != digest:
            problems.append("Target changed while being read")
        ok &= not problems
        rows.append({"name": t["name"], "target_sha256": digest, "settings": settings, "wb_level_tags_present": present_wb,
                     "colorspace": meta["ColorSpace"], "bits_per_sample": meta["BitsPerSample"], "problems": problems})
    dump(QUALIFICATION, {"passed": ok and len(rows) == 8, "manifest_sha256": expected, "rows": rows,
                         "note": "WB levels not read in this stage."})
    print("Qualification", "PASS" if ok else "FAIL")
    require(ok and len(rows) == 8, "Qualification failed")


def compare(expected):
    m, preds = load_manifest(expected)
    q = json.loads(QUALIFICATION.read_text())
    require(q["passed"] is True and q["manifest_sha256"] == expected and len(q["rows"]) == 8, "No matching passed qualification")
    qsha = {r["name"]: r["target_sha256"] for r in q["rows"]}
    by_name = {s["name"]: s for s in preds["scenes"]}
    rows = []
    for t in m["targets"]:
        path = ROOT / t["path"]
        require(sha(path) == qsha[t["name"]] == t["receipt_sha256"], f"Target changed since qualification: {t['name']}")
        levels, _ = parse_makernote(read_exif_metadata(path)["_makernote"], want_wb_levels=True)
        require(all(type(levels.get(k)) is int and levels[k] > 0 for k in "RGB"), f"WB level tags missing: {t['name']}")
        s = by_name[t["name"]]
        rows.append({"name": t["name"], "capture_mode": s["capture_mode"], "predictions": s["predictions"],
                     "observed": [levels["R"], levels["G"], levels["B"]]})
    dump(REPORT, {"manifest_sha256": expected, "qualification_sha256": sha(QUALIFICATION), "rows": rows, "result": compare_rows(rows),
                  "limits": "Metadata only. Supports or fails 0xf00d as a candidate for REQUESTED Auto WB on manual captures; not as-shot neutralisation, pixel or render parity. No policy, decode, app or Round 11 change."})
    print(json.dumps(compare_rows(rows), indent=2)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare-predictions", "prepare-qualification", "qualify", "compare"))
    ap.add_argument("--manifest-sha")
    a = ap.parse_args()
    if a.command == "prepare-predictions":
        prepare_predictions()
    elif a.command == "prepare-qualification":
        prepare_qualification()
    else:
        require(bool(a.manifest_sha), "Frozen manifest SHA required")
        (qualify if a.command == "qualify" else compare)(a.manifest_sha)
