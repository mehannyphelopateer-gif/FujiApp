"""Provia colour-loss pilot (Codex scope 5053165) - DATA PREPARATION ONLY.

For each of the 583 plan scenes (allowlist = provia-color-pilot-plan.json;
K/T8 and every other tier refused): audit the existing Provia TIFF against the
locked neutral-Provia protocol, hash every file, derive the training target
FRESHLY from the current TIFF with the locked loader at the exact .fjlrg
dimensions, compare it to the legacy phase4-target-cache array (provenance
evidence only - never reused unverified), and write a separate hash-bound
cache atomically. No model, no fitting, no monitor/finalHeldOut/dr100Eval.

Protocol (as Codex's F24 audit): FilmMode Provia (0), output DR100, WB mode =
the RAF's as-shot mode (Kelvin value equal for Kelvin captures), WBFineTune
[0,0], HighlightTone/ShadowTone/Saturation/CCE/FXBlue/NR/Clarity/grain 0,
Sharpness 3 (=0); genuine 16-bit RGB (unclipped-sample test), EXIF ColorSpace
1 and the same ICC as a verified sRGB E4 export; orientation 1/3/6/8 and
oriented dimensions matching the .fjlrg aspect; DateTimeOriginal = the RAF's.

  python3 training/provia_color_prep.py      # resumable; writes the audit report at the end
"""
import hashlib
import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuban_exports import FMT, MIN_USABLE_16BIT, NEUTRAL, TAGS, sha256  # noqa: E402
from dataset import _read_with_retry  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from target_tiff import load_xraw_target_linear_matching  # noqa: E402

TRAIN = Path(__file__).resolve().parent
ROOT = TRAIN.parent
CAL = ROOT / "calibration-input"
DIAG = TRAIN / "diagnostics"
PLAN = DIAG / "provia-color-pilot-plan.json"
PLAN_SHA = "8dac1a02baa884bbc0aa1040e05f87c03fd746bd3fe3f581c8b9a6afc0355df1"
CACHE = CAL / "provia-color-target-cache"
LEGACY = CAL / "phase4-target-cache"
AUDIT = DIAG / "provia-color-data-audit.json"
ICC_REF_TIFF = CAL / "Shoot 73" / "xraw-cuban-E4-classic-neg-only.tiff"  # verified sRGB export (base-step1)
ICC_REF_SHA = "704d43184721394004f73bb07832b3d23be7416a20bec3be1a54719171490042"
SPEC = {**NEUTRAL, "FilmMode": 0, "DevelopmentDynamicRange": 100}


def fail(msg):
    print(f"INTEGRITY FAILURE: {msg}", flush=True)
    sys.exit(2)


def makernote(exif_src):
    with Image.open(exif_src) as im:
        sub = im.getexif().get_ifd(0x8769)
    out = {"DateTimeOriginal": sub.get(0x9003), "ColorSpace": sub.get(0xA001)}
    mn = sub.get(0x927C)
    if not isinstance(mn, (bytes, bytearray)) or not mn.startswith(b"FUJIFILM"):
        out["_makernote"] = False
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


def raf_preview(raf):
    with open(raf, "rb") as f:
        head = f.read(92)
        if head[:16] != b"FUJIFILMCCD-RAW ":
            raise ValueError(f"{raf}: not a RAF")
        off, ln = struct.unpack_from(">II", head, 84)
        f.seek(off)
        prev = f.read(ln)
    return prev


def icc_of(path):
    with tifffile.TiffFile(path) as tf:
        p = tf.pages[0]
        return p.tags["InterColorProfile"].value if "InterColorProfile" in p.tags else None


def audit_scene(row, icc_ref):
    shoot = CAL / row["shoot"]
    tif, fj, raf = shoot / "xraw-phase3.tiff", shoot / "browser-phase3-linear.fjlrg", shoot / row["name"]
    problems, info = [], {}
    for p in (tif, fj, raf):
        if not p.exists():
            problems.append(f"missing {p.relative_to(CAL)}")
    if problems:
        return problems, info, None
    prev = _read_with_retry(raf_preview, raf)
    src = makernote(io.BytesIO(prev))
    info["raf_preview_sha256"] = hashlib.sha256(prev).hexdigest()
    info["fjlrg_sha256"] = _read_with_retry(sha256, fj)
    info["tiff_sha256"] = _read_with_retry(sha256, tif)
    s = makernote(tif)
    info["settings"] = {k: s.get(k) for k in (*SPEC, "WhiteBalance", "ColorTemperature", "DateTimeOriginal", "ColorSpace")}
    info["raf_wb"] = {k: src.get(k) for k in ("WhiteBalance", "ColorTemperature", "WBFineTune")}
    for k, want in SPEC.items():
        if s.get(k) != want:
            problems.append(f"{k}={s.get(k)} expected {want}")
    if src.get("WhiteBalance") is None:
        problems.append("RAF as-shot WhiteBalance missing")
    elif s.get("WhiteBalance") != src["WhiteBalance"]:
        problems.append(f"WhiteBalance {s.get('WhiteBalance')} != RAF as-shot {src['WhiteBalance']}")
    if src.get("WhiteBalance") == 0xFF0 and s.get("ColorTemperature") != src.get("ColorTemperature"):
        problems.append(f"Kelvin {s.get('ColorTemperature')} != RAF {src.get('ColorTemperature')}")
    if s.get("DateTimeOriginal") is None or s.get("DateTimeOriginal") != src.get("DateTimeOriginal"):
        problems.append(f"DateTimeOriginal {s.get('DateTimeOriginal')} != RAF {src.get('DateTimeOriginal')}")
    if s.get("ColorSpace") != 1:
        problems.append(f"EXIF ColorSpace {s.get('ColorSpace')} != 1")
    with tifffile.TiffFile(tif) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        orient = int(page.tags["Orientation"].value) if "Orientation" in page.tags else 1
        icc = page.tags["InterColorProfile"].value if "InterColorProfile" in page.tags else None
    info["icc_matches_verified_srgb_export"] = icc is not None and icc == icc_ref
    if not info["icc_matches_verified_srgb_export"]:
        problems.append("ICC profile differs from the verified sRGB export")
    info["orientation"], info["shape"] = orient, list(arr.shape)
    if orient not in (1, 3, 6, 8):
        problems.append(f"orientation {orient}")
    if arr.dtype != np.uint16 or arr.ndim != 3 or arr.shape[2] != 3:
        problems.append(f"not 16-bit RGB: {arr.dtype} {arr.shape}")
    else:
        sub = arr[::7, ::7].ravel()
        usable = sub[(sub > 0) & (sub < 65535)]
        info["usable_unclipped_samples"] = int(usable.size)
        info["fraction_clipped"] = round(float(np.mean(sub == 65535)), 4)
        if usable.size < MIN_USABLE_16BIT:
            problems.append(f"16-bit check inconclusive ({usable.size} samples)")
        else:
            m257 = float(np.mean(usable % 257 == 0))
            levels = int(np.unique(usable).size)
            info["fraction_multiple_of_257"], info["distinct_levels_unclipped"] = round(m257, 4), levels
            if m257 > 0.05 or levels <= 256:
                problems.append(f"not genuine 16-bit ({m257:.2%} multiples of 257, {levels} levels)")
    del arr
    x, w, h = _read_with_retry(load_fjlrg_linear, fj)
    oh, ow = (info["shape"][1], info["shape"][0]) if orient in (6, 8) else (info["shape"][0], info["shape"][1])
    if abs(ow / oh - w / h) > 0.01:
        problems.append(f"oriented TIFF aspect {ow}x{oh} vs .fjlrg {w}x{h}")
    info["fjlrg_wh"] = [w, h]
    return problems, info, (w, h)


def main():
    t_start = time.time()
    plan_bytes = PLAN.read_bytes()
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    if plan_sha != PLAN_SHA:
        fail(f"plan hash {plan_sha} does not match Codex's recorded 8dac1a02...")
    plan = json.loads(plan_bytes)
    rows = plan["scenes"]
    excl = set(plan["excluded_names"])
    names = [r["name"] for r in rows]
    if len(rows) != 583 or len(set(names)) != 583 or excl & set(names):
        fail("plan allowlist malformed or overlaps K/T8")
    if len({r["shoot"] for r in rows}) != 583:
        fail("duplicate shoot assignment in plan")
    if sha256(ICC_REF_TIFF) != ICC_REF_SHA:
        fail("ICC reference export hash changed")
    icc_ref = icc_of(ICC_REF_TIFF)
    target_loader_sha = sha256(TRAIN / "target_tiff.py")
    CACHE.mkdir(exist_ok=True)
    out, problems_all = [], {}
    for i, r in enumerate(rows):
        t0 = time.time()
        npy, side = CACHE / f"{r['shoot']}.npy", CACHE / f"{r['shoot']}.json"
        if side.exists() and npy.exists():
            meta = json.loads(side.read_text())
            if meta.get("npy_sha256") == sha256(npy) and meta.get("tiff_sha256") == sha256(CAL / r["shoot"] / "xraw-phase3.tiff"):
                out.append(meta)
                if meta["problems"]:
                    problems_all[r["name"]] = meta["problems"]
                continue
            fail(f"{r['shoot']}: existing cache sidecar does not validate - refusing to overwrite silently")
        problems, info, wh = audit_scene(r, icc_ref)
        meta = {"name": r["name"], "shoot": r["shoot"], "role": r["role"], "session": r["session"],
                "captureDynamicRange": r["captureDynamicRange"], "problems": problems, **info}
        if wh is not None and not problems:
            y = _read_with_retry(load_xraw_target_linear_matching, CAL / r["shoot"] / "xraw-phase3.tiff", *wh).astype(np.float32)
            if not np.isfinite(y).all() or y.shape != (wh[1], wh[0], 3):
                problems.append(f"derived target non-finite or wrong shape {y.shape}")
            else:
                leg = LEGACY / f"{r['shoot']}.npy"
                if leg.exists():
                    L = np.load(leg)
                    meta["legacy_cache"] = {"exists": True, "shape_equal": list(L.shape) == list(y.shape),
                                            "bit_identical": bool(L.shape == y.shape and np.array_equal(L, y)),
                                            "max_abs_diff": float(np.abs(L - y).max()) if L.shape == y.shape else None,
                                            "sha256": sha256(leg)}
                else:
                    meta["legacy_cache"] = {"exists": False}
                tmp = npy.with_suffix(".npy.partial")
                with open(tmp, "wb") as f:
                    np.save(f, y)
                tmp.rename(npy)
                meta.update({"npy_sha256": sha256(npy), "target_loader_sha256": target_loader_sha,
                             "array_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()})
        meta["seconds"] = round(time.time() - t0, 1)
        if problems:
            problems_all[r["name"]] = problems
            meta["npy_sha256"] = sha256(npy) if npy.exists() else None
        tmp = side.with_suffix(".json.partial")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.rename(side)
        out.append(meta)
        if i % 25 == 0:
            print(f"{i + 1}/583 {r['name']} {meta['seconds']}s problems={len(problems_all)}", flush=True)
    by = {}
    for m in out:
        for key in ("fjlrg_sha256", "tiff_sha256"):
            if m.get(key):
                by.setdefault((key, m[key]), []).append(m["name"])
    dups = {f"{k[0]}:{k[1][:12]}": v for k, v in by.items() if len(v) > 1}
    leg = [m.get("legacy_cache", {}) for m in out]
    report = {"status": "PASS" if not problems_all and not dups else "FAIL - stop, do not train",
              "plan_sha256": plan_sha, "scenes": len(out), "protocol": SPEC | {"WhiteBalance": "RAF as-shot",
                                                                                  "ColorTemperature": "RAF (Kelvin only)"},
              "icc_reference": {"file": str(ICC_REF_TIFF.relative_to(CAL)), "sha256": ICC_REF_SHA},
              "target_loader_sha256": target_loader_sha, "problems": problems_all, "duplicate_hashes": dups,
              "legacy_cache_summary": {"present": sum(l.get("exists", False) for l in leg),
                                       "bit_identical": sum(bool(l.get("bit_identical")) for l in leg),
                                       "max_abs_diff_over_all": max([l["max_abs_diff"] for l in leg if l.get("max_abs_diff") is not None],
                                                                    default=None)},
              "role_counts": {role: sum(m["role"] == role for m in out) for role in ("fit", "development")},
              "elapsed_seconds_this_invocation": round(time.time() - t_start, 1), "per_scene": out}
    AUDIT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_scene"}, indent=1))
    if report["status"] != "PASS":
        sys.exit(3)


if __name__ == "__main__":
    main()
