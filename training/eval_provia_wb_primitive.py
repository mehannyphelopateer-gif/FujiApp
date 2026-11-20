"""WB primitive gate v2: does LibRaw apply the declared per-channel gain at each photosite?

New, separately labelled gate. It does NOT relabel the original joint OLS gate,
which stays FAILED. Scope: the 11 source-Auto F24 training scenes qualified by
metadata only (provia-source-wb-probe.json), source-only coefficients
(0xf00e as-shot vs 0xf00d Auto; archived target levels are never inputs), no
targets, no models, no protected tiers.

Arms differ only in userMul. Both use six-element linear gamma, camera RGB
(outputColor 0), camera/auto WB off and noInterpolation, so imageData() should
be an unmixed native CFA mosaic. Rules frozen before any data:

* Structure (otherwise STOP; no CFA labelling is invented):
  16-bit, 3 colours, identical size in both arms; no pixel with more than one
  non-zero channel; every (row mod 6, col mod 6) phase has exactly one active
  channel across the whole frame; identical 36-phase pattern in both arms.
* Primitive, per scene and channel, on the fixed native grid (offset 15,
  stride 31; coprime to the 6x6 X-Trans period, so all 36 phases are sampled):
  sites with D and E both in [1%, 80%] of full scale; n >= 1000;
  mean |E - g*D| <= 1 code and max <= 3 codes, with
  g = (auto_c / auto_G) / (asShot_c / asShot_G) and green the minimum
  multiplier in both arms (so g_G = 1). Green exact-match fraction reported.
  The tolerance is physical uint16 truncation in scale_colors, not a fit.
* Returned pre_mul ratios are reported descriptively, never as applied-parameter
  readback.

A pass qualifies arm W research inputs only. It says nothing about demosaiced
output, rendered recipes or Fuji parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
MANIFEST = DIAG / "provia-wb-primitive-v2-manifest.json"
REPORT = DIAG / "provia-wb-primitive-v2-report.json"
OUT = ROOT / "training/checkpoints/provia-wb-primitive-v2"
SOURCE_PROBE = DIAG / "provia-source-wb-probe.json"
POOL = DIAG / "provia-color-run-manifest.json"
GAMMA_MANIFEST = DIAG / "provia-gamma-length-probe-manifest.json"

FULL_SCALE = 65535
MASK_LOW, MASK_HIGH = 0.01, 0.80
MIN_N = 1000
MEAN_LIMIT_CODES = 1.0
MAX_LIMIT_CODES = 3.0
GRID_OFFSET, GRID_STRIDE = 15, 31
CFA_PERIOD = 6


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    require(not path.exists(), f"Refusing overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


# --- Pure functions (unit-tested on synthetic data) --------------------------

def expected_gains(as_shot_rgbg, auto_rgbg):
    """Per-channel gain E/D implied by the two declared multiplier sets.

    LibRaw (highlight 0) normalises each arm's multipliers by that arm's
    smallest one; green must be the smallest in both, so the green gain is 1.
    """
    a, s = auto_rgbg[:3], as_shot_rgbg[:3]
    require(a[1] == min(a) and s[1] == min(s), "Green is not the minimum multiplier in both arms")
    return [(a[c] / a[1]) / (s[c] / s[1]) for c in range(3)]


def grid_shape(width, height, offset=GRID_OFFSET, stride=GRID_STRIDE):
    """(rows, cols) of the fixed native sample grid, matching the JS loops."""
    require(width > offset and height > offset, "Image smaller than grid offset")
    return len(range(offset, height, stride)), len(range(offset, width, stride))


def infer_pattern(phase_counts):
    """Infer the 36-phase active-channel pattern from per-phase counts.

    phase_counts: 36*4 values [R, G, B, all-zero] per phase, phase index
    (row mod 6)*6 + (col mod 6). Returns (pattern or None, problems).
    """
    counts = np.asarray(phase_counts, dtype=np.float64)
    require(counts.shape == (CFA_PERIOD * CFA_PERIOD * 4,), "Phase counts must have 144 values")
    require(np.all(np.isfinite(counts)) and np.all(counts >= 0) and np.all(counts == np.floor(counts)), "Phase counts must be finite non-negative integers")
    counts = counts.reshape(CFA_PERIOD * CFA_PERIOD, 4)
    pattern, problems = [], []
    for k, row in enumerate(counts):
        active = [c for c in range(3) if row[c] > 0]
        if len(active) != 1:
            problems.append({"phase": [k // CFA_PERIOD, k % CFA_PERIOD], "active_channels": active,
                             "counts_RGB_zero": row.tolist()})
            pattern.append(None)
        else:
            pattern.append(active[0])
    if not problems and set(pattern) != {0, 1, 2}:
        problems.append({"pattern": pattern, "reason": "Pattern does not contain all three channels"})
    return (pattern if not problems else None), problems


def check_structure(arm_d, arm_e):
    """Return (pattern, problems) for one scene's pair of decoded arms."""
    problems = []
    for name, arm in (("D", arm_d), ("E", arm_e)):
        shape = arm.get("shape", {})
        if arm.get("structure_error"):
            problems.append(f"{name}: {arm['structure_error']}")
        if shape.get("bits") != 16 or shape.get("colors") != 3:
            problems.append(f"{name}: expected 16-bit 3-colour, got {shape}")
        if arm.get("mixed_pixels") != 0:
            problems.append(f"{name}: mixed pixels {arm.get('mixed_pixels')}")
    if problems:
        return None, problems
    if (arm_d["shape"]["width"], arm_d["shape"]["height"]) != (arm_e["shape"]["width"], arm_e["shape"]["height"]):
        return None, ["Arm sizes differ"]
    pattern_d, prob_d = infer_pattern(arm_d["phase_counts"])
    pattern_e, prob_e = infer_pattern(arm_e["phase_counts"])
    problems += [{"arm": "D", **p} for p in prob_d] + [{"arm": "E", **p} for p in prob_e]
    if not problems and pattern_d != pattern_e:
        problems.append("CFA patterns differ between arms")
    return (pattern_d if not problems else None), problems


def active_values(samples, width, height, pattern):
    """Split grid samples into per-channel arrays of the active photosite value.

    samples: flat uint16 array, 3 values per grid site. Returns (values, channel)
    where values holds the active channel's value at each site.
    """
    rows, cols = grid_shape(width, height)
    s = np.asarray(samples, dtype=np.int64).reshape(rows * cols, 3)
    ys = np.repeat(np.arange(GRID_OFFSET, height, GRID_STRIDE)[:rows], cols)
    xs = np.tile(np.arange(GRID_OFFSET, width, GRID_STRIDE)[:cols], rows)
    channel = np.array([pattern[(y % CFA_PERIOD) * CFA_PERIOD + (x % CFA_PERIOD)] for y, x in zip(ys, xs)])
    values = s[np.arange(len(s)), channel]
    inactive = s.copy()
    inactive[np.arange(len(s)), channel] = 0
    require(not inactive.any(), "Non-zero value in an inactive channel at a grid site")
    return values, channel


def primitive_channel(d, e, gain):
    """Quantisation gate for one channel: |E - g*D| over the frozen 1–80% mask."""
    d = np.asarray(d, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    lo, hi = MASK_LOW * FULL_SCALE, MASK_HIGH * FULL_SCALE
    keep = (d >= lo) & (d <= hi) & (e >= lo) & (e <= hi)
    n = int(keep.sum())
    out = {"expected_gain": gain, "n": n, "measurable": n >= MIN_N}
    if n == 0:
        out.update(passed=False, reason="No sites in mask")
        return out
    err = np.abs(e[keep] - gain * d[keep])
    out.update(mean_abs_codes=float(err.mean()), p99_abs_codes=float(np.percentile(err, 99)),
               max_abs_codes=float(err.max()), exact_fraction=float(np.mean(e[keep] == d[keep] * gain)),
               signed_mean_codes=float(np.mean(e[keep] - gain * d[keep])))
    out["passed"] = bool(out["measurable"] and out["mean_abs_codes"] <= MEAN_LIMIT_CODES
                         and out["max_abs_codes"] <= MAX_LIMIT_CODES)
    if not out["measurable"]:
        out["reason"] = f"Short: n={n} < {MIN_N}"
    return out


def evaluate_scene(scene, arm_d, arm_e, samples_d, samples_e):
    """Full per-scene evaluation; structure failures return without a primitive verdict."""
    pattern, problems = check_structure(arm_d, arm_e)
    row = {"name": scene["name"], "session": scene["session"], "shift": scene["shift"], "structure_ok": not problems,
           "structure_problems": problems}
    if problems:
        row["primitive_passed"] = None
        return row
    width, height = arm_d["shape"]["width"], arm_d["shape"]["height"]
    vd, ch = active_values(samples_d, width, height, pattern)
    ve, ch_e = active_values(samples_e, width, height, pattern)
    require(np.array_equal(ch, ch_e), "Grid channel assignment differs")
    gains = scene["expected_gain_RGB"]
    channels = [primitive_channel(vd[ch == c], ve[ch == c], gains[c]) for c in range(3)]
    row.update(cfa_pattern_6x6=[pattern[r * CFA_PERIOD:(r + 1) * CFA_PERIOD] for r in range(CFA_PERIOD)],
               channels_RGB=channels, primitive_passed=all(c["passed"] for c in channels))
    row["returned_pre_mul_descriptive"] = {a: arm.get("returned_metadata", {}).get("pre_mul") for a, arm in (("D", arm_d), ("E", arm_e))}
    return row


# --- CLI ---------------------------------------------------------------------

def prepare():
    """Freeze the F11 subset, options and bindings before any media access."""
    probe = json.loads(SOURCE_PROBE.read_text())
    require(probe["crosschecks_passed"] and probe["screens"]["0xf00d"]["passed"], "Source metadata probe not qualified")
    pool = {s["name"]: s for s in json.loads(POOL.read_text())["scenes"]}
    gamma = json.loads(GAMMA_MANIFEST.read_text())
    scenes = []
    for r in probe["rows"]:
        if r["source_wb_mode"] != 0:
            continue
        require(r["name"] in pool and pool[r["name"]]["shoot"] == r["shoot"], "Scene outside approved training pool")
        records = {v["tag"]: v for v in r["records"]}
        as_shot = records["0xf00e"]["rgbg"][:3]
        auto = records["0xf00d"]["rgbg"][:3]
        as_shot_rgbg, auto_rgbg = as_shot + [as_shot[1]], auto + [auto[1]]
        scenes.append({"name": r["name"], "shoot": r["shoot"], "session": r["session"], "shift": r["shift_steps"],
                       "raf_sha256": r["raf_sha256"], "as_shot_rgbg": as_shot_rgbg, "auto_rgbg": auto_rgbg,
                       "expected_gain_RGB": expected_gains(as_shot_rgbg, auto_rgbg)})
    require(len(scenes) == 11, "Exact F11 source-Auto coverage required")
    base = {k: v for k, v in gamma["base_options"].items() if k != "userMul"}
    require(base.get("outputColor") == 0 and base.get("useCameraWb") is False, "Unexpected inherited options")
    base["gamm"] = [1, 1, 0, 0, 0, 0]
    base["noInterpolation"] = True
    files = [SOURCE_PROBE, POOL, GAMMA_MANIFEST, Path(__file__).resolve(), ROOT / "scripts/probe-provia-wb-primitive.mjs"]
    files += [ROOT / "node_modules/libraw-wasm/dist" / f for f in ("index.js", "worker.js", "libraw.js", "libraw.wasm")]
    files += [ROOT / "node_modules/libraw-wasm/package.json"]
    pin = DIAG / "provia-wb-primitive-source-pin.json"
    files += [pin]
    files += [ROOT / row["path"] for row in json.loads(pin.read_text())["sources"]]
    files += [ROOT / "training/checkpoints/tools/libraw-gamma-source-0221-b860248a89d9/src/postprocessing/dcraw_process.cpp",
              ROOT / "training/checkpoints/tools/libraw-wasm-source-32fd36a9883a/libraw_wrapper.cpp"]
    dump(MANIFEST, {
        "scope": "WB primitive gate v2. F11 source-Auto training scenes only, source-only coefficients, no targets, models or protected tiers. New labelled gate; the original joint OLS gate remains FAILED.",
        "bindings": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "base_options": base, "scenes": sorted(scenes, key=lambda s: s["name"]),
        "output_dir": "training/checkpoints/provia-wb-primitive-v2",
        "sample_grid": {"offset": GRID_OFFSET, "stride": GRID_STRIDE, "space": "post-black-subtraction/scaling sparse native imageData; no demosaic, matrix, non-linear output curve or area average"},
        "structure_rules": "16-bit, 3 colours, equal size both arms; zero mixed pixels; each of 36 (row mod 6, col mod 6) phases has exactly one active channel over the full frame; identical pattern both arms; else STOP.",
        "primitive_rules": f"Per scene and channel: D and E in [{MASK_LOW}, {MASK_HIGH}] of {FULL_SCALE}, n >= {MIN_N}; mean |E-gD| <= {MEAN_LIMIT_CODES} code, max <= {MAX_LIMIT_CODES} codes; g from source multipliers with green minimum both arms. All 11 must pass.",
        "limits": "Raw photosite gain application only. Not demosaiced-output linearity, not rendered-recipe or Fuji parity. Returned pre_mul is descriptive, not applied-parameter readback.",
    })
    print("WB primitive v2 manifest SHA", sha(MANIFEST))


def run(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    m = json.loads(MANIFEST.read_text())
    for p in (MANIFEST, Path(__file__).resolve()):
        committed = subprocess.check_output(["git", "show", "HEAD:" + str(p.relative_to(ROOT))], cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == sha(p), "Code/manifest not committed")
    for f, h in m["bindings"].items():
        require(sha(ROOT / f) == h, "Bound file changed: " + f)
    complete = OUT / "decode-complete.json"
    progress = OUT / "decode-progress.json"
    decoded = json.loads((complete if complete.exists() else progress).read_text())
    require(decoded["manifest_sha256"] == expected, "Decode report from a different manifest")
    by_name = {r["name"]: r for r in decoded["scenes"]}
    require(len(by_name) == len(decoded["scenes"]) and set(by_name) <= {s["name"] for s in m["scenes"]}, "Duplicate/unapproved decoded scene")
    rows = []
    for scene in m["scenes"]:
        if scene["name"] not in by_name:
            continue
        arms = by_name[scene["name"]]["arms"]
        if set(arms) != {"D", "E"} or any(a.get("structure_error") for a in arms.values()):
            pattern, problems = check_structure(arms.get("D", {}), arms.get("E", {}))
            rows.append({"name": scene["name"], "structure_ok": False, "structure_problems": problems, "primitive_passed": None})
            continue
        samples = {}
        for a in ("D", "E"):
            p = ROOT / arms[a]["samples_file"]
            require(p.parent == OUT and sha(p) == arms[a]["samples_sha256"], "Sample file changed")
            samples[a] = np.fromfile(p, dtype="<u2")
        rows.append(evaluate_scene(scene, arms["D"], arms["E"], samples["D"], samples["E"]))
    structure_ok = len(rows) == 11 and all(r["structure_ok"] for r in rows)
    verdict = ("STRUCTURE_FAILURE_STOP" if any(not r["structure_ok"] for r in rows) else
               "INCOMPLETE" if not complete.exists() or not decoded.get("completed") or not structure_ok else
               "PASSED" if all(r["primitive_passed"] for r in rows) else "FAILED")
    dump(REPORT, {"scope": m["scope"], "manifest_sha256": expected, "decode_report_sha256": sha(complete if complete.exists() else progress),
                  "decode_completed": bool(decoded.get("completed")), "verdict": verdict, "rows": rows,
                  "original_joint_WB_gate": "FAILED, unchanged", "limits": m["limits"]})
    print(json.dumps({"verdict": verdict, "scenes": [{r["name"]: r.get("primitive_passed")} for r in rows]}, indent=2))


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
