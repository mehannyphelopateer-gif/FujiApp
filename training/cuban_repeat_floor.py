"""Classic Cuban Neg - X RAW Studio repeat floor (Codex 2026-11-18).

Compares each pilot's first E1/E2 export with its verified repeat (R1/R2) in
the evaluation space: orientation-corrected 16-bit TIFF -> linear sRGB ->
area-downsampled (linear light) to the scene's exact .fjlrg dimensions ->
CIELAB D65, dense (every pixel). Fuji vs Fuji only: no app output, no
fitting, no monitor or finalHeldOut data.

Hold rule (declared before any repeat export exists): HOLD - stop and review
with Codex, no automatic loosening - if on either pilot the E2-vs-R2 repeat
floor exceeds 0.25x the smallest limit it would affect:
  mean ΔE00 > 0.375  (0.25 x 1.5, the E4 aggregate median mean-ΔE00 limit)
  p95  ΔE00 > 1.25   (0.25 x 5.0, the E4 per-scene p95 limit)
E1 (grain) is reported for grain determinism and the detail-statistic floor;
grain is scored statistically, so its pixel ΔE is not held.

  python3 training/cuban_repeat_floor.py  ->  training/diagnostics/cuban-repeat-floor.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402
from cuban_exports import BATCHES, CAL, INSTALL_NAME, REPEAT_OF, REPORT, sha256  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from target_tiff import load_xraw_target_linear_matching  # noqa: E402

OUT = Path(__file__).resolve().parent / "diagnostics" / "cuban-repeat-floor.json"
FJLRG = "browser-phase3-linear.fjlrg"
HOLD_MEAN, HOLD_P95 = 0.25 * 1.5, 0.25 * 5.0
# band-pass L* detail at working (.fjlrg) scale: fine G(1)-G(3) for NR,
# mid-scale G(4)-G(16) for Clarity; repeat floors feed the repeat-aware tests
DETAIL_BANDS = {"fine_s1_s3": (1.0, 3.0), "mid_s4_s16": (4.0, 16.0)}


def eval_lab(shoot, tiff_name, dims):
    lin = load_xraw_target_linear_matching(CAL / shoot / tiff_name, *dims)
    return linear_srgb_to_lab(lin.astype(np.float64)), lin


def detail_rms(L, band):
    lo, hi = DETAIL_BANDS[band]
    a = cv2.GaussianBlur(L, (0, 0), lo) - cv2.GaussianBlur(L, (0, 0), hi)
    return float(np.sqrt(np.mean(a ** 2)))


def pair_stats(lab_a, lab_b, lin_a, lin_b):
    e = ciede2000(lab_a, lab_b)
    det = {}
    for band in DETAIL_BANDS:
        da, db = detail_rms(lab_a[..., 0], band), detail_rms(lab_b[..., 0], band)
        det[band] = {"rms_L": [round(da, 4), round(db, 4)], "rel_diff": round(abs(db - da) / max(da, 1e-9), 5)}
    return {"mean_dE00": round(float(e.mean()), 4), "p95_dE00": round(float(np.percentile(e, 95)), 4),
            "p99_9_dE00": round(float(np.percentile(e, 99.9)), 4), "max_dE00": round(float(e.max()), 4),
            "fraction_dE00_over_0_5": round(float((e > 0.5).mean()), 5),
            "max_abs_linear": float(np.abs(lin_a - lin_b).max()),
            "detail": det}


def main():
    self_test()
    reps = {}
    for kind in ("pilot", "repeat"):
        try:
            rep = json.loads(REPORT[kind].read_text())
        except Exception as e:
            sys.exit(f"refused: {kind} verification report unusable ({type(e).__name__})")
        if rep.get("passed") is not True:
            sys.exit(f"refused: {kind} verification has not passed")
        reps[kind] = {(e["name"], e["variant"]): e["source_sha256"] for e in rep["entries"]}
    rows = json.loads(BATCHES["repeat"][1].read_text())
    res = {"space": "linear-light area-downsample to .fjlrg dims, dense CIELAB D65 / CIEDE2000",
           "hold_rule": {"mean_dE00_E2_vs_R2_max": HOLD_MEAN, "p95_dE00_E2_vs_R2_max": HOLD_P95},
           "scenes": {}, "hold": False, "hold_reasons": []}
    for r in rows:
        files = {v: INSTALL_NAME[v] for v in [*REPEAT_OF, *REPEAT_OF.values()]}
        for v, f in files.items():
            want = reps["repeat" if v in REPEAT_OF else "pilot"][(r["name"], v)]
            if sha256(CAL / r["shoot"] / f) != want:
                sys.exit(f"refused: installed {r['shoot']}/{f} != its verified TIFF")
        _, w, h = load_fjlrg_linear(CAL / r["shoot"] / FJLRG)
        lab, lin = {}, {}
        for v, f in files.items():
            lab[v], lin[v] = eval_lab(r["shoot"], f, (w, h))
        sc = {"dims": [w, h]}
        for rv, o in REPEAT_OF.items():
            sc[f"{o}_vs_{rv}"] = pair_stats(lab[o], lab[rv], lin[o], lin[rv])
        g1 = lab["E1-full-recipe"][..., 0] - lab["E2-no-grain"][..., 0]
        g2 = lab["R1-full-recipe-repeat"][..., 0] - lab["R2-no-grain-repeat"][..., 0]
        sc["grain_residual_L_corr_first_vs_repeat"] = round(float(np.corrcoef(g1.ravel(), g2.ravel())[0, 1]), 4)
        sc["grain_residual_L_std"] = [round(float(g1.std()), 4), round(float(g2.std()), 4)]
        e2 = sc["E2-no-grain_vs_R2-no-grain-repeat"]
        for k, lim in (("mean_dE00", HOLD_MEAN), ("p95_dE00", HOLD_P95)):
            if e2[k] > lim:
                res["hold"] = True
                res["hold_reasons"].append(f"{r['name']}: E2 repeat {k} {e2[k]} > {lim}")
        res["scenes"][r["name"]] = sc
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("HOLD - review with Codex before requesting more exports" if res["hold"] else "repeat floor within hold rule")
    return 1 if res["hold"] else 0


if __name__ == "__main__":
    sys.exit(main())
