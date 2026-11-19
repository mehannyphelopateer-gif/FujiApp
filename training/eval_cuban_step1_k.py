"""Classic Cuban Neg step 1 - the SINGLE frozen E4-only K evaluation.

Written and committed BEFORE any step-1 training result existed. Runs once:
refuses if its output already exists, if the checkpoint freeze manifest is
missing or any frozen hash differs, or if contract v1/v2, folds or run
manifest changed. Only seed 42 decides; 43/44 are report-only and are never
used to pick a winner. K scenes are read here and nowhere in training.

Contract space (v1 == v2 here): reference TIFF via
target_tiff.load_xraw_target_linear_matching at the .fjlrg dims; app =
model linear output at the same dims; Lab via linear_srgb_to_lab; dense
CIEDE2000. Per scene (stage_limits.E4): mean <= 2.0, p95 <= 5.0,
|signed mid dL*| <= 2.0 (mid = ref L* 25-75), rho_provia_to_E4 <= 0.5
over the frame with ref_b = Provia xraw-phase3, ref_s = E4, app_b =
untouched Round 11 seed 42, app_s = step-1 model. Set: >= 7/8 scenes pass
the conjunction. Aggregates: median mean dE00 <= 1.5, |cross-scene mean mid
dL*| <= 1.0, |mean dC* (ref C*>10)| <= 1.5, |mean hue (ref C*>20)| <= 3.0.

"app" here = the step-1 network output only (no shader pipeline, no app
integration). Internal simulation-adaptation check, not unseen-RAW
acceptance and not a whole-recipe pass. Visual review pending (user).
Monitor and finalHeldOut: no code path.

  python3 training/eval_cuban_step1_k.py -> training/diagnostics/cuban-step1-k-eval.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuban_step1 as S  # noqa: E402
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402
from cuban_contract_v2 import CONTRACT_V2  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from target_tiff import load_xraw_target_linear_matching  # noqa: E402

OUT = S.DIAG / "cuban-step1-k-eval.json"
LIM = CONTRACT_V2["stage_limits"]["E4"]
AGG = CONTRACT_V2["whole_recipe"]["aggregate"]
SET_MIN = 7


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = S.HybridPredictor(grid_spatial=c["grid_spatial"], grid_luma=c["grid_luma"], low_res=c["low_res"],
                          base_ch=c["base_ch"], refinement_base_ch=c["refinement_base_ch"],
                          refinement_max_delta=c["refinement_max_delta"], condition_channels=1)
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval(), c


def infer(mc, x, dev):
    m, c = mc
    with torch.no_grad():
        y = S.forward(m, c, torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0), dev)
    return y.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)


def hue_deg(lab):
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))


def metrics(app_lab, ref_lab):
    e = ciede2000(app_lab, ref_lab)
    L = ref_lab[..., 0]
    dL = app_lab[..., 0] - L
    Cr, Ca = np.hypot(ref_lab[..., 1], ref_lab[..., 2]), np.hypot(app_lab[..., 1], app_lab[..., 2])
    bands = {"shadow": L < 25, "mid": (L >= 25) & (L < 75), "high": L >= 75}
    m10, m20 = Cr > 10, Cr > 20
    dh = (hue_deg(app_lab) - hue_deg(ref_lab) + 180) % 360 - 180
    return {"mean_dE00": float(e.mean()), "p95_dE00": float(np.percentile(e, 95)),
            "signed_dL_by_tone_band": {b: (float(dL[r].mean()) if r.any() else None) for b, r in bands.items()},
            "tone_band_occupancy": {b: float(r.mean()) for b, r in bands.items()},
            "signed_dC_ref_C_gt_10": float((Ca - Cr)[m10].mean()) if m10.any() else None,
            "signed_hue_deg_ref_C_gt_20": float(dh[m20].mean()) if m20.any() else None}


def rho(ref_b, ref_s, app_b, app_s):
    num = ciede2000(ref_b + app_s - app_b, ref_s).mean()
    den = ciede2000(ref_b, ref_s).mean()
    return float(num / den), float(den)


def judge(m, r):
    mid = m["signed_dL_by_tone_band"]["mid"]
    checks = {"mean_dE00": m["mean_dE00"] <= LIM["mean_dE00"], "p95_dE00": m["p95_dE00"] <= LIM["p95_dE00"],
              "abs_mid_dL": mid is not None and abs(mid) <= LIM["abs_mid_dL"]}
    if r is not None:
        checks["rho_provia_to_E4"] = r <= LIM["rho_provia_to_E4"]
    return checks


def aggregate(rows):
    def mean_of(key, sub=None):
        v = [(r[key][sub] if sub else r[key]) for r in rows.values()]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None
    a = {"median_mean_dE00": float(np.median([r["mean_dE00"] for r in rows.values()])),
         "cross_scene_mean_mid_dL": mean_of("signed_dL_by_tone_band", "mid"),
         "cross_scene_mean_dC": mean_of("signed_dC_ref_C_gt_10"),
         "cross_scene_mean_hue_deg": mean_of("signed_hue_deg_ref_C_gt_20")}
    a["checks"] = {"median_mean_dE00": a["median_mean_dE00"] <= AGG["median_mean_dE00"]["E4"],
                   "abs_cross_scene_mean_mid_dL": abs(a["cross_scene_mean_mid_dL"]) <= AGG["abs_cross_scene_mean_mid_dL"],
                   "abs_cross_scene_mean_dC": abs(a["cross_scene_mean_dC"]) <= AGG["abs_cross_scene_mean_dC"],
                   "abs_cross_scene_mean_hue_deg": abs(a["cross_scene_mean_hue_deg"]) <= AGG["abs_cross_scene_mean_hue_deg"]}
    return a


def main():
    if OUT.exists():
        S.fail("K evaluation already ran - it is a single frozen evaluation")
    self_test()
    man, folds, allow, k = S.preflight()
    if not S.FREEZE.exists():
        S.fail("checkpoints not frozen (no cuban-step1-freeze.json)")
    fr = json.loads(S.FREEZE.read_text())
    for name, h in fr["checkpoints"].items():
        if S.sha256(S.CKPT / name) != h:
            S.fail(f"{name} != frozen hash")
    if fr["selection_sha256"] != S.sha256(S.SELECTION):
        S.fail("selection file changed after freeze")
    dev = S.device()
    models = {"round11_seed42_untouched": load_model(S.SOURCE, dev)}
    for s in S.FINAL_SEEDS:
        models[f"cn_step1_seed{s}"] = load_model(S.CKPT / f"cn-step1-seed{s}.pt", dev)
    res = {name: {} for name in models}
    files = {}
    for r in man["K_eval_only"]:
        d = S.CAL / r["shoot"]
        fj, e4, pv = d / "browser-phase3-linear.fjlrg", d / "xraw-cuban-E4-classic-neg-only.tiff", d / "xraw-phase3.tiff"
        if S.sha256(fj) != r["fjlrg_sha256"] or S.sha256(e4) != r["e4_sha256"]:
            S.fail(f"K file hash mismatch for {r['name']}")
        files[r["name"]] = {"fjlrg_sha256": r["fjlrg_sha256"], "e4_sha256": r["e4_sha256"], "provia_sha256": S.sha256(pv)}
        x, w, h = load_fjlrg_linear(fj)
        ref_s = linear_srgb_to_lab(load_xraw_target_linear_matching(e4, w, h).astype(np.float64))
        ref_b = linear_srgb_to_lab(load_xraw_target_linear_matching(pv, w, h).astype(np.float64))
        outs = {name: linear_srgb_to_lab(infer(mc, x, dev)) for name, mc in models.items()}
        for name, lab in outs.items():
            m = metrics(lab, ref_s)
            r_val, den = rho(ref_b, ref_s, outs["round11_seed42_untouched"], lab)
            is_base = name == "round11_seed42_untouched"
            m["rho_provia_to_E4"] = None if is_base else r_val
            m["fuji_provia_to_E4_mean_dE00"] = den
            m["checks"] = judge(m, None if is_base else r_val)
            m["pass"] = all(m["checks"].values())
            res[name][r["name"]] = m
        print(r["name"], {n: (round(res[n][r["name"]]["mean_dE00"], 3), round(res[n][r["name"]]["p95_dE00"], 3),
                              res[n][r["name"]]["pass"]) for n in models}, flush=True)
    summary = {}
    for name, rows in res.items():
        n_pass = sum(v["pass"] for v in rows.values())
        agg = aggregate(rows)
        summary[name] = {"scenes_passing": n_pass, "of": len(rows), "set_check": n_pass >= SET_MIN,
                         "aggregates": agg, "step1_E4_pass": n_pass >= SET_MIN and all(agg["checks"].values())}
    S.check_source()
    out = {"label": "internal simulation-adaptation check on K (E4-only); not unseen-RAW acceptance; not a whole-recipe pass",
           "decision_model": "cn_step1_seed42", "decision": summary["cn_step1_seed42"]["step1_E4_pass"],
           "report_only": ["cn_step1_seed43", "cn_step1_seed44"],
           "baseline_context_only": "round11_seed42_untouched (no Classic Negative; rho not applicable)",
           "not_run": "seed42 + app Classic Negative LUT baseline (needs app-pipeline reimplementation)",
           "visual_review": "pending user review",
           "limits": {"E4": LIM, "aggregate": {k: (v["E4"] if isinstance(v, dict) else v) for k, v in AGG.items()},
                      "set_min": SET_MIN},
           "contract_v2_sha256": S.V2_SHA, "freeze_sha256": S.sha256(S.FREEZE), "files": files,
           "summary": summary, "per_scene": res}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"decision": out["decision"], "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
