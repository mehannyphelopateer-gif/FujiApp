"""Phase 4 round 11 - isolated change is CAPTURE-DR CONDITIONING ONLY,
per approval (2026-11-13). Round 9's architecture, data (599 train),
seeds, loss, schedule and monitor set are otherwise unchanged; the TIFF
targets are output-DR100 throughout (Codex-verified), so the only new
information is each scene's capture dynamic-range mode, fed to the grid
encoder as one constant input plane (0 / 0.5 / 1 for DR100 / 200 / 400;
train.capture_dr_condition). +144 parameters at base_ch=16.

Why (docs/phase4-scene-adaptive-scope.md, "Signed-error reconciliation
(2026-11-13)"): the 5 persistent monitor regressions are exactly the 5
monitor scenes captured at DR200 in a 71-scene DR400 monitor tier; the
.fjlrg input carries no DR exposure compensation, the Fuji target does,
and a thumbnail cannot reveal the capture DR. This run tests whether
supplying that metadata fixes the exposure errors. Missing/ambiguous
capture DR raises (never guessed).

Comparison target: ROUND 9 (current selected-best), unchanged 76-scene
monitor tier, reported separately per capture-DR group (DR400: 71
scenes, DR200: 5 scenes). Monitor has no DR100 captures, so nothing
here can establish DR100 accuracy; the 12 DR100 train scenes' signed
errors are reported as a train-tier observation only, not evaluation.

Predeclared success (fixed before this run, verified after) - ALL of:
  1. DR200 union-of-seed regression set strictly smaller than round 9's
     (round 9: 4 of 5 - DSCF0590/0873/0878/0879; DSCF0553 passes L1),
  2. mean signed mid-band luma error over the 5 DR200 scenes (3 seeds)
     below +0.05 (round 9: ~+0.16),
  3. no NEW monitor regressions vs round 9 anywhere (DR400 or DR200),
  4. crop-level L1 on every DR200 scene not worse than round 9.
Round 11 is selected-best only if all four hold; otherwise round 9 is
preserved. "All 5 DR200 scenes non-regressed under all 3 seeds" is
reported as the stronger outcome but is not required.

train + monitor only - finalHeldOut has no code path here.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from dataset import Phase4PairDataset
from diagnose_signed_errors import BANDS, band_stats
from model import BilateralGridPredictor, HybridPredictor, compute_luma_guide
from train import build_arg_parser, train_one_config

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"

BASE_CFG = {
    "grid_spatial": 8, "grid_luma": 9, "base_ch": 16, "lr": 1e-3,
    "epochs": 20, "lr_schedule": "cosine", "early_stopping_patience": 5,
    "grid_tv_weight": 0.0, "luma_mode": "rec709",
    "use_hybrid": True, "refinement_base_ch": 8, "refinement_max_delta": 0.08,
    "capture_dr_conditioning": True,
}
SEEDS = [42, 43, 44]
RUN_PREFIX = "r11-drcond"
EXPECTED_DR200_MONITOR = {"DSCF0553.RAF", "DSCF0590.RAF", "DSCF0873.RAF", "DSCF0878.RAF", "DSCF0879.RAF"}
ROUND9_CHECKPOINTS = {42: "r9-expandeddata-seed42-best.pt", 43: "r9-expandeddata-seed43-best.pt", 44: "r9-expandeddata-seed44-best.pt"}
MID_SIGNED_THRESHOLD = 0.05


def run_one(cfg):
    existing = LOG_DIR / f"{cfg['run_name']}.json"
    if existing.exists():
        print(f"\n=== {cfg['run_name']} already completed, reusing saved log ===")
        return json.loads(existing.read_text())
    ap = build_arg_parser()
    args = ap.parse_args([])
    for k, v in cfg.items():
        setattr(args, k, v)
    print(f"\n=== starting {cfg['run_name']} ===")
    return train_one_config(args)


def load_model_from_ckpt(ckpt_name, device):
    ckpt = torch.load(CKPT_DIR / ckpt_name, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    conditioned = bool(cfg.get("capture_dr_conditioning", False))
    cc = 1 if conditioned else 0
    if cfg.get("use_hybrid", False):
        model = HybridPredictor(
            grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"], low_res=cfg["low_res"],
            base_ch=cfg["base_ch"], refinement_base_ch=cfg["refinement_base_ch"],
            refinement_max_delta=cfg["refinement_max_delta"], condition_channels=cc,
        ).to(device)
    else:
        model = BilateralGridPredictor(
            grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"], low_res=cfg["low_res"],
            base_ch=cfg["base_ch"], use_detail_branch=cfg.get("use_detail_branch", False),
            detail_ch=cfg.get("detail_ch", 8), condition_channels=cc,
        ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, conditioned


def crop_around(arr_hw, cx_norm, cy_norm, crop_frac=0.22):
    h, w = arr_hw.shape[:2]
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    cx, cy = int(cx_norm * w), int(cy_norm * h)
    x0 = max(0, min(w - cw, cx - cw // 2))
    y0 = max(0, min(h - ch, cy - ch // 2))
    return arr_hw[y0:y0 + ch, x0:x0 + cw]


def frame_signed(stats):
    return float(sum(stats[b]["pixel_fraction"] * stats[b]["signed_linear_error"] for b, _, _ in BANDS if "signed_linear_error" in stats.get(b, {})))


def evaluate(checkpoints_by_seed, device, ds, centroid_by_name, signed_names):
    """Per-scene per-seed: model_l1, regressed, crop_l1 (where a centroid
    exists), signed mid-band / frame-weighted luma error (for signed_names)."""
    models = {s: load_model_from_ckpt(c, device) for s, c in checkpoints_by_seed.items()}
    per_scene = {}
    for i in range(len(ds)):
        item = ds[i]
        name = item["name"]
        full_res = item["input"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        stops = item["capture_dr_stops"]
        if not torch.isfinite(stops):
            raise ValueError(f"{name}: capture DR missing/ambiguous in manifest - not guessing")
        condition = (stops / 2.0).view(1).to(device)
        baseline_l1 = (full_res - target).abs().mean().item()
        per_scene[name] = {"baseline_l1": baseline_l1, "capture_dr": int(round(100 * 2 ** float(stops))), "seeds": {}}
        want_signed = name in signed_names
        if want_signed:
            target_luma = compute_luma_guide(target, "rec709").squeeze().cpu().numpy()
            input_luma = compute_luma_guide(full_res, "rec709").squeeze().cpu().numpy()
            target_rgb = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
        for seed, (model, cfg, conditioned) in models.items():
            with torch.no_grad():
                pred, aux = model.predict_and_apply(full_res, cfg["low_res"], cfg.get("luma_mode", "rec709"),
                                                    condition=condition if conditioned else None)
                residual_hw = (pred - target).abs().mean(dim=1).squeeze(0).cpu().numpy()
            model_l1 = float(residual_hw.mean())
            entry = {"model_l1": model_l1, "regressed": model_l1 > baseline_l1}
            if "delta" in aux:
                entry["mean_abs_delta"] = float(aux["delta"].abs().mean().item())
            if name in centroid_by_name:
                cx, cy = centroid_by_name[name]
                entry["crop_l1"] = float(crop_around(residual_hw, cx, cy).mean())
            if want_signed:
                pred_luma = compute_luma_guide(pred, "rec709").squeeze().cpu().numpy()
                pred_rgb = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
                st = band_stats(pred_luma, target_luma, input_luma, pred_rgb, target_rgb)
                entry["mid_signed"] = st["mid"].get("signed_linear_error")
                entry["shadow_signed"] = st["shadow"].get("signed_linear_error")
                entry["highlight_signed"] = st["highlight"].get("signed_linear_error")
                entry["frame_signed"] = frame_signed(st)
            per_scene[name]["seeds"][seed] = entry
    return per_scene


def union_regressed(per_scene, names=None):
    return {n for n, d in per_scene.items() if (names is None or n in names) and any(s["regressed"] for s in d["seeds"].values())}


def mean_metric(per_scene, name, key):
    vals = [per_scene[name]["seeds"][s].get(key) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def group_report(per_scene_r9, per_scene_r11, names, label):
    u9, u11 = union_regressed(per_scene_r9, names), union_regressed(per_scene_r11, names)
    rows = {}
    for n in sorted(names):
        rows[n] = {
            "regressed_seeds_round9": [s for s, v in per_scene_r9[n]["seeds"].items() if v["regressed"]],
            "regressed_seeds_round11": [s for s, v in per_scene_r11[n]["seeds"].items() if v["regressed"]],
            "model_l1_round9": mean_metric(per_scene_r9, n, "model_l1"),
            "model_l1_round11": mean_metric(per_scene_r11, n, "model_l1"),
            "baseline_l1": per_scene_r9[n]["baseline_l1"],
            "crop_l1_round9": mean_metric(per_scene_r9, n, "crop_l1"),
            "crop_l1_round11": mean_metric(per_scene_r11, n, "crop_l1"),
            "mid_signed_round9": mean_metric(per_scene_r9, n, "mid_signed"),
            "mid_signed_round11": mean_metric(per_scene_r11, n, "mid_signed"),
            "frame_signed_round9": mean_metric(per_scene_r9, n, "frame_signed"),
            "frame_signed_round11": mean_metric(per_scene_r11, n, "frame_signed"),
        }
    mids9 = [rows[n]["mid_signed_round9"] for n in names if not np.isnan(rows[n]["mid_signed_round9"])]
    mids11 = [rows[n]["mid_signed_round11"] for n in names if not np.isnan(rows[n]["mid_signed_round11"])]
    return {
        "label": label, "n_scenes": len(names),
        "union_regressed_round9": sorted(u9), "union_regressed_round11": sorted(u11),
        "resolved_vs_round9": sorted(u9 - u11), "persisted_vs_round9": sorted(u9 & u11), "new_regressions_vs_round9": sorted(u11 - u9),
        "mean_mid_signed_round9": float(np.mean(mids9)) if mids9 else None,
        "mean_mid_signed_round11": float(np.mean(mids11)) if mids11 else None,
        "mean_model_l1_round9": float(np.mean([rows[n]["model_l1_round9"] for n in names])),
        "mean_model_l1_round11": float(np.mean([rows[n]["model_l1_round11"] for n in names])),
        "per_scene": rows,
    }


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    results = {}
    for seed in SEEDS:
        cfg = dict(BASE_CFG)
        cfg["seed"] = seed
        cfg["run_name"] = f"{RUN_PREFIX}-seed{seed}"
        results[seed] = run_one(cfg)
        print(f"[{cfg['run_name']}] train scenes used: {results[seed]['train_scenes']}")

    print("\n=== training done, evaluating round 9 (current selected-best) vs round 11 (capture-DR conditioning) ===")
    monitor_ds = Phase4PairDataset("monitor")
    dr_by_name = {s["name"]: s.get("captureDynamicRange") for s in monitor_ds.scenes}
    dr200 = {n for n, d in dr_by_name.items() if d == 200}
    dr400 = {n for n, d in dr_by_name.items() if d == 400}
    other = set(dr_by_name) - dr200 - dr400
    if dr200 != EXPECTED_DR200_MONITOR or other:
        raise ValueError(f"monitor capture-DR composition unexpected: DR200={sorted(dr200)} other={sorted(other)}")

    diag = json.loads((OUT_DIR / "structural_diagnosis.json").read_text())
    centroid_by_name = {r["name"]: tuple(r["largest_centroid_norm"]) for r in diag["regressing_scenes"] if r.get("largest_centroid_norm") and r["name"] in dr200}

    new_checkpoints = {s: f"{RUN_PREFIX}-seed{s}-best.pt" for s in SEEDS}
    signed_names = set(dr_by_name)  # signed errors for every monitor scene (cheap, and DR400 drift matters)
    r9 = evaluate(ROUND9_CHECKPOINTS, device, monitor_ds, centroid_by_name, signed_names)
    r11 = evaluate(new_checkpoints, device, monitor_ds, centroid_by_name, signed_names)

    g200 = group_report(r9, r11, dr200, "monitor DR200 captures")
    g400 = group_report(r9, r11, dr400, "monitor DR400 captures")
    all_new = sorted(union_regressed(r11) - union_regressed(r9))

    crit_fewer_dr200 = len(g200["union_regressed_round11"]) < len(g200["union_regressed_round9"])
    crit_mid_signed = g200["mean_mid_signed_round11"] is not None and g200["mean_mid_signed_round11"] < MID_SIGNED_THRESHOLD
    crit_no_new = len(all_new) == 0
    crit_crop = all(
        not (g200["per_scene"][n]["crop_l1_round11"] > g200["per_scene"][n]["crop_l1_round9"])
        for n in dr200 if not np.isnan(g200["per_scene"][n]["crop_l1_round9"])
    )
    success = crit_fewer_dr200 and crit_mid_signed and crit_no_new and crit_crop
    all_dr200_clear = len(g200["union_regressed_round11"]) == 0
    selected_best = "round11" if success else "round9"

    # Train-tier observation only (not evaluation): the 12 DR100 captures
    # from batch 2, which the model also trained on.
    train_ds = Phase4PairDataset("train")
    dr100_idx = [i for i, s in enumerate(train_ds.scenes) if s.get("captureDynamicRange") == 100]
    train_ds.scenes = [train_ds.scenes[i] for i in dr100_idx]
    t9 = evaluate(ROUND9_CHECKPOINTS, device, train_ds, {}, {s["name"] for s in train_ds.scenes})
    t11 = evaluate(new_checkpoints, device, train_ds, {}, {s["name"] for s in train_ds.scenes})
    dr100_train = group_report(t9, t11, {s["name"] for s in train_ds.scenes}, "TRAIN-tier DR100 captures (observation only, not evaluation)")

    report = {
        "intervention": "capture-DR conditioning only (one constant encoder input plane, 0/0.5/1 for DR100/200/400, +144 params) - round 9's architecture/data/seeds/loss/schedule/monitor set unchanged; TIFF targets output-DR100 throughout. Comparison target is round 9 on the unchanged 76-scene monitor tier, reported per capture-DR group.",
        "seeds": SEEDS,
        "train_scenes_used": results[SEEDS[0]]["train_scenes"],
        "n_params": results[SEEDS[0]].get("n_params"),
        "monitor_capture_dr_composition": {"DR400": len(dr400), "DR200": len(dr200), "DR100": 0},
        "monitor_dr200": g200,
        "monitor_dr400": g400,
        "all_monitor_new_regressions_vs_round9": all_new,
        "train_dr100_observation": dr100_train,
        "predeclared_success_criteria": (
            "ALL of: (1) DR200 union-of-seed regression set strictly smaller than round 9's; (2) mean signed mid-band luma error over the 5 DR200 "
            f"monitor scenes below +{MID_SIGNED_THRESHOLD}; (3) no new monitor regressions vs round 9 anywhere; (4) crop-level L1 on every DR200 scene not worse than round 9."
        ),
        "criterion_1_fewer_dr200_regressions": crit_fewer_dr200,
        "criterion_2_dr200_mid_signed_below_threshold": crit_mid_signed,
        "criterion_3_no_new_monitor_regressions": crit_no_new,
        "criterion_4_no_lost_crop_gains_dr200": crit_crop,
        "predeclared_success_met": success,
        "stronger_outcome_all_dr200_clear_all_seeds": all_dr200_clear,
        "selected_best_model": selected_best,
        "selection_note": (
            "Round 11 cleared the predeclared bar and is the new selected-best model."
            if success else
            "Round 11 did not clear the predeclared bar - Round 9 remains the selected-best model, per instruction. Round 11's checkpoints are kept on disk but not selected."
        ),
        "dr100_caveat": "monitor contains no DR100 captures; nothing here establishes DR100 accuracy - separate evaluation examples are needed for that claim.",
    }
    (OUT_DIR / "round11-drconditioning-comparison.json").write_text(json.dumps(report, indent=2))

    def print_group(g):
        print(f"\n{g['label']} (n={g['n_scenes']}): union regressed round9 {len(g['union_regressed_round9'])} -> round11 {len(g['union_regressed_round11'])}; "
              f"resolved {g['resolved_vs_round9']}; persisted {g['persisted_vs_round9']}; NEW {g['new_regressions_vs_round9']}")
        print(f"  mean mid-band signed: {g['mean_mid_signed_round9']:+.4f} -> {g['mean_mid_signed_round11']:+.4f}   mean model L1 {g['mean_model_l1_round9']:.4f} -> {g['mean_model_l1_round11']:.4f}")
        if g["n_scenes"] <= 12:
            for n, r in g["per_scene"].items():
                print(f"    {n}: L1 {r['model_l1_round9']:.4f}->{r['model_l1_round11']:.4f} (baseline {r['baseline_l1']:.4f}) regressed seeds {r['regressed_seeds_round9']}->{r['regressed_seeds_round11']}  "
                      f"mid_signed {r['mid_signed_round9']:+.3f}->{r['mid_signed_round11']:+.3f}  frame_signed {r['frame_signed_round9']:+.3f}->{r['frame_signed_round11']:+.3f}  "
                      f"crop {r['crop_l1_round9']:.4f}->{r['crop_l1_round11']:.4f}")

    print(f"\ntrain scenes used: {report['train_scenes_used']}   params: {report['n_params']}")
    print_group(g200)
    print_group(g400)
    print_group(dr100_train)
    print(f"\nall monitor NEW regressions vs round9: {all_new}")
    print(f"criteria: fewer_dr200={crit_fewer_dr200}  mid_signed<{MID_SIGNED_THRESHOLD}={crit_mid_signed}  no_new={crit_no_new}  crop_ok={crit_crop}")
    print(f"predeclared success met: {success}   (all 5 DR200 clear under all seeds: {all_dr200_clear})")
    print(f"SELECTED BEST MODEL: {selected_best}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
