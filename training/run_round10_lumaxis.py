"""Phase 4 round 10 - isolated change is grid_luma ONLY, per instruction.
Round 9's architecture/data/seeds/loss/schedule/monitor set are otherwise
completely unchanged: `grid_luma: 9 -> 13`. `refinement_max_delta` stays
at 0.08 (explicitly not touched, per instruction - round 9's diagnosis
showed the refinement net isn't saturating that bound, so it's not the
lever being tested here). No other tuning.

Motivation (training/diagnose_round9_remaining.py, 2026-11-12): the 4
scenes still regressed after round 9 (DSCF0590, DSCF0873, DSCF0878,
DSCF0879) all show the same systematic bias in both round 6 and round 9
- predicted near-black fraction below target, predicted bright fraction
above target, predicted luma std wider than target - insufficient
shadow/highlight compression, not a spatial/seam defect. grid_luma is
the grid's luma-axis resolution (how finely it can apply different
corrections to true shadows vs true highlights vs midtones) and was
never isolated-tested before - round 5 varied grid_spatial only, with
grid_luma explicitly held fixed at 9.

Comparison target: ROUND 9 (the current selected-best), on the unchanged
76-scene monitor tier - not round 6, since round 9 already superseded it.

Predeclared success (checked before this run, verified after, per
instruction): keep round 10 only if EITHER (a) fewer persistent
union-of-seed regressions on monitor than round 9's 4-scene union, OR
(b) crop-level L1 clearly improves (strictly lower, mean over seeds) on
ALL FOUR of DSCF0590/DSCF0873/DSCF0878/DSCF0879 - note DSCF0878 IS
counted in this criterion this round, unlike round 9 where it was
excluded - AND, in either case, no NEW monitor regressions beyond
round 9's existing 4-scene union. DSCF0878 is still reported separately
as the structural outlier (its error location doesn't match the other
3's mechanism), but it counts toward path (b) this round per instruction.

Also records near-black/bright-fraction/luma-std on the 4 problem scenes
as secondary, non-gating observational data (reusing
diagnose_round9_remaining.py's approach) - this directly tests whether
grid_luma moved the hypothesized mechanism at all, which matters for
interpreting the result regardless of whether the pass/fail bar clears.

train + monitor only - finalHeldOut has no code path here.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from dataset import Phase4PairDataset
from diagnose_structural import luma_distribution_stats
from model import BilateralGridPredictor, HybridPredictor, compute_luma_guide
from train import build_arg_parser, train_one_config

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"

BASE_CFG = {
    "grid_spatial": 8, "grid_luma": 13, "base_ch": 16, "lr": 1e-3,
    "epochs": 20, "lr_schedule": "cosine", "early_stopping_patience": 5,
    "grid_tv_weight": 0.0, "luma_mode": "rec709",
    "use_hybrid": True, "refinement_base_ch": 8, "refinement_max_delta": 0.08,
}
SEEDS = [42, 43, 44]

PROBLEM_4 = ["DSCF0590.RAF", "DSCF0873.RAF", "DSCF0878.RAF", "DSCF0879.RAF"]
OUTLIER = "DSCF0878.RAF"
ROUND9_CHECKPOINTS = {42: "r9-expandeddata-seed42-best.pt", 43: "r9-expandeddata-seed43-best.pt", 44: "r9-expandeddata-seed44-best.pt"}


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
    if cfg.get("use_hybrid", False):
        model = HybridPredictor(
            grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"], low_res=cfg["low_res"],
            base_ch=cfg["base_ch"], refinement_base_ch=cfg["refinement_base_ch"],
            refinement_max_delta=cfg["refinement_max_delta"],
        ).to(device)
    else:
        model = BilateralGridPredictor(
            grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"], low_res=cfg["low_res"],
            base_ch=cfg["base_ch"], use_detail_branch=cfg.get("use_detail_branch", False),
            detail_ch=cfg.get("detail_ch", 8),
        ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def crop_around(arr_hw, cx_norm, cy_norm, crop_frac=0.22):
    h, w = arr_hw.shape[:2]
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    cx, cy = int(cx_norm * w), int(cy_norm * h)
    x0 = max(0, min(w - cw, cx - cw // 2))
    y0 = max(0, min(h - ch, cy - ch // 2))
    return arr_hw[y0:y0 + ch, x0:x0 + cw]


def evaluate(checkpoints_by_seed, device, monitor_ds, centroid_by_name):
    models = {s: load_model_from_ckpt(c, device) for s, c in checkpoints_by_seed.items()}
    per_scene = {}
    for i in range(len(monitor_ds)):
        item = monitor_ds[i]
        name = item["name"]
        full_res = item["input"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        baseline_l1 = (full_res - target).abs().mean().item()
        per_scene[name] = {"baseline_l1": baseline_l1, "seeds": {}}
        for seed, (model, cfg) in models.items():
            with torch.no_grad():
                pred, aux = model.predict_and_apply(full_res, cfg["low_res"], cfg.get("luma_mode", "rec709"))
                residual_hw = (pred - target).abs().mean(dim=1).squeeze(0).cpu().numpy()
            model_l1 = float(residual_hw.mean())
            entry = {"model_l1": model_l1, "regressed": model_l1 > baseline_l1}
            if "delta" in aux:
                entry["mean_abs_delta"] = float(aux["delta"].abs().mean().item())
            if name in PROBLEM_4:
                if name in centroid_by_name:
                    cx, cy = centroid_by_name[name]
                    crop = crop_around(residual_hw, cx, cy)
                    entry["crop_l1"] = float(crop.mean())
                pred_luma = compute_luma_guide(pred, cfg.get("luma_mode", "rec709")).squeeze(0).squeeze(0).cpu().numpy()
                dist = luma_distribution_stats(pred_luma)
                entry["pred_near_black_fraction"] = dist["near_black_fraction"]
                entry["pred_bright_fraction"] = dist["bright_fraction"]
                entry["pred_luma_std"] = dist["luma_std"]
            per_scene[name]["seeds"][seed] = entry
    return per_scene


def union_regressed(per_scene):
    return {n for n, d in per_scene.items() if any(s["regressed"] for s in d["seeds"].values())}


def mean_metric(per_scene, name, key):
    vals = [per_scene[name]["seeds"][s].get(key) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    results = {}
    for seed in SEEDS:
        cfg = dict(BASE_CFG)
        cfg["seed"] = seed
        cfg["run_name"] = f"r10-lumaxis13-seed{seed}"
        results[seed] = run_one(cfg)
        print(f"[r10-lumaxis13-seed{seed}] train scenes used: {results[seed]['train_scenes']}")

    print("\n=== training done, evaluating round 9 (current selected-best) vs round 10 (grid_luma 9->13) ===")
    monitor_ds = Phase4PairDataset("monitor")

    diag = json.loads((OUT_DIR / "structural_diagnosis.json").read_text())
    centroid_by_name = {
        r["name"]: tuple(r["largest_centroid_norm"])
        for r in diag["regressing_scenes"]
        if r.get("largest_centroid_norm") and r["name"] in PROBLEM_4
    }

    new_checkpoints = {s: f"r10-lumaxis13-seed{s}-best.pt" for s in SEEDS}
    round9_per_scene = evaluate(ROUND9_CHECKPOINTS, device, monitor_ds, centroid_by_name)
    round10_per_scene = evaluate(new_checkpoints, device, monitor_ds, centroid_by_name)

    round9_union = union_regressed(round9_per_scene)
    round10_union = union_regressed(round10_per_scene)

    resolved_vs_round9 = round9_union - round10_union
    persisted_vs_round9 = round9_union & round10_union
    new_regressions_vs_round9 = round10_union - round9_union

    def status_for(name):
        crop_r9 = mean_metric(round9_per_scene, name, "crop_l1")
        crop_r10 = mean_metric(round10_per_scene, name, "crop_l1")
        return {
            "regressed_seeds_round9": [s for s, v in round9_per_scene[name]["seeds"].items() if v["regressed"]],
            "regressed_seeds_round10": [s for s, v in round10_per_scene[name]["seeds"].items() if v["regressed"]],
            "crop_l1_round9": crop_r9,
            "crop_l1_round10": crop_r10,
            "crop_l1_clearly_improved": crop_r10 < crop_r9,
            "pred_near_black_fraction_round9": mean_metric(round9_per_scene, name, "pred_near_black_fraction"),
            "pred_near_black_fraction_round10": mean_metric(round10_per_scene, name, "pred_near_black_fraction"),
            "pred_bright_fraction_round9": mean_metric(round9_per_scene, name, "pred_bright_fraction"),
            "pred_bright_fraction_round10": mean_metric(round10_per_scene, name, "pred_bright_fraction"),
            "pred_luma_std_round9": mean_metric(round9_per_scene, name, "pred_luma_std"),
            "pred_luma_std_round10": mean_metric(round10_per_scene, name, "pred_luma_std"),
        }

    problem_4_status = {name: status_for(name) for name in PROBLEM_4}

    fewer_persistent_regressions = len(round10_union) < len(round9_union)
    all_four_clearly_improved = all(problem_4_status[n]["crop_l1_clearly_improved"] for n in PROBLEM_4)
    no_new_monitor_regressions = len(new_regressions_vs_round9) == 0
    success = (fewer_persistent_regressions or all_four_clearly_improved) and no_new_monitor_regressions
    selected_best = "round10" if success else "round9"

    report = {
        "intervention": "grid_luma only (9 -> 13) - round 9's architecture/data/seeds/loss/schedule/monitor set completely unchanged, refinement_max_delta untouched. Comparison target is round 9 (current selected-best) on the unchanged 76-scene monitor tier.",
        "seeds": SEEDS,
        "train_scenes_used": results[SEEDS[0]]["train_scenes"],
        "round9_union_regressed": sorted(round9_union),
        "round10_union_regressed": sorted(round10_union),
        "resolved_vs_round9": sorted(resolved_vs_round9),
        "persisted_vs_round9": sorted(persisted_vs_round9),
        "new_regressions_vs_round9": sorted(new_regressions_vs_round9),
        "problem_4_status": problem_4_status,
        "outlier_note_dscf0878": "DSCF0878 counts toward the all-four-clearly-improved path this round, per instruction, but is reported separately above since its error location still doesn't match the other 3's highlight-centroid mechanism.",
        "predeclared_success_met": success,
        "predeclared_success_criteria": "fewer persistent union-of-seed regressions than round 9's 4-scene union, OR crop-level L1 clearly improves (strictly lower) on all 4 of DSCF0590/DSCF0873/DSCF0878/DSCF0879 - AND no new monitor regressions beyond round 9's existing union either way",
        "fewer_persistent_regressions": fewer_persistent_regressions,
        "all_four_clearly_improved": all_four_clearly_improved,
        "no_new_monitor_regressions": no_new_monitor_regressions,
        "selected_best_model": selected_best,
        "selection_note": (
            "Round 10 cleared the predeclared bar and is the new selected-best model."
            if success else
            "Round 10 did not clear the predeclared bar - Round 9 remains the selected-best model, per instruction. Round 10's checkpoints are kept on disk but not selected."
        ),
    }
    (OUT_DIR / "round10-lumaxis-comparison.json").write_text(json.dumps(report, indent=2))

    print(f"\ntrain scenes used: {report['train_scenes_used']}")
    print(f"round9 union regressed ({len(round9_union)}): {sorted(round9_union)}")
    print(f"round10 union regressed ({len(round10_union)}): {sorted(round10_union)}")
    print(f"resolved vs round9: {sorted(resolved_vs_round9)}")
    print(f"persisted vs round9: {sorted(persisted_vs_round9)}")
    print(f"NEW regressions vs round9: {sorted(new_regressions_vs_round9)}")
    print(f"\nfewer_persistent_regressions: {fewer_persistent_regressions}")
    print(f"all_four_clearly_improved: {all_four_clearly_improved}")
    print(f"no_new_monitor_regressions: {no_new_monitor_regressions}")
    print(f"predeclared success met: {success}")
    print(f"SELECTED BEST MODEL: {selected_best}")
    print("\ncrop L1 (round9 -> round10), near-black/bright/std (round9 -> round10):")
    for name in PROBLEM_4:
        s = problem_4_status[name]
        print(f"  {name}: crop_l1 {s['crop_l1_round9']:.4f} -> {s['crop_l1_round10']:.4f}  clearly_improved={s['crop_l1_clearly_improved']}")
        print(f"      near_black {s['pred_near_black_fraction_round9']:.4f} -> {s['pred_near_black_fraction_round10']:.4f}   "
              f"bright {s['pred_bright_fraction_round9']:.4f} -> {s['pred_bright_fraction_round10']:.4f}   "
              f"luma_std {s['pred_luma_std_round9']:.4f} -> {s['pred_luma_std_round10']:.4f}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
