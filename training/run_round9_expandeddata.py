"""Phase 4 round 9 - isolated change is TRAIN-TIER COMPOSITION ONLY, per
instruction. Round 6's hybrid architecture/schedule/loss are otherwise
completely unchanged (rounds 7 and 8 already closed the capacity and
schedule branches without clearing round 6's bar). The only thing
different here is that `calibration-input/phase4-corpus-manifest.json`'s
train tier now has 599 scenes instead of 561 - the 38 capture-batch-2
scenes (2026-11-12), approved into train only, with monitor/finalHeldOut
left untouched. This tests whether the model improves on the known hard
patterns simply from seeing real examples of them during fitting, without
any model-side change.

No code change was needed to pick up the expanded train set - dataset.py
reads it straight from the manifest. precompute_targets.py was run first
to cache the 38 new scenes' target .npy files (a mechanical I/O
prerequisite, not part of the training loop itself).

Predeclared success (checked before this run, verified after): fewer
persistent union-of-seed regressions on monitor than round 6's 5-scene
union, AND no loss of round 6's crop-level gains on the 4 reproducible
scenes. DSCF0878 reported separately, not counted toward success -
but its crop L1 is also tracked as a secondary, non-blocking signal,
since capture batch 2's strongest contribution (DSCF1413-1416) is more
extreme examples of exactly DSCF0878's failure pattern (broad,
low-compactness, near-clipped highlights).

Per instruction: retain round 9 as the selected-best checkpoint set only
if it clears the predeclared bar; otherwise round 6 remains selected-best.
Both checkpoint sets stay on disk regardless - this script only reports
the decision, it doesn't delete anything.

train + monitor only - finalHeldOut has no code path here.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from dataset import CAL_DIR, Phase4PairDataset
from model import BilateralGridPredictor, HybridPredictor
from train import build_arg_parser, train_one_config

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"

BASE_CFG = {
    "grid_spatial": 8, "grid_luma": 9, "base_ch": 16, "lr": 1e-3,
    "epochs": 20, "lr_schedule": "cosine", "early_stopping_patience": 5,
    "grid_tv_weight": 0.0, "luma_mode": "rec709",
    "use_hybrid": True, "refinement_base_ch": 8, "refinement_max_delta": 0.08,
}
SEEDS = [42, 43, 44]

REPRODUCIBLE_4 = ["DSCF0553.RAF", "DSCF0590.RAF", "DSCF0873.RAF", "DSCF0879.RAF"]
OUTLIER = "DSCF0878.RAF"
BASELINE_CHECKPOINTS = {
    42: "r2-b-lrsched-best.pt",
    43: "r2-b-lrsched-seed43-best.pt",
    44: "r2-b-lrsched-seed44-best.pt",
}
ROUND6_CHECKPOINTS = {
    42: "r6-hybrid-seed42-best.pt",
    43: "r6-hybrid-seed43-best.pt",
    44: "r6-hybrid-seed44-best.pt",
}


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


def seam_score(residual_hw, grid_resolution, band=1):
    h, w = residual_hw.shape
    gx = np.abs(np.diff(residual_hw, axis=1))
    gy = np.abs(np.diff(residual_hw, axis=0))
    boundary_cols = [round(i * w / grid_resolution) for i in range(1, grid_resolution)]
    boundary_rows = [round(i * h / grid_resolution) for i in range(1, grid_resolution)]
    boundary_gx = []
    for c in boundary_cols:
        c0, c1 = max(0, c - band), min(w - 1, c + band)
        if c1 > c0:
            boundary_gx.append(gx[:, c0:c1].mean())
    boundary_gy = []
    for r in boundary_rows:
        r0, r1 = max(0, r - band), min(h - 1, r + band)
        if r1 > r0:
            boundary_gy.append(gy[r0:r1, :].mean())
    boundary_grad = float(np.mean(boundary_gx + boundary_gy)) if (boundary_gx or boundary_gy) else 0.0
    interior_grad = float((gx.mean() + gy.mean()) / 2)
    return boundary_grad / max(interior_grad, 1e-8)


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
            if name in REPRODUCIBLE_4 or name == OUTLIER:
                entry["seam_score"] = seam_score(residual_hw, cfg["grid_spatial"])
                if name in centroid_by_name:
                    cx, cy = centroid_by_name[name]
                    crop = crop_around(residual_hw, cx, cy)
                    entry["crop_l1"] = float(crop.mean())
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
        cfg["run_name"] = f"r9-expandeddata-seed{seed}"
        results[seed] = run_one(cfg)
        print(f"[r9-expandeddata-seed{seed}] train scenes used: {results[seed]['train_scenes']}")

    print("\n=== training done, evaluating frozen baseline, round 6 hybrid, and round 9 expanded-data hybrid ===")
    monitor_ds = Phase4PairDataset("monitor")

    diag = json.loads((OUT_DIR / "structural_diagnosis.json").read_text())
    centroid_by_name = {
        r["name"]: tuple(r["largest_centroid_norm"])
        for r in diag["regressing_scenes"]
        if r.get("largest_centroid_norm") and r["name"] in (REPRODUCIBLE_4 + [OUTLIER])
    }

    new_checkpoints = {s: f"r9-expandeddata-seed{s}-best.pt" for s in SEEDS}
    baseline_per_scene = evaluate(BASELINE_CHECKPOINTS, device, monitor_ds, centroid_by_name)
    round6_per_scene = evaluate(ROUND6_CHECKPOINTS, device, monitor_ds, centroid_by_name)
    new_per_scene = evaluate(new_checkpoints, device, monitor_ds, centroid_by_name)

    baseline_union = union_regressed(baseline_per_scene)
    round6_union = union_regressed(round6_per_scene)
    new_union = union_regressed(new_per_scene)

    # primary comparison target is round 6, on the UNCHANGED monitor tier -
    # per instruction, this stays comparable to every round 2-8 number
    # already recorded.
    resolved_vs_round6 = round6_union - new_union
    persisted_vs_round6 = round6_union & new_union
    new_regressions_vs_round6 = new_union - round6_union

    def status_for(name):
        crop_baseline = mean_metric(baseline_per_scene, name, "crop_l1")
        crop_round6 = mean_metric(round6_per_scene, name, "crop_l1")
        crop_round9 = mean_metric(new_per_scene, name, "crop_l1")
        return {
            "regressed_seeds_baseline": [s for s, v in baseline_per_scene[name]["seeds"].items() if v["regressed"]],
            "regressed_seeds_round6": [s for s, v in round6_per_scene[name]["seeds"].items() if v["regressed"]],
            "regressed_seeds_round9": [s for s, v in new_per_scene[name]["seeds"].items() if v["regressed"]],
            "crop_l1_baseline": crop_baseline,
            "crop_l1_round6": crop_round6,
            "crop_l1_round9": crop_round9,
            "crop_l1_lost_gain_vs_round6": crop_round9 > crop_round6,
            "crop_l1_improved_vs_round6": crop_round9 < crop_round6,
            "mean_abs_delta_round9": mean_metric(new_per_scene, name, "mean_abs_delta"),
        }

    reproducible_4_status = {name: status_for(name) for name in REPRODUCIBLE_4}
    outlier_status = {OUTLIER: {**status_for(OUTLIER), "note": "reported separately per instruction, not counted toward success - tracked as a secondary, non-blocking signal given capture batch 2's composition"}}

    fewer_persistent_regressions = len(new_union) < len(round6_union)
    no_lost_crop_gains = not any(reproducible_4_status[n]["crop_l1_lost_gain_vs_round6"] for n in REPRODUCIBLE_4)
    success = fewer_persistent_regressions and no_lost_crop_gains
    selected_best = "round9" if success else "round6"

    report = {
        "intervention": "Train-tier composition only (561 -> 599 scenes, capture batch 2 approved into train) - round 6's hybrid architecture/schedule/loss completely unchanged, same 3 seeds. Comparison target is round 6 on the unchanged 76-scene monitor tier.",
        "seeds": SEEDS,
        "train_scenes_used": results[SEEDS[0]]["train_scenes"],
        "baseline_union_regressed": sorted(baseline_union),
        "round6_union_regressed": sorted(round6_union),
        "round9_union_regressed": sorted(new_union),
        "resolved_vs_round6": sorted(resolved_vs_round6),
        "persisted_vs_round6": sorted(persisted_vs_round6),
        "new_regressions_vs_round6": sorted(new_regressions_vs_round6),
        "reproducible_4_status": reproducible_4_status,
        "outlier_status_dscf0878": outlier_status,
        "predeclared_success_met": success,
        "predeclared_success_criteria": "fewer persistent union-of-seed regressions than round 6 AND no loss of round 6's crop-level gains on any of the 4 reproducible scenes (DSCF0878 excluded, tracked separately)",
        "fewer_persistent_regressions": fewer_persistent_regressions,
        "no_lost_crop_gains": no_lost_crop_gains,
        "selected_best_model": selected_best,
        "selection_note": (
            "Round 9 cleared the predeclared bar and is the new selected-best model."
            if success else
            "Round 9 did not clear the predeclared bar - Round 6 remains the selected-best model, per instruction. Round 9's checkpoints are kept on disk but not selected."
        ),
    }
    (OUT_DIR / "round9-expandeddata-comparison.json").write_text(json.dumps(report, indent=2))

    print(f"\ntrain scenes used: {report['train_scenes_used']}")
    print(f"baseline union regressed ({len(baseline_union)}): {sorted(baseline_union)}")
    print(f"round6 union regressed ({len(round6_union)}): {sorted(round6_union)}")
    print(f"round9 union regressed ({len(new_union)}): {sorted(new_union)}")
    print(f"resolved vs round6: {sorted(resolved_vs_round6)}")
    print(f"persisted vs round6: {sorted(persisted_vs_round6)}")
    print(f"NEW regressions vs round6: {sorted(new_regressions_vs_round6)}")
    print(f"\nfewer_persistent_regressions: {fewer_persistent_regressions}")
    print(f"no_lost_crop_gains: {no_lost_crop_gains}")
    print(f"predeclared success met: {success}")
    print(f"SELECTED BEST MODEL: {selected_best}")
    print("\ncrop L1 (baseline -> round6 -> round9):")
    for name in REPRODUCIBLE_4 + [OUTLIER]:
        s = reproducible_4_status.get(name) or outlier_status.get(name)
        print(f"  {name}: {s['crop_l1_baseline']:.4f} -> {s['crop_l1_round6']:.4f} -> {s['crop_l1_round9']:.4f}   "
              f"lost_gain={s['crop_l1_lost_gain_vs_round6']}   improved={s['crop_l1_improved_vs_round6']}   "
              f"mean|delta|={s['mean_abs_delta_round9']:.4f}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
