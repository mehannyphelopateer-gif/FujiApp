"""Phase 4 round 7 - one controlled follow-up to round 6's hybrid result,
per instruction: round 6 was the first architecture with a real positive
signal (crop-level L1 improved on 4/5 flagged scenes, DSCF0553 moved from
regressing under 2/3 seeds to 1/3), so continue on the hybrid path rather
than treat it as inconclusive. Only the refinement branch's capacity
changes - refinement_base_ch 8 -> 16, doubling it. Residual bound stays
at 0.08 (round 6's mean |delta| was ~0.01-0.02, nowhere near saturating,
so widening the bound isn't justified yet) and the training
schedule/loss are otherwise identical to round 6.

Predeclared success, per instruction: FEWER persistent union-of-seed
regressions AND no loss of round 6's crop-level gains - both measured
against round 6's own hybrid result (refinement_base_ch=8), not just the
original frozen grid-only baseline, since the point is whether more
capacity improves on what round 6 already achieved. If this doesn't clear
that bar, the instruction is explicit: evaluate schedule/loss separately
next, don't combine changes.

Same 3 seeds. DSCF0878 reported separately, not counted toward success.
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
    "use_hybrid": True, "refinement_base_ch": 16, "refinement_max_delta": 0.08,
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
        cfg["run_name"] = f"r7-hybridcap16-seed{seed}"
        results[seed] = run_one(cfg)

    print("\n=== training done, evaluating frozen baseline, round 6 hybrid, and round 7 hybrid-cap16 ===")
    monitor_ds = Phase4PairDataset("monitor")

    diag = json.loads((OUT_DIR / "structural_diagnosis.json").read_text())
    centroid_by_name = {
        r["name"]: tuple(r["largest_centroid_norm"])
        for r in diag["regressing_scenes"]
        if r.get("largest_centroid_norm") and r["name"] in (REPRODUCIBLE_4 + [OUTLIER])
    }

    new_checkpoints = {s: f"r7-hybridcap16-seed{s}-best.pt" for s in SEEDS}
    baseline_per_scene = evaluate(BASELINE_CHECKPOINTS, device, monitor_ds, centroid_by_name)
    round6_per_scene = evaluate(ROUND6_CHECKPOINTS, device, monitor_ds, centroid_by_name)
    new_per_scene = evaluate(new_checkpoints, device, monitor_ds, centroid_by_name)

    baseline_union = union_regressed(baseline_per_scene)
    round6_union = union_regressed(round6_per_scene)
    new_union = union_regressed(new_per_scene)

    # primary comparison target is round 6, per instruction - "no loss of
    # the crop-level gains" only makes sense relative to what round 6
    # actually achieved, not the pre-hybrid baseline.
    resolved_vs_round6 = round6_union - new_union
    persisted_vs_round6 = round6_union & new_union
    new_regressions_vs_round6 = new_union - round6_union

    def status_for(name):
        crop_baseline = mean_metric(baseline_per_scene, name, "crop_l1")
        crop_round6 = mean_metric(round6_per_scene, name, "crop_l1")
        crop_round7 = mean_metric(new_per_scene, name, "crop_l1")
        return {
            "regressed_seeds_baseline": [s for s, v in baseline_per_scene[name]["seeds"].items() if v["regressed"]],
            "regressed_seeds_round6": [s for s, v in round6_per_scene[name]["seeds"].items() if v["regressed"]],
            "regressed_seeds_round7": [s for s, v in new_per_scene[name]["seeds"].items() if v["regressed"]],
            "crop_l1_baseline": crop_baseline,
            "crop_l1_round6": crop_round6,
            "crop_l1_round7": crop_round7,
            "crop_l1_lost_gain_vs_round6": crop_round7 > crop_round6,
            "mean_abs_delta_round7": mean_metric(new_per_scene, name, "mean_abs_delta"),
        }

    reproducible_4_status = {name: status_for(name) for name in REPRODUCIBLE_4}
    outlier_status = {OUTLIER: {**status_for(OUTLIER), "note": "reported separately per instruction, not counted toward success"}}

    fewer_persistent_regressions = len(new_union) < len(round6_union)
    no_lost_crop_gains = not any(reproducible_4_status[n]["crop_l1_lost_gain_vs_round6"] for n in REPRODUCIBLE_4)
    success = fewer_persistent_regressions and no_lost_crop_gains

    report = {
        "intervention": "HybridPredictor with refinement_base_ch 8->16 (doubled), max_delta unchanged at 0.08, schedule/loss unchanged - comparison target is round 6's own hybrid result, not just the pre-hybrid baseline",
        "seeds": SEEDS,
        "baseline_union_regressed": sorted(baseline_union),
        "round6_union_regressed": sorted(round6_union),
        "round7_union_regressed": sorted(new_union),
        "resolved_vs_round6": sorted(resolved_vs_round6),
        "persisted_vs_round6": sorted(persisted_vs_round6),
        "new_regressions_vs_round6": sorted(new_regressions_vs_round6),
        "reproducible_4_status": reproducible_4_status,
        "outlier_status_dscf0878": outlier_status,
        "predeclared_success_met": success,
        "predeclared_success_criteria": "fewer persistent union-of-seed regressions than round 6 AND no loss of round 6's crop-level gains on any of the 4 reproducible scenes (DSCF0878 excluded)",
        "fewer_persistent_regressions": fewer_persistent_regressions,
        "no_lost_crop_gains": no_lost_crop_gains,
    }
    (OUT_DIR / "round7-hybridcapacity-comparison.json").write_text(json.dumps(report, indent=2))

    print(f"\nbaseline union regressed ({len(baseline_union)}): {sorted(baseline_union)}")
    print(f"round6 union regressed ({len(round6_union)}): {sorted(round6_union)}")
    print(f"round7 union regressed ({len(new_union)}): {sorted(new_union)}")
    print(f"resolved vs round6: {sorted(resolved_vs_round6)}")
    print(f"persisted vs round6: {sorted(persisted_vs_round6)}")
    print(f"NEW regressions vs round6: {sorted(new_regressions_vs_round6)}")
    print(f"\nfewer_persistent_regressions: {fewer_persistent_regressions}")
    print(f"no_lost_crop_gains: {no_lost_crop_gains}")
    print(f"predeclared success met: {success}")
    print("\ncrop L1 (baseline -> round6 -> round7):")
    for name in REPRODUCIBLE_4 + [OUTLIER]:
        s = reproducible_4_status.get(name) or outlier_status.get(name)
        print(f"  {name}: {s['crop_l1_baseline']:.4f} -> {s['crop_l1_round6']:.4f} -> {s['crop_l1_round7']:.4f}   "
              f"lost_gain={s['crop_l1_lost_gain_vs_round6']}   mean|delta|={s['mean_abs_delta_round7']:.4f}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
