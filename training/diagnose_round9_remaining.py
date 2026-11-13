"""Phase 4 round 9 - focused before/after diagnosis of the scenes still
regressed after round 9, per instruction: "produce a focused before/after
diagnosis of the remaining unresolved monitor scenes, including what
Round 9 improved, what remains wrong, and one evidence-based next
intervention." Diagnostic-only - no training, no checkpoint selection,
no finalHeldOut access (monitor tier only, same as every prior round-9
artifact).

Scope: the 4 scenes still in round 9's union-of-seed regression set
(training/diagnostics/round9-expandeddata-comparison.json ->
round9_union_regressed): DSCF0590, DSCF0873, DSCF0878, DSCF0879.
DSCF0553 is excluded - round 9 resolved it, nothing to diagnose.

For each scene, for round 6 and round 9 (mean over seeds 42/43/44):
  - crop_l1 (already known from the round 9 comparison, repeated here
    for context)
  - residual connected-component shape (highlight_shape_stats, reused
    from diagnose_structural.py) run on the ERROR map itself (top 0.5%
    of |pred-target|, not on brightness) - characterizes whether the
    worst error is still concentrated in one place, and whether that
    place moved, shrank, or changed shape between round 6 and round 9.
  - distance from the error centroid to the scene's known highlight
    centroid (structural_diagnosis.json's largest_centroid_norm) - tests
    whether round 9 is still failing in the same structural location or
    has shifted the failure elsewhere.
  - predicted (post-correction) luma distribution stats vs the target's,
    to see whether round 9 under/over-shoots brightness in the same
    direction round 6 did, or has changed character.
"""
import json
from pathlib import Path

import numpy as np
import torch

from dataset import Phase4PairDataset
from diagnose_structural import highlight_shape_stats, luma_distribution_stats
from model import BilateralGridPredictor, HybridPredictor, compute_luma_guide

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"

SEEDS = [42, 43, 44]
ROUND6_CHECKPOINTS = {42: "r6-hybrid-seed42-best.pt", 43: "r6-hybrid-seed43-best.pt", 44: "r6-hybrid-seed44-best.pt"}
ROUND9_CHECKPOINTS = {42: "r9-expandeddata-seed42-best.pt", 43: "r9-expandeddata-seed43-best.pt", 44: "r9-expandeddata-seed44-best.pt"}

REMAINING = ["DSCF0590.RAF", "DSCF0873.RAF", "DSCF0878.RAF", "DSCF0879.RAF"]


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


def error_shape_stats(pred_hw3, target_hw3, residual_hw):
    """Connected-component analysis on the brightest 0.5% of the ERROR map
    (reusing highlight_shape_stats's thresholding logic, but feeding it
    the residual instead of luma - characterizes the worst-error region's
    shape/location, independent of where the scene's own highlights are)."""
    return highlight_shape_stats(pred_hw3, residual_hw)


def centroid_dist(c1, c2):
    if c1 is None or c2 is None:
        return None
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def evaluate_scene(name, checkpoints_by_seed, device, scene_item, known_highlight_centroid):
    full_res = scene_item["input"].unsqueeze(0).to(device)
    target = scene_item["target"].unsqueeze(0).to(device)
    target_luma = compute_luma_guide(target, "rec709").squeeze(0).squeeze(0).cpu().numpy()
    target_hw3 = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_dist = luma_distribution_stats(target_luma)

    per_seed = []
    for seed, ckpt_name in checkpoints_by_seed.items():
        model, cfg = load_model_from_ckpt(ckpt_name, device)
        with torch.no_grad():
            pred, _ = model.predict_and_apply(full_res, cfg["low_res"], cfg.get("luma_mode", "rec709"))
        pred_hw3 = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        residual_hw = (pred - target).abs().mean(dim=1).squeeze(0).cpu().numpy()
        crop_l1 = float(residual_hw.mean())

        err_shape = error_shape_stats(pred_hw3, target_hw3, residual_hw)
        pred_luma = compute_luma_guide(pred, "rec709").squeeze(0).squeeze(0).cpu().numpy()
        pred_dist = luma_distribution_stats(pred_luma)

        per_seed.append({
            "seed": seed,
            "mean_abs_residual": crop_l1,
            "error_region_area_fraction": err_shape["largest_area_fraction"],
            "error_region_compactness": err_shape["largest_compactness"],
            "error_region_centroid_norm": err_shape["largest_centroid_norm"],
            "error_to_known_highlight_dist": centroid_dist(err_shape["largest_centroid_norm"], known_highlight_centroid),
            "pred_near_black_fraction": pred_dist["near_black_fraction"],
            "pred_bright_fraction": pred_dist["bright_fraction"],
            "pred_luma_std": pred_dist["luma_std"],
        })

    def mean_of(key):
        vals = [p[key] for p in per_seed if p[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "per_seed": per_seed,
        "mean_abs_residual": mean_of("mean_abs_residual"),
        "error_region_area_fraction": mean_of("error_region_area_fraction"),
        "error_region_compactness": mean_of("error_region_compactness"),
        "error_to_known_highlight_dist": mean_of("error_to_known_highlight_dist"),
        "pred_near_black_fraction": mean_of("pred_near_black_fraction"),
        "pred_bright_fraction": mean_of("pred_bright_fraction"),
        "pred_luma_std": mean_of("pred_luma_std"),
        "target_near_black_fraction": target_dist["near_black_fraction"],
        "target_bright_fraction": target_dist["bright_fraction"],
        "target_luma_std": target_dist["luma_std"],
    }


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    monitor_ds = Phase4PairDataset("monitor")
    by_name = {monitor_ds[i]["name"]: monitor_ds[i] for i in range(len(monitor_ds)) if monitor_ds[i]["name"] in REMAINING}

    struct_diag = json.loads((OUT_DIR / "structural_diagnosis.json").read_text())
    known_centroid_by_name = {r["name"]: tuple(r["largest_centroid_norm"]) for r in struct_diag["regressing_scenes"] if r.get("largest_centroid_norm")}

    report = {}
    for name in REMAINING:
        item = by_name[name]
        known_c = known_centroid_by_name.get(name)
        round6 = evaluate_scene(name, ROUND6_CHECKPOINTS, device, item, known_c)
        round9 = evaluate_scene(name, ROUND9_CHECKPOINTS, device, item, known_c)
        report[name] = {
            "known_highlight_centroid_norm": known_c,
            "round6": round6,
            "round9": round9,
            "residual_delta_round9_minus_round6": round9["mean_abs_residual"] - round6["mean_abs_residual"],
            "error_region_moved_closer_to_known_highlight": (
                (round9["error_to_known_highlight_dist"] < round6["error_to_known_highlight_dist"])
                if round9["error_to_known_highlight_dist"] is not None and round6["error_to_known_highlight_dist"] is not None
                else None
            ),
        }

    (OUT_DIR / "round9-remaining-diagnosis.json").write_text(json.dumps(report, indent=2))

    print("\n=== Round 9 remaining-regression diagnosis (round6 -> round9) ===")
    for name, r in report.items():
        r6, r9 = r["round6"], r["round9"]
        print(f"\n{name}:")
        print(f"  mean_abs_residual:        {r6['mean_abs_residual']:.4f} -> {r9['mean_abs_residual']:.4f}  (delta {r['residual_delta_round9_minus_round6']:+.4f})")
        print(f"  error_region_area_frac:   {r6['error_region_area_fraction']:.5f} -> {r9['error_region_area_fraction']:.5f}")
        print(f"  error_region_compactness: {r6['error_region_compactness']} -> {r9['error_region_compactness']}")
        print(f"  error_to_known_highlight: {r6['error_to_known_highlight_dist']} -> {r9['error_to_known_highlight_dist']}")
        print(f"  pred_near_black_fraction: {r6['pred_near_black_fraction']:.4f} -> {r9['pred_near_black_fraction']:.4f}  (target {r6['target_near_black_fraction']:.4f})")
        print(f"  pred_bright_fraction:     {r6['pred_bright_fraction']:.4f} -> {r9['pred_bright_fraction']:.4f}  (target {r6['target_bright_fraction']:.4f})")
        print(f"  pred_luma_std:            {r6['pred_luma_std']:.4f} -> {r9['pred_luma_std']:.4f}  (target {r6['target_luma_std']:.4f})")

    print("\nfinalHeldOut was not accessed by this diagnostic pass.")


if __name__ == "__main__":
    main()
