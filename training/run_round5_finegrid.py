"""Phase 4 round 5 - one targeted intervention testing the visual seam
diagnosis directly. Holds the 9-zone luma axis, loss (plain L1, no TV
regularization - already ruled out in round 2), and encoder (no detail
branch) fixed; only the bilateral grid's SPATIAL resolution changes,
8x8 -> 16x16. Same param count either way (104,940) - grid_spatial only
changes the head's output spatial size via adaptive pooling, not encoder
capacity, so this isolates grid resolution as the one variable.

Round 4's interpolation-sensitivity test only checked the luma axis; the
2026-09-16 visual inspection found artifacts (a dark ring, color-cast
blotches, desaturation) that all sit at local luminance/color
transitions - consistent with spatial grid cells too coarse to represent
a smooth transition. This tests that directly with two new metrics
alongside the usual monitor L1 / regression count:

  - crop-level L1: error specifically within each flagged scene's
    highlight-region crop (same location the visual inspection used),
    not just whole-frame L1 - did the specific defect region improve?
  - seam score: ratio of residual-error gradient magnitude AT grid-cell
    boundary lines vs the frame average - computed on |pred-target|, not
    raw pixels, so it isolates artifacts attributable to the grid itself
    from ordinary scene-content edges. >1 means error is elevated right
    at cell boundaries (a seam); ~1 means no boundary-specific elevation.
    Computed at each checkpoint's OWN grid resolution's boundary lines.

Same 3 seeds as every prior round. DSCF0878 reported separately, not
counted toward success. train + monitor only - finalHeldOut has no code
path here.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from dataset import CAL_DIR, Phase4PairDataset
from model import BilateralGridPredictor, apply_bilateral_grid, resize_for_network
from train import build_arg_parser, train_one_config

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"

BASE_CFG = {
    "grid_spatial": 16, "grid_luma": 9, "base_ch": 16, "lr": 1e-3,
    "epochs": 20, "lr_schedule": "cosine", "early_stopping_patience": 5,
    "grid_tv_weight": 0.0, "luma_mode": "rec709", "use_detail_branch": False,
}
SEEDS = [42, 43, 44]

REPRODUCIBLE_4 = ["DSCF0553.RAF", "DSCF0590.RAF", "DSCF0873.RAF", "DSCF0879.RAF"]
OUTLIER = "DSCF0878.RAF"
BASELINE_CHECKPOINTS = {
    42: "r2-b-lrsched-best.pt",
    43: "r2-b-lrsched-seed43-best.pt",
    44: "r2-b-lrsched-seed44-best.pt",
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
    model = BilateralGridPredictor(
        grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"],
        low_res=cfg["low_res"], base_ch=cfg["base_ch"],
        use_detail_branch=cfg.get("use_detail_branch", False), detail_ch=cfg.get("detail_ch", 8),
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
    """residual_hw: HxW error map (e.g. mean-abs residual over channels).
    Ratio of mean residual-gradient magnitude at grid-cell boundary lines
    vs the frame-wide average gradient. >1 = elevated error right at
    boundaries (a seam); ~1 = no boundary-specific elevation."""
    h, w = residual_hw.shape
    gx = np.abs(np.diff(residual_hw, axis=1))  # (h, w-1)
    gy = np.abs(np.diff(residual_hw, axis=0))  # (h-1, w)

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
                low_res_in = resize_for_network(full_res, cfg["low_res"])
                grid = model(low_res_in)
                pred = apply_bilateral_grid(grid, full_res, luma_mode=cfg.get("luma_mode", "rec709"))
                residual_hw = (pred - target).abs().mean(dim=1).squeeze(0).cpu().numpy()
            model_l1 = float(residual_hw.mean())
            entry = {"model_l1": model_l1, "regressed": model_l1 > baseline_l1}
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


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    results = {}
    for seed in SEEDS:
        cfg = dict(BASE_CFG)
        cfg["seed"] = seed
        cfg["run_name"] = f"r5-finegrid-seed{seed}"
        results[seed] = run_one(cfg)

    print("\n=== training done, evaluating both baseline and new checkpoints on every monitor scene ===")
    monitor_ds = Phase4PairDataset("monitor")

    diag = json.loads((OUT_DIR / "structural_diagnosis.json").read_text())
    centroid_by_name = {
        r["name"]: tuple(r["largest_centroid_norm"])
        for r in diag["regressing_scenes"]
        if r.get("largest_centroid_norm") and r["name"] in (REPRODUCIBLE_4 + [OUTLIER])
    }

    new_checkpoints = {s: f"r5-finegrid-seed{s}-best.pt" for s in SEEDS}
    baseline_per_scene = evaluate(BASELINE_CHECKPOINTS, device, monitor_ds, centroid_by_name)
    new_per_scene = evaluate(new_checkpoints, device, monitor_ds, centroid_by_name)

    baseline_union = union_regressed(baseline_per_scene)
    new_union = union_regressed(new_per_scene)
    resolved = baseline_union - new_union
    persisted = baseline_union & new_union
    new_regressions = new_union - baseline_union

    def seam_summary(per_scene, name):
        vals = [per_scene[name]["seeds"][s] for s in SEEDS]
        seam = np.mean([v["seam_score"] for v in vals])
        crop = np.mean([v.get("crop_l1", float("nan")) for v in vals])
        return {"mean_seam_score": float(seam), "mean_crop_l1": float(crop)}

    reproducible_4_status = {
        name: {
            "regressed_seeds_before": [s for s, v in baseline_per_scene[name]["seeds"].items() if v["regressed"]],
            "regressed_seeds_after": [s for s, v in new_per_scene[name]["seeds"].items() if v["regressed"]],
            "resolved": name in resolved,
            "seam_before": seam_summary(baseline_per_scene, name),
            "seam_after": seam_summary(new_per_scene, name),
        }
        for name in REPRODUCIBLE_4
    }
    outlier_status = {
        OUTLIER: {
            "regressed_seeds_before": [s for s, v in baseline_per_scene[OUTLIER]["seeds"].items() if v["regressed"]],
            "regressed_seeds_after": [s for s, v in new_per_scene[OUTLIER]["seeds"].items() if v["regressed"]],
            "seam_before": seam_summary(baseline_per_scene, OUTLIER),
            "seam_after": seam_summary(new_per_scene, OUTLIER),
            "note": "reported separately per instruction, not counted toward success",
        }
    }

    resolved_reproducible = [n for n in REPRODUCIBLE_4 if reproducible_4_status[n]["resolved"]]
    success = len(resolved_reproducible) > 0 and len(new_regressions) == 0

    report = {
        "intervention": "grid_spatial 8->16 (16x16x9), luma_mode rec709, no TV weight, no detail branch - grid resolution isolated as the only variable",
        "seeds": SEEDS,
        "baseline_union_regressed": sorted(baseline_union),
        "new_union_regressed": sorted(new_union),
        "resolved": sorted(resolved),
        "persisted": sorted(persisted),
        "new_regressions": sorted(new_regressions),
        "reproducible_4_status": reproducible_4_status,
        "outlier_status_dscf0878": outlier_status,
        "predeclared_success_met": success,
        "predeclared_success_criteria": "resolve >=1 of the 4 reproducible compact-highlight failures AND zero new persistent regressions (DSCF0878 excluded)",
    }
    (OUT_DIR / "round5-finegrid-comparison.json").write_text(json.dumps(report, indent=2))

    print(f"\nbaseline union regressed: {sorted(baseline_union)}")
    print(f"new (16x16x9 grid) union regressed: {sorted(new_union)}")
    print(f"resolved: {sorted(resolved)}")
    print(f"persisted: {sorted(persisted)}")
    print(f"NEW regressions introduced: {sorted(new_regressions)}")
    print(f"\npredeclared success met: {success}  (resolved reproducible: {resolved_reproducible})")
    print("\nseam scores (>1 = elevated error at grid-cell boundaries):")
    for name in REPRODUCIBLE_4 + [OUTLIER]:
        status = reproducible_4_status.get(name) or outlier_status.get(name)
        print(f"  {name}: seam {status['seam_before']['mean_seam_score']:.3f} -> {status['seam_after']['mean_seam_score']:.3f}   "
              f"crop_l1 {status['seam_before']['mean_crop_l1']:.4f} -> {status['seam_after']['mean_crop_l1']:.4f}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
