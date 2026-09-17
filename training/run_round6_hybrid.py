"""Phase 4 round 6 - the hybrid architecture (grid + bounded, identity-safe
RefinementNet), full 3-seed train/monitor round. Round 5 closed the
bilateral-grid-only tuning branch; this is a genuinely different
component per instruction, now cleared through Codex's full export gate
(dynamic-shape ONNX export, ONNX checker, native ONNX Runtime parity to
2.09e-6, and a real-browser ONNX Runtime Web WASM/WebGPU run, all on the
exact single-input preprocessing this trains under).

Same architecture as the approved baseline (grid 8x8x9, base_ch=16,
cosine LR schedule + early stopping, epochs cap 20) plus
use_hybrid=True (refinement_base_ch=8, refinement_max_delta=0.08).
Same 3 seeds, same evaluation code, same crop-level L1 / seam-score
metrics from round 5 - this is the real test of whether a bounded local
correction succeeds where blind grid tuning didn't.

DSCF0878 reported separately, not counted toward success. train +
monitor only - finalHeldOut has no code path here.
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


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    results = {}
    for seed in SEEDS:
        cfg = dict(BASE_CFG)
        cfg["seed"] = seed
        cfg["run_name"] = f"r6-hybrid-seed{seed}"
        results[seed] = run_one(cfg)

    print("\n=== training done, evaluating both baseline and hybrid checkpoints on every monitor scene ===")
    monitor_ds = Phase4PairDataset("monitor")

    diag = json.loads((OUT_DIR / "structural_diagnosis.json").read_text())
    centroid_by_name = {
        r["name"]: tuple(r["largest_centroid_norm"])
        for r in diag["regressing_scenes"]
        if r.get("largest_centroid_norm") and r["name"] in (REPRODUCIBLE_4 + [OUTLIER])
    }

    new_checkpoints = {s: f"r6-hybrid-seed{s}-best.pt" for s in SEEDS}
    baseline_per_scene = evaluate(BASELINE_CHECKPOINTS, device, monitor_ds, centroid_by_name)
    new_per_scene = evaluate(new_checkpoints, device, monitor_ds, centroid_by_name)

    baseline_union = union_regressed(baseline_per_scene)
    new_union = union_regressed(new_per_scene)
    resolved = baseline_union - new_union
    persisted = baseline_union & new_union
    new_regressions = new_union - baseline_union

    def seam_summary(per_scene, name):
        vals = [per_scene[name]["seeds"][s] for s in SEEDS]
        seam = float(np.mean([v["seam_score"] for v in vals]))
        crop = float(np.mean([v.get("crop_l1", float("nan")) for v in vals]))
        delta = float(np.mean([v.get("mean_abs_delta", float("nan")) for v in vals]))
        return {"mean_seam_score": seam, "mean_crop_l1": crop, "mean_abs_delta": delta}

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
        "intervention": "HybridPredictor: grid (unchanged, 8x8x9) + bounded identity-safe RefinementNet, trained jointly end-to-end, export-safe preprocessing (Codex-verified)",
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
    (OUT_DIR / "round6-hybrid-comparison.json").write_text(json.dumps(report, indent=2))

    print(f"\nbaseline union regressed: {sorted(baseline_union)}")
    print(f"new (hybrid) union regressed: {sorted(new_union)}")
    print(f"resolved: {sorted(resolved)}")
    print(f"persisted: {sorted(persisted)}")
    print(f"NEW regressions introduced: {sorted(new_regressions)}")
    print(f"\npredeclared success met: {success}  (resolved reproducible: {resolved_reproducible})")
    print("\nseam scores / crop L1 / mean |delta| (before -> after):")
    for name in REPRODUCIBLE_4 + [OUTLIER]:
        status = reproducible_4_status.get(name) or outlier_status.get(name)
        print(f"  {name}: seam {status['seam_before']['mean_seam_score']:.3f} -> {status['seam_after']['mean_seam_score']:.3f}   "
              f"crop_l1 {status['seam_before']['mean_crop_l1']:.4f} -> {status['seam_after']['mean_crop_l1']:.4f}   "
              f"mean|delta| after={status['seam_after']['mean_abs_delta']:.4f}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
