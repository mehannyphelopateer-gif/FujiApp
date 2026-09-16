"""Phase 4 structural/spatial diagnostic - round 3's WB-stable luma guide
falsified the WB-axis hypothesis (exact same 5 scenes regressed, exact
same seeds, zero change). Per instruction, this looks past the WB/
highlight metadata correlation (real but not causal) at what's actually
different about these 5 scenes' *content*, and separately at whether the
error comes from the grid PREDICTION (the CNN encoder's coefficients
being wrong for this scene) or its SLICING/APPLICATION (coarse trilinear
interpolation smearing a transition the grid can't represent at this
resolution).

Compares the 5 regressing scenes (DSCF0553, DSCF0590, DSCF0873, DSCF0879,
DSCF0878) against 21 matched controls - every OTHER monitor scene in the
same bright-highlight/mixed-lighting categories that never regressed
under any seed - on:
  1. highlight region shape (connected-component analysis on the
     brightest pixels: count, size, aspect ratio, compactness, location)
  2. local luminance distribution shape (histogram, dark+bright
     bimodality)
  3. prediction-vs-slicing split: compares the model's actual trilinear-
     sliced output against a nearest-neighbor-sliced version of the SAME
     predicted grid (grid_sample's 5D nearest mode isn't implemented on
     MPS, so this whole script runs on CPU - it's inference-only on ~26
     scenes, fast enough either way) - if switching interpolation mode
     changes the result a lot for a scene, the grid is rapidly varying
     where that scene's pixels land (a slicing/resolution problem); if it
     barely changes, the grid is locally smooth there and the predicted
     coefficients themselves must be wrong (a prediction problem).

DSCF0878 is kept separate throughout per instruction - it didn't fit the
highlight-concentration pattern of the other 4 in the first diagnosis.

train + monitor only - finalHeldOut has no code path here.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import CAL_DIR, Phase4PairDataset
from model import BilateralGridPredictor, apply_bilateral_grid, compute_luma_guide, resize_for_network

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"

CHECKPOINT = "r2-b-lrsched-seed43-best.pt"  # best-performing seed, consistent with the first diagnosis
REGRESSING = ["DSCF0553.RAF", "DSCF0590.RAF", "DSCF0873.RAF", "DSCF0879.RAF", "DSCF0878.RAF"]
OUTLIER = "DSCF0878.RAF"


def load_model(device):
    ckpt = torch.load(CKPT_DIR / CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BilateralGridPredictor(
        grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"],
        low_res=cfg["low_res"], base_ch=cfg["base_ch"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def highlight_shape_stats(full_res_np, luma_np):
    """full_res_np: HxWx3, luma_np: HxW, both numpy. Connected-component
    analysis on the brightest 0.5% of pixels."""
    h, w = luma_np.shape
    thresh = np.percentile(luma_np, 99.5)
    mask = (luma_np >= thresh).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:  # only background
        return {"n_components": 0, "largest_area_fraction": 0.0, "largest_bbox_aspect": None,
                "largest_compactness": None, "largest_centroid_norm": None}
    # stats[0] is background; find largest real component
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    area = int(stats[idx, cv2.CC_STAT_AREA])
    bw = int(stats[idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
    cy, cx = float(centroids[idx][1]), float(centroids[idx][0])
    return {
        "n_components": int(n_labels - 1),
        "largest_area_fraction": float(area / (h * w)),
        "largest_bbox_aspect": float(bw / bh) if bh > 0 else None,
        "largest_compactness": float(area / (bw * bh)) if bw > 0 and bh > 0 else None,
        "largest_centroid_norm": [float(cx / w), float(cy / h)],
    }


def luma_distribution_stats(luma_np):
    hist, _ = np.histogram(luma_np, bins=32, range=(0, 1), density=True)
    near_black_frac = float((luma_np < 0.05).mean())
    bright_frac = float((luma_np > 0.7).mean())
    mid_frac = 1.0 - near_black_frac - bright_frac
    return {
        "near_black_fraction": near_black_frac,
        "bright_fraction": bright_frac,
        "mid_fraction": mid_frac,
        "dark_and_bright_bimodal_score": near_black_frac * bright_frac,  # high only if BOTH modes are substantial
        "luma_std": float(luma_np.std()),
    }


def prediction_vs_slicing(model, cfg, full_res, device):
    """Returns (trilinear_output, nearest_output, interpolation_sensitivity_l1)."""
    with torch.no_grad():
        low_res_in = resize_for_network(full_res, cfg["low_res"])
        grid = model(low_res_in)

        b, c, h, w = full_res.shape
        luma = compute_luma_guide(full_res, cfg.get("luma_mode", "rec709"))
        ys = torch.linspace(-1, 1, h, device=device)
        xs = torch.linspace(-1, 1, w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.unsqueeze(0).expand(b, h, w)
        gy = gy.unsqueeze(0).expand(b, h, w)
        gz = (luma.clamp(0, 1) * 2 - 1)
        coords = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1)

        def apply(mode):
            sampled = F.grid_sample(grid, coords, mode=mode, padding_mode="border", align_corners=True)
            coeffs = sampled.squeeze(2).view(b, 3, 4, h, w)
            rgb = full_res
            return (coeffs[:, :, 0] * rgb[:, 0:1] + coeffs[:, :, 1] * rgb[:, 1:2]
                    + coeffs[:, :, 2] * rgb[:, 2:3] + coeffs[:, :, 3])

        tri = apply("bilinear")
        near = apply("nearest")
        sensitivity = (tri - near).abs().mean().item()
    return tri, near, sensitivity


def main():
    device = torch.device("cpu")  # 5D nearest grid_sample isn't implemented on MPS
    print(f"device: {device} (deliberate - see module docstring)")

    model, cfg = load_model(device)
    monitor_ds = Phase4PairDataset("monitor")
    by_name = {monitor_ds[i]["name"]: i for i in range(len(monitor_ds))}

    manifest = json.loads((CAL_DIR / "phase4-corpus-manifest.json").read_text())
    meta_by_name = {s["name"]: s for s in manifest["tiers"]["monitor"]["scenes"]}
    controls = [s["name"] for s in manifest["tiers"]["monitor"]["scenes"]
                if s.get("highlightClass") in ("bright-highlight", "mixed-lighting") and s["name"] not in REGRESSING]
    print(f"matched controls: {len(controls)}")

    def analyze(name):
        idx = by_name[name]
        item = monitor_ds[idx]
        full_res = item["input"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        full_res_np = full_res.squeeze(0).permute(1, 2, 0).numpy()
        luma_np = compute_luma_guide(full_res, cfg.get("luma_mode", "rec709")).squeeze(0).numpy()

        shape_stats = highlight_shape_stats(full_res_np, luma_np)
        dist_stats = luma_distribution_stats(luma_np)
        tri, near, sensitivity = prediction_vs_slicing(model, cfg, full_res, device)
        baseline_l1 = (full_res - target).abs().mean().item()
        tri_l1 = (tri - target).abs().mean().item()

        return {
            "name": name,
            "baseline_l1": baseline_l1,
            "model_l1_trilinear": tri_l1,
            "regressed": tri_l1 > baseline_l1,
            "interpolation_sensitivity": sensitivity,  # high = slicing-sensitive, low = prediction-driven
            **shape_stats,
            **dist_stats,
        }

    regressing_results = [analyze(n) for n in REGRESSING]
    control_results = [analyze(n) for n in controls]

    def agg(results, key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    non_outlier_regressing = [r for r in regressing_results if r["name"] != OUTLIER]
    outlier_result = next(r for r in regressing_results if r["name"] == OUTLIER)

    summary = {}
    for key in ["n_components", "largest_area_fraction", "largest_bbox_aspect", "largest_compactness",
                "near_black_fraction", "bright_fraction", "dark_and_bright_bimodal_score", "luma_std",
                "interpolation_sensitivity"]:
        summary[key] = {
            "regressing_4_mean": agg(non_outlier_regressing, key),
            "dscf0878_outlier": outlier_result.get(key),
            "control_mean": agg(control_results, key),
        }

    report = {
        "checkpoint": CHECKPOINT,
        "regressing_scenes": regressing_results,
        "control_scenes": control_results,
        "summary_comparison": summary,
    }
    (OUT_DIR / "structural_diagnosis.json").write_text(json.dumps(report, indent=2))

    print("\n=== summary: regressing-4 (excl. DSCF0878) vs control, DSCF0878 separate ===")
    for key, v in summary.items():
        print(f"{key:32s} regressing4={v['regressing_4_mean']}  DSCF0878={v['dscf0878_outlier']}  control={v['control_mean']}")

    print("\nfinalHeldOut was not accessed by this diagnostic pass.")


if __name__ == "__main__":
    main()
