"""Phase 4 diagnostic pass - per instruction: no further blind hyperparameter
sweeps until we understand WHY 3-5 of 76 monitor scenes regress vs the
no-correction baseline, and why that count varies across seeds.

Runs all 3 round-2 seed checkpoints (r2-b-lrsched, -seed43, -seed44)
against every monitor scene, computes per-scene metrics, the union of
scenes that regressed under at least one seed, a matched-control set
(scenes that never regressed under any seed), compares their RAW/WB/
exposure metadata and session membership, and saves spatial residual
heatmaps for every regressing scene under every seed where it regressed
(to check whether the same scene fails the same way across seeds, or
differently - a strong hint about mechanism either way).

train + monitor only, as always - finalHeldOut has no code path here,
same hard rule as everywhere else in training/.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import CAL_DIR, Phase4PairDataset
from model import BilateralGridPredictor, apply_bilateral_grid, resize_for_network

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"
HEATMAP_DIR = OUT_DIR / "residual_heatmaps"

SEED_CHECKPOINTS = {
    42: "r2-b-lrsched-best.pt",
    43: "r2-b-lrsched-seed43-best.pt",
    44: "r2-b-lrsched-seed44-best.pt",
}


def load_model(ckpt_name, device):
    ckpt = torch.load(CKPT_DIR / ckpt_name, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BilateralGridPredictor(
        grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"],
        low_res=cfg["low_res"], base_ch=cfg["base_ch"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def per_scene_l1_map(pred, target):
    """(1,3,H,W) -> HxW mean-abs-error map (mean over channels)."""
    return (pred - target).abs().mean(dim=1).squeeze(0)


def save_heatmap(residual_map_np, path, vmax=None):
    """residual_map_np: HxW float array of per-pixel abs error."""
    vmax = vmax or float(np.percentile(residual_map_np, 99.5))
    vmax = max(vmax, 1e-6)
    norm = np.clip(residual_map_np / vmax, 0, 1)
    img8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), colored)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    OUT_DIR.mkdir(exist_ok=True)
    HEATMAP_DIR.mkdir(exist_ok=True)

    monitor_ds = Phase4PairDataset("monitor")
    manifest = json.loads((CAL_DIR / "phase4-corpus-manifest.json").read_text())
    meta_by_name = {s["name"]: s for s in manifest["tiers"]["monitor"]["scenes"]}

    models = {}
    for seed, ckpt_name in SEED_CHECKPOINTS.items():
        model, cfg = load_model(ckpt_name, device)
        models[seed] = (model, cfg)
        print(f"loaded seed {seed}: {ckpt_name}  grid={cfg['grid_spatial']}x{cfg['grid_spatial']}x{cfg['grid_luma']}")

    # per-scene, per-seed results
    per_scene = {}  # name -> {seed: {l1, baseline_l1, regressed, regression_magnitude}}
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
                pred = apply_bilateral_grid(grid, full_res)
                residual_map = per_scene_l1_map(pred, target)
                model_l1 = residual_map.mean().item()
            regressed = model_l1 > baseline_l1
            per_scene[name]["seeds"][seed] = {
                "model_l1": model_l1,
                "regressed": regressed,
                "regression_magnitude": max(0.0, model_l1 - baseline_l1),
            }
            if regressed:
                heatmap_path = HEATMAP_DIR / f"{Path(name).stem}_seed{seed}.png"
                save_heatmap(residual_map.cpu().numpy(), heatmap_path)
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(monitor_ds)}")

    # union of regressing scenes (at least one seed) vs control (never regressed)
    regressing_union = [n for n, d in per_scene.items() if any(s["regressed"] for s in d["seeds"].values())]
    control = [n for n, d in per_scene.items() if not any(s["regressed"] for s in d["seeds"].values())]
    print(f"\nregressing (union across 3 seeds): {len(regressing_union)} / {len(per_scene)}")
    print(f"control (never regressed): {len(control)}")

    def meta_summary(names):
        metas = [meta_by_name[n] for n in names if n in meta_by_name]
        if not metas:
            return {}
        manual_wb = sum(1 for m in metas if m.get("wbManualFlag") == 1)
        p99s = [m["p99"] for m in metas if "p99" in m]
        highlight_counts = {}
        for m in metas:
            hc = m.get("highlightClass", "unknown")
            highlight_counts[hc] = highlight_counts.get(hc, 0) + 1
        sessions = {}
        for m in metas:
            s = m["sessionIndex"]
            sessions[s] = sessions.get(s, 0) + 1
        return {
            "n": len(metas),
            "manual_wb_fraction": manual_wb / len(metas),
            "p99_mean": sum(p99s) / len(p99s) if p99s else None,
            "p99_max": max(p99s) if p99s else None,
            "highlight_class_counts": highlight_counts,
            "session_counts": sessions,
        }

    regressing_meta = meta_summary(regressing_union)
    control_meta = meta_summary(control)
    print("\n=== regressing scenes metadata ===")
    print(json.dumps(regressing_meta, indent=2))
    print("\n=== control scenes metadata ===")
    print(json.dumps(control_meta, indent=2))

    # per-scene detail for the regressing set, with metadata attached
    regressing_detail = []
    for n in regressing_union:
        d = per_scene[n]
        m = meta_by_name.get(n, {})
        regressing_detail.append({
            "name": n,
            "shoot": m.get("shoot"),
            "sessionIndex": m.get("sessionIndex"),
            "wbManualFlag": m.get("wbManualFlag"),
            "p99": m.get("p99"),
            "highlightClass": m.get("highlightClass"),
            "baseline_l1": d["baseline_l1"],
            "seeds_regressed_under": [s for s, v in d["seeds"].items() if v["regressed"]],
            "regression_magnitude_by_seed": {s: v["regression_magnitude"] for s, v in d["seeds"].items()},
        })
    regressing_detail.sort(key=lambda r: -max(r["regression_magnitude_by_seed"].values()))

    report = {
        "monitor_scene_count": len(per_scene),
        "regressing_union_count": len(regressing_union),
        "control_count": len(control),
        "regressing_scenes": regressing_detail,
        "regressing_metadata_summary": regressing_meta,
        "control_metadata_summary": control_meta,
        "per_scene_all": per_scene,
    }
    (OUT_DIR / "regression_diagnosis.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT_DIR / 'regression_diagnosis.json'}")
    print(f"wrote {len(list(HEATMAP_DIR.glob('*.png')))} residual heatmaps to {HEATMAP_DIR}")
    print("\nfinalHeldOut was not accessed by this diagnostic pass.")


if __name__ == "__main__":
    main()
