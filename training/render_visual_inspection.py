"""Render side-by-side comparison images for visual inspection of the
persistent failing scenes vs matched successful controls - per instruction,
diagnostic only, no training. For each scene: as-shot input | model output
| Fuji reference (the X RAW Studio target - this project's ground truth),
both full-frame and a crop zoomed on the highlight region (using the
2026-09-16 structural diagnosis's saved centroid), since that's what the
diagnosis found to be structurally different for these scenes.

train + monitor only - finalHeldOut has no code path here.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import CAL_DIR, Phase4PairDataset
from model import BilateralGridPredictor, apply_bilateral_grid, resize_for_network

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "visual_inspection"
CHECKPOINT = "r2-b-lrsched-seed43-best.pt"  # frozen baseline, best-performing seed - same one used throughout

SCENES = {
    "reproducible": ["DSCF0590.RAF", "DSCF0873.RAF", "DSCF0879.RAF"],
    "seed_dependent": ["DSCF0553.RAF"],
    "outlier": ["DSCF0878.RAF"],
    "control": ["DSCF0539.RAF", "DSCF0542.RAF", "DSCF0543.RAF", "DSCF0564.RAF"],
}


def srgb_encode(linear):
    """Inverse of the sRGB EOTF - linear light -> gamma-encoded, for
    viewing. linear: numpy array, 0..1 (will be clamped)."""
    c = np.clip(linear, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def to_uint8(linear_hwc):
    return (srgb_encode(linear_hwc) * 255).clip(0, 255).astype(np.uint8)


def load_model(device):
    ckpt = torch.load(CKPT_DIR / CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BilateralGridPredictor(
        grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"],
        low_res=cfg["low_res"], base_ch=cfg["base_ch"],
        use_detail_branch=cfg.get("use_detail_branch", False), detail_ch=cfg.get("detail_ch", 8),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def make_row(panels, gap=8, label_h=28):
    """panels: list of (label, HxWx3 uint8 array), same H/W. Horizontal
    strip with small text labels underneath isn't available without a
    font lib here, so labels go in the filename instead - just concat."""
    h, w, _ = panels[0][1].shape
    n = len(panels)
    strip = np.full((h, w * n + gap * (n - 1), 3), 40, dtype=np.uint8)
    for i, (_, img) in enumerate(panels):
        x0 = i * (w + gap)
        strip[:, x0:x0 + w] = img
    return strip


def crop_around(hwc, cx_norm, cy_norm, crop_frac=0.22):
    h, w = hwc.shape[:2]
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    cx, cy = int(cx_norm * w), int(cy_norm * h)
    x0 = max(0, min(w - cw, cx - cw // 2))
    y0 = max(0, min(h - ch, cy - ch // 2))
    return hwc[y0:y0 + ch, x0:x0 + cw]


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model, cfg = load_model(device)
    monitor_ds = Phase4PairDataset("monitor")
    by_name = {monitor_ds[i]["name"]: i for i in range(len(monitor_ds))}

    diag = json.loads((Path(__file__).resolve().parent / "diagnostics" / "structural_diagnosis.json").read_text())
    centroid_by_name = {}
    for r in diag["regressing_scenes"] + diag["control_scenes"]:
        if r.get("largest_centroid_norm"):
            centroid_by_name[r["name"]] = r["largest_centroid_norm"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((CAL_DIR / "phase4-corpus-manifest.json").read_text())
    meta_by_name = {s["name"]: s for s in manifest["tiers"]["monitor"]["scenes"]}

    index_lines = []
    for category, names in SCENES.items():
        for name in names:
            idx = by_name[name]
            item = monitor_ds[idx]
            full_res = item["input"].unsqueeze(0).to(device)
            target = item["target"].unsqueeze(0).to(device)
            with torch.no_grad():
                low_res_in = resize_for_network(full_res, cfg["low_res"])
                grid = model(low_res_in)
                pred = apply_bilateral_grid(grid, full_res, luma_mode=cfg.get("luma_mode", "rec709"))
            model_l1 = (pred - target).abs().mean().item()
            baseline_l1 = (full_res - target).abs().mean().item()

            input_np = full_res.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            target_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy()

            input_u8, pred_u8, target_u8 = to_uint8(input_np), to_uint8(pred_np), to_uint8(target_np)

            # full-frame row, downscaled for manageable file size
            scale = 480 / input_u8.shape[1]
            new_size = (int(input_u8.shape[1] * scale), int(input_u8.shape[0] * scale))
            full_row = make_row([
                ("input", np.array(Image.fromarray(input_u8).resize(new_size))),
                ("model", np.array(Image.fromarray(pred_u8).resize(new_size))),
                ("target", np.array(Image.fromarray(target_u8).resize(new_size))),
            ])
            Image.fromarray(full_row).save(OUT_DIR / f"{category}_{Path(name).stem}_full.png")

            # zoomed crop on the highlight region, at native resolution
            cx, cy = centroid_by_name.get(name, (0.5, 0.5))
            crop_row = make_row([
                ("input", crop_around(input_u8, cx, cy)),
                ("model", crop_around(pred_u8, cx, cy)),
                ("target", crop_around(target_u8, cx, cy)),
            ])
            Image.fromarray(crop_row).save(OUT_DIR / f"{category}_{Path(name).stem}_crop.png")

            meta = meta_by_name.get(name, {})
            index_lines.append(
                f"{category:16s} {name:16s} wb_manual={meta.get('wbManualFlag')} p99={meta.get('p99')} "
                f"highlight={meta.get('highlightClass')} model_l1={model_l1:.4f} baseline_l1={baseline_l1:.4f} "
                f"regressed={model_l1 > baseline_l1}"
            )
            print(f"rendered {category}/{name}")

    (OUT_DIR / "index.txt").write_text("\n".join(index_lines))
    print(f"\nwrote comparison images to {OUT_DIR}")
    print("\nfinalHeldOut was not accessed by this visual inspection.")


if __name__ == "__main__":
    main()
