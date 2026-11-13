"""Render input | round-9 output | Fuji target | signed error map for named
scenes in the train or monitor tier - the same panels and colour
convention as diagnose_signed_errors.py (red = model too bright, blue =
too dark). Diagnostic only; finalHeldOut is refused by the dataset.

Usage: python3 render_signed_errors.py --tier train DSCF1419.RAF DSCF1417.RAF
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import Phase4PairDataset
from diagnose_signed_errors import IMG_DIR, ROUND9_CHECKPOINTS, VISUAL_SEED, heatmap, load_model_from_ckpt
from model import compute_luma_guide
from render_visual_inspection import make_row, srgb_encode, to_uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["train", "monitor"], required=True)
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model, cfg = load_model_from_ckpt(ROUND9_CHECKPOINTS[VISUAL_SEED], device)
    ds = Phase4PairDataset(args.tier)
    wanted = set(args.names)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(len(ds)):
        if ds.scenes[i]["name"] not in wanted if hasattr(ds, "scenes") else False:
            continue
        item = ds[i]
        if item["name"] not in wanted:
            continue
        full_res = item["input"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred, _ = model.predict_and_apply(full_res, cfg["low_res"], cfg.get("luma_mode", "rec709"))
        pred_luma = compute_luma_guide(pred, "rec709").squeeze().cpu().numpy()
        target_luma = compute_luma_guide(target, "rec709").squeeze().cpu().numpy()
        input_luma = compute_luma_guide(full_res, "rec709").squeeze().cpu().numpy()
        panels = [
            ("input", to_uint8(full_res.squeeze(0).permute(1, 2, 0).cpu().numpy())),
            ("round9", to_uint8(pred.squeeze(0).permute(1, 2, 0).cpu().numpy())),
            ("target", to_uint8(target.squeeze(0).permute(1, 2, 0).cpu().numpy())),
            ("signed_error", heatmap(srgb_encode(pred_luma) - srgb_encode(target_luma))),
        ]
        scale = 480 / panels[0][1].shape[1]
        size = (int(panels[0][1].shape[1] * scale), int(panels[0][1].shape[0] * scale))
        row = make_row([(l, np.array(Image.fromarray(p).resize(size))) for l, p in panels])
        out = IMG_DIR / f"{Path(item['name']).stem}_{args.tier}_round9_full.png"
        Image.fromarray(row).save(out)
        print(f"{item['name']}: mean luma input {input_luma.mean():.4f}  target {target_luma.mean():.4f}  round9 {pred_luma.mean():.4f}  -> {out.name}")
        wanted.discard(item["name"])
        if not wanted:
            break
    if wanted:
        print(f"not found in {args.tier}: {sorted(wanted)}")


if __name__ == "__main__":
    main()
