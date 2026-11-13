"""Phase 4 - signed-error reconciliation for the 4 scenes still regressed
after round 9 (DSCF0590, DSCF0873, DSCF0878, DSCF0879). Diagnostic only:
no training, no checkpoint selection, no finalHeldOut access.

Why this exists: the 2026-11-12 round-9 diagnosis claimed "shadows not
dark enough / highlights too bright" from *histogram fractions*
(near_black_fraction, bright_fraction of pred vs target). Those are
distributional, not per-pixel, and unsigned |pred-target| residuals can't
give direction either. Round 10 then showed near_black_fraction moving
toward target while crop L1 got worse - a sign the histogram framing was
too coarse. This script measures the SIGNED per-pixel error directly,
binned by the TARGET's luma band, so "too bright in the shadows" means
exactly that: pixels whose Fuji-reference value is a shadow, where the
model's output is brighter than the reference.

Per scene, per seed, per target-luma band:
  - pixel fraction in band
  - signed linear luma error        mean(luma_pred - luma_target)
  - signed relative luma error      mean((luma_pred - luma_target) / (luma_target + eps))
  - signed gamma-encoded error      mean(srgb(luma_pred) - srgb(luma_target))  (perceptual-ish)
  - fraction of pixels too bright   mean(luma_pred > luma_target)
  - the same signed linear error for the UNCORRECTED input vs target,
    i.e. what the correction has to do in that band, and the fraction of
    it the model achieved
  - signed per-channel linear error (R, G, B) - color cast vs brightness

All 76 monitor scenes are evaluated; the 72 non-failing scenes are the
control, reported as a mean and as a sign-agreement count, so it is
visible whether any bias found is specific to the 4 failures or global.

Round 9 (selected-best) is primary; round 6 is also run so sign
stability across rounds is known.

Visuals (diagnostics/signed_errors/): for each of the 4 failures,
input | round 9 pred | target | signed error map (red = model too bright,
blue = too dark, gamma-encoded luma difference, +-0.15 full scale), full
frame and a crop at the structural diagnosis's highlight centroid.
Rendered from round 9 seed 44 (its best-monitor-L1 seed); the numbers
above are over all 3 seeds.

train + monitor only - finalHeldOut has no code path here.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import Phase4PairDataset
from model import BilateralGridPredictor, HybridPredictor, compute_luma_guide
from render_visual_inspection import crop_around, make_row, srgb_encode, to_uint8

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"
IMG_DIR = OUT_DIR / "signed_errors"

SEEDS = [42, 43, 44]
ROUND9_CHECKPOINTS = {42: "r9-expandeddata-seed42-best.pt", 43: "r9-expandeddata-seed43-best.pt", 44: "r9-expandeddata-seed44-best.pt"}
ROUND6_CHECKPOINTS = {42: "r6-hybrid-seed42-best.pt", 43: "r6-hybrid-seed43-best.pt", 44: "r6-hybrid-seed44-best.pt"}
VISUAL_SEED = 44

FAILURES = ["DSCF0590.RAF", "DSCF0873.RAF", "DSCF0878.RAF", "DSCF0879.RAF"]
OUTLIER = "DSCF0878.RAF"

# Target-luma bands (linear light). The 0.05 / 0.7 edges match
# luma_distribution_stats' near-black / bright thresholds so this lines up
# with the earlier histogram-based claim it is checking.
BANDS = [
    ("deep_shadow", 0.0, 0.02),
    ("shadow", 0.02, 0.05),
    ("dark", 0.05, 0.2),
    ("mid", 0.2, 0.5),
    ("light", 0.5, 0.7),
    ("highlight", 0.7, 10.0),
]
EPS = 1e-3
HEATMAP_SCALE = 0.15


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


def band_stats(pred_luma, target_luma, input_luma, pred_rgb, target_rgb):
    """All inputs numpy. pred/target/input_luma: HxW. pred/target_rgb: HxWx3."""
    out = {}
    signed_pred = pred_luma - target_luma
    signed_input = input_luma - target_luma
    gamma_signed = srgb_encode(pred_luma) - srgb_encode(target_luma)
    for name, lo, hi in BANDS:
        mask = (target_luma >= lo) & (target_luma < hi)
        n = int(mask.sum())
        if n == 0:
            out[name] = {"pixel_fraction": 0.0}
            continue
        sp = signed_pred[mask]
        si = signed_input[mask]
        needed = float(si.mean())  # correction the model must apply (input->target), negated sign
        achieved_fraction = None
        if abs(needed) > 1e-4:
            # input error is `needed`; model's remaining error is sp.mean(); fraction closed = 1 - remaining/needed
            achieved_fraction = float(1.0 - sp.mean() / needed)
        out[name] = {
            "pixel_fraction": n / mask.size,
            "signed_linear_error": float(sp.mean()),
            "signed_relative_error": float((sp / (target_luma[mask] + EPS)).mean()),
            "signed_gamma_error": float(gamma_signed[mask].mean()),
            "fraction_too_bright": float((sp > 0).mean()),
            "input_signed_linear_error": needed,
            "fraction_of_needed_correction_achieved": achieved_fraction,
            "signed_channel_error": [float((pred_rgb[..., c][mask] - target_rgb[..., c][mask]).mean()) for c in range(3)],
        }
    return out


def mean_band_stats(list_of_stats):
    """Mean over a list of band-stat dicts (seeds or scenes); None-safe."""
    out = {}
    for name, _, _ in BANDS:
        keys = set()
        for s in list_of_stats:
            keys.update(k for k, v in s.get(name, {}).items() if v is not None)
        agg = {}
        for k in keys:
            vals = [s[name][k] for s in list_of_stats if s.get(name, {}).get(k) is not None]
            if not vals:
                continue
            if isinstance(vals[0], list):
                agg[k] = [float(np.mean([v[c] for v in vals])) for c in range(len(vals[0]))]
            else:
                agg[k] = float(np.mean(vals))
        out[name] = agg
    return out


def heatmap(signed_hw, scale=HEATMAP_SCALE):
    """Diverging map: red = positive (model too bright), blue = negative."""
    e = np.clip(signed_hw / scale, -1, 1)
    img = np.full(signed_hw.shape + (3,), 255, dtype=np.float32)
    pos, neg = e > 0, e < 0
    img[pos, 1] = 255 * (1 - e[pos]); img[pos, 2] = 255 * (1 - e[pos])
    img[neg, 0] = 255 * (1 + e[neg]); img[neg, 1] = 255 * (1 + e[neg])
    return img.clip(0, 255).astype(np.uint8)


def evaluate_round(checkpoints, device, monitor_ds, label, render_names=None, centroid_by_name=None):
    models = {s: load_model_from_ckpt(c, device) for s, c in checkpoints.items()}
    per_scene = {}
    for i in range(len(monitor_ds)):
        item = monitor_ds[i]
        name = item["name"]
        full_res = item["input"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        target_luma = compute_luma_guide(target, "rec709").squeeze().cpu().numpy()
        input_luma = compute_luma_guide(full_res, "rec709").squeeze().cpu().numpy()
        target_rgb = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
        per_seed = []
        for seed, (model, cfg) in models.items():
            with torch.no_grad():
                pred, _ = model.predict_and_apply(full_res, cfg["low_res"], cfg.get("luma_mode", "rec709"))
            pred_luma = compute_luma_guide(pred, "rec709").squeeze().cpu().numpy()
            pred_rgb = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            per_seed.append(band_stats(pred_luma, target_luma, input_luma, pred_rgb, target_rgb))

            if render_names and name in render_names and seed == VISUAL_SEED:
                input_rgb = full_res.squeeze(0).permute(1, 2, 0).cpu().numpy()
                signed_gamma = srgb_encode(pred_luma) - srgb_encode(target_luma)
                panels = [("input", to_uint8(input_rgb)), (label, to_uint8(pred_rgb)),
                          ("target", to_uint8(target_rgb)), ("signed_error", heatmap(signed_gamma))]
                scale = 480 / panels[0][1].shape[1]
                size = (int(panels[0][1].shape[1] * scale), int(panels[0][1].shape[0] * scale))
                full_row = make_row([(l, np.array(Image.fromarray(p).resize(size))) for l, p in panels])
                Image.fromarray(full_row).save(IMG_DIR / f"{Path(name).stem}_{label}_full.png")
                cx, cy = centroid_by_name.get(name, (0.5, 0.5))
                crop_row = make_row([(l, crop_around(p, cx, cy)) for l, p in panels])
                Image.fromarray(crop_row).save(IMG_DIR / f"{Path(name).stem}_{label}_crop.png")

        per_scene[name] = {"per_seed": per_seed, "mean": mean_band_stats(per_seed)}
    return per_scene


def sign_agreement(per_scene, names, band, key="signed_linear_error"):
    """How many of `names` have a positive (too bright) seed-mean error in `band`."""
    vals = [per_scene[n]["mean"].get(band, {}).get(key) for n in names]
    vals = [v for v in vals if v is not None]
    return {"n": len(vals), "n_too_bright": int(sum(v > 0 for v in vals)), "n_too_dark": int(sum(v < 0 for v in vals))}


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    monitor_ds = Phase4PairDataset("monitor")
    diag = json.loads((OUT_DIR / "structural_diagnosis.json").read_text())
    centroid_by_name = {r["name"]: tuple(r["largest_centroid_norm"]) for r in diag["regressing_scenes"] if r.get("largest_centroid_norm")}

    r9 = evaluate_round(ROUND9_CHECKPOINTS, device, monitor_ds, "round9", render_names=set(FAILURES), centroid_by_name=centroid_by_name)
    r6 = evaluate_round(ROUND6_CHECKPOINTS, device, monitor_ds, "round6")

    all_names = list(r9.keys())
    control_names = [n for n in all_names if n not in FAILURES]

    def summarize(per_scene):
        return {
            "failures": {n: per_scene[n]["mean"] for n in FAILURES},
            "failures_mean_excluding_outlier": mean_band_stats([per_scene[n]["mean"] for n in FAILURES if n != OUTLIER]),
            "control_mean": mean_band_stats([per_scene[n]["mean"] for n in control_names]),
            "control_per_scene_mean": {n: per_scene[n]["mean"] for n in control_names},
            "sign_agreement_by_band": {
                band: {"failures": sign_agreement(per_scene, FAILURES, band), "control": sign_agreement(per_scene, control_names, band)}
                for band, _, _ in BANDS
            },
            "per_seed_failures": {n: per_scene[n]["per_seed"] for n in FAILURES},
        }

    report = {
        "bands_by_target_luma_linear": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in BANDS],
        "sign_convention": "positive = model output brighter than Fuji target; negative = darker",
        "n_control_scenes": len(control_names),
        "round9": summarize(r9),
        "round6": summarize(r6),
        "visuals": sorted(p.name for p in IMG_DIR.glob("*.png")),
    }
    (OUT_DIR / "signed-error-diagnosis.json").write_text(json.dumps(report, indent=2))

    def fmt_row(stats):
        cells = []
        for band, _, _ in BANDS:
            b = stats.get(band, {})
            if "signed_linear_error" not in b:
                cells.append(f"{band[:6]:>6s}:   --   ")
                continue
            cells.append(f"{band[:6]:>6s}:{b['signed_linear_error']:+.4f}({b['fraction_too_bright']:.2f})")
        return "  ".join(cells)

    print("\n=== ROUND 9 signed luma error by TARGET band: mean(pred-target) (fraction of pixels too bright) ===")
    for n in FAILURES:
        tag = " [outlier]" if n == OUTLIER else ""
        print(f"{n}{tag}\n   {fmt_row(r9[n]['mean'])}")
    print(f"control mean ({len(control_names)} scenes)\n   {fmt_row(report['round9']['control_mean'])}")

    print("\n=== ROUND 9 uncorrected input vs target (what the correction must do), failures ===")
    for n in FAILURES:
        cells = []
        for band, _, _ in BANDS:
            b = r9[n]["mean"].get(band, {})
            if "input_signed_linear_error" in b:
                ach = b.get("fraction_of_needed_correction_achieved")
                cells.append(f"{band[:6]:>6s}:{b['input_signed_linear_error']:+.4f}" + (f"({ach:.0%})" if ach is not None else ""))
        print(f"{n}\n   {'  '.join(cells)}")

    print("\n=== sign agreement (round 9): scenes with too-bright vs too-dark seed-mean error ===")
    for band, _, _ in BANDS:
        sa = report["round9"]["sign_agreement_by_band"][band]
        print(f"  {band:12s} failures: {sa['failures']['n_too_bright']}↑ {sa['failures']['n_too_dark']}↓ of {sa['failures']['n']}   "
              f"control: {sa['control']['n_too_bright']}↑ {sa['control']['n_too_dark']}↓ of {sa['control']['n']}")

    print(f"\nvisuals: {len(report['visuals'])} images in {IMG_DIR}")
    print("finalHeldOut was not accessed by this diagnostic pass.")


if __name__ == "__main__":
    main()
