"""Companion to diagnose_signed_errors.py - the same signed, target-luma-
banded error analysis, but over the TRAIN tier, round 9 checkpoints.

Question it answers (needed before proposing a loss change): the 4
monitor failures are globally over-brightened by round 9. Is that an
UNDER-FIT - does round 9 also over-brighten the same kind of scene in its
own training set, i.e. the linear-L1 objective tolerates that error there
- or a GENERALIZATION failure, where training scenes of that kind are
fit well and only unseen ones go wrong? A loss change can only help in
the first case. Diagnostic only: no training, no selection, no
finalHeldOut (Phase4PairDataset refuses that tier).

Groups reported, defined from TARGET/manifest properties only (never from
the model's error, to avoid circularity):
  - all 599 train scenes (distribution + top over-lifted tail)
  - capture batch 2 (manifest trip == "Phase4-Capture2", 38 scenes) and
    its DSCF1413-1416 cluster, the scenes added specifically because
    they resemble the failures
  - "low-key" scenes: target deep_shadow+shadow pixel fraction >= 0.5
    (3 of the 4 monitor failures meet this: 0.72, 0.73, 0.62; DSCF0878
    does not, 0.34)
"""
import json
from pathlib import Path

import numpy as np
import torch

from dataset import CAL_DIR, Phase4PairDataset
from diagnose_signed_errors import BANDS, ROUND9_CHECKPOINTS, band_stats, load_model_from_ckpt, mean_band_stats
from model import compute_luma_guide

OUT_DIR = Path(__file__).resolve().parent / "diagnostics"
CLUSTER = ["DSCF1413.RAF", "DSCF1414.RAF", "DSCF1415.RAF", "DSCF1416.RAF"]


def frame_signed(s):
    return float(sum(s[b]["pixel_fraction"] * s[b]["signed_linear_error"] for b, _, _ in BANDS if "signed_linear_error" in s.get(b, {})))


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")
    manifest = json.loads((CAL_DIR / "phase4-corpus-manifest.json").read_text())
    meta = {s["name"]: s for s in manifest["tiers"]["train"]["scenes"]}

    ds = Phase4PairDataset("train")
    models = {s: load_model_from_ckpt(c, device) for s, c in ROUND9_CHECKPOINTS.items()}
    rows = []
    for i in range(len(ds)):
        item = ds[i]
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
        m = mean_band_stats(per_seed)
        low_key_frac = m["deep_shadow"].get("pixel_fraction", 0) + m["shadow"].get("pixel_fraction", 0)
        rows.append({
            "name": name,
            "trip": meta.get(name, {}).get("trip"),
            "sessionIndex": meta.get(name, {}).get("sessionIndex"),
            "low_key_fraction": low_key_frac,
            "frame_signed_error": frame_signed(m),
            "bands": m,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}")

    def group_summary(sel, label):
        if not sel:
            return {"label": label, "n": 0}
        fs = np.array([r["frame_signed_error"] for r in sel])
        mids = np.array([r["bands"]["mid"].get("signed_linear_error", np.nan) for r in sel])
        shadows = np.array([r["bands"]["shadow"].get("signed_linear_error", np.nan) for r in sel])
        needed_mid = np.array([r["bands"]["mid"].get("input_signed_linear_error", np.nan) for r in sel])
        needed_shadow = np.array([r["bands"]["shadow"].get("input_signed_linear_error", np.nan) for r in sel])
        return {
            "label": label, "n": len(sel),
            "frame_signed_mean": float(fs.mean()), "frame_signed_p90": float(np.percentile(fs, 90)), "frame_signed_max": float(fs.max()),
            "mid_signed_mean": float(np.nanmean(mids)), "mid_signed_p90": float(np.nanpercentile(mids, 90)), "mid_signed_max": float(np.nanmax(mids)),
            "shadow_signed_mean": float(np.nanmean(shadows)),
            "n_mid_over_0.05": int(np.nansum(mids > 0.05)), "n_mid_over_0.10": int(np.nansum(mids > 0.10)),
            "needed_mid_mean_input_minus_target": float(np.nanmean(needed_mid)),
            "needed_shadow_mean_input_minus_target": float(np.nanmean(needed_shadow)),
        }

    groups = {
        "all_train": group_summary(rows, "all 599 train scenes"),
        "capture_batch_2": group_summary([r for r in rows if r["trip"] == "Phase4-Capture2"], "capture batch 2 (38)"),
        "capture_batch_2_cluster": group_summary([r for r in rows if r["name"] in CLUSTER], "DSCF1413-1416 cluster"),
        "low_key_ge_0.5": group_summary([r for r in rows if r["low_key_fraction"] >= 0.5], "low-key (target deep+shadow >= 0.5)"),
        "not_low_key": group_summary([r for r in rows if r["low_key_fraction"] < 0.5], "not low-key"),
    }
    top = sorted(rows, key=lambda r: -r["frame_signed_error"])[:12]
    report = {
        "checkpoints": ROUND9_CHECKPOINTS,
        "groups": groups,
        "top_over_lifted_train_scenes": [
            {"name": r["name"], "trip": r["trip"], "low_key_fraction": r["low_key_fraction"], "frame_signed_error": r["frame_signed_error"],
             "mid_signed": r["bands"]["mid"].get("signed_linear_error"), "shadow_signed": r["bands"]["shadow"].get("signed_linear_error")}
            for r in top],
        "cluster_detail": [{"name": r["name"], "low_key_fraction": r["low_key_fraction"], "frame_signed_error": r["frame_signed_error"],
                            "bands": {b: {k: r["bands"][b].get(k) for k in ("pixel_fraction", "signed_linear_error", "input_signed_linear_error")} for b, _, _ in BANDS}}
                           for r in rows if r["name"] in CLUSTER],
        "per_scene": [{"name": r["name"], "trip": r["trip"], "low_key_fraction": r["low_key_fraction"], "frame_signed_error": r["frame_signed_error"],
                       "mid_signed": r["bands"]["mid"].get("signed_linear_error"), "shadow_signed": r["bands"]["shadow"].get("signed_linear_error"),
                       "light_signed": r["bands"]["light"].get("signed_linear_error"), "highlight_signed": r["bands"]["highlight"].get("signed_linear_error")}
                      for r in rows],
    }
    (OUT_DIR / "signed-error-train-survey.json").write_text(json.dumps(report, indent=2))

    print("\n=== ROUND 9 signed error on TRAIN tier, by group ===")
    for g in groups.values():
        if g["n"] == 0:
            print(f"{g['label']}: n=0"); continue
        print(f"{g['label']} (n={g['n']}): frame_signed mean {g['frame_signed_mean']:+.4f} p90 {g['frame_signed_p90']:+.4f} max {g['frame_signed_max']:+.4f} | "
              f"mid mean {g['mid_signed_mean']:+.4f} p90 {g['mid_signed_p90']:+.4f} max {g['mid_signed_max']:+.4f} | shadow mean {g['shadow_signed_mean']:+.4f} | "
              f"n mid>0.05: {g['n_mid_over_0.05']}, >0.10: {g['n_mid_over_0.10']} | needed mid {g['needed_mid_mean_input_minus_target']:+.4f} shadow {g['needed_shadow_mean_input_minus_target']:+.4f}")
    print("\nmost over-lifted train scenes (frame-weighted signed error):")
    for r in report["top_over_lifted_train_scenes"]:
        print(f"  {r['name']:14s} trip={r['trip']}  low_key={r['low_key_fraction']:.2f}  frame {r['frame_signed_error']:+.4f}  mid {r['mid_signed']:+.4f}  shadow {r['shadow_signed']:+.4f}")
    print("\nDSCF1413-1416 cluster:")
    for r in report["cluster_detail"]:
        print(f"  {r['name']}  low_key={r['low_key_fraction']:.2f}  frame {r['frame_signed_error']:+.4f}  " +
              "  ".join(f"{b[:6]}:{r['bands'][b]['signed_linear_error']:+.3f}(need {r['bands'][b]['input_signed_linear_error']:+.3f})" for b, _, _ in BANDS if r['bands'][b]['signed_linear_error'] is not None))
    print("\nfinalHeldOut was not accessed by this diagnostic pass.")


if __name__ == "__main__":
    main()
