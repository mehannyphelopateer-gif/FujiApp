"""F24-only R11 display-domain audit, no fitting or CN-target reads.

Split negative/over-white output tails by size, compare grid/refinement,
and quantify their contribution to displayed Provia colour error. Uses
the frozen R11 and the existing hash-bound Provia cache only. Old evidence
and all K/monitor/finalHeldOut/T8 pixels stay untouched.
"""
import json

import numpy as np
import torch

import run_cn_wb_assay as A
from color_metrics import ciede2000, linear_srgb_to_lab, self_test

OUT = A.S.DIAG / "r11-f24-display-domain-audit.json"
THRESHOLDS = (0.0, 0.0001, 0.001, 0.01, 0.05)


def domain_stats(pred, ref, de):
    low, high = pred < 0, pred > 1
    any_low, any_high = low.any(-1), high.any(-1)
    def region(mask):
        return {"pixel_fraction": float(mask.mean()),
                "mean_dE00": float(de[mask].mean()) if mask.any() else None,
                "share_of_total_dE00_sum": float(de[mask].sum() / de.sum()) if de.sum() > 0 else None}
    tails = {}
    for threshold in THRESHOLDS:
        tails[str(threshold)] = {
            "below_minus_threshold_pixel_fraction": float((pred < -threshold).any(-1).mean()),
            "above_one_plus_threshold_pixel_fraction": float((pred > 1 + threshold).any(-1).mean()),
        }
    return {
        "min": float(pred.min()), "max": float(pred.max()),
        "any_out_of_range_pixel_fraction": float((any_low | any_high).mean()),
        "low_channel_fractions_RGB": low.mean(axis=(0, 1)).tolist(),
        "high_channel_fractions_RGB": high.mean(axis=(0, 1)).tolist(),
        "below_zero": region(any_low), "above_one": region(any_high),
        "both_low_and_high": region(any_low & any_high),
        "in_range": region(~(any_low | any_high)),
        "tail_thresholds_linear": tails,
        "negative_channels_whose_target_exceeds_0.001_pixel_fraction": float((low & (ref > 0.001)).any(-1).mean()),
        "over_one_channels_whose_target_is_below_0.999_pixel_fraction": float((high & (ref < 0.999)).any(-1).mean()),
        "unclipped_linear_l1": float(np.abs(pred - ref).mean()),
        "display_clipped_linear_l1": float(np.abs(np.clip(pred, 0, 1) - ref).mean()),
        "mean_dE00": float(de.mean()), "p95_dE00": float(np.percentile(de, 95)),
    }


def main():
    if OUT.exists():
        raise ValueError("Audit exists; refusing to overwrite evidence")
    self_test()
    manifest = A.load_manifest()
    if len(manifest["scenes"]) != 24:
        raise ValueError("Expected F24 only")
    dev = A.S.device()
    model, config = A.r11_model(dev)
    rows = {}
    for source in manifest["scenes"]:
        name = source["name"]
        x, target = A.load_pairs(manifest, [name])[name]
        with torch.no_grad():
            final, aux = model.predict_and_apply(
                x.to(dev), config["low_res"], config.get("luma_mode", "rec709"),
                condition=torch.tensor([[1.0]], device=dev))
        image = lambda t: t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
        reference = image(target)
        ref_lab = linear_srgb_to_lab(reference)
        stages = {}
        for stage, tensor in (("grid", aux["grid_output"]), ("final", final)):
            pred = image(tensor)
            if not np.isfinite(pred).all():
                raise ValueError("Non-finite frozen R11 output")
            de = ciede2000(linear_srgb_to_lab(pred), ref_lab)
            stages[stage] = domain_stats(pred, reference, de)
        delta = image(aux["delta"])
        row = {
            "session": source["session"], "stages": stages,
            "refinement_mean_abs_delta": float(np.abs(delta).mean()),
            "refinement_near_bound_pixel_fraction": float((np.abs(delta) >= 0.95 * config["refinement_max_delta"]).any(-1).mean()),
            "target_any_near_zero_channel_pixel_fraction": float((reference <= 0.0001).any(-1).mean()),
            "target_any_near_one_channel_pixel_fraction": float((reference >= 0.9999).any(-1).mean()),
            "input_sha256": source["fjlrg_sha256"], "provia_sha256": source["provia_sha256"],
        }
        rows[name] = row
        print(name, "negative", stages["final"]["below_zero"]["pixel_fraction"],
              "overwhite", stages["final"]["above_one"]["pixel_fraction"],
              "grid/final dE", stages["grid"]["mean_dE00"], stages["final"]["mean_dE00"], flush=True)
    A.S.check_source()
    summary = {}
    for stage in ("grid", "final"):
        records = [r["stages"][stage] for r in rows.values()]
        summary[stage] = {
            "median_mean_dE00": float(np.median([r["mean_dE00"] for r in records])),
            "mean_of_scene_mean_dE00": float(np.mean([r["mean_dE00"] for r in records])),
            "median_negative_pixel_fraction": float(np.median([r["below_zero"]["pixel_fraction"] for r in records])),
            "median_overwhite_pixel_fraction": float(np.median([r["above_one"]["pixel_fraction"] for r in records])),
            "median_tail_thresholds_linear": {key: {side: float(np.median([r["tail_thresholds_linear"][key][side] for r in records]))
                for side in records[0]["tail_thresholds_linear"][key]} for key in records[0]["tail_thresholds_linear"]},
        }
    result = {
        "scope": "F24 Provia diagnostic only, R11 in-sample, no fitting or CN/K/monitor/finalHeldOut/T8 targets",
        "source_checkpoint_sha256": A.S.SOURCE_SHA, "manifest_sha256": A.MANIFEST_SHA,
        "note": "Display DeltaE already clamps linear RGB to [0,1]. Adding a final clamp cannot change that scored image. Out-of-range occupancy is not a percentage of image lost, nor proof of sensor clipping or error causation.",
        "summary": summary, "per_scene": rows,
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
