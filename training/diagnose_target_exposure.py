"""Data-only survey (no model, no training, no finalHeldOut): how much
does the Fuji target differ in exposure from the browser-export input,
per scene, across train + monitor? Flags whether capture batch 2's
TIFF targets sit at a systematically different exposure relative to
their .fjlrg inputs than the rest of the corpus does - which would mean
the model's extreme over-brightening of batch-2 TRAIN scenes
(diagnose_signed_errors_train.py) is a data/export mismatch rather than
a modelling problem, and would have to be fixed in the data, not the
loss.

Per scene: mean linear luma of input and target, their ratio, the
input-minus-target signed error per target band ("needed correction"),
trip, session, low-key fraction. Grouped by trip, plus the global
outlier list by log-ratio deviation from the corpus median.
"""
import json
from pathlib import Path

import numpy as np
import torch

from dataset import CAL_DIR, Phase4PairDataset
from diagnose_signed_errors import BANDS
from model import compute_luma_guide

OUT_DIR = Path(__file__).resolve().parent / "diagnostics"


def main():
    manifest = json.loads((CAL_DIR / "phase4-corpus-manifest.json").read_text())
    meta = {}
    for tier in ("train", "monitor"):
        for s in manifest["tiers"][tier]["scenes"]:
            meta[s["name"]] = {**s, "tier": tier}

    rows = []
    for tier in ("train", "monitor"):
        ds = Phase4PairDataset(tier)
        for i in range(len(ds)):
            item = ds[i]
            inp = item["input"].unsqueeze(0)
            tgt = item["target"].unsqueeze(0)
            il = compute_luma_guide(inp, "rec709").squeeze().numpy()
            tl = compute_luma_guide(tgt, "rec709").squeeze().numpy()
            needed = {}
            for b, lo, hi in BANDS:
                m = (tl >= lo) & (tl < hi)
                needed[b] = {"pixel_fraction": float(m.mean()), "input_minus_target": float((il[m] - tl[m]).mean()) if m.any() else None}
            low_key = needed["deep_shadow"]["pixel_fraction"] + needed["shadow"]["pixel_fraction"]
            mi, mt = float(il.mean()), float(tl.mean())
            md = meta.get(item["name"], {})
            rows.append({
                "name": item["name"], "tier": tier, "trip": md.get("trip"), "sessionIndex": md.get("sessionIndex"),
                "wbManualFlag": md.get("wbManualFlag"), "iso": md.get("iso"),
                "mean_luma_input": mi, "mean_luma_target": mt, "target_over_input": mt / max(mi, 1e-6),
                "log2_ratio": float(np.log2(mt / max(mi, 1e-6))), "low_key_fraction": low_key,
                "p99_input": float(np.percentile(il, 99)), "p99_target": float(np.percentile(tl, 99)),
                "needed": needed,
            })
            if (len(rows)) % 100 == 0:
                print(f"  {len(rows)} scenes")

    def summarize(sel, label):
        if not sel:
            return {"label": label, "n": 0}
        lr = np.array([r["log2_ratio"] for r in sel])
        nm = np.array([r["needed"]["mid"]["input_minus_target"] or np.nan for r in sel], dtype=float)
        ns = np.array([r["needed"]["shadow"]["input_minus_target"] or np.nan for r in sel], dtype=float)
        nh = np.array([r["needed"]["highlight"]["input_minus_target"] or np.nan for r in sel], dtype=float)
        return {"label": label, "n": len(sel),
                "log2_ratio_median": float(np.median(lr)), "log2_ratio_p10": float(np.percentile(lr, 10)), "log2_ratio_p90": float(np.percentile(lr, 90)),
                "log2_ratio_min": float(lr.min()), "log2_ratio_max": float(lr.max()),
                "needed_mid_median": float(np.nanmedian(nm)), "needed_shadow_median": float(np.nanmedian(ns)), "needed_highlight_median": float(np.nanmedian(nh)),
                "mean_luma_input_median": float(np.median([r["mean_luma_input"] for r in sel])),
                "mean_luma_target_median": float(np.median([r["mean_luma_target"] for r in sel]))}

    trips = sorted({r["trip"] for r in rows if r["trip"]})
    groups = {"all": summarize(rows, "all train+monitor")}
    for t in trips:
        groups[f"trip:{t}"] = summarize([r for r in rows if r["trip"] == t], f"trip {t}")
    groups["batch2_kelvin"] = summarize([r for r in rows if r["trip"] == "Phase4-Capture2" and r["wbManualFlag"] == 1], "batch 2, manual/Kelvin WB")
    groups["batch2_auto"] = summarize([r for r in rows if r["trip"] == "Phase4-Capture2" and r["wbManualFlag"] == 0], "batch 2, auto WB")
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 1.01)):
        sel_b2 = [r for r in rows if r["trip"] == "Phase4-Capture2" and lo <= r["low_key_fraction"] < hi]
        sel_b1 = [r for r in rows if r["trip"] != "Phase4-Capture2" and lo <= r["low_key_fraction"] < hi]
        groups[f"lowkey_{lo}-{hi}_batch2"] = summarize(sel_b2, f"low-key in [{lo},{hi}) batch 2")
        groups[f"lowkey_{lo}-{hi}_rest"] = summarize(sel_b1, f"low-key in [{lo},{hi}) rest of corpus")

    med = groups["all"]["log2_ratio_median"]
    outliers = sorted(rows, key=lambda r: -abs(r["log2_ratio"] - med))[:20]

    report = {"groups": groups, "corpus_log2_ratio_median": med,
              "top_outliers_by_log2_ratio": [{k: r[k] for k in ("name", "tier", "trip", "sessionIndex", "wbManualFlag", "iso", "mean_luma_input", "mean_luma_target", "log2_ratio", "low_key_fraction", "p99_input", "p99_target")} for r in outliers],
              "per_scene": rows}
    (OUT_DIR / "target-exposure-survey.json").write_text(json.dumps(report, indent=2))

    print("\n=== target/input exposure by group: log2(mean target luma / mean input luma) ===")
    for g in groups.values():
        if g["n"] == 0:
            continue
        print(f"{g['label']:42s} n={g['n']:3d}  log2 ratio median {g['log2_ratio_median']:+.3f}  p10 {g['log2_ratio_p10']:+.3f}  p90 {g['log2_ratio_p90']:+.3f}  "
              f"[{g['log2_ratio_min']:+.2f},{g['log2_ratio_max']:+.2f}] | needed mid {g['needed_mid_median']:+.3f} shadow {g['needed_shadow_median']:+.3f} highlight {g['needed_highlight_median']:+.3f} | "
              f"mean luma in {g['mean_luma_input_median']:.3f} tgt {g['mean_luma_target_median']:.3f}")
    print(f"\ncorpus median log2 ratio {med:+.3f}; top outliers:")
    for r in outliers:
        print(f"  {r['name']:14s} {r['tier']:7s} {str(r['trip']):16s} sess={r['sessionIndex']} wbManual={r['wbManualFlag']} iso={r['iso']}  "
              f"luma in {r['mean_luma_input']:.4f} tgt {r['mean_luma_target']:.4f}  log2 {r['log2_ratio']:+.2f}  low_key {r['low_key_fraction']:.2f}  p99 in {r['p99_input']:.3f} tgt {r['p99_target']:.3f}")
    print("\nfinalHeldOut was not accessed by this diagnostic pass.")


if __name__ == "__main__":
    main()
