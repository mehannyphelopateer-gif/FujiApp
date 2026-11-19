"""Correct correlations from committed F24 numbers, without rereading images.

Preserves the original decomposition report. No inference, fitting or K,
monitor, finalHeldOut or T8 access. Run from any directory.
"""
import hashlib
import json
from pathlib import Path

from rank_metrics import spearman

DIAG = Path(__file__).resolve().parent / "diagnostics"
SOURCE = DIAG / "cn-step1-f24-decomposition.json"
OUT = DIAG / "cn-step1-f24-correlations-tie-corrected.json"


def main():
    source_bytes = SOURCE.read_bytes()
    data = json.loads(source_bytes)
    rows = data["per_scene"]
    cohort = json.loads((DIAG / "cuban-cohort-proposal.json").read_text())
    allowed = {s["name"] for s in cohort["scenes"] if s["role"] == "fit" and s["tier"] == "train"}
    if len(rows) != 24 or set(rows) != allowed:
        raise ValueError("Report must cover exactly the approved F24")
    values = [rows[n] for n in sorted(rows)]
    column = lambda f: [f(r) for r in values]
    metrics = {
        "r11_vs_provia_mean_dE00": column(lambda r: r["1_r11_vs_provia"]["mean_dE00"]),
        "cn_vs_e4_mean_dE00": column(lambda r: r["2_cn_vs_e4"]["mean_dE00"]),
        "oracle_mean_dE00": column(lambda r: r["3_oracle_anchored_provia_plus_D_vs_e4"]["mean_dE00"]),
        "fuji_provia_e4_mean_dE00": column(lambda r: r["2_fuji_provia_vs_e4"]["mean_dE00"]),
        "e_base_mean_da": column(lambda r: r["1_r11_vs_provia"]["frame"]["mean_da"]),
        "e_base_mean_db": column(lambda r: r["1_r11_vs_provia"]["frame"]["mean_db"]),
        "e_base_mean_dL": column(lambda r: r["1_r11_vs_provia"]["frame"]["mean_dL"]),
        "e_delta_mean_da": column(lambda r: r["3_decomposition_e4_bands"]["frame"]["e_delta"]["mean_da"]),
        "e_delta_mean_db": column(lambda r: r["3_decomposition_e4_bands"]["frame"]["e_delta"]["mean_db"]),
    }
    features = {
        "shift_R_steps": column(lambda r: r["source_wb_shift_steps"][0]),
        "shift_B_steps": column(lambda r: r["source_wb_shift_steps"][1]),
        "log_gain_ratio_R": column(lambda r: r["log_gain_ratio_source_over_ref"][0]),
        "log_gain_ratio_B": column(lambda r: r["log_gain_ratio_source_over_ref"][1]),
    }
    if set(metrics) != set(data["spearman_vs_source_wb"]):
        raise ValueError("Unexpected original correlation coverage")
    corrected = {m: {f: spearman(v, fv) for f, fv in features.items()} for m, v in metrics.items()}
    result = {
        "scope": "F24 committed scalar report only; no new pixel reads, inference or fitting",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "method": "Pearson correlation of ranks, ties assigned their average rank",
        "note": "Original report preserved. Still descriptive and confounded with WB mode/session. Frame-offset energy is not WB-attributable energy.",
        "original_correlations": data["spearman_vs_source_wb"],
        "corrected_correlations": corrected,
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(corrected, indent=2))


if __name__ == "__main__":
    main()
