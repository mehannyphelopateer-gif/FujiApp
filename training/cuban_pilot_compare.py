"""Classic Cuban Neg export pilot - descriptive pairwise comparison of the
INSTALLED pilot TIFFs (after `cuban_exports.py verify pilot` + `install pilot`).
No limits, no scoring, no fitting: this only reports what X RAW Studio changed.

  E2 vs E3: Auto WB vs As-Shot WB (rest of recipe identical) - Fuji's output WB
            change. NOT the app's AWB error.
  E4 vs E5: output DR100 vs DR400 (Classic Neg only) - Fuji's DR400 tone change.

  python3 training/cuban_pilot_compare.py   # -> training/diagnostics/cuban-pilot-compare-v2.json

Installed files are re-hashed against the pilot verification report first.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuban_exports import BATCHES, CAL, INSTALL_NAME, REPORT, sha256  # noqa: E402

OUT = Path(__file__).resolve().parent / "diagnostics" / "cuban-pilot-compare-v2.json"
STEP = 4  # spatial subsample
LUM = np.array([0.2126, 0.7152, 0.0722])  # on sRGB-encoded values; descriptive only
BANDS = [("below_p25", 0, 25), ("p25_to_below_p75", 25, 75),
         ("p75_to_below_p99", 75, 99), ("at_or_above_p99", 99, 100)]


def load(path):
    return tifffile.imread(path)[::STEP, ::STEP].astype(np.float64) / 65535.0


def wb_pair(a, b):
    """a = E2 (Auto), b = E3 (As-Shot)."""
    ok = np.all((a > 0.02) & (a < 0.98) & (b > 0.02) & (b < 0.98), axis=2)
    la, lb = np.log2(a[ok]), np.log2(b[ok])
    d = lb - la  # per-channel log2(E3/E2)
    ma, mb = a[ok].mean(0), b[ok].mean(0)
    return {
        "sampled_pixels_identical": bool(np.array_equal(a, b)),
        "valid_pixel_fraction": round(float(ok.mean()), 4),
        "mean_abs_diff_8bit_units": round(float(np.abs(b - a).mean() * 255), 3),
        "p99_abs_diff_8bit_units": round(float(np.percentile(np.abs(b - a), 99) * 255), 3),
        "mean_log2_ratio_E3_over_E2_rgb": [round(float(x), 4) for x in d.mean(0)],
        "median_log2_ratio_E3_over_E2_rgb": [round(float(x), 4) for x in np.median(d, 0)],
        "mean_rgb_E2": [round(float(x), 4) for x in ma],
        "mean_rgb_E3": [round(float(x), 4) for x in mb],
        "chroma_E2_r_over_g_b_over_g": [round(float(ma[0] / ma[1]), 4), round(float(ma[2] / ma[1]), 4)],
        "chroma_E3_r_over_g_b_over_g": [round(float(mb[0] / mb[1]), 4), round(float(mb[2] / mb[1]), 4)],
        "mean_luma_log2_ratio": round(float(np.log2((b[ok] @ LUM) / (a[ok] @ LUM)).mean()), 4),
    }


def dr_pair(a, b):
    """Bands use E4 percentile thresholds, not fixed-size rank quantiles.

    Equal values stay together in the upper interval. Consequently the
    >=p99 interval can contain much more than 1% of an image. Report its
    actual occupancy before and after the near-black exclusion.
    """
    ya, yb = a @ LUM, b @ LUM
    out = {"sampled_pixels_identical": bool(np.array_equal(a, b)),
           "mean_abs_diff_8bit_units": round(float(np.abs(b - a).mean() * 255), 3),
           "clipped_fraction_any_channel": {"E4": round(float((a >= 0.999).any(2).mean()), 5),
                                            "E5": round(float((b >= 0.999).any(2).mean()), 5)},
           "black_fraction_any_channel": {"E4": round(float((a <= 0.001).any(2).mean()), 5),
                                          "E5": round(float((b <= 0.001).any(2).mean()), 5)},
           "luma_percentiles": {}, "bands_by_E4_luma": {},
           "band_definition": "E4 encoded-luma percentile thresholds; ties stay together, not fixed-size quantiles"}
    for p in (1, 5, 25, 50, 75, 95, 99, 99.9):
        out["luma_percentiles"][str(p)] = [round(float(np.percentile(ya, p)), 4),
                                           round(float(np.percentile(yb, p)), 4)]
    edges = np.percentile(ya, [0, 25, 75, 99, 100])
    lo_hi = dict(zip([0, 25, 75, 99, 100], edges))
    for name, lo, hi in BANDS:
        m = (ya >= lo_hi[lo]) & ((ya < lo_hi[hi]) if hi < 100 else (ya <= lo_hi[hi]))
        selected = int(m.sum())
        m &= (ya > 0.002) & (yb > 0.002)
        valid = int(m.sum())
        out["bands_by_E4_luma"][name] = {
            "selected_pixels": selected,
            "selected_fraction": round(selected / ya.size, 6),
            "pixels": valid,
            "valid_fraction": round(valid / ya.size, 6),
            "mean_luma_E4": round(float(ya[m].mean()), 4) if valid else None,
            "mean_luma_E5": round(float(yb[m].mean()), 4) if valid else None,
            "mean_log2_ratio_E5_over_E4": round(float(np.log2(yb[m] / ya[m]).mean()), 4) if valid else None}
    return out


def main():
    try:
        rep = json.loads(REPORT["pilot"].read_text())
    except Exception as e:
        sys.exit(f"refused: pilot verification report unusable ({type(e).__name__}) - run verify + install first")
    if rep.get("passed") is not True:
        sys.exit("refused: pilot verification has not passed")
    verified = {(e["name"], e["variant"]): e["source_sha256"] for e in rep["entries"]}
    rows = json.loads(BATCHES["pilot"][1].read_text())
    res = {"schema_version": 2,
           "note": "descriptive sampled-pixel comparisons of encoded sRGB; log2 ratios are not physical exposure stops; no limits, no scoring, no fitting",
           "supersedes": "cuban-pilot-compare.json (retained unchanged; percentile-band occupancy was mislabeled)",
           "scenes": {}}
    for r in rows:
        paths = {v: CAL / r["shoot"] / INSTALL_NAME[v] for v in
                 ("E2-no-grain", "E3-as-shot-wb", "E4-classic-neg-only", "E5-classic-neg-dr400")}
        for v, p in paths.items():
            if sha256(p) != verified[(r["name"], v)]:
                sys.exit(f"refused: installed {p.relative_to(CAL)} != verified TIFF")
        img = {v: load(p) for v, p in paths.items()}
        wb = {e["variant"]: e["wb"] for e in rep["entries"] if e["name"] == r["name"]}
        res["scenes"][r["name"]] = {
            "asShotWhiteBalance": r.get("asShotWhiteBalance"), "highlightClass": r.get("highlightClass"),
            "wb_metadata": {"E2": wb["E2-no-grain"]["export"], "E3": wb["E3-as-shot-wb"]["export"],
                            "raf": wb["E2-no-grain"]["raf"]},
            "E2_vs_E3_wb": wb_pair(img["E2-no-grain"], img["E3-as-shot-wb"]),
            "E4_vs_E5_dr400": dr_pair(img["E4-classic-neg-only"], img["E5-classic-neg-dr400"])}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
