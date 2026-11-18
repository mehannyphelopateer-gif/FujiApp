"""Classic Cuban Neg evaluation contract - FROZEN single source of truth.

Finalized 2026-11-18 after the repeat floor (training/diagnostics/
cuban-repeat-floor.json, 58816af) and before any app-vs-reference run.
Scoring code must import CONTRACT and refuse to run unless
contract_sha256() equals the value recorded in
docs/phase4-scene-adaptive-scope.md. Changing anything here is a contract
revision: it needs Codex approval, a dated doc entry, and may never be
driven by K or monitor outcomes. Limits are development targets, not a
claim of universal or perceptually indistinguishable Fuji parity.

`python3 training/cuban_contract.py` prints the hash and runs consistency
checks.
"""
import hashlib
import json

CONTRACT = {
    "version": 1,
    "space": {
        "reference": "target_tiff.load_xraw_target_linear_matching: 16-bit TIFF -> linear sRGB, EXIF "
                     "orientation, cv2.INTER_AREA in linear light to the scene's exact "
                     "browser-phase3-linear.fjlrg dimensions",
        "app": "app linear output at the same working dimensions (no resampling if already equal)",
        "lab": "color_metrics.linear_srgb_to_lab (clip [0,1], D65)",
        "delta_e": "color_metrics.ciede2000 (34/34 Sharma pairs exact to 4 dp)",
        "sampling": "dense - every working-scale pixel",
        "scale": "working (.fjlrg) scale for every metric; no full-resolution/40MP claim; "
                 "native-resolution crops are report-only diagnostics",
        "clipped_pixel": "any linear channel >= 0.999 at working scale",
    },
    "repeat_floor": {  # max over both pilots, E2 vs R2 (grain off), working scale
        "mean_dE00": 0.1120, "p95_dE00": 0.2738,
        "detail_rel": {"fine_s1_s3": 3e-05, "mid_s4_s16": 3e-05},
        "grain_std_rel": 0.0006,  # E1/R1 residual-std relative change, max over pilots
        "grain_pixel_pattern_repeatable": False,  # residual corr 0.0188 / 0.0094
    },
    "tone_bands_ref_Lstar": {"shadow": [None, 25], "mid": [25, 75], "high": [75, None]},
    "per_scene_report": ["mean_dE00", "p95_dE00", "signed_dL_by_tone_band",
                         "signed_dC_where_ref_C_gt_10", "signed_hue_deg_where_ref_C_gt_20 (wrapped to +-180)"],
    "stage_limits": {  # per scene; all must hold
        "E4": {"mean_dE00": 2.0, "p95_dE00": 5.0, "abs_mid_dL": 2.0, "rho_provia_to_E4": 0.5},
        "E5": {"mean_dE00": 2.5, "p95_dE00": 6.0, "rho_dr400": 0.5, "clipped_fraction_gap": 0.02},
        "E3": {"mean_dE00": 3.0, "p95_dE00": 7.5, "abs_mid_dL": 2.5, "abs_mean_dC": 3.0, "rho_E5_to_E3": 0.5},
        "E2": {"mean_dE00": 4.0, "p95_dE00": 10.0, "abs_mid_dL": 3.0, "rho_wb_E3_to_E2": 0.6},
    },
    "rho": {
        "definition": "mean_R dE00(Lab(ref_b) + Lab(app_s) - Lab(app_b), Lab(ref_s)) / mean_R dE00(Lab(ref_b), Lab(ref_s))",
        "role": "supplemental contrast test; never replaces absolute limits or visual review",
        "bases": {"rho_provia_to_E4": ["xraw-phase3 (Provia)", "E4"], "rho_dr400": ["E4", "E5"],
                  "rho_E5_to_E3": ["E5", "E3"], "rho_wb_E3_to_E2": ["E3", "E2"]},
    },
    "regions": {  # on the reference BASE image unless stated
        "dr400": "ref E4 L* >= 70 or clipped",
        "C1": "L* >= 60", "C2": "L* <= 35", "C3": "C* >= 20", "C4": "C* >= 30",
        "C5": "hue 200-290 deg and C* >= 15", "C6": "frame", "C7": "frame", "C8": "frame",
        "E3_E4_E2_wb": "frame",
    },
    "measurability": {
        "min_region_fraction": 0.02,
        "min_fuji_effect_mean_dE00": 0.5,
        "min_fuji_effect_vs_floor": 4.0,  # x repeat_floor.mean_dE00 (= 0.448; 0.5 binds)
        "detail_min_vs_floor": 3.0,       # x repeat_floor.detail_rel[band]
        "unmeasurable": "neither pass nor fail; empty region -> null statistics",
    },
    "controls": {  # signed quantity; app sign must match Fuji's measured sign
        "C1-highlight-minus2": {"region": "C1", "signed": "mean dL*", "rho_max": 0.5},
        "C2-shadow-plus1": {"region": "C2", "signed": "mean dL*", "rho_max": 0.5},
        "C3-color-plus4": {"region": "C3", "signed": "mean dC*", "rho_max": 0.5},
        "C4-cce-strong": {"region": "C4", "signed": "mean dL* (dC* reported)", "rho_max": 0.5},
        "C5-fxblue-strong": {"region": "C5", "signed": "mean dL* (dC* reported)", "rho_max": 0.5},
        "C6-nr-minus4": {"region": "C6", "signed": "relative change of fine_s1_s3 L* detail RMS",
                         "detail_band": "fine_s1_s3", "tolerance": "max(0.3*|fuji|, 3*floor)"},
        "C7-clarity-minus4": {"region": "C7", "signed": "relative change of mid_s4_s16 L* detail RMS",
                              "detail_band": "mid_s4_s16", "tolerance": "max(0.3*|fuji|, 3*floor)"},
        "C8-wbshift-r4-b5": {"region": "C8", "signed": "mean (da*, db*)", "direction_cosine_min": 0.8,
                             "rho_max": 0.5},
        "min_measurable_scenes": 6,  # of 8 C scenes; else UNVALIDATED -> more train-only examples
        "unvalidated": "never an automatic pass; recipe cannot be reported validated for that setting",
    },
    "grain_E1": {  # residual g = E1 - E2 (L*, a*, b*), app vs Fuji, working scale; no pixelwise dE
        "bands": "ref E2 L* quintile thresholds (ties kept together; occupancy reported)",
        "std_ratio_per_band": [0.75, 1.33],
        "psd": "radial power spectrum of g_L, Hann window, frequency-weighted centroid over 0.02-0.5 cyc/px",
        "psd_centroid_ratio": [0.80, 1.25],
        "chroma_std_ratio": [0.5, 2.0], "chroma_both_below": 0.3,
        "abs_mean_residual_gap_L": 0.5,
        "visual": "user review of shadow/mid/high working-scale crops per scene, recorded",
    },
    "whole_recipe": {
        "scene_pass": "conjunction: E2 + E4 + E5 + E3 per-scene limits, E1 grain statistics, no visible artifact",
        "set_pass": {"K": "scenes passing conjunction >= 7 of 8", "monitor": ">= 11 of 12 (later, dev-validation)"},
        "aggregate": {"median_mean_dE00": {"E4": 1.5, "E3": 2.5, "E2": 3.0},
                      "abs_cross_scene_mean_mid_dL": 1.0, "abs_cross_scene_mean_dC": 1.5,
                      "abs_cross_scene_mean_hue_deg": 3.0},
        "artifact": "banding, halos, colour blotches or hard clipping onset fails the scene",
        "report": "every scene, signed errors, each failed requirement, visual outcome; "
                  "separate stage pass rates are diagnostic only",
    },
    "sets": {"F": "fit (may drive fitting, never limits)", "K": "internal development check, shares sessions with F",
             "monitor12": "development-validation, excluded from fitting", "finalHeldOut": "locked, never read"},
}


def contract_sha256():
    return hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()


def _check():
    m, f = CONTRACT["measurability"], CONTRACT["repeat_floor"]
    assert m["min_fuji_effect_vs_floor"] * f["mean_dE00"] < m["min_fuji_effect_mean_dE00"]
    # repeat floor must sit within the predeclared hold rule it was measured under
    assert f["mean_dE00"] <= 0.25 * 1.5 and f["p95_dE00"] <= 0.25 * 5.0
    for name, c in CONTRACT["controls"].items():
        if isinstance(c, dict):
            assert c["region"] in CONTRACT["regions"], name
            if "detail_band" in c:
                assert c["detail_band"] in f["detail_rel"], name


if __name__ == "__main__":
    _check()
    print("contract consistent; sha256", contract_sha256())
