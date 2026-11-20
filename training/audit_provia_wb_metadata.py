"""Archived F24 WB metadata/scalar audit. Never opens RAF, TIFF or pixel caches."""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

import provia_color_pilot as P
from audit_provia_distill_artifacts import committed
from rank_metrics import spearman

MANIFEST = P.DIAG / "provia-wb-metadata-manifest.json"
REPORT = P.DIAG / "provia-wb-metadata-audit.json"
SOURCES = ("cn-step1-wb-reference-audit.json", "cn-step1-wb-input-probe-all-F.json",
           "cn-step1-f24-decomposition.json", "cuban-cohort-proposal.json")
LIMIT = .005


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def steps(raw):
    require(isinstance(raw, list) and len(raw) == 2, "Missing fine-tune pair")
    require(all(type(v) is int and v % 20 == 0 and -180 <= v <= 180 for v in raw), "Malformed raw fine-tune units")
    return tuple(v // 20 for v in raw)


def normalized(row):
    vals = [row[k] for k in ("WBLevelR", "WBLevelG", "WBLevelB")]
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in vals), "Invalid recorded WB levels")
    return [vals[0] / vals[1], vals[2] / vals[1]]


def correlation(x, y):
    if len(x) < 2:
        return {"n": len(x), "rho": None, "reason": "fewer than two observations"}
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(x), "rho": None, "reason": "constant variable"}
    return {"n": len(x), "rho": spearman(x, y), "reason": None}


def group_stats(rows):
    ratios = np.array([r["target_over_source_normalized_RB"] for r in rows], dtype=float)
    med = np.median(ratios, axis=0)
    spread = (ratios.max(0) - ratios.min(0)) / med
    sessions = sorted(set(r["session"] for r in rows))
    modes = sorted(set(r["source_wb_mode"] for r in rows))
    measurable = len(rows) >= 2
    coverage = measurable and len(sessions) >= 2 and len(modes) >= 2
    return {"n": len(rows), "names": sorted(r["name"] for r in rows), "sessions": sessions, "wb_modes": modes,
            "ratio_min_RB": ratios.min(0).tolist(), "ratio_max_RB": ratios.max(0).tolist(),
            "ratio_median_RB": med.tolist(), "relative_spread_RB": spread.tolist(),
            "within_group_consistent_at_0_5pct": bool(np.all(spread < LIMIT)) if measurable else None,
            "cross_session_and_mode_coverage": coverage,
            "universal_fine_tune_only_mapping_validated": bool(np.all(spread < LIMIT)) if coverage else None,
            "note": "Singletons unmeasurable. Single-mode groups cannot validate fine-tune-only determinism. Tiny ratio drift/correlations can reflect quantised coefficients and mathematical coupling, not necessarily a different WB space."}


def prepare():
    require(not MANIFEST.exists(), "Metadata manifest already exists")
    P.load_manifest()
    sources = {n: P.sha256(P.DIAG / n) for n in SOURCES}
    out = {
        "scope": "F24 archived metadata and existing residual scalars only; no RAF/TIFF/pixel-cache reads, inference, image fitting or app change",
        "source_hashes": sources, "pilot_manifest": P.MANIFEST_SHA, "source_R11": P.SOURCE_SHA,
        "code": {f: P.sha256(P.TRAIN / f) for f in (Path(__file__).name, "rank_metrics.py")},
        "residual_sign_producer_sha256": P.sha256(P.TRAIN / "diagnose_cn_step1_f24.py"),
        "group_rules": "Raw MakerNote fine-tune /20 must be an integer in[-9,9]. Compute groups only from those raw fields. Relative spread=(max-min)/median; require <0.005 on both ratios. At least2 scenes,2 sessions and2 WB modes required to validate fine-tune-only determinism. Otherwise unvalidated; singleton unmeasurable.",
        "step_check": "Most-supported group with both shifts nonzero, tie lexical. Anchor each channel's unit log slope to that group's median target/source ratio and own shift. Predict other group medians independently per channel, no cross terms or tuning; all relative errors<=0.005 required. Target-derived descriptive reference only, never runtime input.",
        "residual_rule": "Existing frame Lab residual=R11 prediction minus Fuji Provia. Retrospective signs: R step positive with a*/b*, B step negative with a*/b*. Per-session eligibility n>=3 and at least2 fine-tune pairs, per-axis constants null. Require predicted direction in at least2 eligible sessions per one of all4 associations to label retrospective sign consistency; never causal validation.",
        "retrospective": "Ratios, old correlations and the expected simple-step-model failure were already visible before this audit. Results verify/describe existing evidence, not a blind hypothesis test. Mode/session/content confounding remains.",
        "gain_interpretation": "LibRaw archived cam_mul exactly matching source recorded levels is measured. Interpreting proprietary TIFF WB levels as sensor gains and exact post-scale decoder multipliers remains inferred; no new measurement of internal scale_colors is performed.",
    }
    MANIFEST.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print("WB metadata manifest sha256", P.sha256(MANIFEST))


def run(expected_sha):
    require(not REPORT.exists(), "Metadata report already exists")
    require(P.sha256(MANIFEST) == expected_sha, "Metadata manifest changed")
    committed(MANIFEST)
    committed(Path(__file__).resolve())
    man = json.loads(MANIFEST.read_text())
    for f, h in man["code"].items():
        require(P.sha256(P.TRAIN / f) == h, "Bound code changed")
    for f, h in man["source_hashes"].items():
        require(P.sha256(P.DIAG / f) == h, "Archived metadata/scalar evidence changed")
    require(P.sha256(P.TRAIN / "diagnose_cn_step1_f24.py") == man["residual_sign_producer_sha256"], "Residual sign-producer code changed")
    data = P.load_manifest()
    pool = {s["name"]: s for s in data["scenes"]}
    audit, probe, residual, cohort = [json.loads((P.DIAG / n).read_text()) for n in SOURCES]
    fit = {s["name"]: s for s in cohort["scenes"] if s["role"] == "fit"}
    archived = {s["name"]: s for s in audit["scenes"]}
    probes = {s["name"]: s for s in probe["scenes"]}
    require(len(fit) == len(archived) == len(probes) == 24 and set(fit) == set(archived) == set(probes) == set(residual["per_scene"]), "Exact F24 coverage mismatch")
    require(len(audit["scenes"]) == len(probe["scenes"]) == 24 and len([s for s in cohort["scenes"] if s["role"] == "fit"]) == 24, "Duplicate F24 records")
    require(set(fit) <= set(pool), "Non-allowlisted source")
    require(residual["fingerprints"]["r11_seed42"] == P.SOURCE_SHA, "Wrong residual source model")
    require(audit["libraw_probe_sha256"] == man["source_hashes"][SOURCES[1]] and probe["cohort_sha256"] == man["source_hashes"][SOURCES[3]], "Archived probe/cohort chain mismatch")
    require(residual["fingerprints"]["wb_audit"] == man["source_hashes"][SOURCES[0]], "Residual/WB audit chain mismatch")
    rows, groups = {}, collections.defaultdict(list)
    for n in sorted(fit):
        s, a, p = fit[n], archived[n], probes[n]
        src, tar = a["source"], a["provia"]
        shift = steps(src["WBFineTune"])
        require(s["tier"] == "train" and s["shoot"] == a["shoot"] == pool[n]["shoot"], "Source scene identity mismatch")
        require(a["frozen_input_sha256"] == pool[n]["fjlrg_sha256"] and a["source_raf_sha256_at_probe"] == p["source_raf_sha256"], "Archived source fingerprint mismatch")
        require(p["cam_mul"][:3] == [src[k] for k in ("WBLevelR", "WBLevelG", "WBLevelB")], "LibRaw/source numeric levels differ")
        require(src["WBFineTune"] != [0, 0] and src["WhiteBalance"] == int(s["asShotWhiteBalance"], 16) == p["fujiWB"], "Source WB context mismatch")
        require(tar["WBFineTune"] == [0, 0] and tar["WhiteBalance"] == src["WhiteBalance"] and tar["ColorTemperature"] == src["ColorTemperature"], "Neutral target WB settings mismatch")
        require(not a["provia_neutral_settings_mismatches"] and pool[n]["captureDynamicRange"] == 400, "Neutral protocol/capture DR mismatch")
        source, target = normalized(src), normalized(tar)
        f = residual["per_scene"][n]
        require(f["session"] == s["session"] == pool[n]["session"] and f["source_wb_shift_steps"] == list(shift), "Residual metadata identity mismatch")
        row = {"name": n, "shoot": s["shoot"], "session": s["session"], "iso": s["iso"],
               "source_wb_mode": src["WhiteBalance"], "source_kelvin": src["ColorTemperature"],
               "source_fine_tune_raw": src["WBFineTune"], "shift_steps": list(shift),
               "stored_libraw_cam_mul": p["cam_mul"], "source_normalized_RB": source,
               "zero_shift_target_normalized_RB": target,
               "target_over_source_normalized_RB": [target[i] / source[i] for i in (0, 1)],
               "R11_minus_Provia_frame_a_b": [f["1_r11_vs_provia"]["frame"][k] for k in ("mean_da", "mean_db")]}
        rows[n] = row
        groups[shift].append(row)
    stats = {str(k): group_stats(v) for k, v in sorted(groups.items())}
    reference = sorted((k for k in groups if k[0] and k[1]), key=lambda k: (-len(groups[k]), k))[0]
    median = np.array(stats[str(reference)]["ratio_median_RB"])
    slopes = np.log(median) / np.array(reference)
    step_results = {}
    for key in sorted(groups):
        observed = np.array(stats[str(key)]["ratio_median_RB"])
        predicted = np.exp(slopes * np.array(key))
        errors = np.abs(predicted / observed - 1)
        step_results[str(key)] = {"observed_RB": observed.tolist(), "predicted_RB": predicted.tolist(),
                                  "relative_error_RB": errors.tolist(), "within_0_5pct": bool(np.all(errors <= LIMIT))}
        base = groups[key]
        stats[str(key)]["descriptive_ratio_vs_base"] = {
            f"ratio_{c}_vs_zero_shift_base_{b}": correlation([r["zero_shift_target_normalized_RB"][j] for r in base], [r["target_over_source_normalized_RB"][i] for r in base])
            for i, c in enumerate(("R", "B")) for j, b in enumerate(("R", "B"))}
        kelvin = [r for r in base if r["source_kelvin"] is not None]
        stats[str(key)]["ratio_vs_kelvin"] = {c: correlation([r["source_kelvin"] for r in kelvin], [r["target_over_source_normalized_RB"][i] for r in kelvin]) for i, c in enumerate(("R", "B"))}
    correlations, eligible = {}, []
    for session in ["all"] + sorted(set(r["session"] for r in rows.values())):
        selected = list(rows.values()) if session == "all" else [r for r in rows.values() if r["session"] == session]
        ok = len(selected) >= 3 and len({tuple(r["shift_steps"]) for r in selected}) >= 2
        if session != "all" and ok:
            eligible.append(session)
        correlations[str(session)] = {"n": len(selected), "eligible": ok, "associations": {}}
        for i, axis in enumerate(("R", "B")):
            for j, color in enumerate(("a", "b")):
                key = axis + "_vs_" + color
                rec = correlation([r["shift_steps"][i] for r in selected], [r["R11_minus_Provia_frame_a_b"][j] for r in selected]) if ok else {"n": len(selected), "rho": None, "reason": "session not eligible"}
                rec["expected_sign_retrospective"] = 1 if axis == "R" else -1
                correlations[str(session)]["associations"][key] = rec
    supporting = {k: [s for s in eligible if (rec := correlations[str(s)]["associations"][k])["rho"] is not None and rec["rho"] * rec["expected_sign_retrospective"] > 0]
                  for k in ("R_vs_a", "R_vs_b", "B_vs_a", "B_vs_b")}
    out = {"scope": man["scope"], "manifest_sha256": expected_sha,
           "source_zero_shift_target_metadata_mismatch_count": len(rows), "rows": rows, "groups": stats,
           "universal_shift_only_mapping": "UNVALIDATED: no tested group has both cross-session and cross-WB-mode coverage" if not any(g["cross_session_and_mode_coverage"] for g in stats.values()) else "See covered groups",
           "step_check": {"retrospective_known_failure_check": True, "reference_shift": list(reference),
                          "per_channel_unit_log_slopes": slopes.tolist(), "groups": step_results,
                          "all_groups_within_0_5pct": all(v["within_0_5pct"] for v in step_results.values())},
           "retrospective_residual_associations": correlations, "eligible_sessions": eligible,
           "sessions_supporting_each_predicted_sign": supporting,
           "retrospective_sign_consistency_in_at_least_two_sessions_each": all(len(v) >= 2 for v in supporting.values()),
           "limits": "Archived metadata, not fresh decoder applied-gain measurements; proprietary TIFF gain interpretation inferred. Shift/mode/session/content confounded; ratio/base correlations mathematically coupled and quantised. No causal attribution, new runtime calibration, production decoder change, fitting or model selection."}
    P.check_source()
    REPORT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "groups", "retrospective_residual_associations")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--manifest-sha")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        require(bool(args.manifest_sha), "run requires frozen manifest SHA")
        run(args.manifest_sha)
