"""Verify F24 diagnostic coverage, reproduction and derived arithmetic; no inference."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_provia_distill as D

P = D.P
MANIFEST = P.DIAG / "provia-distill-fit-audit-manifest.json"
REPORT = P.DIAG / "provia-distill-fit-audit.json"
OUTPUT = P.DIAG / "provia-distill-fit-audit-verification.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(recorded, expected, label):
    if expected is None:
        require(recorded is None, label + " should be null")
    else:
        require(type(recorded) in (int, float) and math.isfinite(recorded)
                and abs(recorded - expected) <= 1e-10, "Arithmetic mismatch: " + label)


def ratio(start, end):
    require(math.isfinite(start) and math.isfinite(end) and start >= 0 and end >= 0, "Invalid distance")
    return None if start <= 1e-12 else 1 - end / start


def verify(expected_sha):
    require(not OUTPUT.exists(), "Verification output exists")
    require(P.sha256(MANIFEST) == expected_sha, "Manifest changed")
    manifest, result, old = [json.loads(p.read_text()) for p in (MANIFEST, REPORT, D.REPORT)]
    require(result["manifest_sha256"] == expected_sha, "Wrong report manifest")
    require(P.sha256(D.REPORT) == manifest["bindings"]["OOF_report"], "Frozen pilot report changed")
    names = sorted(s["name"] for s in manifest["scenes"])
    require(len(names) == 24 and len(set(names)) == 24 and sorted(result["baselines"]) == names, "Baseline coverage")
    require(sorted(result["endpoint_rows"]) == sorted(str(f["fold"]) for f in manifest["folds"]), "Fold coverage")
    checks, own_fit, distill_fit, distill_oof = {}, [], [], []
    for name in names:
        error = abs(result["baselines"][name]["fuji"]["mean_dE00"] - old["per_scene"][name]["r11"]["mean_dE00"])
        require(error <= 1e-8, "R11 reproduction mismatch")
    for fold in manifest["folds"]:
        fid = str(fold["fold"])
        require(set(result["endpoint_rows"][fid]) == set(D.ARMS), "Arm coverage")
        for arm in D.ARMS:
            endpoint = result["endpoint_rows"][fid][arm]
            require(sorted(endpoint) == names, "Endpoint scene coverage")
            summary = result["fold_summaries"][fid][arm]
            require(summary["own_target"] == ("fuji" if arm == "direct" else "teacher"), "Own target label")
            for subset in ("fit", "adaptation_OOF"):
                selected = fold[subset]
                for name in selected:
                    require(endpoint[name]["subset"] == subset, "Subset label mismatch")
                    if subset == "adaptation_OOF":
                        require(abs(endpoint[name]["fuji"]["mean_dE00"] - old["per_scene"][name][arm]["mean_dE00"]) <= 1e-8,
                                "Frozen adaptation-OOF reproduction mismatch")
                for target in ("fuji", "teacher"):
                    r = summary[subset][target]
                    require(r["n"] == len(selected), "Subset count mismatch")
                    for kind, field in (("objective", "combined"), ("display_dE", "mean_dE00")):
                        values = lambda image: image["objective"][field] if kind == "objective" else image[field]
                        before = [values(result["baselines"][n][target]) for n in selected]
                        after = [values(endpoint[n][target]) for n in selected]
                        start, end = float(np.mean(before)), float(np.mean(after))
                        per_scene = [ratio(b, a) for b, a in zip(before, after)]
                        valid = [v for v in per_scene if v is not None]
                        for key, expected in (("start_" + kind, start), ("end_" + kind, end),
                                              (kind + "_closure", ratio(start, end)),
                                              (kind + "_closure_scene_median", float(np.median(valid)) if valid else None),
                                              (kind + "_closure_scene_nulls", len(per_scene) - len(valid))):
                            close(r[key], expected, fid + arm + subset + target + key)
                    checks[f"f{fid}-{arm}-{subset}-{target}"] = "PASS"
            own = summary["fit"][summary["own_target"]]["objective_closure"]
            own_fit.append(own)
            if arm == "distill":
                distill_fit += [summary["fit"]["teacher"][k] for k in ("objective_closure", "display_dE_closure")]
                distill_oof += [summary["adaptation_OOF"]["teacher"][k] for k in ("objective_closure", "display_dE_closure")]
    displayed_fit = [result["fold_summaries"][str(f["fold"])]["distill"]["fit"]["teacher"]["display_dE_closure"] for f in manifest["folds"]]
    if any(v is None for v in own_fit + displayed_fit + distill_oof):
        label = "INCONCLUSIVE_zero_baseline"
    elif all(v < .25 for v in own_fit + displayed_fit):
        label = "A_under_attained_fit_targets_under_this_budget"
    elif all(v >= .5 for v in distill_fit) and all(v <= .1 for v in distill_oof):
        label = "B_fit_teacher_learning_did_not_transfer_in_this_pilot"
    else:
        label = "MIXED_OR_INCONCLUSIVE"
    require(result["classification"] == label, "Classification mismatch")
    out = {"verification": "PASS", "scope": "Arithmetic/coverage verification only; retrospective train-only diagnostic",
           "manifest_sha256": expected_sha, "report_sha256": P.sha256(REPORT),
           "verifier_sha256": P.sha256(Path(__file__)), "classification": label, "checks": checks,
           "all_R11_and_adaptation_OOF_scores_reproduce_within": 1e-8}
    OUTPUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-sha", required=True)
    verify(p.parse_args().manifest_sha)
