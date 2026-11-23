"""Prospective fit-only inference diagnostic of absolute error and guide use.

No optimisation, new model selection, development-image access or app change.
The cohort is chosen only from frozen fit metadata, never endpoint errors.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

import matched_gamma_data as D
import run_matched_gamma_training as T
from model import compute_luma_guide

PARENT_SHA = "ff0fad18b4fd6ca9e8ca36e728e526b95bf975b6dd07f88da844db50089e62b2"
MANIFEST = D.DIAG / "gamma-guide-diagnostic-manifest.json"
REPORT = D.DIAG / "gamma-guide-diagnostic-report.json"
CASES = D.DIAG / "gamma-guide-error-cases"
INVENTORY = D.DIAG / "provia-v2-source-inventory.json"
PROOF = D.DIAG / "matched-gamma-final-verification.json"
BANDS = ("shadow", "mid", "high")
CODE = ("training/diagnose_gamma_guide.py", "training/test_diagnose_gamma_guide.py",
        "training/gamma_guide_metrics.py", "training/test_gamma_guide_metrics.py")


def bucket(mode):
    D.need(type(mode) is int, "Invalid source WB code")
    if mode == 0:
        return "auto-normal"
    if mode == 1:
        return "auto-white-priority"
    D.need(mode in (256, 512, 770, 4080), "Unknown source WB code")
    return "manual-preset"


def choose_cohort(parent_rows, inventory_rows):
    """Four per nonempty DR x WB stratum, <=2/session/stratum, no backfill.

    Round-robin ascending session IDs; inside a session sort by a salted
    filename SHA. Neither error values nor pixel statistics enter selection.
    """
    inv = {s["name"]: s for s in inventory_rows}
    D.need(len(inv) == len(inventory_rows), "Duplicate source inventory name")
    groups = defaultdict(lambda: defaultdict(list))
    for s in parent_rows:
        if s["role"] != "fit":
            continue
        src = inv[s["name"]]
        D.need(src["shoot"] == s["shoot"], "Source identity mismatch")
        D.need(s["captureDynamicRange"] in (200, 400), "Unexpected fit DR")
        key = (s["captureDynamicRange"], bucket(src["source_wb_mode"]))
        row = {k: s[k] for k in ("name", "shoot", "role", "session", "captureDynamicRange")}
        row.update(source_wb_mode=src["source_wb_mode"], wb_bucket=key[1], stratum=f"DR{key[0]}/{key[1]}")
        groups[key][s["session"]].append(row)
    selected, strata = [], []
    for dr in (200, 400):
        for wb in ("auto-normal", "auto-white-priority", "manual-preset"):
            sessions = groups[(dr, wb)]
            for rows in sessions.values():
                rows.sort(key=lambda r: hashlib.sha256(("gamma-guide-v1/" + r["name"]).encode()).hexdigest())
            chosen = []
            for rank in range(2):
                for session in sorted(sessions):
                    rows = sessions[session]
                    if rank < len(rows) and len(chosen) < 4:
                        chosen.append(rows[rank])
            selected.extend(chosen)
            strata.append({"stratum": f"DR{dr}/{wb}", "eligible": sum(map(len, sessions.values())),
                           "selected": len(chosen), "requested": 4, "short": len(chosen) < 4})
    D.need(len({r["name"] for r in selected}) == len(selected) > 0, "Empty or duplicate diagnostic cohort")
    return selected, strata


def prepare():
    m = D.load_manifest(PARENT_SHA)
    # The parent's already-bound G manifest fixes this source inventory SHA.
    gm = json.loads((D.DIAG / "provia-v2-train-export-manifest.json").read_text())
    expected_inventory = gm["bindings"][str(INVENTORY.relative_to(D.ROOT))]
    D.need(D.sha(INVENTORY) == expected_inventory, "Source inventory changed")
    inv = json.loads(INVENTORY.read_text())
    cohort, strata = choose_cohort(m["scenes"], inv["rows"])
    proof = json.loads(PROOF.read_text())
    D.need(proof["verification_passed"] is True and proof["manifest_sha256"] == PARENT_SHA,
           "Missing completed parent artifact verification")
    entries = proof["artifacts"]
    D.need({(e["seed"], e["arm"]) for e in entries} == {(s, a) for s in D.SEEDS for a in D.ARMS}
           and len(entries) == 6, "Incomplete parent endpoint coverage")
    files = [D.ROOT / f for f in CODE] + [INVENTORY, PROOF, D.MANIFEST]
    manifest = {
        "scope": __doc__.strip(), "parent_manifest_sha256": PARENT_SHA,
        "bindings": {str(p.relative_to(D.ROOT)): D.sha(p) for p in files},
        "cohort_rule": "Four per DR200/400 x AutoNormal/AutoWhitePriority/manual-preset, round-robin session then salted name SHA, <=2 per session per stratum. Short/empty strata reported, never filled based on pixels or errors.",
        "cohort": cohort, "strata": strata, "endpoints": entries,
        "repeat_limits": {"whole_mean_dE00_abs": 1e-6, "whole_p95_dE00_abs": 1e-5},
        "regions": "Target Lab L<25, 25<=L<75, L>=75; unsigned DE00/absolute Lab errors plus signed bias context; empty bands null, not zero-error passes.",
        "guide": "Actual model rec709 float32 guide, clamped [0,1], depth9 align_corners=True; fractional trilinear plane masses, not hard-bin counts alone.",
        "interpretation": "Descriptive mechanism diagnostic, deliberately stratified, not corpus accuracy or acceptance. Concentrated guide use plus shadow-error excess is consistent with a guide hypothesis, never causal proof. No new fitting, model swap, protected-tier reads, or redefinition of the completed parent result.",
    }
    D.write_once_json(MANIFEST, manifest)
    print("Diagnostic manifest SHA", D.sha(MANIFEST), "cohort", len(cohort), flush=True)
    print(json.dumps(strata, indent=2), flush=True)


def load(expected):
    D.need(D.sha(MANIFEST) == expected and D.committed(MANIFEST), "Unfrozen/changed diagnostic manifest")
    manifest = json.loads(MANIFEST.read_text())
    for name, digest in manifest["bindings"].items():
        p = D.ROOT / name
        D.need(D.sha(p) == digest and D.committed(p), "Diagnostic binding changed: " + name)
    m = D.load_manifest(PARENT_SHA)
    cohort, strata = choose_cohort(m["scenes"], json.loads(INVENTORY.read_text())["rows"])
    D.need(cohort == manifest["cohort"] and strata == manifest["strata"], "Diagnostic selection does not reproduce")
    return manifest, m


def aggregate(cases):
    """Equal-scene descriptive summaries; sample is not corpus representative."""
    result = {}
    for seed in D.SEEDS:
        pairs = []
        for case in cases:
            left = case["models"][str(seed)]["legacy"]
            right = case["models"][str(seed)]["linear"]
            whole = right["whole"]["mean_dE00"] - left["whole"]["mean_dE00"]
            delta = {b: right["bands"][b]["dE00_contribution"] - left["bands"][b]["dE00_contribution"] for b in BANDS}
            D.need(abs(sum(delta.values()) - whole) <= 1e-10, "Band contributions do not reconstruct whole error delta")
            pairs.append({"name": case["name"], "whole_delta_mean_dE00": whole, "band_contribution_deltas": delta})
        result[str(seed)] = {"scene_count": len(pairs), "whole_delta_mean_dE00": float(np.mean([p["whole_delta_mean_dE00"] for p in pairs])),
                             "band_contribution_deltas": {b: float(np.mean([p["band_contribution_deltas"][b] for p in pairs])) for b in BANDS},
                             "per_scene": pairs}
    return result


def summaries(cases):
    """Descriptive equal-scene summaries, with empty reference bands explicit."""
    if not cases:
        return {"scene_count": 0, "guides": None, "errors": None}
    guide_result, errors = {}, {}
    for arm in D.ARMS:
        guide_result[arm] = {}
        for region in ("whole",) + BANDS:
            values = [c["guides"][arm]["whole"] if region == "whole" else c["guides"][arm]["bands"][region] for c in cases]
            valid = [v for v in values if v["count"] > 0]
            guide_result[arm][region] = {
                "scenes_with_region": len(valid),
                "mean_effective_planes": float(np.mean([v["effective_planes"] for v in valid])) if valid else None,
                "mean_first_interval_fraction": float(np.mean([v["first_interval_fraction"] for v in valid])) if valid else None,
                "mean_plane_mass": np.mean([v["plane_mass"] for v in valid], axis=0).tolist() if valid else None,
            }
    for seed in D.SEEDS:
        errors[str(seed)] = {}
        for arm in D.ARMS:
            errors[str(seed)][arm] = {}
            for region in ("whole",) + BANDS:
                values = [c["models"][str(seed)][arm]["whole"] if region == "whole" else c["models"][str(seed)][arm]["bands"][region] for c in cases]
                valid = [v for v in values if v["count"] > 0]
                item = {"scenes_with_region": len(valid),
                        "mean_reference_fraction": float(np.mean([v["fraction"] for v in values])),
                        "mean_dE00_contribution": float(np.mean([v["dE00_contribution"] for v in values]))}
                for metric in ("mean_dE00", "p95_dE00", "mean_abs_dL", "mean_abs_da", "mean_abs_db", "mean_abs_dC"):
                    item[metric] = float(np.mean([v[metric] for v in valid])) if valid else None
                item["mean_signed_Lab"] = np.mean([v["mean_signed_Lab"] for v in valid], axis=0).tolist() if valid else None
                errors[str(seed)][arm][region] = item
    return {"scene_count": len(cases), "guides": guide_result, "errors": errors}


def run(expected):
    import gamma_guide_metrics as G
    manifest, m = load(expected)
    D.need(not REPORT.exists() and not CASES.exists(), "Prior diagnostic outputs exist; refusing to overwrite")
    models, cached = {}, {}
    cpu = torch.device("cpu")
    for entry in manifest["endpoints"]:
        seed, arm = entry["seed"], entry["arm"]
        ep, ck, h = T.validate_endpoint(m, seed, arm)
        D.need(h == entry["endpoint_sha256"], "Parent endpoint hash differs")
        eval_path = D.RUNS / f"matched-gamma-seed{seed}-{arm}-eval.json"
        D.need(D.sha(eval_path) == entry["evaluation_sha256"], "Parent evaluation changed")
        cached[(seed, arm)] = json.loads(eval_path.read_text())["scenes"]
        model = D.initialise(m, seed, cpu)
        model.load_state_dict(ck["model"], strict=True)
        models[(seed, arm)] = model.eval()
    pairs = {arm: D.Pairs(m, arm, role="fit") for arm in D.ARMS}
    CASES.mkdir()
    cases = []
    with torch.no_grad():
        for n, row in enumerate(manifest["cohort"], 1):
            name = row["name"]
            data = {arm: pairs[arm].get(name) for arm in D.ARMS}
            D.need(torch.equal(data["legacy"][1], data["linear"][1]) and torch.equal(data["legacy"][2], data["linear"][2]), "Paired targets/DR differ")
            target = data["legacy"][1].squeeze(0).permute(1, 2, 0).numpy()
            target_lab = G.reference_lab(target)
            case = dict(row, guides={}, models={str(seed): {} for seed in D.SEEDS})
            for arm in D.ARMS:
                x, y, cond = data[arm]
                guide = compute_luma_guide(x, "rec709").squeeze(0).numpy()
                case["guides"][arm] = G.guide_statistics(guide, target_lab[..., 0], depth=9)
                for seed in D.SEEDS:
                    pred = D.forward(models[(seed, arm)], m, x, cond, cpu)
                    stats = G.error_statistics(pred.squeeze(0).permute(1, 2, 0).numpy(), target_lab)
                    frozen = cached[(seed, arm)][name]["metrics"]
                    differences = {"mean_dE00": abs(stats["whole"]["mean_dE00"] - frozen["mean_dE00"]),
                                   "p95_dE00": abs(stats["whole"]["p95_dE00"] - frozen["p95_dE00"])}
                    if differences["mean_dE00"] > manifest["repeat_limits"]["whole_mean_dE00_abs"] or differences["p95_dE00"] > manifest["repeat_limits"]["whole_p95_dE00_abs"]:
                        D.write_once_json(D.DIAG / "gamma-guide-repeat-failure.json", {"manifest_sha256": expected, "name": name, "seed": seed, "arm": arm, "differences": differences})
                        D.need(False, "Frozen endpoint metric did not reproduce")
                    stats["repeat_abs_difference"] = differences
                    case["models"][str(seed)][arm] = stats
            D.write_once_json(CASES / (name + ".json"), case)
            cases.append(case)
            print(f"{n}/{len(manifest['cohort'])} {name} six endpoint repeats pass", flush=True)
    report = {"manifest_sha256": expected, "completed": True, "cohort_count": len(cases), "strata": manifest["strata"],
              "paired_error_contributions": aggregate(cases),
              "overall_descriptive": summaries(cases),
              "by_stratum": {s["stratum"]: summaries([c for c in cases if c["stratum"] == s["stratum"]]) for s in manifest["strata"]},
              "case_files": [{"name": c["name"], "file": str((CASES / (c["name"] + ".json")).relative_to(D.ROOT)), "sha256": D.sha(CASES / (c["name"] + ".json"))} for c in cases],
              "interpretation": manifest["interpretation"]}
    D.need(D.sha(D.R11_PATH) == D.R11_SHA, "Selected model changed")
    D.write_once_json(REPORT, report)
    print("Diagnostic complete; no optimisation or model selection", len(cases), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--manifest-sha")
    a = p.parse_args()
    if a.command == "prepare":
        prepare()
    else:
        D.need(bool(a.manifest_sha), "Frozen diagnostic SHA required")
        run(a.manifest_sha)
