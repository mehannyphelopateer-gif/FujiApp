"""Independent read-only pre-score/post-score audit of the frozen pilot.

No training, teacher construction, model inference or protected-tier reads.
The single output is a new audit JSON; existing evidence is never modified.
Exit 3 means a required artifact is missing; exit 1 means integrity failure.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import run_provia_distill as D


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_json(path):
    def bad(value):
        raise ValueError(f"Non-finite JSON constant {value}: {path}")
    return json.loads(path.read_text(), parse_constant=bad)


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid number: {label}")
    return value


def equal_number(a, b, label):
    require(abs(finite(a, label) - finite(b, label)) <= 1e-10, f"Arithmetic mismatch: {label}")


def committed(path):
    rel = str(path.relative_to(D.P.TRAIN.parent))
    r = subprocess.run(["git", "show", "HEAD:" + rel], cwd=D.P.TRAIN.parent, capture_output=True)
    require(r.returncode == 0 and hashlib.sha256(r.stdout).hexdigest() == D.P.sha256(path),
            f"Artifact must be committed and unchanged: {rel}")


def check_endpoint(st, source, fold, arm, expected_order, label):
    require(st["manifest_sha256"] == D.MANIFEST_SHA and st["fold"] == fold
            and st["arm"] == arm and st["epoch"] == D.EPOCHS, "Endpoint identity mismatch: " + label)
    require(st["orders"] == expected_order, "Endpoint order differs from frozen algorithm: " + label)
    require(len(st["history"]) == D.EPOCHS and [v["epoch"] for v in st["history"]] == list(range(1, D.EPOCHS + 1)),
            "Incomplete/invalid history: " + label)
    for row in st["history"]:
        finite(row["train_combined_vs_own_target"], label)
    state = st["model"]
    require(set(state) == set(source), "Model keys mismatch: " + label)
    for k, value in state.items():
        require(value.shape == source[k].shape and value.dtype == source[k].dtype and torch.isfinite(value).all(),
                "Invalid model tensor: " + label + ":" + k)
        if k.startswith("refinement."):
            require(torch.equal(value, source[k]), "Frozen refinement changed: " + label + ":" + k)


def check_results(report, plan, freeze_sha):
    names = sorted(s["name"] for s in plan["scenes"])
    require(report["manifest_sha256"] == D.MANIFEST_SHA and report["freeze_sha256"] == freeze_sha,
            "Results manifest/freeze binding mismatch")
    per = report["per_scene"]
    require(sorted(per) == names, "Results scene coverage differs from F24")
    by = {s["name"]: s for s in plan["scenes"]}
    fold_by = {n: f["fold"] for f in plan["folds"] for n in f["adaptation_OOF"]}
    for n in names:
        require(per[n]["fold"] == fold_by[n] and per[n]["session"] == by[n]["session"],
                "Result fold/session identity mismatch: " + n)
        for arm in ("r11", *D.ARMS):
            finite(per[n][arm]["mean_dE00"], n)
            finite(per[n][arm]["p95_dE00"], n)
        equal_number(per[n]["distill_minus_direct"],
                     per[n]["distill"]["mean_dE00"] - per[n]["direct"]["mean_dE00"], n)
    aggregates = {}
    regressions = {}
    for arm in ("r11", *D.ARMS):
        vals = [per[n][arm]["mean_dE00"] for n in names]
        a = {"mean_dE00": float(np.mean(vals)), "median_dE00": float(np.median(vals)),
             "mean_p95_dE00": float(np.mean([per[n][arm]["p95_dE00"] for n in names]))}
        for k, v in a.items():
            equal_number(report["aggregate"][arm][k], v, arm + ":" + k)
        aggregates[arm] = a
        if arm != "r11":
            regressions[arm] = sorted(n for n in names if per[n][arm]["mean_dE00"] > per[n]["r11"]["mean_dE00"])
    require(report["regressions_vs_r11"] == regressions, "Regression lists mismatch")
    sessions = sorted(set(s["session"] for s in plan["scenes"]))
    require(sorted(report["sessions"]) == sorted(map(str, sessions)), "Session coverage mismatch")
    diffs = []
    for session in sessions:
        subset = [n for n in names if by[n]["session"] == session]
        r = report["sessions"][str(session)]
        require(r["n"] == len(subset), "Session size mismatch")
        for arm in ("r11", *D.ARMS):
            equal_number(r[arm], float(np.mean([per[n][arm]["mean_dE00"] for n in subset])), "session mean")
        delta = float(np.mean([per[n]["distill_minus_direct"] for n in subset]))
        equal_number(r["distill_minus_direct"], delta, "session difference")
        diffs.append(delta)
    require(sorted(report["folds"]) == sorted(str(f["fold"]) for f in plan["folds"]), "Fold coverage mismatch")
    for f in plan["folds"]:
        for arm in ("r11", *D.ARMS):
            equal_number(report["folds"][str(f["fold"])][arm],
                         float(np.mean([per[n][arm]["mean_dE00"] for n in f["adaptation_OOF"]])), "fold mean")
    scr = plan["evaluation"]["pilot_screen_only"]
    mean_diff = aggregates["distill"]["mean_dE00"] - aggregates["direct"]["mean_dE00"]
    median_diff = aggregates["distill"]["median_dE00"] - aggregates["direct"]["median_dE00"]
    p95_diff = aggregates["distill"]["mean_p95_dE00"] - aggregates["direct"]["mean_p95_dE00"]
    for k, v in (("mean_diff", mean_diff), ("median_diff", median_diff), ("p95_diff", p95_diff),
                 ("max_session_diff", max(diffs))):
        equal_number(report["screen"][k], v, k)
    require(report["screen"]["regressions_vs_r11"] == {a: len(v) for a, v in regressions.items()},
            "Screen regression counts mismatch")
    checks = {"mean": mean_diff <= scr["mean_dE00_distill_minus_direct_max"],
              "median": median_diff < scr["median_dE00_distill_minus_direct_strictly_less_than"],
              "p95": p95_diff <= scr["mean_p95_distill_minus_direct_max"],
              "regressions": len(regressions["distill"]) <= len(regressions["direct"]),
              "sessions": max(diffs) <= scr["max_session_mean_dE00_distill_minus_direct"]}
    require(set(report["screen"]["checks"]) == set(checks), "Screen checks differ")
    require(all(type(report["screen"]["checks"][k]) is bool and report["screen"]["checks"][k] == v
                for k, v in checks.items()), "Screen verdict arithmetic mismatch")
    require(type(report["screen"]["pilot_screen_pass"]) is bool
            and report["screen"]["pilot_screen_pass"] == all(checks.values()), "Pilot verdict mismatch")
    return {"arithmetic": "PASS", "pilot_screen_pass": all(checks.values()), "checks": checks}


def audit(with_results=False):
    m, plan, _ = D.load()
    if not D.FREEZE.exists():
        raise FileNotFoundError("Eight-endpoint freeze is not ready")
    for path in (D.FREEZE, D.PLAN, D.MANIFEST, Path(D.__file__)):
        committed(path)
    fr = read_json(D.FREEZE)
    expected = {D.state_path(f["fold"], a, D.EPOCHS).name for f in plan["folds"] for a in D.ARMS}
    require(len(expected) == 8 and set(fr["endpoints"]) == expected, "Freeze must cover exactly eight expected endpoints")
    require(fr["manifest_sha256"] == D.MANIFEST_SHA, "Freeze manifest mismatch")
    teacher_sha = D.P.sha256(D.TEACHER_LOG)
    require(fr["teachers_sha256"] == teacher_sha, "Teacher log changed since freeze")
    log = read_json(D.TEACHER_LOG)
    names = {s["name"] for s in plan["scenes"]}
    require(log["manifest_sha256"] == D.MANIFEST_SHA and set(log["teachers"]) == names,
            "Teacher manifest/coverage mismatch")
    teacher_checks = {}
    for scene in plan["scenes"]:
        n = scene["name"]
        meta = log["teachers"][n]
        path = D.teacher_path(scene["shoot"])
        require(read_json(path.with_suffix(".json")) == meta, "Teacher sidecar/log mismatch: " + n)
        require(meta["name"] == n and meta["fjlrg_sha256"] == scene["fjlrg_sha256"]
                and meta["target_npy_sha256"] == scene["npy_sha256"], "Teacher source identity mismatch: " + n)
        require(D.P.sha256(path) == meta["render_npy_sha256"], "Teacher bytes changed: " + n)
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        w, h = scene["fjlrg_wh"]
        require(arr.dtype == np.float32 and arr.shape == (h, w, 3) and np.isfinite(arr).all(),
                "Teacher dtype/shape/finiteness mismatch: " + n)
        del arr
        gp = D.TEACHERS / (scene["shoot"] + "-grid.npy")
        require(D.P.sha256(gp) == meta["grid_sha256"], "Teacher grid changed: " + n)
        grid = np.load(gp, allow_pickle=False)
        require(grid.dtype == np.float32 and grid.shape == (1, 12, 9, 8, 8) and np.isfinite(grid).all(),
                "Invalid teacher grid: " + n)
        if n in m["bindings"]["reused_grids"]:
            require(meta["grid_sha256"] == m["bindings"]["reused_grids"][n], "Reused grid identity mismatch")
        require(D.P.sha256(D.P.CAL / scene["shoot"] / "browser-phase3-linear.fjlrg") == scene["fjlrg_sha256"],
                "Source input changed: " + n)
        require(D.P.sha256(D.P.CACHE / (scene["shoot"] + ".npy")) == scene["npy_sha256"], "Source target changed: " + n)
        teacher_checks[n] = meta["render_npy_sha256"]
    source = torch.load(D.P.SOURCE, map_location="cpu", weights_only=False)["state_dict"]
    endpoints = {}
    for f in plan["folds"]:
        expected_order = D.fold_order(f["fit"], D.EPOCHS)
        for arm in D.ARMS:
            p = D.state_path(f["fold"], arm, D.EPOCHS)
            if not p.exists():
                raise FileNotFoundError(str(p))
            before = D.P.sha256(p)
            require(before == fr["endpoints"][p.name], "Endpoint changed since freeze: " + p.name)
            st = torch.load(p, map_location="cpu", weights_only=False)
            check_endpoint(st, source, f["fold"], arm, expected_order, p.name)
            require(D.P.sha256(p) == before, "Endpoint changed while being audited")
            endpoints[p.name] = before
            del st
    require(D.P.sha256(D.TEACHER_LOG) == teacher_sha, "Teacher log changed during audit")
    D.P.check_source()
    out = {"integrity": "PASS", "scope": "Read-only artifact audit; no inference/training/protected media reads",
           "manifest_sha256": D.MANIFEST_SHA, "freeze_sha256": D.P.sha256(D.FREEZE),
           "teacher_log_sha256": teacher_sha, "auditor_sha256": D.P.sha256(Path(__file__)),
           "endpoint_checks": endpoints, "teacher_checks": teacher_checks,
           "original_R11_unchanged": True, "frozen_refinement_exact": True, "orders_match_frozen_algorithm": True}
    if with_results:
        out["result_checks"] = check_results(read_json(D.REPORT), plan, out["freeze_sha256"])
        out["result_sha256"] = D.P.sha256(D.REPORT)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--with-results", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Audit output exists; refusing overwrite")
    result = audit(args.with_results)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("endpoint_checks", "teacher_checks")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("NOT_READY:", e, flush=True)
        raise SystemExit(3)
    except (ValueError, KeyError, TypeError, OSError) as e:
        print("INTEGRITY_FAILURE:", e, flush=True)
        raise SystemExit(1)
