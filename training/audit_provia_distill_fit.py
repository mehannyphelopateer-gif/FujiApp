"""Read-only F24 fit-attainment audit of the already-frozen distillation pilot.

Inference only. No optimiser, backward pass, teacher construction, new
checkpoint, new scene, protected-tier or CN reference access.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import audit_provia_distill_artifacts as A
import color_loss as CL
import run_provia_distill as D

P = D.P
MANIFEST = P.DIAG / "provia-distill-fit-audit-manifest.json"
REPORT = P.DIAG / "provia-distill-fit-audit.json"
THIS = Path(__file__).resolve()
SENTINELS = ((0, "direct", "DSCF0391.RAF"), (3, "distill", "DSCF0882.RAF"))


def closure(before, after):
    if not math.isfinite(before) or not math.isfinite(after) or before < 0 or after < 0:
        raise ValueError("Non-finite gap distance")
    return None if before <= 1e-12 else 1 - after / before


def classify(folds):
    """Predeclared diagnostic labels, not product acceptance thresholds."""
    if len(folds) != 4:
        raise ValueError("Expected exactly four folds")
    values = [r[a]["fit"][r[a]["own_target"]][k] for r in folds.values() for a, k in
              (("direct", "objective_closure"), ("distill", "objective_closure"),
               ("distill", "display_dE_closure"))]
    transfer = [r["distill"]["adaptation_OOF"]["teacher"][k] for r in folds.values()
                for k in ("objective_closure", "display_dE_closure")]
    if any(v is None for v in values + transfer):
        return "INCONCLUSIVE_zero_baseline"
    if all(v < 0.25 for v in values):
        return "A_under_attained_fit_targets_under_this_budget"
    if all(r["distill"]["fit"]["teacher"][k] >= 0.5 for r in folds.values()
           for k in ("objective_closure", "display_dE_closure")) and all(v is not None and v <= 0.10 for v in transfer):
        return "B_fit_teacher_learning_did_not_transfer_in_this_pilot"
    return "MIXED_OR_INCONCLUSIVE"


def prepare():
    if MANIFEST.exists():
        raise ValueError("Fit-audit manifest exists; refusing overwrite")
    evidence = A.audit(with_results=True)
    _, plan, _ = D.load()
    out = {
        "scope": "F24 fit-attainment diagnostic, IN-SAMPLE inference only; all DR400; no independent generalisation claim",
        "bindings": {"source_R11": P.SOURCE_SHA, "distill_manifest": D.MANIFEST_SHA,
                     "distill_plan": D.PLAN_SHA, "freeze": evidence["freeze_sha256"],
                     "teacher_log": evidence["teacher_log_sha256"], "OOF_report": evidence["result_sha256"],
                     "code": {f: P.sha256(P.TRAIN / f) for f in
                              (THIS.name, "audit_provia_distill_artifacts.py", "run_provia_distill.py",
                               "model.py", "color_loss.py", "color_metrics.py", "provia_color_pilot.py")}},
        "scenes": plan["scenes"], "folds": plan["folds"],
        "sentinels": [{"fold": f, "arm": a, "name": n} for f, a, n in SENTINELS],
        "reuse_tolerance_mean_dE00": 1e-8,
        "comparisons": "True original R11 start on all 24; each frozen endpoint on all F24, explicitly separated into its fit subset (72 rows per arm) and adaptation-OOF subset (24 rows per arm). All predictions scored against both Fuji Provia and the frozen rendered teacher. These are never finalHeldOut scenes.",
        "metrics": "Unchanged display CIEDE2000 mean/p95 and original display-combined objective with both components, raw tails, tone/chroma/hue; teacher-vs-Fuji attained error (not a lower bound); module weight-change norms",
        "derived": "Per fold/subset/target: ratio-of-means closure 1 - endpoint/start for objective and displayed mean dE, plus median of per-scene closures and null counts. Never online epoch-1 loss as start. Zero start <=1e-12 -> null; negative closure retained. Classify only by per-fold ratio-of-means. All cross-target comparisons explicit.",
        "labels": {
            "retrospective": "The 25%/50% fit thresholds were fixed after seeing small online epoch-1-to-16 changes. These are retrospective diagnostic labels, not independent hypothesis confirmation or product acceptance. The OOF teacher-closure values have not been computed before freezing this audit.",
            "A": "Both arms in every fold close <25% of own display-combined gap AND distill in every fold closes <25% of displayed teacher gap: under-attainment at this budget, not a capacity/schedule diagnosis",
            "B": "Distill in every fold closes >=50% of both teacher fit gaps, but adaptation-OOF teacher closure is <=10% (including negative) on both measures in every fold: fit teacher learning did not transfer in this pilot. The failed Fuji pilot screen is context, not a classification premise.",
            "other": "Mixed/inconclusive, including any unmeasurable baseline; all folds/scenes reported",
        },
        "limits": "R11 trained on F24. Display Lab/dE clamp both target and prediction, including teacher negatives; raw linear loss does not. A scene appears in three fit folds, so 72 rows are not independent; folds are primary, pooled values descriptive only. Checkpoint movement is descriptive, not an Adam drift bound or comparable to grid-coefficient units. No fitting, extension, selection, app/contract/CN change or K/monitor/finalHeldOut/T8/dr100Eval media access.",
    }
    MANIFEST.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print("fit-audit manifest sha256", P.sha256(MANIFEST), flush=True)


def frozen_model(path, device):
    model, config = P.build(device)
    if path != P.SOURCE:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def measure(pred, target, device):
    ref = target.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
    row = D.scene_metrics(pred, ref)
    row["objective"] = {k: float(v.mean().item()) for k, v in CL.per_scene_terms(pred, target.to(device)).items()}
    if not all(math.isfinite(row["objective"][k]) for k in row["objective"]):
        raise ValueError("Non-finite objective")
    return row


def module_movement(state, source):
    out = {}
    for prefix in ("grid.encoder.", "grid.head.", "refinement."):
        keys = [k for k in source if k.startswith(prefix)]
        difference = torch.cat([(state[k] - source[k]).reshape(-1).double() for k in keys])
        before = torch.cat([source[k].reshape(-1).double() for k in keys])
        norm = float(before.norm().item())
        delta = float(difference.norm().item())
        out[prefix] = {"parameters": difference.numel(), "delta_l2": delta,
                       "relative_l2": delta / norm if norm else None,
                       "max_abs_delta": float(difference.abs().max().item())}
    return out


def run(expected_sha):
    if REPORT.exists():
        raise ValueError("Fit-audit report exists; refusing overwrite")
    if P.sha256(MANIFEST) != expected_sha:
        raise ValueError("Fit-audit manifest hash mismatch")
    A.committed(MANIFEST)
    A.committed(THIS)
    manifest = json.loads(MANIFEST.read_text())
    for f, sha in manifest["bindings"]["code"].items():
        if P.sha256(P.TRAIN / f) != sha:
            raise ValueError("Fit-audit bound code changed: " + f)
    evidence = A.audit(with_results=True)
    for key, field in (("freeze", "freeze_sha256"), ("teacher_log", "teacher_log_sha256"), ("OOF_report", "result_sha256")):
        if manifest["bindings"][key] != evidence[field]:
            raise ValueError("Fit-audit evidence changed: " + key)
    _, plan, data = D.load()
    if manifest["scenes"] != plan["scenes"] or manifest["folds"] != plan["folds"]:
        raise ValueError("Fit-audit cohort/folds changed")
    allowed = {s["name"] for s in plan["scenes"]}
    pairs = P.Pairs(data)

    def get(name):
        if name not in allowed:
            raise ValueError("Non-F24 name refused: " + name)
        x, y, condition = pairs.get(name)
        teacher = D.load_teacher(pairs.by[name])
        if y.shape != teacher.shape:
            raise ValueError("Teacher shape mismatch")
        return x, y, teacher, condition

    old = json.loads(D.REPORT.read_text())
    device = P.device()
    source = torch.load(P.SOURCE, map_location="cpu", weights_only=False)["state_dict"]
    checks = {}
    with torch.no_grad():
        for fold, arm, name in SENTINELS:
            if name not in next(f for f in plan["folds"] if f["fold"] == fold)["adaptation_OOF"]:
                raise ValueError("Sentinel not in its frozen adaptation-OOF fold")
            model, config = frozen_model(D.state_path(fold, arm, D.EPOCHS), device)
            x, y, _, condition = get(name)
            row = measure(P.fwd(model, config, x, condition, device), y, device)
            error = abs(row["mean_dE00"] - old["per_scene"][name][arm]["mean_dE00"])
            if error > manifest["reuse_tolerance_mean_dE00"]:
                raise ValueError("Frozen sentinel metric does not reproduce")
            checks[f"f{fold}-{arm}-{name}"] = error
            del model
        model, config = frozen_model(P.SOURCE, device)
        baselines = {}
        for name in sorted(allowed):
            x, y, teacher, condition = get(name)
            pred = P.fwd(model, config, x, condition, device)
            baselines[name] = {"fuji": measure(pred, y, device), "teacher": measure(pred, teacher, device)}
            error = abs(baselines[name]["fuji"]["mean_dE00"] - old["per_scene"][name]["r11"]["mean_dE00"])
            if error > manifest["reuse_tolerance_mean_dE00"]:
                raise ValueError("R11 baseline does not reproduce: " + name)
            checks["r11-" + name] = error
            print("baseline", name, flush=True)
        del model
        rows, movements = {}, {}
        for fold in plan["folds"]:
            fid = str(fold["fold"])
            rows[fid], movements[fid] = {}, {}
            for arm in D.ARMS:
                state = torch.load(D.state_path(fold["fold"], arm, D.EPOCHS), map_location="cpu", weights_only=False)
                movements[fid][arm] = module_movement(state["model"], source)
                model, config = frozen_model(D.state_path(fold["fold"], arm, D.EPOCHS), device)
                rows[fid][arm] = {}
                for name in sorted(allowed):
                    x, y, teacher, condition = get(name)
                    pred = P.fwd(model, config, x, condition, device)
                    row = {"fuji": measure(pred, y, device), "teacher": measure(pred, teacher, device),
                           "subset": "fit" if name in fold["fit"] else "adaptation_OOF"}
                    rows[fid][arm][name] = row
                    if row["subset"] == "adaptation_OOF":
                        if abs(row["fuji"]["mean_dE00"] - old["per_scene"][name][arm]["mean_dE00"]) > manifest["reuse_tolerance_mean_dE00"]:
                            raise ValueError("Existing adaptation-OOF score does not reproduce: " + name)
                    print(row["subset"], fid, arm, name, flush=True)
                del model, state
    summaries = {}
    for fold in plan["folds"]:
        fid = str(fold["fold"])
        summaries[fid] = {}
        for arm in D.ARMS:
            target = "fuji" if arm == "direct" else "teacher"
            summary = {"own_target": target,
                       "last_epoch_online_own_loss_not_endpoint": old["training_history_own_target"][f"f{fid}-{arm}"][-1]["train_combined_vs_own_target"]}
            for subset in ("fit", "adaptation_OOF"):
                summary[subset] = {}
                for against in ("fuji", "teacher"):
                    selected = fold[subset]
                    base_objective = float(np.mean([baselines[n][against]["objective"]["combined"] for n in selected]))
                    end_objective = float(np.mean([rows[fid][arm][n][against]["objective"]["combined"] for n in selected]))
                    base_de = float(np.mean([baselines[n][against]["mean_dE00"] for n in selected]))
                    end_de = float(np.mean([rows[fid][arm][n][against]["mean_dE00"] for n in selected]))
                    objective_closures = [closure(baselines[n][against]["objective"]["combined"], rows[fid][arm][n][against]["objective"]["combined"]) for n in selected]
                    display_closures = [closure(baselines[n][against]["mean_dE00"], rows[fid][arm][n][against]["mean_dE00"]) for n in selected]
                    median = lambda vals: float(np.median([v for v in vals if v is not None])) if any(v is not None for v in vals) else None
                    summary[subset][against] = {
                        "n": len(selected), "start_objective": base_objective, "end_objective": end_objective,
                        "objective_closure": closure(base_objective, end_objective),
                        "objective_closure_scene_median": median(objective_closures),
                        "objective_closure_scene_nulls": objective_closures.count(None),
                        "start_display_dE": base_de, "end_display_dE": end_de,
                        "display_dE_closure": closure(base_de, end_de),
                        "display_dE_closure_scene_median": median(display_closures),
                        "display_dE_closure_scene_nulls": display_closures.count(None),
                    }
            summaries[fid][arm] = summary
    after = A.audit(with_results=True)
    if any(after[k] != evidence[k] for k in ("freeze_sha256", "teacher_log_sha256", "result_sha256")):
        raise ValueError("Frozen evidence changed during fit audit")
    result = {"scope": manifest["scope"], "manifest_sha256": expected_sha, "device": str(device),
              "sentinel_and_R11_reproduction_abs_errors": checks,
              "classification": classify(summaries),
              "fold_summaries": summaries, "module_movement": movements,
              "baselines": baselines, "endpoint_rows": rows, "original_R11_and_frozen_artifacts_unchanged": True,
              "note": "All fit values are in-sample; diagnostic labels do not distinguish update budget, optimisation, information or encoder capacity. No fitting or model selection."}
    tmp = REPORT.with_suffix(".json.partial")
    if tmp.exists():
        raise ValueError("Partial fit-audit report already exists")
    tmp.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    tmp.rename(REPORT)
    print(json.dumps({"classification": result["classification"], "fold_summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--manifest-sha")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    elif not args.manifest_sha:
        raise SystemExit("run requires --manifest-sha")
    else:
        run(args.manifest_sha)
