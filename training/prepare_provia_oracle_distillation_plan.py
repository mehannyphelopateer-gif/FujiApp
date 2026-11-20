"""Metadata-only scope for a paired F24 oracle-output supervision pilot.

No RAF/TIFF/cache pixels are read and no fitting is performed. Existing
train-only diagnostic grid files are hashed for evidence/reuse bindings.
"""
import json
from pathlib import Path

import run_provia_grid_oracle as O

P = O.P
OUT = P.DIAG / "provia-oracle-distillation-plan.json"


def main():
    if OUT.exists():
        P.fail("distillation plan exists; refusing overwrite")
    _, data = O.load()
    cohort_path = P.DIAG / "cuban-cohort-proposal.json"
    folds_path = P.DIAG / "cuban-step1-folds.json"
    cohort = json.loads(cohort_path.read_text())
    folds = json.loads(folds_path.read_text())["folds"]
    by = {r["name"]: r for r in data["scenes"]}
    selected = [r for r in cohort["scenes"] if r["role"] == "fit"]
    names = sorted(r["name"] for r in selected)
    if len(names) != 24 or len(set(names)) != 24 or set(names) - set(by):
        P.fail("F24 is not exactly 24 distinct names in the approved 583-train pool")
    for row in selected:
        old = by[row["name"]]
        if row["tier"] != "train" or any(row[k] != old[k] for k in ("shoot", "session")):
            P.fail("F24 train identity/session mismatch")
    if any(by[n]["captureDynamicRange"] != 400 for n in names):
        P.fail("unexpected F24 capture DR composition; do not silently change scope")
    fold_plan, held = [], []
    for fold in folds:
        dev = sorted(fold["validation_scenes"])
        fit = sorted(set(names) - set(dev))
        if not dev or set(dev) - set(names) or set(by[n]["session"] for n in fit) & set(by[n]["session"] for n in dev):
            P.fail("invalid session fold")
        fold_plan.append({"fold": fold["fold"], "fit": fit, "adaptation_OOF": dev,
                          "held_sessions": sorted(set(by[n]["session"] for n in dev))})
        held += dev
    if len(fold_plan) != 4 or sorted(held) != names:
        P.fail("folds must partition F24 exactly once")
    report = json.loads(O.REPORT.read_text())
    if report["manifest_sha256"] != O.MANIFEST_SHA or len(report["per_scene"]) != 8:
        P.fail("oracle evidence mismatch")
    reuse = {}
    for n in names:
        if n in report["per_scene"]:
            path = O.GRIDS / (Path(n).stem + "-step200.npy")
            sha = report["per_scene"][n]["endpoint_grid_sha256"]
            if P.sha256(path) != sha:
                P.fail("existing oracle grid changed")
            reuse[n] = sha
    if len(reuse) != 4:
        P.fail("expected exactly four reusable F24 teacher grids")
    out = {
        "scope": "Paired session-grouped ADAPTATION-OOF pilot on F24, all DR400; not independent base-model validation",
        "hypothesis": "Rendered reachable oracle targets may teach transferable corrections better than direct Fuji supervision under the matched predictor-only budget.",
        "bindings": {
            "source_R11": P.SOURCE_SHA, "pilot_manifest": P.MANIFEST_SHA,
            "oracle_run_manifest": O.MANIFEST_SHA, "oracle_report": P.sha256(O.REPORT),
            "cohort_metadata": P.sha256(cohort_path), "folds": P.sha256(folds_path),
            "code": {f: P.sha256(P.TRAIN / f) for f in
                     (Path(__file__).name, "model.py", "color_loss.py", "color_metrics.py", "check_provia_grid_oracle.py")},
        },
        "scenes": [by[n] for n in names], "folds": fold_plan,
        "teacher_generation": {
            "reuse_endpoint_grids_sha256": reuse,
            "new_grids": sorted(set(names) - set(reuse)),
            "method": "Exact original per-image oracle protocol: original R11 grid, frozen model, 200 fresh-Adam steps at 1e-3, original display-combined loss, no extension or selection",
            "supervision": "Unclipped float32 linear RGB rendered by oracle_forward with frozen original R11 refinement; hash-bound local cache. No raw coefficient regression.",
            "isolation": "Each teacher uses only its own allowed input/target. No joint statistics or teacher-outcome-driven exclusion. Fit-fold loaders may read only fit-fold teachers; held-fold teachers are not used for updates or selection.",
        },
        "paired_students": {
            "runs": 8, "arms": {"direct": "Original Fuji Provia target", "distill": "Rendered per-image oracle target"},
            "init": "Original R11 seed 42 independently for every arm/fold, not any replay endpoint",
            "trainable": "Entire original grid predictor (encoder plus head), 105084 parameters; refinement weights frozen, recomputed and differentiated through",
            "forward": "Unchanged full-resolution hybrid forward, same RGB/capture-DR input, grid dimensions and guide",
            "optimizer": "fresh Adam, betas=(0.9,0.999), eps=1e-8, weight_decay=0",
            "lr": 1e-5, "epochs": 16, "batch_size": 1, "seed": 42,
            "order": "Same recorded per-fold torch.randperm order for both arms; sorted fit names, fresh Generator(42)",
            "objective": "color_loss.per_scene_terms combined, unchanged definition in both arms; supervision target is the sole between-arm variable",
            "schedule": None, "augmentation": None, "amp": None, "early_stopping": None,
            "endpoint": "epoch 16, no best-epoch selection; exact resume and full source/target hashes required",
        },
        "evaluation": {
            "timing": "Freeze all paired endpoint checkpoints before adaptation-OOF scoring; no intermediate held-fold scoring",
            "target": "Actual Fuji Provia reference, never oracle output",
            "report": "All 24 adaptation-OOF per-scene mean/p95 CIEDE2000, tone/chroma/hue coverage, RGB tails, per-session and per-fold comparisons; R11 baseline and regressions against R11; training-target fit logged separately",
            "pilot_screen_only": {
                "mean_dE00_distill_minus_direct_max": -0.25,
                "median_dE00_distill_minus_direct_strictly_less_than": 0,
                "mean_p95_distill_minus_direct_max": 0,
                "regression_count_vs_R11_distill_no_greater_than_direct": True,
                "max_session_mean_dE00_distill_minus_direct": 0.25,
            },
            "meaning": "A screen pass supports this training-supervision direction only. A miss rejects this scoped budget/cohort, not every distillation method. Original R11 already trained on F24: no independent-generalisation, DR100/200 or recipe-parity claim.",
        },
        "boundary": "Freeze a separate runner/run manifest and smoke exact warm starts, teacher-cache fingerprints, per-fold refusals, paired orders, masked updates, finite backward and exact resume before fitting. No monitor/K/finalHeldOut/T8/dr100Eval media, CN scoring, full-pool run, extra sweeps, app changes or model selection. No new user photos/exports. Codex has scoped only; no teachers or student fits launched.",
    }
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print("distillation plan sha256", P.sha256(OUT), flush=True)


if __name__ == "__main__":
    main()
