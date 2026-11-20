"""Metadata-only plan for one matched training-conversion ablation."""
import hashlib
import json
import math
from pathlib import Path

DIAG = Path(__file__).resolve().parent / "diagnostics"
ROOT = DIAG.parent.parent
LOW = DIAG / "provia-low-rate-run-manifest.json"
LOW_SHA = "defb2f0614e83ada8c645ca8619c0037ebf2913c4667d9fa12133af43cfd907a"
OUT = DIAG / "provia-extended-loss-plan.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or sha(LOW) != LOW_SHA:
        raise ValueError("Existing plan or changed baseline manifest")
    low = json.loads(LOW.read_text())
    probe_path = DIAG / "provia-extended-loss-probes.json"
    probe = json.loads(probe_path.read_text())
    pm = DIAG / "provia-extended-loss-probe-manifest.json"
    manifest = json.loads(pm.read_text())
    if (probe.get("passed") is not True or probe.get("weights_saved") is not False
            or probe.get("source_checkpoint_unchanged") is not True or probe["manifest_sha256"] != sha(pm)
            or set(probe["per_case"]) != set(manifest["names"]) or len(probe["per_case"]) != 8):
        raise ValueError("Eight-case preflight/provenance did not pass")
    for row in probe["per_case"].values():
        if (not all(isinstance(row[k], (int,float)) and not isinstance(row[k],bool) and math.isfinite(row[k])
                    for k in ("before","after","change")) or row["after"] >= row["before"]
                or abs(row["change"]-(row["after"]-row["before"])) > 1e-12):
            raise ValueError("Invalid or non-descending local preflight")
    if any(sha(ROOT/"training"/f)!=h for f,h in manifest["code_sha256"].items()):
        raise ValueError("Bound loss/checker/scorer code changed")
    evaluation = ROOT / "training" / "runs" / "provia-low-rate-candidate-update1868.json"
    ev = json.loads(evaluation.read_text())
    state = ROOT / "training" / "checkpoints" / "provia-low-rate-candidate-block4.pt"
    if ev["manifest_sha256"] != LOW_SHA or ev["state_sha256"] != sha(state):
        raise ValueError("Frozen display-clamped baseline changed")
    result = {
        "status": "PLAN before actual run implementation/fitting; separate run manifest/code must be committed",
        "scope": "One loss-conversion ablation; all 583 scores in-sample fitting/retention, not validation",
        "bindings": {"baseline_low_rate_manifest": LOW_SHA, "baseline_candidate_state": sha(state),
                     "baseline_candidate_evaluation": sha(evaluation), "baseline_report": sha(DIAG/"provia-low-rate-report.json"),
                     "probe_manifest": sha(pm), "probe_report": sha(probe_path),
                     "synthetic_checks": sha(DIAG/"provia-extended-loss-synthetic.json"),
                     "code": manifest["code_sha256"], "source_R11": low["bindings"]["source_checkpoint"]},
        "change": "Only omit RGB hard-clamp before the Lab conversion in the training colour term; use color_loss_extended.per_scene_terms. Piecewise Lab low-XYZ linear branch supplies finite gradients beyond display gamut. Evaluation retains original display clamp.",
        "loss": "0.5*(unclamped linear L1 + extended per-channel Lab L1/100); targets, matrix, white point and weights unchanged",
        "runs": 1,
        "baseline": "Reuse frozen low-rate display-clamped candidate; no duplicate control training",
        "initialization": "Original R11 seed 42, not the old candidate or any probe; fresh Adam same parameters/state policy",
        "unchanged": {"pairs": 583, "updates": 1868, "blocks": 4, "updates_per_block": 467,
                      "seed": 42, "batch_size": 1, "order_sha256": low["bindings"]["order_sha256"],
                      "block_learning_rates": low["block_learning_rates"], "horizon_blocks": 12, "eta_min": 1e-7,
                      "model": "original architecture/preprocessing/capture-DR input, all parameters trainable",
                      "optimizer": "Adam beta 0.9/0.999, eps 1e-8, wd 0, fresh",
                      "AMP": None,"augmentation":None,"accumulation":None,"early_stopping":None,"HPO":None},
        "preflight": "New immutable run manifest/code before fitting; bind original data/helpers and new loss/probes. Same no-op/refusal/full-res finite/resume checks. No edits/monkeypatches to old bound modules. New provia-extended-* output paths.",
        "evaluation": "Update 0 and 1868 only, stream all 583, former roles/session/DR groups and every scene. Compare R11, old low-rate candidate, new extended candidate with unchanged display CIEDE2000/tone/chroma/hue/p95; record RGB tail magnitudes and target-positive negative-channel occupancy.",
        "objective_reporting": "Evaluate BOTH original display-combined and extended-combined definitions on all three models. Different objective definitions cannot be compared as if they were the same loss. Use explicit field names.",
        "review": "Include fixed eight-case panels/tail review; regional associations alone do not prove causality. Keep these cases at ordinary uniform training weight, no exceptions or oversampling.",
        "stop": "Return the fixed endpoint whether positive or negative; no model selection, further rates/seeds/epochs, schedule extension, app/LUT/metric changes, CN scoring or K/monitor/finalHeldOut/T8/dr100Eval access. No new exports/photos. Improvement still needs separate validation.",
    }
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("plan sha256",sha(OUT))


if __name__ == "__main__":
    main()
