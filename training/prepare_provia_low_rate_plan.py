"""Metadata-only lower-step replay plan, informed by frozen local diagnostics.

Does not open images/cache/model tensors or launch fitting. Every actual
training setting except the LR scale stays matched to the prior replay.
"""
import hashlib
import json
from pathlib import Path

DIAG = Path(__file__).resolve().parent / "diagnostics"
OUT = DIAG / "provia-low-rate-plan.json"
REPLAY = DIAG / "provia-replay-run-manifest.json"
REPLAY_SHA = "bc06d9adefa610d0d98d658609a84df72df0f4311dfd5164f13582c1ef082910"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Plan exists; refusing overwrite")
    if sha(REPLAY) != REPLAY_SHA:
        raise ValueError("Frozen replay manifest changed")
    replay = json.loads(REPLAY.read_text())
    probe_path = DIAG / "provia-update-probe-report.json"
    full_path = DIAG / "provia-fullres-gradient-check.json"
    probe = json.loads(probe_path.read_text())
    full = json.loads(full_path.read_text())
    probe_sha = sha(DIAG / "provia-update-probe-manifest.json")
    if (probe.get("source_checkpoint_unchanged") is not True or probe.get("selected_weights") is not False
            or probe.get("manifest_sha256") != probe_sha or full.get("probe_manifest_sha256") != probe_sha
            or full.get("source_checkpoint_sha256") != replay["bindings"]["source_checkpoint"]):
        raise ValueError("Diagnostic provenance/source-restoration mismatch")
    if not full["passed"] or not all(r["passed"] for r in probe["synthetic"].values()):
        raise ValueError("Numerical gradient checks did not pass")
    safe = []
    counts = {}
    for lr in (1e-4, 1e-5, 1e-6):
        by_arm = {}
        for arm, data in probe["local_probes"].items():
            trials = [t for t in data["trials"] if t["lr"] == lr and t["own_loss_change"] is not None]
            if len(trials) != 6:
                raise ValueError("Incomplete fixed local probe coverage")
            by_arm[arm] = sum(t["own_loss_change"] < 0 for t in trials)
        counts[str(lr)] = by_arm
        if all(v == 6 for v in by_arm.values()):
            safe.append(lr)
    # Local descent is the reason for the new setting, not selection by
    # image-quality/validation scores or reuse of a diagnostic weight state.
    if not safe or max(safe) != 1e-5:
        raise ValueError("Recorded evidence does not support the scoped setting")
    rates = [x / 10 for x in replay["training"]["learning_rates_by_467_update_block"]]
    result = {
        "status": "PLAN before lower-step replay implementation/fitting; new run manifest/code still required",
        "scope": "In-sample fitting/retention diagnostic only; no validation or candidate selection",
        "bindings": {"original_replay_run_manifest": REPLAY_SHA,
                     "original_replay_report": sha(DIAG / "provia-replay-report.json"),
                     "source_checkpoint": replay["bindings"]["source_checkpoint"],
                     "local_probe_report": sha(probe_path), "full_resolution_gradient_check": sha(full_path)},
        "evidence_own_case_descent_counts": counts,
        "evidence_limit": "Largest tested rate lowering own-case objective on all six probes for both losses; not a proven full-pool optimum. Conflicting gradients remain, and six cases are not population validation.",
        "change": "Divide the previous replay's entire LR curve by 10: initial 1e-5, cosine eta_min 1e-7, same 12-block horizon, stop after four blocks",
        "block_learning_rates": rates,
        "unchanged": {"pairs": 583, "seed": 42, "source_replay_plan": replay["bindings"]["replay_plan"],
                      "arms": ["control", "candidate"],
                      "source": "original R11, not any probe or replay endpoint",
                      "model": "unchanged architecture, all original parameters trainable",
                      "losses": "same control linear L1 and candidate half linear L1 + half Lab L1/100",
                      "optimizer": "fresh Adam, beta 0.9/0.999, eps 1e-8, weight decay 0",
                      "batch": 1, "gradient_accumulation": None,
                      "order_sha256": replay["order_sha256"], "updates": 1868,
                      "blocks": 4, "updates_per_block": 467, "schedule_horizon_blocks": 12,
                      "AMP": None, "augmentation": None, "early_stopping": None, "runs": 2},
        "implementation": "Separate runner/run manifest committed before fitting; reuse existing hash-bound data/helpers unchanged. No global-variable monkeypatches or edits to old bound code. New provia-low-rate-* paths only.",
        "preflight": "Same no-op/refusal/finite/full-res/resume checks; cross-check scaled rates to CosineAnnealingLR; source and cached data/order unchanged. No extra decode/export if hashes remain valid.",
        "evaluate": "At update 0 and 1868 only: all 583 training scenes, former roles, every session/DR group; own training objective plus unchanged display DeltaE/p95/tone/chroma/hue/clipping, all regressions. Compare both arms to R11 and the same-arm old high-rate replay endpoints, with matched update/order coverage.",
        "reporting": "All scores in-sample, including DR100. Missing hue coverage => null plus valid count; unexpected non-finite model/metric values fail, not blanket sanitisation. Do not rewrite old NaN-bearing evidence.",
        "stop": "Return both fixed endpoints regardless of result; no selection, extension, additional rates/seeds, accumulation change, CN scoring, app change or unlocking K/monitor/finalHeldOut/T8/dr100Eval. A fitting improvement still needs a separately scoped validation plan.",
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"plan_sha256": sha(OUT), "block_learning_rates": rates, "descent_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
