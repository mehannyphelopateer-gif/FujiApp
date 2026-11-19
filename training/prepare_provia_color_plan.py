"""Freeze a metadata-only scope for the Provia colour-loss pilot.

Reads only train-tier rows plus already-known K/T8 exclusion names. No
image/cache/RAF inspection, monitor or finalHeldOut analysis, or fitting.
This is a PLAN, not the later per-file-hash-bound training run manifest.
"""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training" / "diagnostics"
CORPUS = ROOT / "calibration-input" / "phase4-corpus-manifest.json"
K = DIAG / "cuban-step1-run-manifest.json"
T8 = DIAG / "cuban-control-support-selection.json"
OUT = DIAG / "provia-color-pilot-plan.json"
SOURCE = ROOT / "training" / "checkpoints" / "r11-drcond-seed42-best.pt"
SOURCE_SHA = "1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Plan already exists; refusing to overwrite")
    train = json.loads(CORPUS.read_text())["tiers"]["train"]["scenes"]
    excluded = {r["name"] for r in json.loads(K.read_text())["K_eval_only"]}
    excluded |= {r["name"] for r in json.loads(T8.read_text())["scenes"]}
    if len(train) != 599 or len({r["name"] for r in train}) != 599 or len(excluded) != 16:
        raise ValueError("Unexpected source pool; scope needs review")
    if not excluded <= {r["name"] for r in train} or sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Protected names/source checkpoint mismatch")
    allowed = [r for r in train if r["name"] not in excluded]
    if len(allowed) != 583 or any(not r.get("pairReady") or r.get("captureDynamicRange") not in (100, 200, 400) for r in allowed):
        raise ValueError("Pool missing a ready pair or resolved capture DR")
    sessions = sorted({r["sessionIndex"] for r in allowed})
    candidates = []
    for size in (1, 2, 3):
        for ids in itertools.combinations(sessions, size):
            dev = [r for r in allowed if r["sessionIndex"] in ids]
            if not 0.15 <= len(dev) / len(allowed) <= 0.25:
                continue
            if {r["captureDynamicRange"] for r in dev} != {100, 200, 400}:
                continue
            candidates.append(((abs(5 * len(dev) - len(allowed)), size, ids), ids))
    if not candidates:
        raise ValueError("No metadata-only split satisfies the rule")
    ids = min(candidates)[1]
    rows = []
    for r in sorted(allowed, key=lambda r: r["name"]):
        rows.append({"name": r["name"], "shoot": r["shoot"], "session": r["sessionIndex"],
                     "captureDynamicRange": r["captureDynamicRange"],
                     "role": "development" if r["sessionIndex"] in ids else "fit"})
    counts = {role: {"n": sum(r["role"] == role for r in rows),
                     "capture_dr": dict(Counter(r["captureDynamicRange"] for r in rows if r["role"] == role))}
              for role in ("fit", "development")}
    result = {
        "status": "Scope fixed before pixel/cache preparation or pilot training; not a runnable data manifest",
        "purpose": "Compare full-pool Provia fine-tuning: linear L1 vs a fixed paired pixel-colour objective",
        "source_hashes": {"corpus_manifest": sha(CORPUS), "K_exclusions": sha(K), "T8_exclusions": sha(T8),
                          "r11_checkpoint": SOURCE_SHA, "color_loss_code": sha(ROOT / "training" / "color_loss.py"),
                          "synthetic_loss_checks": sha(DIAG / "provia-color-loss-checks.json")},
        "excluded_names": sorted(excluded), "excluded_pixel_access": "K, T8, monitor, finalHeldOut, dr100Eval, rejected capture attempts",
        "split_rule": "Train tier only, exclude K8+T8. Hold out 1-3 whole sessions, 15-25% of eligible scenes, covering all three capture DR values. Minimise abs(5*n_dev-n_pool), then session count, then lexical session tuple. Metadata counts only.",
        "development_sessions": ids, "counts": counts,
        "DR100_limitation": "All 12 eligible DR100 captures belong to development session 14; none in fine-tuning fit. Both arms inherit R11's earlier exposure to all these scenes. Development tests retention under this adaptation, not fresh DR100 or unseen-session generalisation. No scene reassignment/replication to hide the gap.",
        "model": "Unchanged R11 HybridPredictor architecture, all original parameters trainable in both arms, source loaded read-only into independent copies",
        "control_loss": "mean abs(unclamped linear pred - target)",
        "candidate_loss": "0.5 * (linear L1 + mean abs(display-clamped D65 Lab pred - target)/100), pixel-aligned",
        "training": {"seed": 42, "batch_size": 1, "epochs": 12, "lr": 0.0001,
                     "optimizer": "fresh Adam, betas 0.9/0.999, eps 1e-8, weight decay 0",
                     "schedule": "cosine T_max=12, eta_min=1e-6, no early stopping",
                     "ordering": "identical per-epoch scene order across the two arms", "augmentation": None,
                     "HPO": None, "runs": 2},
        "evaluations": "Epochs 0,4,8,12; epoch 12 is predeclared endpoint, no checkpoint picking from curves. Save recoverable model/optimizer/scheduler/RNG/order state each epoch.",
        "primary": "equal-scene development mean DeltaE00, candidate minus matched control and unchanged R11; report every scene/session/DR group",
        "screening": "Further consideration requires endpoint mean DeltaE00 strictly below both control and R11, with mean per-scene p95 no worse than either. Every regression and its magnitude remains reported. This is a development screen only; no model selection/integration/accuracy acceptance follows automatically.",
        "preflight_required": "Separate implementation/data manifest committed before fitting; explicit allowlist loader; per-file hashes; source TIFF settings/profile/16-bit validation; hash-bound targets freshly derived or verified from current TIFFs (legacy cache alone insufficient); synthetic/full-working-resolution smoke; no-op warm-start; refusal tests; source hash unchanged",
        "compute": "local MPS only; no cloud spend; measure throughput before an ETA",
        "stop": "Return the two-run report at endpoint regardless of result; no new seeds/hyperparameters/monitor access/CN scoring/final fit/app changes",
        "scenes": rows,
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"plan_sha256": sha(OUT), "development_sessions": ids, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
