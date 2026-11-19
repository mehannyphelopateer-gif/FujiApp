"""Metadata/order-only plan for a bounded Provia retention diagnostic.

No images, caches or checkpoint tensors are opened. Former development
rows were already canonical train rows; all 583 eligible rows are now
replayed, so nothing in this assay is independent validation.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

DIAG = Path(__file__).resolve().parent / "diagnostics"
RUN = DIAG / "provia-color-run-manifest.json"
RUN_SHA = "d6c04efd70df63aa38a9f2f08e0b291f858d5085a9670ffb7d3f1b3f4c6cf254"
REPORT = DIAG / "provia-color-pilot-report.json"
OUT = DIAG / "provia-replay-plan.json"
BLOCK_SIZE, BLOCKS, SEED = 467, 4, 42


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uniform_order(names, count, seed):
    if not names or len(names) != len(set(names)) or count < len(names):
        raise ValueError("Expected unique names and at least one full permutation")
    generator = torch.Generator().manual_seed(seed)
    order = []
    while len(order) < count:
        order.extend(names[i] for i in torch.randperm(len(names), generator=generator).tolist())
    return order[:count]


def main():
    if OUT.exists():
        raise ValueError("Replay plan exists; refusing to overwrite")
    if sha(RUN) != RUN_SHA:
        raise ValueError("Original data/run manifest changed")
    source = json.loads(RUN.read_text())
    previous = json.loads(REPORT.read_text())
    if previous["manifest_sha256"] != RUN_SHA:
        raise ValueError("Previous report has unexpected provenance")
    scenes = source["scenes"]
    if len(scenes) != 583 or Counter(s["role"] for s in scenes) != {"fit": 467, "development": 116}:
        raise ValueError("Unexpected original pool/roles")
    names = [s["name"] for s in scenes]
    count = BLOCK_SIZE * BLOCKS
    order = uniform_order(names, count, SEED)
    visits = Counter(order)
    if set(visits) != set(names) or set(visits.values()) != {3, 4}:
        raise ValueError("Unexpected replay coverage")
    old_plan = json.loads((DIAG / "provia-color-pilot-plan.json").read_text())
    if set(names) & set(old_plan["excluded_names"]):
        raise ValueError("Protected K/T8 name included")
    lr, eta = 1e-4, 1e-6
    result = {
        "status": "PLAN fixed before replay implementation/fitting; actual run manifest still required",
        "question": "Does retaining all eligible inherited training examples prevent the degradation observed when 116 previously fitted examples were withheld?",
        "scope": "Full-replay retention/fitting diagnostic; all scores IN-SAMPLE, not validation or acceptance",
        "source_run_manifest_sha256": RUN_SHA,
        "previous_pilot_report_sha256": sha(REPORT),
        "source_checkpoint_sha256": source["bindings"]["source_checkpoint"],
        "data": "Reuse the exact 583 verified/hash-bound Provia pairs from the old run manifest, with its code/evidence bindings; all are canonical train rows. K/T8/monitor/finalHeldOut/dr100Eval remain excluded.",
        "coverage_by_capture_dr": dict(Counter(s["captureDynamicRange"] for s in scenes)),
        "arms": source["arms"],
        "initialization": "Separate read-only copies of original R11 seed 42, not either failed pilot endpoint; all original parameters trainable",
        "training": {"seed": SEED, "batch": 1, "optimizer": source["training"]["optimizer"],
                     "learning_rates_by_467_update_block": [eta + (lr - eta) * (1 + math.cos(math.pi * b / 12)) / 2 for b in range(BLOCKS)],
                     "schedule": "Same first four blocks of original cosine T_max=12/eta_min=1e-6; advance only each 467 completed updates; no horizon rescaling",
                     "blocks": BLOCKS, "updates_per_block": BLOCK_SIZE, "total_updates": count,
                     "note": "Blocks are not full 583-scene epochs. Exact update count matches the earlier pilot's first four 467-scene epochs.",
                     "AMP": None, "augmentation": None, "HPO": None, "early_stopping": None, "runs": 2},
        "ordering": {"algorithm": "Concatenate full torch.randperm(583) permutations using one Generator(seed=42), truncate to 1868; same recorded order for both arms",
                     "source_name_order": names, "sha256_compact_JSON_order": hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest(),
                     "visit_counts": dict(sorted(visits.items())), "updates": order},
        "evaluation": "At update 0 and 1868 only, stream all 583 pairs. Report all-pool, former-fit 467, former-development 116, per-session and per-DR groups, every scene and worst regression. All are training scores now; label this in every report. Compare to R11 and the frozen old epoch-4 results only with differing training coverage labelled.",
        "report_metrics": "Unclamped linear L1, display DeltaE mean/median/mean-p95, signed tone Lab/chroma/hue and clipping; candidate-control and each-R11 differences",
        "stop": "Return both fixed endpoints regardless of result. No checkpoint selection, acceptance, longer schedule, extra seed, app change or automatic follow-up.",
        "preflight": "New runner/manifest committed before fitting; bind old verified manifest/cache/code/source hashes, replay order and schedule. Refusal/no-op/finite/full-resolution checks, exact resume, source immutable; do not modify prior scripts/evidence or corpus tiers.",
        "limitation": "R11 had originally fit these scenes and both replay arms now fit them too. This restores comparable fitting status but cannot measure unseen-session generalisation; sampling/order and coverage change together, and one seed is diagnostic only.",
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"plan_sha256": sha(OUT), "updates": count, "coverage": len(visits),
                      "visits": dict(Counter(visits.values())), "capture_dr": result["coverage_by_capture_dr"]}, indent=2))


if __name__ == "__main__":
    main()
