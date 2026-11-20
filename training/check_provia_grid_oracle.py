"""Prepare/smoke a TRAIN-ONLY, per-image free-grid diagnostic; never fit.

Bypass the encoder with a detached copy of its predicted grid. Keep the
original slicing and the frozen refinement network in the forward path.
This is target-assisted diagnostic machinery, NOT an inference fallback.
No optimiser, checkpoint writing, or protected-tier loader is provided.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import sys
from pathlib import Path

import torch

import color_loss as CL
import provia_extended as E
from model import apply_bilateral_grid, resize_for_network

P = E.P
PLAN = P.DIAG / "provia-grid-oracle-plan.json"
SMOKE = P.DIAG / "provia-grid-oracle-preflight.json"
THIS = Path(__file__).resolve()


def oracle_forward(model, config, image, grid):
    """Same hybrid decoder; only the per-image grid is an independent leaf."""
    coarse = apply_bilateral_grid(grid, image, config.get("luma_mode", "rec709"))
    return model.refinement(image, coarse)[0]


def prepare():
    if PLAN.exists():
        P.fail("oracle plan exists; refusing overwrite")
    _, data, _, _ = E.load()
    by = {s["name"]: s for s in data["scenes"]}
    if len(E.REVIEW_CASES) != 8 or len(set(E.REVIEW_CASES)) != 8 or set(E.REVIEW_CASES) - set(by):
        P.fail("oracle cases must be the fixed eight train-only review cases")
    out = {
        "scope": "Target-assisted per-image diagnostic, not validation or a deployable model",
        "question": "Can different coefficients in the existing 8x8x9 grid reduce the residual, with the encoder bypassed and refinement frozen?",
        "bindings": {
            "source_R11": P.SOURCE_SHA,
            "pilot_manifest": P.MANIFEST_SHA,
            "extended_manifest": E.MANIFEST_SHA,
            "extended_report": P.sha256(E.REPORT),
            "code": {p: P.sha256(P.TRAIN / p) for p in
                     (THIS.name, "model.py", "color_loss.py", "color_metrics.py")},
        },
        "scenes": [by[n] for n in E.REVIEW_CASES],
        "initialization": "Independent original R11 predicted grid for each scene, not either replay endpoint",
        "frozen": "All original model parameters, refinement weights, input, capture-DR condition, luma guide, grid dimensions, source checkpoint",
        "trainable": "Only one detached grid tensor per scene (1,12,9,8,8); refinement recomputed on the updated grid output",
        "proposed_runner": {
            "optimizer": "fresh Adam per scene, betas=(0.9,0.999), eps=1e-8, weight_decay=0",
            "lr": 0.001, "steps": 200, "schedule": None,
            "loss": "color_loss.per_scene_terms combined (original display-clamped loss)",
            "observations": [0, 25, 50, 100, 200],
            "endpoint": 200,
            "sampling": "all pixels at the existing working resolution; no augmentation",
            "save_deployable_weights": False,
        },
        "report": [
            "unchanged display CIEDE2000 mean/p95, linear L1, both components and combined objective",
            "loss trace, grid gradient norm, coefficient distance from R11, finiteness, negative/over-one tails",
            "working-scale panels at step 0 and fixed endpoint, all eight scenes reported",
            "compare original R11 and frozen low-rate/extended results for context only",
        ],
        "interpretation": [
            "A substantial display improvement demonstrates coefficient headroom on those scenes, not that the encoder can predict those coefficients from an unseen input.",
            "An unsuccessful finite-budget fit is inconclusive about representation: optimiser and loss limitations remain. Do not declare an architectural ceiling from non-improvement.",
            "Do not equate this target-assisted oracle with generalisation, recipe readiness, or pixel-perfect parity.",
        ],
        "boundary": "Commit a separate hash-bound runner and run manifest before the 8x200-step diagnostic. No full-corpus fit, sweep, extra scenes, selection, app/CN change or K/monitor/finalHeldOut/T8/dr100Eval access.",
    }
    PLAN.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print("oracle plan sha256", P.sha256(PLAN), flush=True)


def smoke(expected_sha):
    if P.sha256(PLAN) != expected_sha:
        P.fail("oracle plan hash mismatch")
    if SMOKE.exists():
        P.fail("oracle preflight exists; refusing overwrite")
    plan = json.loads(PLAN.read_text())
    if plan["bindings"]["source_R11"] != P.SOURCE_SHA:
        P.fail("source binding mismatch")
    for name, sha in plan["bindings"]["code"].items():
        if P.sha256(P.TRAIN / name) != sha:
            P.fail("bound oracle code changed: " + name)
    _, data, _, _ = E.load()
    if P.sha256(E.REPORT) != plan["bindings"]["extended_report"]:
        P.fail("extended evidence changed")
    loader = P.Pairs(data)
    by = {s["name"]: s for s in data["scenes"]}
    if plan["scenes"] != [by[n] for n in E.REVIEW_CASES]:
        P.fail("oracle scene binding mismatch")
    device = P.device()
    model, config = P.build(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if config["grid_spatial"] != 8 or config["grid_luma"] != 9:
        P.fail("unexpected original grid shape")
    records = {}
    for name in E.REVIEW_CASES:
        x, target, condition = loader.get(name)
        x, target, condition = x.to(device), target.to(device), condition.to(device)
        with torch.no_grad():
            original = P.fwd(model, config, x, condition, device)
            initial_grid = model.grid(resize_for_network(x, config["low_res"]), condition)
        grid = torch.nn.Parameter(initial_grid.detach().clone())
        result = oracle_forward(model, config, x, grid)
        difference = float((result.detach() - original).abs().max().item())
        if difference != 0.0 or tuple(grid.shape) != (1, 12, 9, 8, 8):
            P.fail("oracle warm start is not exactly original R11")
        loss = CL.per_scene_terms(result, target)["combined"].mean()
        loss.backward()
        if not torch.isfinite(loss) or grid.grad is None or not torch.isfinite(grid.grad).all():
            P.fail("non-finite oracle loss/gradient")
        gradient_norm = float(grid.grad.norm().item())
        if gradient_norm <= 0 or any(p.grad is not None for p in model.parameters()):
            P.fail("grid has no gradient or frozen model received gradients")
        records[name] = {"max_abs_vs_original_R11": difference,
                         "combined": float(loss.item()), "grid_gradient_norm": gradient_norm,
                         "model_gradients": False, "optimizer_steps": 0}
        print(name, records[name], flush=True)
        del x, target, condition, initial_grid, grid, original, result, loss
    P.check_source()
    SMOKE.write_text(json.dumps({"scope": "No fitting; no optimizer created; no checkpoint saved",
                                "plan_sha256": expected_sha, "device": str(device),
                                "per_scene": records, "source_unchanged": True}, indent=2,
                               allow_nan=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "prepare":
        prepare()
    elif len(sys.argv) == 3 and sys.argv[1] == "smoke":
        smoke(sys.argv[2])
    else:
        raise SystemExit("Use prepare | smoke <frozen-plan-sha256>")
