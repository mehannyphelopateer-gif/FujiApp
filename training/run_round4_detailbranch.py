"""Phase 4 round 4 - one targeted encoder intervention, per instruction.
Round 3 falsified the WB-axis hypothesis; the follow-up structural
diagnosis found the 4 reproducible regressions differ from matched
controls specifically in highlight-blob *shape* (fewer, larger, more
compact/rounder), not overall brightness, and that a prediction-vs-
slicing test showed no elevated grid roughness for these scenes -
pointing at the encoder's predicted coefficients themselves, not the
grid's resolution.

This tests that directly: model.BilateralGridPredictor's use_detail_branch
adds a shallow, separate [luma, local-highpass] branch (only 2 stride-2
layers vs the main encoder's 5, so far more spatial detail survives) that
gets max-pooled (not averaged) and fused into the grid-coefficient head.
The 8x8x9 grid, its trilinear slicing, and luma_mode (back to rec709 -
round 3's WB-stable guide is not part of this intervention) are all
unchanged from the frozen baseline. +1.6% params.

Trains the same 3 seeds (42/43/44) as every prior round, then evaluates
every monitor scene against both the frozen baseline and the new
checkpoints with identical code. Predeclared success: resolve at least
one of the 4 reproducible compact-highlight failures (DSCF0553, DSCF0590,
DSCF0873, DSCF0879) without creating new persistent regressions.
DSCF0878 (broadly-lit, different interpolation-sensitivity pattern)
reported separately, not counted toward success.

train + monitor only - finalHeldOut has no code path here.
"""
import json
import os
from pathlib import Path

# Must happen before torch is imported by anything - see the 2026-09-15
# incident in run_round3_wbstable.py's history: this script also imports
# torch directly (for its post-training evaluation pass) ahead of
# `from train import ...`, which is where this normally gets set.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from dataset import CAL_DIR, Phase4PairDataset
from model import BilateralGridPredictor, apply_bilateral_grid, resize_for_network
from train import build_arg_parser, train_one_config

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"

BASE_CFG = {
    "grid_spatial": 8, "grid_luma": 9, "base_ch": 16, "lr": 1e-3,
    "epochs": 20, "lr_schedule": "cosine", "early_stopping_patience": 5,
    "grid_tv_weight": 0.0, "luma_mode": "rec709",
    "use_detail_branch": True, "detail_ch": 8,
}
SEEDS = [42, 43, 44]

REPRODUCIBLE_4 = ["DSCF0553.RAF", "DSCF0590.RAF", "DSCF0873.RAF", "DSCF0879.RAF"]
OUTLIER = "DSCF0878.RAF"
BASELINE_CHECKPOINTS = {
    42: "r2-b-lrsched-best.pt",
    43: "r2-b-lrsched-seed43-best.pt",
    44: "r2-b-lrsched-seed44-best.pt",
}


def run_one(cfg):
    existing = LOG_DIR / f"{cfg['run_name']}.json"
    if existing.exists():
        print(f"\n=== {cfg['run_name']} already completed, reusing saved log ===")
        return json.loads(existing.read_text())
    ap = build_arg_parser()
    args = ap.parse_args([])
    for k, v in cfg.items():
        setattr(args, k, v)
    print(f"\n=== starting {cfg['run_name']} ===")
    return train_one_config(args)


def load_model_from_ckpt(ckpt_name, device):
    ckpt = torch.load(CKPT_DIR / ckpt_name, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BilateralGridPredictor(
        grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"],
        low_res=cfg["low_res"], base_ch=cfg["base_ch"],
        use_detail_branch=cfg.get("use_detail_branch", False),
        detail_ch=cfg.get("detail_ch", 8),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def evaluate_all_seeds(checkpoints_by_seed, device, monitor_ds):
    models = {s: load_model_from_ckpt(c, device) for s, c in checkpoints_by_seed.items()}
    per_scene = {}
    for i in range(len(monitor_ds)):
        item = monitor_ds[i]
        name = item["name"]
        full_res = item["input"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        baseline_l1 = (full_res - target).abs().mean().item()
        per_scene[name] = {"baseline_l1": baseline_l1, "seeds": {}}
        for seed, (model, cfg) in models.items():
            with torch.no_grad():
                low_res_in = resize_for_network(full_res, cfg["low_res"])
                grid = model(low_res_in)
                pred = apply_bilateral_grid(grid, full_res, luma_mode=cfg.get("luma_mode", "rec709"))
                model_l1 = (pred - target).abs().mean().item()
            per_scene[name]["seeds"][seed] = {"model_l1": model_l1, "regressed": model_l1 > baseline_l1}
    return per_scene


def union_regressed(per_scene):
    return {n for n, d in per_scene.items() if any(s["regressed"] for s in d["seeds"].values())}


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    results = {}
    for seed in SEEDS:
        cfg = dict(BASE_CFG)
        cfg["seed"] = seed
        cfg["run_name"] = f"r4-detailbranch-seed{seed}"
        results[seed] = run_one(cfg)

    print("\n=== training done, evaluating both baseline and new checkpoints on every monitor scene ===")
    monitor_ds = Phase4PairDataset("monitor")

    new_checkpoints = {s: f"r4-detailbranch-seed{s}-best.pt" for s in SEEDS}
    baseline_per_scene = evaluate_all_seeds(BASELINE_CHECKPOINTS, device, monitor_ds)
    new_per_scene = evaluate_all_seeds(new_checkpoints, device, monitor_ds)

    baseline_union = union_regressed(baseline_per_scene)
    new_union = union_regressed(new_per_scene)
    resolved = baseline_union - new_union
    persisted = baseline_union & new_union
    new_regressions = new_union - baseline_union

    reproducible_4_status = {
        name: {
            "regressed_seeds_before": [s for s, v in baseline_per_scene[name]["seeds"].items() if v["regressed"]],
            "regressed_seeds_after": [s for s, v in new_per_scene[name]["seeds"].items() if v["regressed"]],
            "resolved": name in resolved,
        }
        for name in REPRODUCIBLE_4
    }
    outlier_status = {
        OUTLIER: {
            "regressed_seeds_before": [s for s, v in baseline_per_scene[OUTLIER]["seeds"].items() if v["regressed"]],
            "regressed_seeds_after": [s for s, v in new_per_scene[OUTLIER]["seeds"].items() if v["regressed"]],
            "note": "reported separately per instruction - broadly-lit/different interpolation-sensitivity pattern, not counted toward success",
        }
    }

    resolved_reproducible = [n for n in REPRODUCIBLE_4 if reproducible_4_status[n]["resolved"]]
    success = len(resolved_reproducible) > 0 and len(new_regressions) == 0

    report = {
        "intervention": "use_detail_branch=True (+1.6% params), luma_mode back to rec709, grid/slicing unchanged from frozen baseline",
        "seeds": SEEDS,
        "baseline_union_regressed": sorted(baseline_union),
        "new_union_regressed": sorted(new_union),
        "resolved": sorted(resolved),
        "persisted": sorted(persisted),
        "new_regressions": sorted(new_regressions),
        "reproducible_4_status": reproducible_4_status,
        "outlier_status_dscf0878": outlier_status,
        "predeclared_success_met": success,
        "predeclared_success_criteria": "resolve >=1 of the 4 reproducible compact-highlight failures AND zero new persistent regressions (DSCF0878 excluded from the count)",
        "baseline_per_scene": baseline_per_scene,
        "new_per_scene": new_per_scene,
    }
    (OUT_DIR / "round4-detailbranch-comparison.json").write_text(json.dumps(report, indent=2))

    print(f"\nbaseline union regressed: {sorted(baseline_union)}")
    print(f"new (detail-branch) union regressed: {sorted(new_union)}")
    print(f"resolved: {sorted(resolved)}")
    print(f"persisted: {sorted(persisted)}")
    print(f"NEW regressions introduced: {sorted(new_regressions)}")
    print(f"\npredeclared success met: {success}  (resolved reproducible: {resolved_reproducible})")
    print(f"\nDSCF0878 (outlier, reported separately): before={outlier_status[OUTLIER]['regressed_seeds_before']} after={outlier_status[OUTLIER]['regressed_seeds_after']}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
