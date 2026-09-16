"""Phase 4 round 3 - one targeted intervention, per instruction (no more
blind sweeps): keep the winning model (round 1's cand-b-finer-grid /
round 2's r2-b-lrsched architecture, 8x8 spatial x 9 luma grid, base_ch=16,
cosine LR schedule + early stopping) completely unchanged except for the
coordinate used to index the bilateral grid's luma axis - see
model.compute_luma_guide's "green_channel" mode. The as-shot-WB color
image stays both the model's input and output throughout; only the
lookup coordinate for grid_sample changes.

Trains 3 seeds (42/43/44, matching round 2's stability check exactly),
then evaluates every monitor scene against each of the 3 new checkpoints
and compares against the frozen round-2 baseline's per-scene results
(recomputed fresh here for a fair apples-to-apples comparison, both using
the same evaluation code).

Predeclared success (per instruction): fewer union-of-seed monitor
regressions, especially improvement on the 4 reproducible Auto-WB/
highlight failures (DSCF0878, DSCF0873, DSCF0879, DSCF0590), with no new
persistent regressions. DSCF0878 is reported separately, not tuned to -
its residual pattern didn't match the other 3 in the 2026-09-15 diagnosis.

train + monitor only - finalHeldOut has no code path here, same as
everywhere else in training/.
"""
import json
import os
from pathlib import Path

# Must happen before torch is imported by anything - train.py normally
# sets this at its own module load, but this script is the first one to
# `import torch` directly (for the post-training evaluation pass) ahead
# of `from train import ...`, which left grid_sampler_3d_backward's MPS
# fallback disabled and crashed every training run with a hard
# NotImplementedError (2026-09-15). Set it here first, defensively.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from dataset import CAL_DIR, Phase4PairDataset
from model import BilateralGridPredictor, apply_bilateral_grid, resize_for_network
from train import build_arg_parser, train_one_config

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"

# Exact architecture/schedule as round 2's winner (r2-b-lrsched), only
# luma_mode changes.
BASE_CFG = {
    "grid_spatial": 8, "grid_luma": 9, "base_ch": 16, "lr": 1e-3,
    "epochs": 20, "lr_schedule": "cosine", "early_stopping_patience": 5,
    "grid_tv_weight": 0.0, "luma_mode": "green_channel",
}
SEEDS = [42, 43, 44]

# The 2026-09-15 diagnosis's findings, for comparison.
BASELINE_REPRODUCIBLE_4 = ["DSCF0878.RAF", "DSCF0873.RAF", "DSCF0879.RAF", "DSCF0590.RAF"]
BASELINE_SEED_DEPENDENT = ["DSCF0553.RAF"]
BASELINE_OUTLIER = "DSCF0878.RAF"  # different residual pattern - report separately, don't tune to it
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
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def evaluate_all_seeds(checkpoints_by_seed, device, monitor_ds):
    """checkpoints_by_seed: {seed: ckpt_filename}. Returns per-scene dict:
    name -> {baseline_l1, seeds: {seed: {model_l1, regressed}}}."""
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
            per_scene[name]["seeds"][seed] = {
                "model_l1": model_l1,
                "regressed": model_l1 > baseline_l1,
            }
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
        cfg["run_name"] = f"r3-wbstable-seed{seed}"
        results[seed] = run_one(cfg)

    print("\n=== training done, evaluating both baseline and new checkpoints on every monitor scene ===")
    monitor_ds = Phase4PairDataset("monitor")

    new_checkpoints = {s: f"r3-wbstable-seed{s}-best.pt" for s in SEEDS}
    baseline_per_scene = evaluate_all_seeds(BASELINE_CHECKPOINTS, device, monitor_ds)
    new_per_scene = evaluate_all_seeds(new_checkpoints, device, monitor_ds)

    baseline_union = union_regressed(baseline_per_scene)
    new_union = union_regressed(new_per_scene)

    resolved = baseline_union - new_union       # regressed before, not now
    persisted = baseline_union & new_union      # regressed before and still now
    new_regressions = new_union - baseline_union  # NEW failures the old model didn't have

    reproducible_4_status = {
        name: {
            "was_regressed_all_3_seeds_before": name in baseline_union,
            "regressed_seeds_before": [s for s, v in baseline_per_scene[name]["seeds"].items() if v["regressed"]],
            "regressed_seeds_after": [s for s, v in new_per_scene[name]["seeds"].items() if v["regressed"]],
        }
        for name in BASELINE_REPRODUCIBLE_4
    }

    outlier_status = {
        BASELINE_OUTLIER: {
            "regressed_seeds_before": [s for s, v in baseline_per_scene[BASELINE_OUTLIER]["seeds"].items() if v["regressed"]],
            "regressed_seeds_after": [s for s, v in new_per_scene[BASELINE_OUTLIER]["seeds"].items() if v["regressed"]],
            "note": "reported separately per instruction - its residual pattern differed from the other 3 in the 2026-09-15 diagnosis, not tuned to specifically",
        }
    }

    success = (
        len(new_union) < len(baseline_union)
        and len(new_regressions) == 0
        and any(reproducible_4_status[n]["was_regressed_all_3_seeds_before"] and not reproducible_4_status[n]["regressed_seeds_after"] for n in BASELINE_REPRODUCIBLE_4 if n != BASELINE_OUTLIER)
    )

    report = {
        "intervention": "luma_mode=green_channel (WB-stable luma guide), architecture/schedule otherwise identical to round 2's winner",
        "seeds": SEEDS,
        "baseline_union_regressed": sorted(baseline_union),
        "new_union_regressed": sorted(new_union),
        "resolved": sorted(resolved),
        "persisted": sorted(persisted),
        "new_regressions": sorted(new_regressions),
        "reproducible_4_status": reproducible_4_status,
        "outlier_status_dscf0878": outlier_status,
        "predeclared_success_met": success,
        "predeclared_success_criteria": "fewer union-of-seed regressions AND zero new persistent regressions AND at least one of the 4 reproducible (excluding the DSCF0878 outlier) resolved under all seeds",
        "baseline_per_scene": baseline_per_scene,
        "new_per_scene": new_per_scene,
    }
    (OUT_DIR / "round3-wbstable-comparison.json").write_text(json.dumps(report, indent=2))

    print(f"\nbaseline union regressed: {sorted(baseline_union)}")
    print(f"new (WB-stable) union regressed: {sorted(new_union)}")
    print(f"resolved: {sorted(resolved)}")
    print(f"persisted: {sorted(persisted)}")
    print(f"NEW regressions introduced: {sorted(new_regressions)}")
    print(f"\npredeclared success met: {success}")
    print(f"\nDSCF0878 (outlier, reported separately): before={outlier_status[BASELINE_OUTLIER]['regressed_seeds_before']} after={outlier_status[BASELINE_OUTLIER]['regressed_seeds_after']}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
