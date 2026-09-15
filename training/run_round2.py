"""Phase 4 model-selection, round 2 - predeclared, small, targeted at the
5 monitor regressions and epoch-8->15 drift round 1's winner
(cand-b-finer-grid, 8x8 spatial x 9 luma grid) showed. Not open-ended
tuning: three runs, fixed in advance -

  1. b-lrsched:  same architecture, + cosine LR schedule + early stopping
                 (targets the epoch-8->15 monitor drift)
  2. b-smooth:   same architecture, + grid total-variation regularization
                 (targets the 5 regressions specifically - discourages the
                 grid from fitting a locally noisy cell to one hard scene)
  3. winner x2 more seeds: whichever of 1/2 wins (fewer regressions first,
                 then worst-case regression magnitude, then monitor L1),
                 repeated with 2 additional seeds to check stability.

Loss stays plain linear-RGB L1 (+ the TV term only in run 2, and never in
the reported/compared metric - see train.py). No perceptual loss.
finalHeldOut is not accessed - trains/evaluates on train+monitor only,
same hard-refusal as round 1 (dataset.list_scenes has no code path to it).
"""
import json
from pathlib import Path

from train import build_arg_parser, train_one_config

LOG_DIR = Path(__file__).resolve().parent / "runs"

# cand-b-finer-grid's architecture, round 1's winner - held fixed as the
# base for every round-2 variant, per instruction.
BASE_ARCH = {"grid_spatial": 8, "grid_luma": 9, "base_ch": 16, "lr": 1e-3}

RUN1 = {
    "run_name": "r2-b-lrsched",
    **BASE_ARCH,
    "epochs": 20,  # raised cap - a schedule benefits from more room; bounded by early stopping below
    "lr_schedule": "cosine",
    "early_stopping_patience": 5,
    "grid_tv_weight": 0.0,
    "seed": 42,
}
RUN2 = {
    "run_name": "r2-b-smooth",
    **BASE_ARCH,
    "epochs": 15,  # matches round 1's cand-b exactly, isolating the regularization variable
    "lr_schedule": "none",
    "early_stopping_patience": None,
    # Calibrated, not swept: raw grid_smoothness_loss on an untrained model
    # of this shape measures ~1.06; weight 0.005 -> ~0.005 contribution,
    # roughly 10% of the ~0.05 initial reconstruction L1 - meaningful
    # regularization pressure without dominating the reconstruction
    # objective. Chosen once, ahead of running this.
    "grid_tv_weight": 0.005,
    "seed": 42,
}


def run_one(cfg):
    ap = build_arg_parser()
    args = ap.parse_args([])
    for k, v in cfg.items():
        setattr(args, k, v)
    print(f"\n=== starting {cfg['run_name']} ===")
    return train_one_config(args)


def selection_key(summary):
    # fewer regressions first (round 2's explicit target), then worst-case
    # regression magnitude, then overall monitor L1.
    return (summary["final_monitor_regressed"], summary["history"][-1]["monitor_worst_regression"], summary["best_monitor_l1"])


def main():
    r1 = run_one(RUN1)
    r2 = run_one(RUN2)

    candidates = [r1, r2]
    candidates.sort(key=selection_key)
    winner = candidates[0]
    print(f"\nround-2 winner (of lrsched vs smooth): {winner['run_name']}")

    winner_cfg = dict(winner["config"])
    winner_cfg["run_name"] = f"{winner['run_name']}-seed43"
    winner_cfg["seed"] = 43
    r_seed43 = run_one(winner_cfg)

    winner_cfg2 = dict(winner["config"])
    winner_cfg2["run_name"] = f"{winner['run_name']}-seed44"
    winner_cfg2["seed"] = 44
    r_seed44 = run_one(winner_cfg2)

    seed_runs = [winner, r_seed43, r_seed44]
    monitor_l1s = [r["best_monitor_l1"] for r in seed_runs]
    regressed_counts = [r["final_monitor_regressed"] for r in seed_runs]
    worst_regressions = [r["history"][-1]["monitor_worst_regression"] for r in seed_runs]

    import statistics
    stability = {
        "seeds": [r["config"]["seed"] for r in seed_runs],
        "best_monitor_l1_mean": statistics.mean(monitor_l1s),
        "best_monitor_l1_stdev": statistics.stdev(monitor_l1s) if len(monitor_l1s) > 1 else 0.0,
        "best_monitor_l1_values": monitor_l1s,
        "regressed_counts": regressed_counts,
        "worst_regressions": worst_regressions,
    }

    comparison = {
        "round1_baseline": "cand-b-finer-grid: best_monitor_l1=0.03809, regressed=5/76, best_epoch=8",
        "run1_lrsched": {"run_name": r1["run_name"], "best_monitor_l1": r1["best_monitor_l1"], "best_epoch": r1["best_epoch"],
                          "epochs_run": r1["epochs_run"], "stopped_early": r1["stopped_early"],
                          "final_monitor_regressed": r1["final_monitor_regressed"],
                          "worst_regression": r1["history"][-1]["monitor_worst_regression"]},
        "run2_smooth": {"run_name": r2["run_name"], "best_monitor_l1": r2["best_monitor_l1"], "best_epoch": r2["best_epoch"],
                         "final_monitor_regressed": r2["final_monitor_regressed"],
                         "worst_regression": r2["history"][-1]["monitor_worst_regression"]},
        "winner": winner["run_name"],
        "winner_seed_runs": [r["run_name"] for r in seed_runs],
        "stability": stability,
    }
    (LOG_DIR / "round2-comparison.json").write_text(json.dumps(comparison, indent=2))

    print("\n=== ROUND 2 SUMMARY ===")
    print(f"run1 (lrsched): best_monitor_l1={r1['best_monitor_l1']:.5f}  regressed={r1['final_monitor_regressed']}/76  "
          f"worst_regression={r1['history'][-1]['monitor_worst_regression']:.5f}  best_epoch={r1['best_epoch']}  "
          f"epochs_run={r1['epochs_run']}  stopped_early={r1['stopped_early']}")
    print(f"run2 (smooth):  best_monitor_l1={r2['best_monitor_l1']:.5f}  regressed={r2['final_monitor_regressed']}/76  "
          f"worst_regression={r2['history'][-1]['monitor_worst_regression']:.5f}  best_epoch={r2['best_epoch']}")
    print(f"\nwinner: {winner['run_name']}")
    print(f"seed stability across seeds {stability['seeds']}: "
          f"mean monitor_l1={stability['best_monitor_l1_mean']:.5f}  stdev={stability['best_monitor_l1_stdev']:.5f}")
    print(f"regressed counts per seed: {regressed_counts}")
    print(f"worst regressions per seed: {[round(w,5) for w in worst_regressions]}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
