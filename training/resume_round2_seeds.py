"""Resume round 2 after the 2026-09-15 external-drive I/O crash killed
run_round2.py partway through the seed43 stability run (r2-b-lrsched and
r2-b-smooth had already finished and were on disk; round2-comparison.json
was never written). Re-derives the winner from the two saved run logs
(same rule as run_round2.py's selection_key) and runs the 2 remaining
seed-stability runs, then writes the same final comparison run_round2.py
would have. Not a general resume mechanism - a one-off for this incident.
"""
import json
import statistics
from pathlib import Path

from train import build_arg_parser, train_one_config

LOG_DIR = Path(__file__).resolve().parent / "runs"


def selection_key(summary):
    return (summary["final_monitor_regressed"], summary["history"][-1]["monitor_worst_regression"], summary["best_monitor_l1"])


def run_one(cfg):
    # Idempotent: if this exact run already completed (its log exists on
    # disk), reuse it rather than re-run - makes this script safe to
    # re-invoke after a crash without losing already-finished work. Only
    # skips on a clean, fully-written log; a run that crashed mid-way never
    # wrote one, so it correctly re-runs from scratch.
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


def main():
    if (LOG_DIR / "round2-comparison.json").exists():
        print("round2-comparison.json already exists - nothing left to resume")
        return
    r1 = json.loads((LOG_DIR / "r2-b-lrsched.json").read_text())
    r2 = json.loads((LOG_DIR / "r2-b-smooth.json").read_text())
    winner = min([r1, r2], key=selection_key)
    print(f"winner (re-derived from saved logs): {winner['run_name']}")

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
        "incident_note": (
            "The original run_round2.py process crashed 2026-09-15 mid-seed43 "
            "on a transient external-drive I/O error (OSError: Device not "
            "configured). r2-b-lrsched and r2-b-smooth had already completed "
            "and are unaffected. Resumed via resume_round2_seeds.py, which "
            "re-derives the winner from the saved run logs rather than "
            "re-running the already-completed pair. dataset.py gained retry "
            "logic for transient OSErrors as a result."
        ),
    }
    (LOG_DIR / "round2-comparison.json").write_text(json.dumps(comparison, indent=2))

    print("\n=== ROUND 2 SUMMARY (resumed) ===")
    print(f"winner: {winner['run_name']}")
    print(f"seed stability across seeds {stability['seeds']}: "
          f"mean monitor_l1={stability['best_monitor_l1_mean']:.5f}  stdev={stability['best_monitor_l1_stdev']:.5f}")
    print(f"regressed counts per seed: {regressed_counts}")
    print(f"worst regressions per seed: {[round(w,5) for w in worst_regressions]}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
