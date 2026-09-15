"""Phase 4 model-selection driver - runs a small, reasoned set of candidate
architecture/hyperparameter configs (not an exhaustive search - three
candidates chosen to test the axes SS4 flags as open questions: grid
resolution and encoder capacity), each via train_one_config, then picks
the candidate with the best monitor-tier L1 (not train - the point of a
monitor tier is to select against data the optimizer never sees).

Trains/evaluates on train+monitor ONLY. Never touches finalHeldOut -
selection here does not authorize a held-out run; that is a separate,
explicit later step per docs/phase4-scene-adaptive-scope.md SS7.
"""
import json
import time
from pathlib import Path

from train import build_arg_parser, train_one_config

LOG_DIR = Path(__file__).resolve().parent / "runs"

CANDIDATES = [
    {
        "run_name": "cand-a-baseline",
        "grid_spatial": 4, "grid_luma": 7, "base_ch": 16, "lr": 1e-3,
        "note": "SS4's suggested starting point - matches Phase 3's own hand-fit grid resolution",
    },
    {
        "run_name": "cand-b-finer-grid",
        "grid_spatial": 8, "grid_luma": 9, "base_ch": 16, "lr": 1e-3,
        "note": "tests SS4's 'a learned model may benefit from finer resolution than hand-fitting could support'",
    },
    {
        "run_name": "cand-c-higher-capacity",
        "grid_spatial": 4, "grid_luma": 7, "base_ch": 32, "lr": 1e-3,
        "note": "tests whether more encoder capacity helps at this corpus size (2x base channels, ~4x conv params)",
    },
]


def main():
    ap = build_arg_parser()
    ap.add_argument("--candidates", default=None, help="comma-separated run_names to run (default: all)")
    args = ap.parse_args()

    wanted = set(args.candidates.split(",")) if args.candidates else None
    results = []
    t0 = time.time()

    for cand in CANDIDATES:
        if wanted and cand["run_name"] not in wanted:
            continue
        run_args = ap.parse_args([])  # fresh defaults
        for k, v in vars(args).items():
            setattr(run_args, k, v)
        for k, v in cand.items():
            if k == "note":
                continue
            setattr(run_args, k, v)
        print(f"\n=== starting {cand['run_name']} ({cand['note']}) ===")
        summary = train_one_config(run_args)
        summary["note"] = cand["note"]
        results.append(summary)

    total_time = time.time() - t0
    results.sort(key=lambda r: r["best_monitor_l1"])
    selected = results[0] if results else None

    comparison = {
        "candidates": results,
        "selected": selected["run_name"] if selected else None,
        "selection_rule": "lowest best-epoch monitor L1 across candidates",
        "total_seconds": round(total_time, 1),
    }
    (LOG_DIR / "model-selection-comparison.json").write_text(json.dumps(comparison, indent=2))

    print("\n=== MODEL SELECTION SUMMARY ===")
    for r in results:
        print(
            f"{r['run_name']:24s} params={r['n_params']:>8,}  "
            f"best_epoch={r['best_epoch']:>3}  best_monitor_l1={r['best_monitor_l1']:.5f}  "
            f"final_monitor_regressed={r['final_monitor_regressed']}/{r['final_monitor_n']}  "
            f"total_s={r['total_seconds']:.0f}"
        )
    if selected:
        print(f"\nSELECTED: {selected['run_name']} (best_monitor_l1={selected['best_monitor_l1']:.5f})")
        print(f"checkpoint: {selected['best_checkpoint']}")
    print("\nfinalHeldOut was not accessed by this run.")


if __name__ == "__main__":
    main()
