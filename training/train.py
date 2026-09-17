"""Phase 4 training loop - local only, Apple MPS backend.
docs/phase4-scene-adaptive-scope.md SS1/SS4/SS5 + the 2026-09-14 compute
decision (local PyTorch/MPS, no cloud spend).

Trains on the 'train' tier only, reports monitor-tier metrics each epoch
for model-selection visibility. This is the fitting loop itself, run once
per candidate architecture/hyperparameter combination - see
run_model_selection.py (round 1) and run_round2.py (round 2: LR
schedule/early stopping, grid smoothness regularization, seed stability)
for the actual multi-candidate comparisons.

Baseline: per §7's real acceptance criterion ("every scene at least as
good as no correction at all"), every run also reports the no-correction
L1 (input vs target, untouched) so train/monitor numbers are meaningful
against that bar, not just as raw numbers - plus a per-scene "regressed"
count (model worse than no-correction on that scene) and the single
worst-case regression magnitude/scene. This is a fast proxy for §7's real
gate, not the gate itself - the real one runs leave-one-session-out and,
at the end, once against finalHeldOut.

Loss stays plain linear-RGB L1 reconstruction (+ an optional grid
smoothness regularizer, off by default) - no perceptual loss, per
instruction: the goal is pixel-level Fuji parity, not perceptual
similarity.

NEVER imports or references the finalHeldOut tier - dataset.list_scenes
refuses that tier outright, so there is nothing to accidentally wire in
here.
"""
import argparse
import json
import os
import random
import resource
import time
from pathlib import Path

# Must be set before torch touches MPS - grid_sampler_3d_backward isn't
# implemented on MPS yet (verified 2026-09-14 against torch 2.11.0); this
# falls back to CPU for just that op, rest of the graph stays on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Phase4PairDataset
from model import BilateralGridPredictor, HybridPredictor, grid_smoothness_loss

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "runs"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("WARNING: MPS not available, falling back to CPU (will be slow)")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds DataLoader(shuffle=True)'s default generator


def per_scene_l1(pred, target):
    """(B,3,H,W) -> (B,) mean absolute error per sample."""
    return (pred - target).abs().mean(dim=(1, 2, 3))


def run_epoch(model, loader, device, optimizer=None, low_res=256, max_batches=None,
              collect_baseline=False, grid_tv_weight=0.0, luma_mode="rec709"):
    training = optimizer is not None
    model.train(training)
    total_loss, total_baseline, count = 0.0, 0.0, 0
    regressed = 0
    worst_regression = 0.0
    worst_regression_name = None

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        full_res = batch["input"].to(device)
        target = batch["target"].to(device)

        pred, aux = model.predict_and_apply(full_res, low_res, luma_mode)

        per_scene = per_scene_l1(pred, target)
        recon_loss = per_scene.mean()
        loss = recon_loss
        if training and grid_tv_weight > 0 and "grid" in aux:
            loss = loss + grid_tv_weight * grid_smoothness_loss(aux["grid"])

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reported/compared loss is always the pure reconstruction L1, not
        # the TV-augmented training objective - so candidates with
        # different regularization stay comparable on the same metric.
        total_loss += recon_loss.item() * full_res.shape[0]
        count += full_res.shape[0]

        if collect_baseline:
            baseline_per_scene = per_scene_l1(full_res, target)
            total_baseline += baseline_per_scene.sum().item()
            diffs = (per_scene.detach() - baseline_per_scene)
            regressed += (diffs > 0).sum().item()
            batch_worst, batch_worst_idx = diffs.max(dim=0)
            if batch_worst.item() > worst_regression:
                worst_regression = batch_worst.item()
                worst_regression_name = batch["name"][batch_worst_idx.item()]

    result = {"l1": total_loss / max(count, 1), "n": count}
    if collect_baseline:
        result["baseline_l1"] = total_baseline / max(count, 1)
        result["regressed"] = regressed
        result["worst_regression"] = worst_regression
        result["worst_regression_name"] = worst_regression_name
    return result


def mem_snapshot(device):
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024) if hasattr(os, "uname") else None
    snap = {"peak_rss_mb": round(peak_rss_mb, 1) if peak_rss_mb else None}
    if device.type == "mps":
        snap["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)
        snap["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
    return snap


def build_model(args):
    if getattr(args, "use_hybrid", False):
        return HybridPredictor(
            grid_spatial=args.grid_spatial,
            grid_luma=args.grid_luma,
            low_res=args.low_res,
            base_ch=args.base_ch,
            refinement_base_ch=args.refinement_base_ch,
            refinement_max_delta=args.refinement_max_delta,
        )
    return BilateralGridPredictor(
        grid_spatial=args.grid_spatial,
        grid_luma=args.grid_luma,
        low_res=args.low_res,
        base_ch=args.base_ch,
        use_detail_branch=args.use_detail_branch,
        detail_ch=args.detail_ch,
    )


def train_one_config(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[{args.run_name}] device: {device}  seed: {args.seed}")

    train_ds = Phase4PairDataset("train")
    monitor_ds = Phase4PairDataset("monitor")
    if args.limit_train:
        train_ds.scenes = train_ds.scenes[: args.limit_train]
    if args.limit_monitor:
        monitor_ds.scenes = monitor_ds.scenes[: args.limit_monitor]
    print(f"[{args.run_name}] train scenes: {len(train_ds)}  monitor scenes: {len(monitor_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    monitor_loader = DataLoader(monitor_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    history = []
    run_t0 = time.time()
    best_monitor_l1 = float("inf")
    best_epoch = None
    best_state = None
    epochs_since_improvement = 0
    stopped_early = False

    for epoch in range(args.epochs):
        t0 = time.time()
        train_stats = run_epoch(
            model, train_loader, device, optimizer, args.low_res, args.max_batches,
            collect_baseline=(epoch == args.epochs - 1), grid_tv_weight=args.grid_tv_weight,
            luma_mode=args.luma_mode,
        )
        monitor_stats = run_epoch(model, monitor_loader, device, None, args.low_res, args.max_batches,
                                   collect_baseline=True, luma_mode=args.luma_mode)
        if scheduler is not None:
            scheduler.step()
        epoch_time = time.time() - t0
        mem = mem_snapshot(device)
        current_lr = optimizer.param_groups[0]["lr"]

        if monitor_stats["l1"] < best_monitor_l1:
            best_monitor_l1 = monitor_stats["l1"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        row = {
            "epoch": epoch + 1,
            "train_l1": train_stats["l1"],
            "monitor_l1": monitor_stats["l1"],
            "monitor_baseline_l1": monitor_stats["baseline_l1"],
            "monitor_regressed": monitor_stats["regressed"],
            "monitor_worst_regression": monitor_stats["worst_regression"],
            "monitor_worst_regression_name": monitor_stats["worst_regression_name"],
            "monitor_n": monitor_stats["n"],
            "lr": current_lr,
            "epoch_seconds": round(epoch_time, 1),
            **mem,
        }
        history.append(row)
        print(
            f"[{args.run_name}] epoch {epoch+1}/{args.epochs}  "
            f"train_l1={train_stats['l1']:.5f}  monitor_l1={monitor_stats['l1']:.5f}  "
            f"(baseline {monitor_stats['baseline_l1']:.5f}, regressed {monitor_stats['regressed']}/{monitor_stats['n']}, "
            f"worst_regression={monitor_stats['worst_regression']:.5f})  lr={current_lr:.2e}  "
            f"{epoch_time:.1f}s  mem={mem}"
        )

        if args.early_stopping_patience and epochs_since_improvement >= args.early_stopping_patience:
            print(f"[{args.run_name}] early stop at epoch {epoch+1} (no monitor improvement for {args.early_stopping_patience} epochs)")
            stopped_early = True
            break

    final_train_stats = train_stats
    final_monitor_stats = monitor_stats
    total_time = time.time() - run_t0

    # Selection is by best MONITOR epoch, not the last one - with a modest
    # epoch budget the final epoch isn't necessarily the best generalizing
    # one. Both checkpoints are kept so the comparison is honest.
    final_ckpt_path = CHECKPOINT_DIR / f"{args.run_name}-final.pt"
    best_ckpt_path = CHECKPOINT_DIR / f"{args.run_name}-best.pt"
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "epoch": history[-1]["epoch"]}, final_ckpt_path)
    torch.save({"state_dict": best_state, "config": vars(args), "epoch": best_epoch}, best_ckpt_path)

    summary = {
        "run_name": args.run_name,
        "config": {
            "grid_spatial": args.grid_spatial,
            "grid_luma": args.grid_luma,
            "base_ch": args.base_ch,
            "use_detail_branch": args.use_detail_branch,
            "detail_ch": args.detail_ch,
            "use_hybrid": getattr(args, "use_hybrid", False),
            "refinement_base_ch": args.refinement_base_ch,
            "refinement_max_delta": args.refinement_max_delta,
            "low_res": args.low_res,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "early_stopping_patience": args.early_stopping_patience,
            "grid_tv_weight": args.grid_tv_weight,
            "luma_mode": args.luma_mode,
        },
        "n_params": n_params,
        "train_scenes": len(train_ds),
        "monitor_scenes": len(monitor_ds),
        "final_train_l1": final_train_stats["l1"],
        "final_monitor_l1": final_monitor_stats["l1"],
        "final_monitor_baseline_l1": final_monitor_stats["baseline_l1"],
        "final_monitor_regressed": final_monitor_stats["regressed"],
        "final_monitor_n": final_monitor_stats["n"],
        "best_epoch": best_epoch,
        "best_monitor_l1": best_monitor_l1,
        "epochs_run": history[-1]["epoch"],
        "stopped_early": stopped_early,
        "total_seconds": round(total_time, 1),
        "history": history,
        "final_checkpoint": str(final_ckpt_path),
        "best_checkpoint": str(best_ckpt_path),
    }
    log_path = LOG_DIR / f"{args.run_name}.json"
    log_path.write_text(json.dumps(summary, indent=2))
    print(f"[{args.run_name}] best monitor epoch {best_epoch}: monitor_l1={best_monitor_l1:.5f}")
    print(f"[{args.run_name}] saved checkpoints: {final_ckpt_path.name}, {best_ckpt_path.name}")
    print(f"[{args.run_name}] saved run log: {log_path}")
    return summary


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--batch-size", type=int, default=1,
        help=(
            "images per batch. Default 1 because scenes are NOT all the same "
            "resolution (.fjlrg export fixes the long edge at 1536px, but the "
            "short edge varies a few px by scene aspect ratio - confirmed "
            "2026-09-14, e.g. 1026x1536 / 1536x1026 / 1023x1536 all appear in "
            "this corpus), and PyTorch's default collate can't stack mismatched "
            "sizes. A batch>1 run WILL eventually hit a shuffle that mixes "
            "sizes and crash - batching would need a custom collate_fn (pad or "
            "resize-to-common-size) to be safe, not implemented. Measured: "
            "batch 4 vs 8 gave no real speedup anyway (I/O and the MPS "
            "grid_sample CPU-fallback dominate, not GPU parallelism), so 1 "
            "isn't a real performance cost here."
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    ap.add_argument("--early-stopping-patience", type=int, default=None,
                     help="stop if monitor L1 hasn't improved for N epochs (None = disabled, always run --epochs)")
    ap.add_argument("--grid-tv-weight", type=float, default=0.0,
                     help="weight on grid_smoothness_loss added to the training objective (0 = off)")
    ap.add_argument("--luma-mode", choices=["rec709", "green_channel"], default="rec709",
                     help="coordinate used to index the bilateral grid's luma axis - see model.compute_luma_guide. "
                          "'green_channel' is the WB-stable guide (2026-09-15 targeted intervention); the as-shot-WB "
                          "color image is always both the model's input and output either way, only this changes.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low-res", type=int, default=256)
    ap.add_argument("--grid-spatial", type=int, default=4)
    ap.add_argument("--grid-luma", type=int, default=7)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--use-detail-branch", action="store_true",
                     help="2026-09-16 targeted intervention: shallow high-res [luma, local-highpass] branch "
                          "fused into the grid-coefficient head, preserving compact-highlight shape that the "
                          "main encoder's 5 stride-2 layers otherwise discard. Grid/slicing unchanged either way.")
    ap.add_argument("--detail-ch", type=int, default=8)
    ap.add_argument("--use-hybrid", action="store_true",
                     help="2026-09-17: bilateral grid (unchanged) + RefinementNet, a small multiscale "
                          "full-resolution residual-correction net with a bounded, identity-safe output "
                          "(zero-init last layer, tanh-clamped delta) - see model.HybridPredictor. Trained "
                          "jointly end-to-end with the grid encoder. Genuinely different from --use-detail-branch, "
                          "which only changes what the grid *encoder* sees, not the correction mechanism itself.")
    ap.add_argument("--refinement-base-ch", type=int, default=8)
    ap.add_argument("--refinement-max-delta", type=float, default=0.08,
                     help="hard clamp on the refinement net's per-pixel correction magnitude (linear light, 0..1 domain)")
    ap.add_argument("--limit-train", type=int, default=None, help="cap train scenes (smoke-testing)")
    ap.add_argument("--limit-monitor", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches/epoch (smoke-testing)")
    ap.add_argument("--run-name", default="phase4-bilateral-grid")
    return ap


if __name__ == "__main__":
    train_one_config(build_arg_parser().parse_args())
