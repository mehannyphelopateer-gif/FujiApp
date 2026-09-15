"""Phase 4 training loop - local only, Apple MPS backend.
docs/phase4-scene-adaptive-scope.md SS1/SS4/SS5 + the 2026-09-14 compute
decision (local PyTorch/MPS, no cloud spend).

Trains on the 'train' tier only, reports monitor-tier metrics each epoch
for model-selection visibility. This is the fitting loop itself, run once
per candidate architecture/hyperparameter combination - see
run_model_selection.py for the actual multi-candidate comparison.

Baseline: per §7's real acceptance criterion ("every scene at least as
good as no correction at all"), every run also reports the no-correction
L1 (input vs target, untouched) so train/monitor numbers are meaningful
against that bar, not just as raw numbers - plus a per-scene "regressed"
count (model worse than no-correction on that scene). This is a fast
proxy for §7's real gate, not the gate itself - the real one runs
leave-one-session-out and, at the end, once against finalHeldOut.

NEVER imports or references the finalHeldOut tier - dataset.list_scenes
refuses that tier outright, so there is nothing to accidentally wire in
here.
"""
import argparse
import json
import os
import resource
import time
from pathlib import Path

# Must be set before torch touches MPS - grid_sampler_3d_backward isn't
# implemented on MPS yet (verified 2026-09-14 against torch 2.11.0); this
# falls back to CPU for just that op, rest of the graph stays on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader

from dataset import Phase4PairDataset
from model import BilateralGridPredictor, apply_bilateral_grid, resize_for_network

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "runs"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("WARNING: MPS not available, falling back to CPU (will be slow)")
    return torch.device("cpu")


def per_scene_l1(pred, target):
    """(B,3,H,W) -> (B,) mean absolute error per sample."""
    return (pred - target).abs().mean(dim=(1, 2, 3))


def run_epoch(model, loader, device, optimizer=None, low_res=256, max_batches=None, collect_baseline=False):
    training = optimizer is not None
    model.train(training)
    total_loss, total_baseline, count = 0.0, 0.0, 0
    regressed = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        full_res = batch["input"].to(device)
        target = batch["target"].to(device)

        low_res_in = resize_for_network(full_res, low_res)
        grid = model(low_res_in)
        pred = apply_bilateral_grid(grid, full_res)

        per_scene = per_scene_l1(pred, target)
        loss = per_scene.mean()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * full_res.shape[0]
        count += full_res.shape[0]

        if collect_baseline:
            baseline_per_scene = per_scene_l1(full_res, target)
            total_baseline += baseline_per_scene.sum().item()
            regressed += (per_scene.detach() > baseline_per_scene).sum().item()

    result = {"l1": total_loss / max(count, 1), "n": count}
    if collect_baseline:
        result["baseline_l1"] = total_baseline / max(count, 1)
        result["regressed"] = regressed
    return result


def mem_snapshot(device):
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024) if hasattr(os, "uname") else None
    snap = {"peak_rss_mb": round(peak_rss_mb, 1) if peak_rss_mb else None}
    if device.type == "mps":
        snap["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)
        snap["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
    return snap


def build_model(args):
    return BilateralGridPredictor(
        grid_spatial=args.grid_spatial,
        grid_luma=args.grid_luma,
        low_res=args.low_res,
        base_ch=args.base_ch,
    )


def train_one_config(args):
    device = get_device()
    print(f"[{args.run_name}] device: {device}")

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

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    history = []
    run_t0 = time.time()
    best_monitor_l1 = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(args.epochs):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, device, optimizer, args.low_res, args.max_batches, collect_baseline=(epoch == args.epochs - 1))
        monitor_stats = run_epoch(model, monitor_loader, device, None, args.low_res, args.max_batches, collect_baseline=True)
        epoch_time = time.time() - t0
        mem = mem_snapshot(device)

        if monitor_stats["l1"] < best_monitor_l1:
            best_monitor_l1 = monitor_stats["l1"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        row = {
            "epoch": epoch + 1,
            "train_l1": train_stats["l1"],
            "monitor_l1": monitor_stats["l1"],
            "monitor_baseline_l1": monitor_stats["baseline_l1"],
            "monitor_regressed": monitor_stats["regressed"],
            "monitor_n": monitor_stats["n"],
            "epoch_seconds": round(epoch_time, 1),
            **mem,
        }
        history.append(row)
        print(
            f"[{args.run_name}] epoch {epoch+1}/{args.epochs}  "
            f"train_l1={train_stats['l1']:.5f}  monitor_l1={monitor_stats['l1']:.5f}  "
            f"(baseline {monitor_stats['baseline_l1']:.5f}, regressed {monitor_stats['regressed']}/{monitor_stats['n']})  "
            f"{epoch_time:.1f}s  mem={mem}"
        )

    final_train_stats = train_stats
    final_monitor_stats = monitor_stats
    total_time = time.time() - run_t0

    # Selection is by best MONITOR epoch, not the last one - with a modest
    # epoch budget the final epoch isn't necessarily the best generalizing
    # one. Both checkpoints are kept so the comparison is honest.
    final_ckpt_path = CHECKPOINT_DIR / f"{args.run_name}-final.pt"
    best_ckpt_path = CHECKPOINT_DIR / f"{args.run_name}-best.pt"
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "epoch": args.epochs}, final_ckpt_path)
    torch.save({"state_dict": best_state, "config": vars(args), "epoch": best_epoch}, best_ckpt_path)

    summary = {
        "run_name": args.run_name,
        "config": {
            "grid_spatial": args.grid_spatial,
            "grid_luma": args.grid_luma,
            "base_ch": args.base_ch,
            "low_res": args.low_res,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
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
    ap.add_argument("--low-res", type=int, default=256)
    ap.add_argument("--grid-spatial", type=int, default=4)
    ap.add_argument("--grid-luma", type=int, default=7)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--limit-train", type=int, default=None, help="cap train scenes (smoke-testing)")
    ap.add_argument("--limit-monitor", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches/epoch (smoke-testing)")
    ap.add_argument("--run-name", default="phase4-bilateral-grid")
    return ap


if __name__ == "__main__":
    train_one_config(build_arg_parser().parse_args())
