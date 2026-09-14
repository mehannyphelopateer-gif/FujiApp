"""Phase 4 training loop - local only, Apple MPS backend.
docs/phase4-scene-adaptive-scope.md SS1/SS4/SS5 + the 2026-09-14 compute
decision (local PyTorch/MPS, no cloud spend).

Trains on the 'train' tier only. Reports monitor-tier loss periodically for
visibility, but model SELECTION against monitor and the eventual
leave-one-session-out zero-regression gate (SS7) are separate, later steps
- this script is the fitting loop itself, run repeatedly across
architecture/hyperparameter attempts.

NEVER imports or references the finalHeldOut tier - dataset.list_scenes
refuses that tier outright, so there is nothing to accidentally wire in
here.
"""
import argparse
import os
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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("WARNING: MPS not available, falling back to CPU (will be slow)")
    return torch.device("cpu")


def l1_loss(pred, target):
    return (pred - target).abs().mean()


def run_epoch(model, loader, device, optimizer=None, low_res=256, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss, count = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        full_res = batch["input"].to(device)
        target = batch["target"].to(device)

        low_res_in = resize_for_network(full_res, low_res)
        grid = model(low_res_in)
        pred = apply_bilateral_grid(grid, full_res)
        loss = l1_loss(pred, target)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--low-res", type=int, default=256)
    ap.add_argument("--limit-train", type=int, default=None, help="cap train scenes (smoke-testing)")
    ap.add_argument("--limit-monitor", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches/epoch (smoke-testing)")
    ap.add_argument("--checkpoint-name", default="phase4-bilateral-grid.pt")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")

    train_ds = Phase4PairDataset("train")
    monitor_ds = Phase4PairDataset("monitor")
    if args.limit_train:
        train_ds.scenes = train_ds.scenes[: args.limit_train]
    if args.limit_monitor:
        monitor_ds.scenes = monitor_ds.scenes[: args.limit_monitor]
    print(f"train scenes: {len(train_ds)}  monitor scenes: {len(monitor_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    monitor_loader = DataLoader(monitor_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BilateralGridPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer, args.low_res, args.max_batches)
        monitor_loss = run_epoch(model, monitor_loader, device, None, args.low_res, args.max_batches)
        print(
            f"epoch {epoch+1}/{args.epochs}  train_l1={train_loss:.5f}  "
            f"monitor_l1={monitor_loss:.5f}  ({time.time()-t0:.1f}s)"
        )

    ckpt_path = CHECKPOINT_DIR / args.checkpoint_name
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
