"""Precompute + cache each train/monitor scene's target TIFF, downsampled to
match its .fjlrg resolution, as a .npy - the full 16-bit TIFF decode is slow
enough (Phase 3 precedent: 9-60+ min for 230-270 scenes) that doing it once
up front beats re-decoding every training epoch.

Only ever touches train/monitor scenes (via dataset.list_scenes, which
refuses 'finalHeldOut' outright) - never the held-out tier.

Usage: python3 precompute_targets.py [--tier train|monitor|both] [--limit N]
"""
import argparse
import time

from dataset import CAL_DIR, TARGET_CACHE_DIR, cached_target_path, list_scenes
from fjlrg import load_fjlrg_linear
from target_tiff import load_xraw_target_linear_matching
import numpy as np


def precompute(tier, limit=None):
    scenes = list_scenes(tier)
    if limit:
        scenes = scenes[:limit]
    TARGET_CACHE_DIR.mkdir(exist_ok=True)

    done, skipped, failed = 0, 0, []
    for i, scene in enumerate(scenes):
        shoot = scene["shoot"]
        out_path = cached_target_path(shoot)
        if out_path.exists():
            skipped += 1
            continue
        shoot_dir = CAL_DIR / shoot
        t0 = time.time()
        try:
            _, w, h = load_fjlrg_linear(shoot_dir / "browser-phase3-linear.fjlrg")
            target = load_xraw_target_linear_matching(shoot_dir / "xraw-phase3.tiff", w, h)
            np.save(out_path, target.astype(np.float32))
            done += 1
        except Exception as e:
            failed.append((scene["name"], str(e)))
            continue
        elapsed = time.time() - t0
        print(f"[{tier}] {i+1}/{len(scenes)} {shoot} ({scene['name']}) - {elapsed:.1f}s")

    print(f"\n{tier}: precomputed {done}, already cached {skipped}, failed {len(failed)}")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")
    return done, skipped, failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["train", "monitor", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="cap scenes per tier (for smoke-testing)")
    args = ap.parse_args()

    tiers = ["train", "monitor"] if args.tier == "both" else [args.tier]
    for t in tiers:
        precompute(t, limit=args.limit)
