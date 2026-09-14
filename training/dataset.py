"""Phase 4 train/monitor dataset - reads calibration-input/phase4-corpus-manifest.json
and yields (input_linear, target_linear) pairs, both already at the .fjlrg's
resolution. Target TIFFs are slow to decode (7728x5152 16-bit) - see
precompute_targets.py, which caches each one down to a .npy at the matching
resolution once so training iterations don't re-decode the full TIFF.

Hard rule, not a suggestion: this module must never load a finalHeldOut
scene. See docs/phase4-scene-adaptive-scope.md SS7/SS8.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fjlrg import load_fjlrg_linear
from target_tiff import load_xraw_target_linear_matching

REPO_ROOT = Path(__file__).resolve().parent.parent
CAL_DIR = REPO_ROOT / "calibration-input"
MANIFEST_PATH = CAL_DIR / "phase4-corpus-manifest.json"
TARGET_CACHE_DIR = CAL_DIR / "phase4-target-cache"


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def list_scenes(tier):
    """tier: 'train' or 'monitor' ONLY. Raises if asked for finalHeldOut -
    that tier is not accessible through this loader at all."""
    if tier not in ("train", "monitor"):
        raise ValueError(
            f"list_scenes({tier!r}) refused - this dataset only ever loads "
            f"'train' or 'monitor'. finalHeldOut scenes must not be accessed "
            f"until train/monitor model selection is frozen."
        )
    manifest = load_manifest()
    scenes = manifest["tiers"][tier]["scenes"]
    ready = [s for s in scenes if s.get("pairReady")]
    if len(ready) != len(scenes):
        missing = len(scenes) - len(ready)
        print(f"WARNING: {missing} {tier} scenes are not pairReady yet, skipping them")
    return ready


def cached_target_path(shoot):
    return TARGET_CACHE_DIR / f"{shoot}.npy"


class Phase4PairDataset(Dataset):
    """tier: 'train' or 'monitor'. require_cached_target: if True (default),
    raises on a scene whose target hasn't been precomputed yet, rather than
    silently falling back to a slow full-TIFF decode mid-training - run
    precompute_targets.py first."""

    def __init__(self, tier, require_cached_target=True):
        self.tier = tier
        self.scenes = list_scenes(tier)
        self.require_cached_target = require_cached_target

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        shoot = scene["shoot"]
        shoot_dir = CAL_DIR / shoot

        input_linear, w, h = load_fjlrg_linear(shoot_dir / "browser-phase3-linear.fjlrg")

        cache_path = cached_target_path(shoot)
        if cache_path.exists():
            target_linear = np.load(cache_path)
        elif self.require_cached_target:
            raise FileNotFoundError(
                f"{shoot}: no cached target at {cache_path} - run "
                f"training/precompute_targets.py first"
            )
        else:
            target_linear = load_xraw_target_linear_matching(
                shoot_dir / "xraw-phase3.tiff", w, h
            )

        if target_linear.shape != input_linear.shape:
            raise ValueError(
                f"{shoot}: shape mismatch input={input_linear.shape} "
                f"target={target_linear.shape}"
            )

        # HxWx3 -> 3xHxW for PyTorch conv convention.
        input_t = torch.from_numpy(np.ascontiguousarray(input_linear.transpose(2, 0, 1)))
        target_t = torch.from_numpy(np.ascontiguousarray(target_linear.transpose(2, 0, 1)))
        return {"input": input_t, "target": target_t, "name": scene["name"], "shoot": shoot}
