"""Strict train-only data contract for matched legacy/G gamma experiments.

This is research infrastructure. It does not replace either codec, change the
app, load R11 weights, or permit protected evaluation tiers.
"""
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from fjlrg import load_fjlrg_linear
import fjlrg_v2 as V
from model import HybridPredictor

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "training"
DIAG, CKPT, RUNS = TRAIN / "diagnostics", TRAIN / "checkpoints", TRAIN / "runs"
MANIFEST = DIAG / "matched-gamma-training-manifest-v2.json"
POOL = DIAG / "provia-color-run-manifest.json"
POOL_SHA = "d6c04efd70df63aa38a9f2f08e0b291f858d5085a9670ffb7d3f1b3f4c6cf254"
R11_PATH = CKPT / "r11-drcond-seed42-best.pt"
R11_SHA = "1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c"
ARMS = ("legacy", "linear")
SEEDS = (42, 43, 44)


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    with path.open("x") as f:
        f.write(payload)


def committed(path):
    path = Path(path)
    content = subprocess.check_output(["git", "show", "HEAD:" + str(path.relative_to(ROOT))], cwd=ROOT)
    return hashlib.sha256(content).hexdigest() == sha(path)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def state_digest(state):
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def fresh_model(config, seed):
    need(type(seed) is int and seed in SEEDS, "Unknown paired seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return HybridPredictor(
        grid_spatial=config["grid_spatial"], grid_luma=config["grid_luma"],
        low_res=config["low_res"], base_ch=config["base_ch"],
        refinement_base_ch=config["refinement_base_ch"],
        refinement_max_delta=config["refinement_max_delta"], condition_channels=1,
    )


def initialise(m, seed, target_device):
    model = fresh_model(m["model_config"], seed)
    need(state_digest(model.state_dict()) == m["initial_state_sha256"][str(seed)],
         "Initial state differs from frozen paired initialisation")
    return model.to(target_device)


def forward(model, m, x, cond, target_device):
    config = m["model_config"]
    result, _ = model.predict_and_apply(x.to(target_device), config["low_res"],
                                        config["luma_mode"], condition=cond.to(target_device))
    return result


def make_orders(names, seed, epochs=20):
    generator = torch.Generator().manual_seed(seed)
    return [[names[i] for i in torch.randperm(len(names), generator=generator).tolist()]
            for _ in range(epochs)]


def validate_manifest(m):
    need(m["pool_sha256"] == POOL_SHA and sha(POOL) == POOL_SHA, "Train allowlist changed")
    pool = json.loads(POOL.read_text())["scenes"]
    rows = m["scenes"]
    index = {r["name"]: r for r in rows}
    need(len(rows) == len(index) == len({r["shoot"] for r in rows}) == 583, "Expected 583 unique train identities")
    need(set(index) == {r["name"] for r in pool}, "Scene names differ from train allowlist")
    for old in pool:
        row = index[old["name"]]
        for key in ("name", "shoot", "role", "session", "captureDynamicRange", "fjlrg_wh", "npy_sha256", "tiff_sha256", "fjlrg_sha256"):
            need(row[key] == old[key], "Frozen scene metadata differs: " + key)
        need(row["role"] in ("fit", "development"), "Protected or unknown role")
        need(row["captureDynamicRange"] in (100, 200, 400), "Unknown capture DR")
        need(set(row["inputs"]) == set(ARMS), "Unknown input arm")
        need(row["inputs"]["legacy"]["path"] == f"calibration-input/{row['shoot']}/browser-phase3-linear.fjlrg"
             and row["inputs"]["legacy"]["sha256"] == old["fjlrg_sha256"], "Legacy path/hash mismatch")
        need(row["target_path"] == f"calibration-input/provia-color-target-cache/{row['shoot']}.npy", "Target path escaped train cache")
        need(row["inputs"]["linear"]["path"] == f"training/checkpoints/provia-input-v2-train-g/{row['shoot']}/input-v2-linear-asshot.fjlrg",
             "Linear path escaped verified G exports")
    fit = [r["name"] for r in rows if r["role"] == "fit"]
    dev = [r for r in rows if r["role"] == "development"]
    need(len(fit) == 467 and len(dev) == 116, "Fit/development counts changed")
    need(not ({r["session"] for r in rows if r["role"] == "fit"} & {r["session"] for r in dev}), "Session leakage")
    need(m["training"] == {"seeds": list(SEEDS), "epochs": 20, "lr": 0.001, "eta_min": 1e-5, "batch_size": 1},
         "Frozen optimisation budget changed")
    need(m["model_config"] == {"grid_spatial": 8, "grid_luma": 9, "base_ch": 16,
                               "refinement_base_ch": 8, "refinement_max_delta": 0.08,
                               "low_res": 256, "luma_mode": "rec709", "condition_channels": 1},
         "Frozen architecture changed")
    need(set(m["orders"]) == set(m["initial_state_sha256"]) == {str(s) for s in SEEDS}, "Paired seed coverage changed")
    for seed in SEEDS:
        need(m["orders"][str(seed)] == make_orders(fit, seed), "Frozen sample orders do not reproduce")


def load_manifest(expected_sha):
    need(isinstance(expected_sha, str) and len(expected_sha) == 64 and sha(MANIFEST) == expected_sha,
         "Missing, unpinned or changed training manifest")
    need(committed(MANIFEST), "Training manifest must be committed before any run")
    m = json.loads(MANIFEST.read_text())
    for name, expected in m["bindings"].items():
        path = ROOT / name
        need(sha(path) == expected, "Bound file changed: " + name)
        if name.startswith("training/") and (name.endswith(".py") or name.endswith(".json")):
            need(committed(path), "Bound research code/evidence must be committed: " + name)
    need(sha(R11_PATH) == R11_SHA, "Selected R11 checkpoint changed")
    need(m["environment"]["torch"] == torch.__version__
         and m["environment"]["numpy"] == np.__version__
         and m["environment"]["mps_available"] == torch.backends.mps.is_available(),
         "Frozen training environment changed")
    validate_manifest(m)
    m["_manifest_sha256"] = expected_sha
    return m


class Pairs:
    """Only manifest-allowlisted pairs; caller may further restrict to fit/dev."""

    def __init__(self, m, arm, role=None):
        need(arm in ARMS, "Unknown input arm")
        need(role in (None, "fit", "development"), "Protected or unknown role refused")
        self.arm, self.m = arm, m
        self.by = {r["name"]: r for r in m["scenes"] if role is None or r["role"] == role}

    def get(self, name):
        need(name in self.by, "Non-allowlisted scene refused: " + str(name))
        row = self.by[name]
        spec = row["inputs"][self.arm]
        path, target = ROOT / spec["path"], ROOT / row["target_path"]
        need(sha(path) == spec["sha256"], "Input hash mismatch: " + name)
        need(sha(target) == row["npy_sha256"], "Target cache hash mismatch: " + name)
        if self.arm == "legacy":
            x, width, height = load_fjlrg_linear(path)
        else:
            need(sha(V.sidecar_path(path)) == spec["sidecar_sha256"], "Sidecar hash mismatch: " + name)
            frame = V.load_fjlrg_v2(path, expected_contract=V.CONTRACT_ASSHOT,
                                    expected_source_sha256=spec["source_raf_sha256"],
                                    expected_wasm_sha256=spec["wasm_sha256"])
            x, width, height = frame.to_unit_float32(), frame.width, frame.height
            need(frame.provenance["capture_dynamic_range"] == row["captureDynamicRange"], "DR provenance mismatch")
        y = np.load(target, allow_pickle=False)
        need([width, height] == row["fjlrg_wh"] and x.shape == y.shape == (height, width, 3), "Pair geometry mismatch")
        need(x.dtype == y.dtype == np.float32 and np.isfinite(x).all() and np.isfinite(y).all(), "Non-finite or wrong-dtype pair")
        need(0 <= x.min() <= x.max() <= 1, "Input outside unit interval: " + name)
        # Keep audited targets bit-identical to the original training pipeline.
        # Float32 area resizing can put nominal white a few ULP above 1. This
        # is not an input-codec violation; do not clamp/rewrite those targets.
        tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).unsqueeze(0)
        condition = torch.tensor([np.log2(row["captureDynamicRange"] / 100) / 2], dtype=torch.float32)
        return tensor(x), tensor(y), condition
