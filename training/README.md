# Phase 4 training

Local-only PyTorch pipeline for the scene-adaptive bilateral-grid model —
see `docs/phase4-scene-adaptive-scope.md` §§1, 4, 5 for the architecture
rationale and the 2026-09-14 "local PyTorch/MPS, no cloud spend" decision
this follows.

Trains and evaluates against `train`/`monitor` only. `dataset.list_scenes`
refuses the string `"finalHeldOut"` outright — there is no code path in
this directory that can load that tier. Do not add one before train/monitor
model selection is frozen.

## Setup

```
pip3 install -r training/requirements.txt
```

Already verified against this machine (Apple M1 Pro, 16 GB, torch 2.11.0,
2026-09-14).

## Two MPS gaps found and worked around

1. **`grid_sampler_3d_backward` isn't implemented on MPS.** The bilateral-
   grid slicing op (`model.apply_bilateral_grid`) uses 5D `grid_sample` —
   forward pass runs fine on MPS, but the backward pass doesn't exist yet
   for that device. `train.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before
   touching MPS, which falls back to CPU for just this one op's backward;
   the rest of the graph stays on MPS. Measured cost at full working
   resolution (~1536px): under half a second per step, not a bottleneck.
2. **`F.interpolate(mode="area")` requires evenly-divisible sizes on MPS**
   (delegates to `adaptive_avg_pool2d`, which has that restriction on this
   backend) — this project's images (e.g. 1536×1026 → 256×256) aren't
   evenly divisible. `model.resize_for_network` uses antialiased bilinear
   instead, which is unrestricted and MPS-native.

Both are noted here so a future PyTorch upgrade that closes either gap is a
reason to revisit, not a mystery to re-debug.

## Pipeline

1. **`precompute_targets.py`** — decodes each train/monitor scene's
   `xraw-phase3.tiff` (full sensor resolution, 16-bit) once, converts
   sRGB→linear, and area-averages it down to match that scene's
   `browser-phase3-linear.fjlrg` resolution, caching the result to
   `calibration-input/phase4-target-cache/<Shoot N>.npy`. Slow per scene
   (~5s; matches the "9-60+ min for 230-270 scenes" precedent in the scope
   doc for the full corpus) — run once, not per training run.

   ```
   python3 training/precompute_targets.py            # all train+monitor
   python3 training/precompute_targets.py --tier train --limit 20   # smoke-test a subset
   ```

2. **`train.py`** — the fitting loop itself, meant to be run repeatedly
   across architecture/hyperparameter attempts.

   ```
   python3 training/train.py --epochs 20 --batch-size 4
   ```

   Reports monitor-tier L1 loss each epoch for visibility only — actual
   model *selection* against monitor, and the leave-one-session-out
   zero-regression gate (scope doc §7) against `finalHeldOut`, are
   separate steps this script doesn't do.

3. **ONNX export** — not built yet. Write this once an actual checkpoint
   is worth freezing, not before.

## Status (2026-09-14)

Pipeline built and validated end-to-end (data loading, MPS forward pass,
CPU-fallback backward, optimizer step, checkpoint save) against 4 real
train + 2 real monitor scenes — loss drops cleanly on that tiny overfit
check (0.287 → 0.187 → 0.107 over 3 epochs). **This is a smoke test, not a
trained model.** Full-corpus target precompute (637 scenes, ~50min) and
real training/architecture iteration haven't happened yet.
