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

3. **ONNX export** — Round 11's reproducible export and browser parity
   gate is documented below. This is inference/export only.

## Round 11 ONNX gate (2026-11-14)

Selected Round 11 seed-42/43 checkpoints export as ONNX opset 20 graphs.
The graph takes `linear_rgb` (float32 `[1,3,H,W]`, dynamic height/width,
linear sRGB primaries and as-shot WB) and `capture_dr_condition`
(float32 `[1,1]`, `log2(captureDynamicRange/100)/2`: DR100=0, DR200=0.5,
DR400=1). The fourth encoder plane is constructed inside the graph;
full-resolution RGB is not exposure-scaled or concatenated with a plane
on the host. Reject missing/ambiguous capture DR instead of defaulting.
The output is unclamped linear RGB, matching training. Target output DR
remains 100, separate from the capture DR input.

Reproduce with the existing local training dependencies plus:

```sh
python3 -m pip install -r training/requirements-onnx.txt
python3 training/export_round11_onnx.py
npm install --prefix training/checkpoints/onnx-web-deps --no-audit --no-fund --save-exact onnxruntime-web@1.31.1
node scripts/verify-round11-onnx-web.mjs --ort-dir training/checkpoints/onnx-web-deps/node_modules/onnxruntime-web/dist
```

The Python script compares native ONNX Runtime against the original
trained PyTorch model and writes test vectors and reports under the
gitignored `training/checkpoints/round11-onnx/`. It only resolves real
examples against train/monitor: DSCF1419 (capture DR100, train observation
only), DSCF0878 (DR200), DSCF0538 (DR400), at full working resolution,
plus a dynamic 73x111 RGB input tested at all three DR values. The Node
script serves only those allowlisted files on loopback, tests both seeds
in Chromium WASM and WebGPU, records GPU-adapter identity and numerical
error, and closes the browser/server. Chromium on macOS uses Metal;
the recorded adapter was Apple, metal-3. No model training or held-out
scene processing occurs. Timings include output readback/comparison and
are not a production performance benchmark.

Two export/runtime fixes are essential:

- Export a copy of GroupNorm using centered variance with separate
  width/height/channel reductions. The default flattened
  InstanceNormalization path lost WASM precision on full images.
  Checkpoint weights/training modules are unchanged, and the export is
  compared against the original model, not against the rewritten copy.
- Set WebGPU `preferredLayout: "NCHW"` explicitly. Runtime 1.31.1's
  default layout produced substantial refinement-branch drift; NCHW
  restores agreement. Resizing, DR conditioning, and grid slicing
  already agreed before either fix. Do not omit this session option.

`--diagnostic-stages` on the Python exporter (use a separate
`--output-dir`) includes resize/grid/refinement outputs for isolating
backend errors. The browser validator automatically measures these when
present. `--webgpu-layout NHWC` is diagnostic only, not the approved
runtime configuration.

Both seeds passed 12 native comparisons and 24 browser comparisons in
total. Maximum browser absolute error was 2.51e-6 and maximum mean
absolute error 9.80e-8 versus original PyTorch. The predeclared limits
were max 5e-4 / mean 5e-5 for browsers and max 1e-4 / mean 1e-5 for
native ONNX. Versioned models and evidence are in `exports/round11/`.

This validates numerical conversion at the existing 1536px working
resolution, not DR100 generalization, full 40MP inference, all recipes,
or application integration. Independent DR100 evaluation and the locked
final acceptance test remain outstanding.

## Status (2026-09-14)

Pipeline built and validated end-to-end (data loading, MPS forward pass,
CPU-fallback backward, optimizer step, checkpoint save) against 4 real
train + 2 real monitor scenes — loss drops cleanly on that tiny overfit
check (0.287 → 0.187 → 0.107 over 3 epochs). **This is a smoke test, not a
trained model.** Full-corpus target precompute (637 scenes, ~50min) and
real training/architecture iteration haven't happened yet.

## Current research: neutral-WB input study (2026-11-25)

Earlier dated status sections are historical. R11 seed42 remains selected.
The new isolated study compares fresh G/as-shot against source-only neutral
WB inputs on455FIT/116internal-development scenes (12unsupported FIT
presets excluded from both arms). The full455FIT input qualification and
41-row development canary check passed. Full development qualification is
still running; this is not a model result.

`neutral_wb_study_data.py` supports the strict G/W `.fjlrg` and P `.fjnrgb`
contracts, with identical audited target arrays and capture-DR conditions.
It defaults to FIT-only access, preserves target float32 epsilon values,
and refuses changed hashes, unsupported kinds, protected roles and unknown
names. Both complete/published input receipts are required before preparing
`diagnostics/neutral-wb-study-manifest.json`.

The operator sequence is prepare -> commit manifest ->571-pair preflight
-> disposable MPS/CPU-resume smoke -> commit both proofs -> six fixed
20-epoch runs -> independent report/canary verification. Entrypoints:

```text
python3 training/prepare_neutral_wb_study.py prepare
python3 training/prepare_neutral_wb_study.py preflight --manifest-sha <printed SHA>
python3 training/run_neutral_wb_study.py smoke --manifest-sha <same SHA>
python3 training/run_neutral_wb_study.py all --manifest-sha <same SHA>
python3 training/verify_neutral_wb_study.py --manifest-sha <same SHA>
```

Three paired seeds42/43/44 use fresh identical initialization/orders, the
existing R11 architecture/rec709 guide/linear-L1 optimizer recipe and old
geometry, without R11 weights or epoch selection. Every seed has9100
updates per arm. Five internal screen criteria are fixed in
`neutral_wb_study_data.SCREEN`; results never automatically select or ship
a model. Development treated79 are allDR400, whereas identity37 comprise
25DR200/12DR100: this confound is reported, and DR100 has no FIT support.
K/monitor/finalHeldOut/T8/dr100Eval remain inaccessible.

The one-shot continuation handles this sequence locally, publishes only
its exact evidence paths, and stops on any integrity/process failure:

```text
python3 -u training/continue_neutral_wb_study.py --run
```

It must itself be committed before launch. Logs and process state live in
`training/coordination/.local/neutral-wb-study-pipeline.log` and
`neutral-wb-study-pipeline-state.json`. Input qualification completion is
not training start: production starts only at the first `TRAINING_STARTED`
optimizer marker. Completion requests a real read-only Claude CLI review;
quota failure is recorded as pending. No automatic app/model selection.

### Neutral-WB study result (2026-11-26)

All six runs and independent verification completed (41303a1). The frozen
internal screen failed4/5: pooled development colour error improved
7.845→6.343DeltaE00, but seed42's identity37group worsened+1.106 against
the+0.10limit. DR100has no FIT support; its12development cases are all
session14and flip direction across seeds. No model was selected. R11/app
are unchanged, and protected sets remain closed.

`diagnose_neutral_wb_dr100_spread.py` is a post-hoc scalar-JSON-only
diagnostic, not inference or a new screen. It binds the completed evidence,
reproduces identity means from DR-group contributions, and records
observed seed ranges without treating three seeds as a noise distribution.
The report is `diagnostics/neutral-wb-dr100-seed-spread.json`.

### Follow-up condition-response diagnostic (2026-11-26)

New scope only:37identity internal-development sources, all six frozen
endpoints, CPU conditions[0,0.5,1]. The runner must first reproduce all222
actual-condition cases, then runs444counterfactuals. It emits scalar cells;
the separate verifier rebuilds group arithmetic and checks30CPU canaries.
All17new synthetic tests pass. Code/scope must be committed before
execution. No training, app change, alternative condition selection or
protected-tier access. Original study remains FAIL.

```text
python3 training/probe_neutral_wb_condition.py prepare
python3 training/probe_neutral_wb_condition.py run --scope-sha <committed scope SHA>
python3 training/verify_neutral_wb_condition_probe.py --scope-sha <same SHA>
```

Actual Claude drafted the plan. Codex corrected interpolation/capture-DR
wording, implemented runner/verifier after the Claude CLI usage-limit
stop, and executed the synthetic checks. Claude has not reviewed that
implementation yet; that will be reported separately rather than assumed.
