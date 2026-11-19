# Classic Negative Step-1 export handoff

Codex's independent export work, 2026-11-19. No training, app integration,
K evaluation, monitor or finalHeldOut image processing occurs in these tools.
Claude owns training/selection and the single E4-only K check. Numerical
export parity is separate from recipe/simulation accuracy acceptance.

## Checkpoint interface

The trainer's current checkpoint format works unchanged: `state_dict`,
Round-11-compatible `config`, and its `cuban_step1` metadata. Encoder weights
must remain identical to the original Round 11 seed-42 source. Native export
uses the existing GroupNorm conversion on an export copy, while expected
vectors come from the original PyTorch graph.

The export tests use only the fixed F fixtures DSCF0523, DSCF0964 and
DSCF0514, plus dynamic 73x111 and 111x73 synthetic inputs. All are evaluated
with capture-DR400 conditioning, output DR100. No target TIFFs are loaded.

## One-shot background export stage

```sh
python3 -u training/run_codex_cn_export_gate.py \
  --control-dir training/checkpoints/codex-cn-overnight-20261119 \
  --wait-hours 12
```

This finite dependency stage waits for
`training/diagnostics/cuban-step1-freeze.json`, at most 12 hours. It does not
start or alter training. On arrival it checks the frozen run/folds/contract,
all final checkpoint hashes, finite logged losses/selection, F24-only
adaptation metadata and bit-identical frozen encoder weights. It protects
the original Round 11 checkpoint and original app ONNX hashes.

It then generates its own export provenance and runs native ONNX, WASM and
WebGPU parity once for seed 42. It never evaluates K or interprets a K pass.
Even a numerical parity pass has `accuracy_acceptance=false` and does not
authorize app integration. Failed integrity/numerical checks stop the stage;
there is no automatic threshold change, seed swap or further tuning.

Its own gitignored control folder holds:

- `worker.log`: stage/export/runtime messages.
- `status.json`: waiting, exporting, browser checking, stopped, timeout or
  numerical-parity-passed, with the exact reason/report path.
- `export-provenance.json`: immutable run/fold/checkpoint associations.
- `parity/`: generated ONNX, float vectors and native/browser gate reports.

The script intentionally requires a new control directory/lock, to prevent
duplicate or stale runs. Native/browser subprocesses each have a 10-minute
ceiling. An eventual selected model still needs its recipe accuracy review,
versioned export artifact, app integration and app-level verification.

## Manual candidate export

```sh
python3 training/export_cn_step1_onnx.py \
  --checkpoint training/checkpoints/cn-step1-seed42.pt \
  --provenance /path/to/export-provenance.json \
  --output-dir training/checkpoints/cn-step1-export-new
node scripts/verify-cn-step1-onnx-web.mjs \
  --vector-dir training/checkpoints/cn-step1-export-new \
  --ort-dir training/checkpoints/onnx-web-deps/node_modules/onnxruntime-web/dist
```

The checkpoint stays read-only. Provenance contains
`target_film_simulation="Classic Negative"`, `target_output_dr=100`,
`seed`, `checkpoint_sha256`, `contract_sha256`, `adaptation_fit_names`
(exactly F24), and `run_manifest`/`fold_manifest` objects containing the
repo-relative `path` and frozen `sha256`. The background stage constructs
this from the trainer's existing freeze manifest; no training format change
is required. The output directory must be new.

Limits remain native max/mean 1e-4/1e-5 and browser 5e-4/5e-5. Browser runtime
is pinned to 1.31.1 and WebGPU explicitly uses NCHW, with a non-fallback
adapter. All model/input/expected-vector hashes and exact case coverage are
checked. A failed rerun removes the old successful CN gate stamp. GPU info
reads the current `adapter.info.isFallbackAdapter` property, with the older
adapter property as a compatibility fallback. ORT severity-W logs are
preserved as warnings, including ordinary CPU shape-node assignment;
actual console/page errors still fail the CN gate. Numerical limits and
model execution are unchanged by that logging correction.

## Architecture plumbing smoke test

```sh
python3 training/export_cn_step1_onnx.py --smoke-round11 \
  --output-dir training/checkpoints/cn-export-smoke-new
node scripts/verify-cn-step1-onnx-web.mjs \
  --vector-dir training/checkpoints/cn-export-smoke-new \
  --ort-dir training/checkpoints/onnx-web-deps/node_modules/onnxruntime-web/dist \
  --allow-provia-smoke
```

This deliberately uses frozen Provia weights and labels every report
`provia-plumbing-smoke`. It cannot serve as a trained Classic Negative
acceptance result. The smoke test verifies architecture/preprocessing/export
compatibility only. The actual adapted weights must repeat the full gate.

Recorded checks: five native and ten browser cases; negative cases reject
missing smoke opt-in, a `passed=true` record with out-of-limit numerical
error, and a changed input vector. The bounded waiter also exits with a
clear timeout when no frozen candidate arrives. Evidence is in
`training/diagnostics/cn-export-plumbing-native.json`,
`cn-export-plumbing-browser.json` and `codex-cn-step1-preflight.json`.

The model output is already rendered Classic Negative in linear sRGB,
output DR100. Applying a Classic Negative simulation LUT afterward would
apply the simulation again. Full Cuban Neg output DR400, recipe controls,
AWB, grain and live-app integration remain separately gated work.
