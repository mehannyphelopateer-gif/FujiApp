# Round 11 ONNX candidates

These seed-42/43 models are export candidates, not deployed app assets.
They include the trained capture-DR conditioning and export-only stable
GroupNorm arithmetic. No checkpoint weights were changed or retrained.

See `../../README.md` and `../../export_round11_onnx.py` for the input
contract, dependency versions, and reproduction commands. Configure ONNX
Runtime Web 1.31.1 with WebGPU `preferredLayout: "NCHW"`; WASM is also
verified. Both use float32.

`native-report.json` records checkpoint/model SHA-256 values and native
parity; `browser-report.json` records the actual Chromium, Apple Metal
adapter, runtime settings, cases, tolerances, and browser parity. Raw
test vectors remain local under `training/checkpoints/round11-onnx/`
and are regenerable from the existing train/monitor corpus. No held-out
photos or targets were opened to produce these artifacts.

The DR100 photo in the export gate checks runtime parity only; because
it belongs to training, it cannot establish unseen-DR100 accuracy.
