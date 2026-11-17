# Local experimental Preview (Round 11, 2026-11-17)

This integrates frozen seed 42 for personal testing: X100VI RAF ->
Provia, output DR100, as-shot WB, otherwise neutral settings, at a
1536px long edge. It is not a release approval or a claim of Fuji-perfect
parity across all recipes. The same-session limitation on the DR100
acceptance result still applies. FinalHeldOut stays locked.

## Open and test

- Browser: `npm run dev`, open its local URL, select Preview, then
  **Experimental Provia**. Use that mode's **Upload X100VI RAF** button.
- Xcode: `npm run build:experimental && npx cap sync ios`, then open
  the existing project with `npx cap open ios` and build/run normally.
  This uses bundled local runtime/model assets, with WASM fallback when
  WebGPU is unavailable. Native WKWebView device testing is still needed;
  browser verification alone does not establish native compatibility.
- Return to **Standard Preview** for the existing recipe workflow.
- Save preview PNG exports the displayed 8-bit image at the working
  resolution. It does not claim full-sensor resolution or 16-bit export.

The experimental switch is present only in development or builds made
with `VITE_EXPERIMENTAL_PROCESSOR=1`. Ordinary `npm run build` omits it
and its model/runtime chunks. No production deployment is performed.
PWA inference assets are cached after their first successful load;
native bundles include them directly.

## Exact processing path

1. RAF header offsets extract the real embedded JPEG for EXIF only.
   The existing Fuji IFD reader verifies X100VI and resolves capture DR
   from DevelopmentDynamicRange / recorded AutoDynamicRange. Disagreeing,
   missing, unsupported, or D Range Priority metadata is rejected.
2. `decodeProcessorLinearRaf` shares the calibration decoder but pins
   highlight mode 0, independent of diagnostic URL flags. It always uses
   LibRaw WASM, including inside native WebViews: CIRAWFilter/JPEG data
   cannot enter this model. It retains full demosaic -> area downsample
   -> 16-bit linear RGB -> the exact FJLRGB16 container convention.
3. The container becomes float32 NCHW RGB / 65535. The model receives
   capture DR as `log2(DR/100)/2`, separately from output DR100.
4. SHA-256 pins the committed seed-42 ONNX. ONNX Runtime Web 1.31.1
   uses its matching Asyncify WASM/loader and explicit WebGPU NCHW layout;
   WASM is the fallback. Numerical output remains float32 linear RGB.
5. Only display/export clamps and encodes sRGB to an 8-bit canvas. No
   legacy shader, Provia LUT, gray-world AWB, inverse LUT, DR simulation,
   or shadow-recovery correction runs over the result.

The two modes have separate upload/render/export controls, so normal
recipe settings cannot silently affect experimental output and an old
standard render cannot be exported as if it came from the new model.
Uploads clear prior output; unsupported files fail explicitly. Unmounting
cancels UI delivery of a pending job; LibRaw is disposed once its decode
finishes. Processor RAF jobs are serialized to bound peak allocation.

## Reproduce the integration check

```sh
npm install
npx playwright install chromium --only-shell
python3 training/export_round11_onnx.py  # regenerate native reference vectors if missing
npm run dev -- --host 127.0.0.1 --port 5182 --strictPort
# In another terminal:
FUJIAPP_URL=http://127.0.0.1:5182 node scripts/verify-round11-app.mjs
```

This checks only three explicitly named existing train/monitor fixtures
(capture DR100/200/400), not DR100Eval or finalHeldOut. It checks the
new app inference entry against original PyTorch vectors in both
backends, original RAF decode byte equality with calibration input,
capture metadata, rejection of unknown DR, upload-to-canvas UI behavior,
invalid replacement clearing, and switching back to normal Preview.
Reports/screenshots stay under gitignored `training/checkpoints/`.

For the packaged experimental build, serve `dist` with `npm run preview`
and run the same script with `FUJIAPP_BUILT=1` and that server's
`FUJIAPP_URL`. Add `FUJIAPP_FORCE_WASM=1` to simulate missing WebGPU.
The packaged check uploads a real approved RAF, downloads its PNG,
compares every displayed/exported RGB byte against the original
PyTorch output after sRGB encoding, checks invalid replacement clearing,
and returns to Standard Preview.

### Verified on this Mac (2026-11-17)

- Typecheck, targeted ESLint, ordinary build, and experimental build pass.
- App inference matches original PyTorch vectors for capture DR100,
  DR200, and DR400 in WASM and WebGPU: maximum absolute error 2.51e-6.
- The app's full RAF decode is byte-identical to the corresponding
  calibration `.fjlrg`, even when a URL requests highlight mode 5.
- Development UI upload/error-clearing/mode-switch checks pass.
- Packaged UI passes with WebGPU available and with WebGPU deliberately
  unavailable (WASM fallback), including PNG download. PNG differs from
  the frozen reference by at most one RGB byte level.
- Ordinary release output was checked to contain neither model nor ORT
  assets. Experimental assets were synced into `ios/App/App/public`, and
  the copied ONNX SHA-256 was checked against the frozen seed-42 value.

Evidence copies: `training/exports/round11/app-*-report.json`.
No training, checkpoint replacement, deployment, or finalHeldOut access
was performed. The test fixtures belong only to existing train/monitor.

Next validation stages remain separate: native Mac/WebView testing,
broader end-to-end recipe behavior, full-resolution export, and the
explicitly authorized finalHeldOut gate before release.
