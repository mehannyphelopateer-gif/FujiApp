# WebGL Engine

Owns the GPU rendering pipeline for the recipe preview.

- `glUtils.ts` — shader compilation, program linking, full-screen quad, image texture upload.
- `shaders/vertexShader.ts` — pass-through vertex shader for the full-screen quad.
- `shaders/fragmentShader.ts` — the recipe pipeline: sharpness convolution → Hald CLUT film
  simulation swap → white balance shift → highlight/shadow tone curve → Color Chrome Effect →
  grain overlay. See the file's header comment for the exact ordering and why.
- `lut.ts` — resolves a `BaseFilmSimulation` to its Hald CLUT PNG (`/public/luts/`), loads and
  caches the image, uploads it as a texture.
- `useWebGLRenderer.ts` — the hook `ImageViewport` calls. Compiles the program once per
  `<canvas>`, re-uploads the image texture when the source changes, re-loads the LUT texture
  when `recipeAdjustment.baseFilmSimulation` changes, and does a uniform-only redraw (no
  recompile) whenever any other part of `recipeAdjustment` changes — that's what keeps recipe
  switching feeling instantaneous.

Known caveats (see `src/lib/exif/README.md` and the plan doc for more):
- The tone curve, Color Chrome Effect, and grain math in `fragmentShader.ts` are parametric
  approximations, not exact fits to Fuji's published color science.
- Of the 14 Hald CLUT PNGs in `/public/luts/`: `classic-chrome`, `provia`, and `velvia` are
  derived directly from this app's own camera via `scripts/derive-luts-from-calibration.mjs`
  (real calibration photos, fitted per the plan doc's methodology). 8 more (`astia`,
  `classic-negative`, `eterna`, `eterna-bleach-bypass`, `nostalgic-neg`, `pro-neg-hi`,
  `pro-neg-std`, `reala-ace`) are adapted from a third-party source (Adobe Camera Raw's own
  Fuji-matching profiles, not Fuji's own conversion software) — see `THIRD_PARTY_LICENSES.md`
  at the repo root for the source, license terms (CC BY-NC-SA 4.0 — non-commercial,
  share-alike, attribution), and conversion method. The remaining 3 (`acros`, `monochrome`,
  `sepia`) are still the original hand-guessed placeholders from
  `scripts/generate-placeholder-luts.mjs` — extending self-calibration to the other 8
  abpy-derived sims is the planned next step (see the plan doc); swap files in this folder for
  any real source with no code changes required, as long as the pixel layout matches (see
  `lut.ts` and the shader's `haldUV()` for the exact spec).
