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
- Of the 14 Hald CLUT PNGs in `/public/luts/`, 11 (everything except `acros`, `monochrome`,
  `sepia`) are adapted from a third-party source (Adobe Camera Raw's own Fuji-matching
  profiles, not Fuji's own conversion software) — see `THIRD_PARTY_LICENSES.md` at the repo
  root for the source, license terms (CC BY-NC-SA 4.0 — non-commercial, share-alike,
  attribution), and conversion method. The remaining 3 are still the original hand-guessed
  placeholders from `scripts/generate-placeholder-luts.mjs`. `scripts/derive-luts-from-calibration.mjs`
  (see the plan doc) is the planned path to eventually replace all of these with LUTs derived
  directly from this app's own camera — swap files in this folder for that or any other real
  source with no code changes required, as long as the pixel layout matches (see `lut.ts` and
  the shader's `haldUV()` for the exact spec).
