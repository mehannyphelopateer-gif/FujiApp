#!/usr/bin/env node
// Measures the camera's REAL response to white balance shift, white
// balance MODE, highlight tone, shadow tone, saturation ("color"), and
// sharpness dial values, replacing the hand-picked constants in
// src/engine/webgl/shaders/fragmentShader.ts's applyWhiteBalance/
// applyToneCurve/applySaturation/applySharpness (and useWebGLRenderer.ts's
// getWbModeGain lookup) with calibrated control points. See
// ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section for the full
// methodology and why these knobs weren't covered by the Phase 1/2
// film-simulation LUT calibration.
//
// Usage: node scripts/derive-parametric-curves.mjs [input-dir] [output-file]
// input-dir defaults to ./calibration-input, output-file to
// src/engine/webgl/generated/wbToneSaturationCurves.ts.
//
// Expects at least one shoot folder (any of the existing
// calibration-input/<shoot> folders) containing calib-provia.jpg (the
// zero/baseline point, from a Phase 1/2 run) plus this script's own
// PARAMETRIC_CALIBRATION_RECIPES outputs (calib-wb-red-*.jpg,
// calib-highlight-*.jpg, etc. — see src/lib/camera/calibrationRecipes.ts).
//
// WB shift and highlight/shadow tone are measured at every integer step
// (not just a sparse handful) after the first real shoot's 4-5-point
// piecewise-linear curves produced a visibly wrong "Classic Cuban Neg"
// render — that recipe's whiteBalance.shift.blue=-5 and shadowTone=1 both
// landed in the GAPS between sampled points, and the real response curve
// isn't necessarily straight between them. PARAMETRIC_CALIBRATION_RECIPES_
// ROUND_2 supplies the extra points.
//
// EVERY shoot folder that has a given test file contributes to that
// point's measurement (pooled samples, one median across all of them —
// same approach derive-luts-from-calibration.mjs already uses for the
// film-sim LUTs). This was originally "first matching folder only,"
// on the theory that WB/tone/saturation are pure camera-engine properties
// independent of scene content, so one scene should be enough — that
// assumption held for tone/saturation/Color Chrome (confirmed via real
// cross-scene isolation tests) but NOT for WB shift: a curve measured
// from one scene (a bright, moderately-lit hallway) applied a real,
// correctly-measured ~12%/21% channel shift that still overshot badly on
// a very differently-lit scene (a dark, high-contrast museum photo with
// an already-extreme existing color cast). Pooling data from more than
// one lighting condition, once available, should produce a curve that's
// less overfit to any single scene's characteristics.
//
// Measurement approach per axis (all relative to the SAME baseline image,
// pixel-correspondence preserved since it's the same physical scene):
// - White balance gain (red/blue): median per-pixel channel ratio
//   (shifted / baseline) over well-exposed, non-clipped pixels — a direct
//   empirical gain, replacing applyWhiteBalance's hand-picked `0.015`
//   scale constant entirely.
// - Highlight tone: fragmentShader.ts's applyToneCurve models this as
//   `result = color * (1 - highlightWeight * hAmt * 0.5)` where
//   `highlightWeight = smoothstep(0.5, 1.0, luma)`. For each qualifying
//   pixel (weight >= 0.5), solves hAmt = 2*(1 - ratio) / weight — dividing
//   out that PIXEL'S OWN exact weight rather than assuming weight ≈ 1,
//   since smoothstep doesn't reach 1 until luma = 1.0 exactly; a synthetic
//   ground-truth check confirmed the naive assume-weight-is-1 version
//   biased the result low by ~25%. Takes the median implied hAmt across
//   qualifying pixels — the shader's own `clamp(highlight/6.0, -1, 1)`
//   guess is replaced by this real value, keeping the same shape.
// - Shadow tone: same idea, but the shader's model is additive
//   (`result += shadowWeight * sAmt * 0.15`, `shadowWeight = 1 -
//   smoothstep(0.0, 0.5, luma)`) — solves sAmt = delta / (0.15 * weight)
//   per qualifying pixel, same exact-weight-division reasoning as highlight.
// - Saturation: the shader's `applySaturation` blends toward/away from luma
//   by `factor = 1 + clamp(delta/8, -1, 1)`. Measuring the real median
//   chroma ratio (target chroma / baseline chroma, over pixels with
//   meaningful baseline chroma) gives the calibrated `factor` directly.
// - Sharpness: a first-order linear approximation of applySharpness's local
//   contrast response (see measureSharpenAmount's doc comment) — measures a
//   simple Laplacian at each real-detail pixel in both images and solves
//   for the implied amount from the ratio. Needs a much larger sample size
//   than the other axes (SHARPNESS_SAMPLE_SIZE) since it's a genuinely
//   high-frequency spatial signal that a small downsample would destroy.
// - White balance MODE: reuses the exact same measureChannelGain as WB
//   shift (a mode is just a different fixed color-temperature baseline, no
//   different in kind from a shift) — one real photo per mode at shift 0,
//   compared against the Auto/Provia baseline. "Kelvin" (a continuous
//   temperature dial, not a fixed preset) is out of scope for this pass.
//
// Every curve is forced to include an explicit (0, neutral) control point
// (gain 1.0 / hAmt 0 / sAmt 0 / factor 1.0) — a zero dial value MUST be a
// true no-op by construction, this isn't something to fit from noisy data.

import { readdirSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { loadSamplePixels } from "./lib/hald-clut-fitting.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const inputDir = process.argv[2] ?? join(__dirname, "..", "calibration-input");
const outputFile =
  process.argv[3] ?? join(__dirname, "..", "src", "engine", "webgl", "generated", "wbToneSaturationCurves.ts");

const SAMPLE_SIZE = 256;
// Sharpening is a high-frequency spatial effect — a 256px downsample would
// low-pass-filter away most of the real detail it acts on, so sharpness
// measurement uses a much larger sample size (same reasoning as
// derive-grain-stats.mjs's MAX_DIMENSION for the same kind of high-
// frequency signal).
const SHARPNESS_SAMPLE_SIZE = 1024;
const TRIM_FRACTION = 0.08;

// Well-exposed midtone band for WB gain measurement — avoids near-black/
// near-white pixels where a channel ratio is numerically unstable.
const MIDTONE_MIN = 0.1;
const MIDTONE_MAX = 0.9;

const MIN_CHROMA_FOR_SATURATION = 0.05;
// Below this, a sample's highlightWeight/shadowWeight is small enough that
// dividing it out (below) amplifies noise more than it corrects bias.
const MIN_ZONE_WEIGHT = 0.5;

function luma(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Exact port of GLSL's smoothstep — used to correctly weight highlight/shadow-zone samples, matching applyToneCurve exactly. */
function smoothstep(edge0, edge1, x) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/**
 * Median per-channel gain (target/baseline) over well-exposed pixels, in
 * the SAME gamma-encoded space fragmentShader.ts's applyWhiteBalance
 * applies it in directly. TRIED AND REVERTED: measuring/applying this in
 * linear light instead (converting via the sRGB transfer function) on the
 * theory that WB is physically a linear-light correction — real hardware
 * evidence contradicted it (see applyWhiteBalance's doc comment for the
 * actual math and the real-photo test that disproved it), so this stays
 * a direct gamma-space ratio.
 *
 * `pairs` is a list of {baseline, target} from every contributing shoot
 * folder (see the file header comment for why WB pools across scenes) —
 * samples from all of them are pooled into one ratio list before taking
 * the median, not averaged per-folder first.
 */
function measureChannelGain(pairs) {
  const ratiosR = [];
  const ratiosB = [];
  for (const { baseline, target } of pairs) {
    const count = baseline.length / 3;
    for (let i = 0; i < count; i++) {
      const br = baseline[i * 3];
      const bb = baseline[i * 3 + 2];
      if (br > MIDTONE_MIN && br < MIDTONE_MAX) ratiosR.push(target[i * 3] / br);
      if (bb > MIDTONE_MIN && bb < MIDTONE_MAX) ratiosB.push(target[i * 3 + 2] / bb);
    }
  }
  return { red: median(ratiosR) ?? 1, blue: median(ratiosB) ?? 1 };
}

/**
 * Calibrated hAmt, matching applyToneCurve's exact model
 * (result = color * (1 - highlightWeight * hAmt * 0.5)): for each pixel,
 * solves for the implied hAmt from the real measured ratio and that
 * pixel's own exact highlightWeight (smoothstep(0.5, 1.0, luma)), rather
 * than assuming highlightWeight ≈ 1 for "highlight enough" pixels — that
 * assumption alone was confirmed (via a synthetic ground-truth check) to
 * bias the naive version low by ~25%, since smoothstep doesn't reach 1
 * until luma = 1.0 exactly. Only pixels with weight >= MIN_ZONE_WEIGHT are
 * used, since dividing by a small weight amplifies noise.
 */
function measureHighlightAmount(pairs) {
  const implied = [];
  for (const { baseline, target } of pairs) {
    const count = baseline.length / 3;
    for (let i = 0; i < count; i++) {
      const br = baseline[i * 3];
      const bg = baseline[i * 3 + 1];
      const bb = baseline[i * 3 + 2];
      const bl = luma(br, bg, bb);
      const weight = smoothstep(0.5, 1.0, bl);
      if (weight < MIN_ZONE_WEIGHT || bl <= 0.001) continue;
      const tl = luma(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
      implied.push((2 * (1 - tl / bl)) / weight);
    }
  }
  const amount = median(implied);
  return amount === null ? 0 : Math.max(-1, Math.min(1, amount));
}

/** Calibrated sAmt, matching applyToneCurve's exact model (result += shadowWeight * sAmt * 0.15) — see measureHighlightAmount's doc comment for why this divides out the real per-pixel shadowWeight instead of assuming it's 1. */
function measureShadowAmount(pairs) {
  const implied = [];
  for (const { baseline, target } of pairs) {
    const count = baseline.length / 3;
    for (let i = 0; i < count; i++) {
      const br = baseline[i * 3];
      const bg = baseline[i * 3 + 1];
      const bb = baseline[i * 3 + 2];
      const bl = luma(br, bg, bb);
      const weight = 1 - smoothstep(0.0, 0.5, bl);
      if (weight < MIN_ZONE_WEIGHT) continue;
      const tl = luma(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
      implied.push((tl - bl) / (0.15 * weight));
    }
  }
  const amount = median(implied);
  return amount === null ? 0 : Math.max(-1, Math.min(1, amount));
}

/** Calibrated saturation factor: median chroma ratio (target/baseline) over meaningfully-saturated pixels. */
function measureSaturationFactor(pairs) {
  const ratios = [];
  for (const { baseline, target } of pairs) {
    const count = baseline.length / 3;
    for (let i = 0; i < count; i++) {
      const br = baseline[i * 3];
      const bg = baseline[i * 3 + 1];
      const bb = baseline[i * 3 + 2];
      const bChroma = Math.max(br, bg, bb) - Math.min(br, bg, bb);
      if (bChroma < MIN_CHROMA_FOR_SATURATION) continue;
      const tr = target[i * 3];
      const tg = target[i * 3 + 1];
      const tb = target[i * 3 + 2];
      const tChroma = Math.max(tr, tg, tb) - Math.min(tr, tg, tb);
      ratios.push(tChroma / bChroma);
    }
  }
  const ratio = median(ratios);
  return ratio === null ? 1 : ratio;
}

function toLumaGrid(pixels, size) {
  const grid = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) grid[i] = luma(pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]);
  return grid;
}

function laplacianAt(grid, size, x, y) {
  return (
    4 * grid[y * size + x] - grid[(y - 1) * size + x] - grid[(y + 1) * size + x] - grid[y * size + x - 1] - grid[y * size + x + 1]
  );
}

// Below this, a pixel is in a flat/low-detail region with no real edge to
// measure sharpening against — including it just adds noise.
const MIN_LAPLACIAN_FOR_SHARPNESS = 0.02;

/**
 * Calibrated sharpen amount, using a first-order linear approximation of
 * applySharpness's local-contrast response: for a 4-neighbor unsharp mask,
 * sharpening scales local high-frequency content (approximated here by a
 * simple Laplacian) by roughly (1 + amount*0.5); the shader's blur branch
 * (negative amount) attenuates it by roughly the same factor from the other
 * direction. Both directions collapse to one solve: for each pixel with
 * real detail (|baseline Laplacian| above a noise floor), the implied
 * amount is 2*(targetLaplacian/baselineLaplacian - 1). Takes the median
 * across qualifying pixels for robustness against real-photo noise.
 *
 * KNOWN LIMITATION, confirmed via synthetic ground-truth testing (unlike
 * highlight/shadow/saturation above, which solve the shader's EXACT
 * formula and were confirmed exact): this first-order model only
 * approximates the real cascaded 2D convolution, and measured a
 * consistent ~20% undershoot on synthetic detail, plus poor discrimination
 * at strong negative (blur) amounts since the shader's own blur branch
 * saturates once amount <= -2 (its `clamp(-amount*0.5, 0, 1)` hits 1) —
 * amounts -2 and -3.5 are genuinely indistinguishable in the shader's own
 * output, not a measurement flaw. Treat this axis's calibrated curve as
 * directionally correct and roughly right in magnitude, not as precise as
 * the other axes — worth an extra visual side-by-side check once real
 * data lands (same spirit as grain's stochastic-noise caveat).
 */
function measureSharpenAmount(pairs, size) {
  const implied = [];
  for (const { baseline, target } of pairs) {
    const baseGrid = toLumaGrid(baseline, size);
    const targetGrid = toLumaGrid(target, size);
    for (let y = 1; y < size - 1; y++) {
      for (let x = 1; x < size - 1; x++) {
        const bL = laplacianAt(baseGrid, size, x, y);
        if (Math.abs(bL) < MIN_LAPLACIAN_FOR_SHARPNESS) continue;
        const tL = laplacianAt(targetGrid, size, x, y);
        implied.push(2 * (tL / bL - 1));
      }
    }
  }
  const amount = median(implied);
  return amount === null ? 0 : Math.max(-4, Math.min(4, amount));
}

/** Every directory under `dir` that directly contains calib-provia.jpg — one entry per calibration shoot. */
function findShootFolders(dir) {
  const found = [];
  function walk(current) {
    let entries;
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    if (entries.some((e) => e.name === "calib-provia.jpg")) found.push(current);
    for (const entry of entries) {
      if (entry.isDirectory()) walk(join(current, entry.name));
    }
  }
  walk(dir);
  return found;
}

async function loadPixels(path) {
  return loadSamplePixels(path, { sampleSize: SAMPLE_SIZE, trimFraction: TRIM_FRACTION });
}

async function main() {
  const shootFolders = findShootFolders(inputDir);
  if (shootFolders.length === 0) {
    console.error(
      `No shoot folder under ${inputDir} has calib-provia.jpg — run the Camera tab's Advanced > LUT Calibration ` +
        "Capture first, then Advanced > Parametric Calibration Capture against the same RAF.",
    );
    process.exit(1);
  }
  console.log(`Found ${shootFolders.length} shoot folder(s):`);
  for (const folder of shootFolders) console.log(`  ${folder}`);

  // Baselines are loaded once per folder, lazily, and reused across every
  // test point that folder contributes to.
  const baselineCache = new Map();
  const sharpnessBaselineCache = new Map();
  async function getBaseline(folder) {
    if (!baselineCache.has(folder)) {
      baselineCache.set(folder, await loadPixels(join(folder, "calib-provia.jpg")));
    }
    return baselineCache.get(folder);
  }
  async function getSharpnessBaseline(folder) {
    if (!sharpnessBaselineCache.has(folder)) {
      sharpnessBaselineCache.set(
        folder,
        await loadSamplePixels(join(folder, "calib-provia.jpg"), { sampleSize: SHARPNESS_SAMPLE_SIZE, trimFraction: TRIM_FRACTION }),
      );
    }
    return sharpnessBaselineCache.get(folder);
  }

  /** Pairs {baseline, target} from every shoot folder that has calib-<slug>.jpg, for pooling. */
  async function collectPairs(slug) {
    const pairs = [];
    for (const folder of shootFolders) {
      const path = join(folder, `calib-${slug}.jpg`);
      if (!existsSync(path)) continue;
      pairs.push({ baseline: await getBaseline(folder), target: await loadPixels(path) });
    }
    return pairs;
  }

  async function collectSharpnessPairs(slug) {
    const pairs = [];
    for (const folder of shootFolders) {
      const path = join(folder, `calib-${slug}.jpg`);
      if (!existsSync(path)) continue;
      pairs.push({
        baseline: await getSharpnessBaseline(folder),
        target: await loadSamplePixels(path, { sampleSize: SHARPNESS_SAMPLE_SIZE, trimFraction: TRIM_FRACTION }),
      });
    }
    return pairs;
  }

  // --- White balance ---
  const wbRedPoints = [{ shift: 0, gain: 1 }];
  const wbBluePoints = [{ shift: 0, gain: 1 }];
  for (const shift of [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
    const suffix = shift < 0 ? `m${-shift}` : `p${shift}`;
    const redPairs = await collectPairs(`wb-red-${suffix}`);
    if (redPairs.length > 0) {
      const { red } = measureChannelGain(redPairs);
      wbRedPoints.push({ shift, gain: red });
      console.log(`  wb-red-${suffix}: gain=${red.toFixed(4)} (${redPairs.length} shoot(s))`);
    }
    const bluePairs = await collectPairs(`wb-blue-${suffix}`);
    if (bluePairs.length > 0) {
      const { blue } = measureChannelGain(bluePairs);
      wbBluePoints.push({ shift, gain: blue });
      console.log(`  wb-blue-${suffix}: gain=${blue.toFixed(4)} (${bluePairs.length} shoot(s))`);
    }
  }
  wbRedPoints.sort((a, b) => a.shift - b.shift);
  wbBluePoints.sort((a, b) => a.shift - b.shift);

  // --- Highlight / shadow tone ---
  const highlightPoints = [{ value: 0, amount: 0 }];
  for (const value of [-2, -1, 1, 2, 3, 4]) {
    const suffix = value < 0 ? `m${-value}` : `p${value}`;
    const pairs = await collectPairs(`highlight-${suffix}`);
    if (pairs.length > 0) {
      const amount = measureHighlightAmount(pairs);
      highlightPoints.push({ value, amount });
      console.log(`  highlight-${suffix}: hAmt=${amount.toFixed(4)} (${pairs.length} shoot(s))`);
    }
  }
  highlightPoints.sort((a, b) => a.value - b.value);

  const shadowPoints = [{ value: 0, amount: 0 }];
  for (const value of [-2, -1, 1, 2, 3, 4]) {
    const suffix = value < 0 ? `m${-value}` : `p${value}`;
    const pairs = await collectPairs(`shadow-${suffix}`);
    if (pairs.length > 0) {
      const amount = measureShadowAmount(pairs);
      shadowPoints.push({ value, amount });
      console.log(`  shadow-${suffix}: sAmt=${amount.toFixed(4)} (${pairs.length} shoot(s))`);
    }
  }
  shadowPoints.sort((a, b) => a.value - b.value);

  // --- Saturation ---
  const saturationPoints = [{ value: 0, factor: 1 }];
  for (const value of [-4, -2, 2, 4]) {
    const suffix = value < 0 ? `m${-value}` : `p${value}`;
    const pairs = await collectPairs(`saturation-${suffix}`);
    if (pairs.length > 0) {
      const factor = measureSaturationFactor(pairs);
      saturationPoints.push({ value, factor });
      console.log(`  saturation-${suffix}: factor=${factor.toFixed(4)} (${pairs.length} shoot(s))`);
    }
  }
  saturationPoints.sort((a, b) => a.value - b.value);

  // --- Sharpness (needs a much larger sample than the other axes — see SHARPNESS_SAMPLE_SIZE) ---
  const sharpenPoints = [{ value: 0, amount: 0 }];
  for (const value of [-4, -2, 2, 4]) {
    const suffix = value < 0 ? `m${-value}` : `p${value}`;
    const pairs = await collectSharpnessPairs(`sharpness-${suffix}`);
    if (pairs.length > 0) {
      const amount = measureSharpenAmount(pairs, SHARPNESS_SAMPLE_SIZE);
      sharpenPoints.push({ value, amount });
      console.log(`  sharpness-${suffix}: amount=${amount.toFixed(4)} (${pairs.length} shoot(s))`);
    }
  }
  sharpenPoints.sort((a, b) => a.value - b.value);

  // --- White balance MODE (shift held at 0) ---
  // CONFIRMED (first real Phase 3 shoot, all 7 modes measured EXACTLY
  // {red:1, blue:1} — no signal at all): this matches a hardware limitation
  // already documented in src/lib/camera/patchRawProfile.ts — the RAW-
  // conversion engine ignores the requested WB mode entirely and always
  // renders using the RAF's as-shot WB, confirmed there against real
  // hardware across three source photos before this script existed. This
  // isn't a missing-data gap the way sharpness's "no derive script yet" was
  // — it's a firmware-level ceiling on what convertWithRecipe can ever
  // exercise for this field. Every value below being 1.0 is the honestly
  // measured, correct answer, not a placeholder waiting on more data.
  const wbModeGain = { Auto: { red: 1, blue: 1 } };
  for (const mode of ["Daylight", "Shade", "Fluorescent1", "Fluorescent2", "Fluorescent3", "Incandescent", "Underwater"]) {
    const pairs = await collectPairs(`wbmode-${mode.toLowerCase()}`);
    if (pairs.length > 0) {
      const gain = measureChannelGain(pairs);
      wbModeGain[mode] = gain;
      console.log(`  wbmode-${mode.toLowerCase()}: red=${gain.red.toFixed(4)}, blue=${gain.blue.toFixed(4)} (${pairs.length} shoot(s))`);
    }
  }

  mkdirSync(dirname(outputFile), { recursive: true });
  const contents = `// GENERATED by scripts/derive-parametric-curves.mjs — do not hand-edit.
// Rerun the script after a new Phase 3 calibration shoot to refresh these.
// See ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section.
//
// WB_MODE_GAIN is all {red:1,blue:1} by design, not a placeholder: the
// RAW-conversion engine ignores requested WB mode entirely (confirmed
// against real hardware, see src/lib/camera/patchRawProfile.ts and this
// script's WB-mode measurement loop) — there is no real signal this app
// can ever measure for that field via convertWithRecipe.

export interface WbGainPoint {
  shift: number;
  gain: number;
}

export interface ToneAmountPoint {
  value: number;
  amount: number;
}

export interface SaturationFactorPoint {
  value: number;
  factor: number;
}

export interface SharpenAmountPoint {
  value: number;
  amount: number;
}

export interface WbModeGain {
  red: number;
  blue: number;
}

export const WB_RED_GAIN_CURVE: WbGainPoint[] = ${JSON.stringify(wbRedPoints)};
export const WB_BLUE_GAIN_CURVE: WbGainPoint[] = ${JSON.stringify(wbBluePoints)};
export const HIGHLIGHT_TONE_CURVE: ToneAmountPoint[] = ${JSON.stringify(highlightPoints)};
export const SHADOW_TONE_CURVE: ToneAmountPoint[] = ${JSON.stringify(shadowPoints)};
export const SATURATION_FACTOR_CURVE: SaturationFactorPoint[] = ${JSON.stringify(saturationPoints)};
export const SHARPEN_AMOUNT_CURVE: SharpenAmountPoint[] = ${JSON.stringify(sharpenPoints)};
export const WB_MODE_GAIN: Record<string, WbModeGain> = ${JSON.stringify(wbModeGain)};
`;
  writeFileSync(outputFile, contents);
  console.log(`\nWrote ${outputFile}`);
}

main();
