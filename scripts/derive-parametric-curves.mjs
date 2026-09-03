#!/usr/bin/env node
// Measures the camera's REAL response to white balance shift, highlight
// tone, shadow tone, and saturation ("color") dial values, replacing the
// hand-picked constants in src/engine/webgl/shaders/fragmentShader.ts's
// applyWhiteBalance/applyToneCurve/applySaturation with calibrated control
// points. See ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section
// for the full methodology and why these five knobs weren't covered by the
// Phase 1/2 film-simulation LUT calibration.
//
// Usage: node scripts/derive-parametric-curves.mjs [input-dir] [output-file]
// input-dir defaults to ./calibration-input, output-file to
// src/engine/webgl/generated/wbToneSaturationCurves.ts.
//
// Expects one shoot folder (any of the existing calibration-input/<shoot>
// folders) containing calib-provia.jpg (the zero/baseline point, from a
// Phase 1/2 run) plus this script's own PARAMETRIC_CALIBRATION_RECIPES
// outputs (calib-wb-red-*.jpg, calib-highlight-*.jpg, etc. — see
// src/lib/camera/calibrationRecipes.ts). Only the FIRST shoot folder that
// has both calib-provia.jpg and at least one calib-wb-*.jpg is used — unlike
// the film-sim LUT fit, these are camera-engine properties, not scene-color
// properties, so pooling multiple scenes doesn't add real signal (see the
// plan doc for the reasoning) and would just complicate "which scene do
// these come from" bookkeeping.
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

/** Median per-channel gain (target/baseline) over well-exposed pixels — for white balance. */
function measureChannelGain(baseline, target) {
  const ratiosR = [];
  const ratiosB = [];
  const count = baseline.length / 3;
  for (let i = 0; i < count; i++) {
    const br = baseline[i * 3];
    const bg = baseline[i * 3 + 1];
    const bb = baseline[i * 3 + 2];
    if (br > MIDTONE_MIN && br < MIDTONE_MAX) ratiosR.push(target[i * 3] / br);
    if (bb > MIDTONE_MIN && bb < MIDTONE_MAX) ratiosB.push(target[i * 3 + 2] / bb);
    void bg;
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
function measureHighlightAmount(baseline, target) {
  const implied = [];
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
  const amount = median(implied);
  return amount === null ? 0 : Math.max(-1, Math.min(1, amount));
}

/** Calibrated sAmt, matching applyToneCurve's exact model (result += shadowWeight * sAmt * 0.15) — see measureHighlightAmount's doc comment for why this divides out the real per-pixel shadowWeight instead of assuming it's 1. */
function measureShadowAmount(baseline, target) {
  const implied = [];
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
  const amount = median(implied);
  return amount === null ? 0 : Math.max(-1, Math.min(1, amount));
}

/** Calibrated saturation factor: median chroma ratio (target/baseline) over meaningfully-saturated pixels. */
function measureSaturationFactor(baseline, target) {
  const ratios = [];
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
  const ratio = median(ratios);
  return ratio === null ? 1 : ratio;
}

function findShootFolder(dir) {
  const candidates = [];
  function walk(current) {
    let entries;
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    if (entries.some((e) => e.name === "calib-provia.jpg") && entries.some((e) => /^calib-wb-/.test(e.name))) {
      candidates.push(current);
    }
    for (const entry of entries) {
      if (entry.isDirectory()) walk(join(current, entry.name));
    }
  }
  walk(dir);
  return candidates[0] ?? null;
}

async function loadPixels(path) {
  return loadSamplePixels(path, { sampleSize: SAMPLE_SIZE, trimFraction: TRIM_FRACTION });
}

async function main() {
  const shootFolder = findShootFolder(inputDir);
  if (!shootFolder) {
    console.error(
      `No shoot folder under ${inputDir} has both calib-provia.jpg and a calib-wb-*.jpg — run the Camera tab's ` +
        "Advanced > Parametric Calibration Capture against a RAF that already has a Phase 1/2 shoot folder first.",
    );
    process.exit(1);
  }
  console.log(`Using shoot folder: ${shootFolder}`);

  const baseline = await loadPixels(join(shootFolder, "calib-provia.jpg"));

  async function loadIfExists(slug) {
    const path = join(shootFolder, `calib-${slug}.jpg`);
    return existsSync(path) ? loadPixels(path) : null;
  }

  // --- White balance ---
  const wbRedPoints = [{ shift: 0, gain: 1 }];
  const wbBluePoints = [{ shift: 0, gain: 1 }];
  for (const shift of [-9, -4, 4, 9]) {
    const suffix = shift < 0 ? `m${-shift}` : `p${shift}`;
    const redPixels = await loadIfExists(`wb-red-${suffix}`);
    if (redPixels) {
      const { red } = measureChannelGain(baseline, redPixels);
      wbRedPoints.push({ shift, gain: red });
      console.log(`  wb-red-${suffix}: gain=${red.toFixed(4)}`);
    }
    const bluePixels = await loadIfExists(`wb-blue-${suffix}`);
    if (bluePixels) {
      const { blue } = measureChannelGain(baseline, bluePixels);
      wbBluePoints.push({ shift, gain: blue });
      console.log(`  wb-blue-${suffix}: gain=${blue.toFixed(4)}`);
    }
  }
  wbRedPoints.sort((a, b) => a.shift - b.shift);
  wbBluePoints.sort((a, b) => a.shift - b.shift);

  // --- Highlight / shadow tone ---
  const highlightPoints = [{ value: 0, amount: 0 }];
  for (const value of [-2, 2, 4]) {
    const suffix = value < 0 ? `m${-value}` : `p${value}`;
    const pixels = await loadIfExists(`highlight-${suffix}`);
    if (pixels) {
      const amount = measureHighlightAmount(baseline, pixels);
      highlightPoints.push({ value, amount });
      console.log(`  highlight-${suffix}: hAmt=${amount.toFixed(4)}`);
    }
  }
  highlightPoints.sort((a, b) => a.value - b.value);

  const shadowPoints = [{ value: 0, amount: 0 }];
  for (const value of [-2, 2, 4]) {
    const suffix = value < 0 ? `m${-value}` : `p${value}`;
    const pixels = await loadIfExists(`shadow-${suffix}`);
    if (pixels) {
      const amount = measureShadowAmount(baseline, pixels);
      shadowPoints.push({ value, amount });
      console.log(`  shadow-${suffix}: sAmt=${amount.toFixed(4)}`);
    }
  }
  shadowPoints.sort((a, b) => a.value - b.value);

  // --- Saturation ---
  const saturationPoints = [{ value: 0, factor: 1 }];
  for (const value of [-4, -2, 2, 4]) {
    const suffix = value < 0 ? `m${-value}` : `p${value}`;
    const pixels = await loadIfExists(`saturation-${suffix}`);
    if (pixels) {
      const factor = measureSaturationFactor(baseline, pixels);
      saturationPoints.push({ value, factor });
      console.log(`  saturation-${suffix}: factor=${factor.toFixed(4)}`);
    }
  }
  saturationPoints.sort((a, b) => a.value - b.value);

  mkdirSync(dirname(outputFile), { recursive: true });
  const contents = `// GENERATED by scripts/derive-parametric-curves.mjs — do not hand-edit.
// Rerun the script after a new Phase 3 calibration shoot to refresh these.
// See ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section.

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

export const WB_RED_GAIN_CURVE: WbGainPoint[] = ${JSON.stringify(wbRedPoints)};
export const WB_BLUE_GAIN_CURVE: WbGainPoint[] = ${JSON.stringify(wbBluePoints)};
export const HIGHLIGHT_TONE_CURVE: ToneAmountPoint[] = ${JSON.stringify(highlightPoints)};
export const SHADOW_TONE_CURVE: ToneAmountPoint[] = ${JSON.stringify(shadowPoints)};
export const SATURATION_FACTOR_CURVE: SaturationFactorPoint[] = ${JSON.stringify(saturationPoints)};
`;
  writeFileSync(outputFile, contents);
  console.log(`\nWrote ${outputFile}`);
}

main();
