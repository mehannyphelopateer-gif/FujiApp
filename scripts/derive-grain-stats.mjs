#!/usr/bin/env node
// Measures the camera's REAL grain noise statistics (amplitude + a relative
// spatial-scale ratio) at all 4 discrete strength x size combinations,
// replacing the hand-picked GRAIN_STRENGTH_WEAK/STRONG and GRAIN_SIZE_SCALE
// constants in src/engine/webgl/useWebGLRenderer.ts. See
// ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section.
//
// Usage: node scripts/derive-grain-stats.mjs [input-dir] [output-file]
// input-dir defaults to ./calibration-input, output-file to
// src/engine/webgl/generated/grainCalibration.ts.
//
// Grain is stochastic (real noise), not a deterministic color mapping, so
// no Hald-CLUT approach applies here — only the shader's existing
// strength/sizeScale CONSTANTS are corrected, not applyGrain()'s algorithm
// shape (a full noise-model rewrite is out of scope for this pass).
//
// Amplitude: measured as the standard deviation of (image - Gaussian-blur
// of image) within a low-detail/flat region of the frame (the block with
// the LOWEST baseline/Off variance — real detail contributes far more
// variance than grain noise, so the flattest block isolates grain from
// scene content), with the baseline's OWN residual noise floor (sensor/JPEG
// compression noise present even with grain Off) subtracted in quadrature
// so only the grain effect's OWN added noise is measured. WeakSmall +
// WeakLarge are averaged into one GRAIN_STRENGTH_WEAK (amplitude shouldn't
// depend on blob size), same for Strong.
//
// Spatial scale: this script measures each setting's noise autocorrelation
// falloff distance (a real, well-defined blob-size proxy) and reports the
// RATIO between Small and Large. The shader's `sizeScale` uniform doesn't
// have a known closed-form mapping from "real blob size in pixels" to its
// value (it's a coordinate-scale multiplier into a chaotic hash function,
// not a literal blur radius), so this script keeps today's Small value as
// the anchor and only recalibrates Large RELATIVE to it via the measured
// ratio — a real, if partial, calibration. If the real ratio is wildly
// different from today's guess, that's worth a visual side-by-side check
// (see the plan doc's Verification section for Grain), and potentially a
// follow-up to properly model the hash function's frequency response
// instead of just rescaling it.

import sharp from "sharp";
import { readdirSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const inputDir = process.argv[2] ?? join(__dirname, "..", "calibration-input");
const outputFile = process.argv[3] ?? join(__dirname, "..", "src", "engine", "webgl", "generated", "grainCalibration.ts");

// Grain is high-frequency detail — resize far less aggressively than the
// Hald-CLUT fitting scripts do, so real grain texture survives.
const MAX_DIMENSION = 1200;
const BLOCK_SIZE = 48; // px, for finding the flattest (lowest-baseline-variance) region
const AUTOCORR_MAX_LAG = 16; // px

// Today's hand-picked values, kept as the anchor for Large's relative
// recalibration and as the fallback if no calibration data exists yet.
const DEFAULT_STRENGTH_WEAK = 0.035;
const DEFAULT_STRENGTH_STRONG = 0.08;
const DEFAULT_SIZE_SCALE_SMALL = 0.9;
const DEFAULT_SIZE_SCALE_LARGE = 0.35;

async function loadGrayscale(path) {
  const image = sharp(path).rotate();
  const meta = await image.metadata();
  const scale = Math.min(1, MAX_DIMENSION / Math.max(meta.width, meta.height));
  const width = Math.round((meta.orientation ?? 1) >= 5 ? meta.height * scale : meta.width * scale);
  const height = Math.round((meta.orientation ?? 1) >= 5 ? meta.width * scale : meta.height * scale);

  const { data, info } = await image
    .resize(width, height, { fit: "fill" })
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixels = new Float64Array(info.width * info.height);
  for (let i = 0; i < pixels.length; i++) pixels[i] = data[i] / 255;
  return { pixels, width: info.width, height: info.height };
}

function blockVariance(pixels, width, height, bx, by) {
  const x0 = bx * BLOCK_SIZE;
  const y0 = by * BLOCK_SIZE;
  const x1 = Math.min(width, x0 + BLOCK_SIZE);
  const y1 = Math.min(height, y0 + BLOCK_SIZE);
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const v = pixels[y * width + x];
      sum += v;
      sumSq += v * v;
      count++;
    }
  }
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

/** Finds the flattest block by BASELINE variance, returns its bounds. */
function findFlattestBlock(baselinePixels, width, height) {
  const blocksX = Math.floor(width / BLOCK_SIZE);
  const blocksY = Math.floor(height / BLOCK_SIZE);
  let best = { bx: 0, by: 0, variance: Infinity };
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      const variance = blockVariance(baselinePixels, width, height, bx, by);
      if (variance < best.variance) best = { bx, by, variance };
    }
  }
  return {
    x0: best.bx * BLOCK_SIZE,
    y0: best.by * BLOCK_SIZE,
    x1: Math.min(width, best.bx * BLOCK_SIZE + BLOCK_SIZE),
    y1: Math.min(height, best.by * BLOCK_SIZE + BLOCK_SIZE),
  };
}

/** 3x3 box-blur residual (pixel - local mean) within the given block bounds. */
function residualInBlock(pixels, width, block) {
  const values = [];
  for (let y = block.y0 + 1; y < block.y1 - 1; y++) {
    for (let x = block.x0 + 1; x < block.x1 - 1; x++) {
      let sum = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) sum += pixels[(y + dy) * width + (x + dx)];
      }
      const blurred = sum / 9;
      values.push(pixels[y * width + x] - blurred);
    }
  }
  return values;
}

function stddev(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

/** Horizontal autocorrelation of the residual signal within a block, at a given lag. */
function autocorrelation(pixels, width, block, lag) {
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  const residuals = [];
  for (let y = block.y0 + 1; y < block.y1 - 1; y++) {
    const row = [];
    for (let x = block.x0 + 1; x < block.x1 - 1; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) s += pixels[(y + dy) * width + (x + dx)];
      }
      row.push(pixels[y * width + x] - s / 9);
    }
    residuals.push(row);
  }
  for (const row of residuals) {
    for (let i = 0; i + lag < row.length; i++) {
      sum += row[i] * row[i + lag];
      sumSq += row[i] * row[i];
      count++;
    }
  }
  return count > 0 ? sum / (sumSq || 1e-9) : 0;
}

/** Lag (px) at which horizontal autocorrelation first drops below 1/e — a blob-size proxy. */
function correlationLength(pixels, width, block) {
  for (let lag = 1; lag <= AUTOCORR_MAX_LAG; lag++) {
    if (autocorrelation(pixels, width, block, lag) < 1 / Math.E) return lag;
  }
  return AUTOCORR_MAX_LAG;
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
    if (entries.some((e) => e.name === "calib-provia.jpg") && entries.some((e) => e.name === "calib-grain-weak-small.jpg")) {
      candidates.push(current);
    }
    for (const entry of entries) {
      if (entry.isDirectory()) walk(join(current, entry.name));
    }
  }
  walk(dir);
  return candidates[0] ?? null;
}

async function measureSetting(baseline, baselineBlock, folder, slug) {
  const path = join(folder, `calib-${slug}.jpg`);
  if (!existsSync(path)) return null;
  const { pixels, width } = await loadGrayscale(path);

  const targetResidual = residualInBlock(pixels, width, baselineBlock);
  const baselineResidual = residualInBlock(baseline.pixels, baseline.width, baselineBlock);
  const targetStd = stddev(targetResidual);
  const baselineStd = stddev(baselineResidual);
  const amplitude = Math.sqrt(Math.max(0, targetStd ** 2 - baselineStd ** 2));

  const corrLength = correlationLength(pixels, width, baselineBlock);
  return { amplitude, corrLength };
}

async function main() {
  const shootFolder = findShootFolder(inputDir);
  if (!shootFolder) {
    console.error(
      `No shoot folder under ${inputDir} has both calib-provia.jpg and calib-grain-weak-small.jpg — run the ` +
        "Camera tab's Advanced > Parametric Calibration Capture against a RAF that already has a Phase 1/2 shoot folder first.",
    );
    process.exit(1);
  }
  console.log(`Using shoot folder: ${shootFolder}`);

  const baseline = await loadGrayscale(join(shootFolder, "calib-provia.jpg"));
  const block = findFlattestBlock(baseline.pixels, baseline.width, baseline.height);
  console.log(`Flattest block: (${block.x0},${block.y0})-(${block.x1},${block.y1})`);

  const results = {};
  for (const slug of ["grain-weak-small", "grain-strong-small", "grain-weak-large", "grain-strong-large"]) {
    const result = await measureSetting(baseline, block, shootFolder, slug);
    if (result) {
      results[slug] = result;
      console.log(`  ${slug}: amplitude=${result.amplitude.toFixed(4)}, corrLength=${result.corrLength}px`);
    } else {
      console.log(`  ${slug}: not found — skipping.`);
    }
  }

  const weakAmplitudes = [results["grain-weak-small"]?.amplitude, results["grain-weak-large"]?.amplitude].filter(
    (v) => v !== undefined,
  );
  const strongAmplitudes = [results["grain-strong-small"]?.amplitude, results["grain-strong-large"]?.amplitude].filter(
    (v) => v !== undefined,
  );
  const strengthWeak = weakAmplitudes.length > 0 ? weakAmplitudes.reduce((a, b) => a + b, 0) / weakAmplitudes.length : DEFAULT_STRENGTH_WEAK;
  const strengthStrong =
    strongAmplitudes.length > 0 ? strongAmplitudes.reduce((a, b) => a + b, 0) / strongAmplitudes.length : DEFAULT_STRENGTH_STRONG;

  let sizeScaleSmall = DEFAULT_SIZE_SCALE_SMALL;
  let sizeScaleLarge = DEFAULT_SIZE_SCALE_LARGE;
  const smallCorrLengths = [results["grain-weak-small"]?.corrLength, results["grain-strong-small"]?.corrLength].filter(
    (v) => v !== undefined,
  );
  const largeCorrLengths = [results["grain-weak-large"]?.corrLength, results["grain-strong-large"]?.corrLength].filter(
    (v) => v !== undefined,
  );
  if (smallCorrLengths.length > 0 && largeCorrLengths.length > 0) {
    const avgSmall = smallCorrLengths.reduce((a, b) => a + b, 0) / smallCorrLengths.length;
    const avgLarge = largeCorrLengths.reduce((a, b) => a + b, 0) / largeCorrLengths.length;
    const ratio = avgLarge / avgSmall; // real "Large is N times blobbier than Small" measurement
    sizeScaleLarge = sizeScaleSmall / ratio;
    console.log(`\nMeasured Large/Small blob-size ratio: ${ratio.toFixed(2)}x`);
  }

  mkdirSync(dirname(outputFile), { recursive: true });
  const contents = `// GENERATED by scripts/derive-grain-stats.mjs — do not hand-edit.
// Rerun the script after a new Phase 3 calibration shoot to refresh these.
// See ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section — amplitude
// is a real calibrated measurement, sizeScale.Large is calibrated RELATIVE
// to sizeScale.Small (see the script's header comment for why the absolute
// scale can't be derived the same way).

export const GRAIN_STRENGTH_WEAK = ${strengthWeak};
export const GRAIN_STRENGTH_STRONG = ${strengthStrong};
export const GRAIN_SIZE_SCALE: Record<"Small" | "Large", number> = { Small: ${sizeScaleSmall}, Large: ${sizeScaleLarge} };
`;
  writeFileSync(outputFile, contents);
  console.log(`\nWrote ${outputFile}`);
}

main();
