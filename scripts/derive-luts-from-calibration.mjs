#!/usr/bin/env node
// Derives real Hald CLUT PNGs from calibration image pairs exported by
// CalibrationCapture.tsx (Camera tab -> Advanced -> LUT Calibration
// Capture), replacing the hand-guessed placeholders from
// generate-placeholder-luts.mjs with LUTs fitted to real camera output.
// See ~/.claude/plans/indexed-inventing-wren.md for the full methodology
// and rationale (global affine fit + local IDW-corrected coarse grid +
// trilinear upsample to the final 64-level Hald CLUT).
//
// Usage:
//   node scripts/derive-luts-from-calibration.mjs [input-dir] [output-dir]
// input-dir defaults to ./calibration-input, output-dir to ./public/luts
// (only overwrites PNGs for film sims actually found in the input).
//
// Expected input layout — one subfolder per calibration SHOOT (one source
// RAF run once through the capture tool), e.g.:
//   calibration-input/shoot1/calib-neutral.jpg
//   calibration-input/shoot1/calib-classic-chrome.jpg
//   calibration-input/shoot1/calib-velvia.jpg
//   calibration-input/shoot1/calib-provia.jpg
//   calibration-input/shoot2/calib-neutral.jpg
//   calibration-input/shoot2/calib-classic-chrome.jpg
//   ...
// A neutral image is only ever paired with sim images from its OWN shoot
// folder — pairing across different source photos would compare unrelated
// scenes, not the same scene before/after. Samples from every shoot folder
// that has a given film sim are pooled together into one fit for that sim.
//
// Fitting happens directly in sRGB-gamma-encoded 0..1 pixel space (raw
// 8-bit JPEG values / 255), matching exactly what the WebGL shader
// consumes at runtime — no gamma linearization anywhere in this pipeline,
// intentionally, for self-consistency with fragmentShader.ts's apply3DLut().

import sharp from "sharp";
import { PNG } from "pngjs";
import { readdirSync, statSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const inputDir = process.argv[2] ?? join(__dirname, "..", "calibration-input");
const outputDir = process.argv[3] ?? join(__dirname, "..", "public", "luts");

// Must match src/engine/webgl/shaders/fragmentShader.ts's haldUV()/apply3DLut()
// and generate-placeholder-luts.mjs's layout exactly.
const LEVELS = 64;
const N = Math.sqrt(LEVELS); // 8
const SIDE = LEVELS * N; // 512

const SAMPLE_SIZE = 256; // common square resize target for correspondence sampling
const TRIM_FRACTION = 0.08; // fraction trimmed off each edge before resize, per image
const GRID_LEVELS = 17; // coarse control-grid resolution for the local correction layer
const CORRECTION_CLAMP = 0.14; // max |local correction| per channel (0..1 scale)
const IDW_RADIUS = 3; // grid-node search radius (in grid cells) for IDW falloff
const IDW_EPSILON = 0.5;
const MAX_SAMPLES_PER_CELL = 500; // cap per grid cell so one dense region can't dominate memory/sort cost

function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}

/** Finds every directory under `dir` that directly contains a calib-neutral.jpg. */
function findShootFolders(dir) {
  const found = [];
  function walk(current) {
    let entries;
    try {
      entries = readdirSync(current);
    } catch {
      return;
    }
    if (entries.includes("calib-neutral.jpg")) found.push(current);
    for (const entry of entries) {
      const full = join(current, entry);
      if (statSync(full).isDirectory()) walk(full);
    }
  }
  walk(dir);
  return found;
}

/**
 * Loads a JPEG, center-trims, resizes to a common square, returns a flat
 * Float32Array of RGB in 0..1 (row-major, no alpha).
 *
 * `.rotate()` with no argument tells sharp to auto-orient using the image's
 * own EXIF Orientation tag before doing anything else — required here
 * because the camera's own JPEG conversion (calib-<slug>.jpg) and
 * decodeNeutralRaf's output (calib-neutral.jpg) represent a portrait-held
 * shot two different, equally valid ways: the camera leaves pixels in
 * sensor-native (landscape) order and sets an Orientation tag (e.g. 8) for
 * viewers to rotate on display, while decodeNeutralRaf physically rotates
 * the pixel buffer itself and leaves Orientation at 1 (verified directly
 * against real calibration shoots with Python/PIL's EXIF reader — sips'
 * own `-g orientation` silently returns nil for both, which is what led an
 * earlier version of this script to (wrongly) suspect a decodeNeutralRaf
 * bug and paper over it with a fragile empirical rotation-correlation
 * guess instead of this straightforward fix). Without auto-orienting both
 * images the same way, pixel (x,y) in one doesn't correspond to the same
 * real-world scene point as pixel (x,y) in the other, corrupting the fit.
 */
async function loadSamplePixels(path) {
  const image = sharp(path).rotate();
  const meta = await image.metadata();
  const swapped = (meta.orientation ?? 1) >= 5; // orientations 5-8 swap width/height
  const width = swapped ? meta.height : meta.width;
  const height = swapped ? meta.width : meta.height;
  const trimX = Math.round(width * TRIM_FRACTION);
  const trimY = Math.round(height * TRIM_FRACTION);

  const { data } = await image
    .extract({
      left: trimX,
      top: trimY,
      width: width - trimX * 2,
      height: height - trimY * 2,
    })
    .resize(SAMPLE_SIZE, SAMPLE_SIZE, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixels = new Float32Array((data.length / 3) * 3);
  for (let i = 0; i < data.length; i++) pixels[i] = data[i] / 255;
  return pixels;
}

/** Solves a 4x4 linear system via Gaussian elimination with partial pivoting. */
function solve4x4(A, b) {
  const M = A.map((row) => row.slice());
  const y = b.slice();
  const n = 4;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];
    [y[col], y[pivot]] = [y[pivot], y[col]];
    const diag = M[col][col] || 1e-9;
    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / diag;
      for (let c = col; c < n; c++) M[row][c] -= factor * M[col][c];
      y[row] -= factor * y[col];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = y[row];
    for (let c = row + 1; c < n; c++) sum -= M[row][c] * x[c];
    x[row] = sum / (M[row][row] || 1e-9);
  }
  return x;
}

/** Fits output_channel ~= a*R + b*G + c*B + d via ordinary least squares, one solve per output channel. */
function fitGlobalAffine(neutral, target, sampleCount) {
  const XtX = Array.from({ length: 4 }, () => new Array(4).fill(0));
  const XtY = [new Array(4).fill(0), new Array(4).fill(0), new Array(4).fill(0)];

  for (let i = 0; i < sampleCount; i++) {
    const r = neutral[i * 3];
    const g = neutral[i * 3 + 1];
    const b = neutral[i * 3 + 2];
    const row = [r, g, b, 1];
    for (let a = 0; a < 4; a++) {
      for (let c = 0; c < 4; c++) XtX[a][c] += row[a] * row[c];
      for (let ch = 0; ch < 3; ch++) XtY[ch][a] += row[a] * target[i * 3 + ch];
    }
  }

  return [0, 1, 2].map((ch) => solve4x4(XtX, XtY[ch]));
}

function evalAffine(coeffs, r, g, b) {
  return [0, 1, 2].map((ch) => coeffs[ch][0] * r + coeffs[ch][1] * g + coeffs[ch][2] * b + coeffs[ch][3]);
}

/** Bins (neutral -> residual-from-global-affine) samples into a GRID_LEVELS^3 grid, median-aggregated per cell. */
function buildResidualGrid(neutral, target, sampleCount, affine) {
  const cellCount = GRID_LEVELS ** 3;
  const buckets = Array.from({ length: cellCount }, () => []);

  for (let i = 0; i < sampleCount; i++) {
    const r = neutral[i * 3];
    const g = neutral[i * 3 + 1];
    const b = neutral[i * 3 + 2];
    const rL = Math.min(GRID_LEVELS - 1, Math.round(r * (GRID_LEVELS - 1)));
    const gL = Math.min(GRID_LEVELS - 1, Math.round(g * (GRID_LEVELS - 1)));
    const bL = Math.min(GRID_LEVELS - 1, Math.round(b * (GRID_LEVELS - 1)));
    const cell = (rL * GRID_LEVELS + gL) * GRID_LEVELS + bL;
    if (buckets[cell].length >= MAX_SAMPLES_PER_CELL) continue;

    const predicted = evalAffine(affine, r, g, b);
    buckets[cell].push([
      target[i * 3] - predicted[0],
      target[i * 3 + 1] - predicted[1],
      target[i * 3 + 2] - predicted[2],
    ]);
  }

  function median(values) {
    const sorted = values.slice().sort((a, b2) => a - b2);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  const grid = new Array(cellCount).fill(null);
  for (let cell = 0; cell < cellCount; cell++) {
    const samples = buckets[cell];
    if (samples.length === 0) continue;
    grid[cell] = [0, 1, 2].map((ch) => median(samples.map((s) => s[ch])));
  }
  return grid;
}

/** IDW-smooths the sparse residual grid so every node (occupied or not) has a bounded, distance-decayed correction. */
function smoothGrid(grid) {
  const smoothed = new Array(grid.length).fill(null);
  for (let rL = 0; rL < GRID_LEVELS; rL++) {
    for (let gL = 0; gL < GRID_LEVELS; gL++) {
      for (let bL = 0; bL < GRID_LEVELS; bL++) {
        const cell = (rL * GRID_LEVELS + gL) * GRID_LEVELS + bL;
        let weightSum = 0;
        const acc = [0, 0, 0];
        for (let dr = -IDW_RADIUS; dr <= IDW_RADIUS; dr++) {
          const r2 = rL + dr;
          if (r2 < 0 || r2 >= GRID_LEVELS) continue;
          for (let dg = -IDW_RADIUS; dg <= IDW_RADIUS; dg++) {
            const g2 = gL + dg;
            if (g2 < 0 || g2 >= GRID_LEVELS) continue;
            for (let db = -IDW_RADIUS; db <= IDW_RADIUS; db++) {
              const b2 = bL + db;
              if (b2 < 0 || b2 >= GRID_LEVELS) continue;
              const neighbor = grid[(r2 * GRID_LEVELS + g2) * GRID_LEVELS + b2];
              if (!neighbor) continue;
              const dist2 = dr * dr + dg * dg + db * db;
              const weight = 1 / (dist2 + IDW_EPSILON);
              weightSum += weight;
              acc[0] += weight * neighbor[0];
              acc[1] += weight * neighbor[1];
              acc[2] += weight * neighbor[2];
            }
          }
        }
        smoothed[cell] =
          weightSum > 0
            ? acc.map((v) => Math.max(-CORRECTION_CLAMP, Math.min(CORRECTION_CLAMP, v / weightSum)))
            : [0, 0, 0];
      }
    }
  }
  return smoothed;
}

function sampleGridTrilinear(grid, r, g, b) {
  const pos = [r, g, b].map((v) => clamp01(v) * (GRID_LEVELS - 1));
  const lo = pos.map(Math.floor);
  const frac = pos.map((v, i) => v - lo[i]);
  const hi = lo.map((v) => Math.min(GRID_LEVELS - 1, v + 1));

  function at(rL, gL, bL) {
    return grid[(rL * GRID_LEVELS + gL) * GRID_LEVELS + bL];
  }
  function lerp3(a, b2, t) {
    return [0, 1, 2].map((i) => a[i] * (1 - t) + b2[i] * t);
  }

  const c000 = at(lo[0], lo[1], lo[2]);
  const c100 = at(hi[0], lo[1], lo[2]);
  const c010 = at(lo[0], hi[1], lo[2]);
  const c110 = at(hi[0], hi[1], lo[2]);
  const c001 = at(lo[0], lo[1], hi[2]);
  const c101 = at(hi[0], lo[1], hi[2]);
  const c011 = at(lo[0], hi[1], hi[2]);
  const c111 = at(hi[0], hi[1], hi[2]);

  const c00 = lerp3(c000, c100, frac[0]);
  const c10 = lerp3(c010, c110, frac[0]);
  const c01 = lerp3(c001, c101, frac[0]);
  const c11 = lerp3(c011, c111, frac[0]);
  const c0 = lerp3(c00, c10, frac[1]);
  const c1 = lerp3(c01, c11, frac[1]);
  return lerp3(c0, c1, frac[2]);
}

/** Writes a Hald CLUT PNG, sampling `lookup(r, g, b)` (each 0..1) at every one of the 64^3 cells. */
function writeLutPng(lookup, outPath) {
  const png = new PNG({ width: SIDE, height: SIDE });
  for (let y = 0; y < SIDE; y++) {
    for (let x = 0; x < SIDE; x++) {
      const b = Math.floor(y / N);
      const gHi = y % N;
      const gLo = Math.floor(x / LEVELS);
      const r = x % LEVELS;
      const g = gHi * N + gLo;

      const [tr, tg, tb] = lookup(r / (LEVELS - 1), g / (LEVELS - 1), b / (LEVELS - 1));

      const idx = (SIDE * y + x) << 2;
      png.data[idx] = Math.round(clamp01(tr) * 255);
      png.data[idx + 1] = Math.round(clamp01(tg) * 255);
      png.data[idx + 2] = Math.round(clamp01(tb) * 255);
      png.data[idx + 3] = 255;
    }
  }
  writeFileSync(outPath, PNG.sync.write(png));
}

async function deriveLutForSim(slug, shootFolders) {
  const neutralChunks = [];
  const targetChunks = [];

  for (const folder of shootFolders) {
    const simPath = join(folder, `calib-${slug}.jpg`);
    if (!existsSync(simPath)) continue;
    const neutralPath = join(folder, "calib-neutral.jpg");
    console.log(`  reading pair: ${neutralPath} / ${simPath}`);
    neutralChunks.push(await loadSamplePixels(neutralPath));
    targetChunks.push(await loadSamplePixels(simPath));
  }

  if (neutralChunks.length === 0) {
    console.log(`  no calib-${slug}.jpg found in any shoot folder — skipping.`);
    return false;
  }

  const totalSamples = neutralChunks.reduce((sum, arr) => sum + arr.length / 3, 0);
  const neutral = new Float32Array(totalSamples * 3);
  const target = new Float32Array(totalSamples * 3);
  let offset = 0;
  for (let i = 0; i < neutralChunks.length; i++) {
    neutral.set(neutralChunks[i], offset);
    target.set(targetChunks[i], offset);
    offset += neutralChunks[i].length;
  }

  console.log(`  fitting from ${totalSamples} pixel-correspondence samples (${shootFolders.length} shoot(s) contributed)…`);
  const affine = fitGlobalAffine(neutral, target, totalSamples);
  const residualGrid = buildResidualGrid(neutral, target, totalSamples, affine);
  const smoothedGrid = smoothGrid(residualGrid);

  writeLutPng((r, g, b) => {
    const base = evalAffine(affine, r, g, b);
    const correction = sampleGridTrilinear(smoothedGrid, r, g, b);
    return [base[0] + correction[0], base[1] + correction[1], base[2] + correction[2]];
  }, join(outputDir, `${slug}.png`));

  return true;
}

async function main() {
  if (!existsSync(inputDir)) {
    console.error(`Input directory not found: ${inputDir}`);
    console.error("Run the Camera tab's Advanced > LUT Calibration Capture, export the files, and organize them into per-shoot subfolders here first.");
    process.exit(1);
  }
  mkdirSync(outputDir, { recursive: true });

  const shootFolders = findShootFolders(inputDir);
  if (shootFolders.length === 0) {
    console.error(`No shoot folders found under ${inputDir} (looking for subfolders containing calib-neutral.jpg).`);
    process.exit(1);
  }
  console.log(`Found ${shootFolders.length} shoot folder(s):`);
  for (const folder of shootFolders) console.log(`  ${folder}`);

  // Discover which slugs actually have data, from whatever calib-*.jpg files exist.
  const slugs = new Set();
  for (const folder of shootFolders) {
    for (const entry of readdirSync(folder)) {
      const match = /^calib-(?!neutral(?:\.|$))(.+)\.jpg$/.exec(entry);
      if (match) slugs.add(match[1]);
    }
  }

  let derived = 0;
  for (const slug of slugs) {
    console.log(`\nDeriving LUT for "${slug}"…`);
    if (await deriveLutForSim(slug, shootFolders)) derived++;
  }

  console.log(`\nDone — wrote ${derived} LUT(s) to ${outputDir}.`);
  console.log("Next: sanity-check for banding, then validate against a held-out RAF (see the plan doc's Verification section).");
}

main();
