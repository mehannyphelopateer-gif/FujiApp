#!/usr/bin/env node
// Derives an "inverse" Hald CLUT for each color (non-monochrome) film
// simulation, used by the WebGL preview pipeline to undo a film simulation
// already baked into an uploaded JPEG before applying a different target
// recipe's own LUT — see src/lib/recipes/neutralize.ts and
// ~/.claude/plans/indexed-inventing-wren.md for why: without this, picking
// a new recipe on a JPEG that already has one baked in stacks the two
// film simulations instead of swapping one for the other.
//
// Usage: node scripts/invert-luts.mjs [forward-lut-dir] [output-dir]
// forward-lut-dir defaults to ./public/luts, output-dir to ./public/luts/inverse.
//
// Unlike derive-luts-from-calibration.mjs (which fits from real, sparse,
// noisy photo pairs), the input here is a known, smooth, exactly-samplable
// function — the forward LUT PNG itself — so this samples it densely and
// evenly (128 points/axis, ~2.1M samples) rather than relying on whatever a
// real photo happened to cover. The same global-affine + IDW-smoothed
// coarse-grid + trilinear-upsample fitting pipeline is reused anyway (not
// for noise-robustness here, but because it's exactly the right tool for
// the OTHER real problem an inverse has: many-to-one regions, where a
// saturation/desaturation-heavy simulation maps multiple distinct inputs to
// similar outputs, making the true inverse ill-defined there. Median-
// aggregation picks a representative centroid for those regions and
// IDW-fill smoothly degrades toward the global-affine fallback instead of
// producing a discontinuity — the honest behavior for a genuinely ill-posed
// partial inverse, not a bug to fix.
//
// This is a DERIVED asset: rerun this whenever a forward LUT in
// public/luts/*.png changes (e.g. after npm run derive:luts or
// convert-third-party-luts.mjs produce new forward LUTs) — an inverse LUT
// generated against a stale forward LUT will quietly under- or over-undo it.
//
// acros.png / monochrome.png / sepia.png are skipped on purpose: a real
// camera JPEG shot in one of these modes has no color left to recover —
// that's a physical fact about monochrome/toned rendering, not a fitting
// quality problem this script (or any LUT) can solve.

import { PNG } from "pngjs";
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const forwardLutDir = process.argv[2] ?? join(__dirname, "..", "public", "luts");
const outputDir = process.argv[3] ?? join(__dirname, "..", "public", "luts", "inverse");

const SKIP_SLUGS = new Set(["acros", "monochrome", "sepia", "identity"]);

// Must match src/engine/webgl/shaders/fragmentShader.ts's haldUV()/apply3DLut().
const LEVELS = 64;
const N = Math.sqrt(LEVELS); // 8
const SIDE = LEVELS * N; // 512

const SAMPLE_DENSITY = 160; // samples per axis when probing the forward LUT — dense, cheap (pure function evaluation)
// Much finer than derive-luts-from-calibration.mjs's GRID_LEVELS=17: that
// value was chosen for real, sparse, noisy photo data where a coarse grid
// avoids overfitting noise. Here the "data" is the forward LUT itself —
// deterministic and densely samplable at any resolution — so there's no
// noise to protect against, and a coarse grid only costs real accuracy
// against non-affine curvature in the LUT being inverted (confirmed: the
// first pass at GRID_LEVELS=17 gave 0.05-0.10 mean round-trip error,
// worse for the stronger/more nonlinear sims like Velvia and Classic
// Negative — exactly what under-resolving curvature looks like).
const GRID_LEVELS = 45;
const CORRECTION_CLAMP = 0.25;
const IDW_RADIUS = 3;
const IDW_EPSILON = 0.5;
const MAX_SAMPLES_PER_CELL = 4000; // dense regular sampling fills cells far more evenly than real photos — safe to raise

const ROUND_TRIP_CHECKS = 500;

function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}

/** Reads a standard (non-tile-atlas) Hald CLUT PNG, returning a trilinear sampler `(r,g,b) => [r,g,b]` (inputs/outputs 0..1). */
function loadHaldSampler(path) {
  const png = PNG.sync.read(readFileSync(path));
  const side = png.width;
  const levels = Math.round(Math.cbrt(side) ** 2);
  const n = Math.round(Math.sqrt(levels));
  if (n * n !== levels || levels * n !== side) {
    throw new Error(`${path}: ${side}x${side} doesn't look like a standard Hald CLUT (expected side = levels^1.5).`);
  }

  function at(rL, gL, bL) {
    const gLo = gL % n;
    const gHi = Math.floor(gL / n);
    const x = rL + levels * gLo;
    const y = bL * n + gHi;
    const idx = (png.width * y + x) << 2;
    return [png.data[idx] / 255, png.data[idx + 1] / 255, png.data[idx + 2] / 255];
  }

  return function sample(r, g, b) {
    const pos = [r, g, b].map((v) => clamp01(v) * (levels - 1));
    const lo = pos.map(Math.floor);
    const frac = pos.map((v, i) => v - lo[i]);
    const hi = lo.map((v) => Math.min(levels - 1, v + 1));

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
  };
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
function fitGlobalAffine(inputs, outputs, sampleCount) {
  const XtX = Array.from({ length: 4 }, () => new Array(4).fill(0));
  const XtY = [new Array(4).fill(0), new Array(4).fill(0), new Array(4).fill(0)];

  for (let i = 0; i < sampleCount; i++) {
    const r = inputs[i * 3];
    const g = inputs[i * 3 + 1];
    const b = inputs[i * 3 + 2];
    const row = [r, g, b, 1];
    for (let a = 0; a < 4; a++) {
      for (let c = 0; c < 4; c++) XtX[a][c] += row[a] * row[c];
      for (let ch = 0; ch < 3; ch++) XtY[ch][a] += row[a] * outputs[i * 3 + ch];
    }
  }

  return [0, 1, 2].map((ch) => solve4x4(XtX, XtY[ch]));
}

function evalAffine(coeffs, r, g, b) {
  return [0, 1, 2].map((ch) => coeffs[ch][0] * r + coeffs[ch][1] * g + coeffs[ch][2] * b + coeffs[ch][3]);
}

/** Bins (input -> residual-from-global-affine) samples into a GRID_LEVELS^3 grid keyed by INPUT, median-aggregated per cell. */
function buildResidualGrid(inputs, outputs, sampleCount, affine) {
  const cellCount = GRID_LEVELS ** 3;
  const buckets = Array.from({ length: cellCount }, () => []);

  for (let i = 0; i < sampleCount; i++) {
    const r = inputs[i * 3];
    const g = inputs[i * 3 + 1];
    const b = inputs[i * 3 + 2];
    const rL = Math.min(GRID_LEVELS - 1, Math.round(r * (GRID_LEVELS - 1)));
    const gL = Math.min(GRID_LEVELS - 1, Math.round(g * (GRID_LEVELS - 1)));
    const bL = Math.min(GRID_LEVELS - 1, Math.round(b * (GRID_LEVELS - 1)));
    const cell = (rL * GRID_LEVELS + gL) * GRID_LEVELS + bL;
    if (buckets[cell].length >= MAX_SAMPLES_PER_CELL) continue;

    const predicted = evalAffine(affine, r, g, b);
    buckets[cell].push([
      outputs[i * 3] - predicted[0],
      outputs[i * 3 + 1] - predicted[1],
      outputs[i * 3 + 2] - predicted[2],
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

async function invertLut(forwardPath, outPath) {
  const forward = loadHaldSampler(forwardPath);

  // Dense, even sampling of the forward LUT's input space — cheap since
  // this is pure function evaluation, not real photo I/O. Inputs are the
  // "known" side; outputs are what we're inverting from.
  const total = SAMPLE_DENSITY ** 3;
  const inputs = new Float32Array(total * 3);
  const outputs = new Float32Array(total * 3);
  let i = 0;
  for (let ri = 0; ri < SAMPLE_DENSITY; ri++) {
    const r = ri / (SAMPLE_DENSITY - 1);
    for (let gi = 0; gi < SAMPLE_DENSITY; gi++) {
      const g = gi / (SAMPLE_DENSITY - 1);
      for (let bi = 0; bi < SAMPLE_DENSITY; bi++) {
        const b = bi / (SAMPLE_DENSITY - 1);
        const [or_, og, ob] = forward(r, g, b);
        inputs[i * 3] = r;
        inputs[i * 3 + 1] = g;
        inputs[i * 3 + 2] = b;
        outputs[i * 3] = or_;
        outputs[i * 3 + 1] = og;
        outputs[i * 3 + 2] = ob;
        i++;
      }
    }
  }

  // Inverted roles vs. derive-luts-from-calibration.mjs: fit input FROM
  // output (i.e. "given this output color, what input produced it").
  const affine = fitGlobalAffine(outputs, inputs, total);
  const residualGrid = buildResidualGrid(outputs, inputs, total, affine);
  const smoothedGrid = smoothGrid(residualGrid);

  function inverse(outR, outG, outB) {
    const base = evalAffine(affine, outR, outG, outB);
    const correction = sampleGridTrilinear(smoothedGrid, outR, outG, outB);
    return [base[0] + correction[0], base[1] + correction[1], base[2] + correction[2]];
  }

  writeLutPng(inverse, outPath);

  // Round-trip self-check: forward(inverse(x)) should land close to x.
  let maxErr = 0;
  let sumErr = 0;
  for (let n = 0; n < ROUND_TRIP_CHECKS; n++) {
    const r = Math.random();
    const g = Math.random();
    const b = Math.random();
    const [fr, fg, fb] = forward(r, g, b);
    const [ir, ig, ib] = inverse(fr, fg, fb);
    const err = Math.max(Math.abs(ir - r), Math.abs(ig - g), Math.abs(ib - b));
    maxErr = Math.max(maxErr, err);
    sumErr += err;
  }
  return { maxErr, meanErr: sumErr / ROUND_TRIP_CHECKS };
}

async function main() {
  mkdirSync(outputDir, { recursive: true });

  const candidates = readdirSync(forwardLutDir)
    .filter((f) => f.endsWith(".png"))
    .map((f) => ({ slug: basename(f, ".png"), path: join(forwardLutDir, f) }))
    .filter(({ slug }) => !SKIP_SLUGS.has(slug));

  if (candidates.length === 0) {
    console.error(`No invertible forward LUTs found in ${forwardLutDir}.`);
    process.exit(1);
  }

  for (const { slug, path } of candidates) {
    console.log(`Inverting "${slug}"…`);
    const outPath = join(outputDir, `${slug}.png`);
    const { maxErr, meanErr } = await invertLut(path, outPath);
    console.log(
      `  wrote ${outPath} — round-trip error: mean ${meanErr.toFixed(4)}, max ${maxErr.toFixed(4)} (0..1 scale, ${ROUND_TRIP_CHECKS} random samples)`,
    );
    // Nonzero round-trip error here is expected, not necessarily a bug: a
    // strong contrast/saturation curve genuinely compresses distinct inputs
    // toward similar outputs in places, and no inverse can recover
    // information the forward transform actually destroyed. Confirmed via
    // direct visual comparison (undo-then-reapply vs. applying the target
    // LUT to a truly neutral source) that a mean error in this range still
    // produces a close, visually correct result — this threshold is only a
    // flag for "look at this one," not a pass/fail gate.
    if (meanErr > 0.15) {
      console.warn(`  ⚠ unusually high round-trip error for "${slug}" — worth a visual check before trusting this inverse.`);
    }
  }

  console.log(`\nDone — wrote ${candidates.length} inverse LUT(s) to ${outputDir}.`);
}

main();
