// Shared Hald-CLUT fitting machinery, extracted from derive-luts-from-
// calibration.mjs and invert-luts.mjs (which used to duplicate all of this
// verbatim) so a third script (derive-color-chrome-luts.mjs) doesn't need a
// third copy. Fitting method: global affine (ordinary least squares) plus a
// local IDW-smoothed coarse-grid correction, trilinearly upsampled to the
// final 64-level Hald CLUT — see ~/.claude/plans/indexed-inventing-wren.md
// for the full rationale. Tunable constants (grid resolution, correction
// clamp, IDW radius/epsilon, per-cell sample cap) are passed in by the
// caller rather than hard-coded here, since real-photo fits (sparse, noisy)
// and LUT-inversion fits (dense, exact) want different values.

import sharp from "sharp";
import { PNG } from "pngjs";
import { readFileSync, writeFileSync } from "node:fs";

// Must match src/engine/webgl/shaders/fragmentShader.ts's haldUV()/apply3DLut().
export const LEVELS = 64;
export const N = Math.sqrt(LEVELS); // 8
export const SIDE = LEVELS * N; // 512

export function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}

/**
 * Loads a JPEG, center-trims, resizes to a common square, returns a flat
 * Float32Array of RGB in 0..1 (row-major, no alpha).
 *
 * `.rotate()` with no argument tells sharp to auto-orient using the image's
 * own EXIF Orientation tag before doing anything else — required because a
 * camera JPEG conversion and a locally-decoded/rendered comparison image can
 * represent the same portrait-held shot two different, equally valid ways
 * (sensor-native pixels + an Orientation tag vs. physically pre-rotated
 * pixels) — see derive-luts-from-calibration.mjs's git history for the real
 * bug this caused before this fix. Without auto-orienting both images the
 * same way, pixel (x,y) in one doesn't correspond to the same real-world
 * scene point as pixel (x,y) in the other, corrupting any fit.
 */
export async function loadSamplePixels(path, { sampleSize, trimFraction }) {
  const image = sharp(path).rotate();
  const meta = await image.metadata();
  const swapped = (meta.orientation ?? 1) >= 5; // orientations 5-8 swap width/height
  const width = swapped ? meta.height : meta.width;
  const height = swapped ? meta.width : meta.height;
  const trimX = Math.round(width * trimFraction);
  const trimY = Math.round(height * trimFraction);

  const { data } = await image
    .extract({
      left: trimX,
      top: trimY,
      width: width - trimX * 2,
      height: height - trimY * 2,
    })
    .resize(sampleSize, sampleSize, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixels = new Float32Array((data.length / 3) * 3);
  for (let i = 0; i < data.length; i++) pixels[i] = data[i] / 255;
  return pixels;
}

/** Reads a standard (non-tile-atlas) Hald CLUT PNG, returning a trilinear sampler `(r,g,b) => [r,g,b]` (inputs/outputs 0..1). */
export function loadHaldSampler(path) {
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
export function solve4x4(A, b) {
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
export function fitGlobalAffine(inputs, outputs, sampleCount) {
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

export function evalAffine(coeffs, r, g, b) {
  return [0, 1, 2].map((ch) => coeffs[ch][0] * r + coeffs[ch][1] * g + coeffs[ch][2] * b + coeffs[ch][3]);
}

/** Bins (input -> residual-from-global-affine) samples into a gridLevels^3 grid, median-aggregated per cell. */
export function buildResidualGrid(inputs, outputs, sampleCount, affine, { gridLevels, maxSamplesPerCell }) {
  const cellCount = gridLevels ** 3;
  const buckets = Array.from({ length: cellCount }, () => []);

  for (let i = 0; i < sampleCount; i++) {
    const r = inputs[i * 3];
    const g = inputs[i * 3 + 1];
    const b = inputs[i * 3 + 2];
    const rL = Math.min(gridLevels - 1, Math.round(r * (gridLevels - 1)));
    const gL = Math.min(gridLevels - 1, Math.round(g * (gridLevels - 1)));
    const bL = Math.min(gridLevels - 1, Math.round(b * (gridLevels - 1)));
    const cell = (rL * gridLevels + gL) * gridLevels + bL;
    if (buckets[cell].length >= maxSamplesPerCell) continue;

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
export function smoothGrid(grid, { gridLevels, correctionClamp, idwRadius, idwEpsilon }) {
  const smoothed = new Array(grid.length).fill(null);
  for (let rL = 0; rL < gridLevels; rL++) {
    for (let gL = 0; gL < gridLevels; gL++) {
      for (let bL = 0; bL < gridLevels; bL++) {
        const cell = (rL * gridLevels + gL) * gridLevels + bL;
        let weightSum = 0;
        const acc = [0, 0, 0];
        for (let dr = -idwRadius; dr <= idwRadius; dr++) {
          const r2 = rL + dr;
          if (r2 < 0 || r2 >= gridLevels) continue;
          for (let dg = -idwRadius; dg <= idwRadius; dg++) {
            const g2 = gL + dg;
            if (g2 < 0 || g2 >= gridLevels) continue;
            for (let db = -idwRadius; db <= idwRadius; db++) {
              const b2 = bL + db;
              if (b2 < 0 || b2 >= gridLevels) continue;
              const neighbor = grid[(r2 * gridLevels + g2) * gridLevels + b2];
              if (!neighbor) continue;
              const dist2 = dr * dr + dg * dg + db * db;
              const weight = 1 / (dist2 + idwEpsilon);
              weightSum += weight;
              acc[0] += weight * neighbor[0];
              acc[1] += weight * neighbor[1];
              acc[2] += weight * neighbor[2];
            }
          }
        }
        smoothed[cell] =
          weightSum > 0
            ? acc.map((v) => Math.max(-correctionClamp, Math.min(correctionClamp, v / weightSum)))
            : [0, 0, 0];
      }
    }
  }
  return smoothed;
}

export function sampleGridTrilinear(grid, r, g, b, gridLevels) {
  const pos = [r, g, b].map((v) => clamp01(v) * (gridLevels - 1));
  const lo = pos.map(Math.floor);
  const frac = pos.map((v, i) => v - lo[i]);
  const hi = lo.map((v) => Math.min(gridLevels - 1, v + 1));

  function at(rL, gL, bL) {
    return grid[(rL * gridLevels + gL) * gridLevels + bL];
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

/** Writes a Hald CLUT PNG, sampling `lookup(r, g, b)` (each 0..1) at every one of the LEVELS^3 cells. */
export function writeLutPng(lookup, outPath) {
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

/**
 * Fits a full RGB -> RGB Hald CLUT (affine + IDW-corrected grid) from paired
 * (inputs, outputs) sample arrays. `gridConfig` overrides the defaults below
 * — real-photo fits want a coarser grid (fewer, noisier samples) than a
 * dense/exact LUT-inversion fit does.
 */
export function fitHaldClut(inputs, outputs, sampleCount, gridConfig) {
  const config = {
    gridLevels: 17,
    correctionClamp: 0.14,
    idwRadius: 3,
    idwEpsilon: 0.5,
    maxSamplesPerCell: 500,
    ...gridConfig,
  };
  const affine = fitGlobalAffine(inputs, outputs, sampleCount);
  const residualGrid = buildResidualGrid(inputs, outputs, sampleCount, affine, config);
  const smoothedGrid = smoothGrid(residualGrid, config);

  return function lookup(r, g, b) {
    const base = evalAffine(affine, r, g, b);
    const correction = sampleGridTrilinear(smoothedGrid, r, g, b, config.gridLevels);
    return [base[0] + correction[0], base[1] + correction[1], base[2] + correction[2]];
  };
}
