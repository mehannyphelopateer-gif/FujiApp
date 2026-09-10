#!/usr/bin/env node
// Phase 3 fitting core: a bilateral-grid-style local color correction —
// spatial tiles x luma zones, each bin's per-channel correction predicted
// by ridge regression on LOCAL (per-tile) features, with a spatial
// Laplacian smoothness penalty coupling adjacent tiles. See
// docs/phase3-raw-aware-processor-plan.md for the full design and why
// this replaces Phase 2's scalar-summary-conditioned single curve.
//
// Operates entirely in LINEAR light (both sides converted before any
// comparison — see scripts/lib/phase3-linear-rgb.mjs) on Codex's 16-bit
// browser export (docs/phase3-linear-rgb-format.md) vs. the X RAW Studio
// target (16-bit TIFF preferred, 8-bit JPEG fallback per
// docs/phase3-export-protocol.md).
//
// STATUS as of this writing: only one real browser-side sample exists
// (Shoot 127) and ZERO real X RAW Studio Phase 3 targets exist yet. This
// script's fitting math is smoke-tested against SYNTHETIC data
// (--smoke-test) to verify the algorithm itself is correct before any
// real fit is attempted — do not treat a real-data run as valid until
// both sides' actual Phase 3 exports exist for the full training range.
//
// Usage:
//   node scripts/derive-phase3-spatial-processor.mjs --smoke-test
//   node scripts/derive-phase3-spatial-processor.mjs [input-dir]   (real run, once data exists)

import { readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadFjlrgLinear, loadXrawTargetLinear, resizeLinearRgbTo } from "./lib/phase3-linear-rgb.mjs";

const MIN_SCENES_TO_FIT = 3; // below this, leave-one-out is meaningless (same floor Phase 2 used)
const FIRST_HELD_OUT_SHOOT = 387; // Shoots 387-426 are the locked final-test range — never load these here
const LAST_HELD_OUT_SHOOT = 426;

const SPATIAL_GRID_SIZE = 4; // 4x4 = 16 tiles, per the architecture doc's starting point
const REGRESSION_GATE_TOLERANCE = 2.0; // same units as the MAE reported below (0-255 equivalent)

// Perceptual luma-zone edges, expressed as LINEAR-light values. Phase 2's
// zones (0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9) were defined in sRGB-gamma
// space; converting them through the sRGB EOTF preserves the same
// perceptual meaning (same "shadow/low/mid/high/highlight" boundaries a
// person would perceive) while operating on the linear numbers this
// pipeline actually works with — linear and gamma-encoded values have very
// different numeric distributions (most of the useful tonal range sits
// close to 0 in linear light), so reusing the gamma-space numbers directly
// would bin almost everything into one zone.
function srgbGammaToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}
const PERCEPTUAL_EDGES = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9];
const LUMA_ZONE_EDGES = PERCEPTUAL_EDGES.map(srgbGammaToLinear);

export function luma(r, g, b) { return 0.2126 * r + 0.7152 * g + 0.0722 * b; }
export function zoneIndex(l) { for (let i = LUMA_ZONE_EDGES.length - 1; i >= 0; i--) if (l >= LUMA_ZONE_EDGES[i]) return i; return 0; }
export function tileIndex(x, y, width, height) {
  const tx = Math.min(SPATIAL_GRID_SIZE - 1, Math.floor((x / width) * SPATIAL_GRID_SIZE));
  const ty = Math.min(SPATIAL_GRID_SIZE - 1, Math.floor((y / height) * SPATIAL_GRID_SIZE));
  return ty * SPATIAL_GRID_SIZE + tx;
}
function median(values) {
  if (!values.length) return null;
  const s = values.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}
/** MAE reported on the same 0-255 scale as every Phase 1/2 script, even
 * though this pipeline operates on 0..1 linear values internally — keeps
 * results comparable across phases. */
export function maeLinear255(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return (s / a.length) * 255;
}

/** For one scene: bins every pixel into (tile, zone), computing per-bin
 * median (target - browser) delta and per-bin LOCAL features (mean linear
 * luma, near-black/near-white fraction within that tile) — the local
 * analog of Phase 2's whole-scene p99/nearBlackFraction. Assumes browser
 * and target are already resampled to the same dimensions by the caller. */
export function computeSceneBins(browser, target, width, height) {
  const binCount = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE * LUMA_ZONE_EDGES.length;
  const bins = Array.from({ length: binCount }, () => ({ r: [], g: [], b: [], lumas: [] }));

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 3;
      const br = browser[offset], bg = browser[offset + 1], bb = browser[offset + 2];
      const tr = target[offset], tg = target[offset + 1], tb = target[offset + 2];
      const l = luma(br, bg, bb);
      const tile = tileIndex(x, y, width, height);
      const zone = zoneIndex(l);
      const bin = bins[tile * LUMA_ZONE_EDGES.length + zone];
      bin.r.push(tr - br); bin.g.push(tg - bg); bin.b.push(tb - bb);
      bin.lumas.push(l);
    }
  }

  return bins.map((bin) => {
    if (bin.r.length === 0) return null;
    const nearBlack = bin.lumas.filter((l) => l < 0.02).length / bin.lumas.length;
    const nearWhite = bin.lumas.filter((l) => l > 0.5).length / bin.lumas.length; // 0.5 linear ~= very bright given sRGB EOTF compression
    const meanLuma = bin.lumas.reduce((a, b) => a + b, 0) / bin.lumas.length;
    return {
      delta: [median(bin.r), median(bin.g), median(bin.b)],
      features: [1, meanLuma, nearBlack, nearWhite],
      sampleCount: bin.r.length,
    };
  });
}

function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row) => row.slice());
  const y = b.slice();
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
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

/** Adjacency (4-connected) among the SPATIAL_GRID_SIZE^2 tiles, used by
 * the joint Laplacian-regularized solve below. */
function tileNeighbors(tile) {
  const x = tile % SPATIAL_GRID_SIZE, y = Math.floor(tile / SPATIAL_GRID_SIZE);
  const out = [];
  if (x > 0) out.push(tile - 1);
  if (x < SPATIAL_GRID_SIZE - 1) out.push(tile + 1);
  if (y > 0) out.push(tile - SPATIAL_GRID_SIZE);
  if (y < SPATIAL_GRID_SIZE - 1) out.push(tile + SPATIAL_GRID_SIZE);
  return out;
}

/**
 * Joint ridge fit for ONE luma zone across all spatial tiles at once,
 * coupled by a Laplacian smoothness penalty between adjacent tiles.
 * Solving per-tile independently (Phase 2's approach) can't express "this
 * tile should look like its neighbors" — solving all tiles' coefficients
 * together, for a fixed zone, is what actually lets that prior do
 * anything. Returns one P-length coefficient vector per tile per channel.
 *
 * trainingByTile[tile] = { features: number[][], deltas: number[][] } |
 * null (no samples for that tile across the whole training set — that
 * tile's block stays all-zero via the ridge term, same "unseen category
 * gets zero effect" mechanic used for Phase 2's WB-preset pooling).
 */
export function fitZoneJoint(trainingByTile, numTiles, numFeatures, ridgeLambda, smoothnessLambda) {
  const blockSize = numFeatures;
  const totalSize = numTiles * blockSize;
  const results = [0, 1, 2].map(() => ({ XtX: Array.from({ length: totalSize }, () => new Array(totalSize).fill(0)), XtY: new Array(totalSize).fill(0) }));

  for (let tile = 0; tile < numTiles; tile++) {
    const data = trainingByTile[tile];
    const base = tile * blockSize;
    if (data) {
      for (let i = 0; i < data.features.length; i++) {
        const x = data.features[i];
        for (let a = 0; a < blockSize; a++) {
          for (let c = 0; c < blockSize; c++) {
            for (let ch = 0; ch < 3; ch++) results[ch].XtX[base + a][base + c] += x[a] * x[c];
          }
          for (let ch = 0; ch < 3; ch++) results[ch].XtY[base + a] += x[a] * data.deltas[i][ch];
        }
      }
    }
    for (let a = 0; a < blockSize; a++) for (let ch = 0; ch < 3; ch++) results[ch].XtX[base + a][base + a] += ridgeLambda;

    // Laplacian smoothness: for each neighbor pair, add
    // mu*(w_tile - w_neighbor)^2 to the objective, which contributes
    // +mu to both self-blocks' diagonals and -mu to the cross-block terms.
    for (const neighbor of tileNeighbors(tile)) {
      if (neighbor <= tile) continue; // add each undirected pair once
      const neighborBase = neighbor * blockSize;
      for (let a = 0; a < blockSize; a++) {
        for (let ch = 0; ch < 3; ch++) {
          results[ch].XtX[base + a][base + a] += smoothnessLambda;
          results[ch].XtX[neighborBase + a][neighborBase + a] += smoothnessLambda;
          results[ch].XtX[base + a][neighborBase + a] -= smoothnessLambda;
          results[ch].XtX[neighborBase + a][base + a] -= smoothnessLambda;
        }
      }
    }
  }

  const solved = results.map((r) => solveLinearSystem(r.XtX, r.XtY));
  const perTile = [];
  for (let tile = 0; tile < numTiles; tile++) {
    const base = tile * blockSize;
    perTile.push(solved.map((channelSolution) => channelSolution.slice(base, base + blockSize)));
  }
  return perTile; // perTile[tile] = [ [wr...], [wg...], [wb...] ]
}

export function applyModel(model, browser, width, height) {
  const out = new Float32Array(browser.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 3;
      const r = browser[offset], g = browser[offset + 1], b = browser[offset + 2];
      const l = luma(r, g, b);
      const tile = tileIndex(x, y, width, height);
      const zone = zoneIndex(l);
      const coeffs = model[zone][tile]; // [ [wr...], [wg...], [wb...] ]
      const featureVector = [1, l, l < 0.02 ? 1 : 0, l > 0.5 ? 1 : 0]; // per-pixel approx of the per-bin training features
      const dr = coeffs[0].reduce((s, w, i) => s + w * featureVector[i], 0);
      const dg = coeffs[1].reduce((s, w, i) => s + w * featureVector[i], 0);
      const db = coeffs[2].reduce((s, w, i) => s + w * featureVector[i], 0);
      out[offset] = Math.min(1, Math.max(0, r + dr));
      out[offset + 1] = Math.min(1, Math.max(0, g + dg));
      out[offset + 2] = Math.min(1, Math.max(0, b + db));
    }
  }
  return out;
}

// --- Smoke test: synthetic data, verifies the fitting math is correct
// (recovers a known relationship) before any real fit is attempted. ---
async function smokeTest() {
  console.log("Running Phase 3 pipeline smoke test on SYNTHETIC data (no real X RAW Studio targets exist yet).\n");
  const width = 64, height = 64;
  const numTiles = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
  const numFeatures = 4;

  // Synthesize N "scenes": random browser linear RGB, target = browser +
  // a KNOWN, tile-varying lift (stronger lift in tile 0, weaker elsewhere)
  // plus small noise — if the fit recovers something close to that known
  // pattern, the algorithm (binning, joint solve, Laplacian coupling,
  // application) is working correctly end to end.
  const N_SCENES = 12;
  const scenes = [];
  for (let s = 0; s < N_SCENES; s++) {
    const browser = new Float32Array(width * height * 3);
    const target = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const offset = (y * width + x) * 3;
        const base = 0.05 + Math.random() * 0.3;
        const tile = tileIndex(x, y, width, height);
        const knownLift = tile === 0 ? 0.05 : 0.0; // tile 0 needs a real, detectable lift; others don't
        for (let ch = 0; ch < 3; ch++) {
          const v = Math.max(0, Math.min(1, base + (Math.random() - 0.5) * 0.02));
          browser[offset + ch] = v;
          target[offset + ch] = Math.max(0, Math.min(1, v + knownLift + (Math.random() - 0.5) * 0.005));
        }
      }
    }
    scenes.push({ name: `synthetic-${s}`, browser, target, width, height });
  }

  for (const scene of scenes) scene.bins = computeSceneBins(scene.browser, scene.target, scene.width, scene.height);

  console.log("Fitting joint per-zone models (with Laplacian smoothness) via leave-one-out...");
  const RIDGE_LAMBDA = 0.1, SMOOTHNESS_LAMBDA = 1;
  let regressions = 0;
  const results = [];
  for (let i = 0; i < scenes.length; i++) {
    const held = scenes[i];
    const training = scenes.filter((_, j) => j !== i);
    const model = [];
    for (let zone = 0; zone < LUMA_ZONE_EDGES.length; zone++) {
      const byTile = Array.from({ length: numTiles }, (_, tile) => {
        const pooled = { features: [], deltas: [] };
        for (const s of training) {
          const bin = s.bins[tile * LUMA_ZONE_EDGES.length + zone];
          if (bin) { pooled.features.push(bin.features); pooled.deltas.push(bin.delta); }
        }
        return pooled.features.length > 0 ? pooled : null;
      });
      model.push(fitZoneJoint(byTile, numTiles, numFeatures, RIDGE_LAMBDA, SMOOTHNESS_LAMBDA));
    }
    const before = maeLinear255(held.browser, held.target);
    const corrected = applyModel(model, held.browser, held.width, held.height);
    const after = maeLinear255(corrected, held.target);
    if (after > before + REGRESSION_GATE_TOLERANCE) regressions++;
    results.push({ name: held.name, before, after });
  }

  for (const r of results) console.log(`  ${r.name}: MAE ${r.before.toFixed(2)} -> ${r.after.toFixed(2)}`);
  console.log(`\n${regressions}/${results.length} regressed. On synthetic data with a real, learnable tile-0-only lift, this should be 0 with the correction concentrated in tile 0.`);
  console.log("\nSmoke test complete. This validates the ALGORITHM only — no real Phase 3 data was used.");
}

/** Finds a shoot's X RAW Studio target file, preferring the 16-bit TIFF
 * per the locked export protocol, falling back to a JPEG only if no TIFF
 * exists in that folder. */
function findTargetPath(folder) {
  const tiffPath = join(folder, "xraw-phase3.tiff");
  if (existsSync(tiffPath)) return tiffPath;
  const jpgPath = join(folder, "xraw-phase3.jpg");
  if (existsSync(jpgPath)) return jpgPath;
  return null;
}

async function loadRealScenes(inputDir) {
  const shootDirs = readdirSync(inputDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && /^Shoot \d+$/.test(d.name))
    .map((d) => ({ name: d.name, num: parseInt(d.name.replace("Shoot ", ""), 10) }))
    .filter((d) => !(d.num >= FIRST_HELD_OUT_SHOOT && d.num <= LAST_HELD_OUT_SHOOT)) // hard-enforced lock, not just convention
    .sort((a, b) => a.num - b.num);

  const scenes = [];
  for (const { name } of shootDirs) {
    const folder = join(inputDir, name);
    const fjlrgPath = join(folder, "browser-phase3-linear.fjlrg");
    const targetPath = findTargetPath(folder);
    if (!existsSync(fjlrgPath) || !targetPath) continue; // incomplete pair — skip, don't error, more scenes arrive incrementally

    const browserImg = loadFjlrgLinear(fjlrgPath);
    const targetFull = await loadXrawTargetLinear(targetPath);
    const target = resizeLinearRgbTo(targetFull, browserImg.width, browserImg.height);
    scenes.push({ name, browser: browserImg.data, target: target.data, width: browserImg.width, height: browserImg.height });
  }
  return scenes;
}

async function main() {
  if (process.argv.includes("--smoke-test")) {
    await smokeTest();
    return;
  }

  const inputDir = process.argv[2] ?? join(new URL(".", import.meta.url).pathname, "..", "calibration-input");
  console.log(`Loading real Phase 3 scene pairs from ${inputDir} (Shoots ${FIRST_HELD_OUT_SHOOT}-${LAST_HELD_OUT_SHOOT} excluded — locked final-test range)...`);
  const scenes = await loadRealScenes(inputDir);
  console.log(`Found ${scenes.length} complete real scene pair(s): ${scenes.map((s) => s.name).join(", ") || "(none)"}`);

  if (scenes.length < MIN_SCENES_TO_FIT) {
    console.error(
      `\nOnly ${scenes.length} complete pair(s) available, need at least ${MIN_SCENES_TO_FIT} for leave-one-out to ` +
        "mean anything. Not fitting. Export more browser-phase3-linear.fjlrg / xraw-phase3.tiff pairs and rerun — " +
        "see docs/phase3-export-protocol.md.",
    );
    process.exit(1);
  }

  for (const scene of scenes) scene.bins = computeSceneBins(scene.browser, scene.target, scene.width, scene.height);

  console.log("\nFitting joint per-zone models (with Laplacian smoothness) via leave-one-out...");
  const RIDGE_LAMBDA = 0.1, SMOOTHNESS_LAMBDA = 1; // starting values carried from the smoke test — not yet cross-validated against real data
  const numTiles = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
  let regressions = 0;
  const results = [];
  for (let i = 0; i < scenes.length; i++) {
    const held = scenes[i];
    const training = scenes.filter((_, j) => j !== i);
    const model = [];
    for (let zone = 0; zone < LUMA_ZONE_EDGES.length; zone++) {
      const byTile = Array.from({ length: numTiles }, (_, tile) => {
        const pooled = { features: [], deltas: [] };
        for (const s of training) {
          const bin = s.bins[tile * LUMA_ZONE_EDGES.length + zone];
          if (bin) { pooled.features.push(bin.features); pooled.deltas.push(bin.delta); }
        }
        return pooled.features.length > 0 ? pooled : null;
      });
      model.push(fitZoneJoint(byTile, numTiles, 4, RIDGE_LAMBDA, SMOOTHNESS_LAMBDA));
    }
    const before = maeLinear255(held.browser, held.target);
    const corrected = applyModel(model, held.browser, held.width, held.height);
    const after = maeLinear255(corrected, held.target);
    if (after > before + REGRESSION_GATE_TOLERANCE) regressions++;
    results.push({ name: held.name, before, after });
    console.log(`  ${held.name}: MAE ${before.toFixed(2)} -> ${after.toFixed(2)}${after > before + REGRESSION_GATE_TOLERANCE ? "  [REGRESSION]" : ""}`);
  }

  console.log(`\n${regressions}/${results.length} scenes regressed beyond tolerance (${REGRESSION_GATE_TOLERANCE} MAE).`);
  if (regressions > 0) {
    console.error("\nGate failed. Not proceeding to a final fit or the held-out test.");
    process.exit(1);
  }
  console.log("\nGate passed on the current training pool. Still not approved to ship — see docs/phase3-raw-aware-processor-plan.md for the full acceptance sequence (validation-set check, then the one-shot held-out test).");
}

// Only run as a CLI entry point — importing this module for its exported
// functions (e.g. to test them directly) must not trigger the CLI logic.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main();
}
