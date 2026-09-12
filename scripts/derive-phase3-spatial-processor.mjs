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

import { readdirSync, existsSync, readFileSync } from "node:fs";
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
export const LUMA_ZONE_EDGES = PERCEPTUAL_EDGES.map(srgbGammaToLinear);

export function luma(r, g, b) { return 0.2126 * r + 0.7152 * g + 0.0722 * b; }

// Local color-cast features, the same lever that was Phase 2's single
// biggest win (CFA per-channel ratios) — added after diagnosing why the
// hyperparameter grid search couldn't fix 4 stubbornly-regressing scenes
// (Shoots 180, 229, 319, 383): the per-bin prediction error for those
// scenes was 100-260 (of 255) even where their meanLuma/nearWhite features
// sat well INSIDE the training range, meaning the error wasn't
// extrapolation — it was that luma alone can't distinguish, say, a warm
// sunset highlight from a neutral-white one, so every near-white pixel in
// a (tile, zone) bin got the same single learned correction regardless of
// its actual color. An eps-stabilized ratio to green, clamped, keeps rare
// near-zero-green pixels from producing huge unstable feature values.
const CHROMA_EPS = 1e-3;
const CHROMA_CLAMP = 3;
function chromaFeatures(r, g, b) {
  const rg = Math.min(CHROMA_CLAMP, Math.max(-CHROMA_CLAMP, (r + CHROMA_EPS) / (g + CHROMA_EPS) - 1));
  const bg = Math.min(CHROMA_CLAMP, Math.max(-CHROMA_CLAMP, (b + CHROMA_EPS) / (g + CHROMA_EPS) - 1));
  return [rg, bg];
}
export const BIN_FEATURE_COUNT = 6; // [1, meanLuma, nearBlack, nearWhite, rgRatio, bgRatio]

// RAW-domain highlight-TOPOLOGY features (Codex's proposal, see
// src/lib/raw/rawService.ts's computeHighlightTopology), appended after
// BIN_FEATURE_COUNT's 6 when a scene has a raw-highlight-topology.json
// sidecar and --with-topology is passed. Unlike the 6 base features (which
// vary per pixel/bin), these are mostly whole-SCENE constants — the same
// value repeated across every bin of that scene — except the per-tile
// near-saturation fraction, which varies by tile but not by luma zone.
// This is exactly the signal the base features can't express: a scalar
// near-white FRACTION can't tell a broad dominant glow (one huge connected
// blob) apart from scattered small specular highlights that sum to the
// same total area — highlightConcentration is built specifically to.
export const TOPOLOGY_FEATURE_COUNT = 8;
function topologyFeatureVector(topology, tile) {
  if (!topology) return new Array(TOPOLOGY_FEATURE_COUNT).fill(0);
  const chroma = topology.highlightChroma ?? { rg: 0, bg: 0 };
  return [
    topology.nearSaturationFraction.p90,
    topology.nearSaturationFraction.p95,
    topology.nearSaturationFraction.p99,
    topology.perTileNearSaturationFraction99[tile] ?? 0,
    topology.largestRegionAreaFraction,
    topology.highlightConcentration,
    chroma.rg,
    chroma.bg,
  ];
}
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
export function computeSceneBins(browser, target, width, height, topology = null) {
  const binCount = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE * LUMA_ZONE_EDGES.length;
  const bins = Array.from({ length: binCount }, () => ({ r: [], g: [], b: [], lumas: [], rg: [], bg: [] }));

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
      const [rgRatio, bgRatio] = chromaFeatures(br, bg, bb);
      bin.rg.push(rgRatio); bin.bg.push(bgRatio);
    }
  }

  return bins.map((bin, index) => {
    if (bin.r.length === 0) return null;
    const nearBlack = bin.lumas.filter((l) => l < 0.02).length / bin.lumas.length;
    const nearWhite = bin.lumas.filter((l) => l > 0.5).length / bin.lumas.length; // 0.5 linear ~= very bright given sRGB EOTF compression
    const meanLuma = bin.lumas.reduce((a, b) => a + b, 0) / bin.lumas.length;
    // Median, not mean, for the chroma ratios — same reasoning as the
    // delta itself: robust against the occasional near-zero-green outlier
    // pixel the clamp doesn't fully tame.
    const tile = Math.floor(index / LUMA_ZONE_EDGES.length);
    const baseFeatures = [1, meanLuma, nearBlack, nearWhite, median(bin.rg), median(bin.bg)];
    return {
      delta: [median(bin.r), median(bin.g), median(bin.b)],
      features: topology ? baseFeatures.concat(topologyFeatureVector(topology, tile)) : baseFeatures,
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

export function applyModel(model, browser, width, height, topology = null) {
  const out = new Float32Array(browser.length);
  const numTiles = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
  // Topology features are constant per tile (mostly whole-scene constants,
  // plus one per-tile value) — precompute once per tile rather than
  // recomputing per pixel.
  const topologyByTile = topology ? Array.from({ length: numTiles }, (_, tile) => topologyFeatureVector(topology, tile)) : null;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 3;
      const r = browser[offset], g = browser[offset + 1], b = browser[offset + 2];
      const l = luma(r, g, b);
      const tile = tileIndex(x, y, width, height);
      const zone = zoneIndex(l);
      const coeffs = model[zone][tile]; // [ [wr...], [wg...], [wb...] ]
      const [rgRatio, bgRatio] = chromaFeatures(r, g, b);
      const baseFeatures = [1, l, l < 0.02 ? 1 : 0, l > 0.5 ? 1 : 0, rgRatio, bgRatio]; // per-pixel approx of the per-bin training features
      const featureVector = topologyByTile ? baseFeatures.concat(topologyByTile[tile]) : baseFeatures;
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
  const numFeatures = BIN_FEATURE_COUNT;

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

export async function loadRealScenes(inputDir) {
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
    const topologyPath = join(folder, "raw-highlight-topology.json");
    const topology = existsSync(topologyPath) ? JSON.parse(readFileSync(topologyPath, "utf8")) : null;
    scenes.push({ name, browser: browserImg.data, target: target.data, width: browserImg.width, height: browserImg.height, topology });
  }
  return scenes;
}

async function main() {
  if (process.argv.includes("--smoke-test")) {
    await smokeTest();
    return;
  }

  const positionalArgs = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
  const inputDir = positionalArgs[0] ?? join(new URL(".", import.meta.url).pathname, "..", "calibration-input");
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

  // Manifest-driven training/validation split (docs/phase3-raw-aware-processor-plan.md):
  // hyperparameter selection and LOO fitting must only ever see the 220
  // training scenes — the 40 validation scenes exist purely as a genuine
  // out-of-sample check on the model FROZEN from that search, never as
  // material the search itself can react to. Pooling all 260 into one LOO
  // set (the prior behavior) let the "search" implicitly overfit to the
  // validation scenes too, since every scene took a turn being held out
  // during hyperparameter selection.
  const manifestPath = join(inputDir, "phase3-corpus-manifest.json");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const trainShootNames = new Set(manifest.trainShoots.map((entry) => entry.shoot));
  const validationShootNames = new Set(manifest.validationShoots.map((entry) => entry.shoot));
  const trainingScenes = scenes.filter((s) => trainShootNames.has(s.name));
  const validationScenes = scenes.filter((s) => validationShootNames.has(s.name));
  const unrecognized = scenes.filter((s) => !trainShootNames.has(s.name) && !validationShootNames.has(s.name));
  if (unrecognized.length > 0) {
    console.error(`\nFound scene pair(s) not listed in the manifest's train or validation sets: ${unrecognized.map((s) => s.name).join(", ")}. Refusing to guess which tier they belong to.`);
    process.exit(1);
  }
  console.log(`Manifest split: ${trainingScenes.length} training scene(s), ${validationScenes.length} validation scene(s).`);

  // Group-aware LOO: near-duplicate frames from the same shooting sequence
  // (e.g. 13 gilded-palace-ceiling photos, all similar lighting/color) must
  // not get to "help" each other during leave-one-out — if a palace scene's
  // held-out fold still has 12 other palace scenes in its training pool,
  // LOO would look artificially good on that category without the model
  // actually having learned anything generalizable, just near-duplicate
  // memorization. Default group = the scene's own name (singleton), which
  // reproduces plain LOO exactly for every scene not manually grouped in
  // the manifest.
  const groupOf = new Map(trainingScenes.map((s) => [s.name, s.name]));
  for (const entry of manifest.trainShoots) {
    if (entry.group) groupOf.set(entry.shoot, entry.group);
  }
  const groupMembers = new Map();
  for (const s of trainingScenes) {
    const g = groupOf.get(s.name);
    if (!groupMembers.has(g)) groupMembers.set(g, []);
    groupMembers.get(g).push(s.name);
  }
  const namedGroups = [...groupMembers.entries()].filter(([g, members]) => members.length > 1);
  for (const [group, members] of namedGroups) {
    console.log(`Grouped LOO: "${group}" (${members.length} scenes, held out together): ${members.join(", ")}`);
  }

  if (trainingScenes.length < MIN_SCENES_TO_FIT) {
    console.error(`\nOnly ${trainingScenes.length} training pair(s) available, need at least ${MIN_SCENES_TO_FIT}. Not fitting.`);
    process.exit(1);
  }

  // Codex's proposed RAW-domain highlight-topology feature family
  // (src/lib/raw/rawService.ts's computeHighlightTopology), gated behind
  // an explicit flag so the run can be A/B compared against the baseline
  // per Codex's acceptance criterion: keep it only if it reduces the
  // regression count without creating new regressions anywhere in the
  // 230-scene pool or the palace group test.
  const withTopology = process.argv.includes("--with-topology");
  if (withTopology) {
    const missingTopology = scenes.filter((s) => !s.topology);
    if (missingTopology.length > 0) {
      console.error(`\n--with-topology requires every scene to have a raw-highlight-topology.json sidecar. Missing for: ${missingTopology.map((s) => s.name).join(", ")}`);
      process.exit(1);
    }
    console.log(`Highlight-topology features ENABLED (${BIN_FEATURE_COUNT + TOPOLOGY_FEATURE_COUNT} features/bin instead of ${BIN_FEATURE_COUNT}).`);
  }
  const numFeatures = withTopology ? BIN_FEATURE_COUNT + TOPOLOGY_FEATURE_COUNT : BIN_FEATURE_COUNT;

  // The expensive part (loading 260 full-resolution TIFFs via sips
  // subprocesses, then per-pixel binning) happens exactly ONCE here,
  // regardless of how many (ridge, smoothness) candidates get tried below
  // — a first real run without this separation took ~72 minutes for a
  // single hardcoded hyperparameter pair, which would make any real grid
  // search prohibitively slow if loading were repeated per candidate.
  const loadStart = Date.now();
  for (const scene of scenes) scene.bins = computeSceneBins(scene.browser, scene.target, scene.width, scene.height, withTopology ? scene.topology : null);
  console.log(`Loaded and binned ${scenes.length} scenes in ${((Date.now() - loadStart) / 1000).toFixed(0)}s.`);

  const numTiles = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;

  // Fits one zone's joint per-tile model from a pool of training scenes
  // (excluding whichever scenes the caller already filtered out, e.g. a
  // LOO fold's held-out scene). Shared by both the LOO search below and
  // the final full-training-set fit, so "how a zone gets fit" only exists
  // in one place.
  function fitAllZones(trainingPool, ridgeLambda, smoothnessLambda) {
    const model = [];
    for (let zone = 0; zone < LUMA_ZONE_EDGES.length; zone++) {
      const byTile = Array.from({ length: numTiles }, (_, tile) => {
        const pooled = { features: [], deltas: [] };
        for (const s of trainingPool) {
          const bin = s.bins[tile * LUMA_ZONE_EDGES.length + zone];
          if (bin) { pooled.features.push(bin.features); pooled.deltas.push(bin.delta); }
        }
        return pooled.features.length > 0 ? pooled : null;
      });
      model.push(fitZoneJoint(byTile, numTiles, numFeatures, ridgeLambda, smoothnessLambda));
    }
    return model;
  }

  function evaluateModel(model, evalScenes) {
    const results = [];
    let regressions = 0;
    for (const scene of evalScenes) {
      const before = maeLinear255(scene.browser, scene.target);
      const corrected = applyModel(model, scene.browser, scene.width, scene.height, withTopology ? scene.topology : null);
      const after = maeLinear255(corrected, scene.target);
      const isRegression = after > before + REGRESSION_GATE_TOLERANCE;
      if (isRegression) regressions++;
      results.push({ name: scene.name, before, after, isRegression });
    }
    const meanBefore = results.reduce((s, r) => s + r.before, 0) / results.length;
    const meanAfter = results.reduce((s, r) => s + r.after, 0) / results.length;
    return { regressions, results, meanBefore, meanAfter };
  }

  // Leave-one-GROUP-out over the TRAINING scenes only — the 40 validation
  // scenes never take a turn being held out here, so they can't leak into
  // hyperparameter selection (see the manifest-split comment above main).
  // For an ungrouped scene this is exactly plain LOO (its group is a
  // singleton); for a manifest-tagged group (e.g. the palace-ceiling
  // sequence) every fold excludes ALL members of that scene's group from
  // the training pool, not just the one scene being scored.
  function runLoo(ridgeLambda, smoothnessLambda) {
    const results = [];
    for (let i = 0; i < trainingScenes.length; i++) {
      const held = trainingScenes[i];
      const heldGroup = groupOf.get(held.name);
      const pool = trainingScenes.filter((s) => groupOf.get(s.name) !== heldGroup);
      const model = fitAllZones(pool, ridgeLambda, smoothnessLambda);
      const before = maeLinear255(held.browser, held.target);
      const corrected = applyModel(model, held.browser, held.width, held.height, withTopology ? held.topology : null);
      const after = maeLinear255(corrected, held.target);
      const isRegression = after > before + REGRESSION_GATE_TOLERANCE;
      results.push({ name: held.name, group: heldGroup, before, after, isRegression });
    }
    const regressions = results.filter((r) => r.isRegression).length;
    const meanBefore = results.reduce((s, r) => s + r.before, 0) / results.length;
    const meanAfter = results.reduce((s, r) => s + r.after, 0) / results.length;
    return { regressions, results, meanBefore, meanAfter };
  }

  // Grid search over (ridgeLambda, smoothnessLambda), reusing the
  // precomputed bins above for every candidate — the first real run
  // (0.1, 1) regressed 4/260 scenes (before this training/validation
  // split existed) with mean MAE 26.23 -> 9.83, close enough to a clean
  // pass that tuning regularization is worth trying before considering
  // this a model-form problem.
  const RIDGE_CANDIDATES = [0.03, 0.1, 0.3, 1, 3];
  const SMOOTHNESS_CANDIDATES = [0.3, 1, 3, 10, 30];
  console.log(`\nSearching (ridgeLambda, smoothnessLambda) via leave-one-out on the ${trainingScenes.length} training scenes — ${RIDGE_CANDIDATES.length * SMOOTHNESS_CANDIDATES.length} combinations, reusing the loaded/binned data above:`);
  let best = null;
  for (const ridgeLambda of RIDGE_CANDIDATES) {
    for (const smoothnessLambda of SMOOTHNESS_CANDIDATES) {
      const searchStart = Date.now();
      const { regressions, meanBefore, meanAfter, results } = runLoo(ridgeLambda, smoothnessLambda);
      const seconds = ((Date.now() - searchStart) / 1000).toFixed(0);
      console.log(`  ridge=${ridgeLambda} smoothness=${smoothnessLambda}: mean ${meanBefore.toFixed(2)} -> ${meanAfter.toFixed(2)}, regressions=${regressions} (${seconds}s)`);
      if (best === null || regressions < best.regressions || (regressions === best.regressions && meanAfter < best.meanAfter)) {
        best = { ridgeLambda, smoothnessLambda, regressions, meanAfter, meanBefore, results };
      }
    }
  }

  console.log(`\nBest on training LOO: ridge=${best.ridgeLambda} smoothness=${best.smoothnessLambda} (mean ${best.meanBefore.toFixed(2)} -> ${best.meanAfter.toFixed(2)}, ${best.regressions}/${trainingScenes.length} regressions)`);
  console.log("\nPer-scene results at best hyperparameters (training LOO):");
  for (const r of best.results) console.log(`  ${r.name}: MAE ${r.before.toFixed(2)} -> ${r.after.toFixed(2)}${r.isRegression ? "  [REGRESSION]" : ""}`);

  // Grouped-cohort breakdown: reported separately from the overall number
  // above because a grouped LOO result is a genuinely different (harder,
  // more honest) test than ordinary per-scene LOO — near-duplicate frames
  // were excluded from each other's training pool, so this number reflects
  // whether the model generalizes to the CATEGORY, not whether it
  // memorized these 13 specific photos.
  for (const [group, members] of namedGroups) {
    const groupResults = best.results.filter((r) => r.group === group);
    const groupRegressions = groupResults.filter((r) => r.isRegression).length;
    const groupMeanBefore = groupResults.reduce((s, r) => s + r.before, 0) / groupResults.length;
    const groupMeanAfter = groupResults.reduce((s, r) => s + r.after, 0) / groupResults.length;
    console.log(`\nGrouped LOO result for "${group}" (${members.length} scenes, evaluated with the whole group excluded from training each time): mean ${groupMeanBefore.toFixed(2)} -> ${groupMeanAfter.toFixed(2)}, ${groupRegressions}/${groupResults.length} regressions`);
  }

  if (best.regressions > 0) {
    console.error(`\n${best.regressions}/${trainingScenes.length} training scenes regressed beyond tolerance (${REGRESSION_GATE_TOLERANCE} MAE) even at the best hyperparameters found. Gate failed. Not proceeding to a final fit, the validation check, or the held-out test.`);
    process.exit(1);
  }
  console.log("\nTraining-pool gate passed. Fitting a final model on ALL training scenes at the selected hyperparameters, then checking it against the validation tier (scenes never used in the search above)...");

  // Freeze the model: refit once on every training scene (no held-out
  // fold this time) at the hyperparameters the LOO search selected, then
  // apply that single frozen model to each validation scene. This is a
  // genuine out-of-sample check — validation scenes influenced neither
  // the fitting nor the hyperparameter choice.
  const finalModel = fitAllZones(trainingScenes, best.ridgeLambda, best.smoothnessLambda);
  const validation = evaluateModel(finalModel, validationScenes);
  console.log(`\nValidation-tier result: mean ${validation.meanBefore.toFixed(2)} -> ${validation.meanAfter.toFixed(2)}, ${validation.regressions}/${validationScenes.length} regressions`);
  for (const r of validation.results) console.log(`  ${r.name}: MAE ${r.before.toFixed(2)} -> ${r.after.toFixed(2)}${r.isRegression ? "  [REGRESSION]" : ""}`);

  if (validation.regressions > 0) {
    console.error(`\n${validation.regressions}/${validationScenes.length} validation scenes regressed beyond tolerance. Gate failed — this model does not generalize past the training pool it was tuned on. Not proceeding to the held-out test.`);
    process.exit(1);
  }
  console.log("\nValidation-tier gate passed. Still not approved to ship — the one-shot held-out test (Shoots 387-426) is the only remaining step, and requires explicit team agreement that the architecture and training are frozen before it is ever run. See docs/phase3-raw-aware-processor-plan.md.");
}

// Only run as a CLI entry point — importing this module for its exported
// functions (e.g. to test them directly) must not trigger the CLI logic.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main();
}
