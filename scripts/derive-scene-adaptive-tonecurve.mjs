#!/usr/bin/env node
// Fits a scene-adaptive per-luma-zone tone correction mapping the browser
// LibRaw as-shot-WB decode toward X RAW Studio's as-shot-WB conversion —
// the real, feature-conditioned successor to the ruled-out global
// approaches. See docs/scene-adaptive-processor-architecture.md for the
// full design and the negative results (global affine, LibRaw highlight
// modes, global gamma/output-power, decoded-pixel scene features) this
// replaces.
//
// Correction shape: the same per-luma-zone control-point lift curve
// already proven out in derive-shadow-chroma-recovery.mjs, generalized so
// each zone's per-channel lift is a RIDGE-REGULARIZED LINEAR FUNCTION of a
// small runtime feature vector, not a fixed constant. Features are
// RAW-domain sensor statistics (percentiles.p99, nearBlackFraction) —
// confirmed this session to correlate with correction benefit (r=-0.56,
// r=0.52 respectively) and, critically, to be computable from RAF/RAW data
// alone, unlike decoded-pixel statistics (which are corrupted by the exact
// flaw being corrected — see the architecture doc). EXIF exposure bias was
// also tried as a third feature — reverted, see featureVector()'s comment
// for why it made the regression count worse despite a slightly better mean.
//
// Validation, per the architecture doc's explicit rule: leave-one-out
// cross-validation within the TRAINING pool only. A separate, disjoint
// HELD-OUT set (calibration-input/phase2-held-out-manifest.json's
// heldOutShoots) is reserved and must never be touched by this script's
// fitting or regularization-selection loop — only a separate, single-shot
// final-test script (derive-scene-adaptive-tonecurve-finaltest.mjs) may
// read it, and only after this script's internal validation already
// passes cleanly.
//
// Usage: node scripts/derive-scene-adaptive-tonecurve.mjs [input-dir] [output-file]

import { readdirSync, readFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { loadSamplePixels } from "./lib/hald-clut-fitting.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const inputDir = process.argv[2] ?? join(__dirname, "..", "calibration-input");
const outputFile =
  process.argv[3] ?? join(__dirname, "..", "src", "engine", "webgl", "generated", "sceneAdaptiveToneCurve.ts");

const SAMPLE_SIZE = 128;
const TRIM_FRACTION = 0.08;
const LUMA_BUCKET_EDGES = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]; // 7 zones
const RIDGE_LAMBDAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]; // searched via LOO
const REGRESSION_GATE_TOLERANCE = 2.0; // MAE points a held-out scene may worsen by and still "pass"

function luma(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function bucketIndex(l) {
  for (let i = LUMA_BUCKET_EDGES.length - 1; i >= 0; i--) if (l >= LUMA_BUCKET_EDGES[i]) return i;
  return 0;
}
function mae(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return (sum / a.length) * 255;
}
function median(values) {
  if (values.length === 0) return null;
  const s = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
}

const HIGHLIGHT_CELL_EPSILON = 0.0001; // a cell counts as "has highlight" above this nearWhiteFraction98

/** Compact descriptors derived from Codex's 3x3 spatial RAW occupancy grid
 * (commit 69305a2) — per Codex's request, feeding all 9 cells x 3 fields
 * directly into ridge regression would be far too many parameters for
 * ~100 scenes. These four are designed around the specific failure this
 * grid was built to fix: distinguishing a broad bright region (already
 * well handled by p99) from sparse, spatially localized specular/flash
 * highlights against an otherwise near-black scene (Shoots 53/55/84),
 * and a uniformly dark scene from one with a real (if small) non-black
 * subject region (Shoots 38/65/72/74 vs. everything else). */
function spatialGridDescriptors(spatialGrid) {
  if (!spatialGrid) return null;
  const cells = spatialGrid.cells;
  const highlightFracs = cells.map((c) => c.nearWhiteFraction98);
  const shadowFracs = cells.map((c) => c.nearBlackFraction);
  const maxCellHighlight = Math.max(...highlightFracs);
  const numCellsWithHighlight = highlightFracs.filter((f) => f > HIGHLIGHT_CELL_EPSILON).length;
  const maxCellShadow = Math.max(...shadowFracs);
  const minCellShadow = Math.min(...shadowFracs);
  const shadowMean = shadowFracs.reduce((a, b) => a + b, 0) / shadowFracs.length;
  const shadowDispersion = Math.sqrt(shadowFracs.reduce((a, f) => a + (f - shadowMean) ** 2, 0) / shadowFracs.length);
  return { maxCellHighlight, numCellsWithHighlight, maxCellShadow, minCellShadow, shadowDispersion };
}

/** Loads a scene's RAW-domain features (p99, nearBlackFraction, and the
 * spatial-grid descriptors above where available) from either a per-shoot
 * raw-sensor-features.json sidecar, or the phase2 corpus manifest keyed by
 * RAF filename — both were produced by Codex's pre-demosaic sensor-mosaic
 * extractor, never from decoded pixels. Only the sidecar carries the
 * spatial grid so far (all 103 training scenes have one); the corpus
 * fallback path (used only if a sidecar is missing) does not. */
function loadFeatures(folder, corpusByFileName) {
  const rafName = readdirSync(folder).find((f) => /\.raf$/i.test(f));

  let p99, nearBlackFraction, grid = null;
  const sidecar = join(folder, "raw-sensor-features.json");
  if (existsSync(sidecar)) {
    const j = JSON.parse(readFileSync(sidecar, "utf8"));
    p99 = j.percentiles.p99;
    nearBlackFraction = j.nearBlackFraction;
    grid = spatialGridDescriptors(j.spatialGrid);
  } else {
    const rec = corpusByFileName[rafName];
    if (!rec) throw new Error(`No RAW-domain features found for ${folder} (checked sidecar and phase2 corpus)`);
    p99 = rec.percentiles.p99;
    nearBlackFraction = rec.nearBlackFraction;
  }

  return { p99, nearBlackFraction, grid };
}

function featureVector(f) {
  // [intercept, p99, nearBlackFraction, p99*nearBlackFraction]. The
  // interaction term lets the model express "low p99 means something
  // different depending on how black-dominated the scene is" instead of
  // treating the two features as independent additive effects — the 13
  // remaining regressions with a pure additive model clustered specifically
  // in the low-p99 range, which is exactly where an interaction would
  // matter most. This is the best model found so far: mean LOO MAE 23.30,
  // 11/103 regressions.
  //
  // Tried and rejected, each added ALONE per the one-at-a-time discipline
  // (see spatialGridDescriptors() for what these are and why they were
  // built — Codex's structural audit of the persistent failures):
  //   - EXIF exposureBiasEV (r=-0.43 alone): mean improved slightly
  //     (23.83) but regressions got worse (13 -> 16) — only clean at the
  //     bias extremes, noisy in the middle most scenes sit in.
  //   - grid.maxCellHighlight: mean 23.45 (worse), regressions unchanged
  //     at 11 (same scenes).
  //   - grid.numCellsWithHighlight: mean improved (23.03) but regressions
  //     got worse (11 -> 13), and Shoot 73 regressed dramatically (6.17 ->
  //     33.30) despite being in neither structural failure group.
  //   - grid.minCellShadow + grid.maxCellShadow: worse on both axes
  //     (23.54 mean, 12 regressions).
  //   - grid.shadowDispersion: a wash (23.29 mean, same 11 regressions,
  //     same scenes) — no harm, no help, not worth the added parameter.
  //   - grid.maxCellHighlight * nearBlackFraction (the literal Group-B
  //     signature: black-dominated AND some sparse highlight): best mean
  //     yet (22.68) but regressions still got worse (11 -> 12, new
  //     failure on a previously-fine scene).
  // Consistent pattern across all six attempts: everything that improves
  // the mean does so by trading in new regressions elsewhere, never by
  // reducing the regression count itself. The 3x3 grid's compact scalar
  // summaries don't appear to carry the signal needed — a genuinely
  // different feature or representation is likely needed, not another
  // grid-derived scalar. See docs/scene-adaptive-processor-architecture.md.
  return [1, f.p99, f.nearBlackFraction, f.p99 * f.nearBlackFraction];
}

/** Computes ONE scene's own per-zone median (xraw - browser) delta, once.
 * Expensive part (iterating every pixel sample) — called exactly once per
 * scene, never repeated across LOO folds or lambda candidates. */
function computeSceneZoneDeltas(entry) {
  const buckets = LUMA_BUCKET_EDGES.map(() => ({ r: [], g: [], b: [] }));
  for (let i = 0; i < entry.browser.length; i += 3) {
    const br = entry.browser[i], bg = entry.browser[i + 1], bb = entry.browser[i + 2];
    const xr = entry.xraw[i], xg = entry.xraw[i + 1], xb = entry.xraw[i + 2];
    const bucket = buckets[bucketIndex(luma(br, bg, bb))];
    bucket.r.push(xr - br);
    bucket.g.push(xg - bg);
    bucket.b.push(xb - bb);
  }
  return LUMA_BUCKET_EDGES.map((_, z) => {
    const dr = median(buckets[z].r), dg = median(buckets[z].g), db = median(buckets[z].b);
    return dr === null ? null : [dr, dg, db]; // null = this scene had no samples in this zone
  });
}

/** Cheap: assembles per-zone (feature, delta) training arrays from
 * ALREADY-COMPUTED per-scene zone deltas — no pixel iteration here, safe
 * to call once per LOO fold per lambda candidate. */
function assembleZoneData(entries) {
  const zoneData = LUMA_BUCKET_EDGES.map(() => ({ features: [], deltas: [] }));
  for (const entry of entries) {
    const fv = featureVector(entry.features);
    entry.zoneDeltas.forEach((delta, z) => {
      if (delta === null) return;
      zoneData[z].features.push(fv);
      zoneData[z].deltas.push(delta);
    });
  }
  return zoneData;
}

/** Ridge regression: solve (X^T X + lambda*I) w = X^T y, per output channel. */
function ridgeFit(features, targets, lambda) {
  const p = features[0].length;
  const XtX = Array.from({ length: p }, () => new Array(p).fill(0));
  const XtY = [new Array(p).fill(0), new Array(p).fill(0), new Array(p).fill(0)];
  for (let i = 0; i < features.length; i++) {
    const x = features[i];
    for (let a = 0; a < p; a++) {
      for (let c = 0; c < p; c++) XtX[a][c] += x[a] * x[c];
      for (let ch = 0; ch < 3; ch++) XtY[ch][a] += x[a] * targets[i][ch];
    }
  }
  for (let a = 0; a < p; a++) XtX[a][a] += lambda; // regularizes the intercept too — kept simple/symmetric given how few features there are
  return [0, 1, 2].map((ch) => solveLinearSystem(XtX, XtY[ch]));
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

function predictZoneDelta(model, zone, features) {
  const fv = featureVector(features);
  const coeffs = model[zone]; // [ [wr...], [wg...], [wb...] ]
  return [0, 1, 2].map((ch) => coeffs[ch].reduce((sum, w, i) => sum + w * fv[i], 0));
}

function applyModelWithFeatures(model, features, browserSamples) {
  const zoneDeltas = LUMA_BUCKET_EDGES.map((_, z) => predictZoneDelta(model, z, features));
  const out = new Float32Array(browserSamples.length);
  for (let i = 0; i < browserSamples.length; i += 3) {
    const r = browserSamples[i], g = browserSamples[i + 1], b = browserSamples[i + 2];
    const z = bucketIndex(luma(r, g, b));
    const [dr, dg, db] = zoneDeltas[z];
    out[i] = Math.min(1, Math.max(0, r + dr));
    out[i + 1] = Math.min(1, Math.max(0, g + dg));
    out[i + 2] = Math.min(1, Math.max(0, b + db));
  }
  return out;
}

async function main() {
  const shootDirs = readdirSync(inputDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && /^Shoot \d+$/.test(d.name))
    .map((d) => d.name);

  let heldOutShoots = new Set();
  const manifestPath = join(inputDir, "phase2-held-out-manifest.json");
  if (existsSync(manifestPath)) {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    heldOutShoots = new Set(manifest.heldOutShoots.map((s) => s.shoot));
  }

  const corpusPath = join(inputDir, "phase2-raw-sensor-corpus.json");
  const corpusByFileName = {};
  if (existsSync(corpusPath)) {
    const corpus = JSON.parse(readFileSync(corpusPath, "utf8"));
    for (const rec of corpus.records) corpusByFileName[rec.fileName] = rec;
  }

  const trainingShoots = shootDirs.filter((name) => {
    if (heldOutShoots.has(name)) return false;
    const folder = join(inputDir, name);
    return existsSync(join(folder, "browser-as-shot-baseline.jpg")) && existsSync(join(folder, "xraw-as-shot-baseline.jpg"));
  });

  console.log(`Training pool: ${trainingShoots.length} scenes (excluding ${heldOutShoots.size} reserved held-out scenes)`);

  const entries = [];
  for (const name of trainingShoots) {
    const folder = join(inputDir, name);
    const browser = await loadSamplePixels(join(folder, "browser-as-shot-baseline.jpg"), { sampleSize: SAMPLE_SIZE, trimFraction: TRIM_FRACTION });
    const xraw = await loadSamplePixels(join(folder, "xraw-as-shot-baseline.jpg"), { sampleSize: SAMPLE_SIZE, trimFraction: TRIM_FRACTION });
    const features = loadFeatures(folder, corpusByFileName);
    entries.push({ name, browser, xraw, features });
  }

  // Expensive per-pixel bucketing happens exactly once per scene here —
  // every LOO fold and every lambda candidate below reuses these.
  for (const entry of entries) entry.zoneDeltas = computeSceneZoneDeltas(entry);
  console.log(`Precomputed per-scene zone deltas for ${entries.length} scenes.`);

  // Leave-one-out cross-validation, searched over ridge lambda. For each
  // candidate lambda, refit on all-but-one and predict the held-out
  // scene's own correction from ITS OWN features (not from the training
  // fold) — this is the actual generalization test, not just a smoothness
  // check on the training residuals.
  console.log("\nSearching ridge lambda via leave-one-out (reporting mean held-out MAE per lambda):");
  let bestLambda = null, bestMeanAfter = Infinity, bestResults = null;
  for (const lambda of RIDGE_LAMBDAS) {
    const results = [];
    for (let i = 0; i < entries.length; i++) {
      const heldOut = entries[i];
      const training = entries.filter((_, j) => j !== i);
      const zoneData = assembleZoneData(training);
      const model = zoneData.map(({ features, deltas }) =>
        features.length === 0 ? [[0, 0, 0], [0, 0, 0], [0, 0, 0]] : ridgeFit(features, deltas, lambda),
      );
      const before = mae(heldOut.browser, heldOut.xraw);
      const corrected = applyModelWithFeatures(model, heldOut.features, heldOut.browser);
      const after = mae(corrected, heldOut.xraw);
      results.push({ name: heldOut.name, before, after });
    }
    const meanAfter = results.reduce((s, r) => s + r.after, 0) / results.length;
    const meanBefore = results.reduce((s, r) => s + r.before, 0) / results.length;
    console.log(`  lambda=${lambda}: mean MAE ${meanBefore.toFixed(2)} -> ${meanAfter.toFixed(2)}`);
    if (meanAfter < bestMeanAfter) { bestMeanAfter = meanAfter; bestLambda = lambda; bestResults = results; }
  }

  console.log(`\nBest lambda: ${bestLambda} (mean held-out MAE ${bestMeanAfter.toFixed(2)})`);
  console.log("\nPer-scene leave-one-out results at best lambda:");
  let regressions = 0;
  for (const r of bestResults.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))) {
    const flag = r.after > r.before + REGRESSION_GATE_TOLERANCE ? "  [REGRESSION]" : "";
    if (flag) regressions++;
    console.log(`  ${r.name}: MAE ${r.before.toFixed(2)} -> ${r.after.toFixed(2)}${flag}`);
  }

  console.log(`\n${regressions}/${bestResults.length} scenes regressed beyond tolerance (${REGRESSION_GATE_TOLERANCE} MAE).`);
  if (regressions > 0) {
    console.error(
      "\nGate failed: this model does not generalize within the training pool even before touching the held-out " +
        "set. Not fitting a final model, not writing output. Do not proceed to the held-out final test with a model " +
        "that already fails internal cross-validation.",
    );
    process.exit(1);
  }

  console.log("\nInternal cross-validation gate PASSED — every training-pool scene held up under leave-one-out.");
  console.log("This does NOT mean the model is approved to ship. Run the separate, single-shot final test against");
  console.log("the 16 reserved held-out scenes next — see derive-scene-adaptive-tonecurve-finaltest.mjs.");

  // Fit the final model on ALL training entries (for the final-test script
  // to consume) — still not written to the shipped generated/ location
  // until the held-out test also passes.
  const zoneData = assembleZoneData(entries);
  const finalModel = zoneData.map(({ features, deltas }) =>
    features.length === 0 ? [[0, 0, 0], [0, 0, 0], [0, 0, 0]] : ridgeFit(features, deltas, bestLambda),
  );

  const stagingPath = join(inputDir, "scene-adaptive-model-staged.json");
  writeFileSync(stagingPath, JSON.stringify({ lambda: bestLambda, lumaBucketEdges: LUMA_BUCKET_EDGES, model: finalModel, trainedOn: trainingShoots.length }, null, 2));
  console.log(`\nStaged (unapproved) model written to ${stagingPath} for the final-test script to consume.`);
}

main();
