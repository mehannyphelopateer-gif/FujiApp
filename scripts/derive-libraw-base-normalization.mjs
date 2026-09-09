#!/usr/bin/env node
// Derives a base color-correction affine matrix that maps the browser
// LibRaw neutral decode (Preview tab, RAF path) toward X RAW Studio's own
// neutral-Provia conversion of the identical RAF.
//
// Why this exists: comparing DSCF0752.RAF decoded via browser LibRaw
// (plain Provia, DR100, Auto WB 0/0, every other control zero/off) against
// `xraw-provia-auto-neutral.jpg` measured MAE 22.31/255 — *before* any
// recipe-specific control (DR400, Clarity, Grain, Color Chrome, WB shift)
// is even applied. Ruled out as explanations: half-size vs full-size
// demosaic (full-size measured worse: 20.15 MAE, 5x slower) and this app's
// AWB gray-world damping (swept 0/25/50/75%, 50% already near-optimal).
// What's left is a genuine base color/tone mismatch between LibRaw's and
// X RAW Studio's processing of the same sensor data — a single fixed
// correction, not something any per-recipe control can fix.
//
// Deliberately a single GLOBAL AFFINE fit (out = A*in + b, one solve per
// channel via fitGlobalAffine), not a full local Hald CLUT: a prior
// single-scene local-grid LUT experiment cut error on the scene it was fit
// to (19.82 -> 9.86 MAE) but made an unrelated recipe (Provia) worse
// (~17.80 MAE) — a low-parameter global fit pooled across multiple scenes
// is far less able to memorize one scene's specific color content. If a
// validated global affine still leaves meaningful residual error, a
// local-grid layer (see fitHaldClut in lib/hald-clut-fitting.mjs) can be
// layered on top later, but only once there's enough scene coverage to
// trust it.
//
// Usage:
//   node scripts/derive-libraw-base-normalization.mjs [input-dir] [output-file]
// input-dir defaults to ./calibration-input, output-file to
// src/engine/webgl/generated/libRawBaseNormalization.ts
//
// Expected input layout — one pair per calibration shoot folder:
//   calibration-input/Shoot N/browser-baseline.jpg   (browser LibRaw decode,
//     plain Provia / DR100 / Auto WB 0-0 / everything else zero-off, of
//     that shoot's RAF, exported from the Preview tab)
//   calibration-input/Shoot N/xraw-baseline.jpg          -- OR --
//   calibration-input/Shoot N/xraw-provia-auto-neutral.jpg
//     (X RAW Studio's neutral-Provia/Auto-WB conversion of the SAME RAF;
//     both filenames are pre-existing conventions in this repo for the
//     same thing)
//
// `browser-baseline.jpg` does not exist anywhere yet as of this writing —
// producing it requires an actual browser (this script was written without
// one available). As of the initial run, `xraw-baseline.jpg`/
// `xraw-provia-auto-neutral.jpg` already exist, with no browser-baseline
// counterpart yet, in: Shoot 8, Shoot 9, Shoot 10, Shoot 11 — meaning no
// new photography is needed, only new browser-side renders of RAFs already
// in the repo.

import { readdirSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { loadSamplePixels, fitGlobalAffine, evalAffine } from "./lib/hald-clut-fitting.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const inputDir = process.argv[2] ?? join(__dirname, "..", "calibration-input");
const outputFile =
  process.argv[3] ?? join(__dirname, "..", "src", "engine", "webgl", "generated", "libRawBaseNormalization.ts");

const SAMPLE_SIZE = 256; // common square resize target for correspondence sampling
const TRIM_FRACTION = 0.08; // fraction trimmed off each edge before resize, per image

const BROWSER_BASELINE_NAME = "browser-baseline.jpg";
const XRAW_BASELINE_NAMES = ["xraw-baseline.jpg", "xraw-provia-auto-neutral.jpg"];

/** Finds every directory under `dir` that has both a browser-baseline render and a matching X RAW Studio neutral export. */
function findPairs(dir) {
  const found = [];
  function walk(current) {
    let entries;
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    const names = entries.map((e) => e.name);
    if (names.includes(BROWSER_BASELINE_NAME)) {
      const xrawName = XRAW_BASELINE_NAMES.find((name) => names.includes(name));
      if (xrawName) found.push({ folder: current, xrawName });
    }
    for (const entry of entries) {
      if (entry.isDirectory()) walk(join(current, entry.name));
    }
  }
  walk(dir);
  return found;
}

function mae(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return (sum / a.length) * 255;
}

function applyAffineToSamples(affine, samples) {
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 3) {
    const [r, g, b] = evalAffine(affine, samples[i], samples[i + 1], samples[i + 2]);
    out[i] = r;
    out[i + 1] = g;
    out[i + 2] = b;
  }
  return out;
}

function poolSamples(entries) {
  const totalSamples = entries.reduce((sum, e) => sum + e.browser.length / 3, 0);
  const browser = new Float32Array(totalSamples * 3);
  const xraw = new Float32Array(totalSamples * 3);
  let offset = 0;
  for (const entry of entries) {
    browser.set(entry.browser, offset);
    xraw.set(entry.xraw, offset);
    offset += entry.browser.length;
  }
  return { browser, xraw, totalSamples };
}

async function main() {
  const pairs = findPairs(inputDir);
  if (pairs.length === 0) {
    console.error(
      `No shoot folder under ${inputDir} has both ${BROWSER_BASELINE_NAME} and an xraw baseline ` +
        `(${XRAW_BASELINE_NAMES.join(" or ")}). This script needs at least one browser-decoded RAF render before it ` +
        "can fit anything — see this file's header comment for which shoot folders already have the X RAW Studio " +
        "half of the pair waiting.",
    );
    process.exit(1);
  }

  console.log(`Found ${pairs.length} browser/xraw baseline pair(s):`);
  const entries = [];
  for (const { folder, xrawName } of pairs) {
    console.log(`  ${folder} (${xrawName})`);
    const browser = await loadSamplePixels(join(folder, BROWSER_BASELINE_NAME), {
      sampleSize: SAMPLE_SIZE,
      trimFraction: TRIM_FRACTION,
    });
    const xraw = await loadSamplePixels(join(folder, xrawName), { sampleSize: SAMPLE_SIZE, trimFraction: TRIM_FRACTION });
    entries.push({ folder, browser, xraw });
  }

  if (entries.length < 2) {
    console.warn(
      "\nOnly one scene available — fitting anyway, but this is exactly the situation that overfit last time " +
        "(a single-scene fit that didn't generalize to other recipes/scenes). Do not treat this fit as trustworthy " +
        "until at least 2-3 more scenes are added and held-out validation below has something to hold out.",
    );
  } else {
    console.log("\nLeave-one-scene-out validation:");
    for (let i = 0; i < entries.length; i++) {
      const heldOut = entries[i];
      const training = entries.filter((_, j) => j !== i);
      const { browser: trainBrowser, xraw: trainXraw, totalSamples: trainCount } = poolSamples(training);
      const affine = fitGlobalAffine(trainBrowser, trainXraw, trainCount);

      const before = mae(heldOut.browser, heldOut.xraw);
      const corrected = applyAffineToSamples(affine, heldOut.browser);
      const after = mae(corrected, heldOut.xraw);
      console.log(`  held out ${heldOut.folder}: MAE ${before.toFixed(2)} -> ${after.toFixed(2)} (fit on the other ${training.length} scene(s))`);
    }
  }

  const { browser, xraw, totalSamples } = poolSamples(entries);
  const finalAffine = fitGlobalAffine(browser, xraw, totalSamples);
  const finalCorrected = applyAffineToSamples(finalAffine, browser);
  console.log(
    `\nFinal affine fit on all ${entries.length} scene(s) pooled: in-sample MAE ${mae(browser, xraw).toFixed(2)} -> ${mae(finalCorrected, xraw).toFixed(2)}`,
  );
  console.log("(In-sample error only shows whether the fit converged — trust the held-out numbers above for whether it generalizes.)");

  mkdirSync(dirname(outputFile), { recursive: true });
  const contents = `// GENERATED by scripts/derive-libraw-base-normalization.mjs — do not hand-edit.
// Rerun after adding a new browser-baseline.jpg/xraw-baseline.jpg (or
// xraw-provia-auto-neutral.jpg) pair to a calibration shoot folder. See this
// script's header comment for the methodology and what's been ruled out.
//
// Maps a browser LibRaw neutral-Provia decode's RGB toward X RAW Studio's
// own neutral-Provia conversion of the same RAF: out_channel = a*R + b*G +
// c*B + d, coefficients in [a, b, c, d] order, one row per output channel
// (R, G, B). Apply in the same sRGB-gamma-encoded 0..1 pixel space the rest
// of the WebGL pipeline already works in (no linearization step).

export const LIBRAW_BASE_NORMALIZATION_AFFINE: [number, number, number, number][] = ${JSON.stringify(finalAffine)};
`;
  writeFileSync(outputFile, contents);
  console.log(`\nWrote ${outputFile}`);
}

main();
