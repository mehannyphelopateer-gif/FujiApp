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

import { readdirSync, statSync, mkdirSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { loadSamplePixels, fitHaldClut, writeLutPng } from "./lib/hald-clut-fitting.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const inputDir = process.argv[2] ?? join(__dirname, "..", "calibration-input");
const outputDir = process.argv[3] ?? join(__dirname, "..", "public", "luts");

const SAMPLE_SIZE = 256; // common square resize target for correspondence sampling
const TRIM_FRACTION = 0.08; // fraction trimmed off each edge before resize, per image
const GRID_CONFIG = {
  gridLevels: 17, // coarse control-grid resolution for the local correction layer
  correctionClamp: 0.14, // max |local correction| per channel (0..1 scale)
  idwRadius: 3, // grid-node search radius (in grid cells) for IDW falloff
  idwEpsilon: 0.5,
  maxSamplesPerCell: 500, // cap per grid cell so one dense region can't dominate memory/sort cost
};

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

async function deriveLutForSim(slug, shootFolders) {
  const neutralChunks = [];
  const targetChunks = [];

  for (const folder of shootFolders) {
    const simPath = join(folder, `calib-${slug}.jpg`);
    if (!existsSync(simPath)) continue;
    const neutralPath = join(folder, "calib-neutral.jpg");
    console.log(`  reading pair: ${neutralPath} / ${simPath}`);
    neutralChunks.push(await loadSamplePixels(neutralPath, { sampleSize: SAMPLE_SIZE, trimFraction: TRIM_FRACTION }));
    targetChunks.push(await loadSamplePixels(simPath, { sampleSize: SAMPLE_SIZE, trimFraction: TRIM_FRACTION }));
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
  const lookup = fitHaldClut(neutral, target, totalSamples, GRID_CONFIG);
  writeLutPng(lookup, join(outputDir, `${slug}.png`));

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
