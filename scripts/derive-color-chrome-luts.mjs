#!/usr/bin/env node
// Derives real Hald CLUTs for Color Chrome Effect and Color Chrome FX Blue
// at their Weak/Strong states, replacing fragmentShader.ts's applyColorChrome
// parametric approximation. See ~/.claude/plans/indexed-inventing-wren.md's
// Phase 3 section for the full methodology.
//
// Both effects are discrete 3-step camera enums (Off/Weak/Strong — see
// src/types/recipe.ts's EffectStrength), and src/lib/recipes/neutralize.ts's
// neutralizedStrength() is mathematically always exactly 0/0.5/1 for any
// Off/Weak/Strong pair — so only 2 non-identity real states need
// calibrating per effect, each treated as a deterministic RGB->RGB Hald-CLUT
// problem exactly like the film-simulation LUTs (reuses
// scripts/lib/hald-clut-fitting.mjs).
//
// Usage: node scripts/derive-color-chrome-luts.mjs [input-dir] [output-dir]
// input-dir defaults to ./calibration-input, output-dir to ./public/luts.
//
// Baseline is calib-provia.jpg (Color Chrome Off, FX Blue Off — the
// existing Phase 1/2 zero point), not a fresh neutral decode: both effects
// operate on already-rendered/tone-mapped pixels, same reasoning as
// derive-parametric-curves.mjs. Only the FIRST shoot folder with both
// calib-provia.jpg and calib-cce-weak.jpg is used (see that script's
// header comment for why pooling multiple scenes isn't needed here).
//
// calib-cce-strong-fxblue-strong.jpg (if present) is a HELD-OUT validation
// file, never fit into anything — it checks that composing the two
// independently-fit LUTs sequentially (the way the shader applies them)
// actually matches a real photo with both effects on at once.

import { readdirSync, existsSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { loadSamplePixels, fitHaldClut, writeLutPng } from "./lib/hald-clut-fitting.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const inputDir = process.argv[2] ?? join(__dirname, "..", "calibration-input");
const outputDir = process.argv[3] ?? join(__dirname, "..", "public", "luts");

const SAMPLE_SIZE = 256;
const TRIM_FRACTION = 0.08;
const GRID_CONFIG = {
  gridLevels: 17,
  correctionClamp: 0.14,
  idwRadius: 3,
  idwEpsilon: 0.5,
  maxSamplesPerCell: 500,
};

const TARGETS = [
  { slug: "cce-weak", outPath: ["color-chrome", "weak.png"] },
  { slug: "cce-strong", outPath: ["color-chrome", "strong.png"] },
  { slug: "fxblue-weak", outPath: ["fx-blue", "weak.png"] },
  { slug: "fxblue-strong", outPath: ["fx-blue", "strong.png"] },
];

function findShootFolder(dir) {
  const candidates = [];
  function walk(current) {
    let entries;
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    if (entries.some((e) => e.name === "calib-provia.jpg") && entries.some((e) => e.name === "calib-cce-weak.jpg")) {
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

/** Mean absolute error between a lookup fit and real target pixels, at the baseline's own sample points. */
function meanAbsError(lookup, baseline, target) {
  const count = baseline.length / 3;
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const [pr, pg, pb] = lookup(baseline[i * 3], baseline[i * 3 + 1], baseline[i * 3 + 2]);
    sum +=
      (Math.abs(pr - target[i * 3]) + Math.abs(pg - target[i * 3 + 1]) + Math.abs(pb - target[i * 3 + 2])) / 3;
  }
  return sum / count;
}

async function main() {
  const shootFolder = findShootFolder(inputDir);
  if (!shootFolder) {
    console.error(
      `No shoot folder under ${inputDir} has both calib-provia.jpg and calib-cce-weak.jpg — run the Camera ` +
        "tab's Advanced > Parametric Calibration Capture against a RAF that already has a Phase 1/2 shoot folder first.",
    );
    process.exit(1);
  }
  console.log(`Using shoot folder: ${shootFolder}`);

  const baseline = await loadPixels(join(shootFolder, "calib-provia.jpg"));
  const fittedLookups = {};

  for (const { slug, outPath } of TARGETS) {
    const targetPath = join(shootFolder, `calib-${slug}.jpg`);
    if (!existsSync(targetPath)) {
      console.log(`  ${slug}: no calib-${slug}.jpg found — skipping.`);
      continue;
    }
    console.log(`Deriving LUT for "${slug}"…`);
    const target = await loadPixels(targetPath);
    const lookup = fitHaldClut(baseline, target, baseline.length / 3, GRID_CONFIG);
    fittedLookups[slug] = lookup;

    const outFile = join(outputDir, ...outPath);
    mkdirSync(dirname(outFile), { recursive: true });
    writeLutPng(lookup, outFile);
    console.log(`  wrote ${outFile} (fit error: ${meanAbsError(lookup, baseline, target).toFixed(4)})`);
  }

  // Held-out validation: compose cce-strong then fxblue-strong (the same
  // order applyColorChrome/applyChromeLut runs in the shader) and compare
  // against a real photo with both effects on simultaneously.
  const comboPath = join(shootFolder, "calib-cce-strong-fxblue-strong.jpg");
  if (existsSync(comboPath) && fittedLookups["cce-strong"] && fittedLookups["fxblue-strong"]) {
    console.log("\nValidating composed cce-strong + fxblue-strong against held-out real photo…");
    const combo = await loadPixels(comboPath);
    const composed = (r, g, b) => {
      const [cr, cg, cb] = fittedLookups["cce-strong"](r, g, b);
      return fittedLookups["fxblue-strong"](cr, cg, cb);
    };
    const error = meanAbsError(composed, baseline, combo);
    console.log(`  composed-pipeline mean abs error vs real combo photo: ${error.toFixed(4)}`);
    if (error > 0.05) {
      console.warn(
        "  ⚠ this is notably higher than the individual per-effect fit errors — sequential LUT composition " +
          "may not be accurate enough for recipes that use both effects at Strong simultaneously; " +
          "consider a joint fit if this matters for your recipes.",
      );
    }
  } else if (!existsSync(comboPath)) {
    console.log(
      "\n(No calib-cce-strong-fxblue-strong.jpg found — skipping the composed-effect validation check. " +
        "Not required, but recommended if any of your recipes use both effects at Strong together.)",
    );
  }

  console.log("\nDone.");
}

main();
