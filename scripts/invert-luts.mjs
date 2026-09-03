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

import { mkdirSync, readdirSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { loadHaldSampler, fitHaldClut, writeLutPng } from "./lib/hald-clut-fitting.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const forwardLutDir = process.argv[2] ?? join(__dirname, "..", "public", "luts");
const outputDir = process.argv[3] ?? join(__dirname, "..", "public", "luts", "inverse");

const SKIP_SLUGS = new Set(["acros", "monochrome", "sepia", "identity"]);

const SAMPLE_DENSITY = 160; // samples per axis when probing the forward LUT — dense, cheap (pure function evaluation)
// Much finer than derive-luts-from-calibration.mjs's gridLevels=17: that
// value was chosen for real, sparse, noisy photo data where a coarse grid
// avoids overfitting noise. Here the "data" is the forward LUT itself —
// deterministic and densely samplable at any resolution — so there's no
// noise to protect against, and a coarse grid only costs real accuracy
// against non-affine curvature in the LUT being inverted (confirmed: the
// first pass at gridLevels=17 gave 0.05-0.10 mean round-trip error,
// worse for the stronger/more nonlinear sims like Velvia and Classic
// Negative — exactly what under-resolving curvature looks like).
const GRID_CONFIG = {
  gridLevels: 45,
  correctionClamp: 0.25,
  idwRadius: 3,
  idwEpsilon: 0.5,
  maxSamplesPerCell: 4000, // dense regular sampling fills cells far more evenly than real photos — safe to raise
};

const ROUND_TRIP_CHECKS = 500;

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
  const inverse = fitHaldClut(outputs, inputs, total, GRID_CONFIG);

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
