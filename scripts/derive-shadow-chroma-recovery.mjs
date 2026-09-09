#!/usr/bin/env node
// Measures how much real blue-channel signal this app's native RAW decode
// loses in shadows relative to the real camera's own color science, and
// derives a calibrated additive recovery curve — a NEW correction, separate
// from (and applied earlier in the pipeline than) the existing WB/AWB/
// shadow-tone machinery.
//
// Root cause (see ~/.claude/plans/indexed-inventing-wren.md's Phase 3
// "Round 7"/"Round 10"): CIRAWFilter's chroma noise reduction (now disabled,
// see RawDecoderPlugin.swift) still leaves the neutral decode's blue
// channel hard-clipped to exactly 0 in a large fraction of true-shadow
// pixels — real color information a real Fuji JPEG conversion of the same
// RAF clearly preserves. No MULTIPLICATIVE gain (WB shift, AWB, mode) can
// ever recover a channel that's already exactly 0 (0 * anything = 0), so
// this needs a genuinely different, ADDITIVE correction.
//
// Compares this app's own neutral decode (`<shoot>/app-baseline.jpg`, or
// Shoot 6's differently-named `baseline-provia.jpg`) against a real X RAW
// Studio conversion of the IDENTICAL RAF at the same settings (Provia, Auto
// WB, no shift — `<shoot>/xraw-baseline.jpg` / `DSCF0752(5).jpg`), pooled
// across every shoot folder that has both, exactly the same "don't trust a
// single scene" principle as every other Phase 3 axis (see the plan doc's
// Round 6 for why this mattered for white balance).
//
// The two images are decoded at different resolutions (this app's neutral
// decode vs. X RAW Studio's own export size) but represent the same
// framing, so samples are taken at matching FRACTIONAL coordinates across
// the frame, not raw pixel indices.
//
// Usage: node scripts/derive-shadow-chroma-recovery.mjs [input-dir] [output-file]

import sharp from "sharp";
import { readdirSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const inputDir = process.argv[2] ?? join(__dirname, "..", "calibration-input");
const outputFile =
  process.argv[3] ?? join(__dirname, "..", "src", "engine", "webgl", "generated", "shadowChromaRecovery.ts");

// Only the shadow range needs correcting — everything above this luma
// already matches the real camera closely (confirmed directly: midtone
// luma ratios of 1.0-1.1 across every real-photo test this session, vs.
// 0.5-0.9 in true shadows). Control points beyond this range are pinned to
// zero lift so a normal, well-exposed photo's midtones/highlights are
// completely unaffected.
const LUMA_BUCKET_EDGES = [0, 0.05, 0.1, 0.15, 0.2, 0.3];
const GRID_STEP = 0.03; // fractional-coordinate sampling density across the frame

// A handful of candidate (app-baseline, xraw-baseline) filename pairs —
// Shoot 6 predates the app-baseline.jpg/xraw-baseline.jpg naming
// convention adopted from Shoot 8 onward.
const BASELINE_PAIR_NAMES = [
  ["app-baseline.jpg", "xraw-baseline.jpg"],
  ["baseline-provia.jpg", "DSCF0752(5).jpg"],
];

function findShootFolders(dir) {
  const found = [];
  function walk(current) {
    let entries;
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    for (const [appName, xrawName] of BASELINE_PAIR_NAMES) {
      if (entries.some((e) => e.name === appName) && entries.some((e) => e.name === xrawName)) {
        found.push({ folder: current, appName, xrawName });
        break;
      }
    }
    for (const entry of entries) {
      if (entry.isDirectory()) walk(join(current, entry.name));
    }
  }
  walk(dir);
  return found;
}

async function loadRotated(path) {
  const { data, info } = await sharp(path).rotate().raw().toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height, channels: info.channels };
}

function sampleAt(img, fx, fy) {
  const x = Math.min(img.width - 1, Math.floor(fx * img.width));
  const y = Math.min(img.height - 1, Math.floor(fy * img.height));
  const i = (y * img.width + x) * img.channels;
  return [img.data[i] / 255, img.data[i + 1] / 255, img.data[i + 2] / 255];
}

function luma(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function bucketIndexFor(appLuma) {
  for (let i = LUMA_BUCKET_EDGES.length - 1; i >= 0; i--) {
    if (appLuma >= LUMA_BUCKET_EDGES[i]) return i;
  }
  return 0;
}

async function main() {
  const pairs = findShootFolders(inputDir);
  if (pairs.length === 0) {
    console.error(
      `No shoot folder under ${inputDir} has a recognized app/xraw baseline pair — see this script's header ` +
        "comment for the expected filenames.",
    );
    process.exit(1);
  }
  console.log(`Found ${pairs.length} baseline pair(s):`);
  for (const { folder } of pairs) console.log(`  ${folder}`);

  // One bucket list per luma bin, collecting (xrawBlue - appBlue) deltas.
  const blueDeltaBuckets = LUMA_BUCKET_EDGES.map(() => []);

  for (const { folder, appName, xrawName } of pairs) {
    const app = await loadRotated(join(folder, appName));
    const xraw = await loadRotated(join(folder, xrawName));
    let sampleCount = 0;
    for (let fy = 0.05; fy <= 0.95; fy += GRID_STEP) {
      for (let fx = 0.05; fx <= 0.95; fx += GRID_STEP) {
        const [ar, ag, ab] = sampleAt(app, fx, fy);
        const appLuma = luma(ar, ag, ab);
        if (appLuma > LUMA_BUCKET_EDGES[LUMA_BUCKET_EDGES.length - 1]) continue; // outside the shadow range
        const [, , xb] = sampleAt(xraw, fx, fy);
        blueDeltaBuckets[bucketIndexFor(appLuma)].push(xb - ab);
        sampleCount++;
      }
    }
    console.log(`  ${folder}: ${sampleCount} shadow-range samples`);
  }

  console.log("\nPer-bucket sample counts:");
  for (let i = 0; i < LUMA_BUCKET_EDGES.length; i++) {
    console.log(`  luma>=${LUMA_BUCKET_EDGES[i].toFixed(2)}: n=${blueDeltaBuckets[i].length}`);
  }

  const points = LUMA_BUCKET_EDGES.map((edge, i) => {
    const delta = median(blueDeltaBuckets[i]);
    return { luma: edge, blueLift: delta === null ? 0 : Math.max(0, delta) };
  });

  // Anchor the top of the range to exactly 0 lift, whatever the nearest
  // bucket's raw median said — this is what makes the correction vanish
  // cleanly into the already-accurate midtone range instead of leaving a
  // visible seam at the boundary.
  points[points.length - 1].blueLift = 0;

  console.log("\nCalibrated shadow blue-recovery curve:");
  for (const point of points) {
    console.log(`  luma=${point.luma.toFixed(2)} -> blueLift=${point.blueLift.toFixed(4)} (0-1 scale)`);
  }

  mkdirSync(dirname(outputFile), { recursive: true });
  const contents = `// GENERATED by scripts/derive-shadow-chroma-recovery.mjs — do not hand-edit.
// Rerun after adding a new app-baseline.jpg/xraw-baseline.jpg pair to a
// calibration shoot folder. See ~/.claude/plans/indexed-inventing-wren.md's
// Phase 3 "Round 16" for the methodology and root-cause explanation.

export interface ShadowBlueLiftPoint {
  luma: number;
  blueLift: number;
}

export const SHADOW_BLUE_LIFT_CURVE: ShadowBlueLiftPoint[] = ${JSON.stringify(points)};
`;
  writeFileSync(outputFile, contents);
  console.log(`\nWrote ${outputFile}`);
}

main();
