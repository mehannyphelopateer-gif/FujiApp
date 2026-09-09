#!/usr/bin/env node
// Compares a real camera/X RAW Studio render to a FujiApp browser render.
// Both inputs are auto-oriented and resampled to a shared, proportional
// working size, so different export resolutions do not obscure color/tone
// error. This is a validation tool only: it never writes either image.
//
// Usage: node scripts/compare-render-pair.mjs <reference.jpg> <candidate.jpg>

import sharp from "sharp";

const [referencePath, candidatePath] = process.argv.slice(2);
if (!referencePath || !candidatePath) {
  console.error("Usage: node scripts/compare-render-pair.mjs <reference.jpg> <candidate.jpg>");
  process.exit(1);
}

const MAX_PIXELS = 512 * 768;
const LUMA_BINS = 5;

function luma(rgb) {
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

async function metadataAfterOrientation(path) {
  // metadata() reports the source dimensions even when rotate() is chained;
  // apply the EXIF orientation swap ourselves for orientations 5–8.
  const metadata = await sharp(path).metadata();
  if (!metadata.width || !metadata.height) throw new Error(`Could not read dimensions for ${path}`);
  const rotated = (metadata.orientation ?? 1) >= 5;
  return {
    width: rotated ? metadata.height : metadata.width,
    height: rotated ? metadata.width : metadata.height,
  };
}

function commonSize(reference, candidate) {
  const aspect = reference.width / reference.height;
  const candidateAspect = candidate.width / candidate.height;
  if (Math.abs(aspect - candidateAspect) > 0.01) {
    throw new Error(
      `Images have different aspect ratios after orientation (${aspect.toFixed(4)} vs ${candidateAspect.toFixed(4)}); refusing to compare unrelated framing.`,
    );
  }

  const scale = Math.min(1, Math.sqrt(MAX_PIXELS / (reference.width * reference.height)));
  return {
    width: Math.max(1, Math.round(reference.width * scale)),
    height: Math.max(1, Math.round(reference.height * scale)),
  };
}

async function loadPixels(path, size) {
  return sharp(path).rotate().resize(size.width, size.height, { fit: "fill" }).removeAlpha().raw().toBuffer();
}

function formatRgb(values, count) {
  return values.map((value) => (value / count).toFixed(2)).join(", ");
}

async function main() {
  const [referenceMeta, candidateMeta] = await Promise.all([
    metadataAfterOrientation(referencePath),
    metadataAfterOrientation(candidatePath),
  ]);
  const size = commonSize(referenceMeta, candidateMeta);
  const [reference, candidate] = await Promise.all([loadPixels(referencePath, size), loadPixels(candidatePath, size)]);

  let mae = 0;
  let squaredError = 0;
  const bins = Array.from({ length: LUMA_BINS }, () => ({ count: 0, meanError: [0, 0, 0], mae: [0, 0, 0] }));

  for (let i = 0; i < reference.length; i += 3) {
    const referenceRgb = [reference[i], reference[i + 1], reference[i + 2]];
    const bin = bins[Math.min(LUMA_BINS - 1, Math.floor((luma(referenceRgb) / 255) * LUMA_BINS))];
    bin.count++;

    for (let channel = 0; channel < 3; channel++) {
      const error = candidate[i + channel] - reference[i + channel];
      mae += Math.abs(error);
      squaredError += error ** 2;
      bin.meanError[channel] += error;
      bin.mae[channel] += Math.abs(error);
    }
  }

  const channelCount = reference.length;
  console.log(`Reference: ${referencePath}`);
  console.log(`Candidate: ${candidatePath}`);
  console.log(`Comparison: ${size.width}×${size.height} after EXIF orientation`);
  console.log(`MAE: ${(mae / channelCount).toFixed(2)} / 255`);
  console.log(`RMSE: ${Math.sqrt(squaredError / channelCount).toFixed(2)} / 255`);
  console.log("\nReference-luma bin | pixels | mean candidate − reference RGB | channel MAE RGB");
  for (let index = 0; index < bins.length; index++) {
    const bin = bins[index];
    const low = (index / LUMA_BINS).toFixed(1);
    const high = ((index + 1) / LUMA_BINS).toFixed(1);
    console.log(
      `${low}–${high} | ${bin.count} | ${formatRgb(bin.meanError, bin.count)} | ${formatRgb(bin.mae, bin.count)}`,
    );
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
