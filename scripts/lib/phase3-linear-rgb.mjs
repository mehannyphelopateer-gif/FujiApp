// Reader for Codex's Phase 3 linear-RGB calibration container
// (docs/phase3-linear-rgb-format.md) and for whichever X RAW Studio target
// format ends up being used (16-bit TIFF preferred per
// docs/phase3-export-protocol.md, 8-bit JPEG fallback if this X RAW Studio
// version doesn't support 16-bit TIFF export).
//
// Verified against Codex's real Shoot 127 sample (commit 2a0c745): magic,
// version, dimensions, channel/bit-depth fields, and payload size all
// match this parser's expectations exactly.

import { readFileSync } from "node:fs";
import sharp from "sharp";

const FJLRGB_MAGIC = "FJLRGB16";

/** Parses a .fjlrg file into { width, height, data } where data is a
 * Float32Array of RGB values normalized to 0..1 linear light (divided by
 * 65535, the container's uint16 range — NOT sRGB-gamma-encoded). */
export function loadFjlrgLinear(path) {
  const buf = readFileSync(path);
  const magic = buf.toString("ascii", 0, 8);
  if (magic !== FJLRGB_MAGIC) throw new Error(`${path}: bad magic "${magic}", expected "${FJLRGB_MAGIC}"`);
  const version = buf.readUInt16LE(8);
  if (version !== 1) throw new Error(`${path}: unsupported format version ${version}`);
  const headerBytes = buf.readUInt16LE(10);
  const width = buf.readUInt32LE(12);
  const height = buf.readUInt32LE(16);
  const channels = buf.readUInt16LE(20);
  const bitsPerChannel = buf.readUInt16LE(22);
  const payloadByteCount = buf.readUInt32LE(24);
  const flags = buf.readUInt32LE(28);
  const isLinear = !!(flags & 1);
  const hasSrgbPrimaries = !!(flags & 2);

  if (channels !== 3) throw new Error(`${path}: expected 3 channels, got ${channels}`);
  if (bitsPerChannel !== 16) throw new Error(`${path}: expected 16 bits/channel, got ${bitsPerChannel}`);
  if (!isLinear) throw new Error(`${path}: flags indicate non-linear transfer — this loader assumes linear light throughout`);
  const expectedPayload = width * height * channels * 2;
  if (payloadByteCount !== expectedPayload || buf.length - headerBytes !== expectedPayload) {
    throw new Error(`${path}: payload size mismatch (declared ${payloadByteCount}, expected ${expectedPayload}, actual ${buf.length - headerBytes})`);
  }

  const sampleCount = width * height * channels;
  const data = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    data[i] = buf.readUInt16LE(headerBytes + i * 2) / 65535;
  }
  return { width, height, data, hasSrgbPrimaries };
}

/** Loads an X RAW Studio target (16-bit TIFF preferred, 8-bit JPEG
 * fallback per the locked export protocol) as linear-light RGB, matching
 * loadFjlrgLinear's output shape. sRGB inputs (both 16-bit TIFF and 8-bit
 * JPEG from X RAW Studio are sRGB-gamma-encoded, not linear) are converted
 * to linear light here so both sides of every comparison in this pipeline
 * are in the same linear space — never compare one linear array against
 * one gamma-encoded array directly. */
export async function loadXrawTargetLinear(path) {
  const isTiff = /\.tiff?$/i.test(path);
  const image = sharp(path).rotate();
  const meta = await image.metadata();
  const is16Bit = meta.depth === "ushort" || meta.depth === "short";

  if (isTiff && is16Bit) {
    // sharp/libvips preserves native bit depth through .raw() when the
    // pipeline hasn't been asked to reduce it — NOT YET VERIFIED against a
    // real X RAW Studio 16-bit TIFF (none exist in this repo as of this
    // writing, only Codex's browser-side sample). Verify this branch
    // against the first real X RAW Studio Phase 3 export before trusting
    // it for an actual fit.
    const { data, info } = await image.removeAlpha().raw({ depth: "ushort" }).toBuffer({ resolveWithObject: true });
    const sampleCount = info.width * info.height * info.channels;
    const out = new Float32Array(sampleCount);
    const view = new Uint16Array(data.buffer, data.byteOffset, sampleCount);
    for (let i = 0; i < sampleCount; i++) out[i] = srgbGammaToLinear(view[i] / 65535);
    return { width: info.width, height: info.height, data: out, sourceBitDepth: 16 };
  }

  // 8-bit fallback (JPEG, or an 8-bit TIFF) — same precision ceiling as
  // every Phase 2 comparison; only used if 16-bit TIFF genuinely isn't
  // available from this X RAW Studio version.
  const { data, info } = await image.removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const sampleCount = info.width * info.height * info.channels;
  const out = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) out[i] = srgbGammaToLinear(data[i] / 255);
  return { width: info.width, height: info.height, data: out, sourceBitDepth: 8 };
}

/** Standard sRGB EOTF (gamma-encoded -> linear light), IEC 61966-2-1. */
function srgbGammaToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}
