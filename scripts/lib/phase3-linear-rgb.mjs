// Reader for Codex's Phase 3 linear-RGB calibration container
// (docs/phase3-linear-rgb-format.md) and for whichever X RAW Studio target
// format ends up being used (16-bit TIFF preferred per
// docs/phase3-export-protocol.md, 8-bit JPEG fallback if this X RAW Studio
// version doesn't support 16-bit TIFF export).
//
// Verified against Codex's real Shoot 127 sample (commit 2a0c745): magic,
// version, dimensions, channel/bit-depth fields, and payload size all
// match this parser's expectations exactly.

import { readFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import sharp from "sharp";
import { PNG } from "pngjs";

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
  // failOn: "none" — X RAW Studio's real TIFF exports carry a Copyright
  // tag with an embedded null byte (a benign, common quirk of some RAW-
  // converter TIFF writers, confirmed against a real Shoot 127 export),
  // which libvips otherwise treats as fatal by default. This is a
  // metadata cosmetic issue, not pixel-data corruption — verified by
  // successfully decoding the same file once this is relaxed.
  const meta = await sharp(path, { failOn: "none" }).metadata();
  const is16Bit = meta.depth === "ushort" || meta.depth === "short";

  if (isTiff && is16Bit) return loadTiff16BitLinear(path, meta);

  // 8-bit fallback (JPEG, or an 8-bit TIFF) — same precision ceiling as
  // every Phase 2 comparison; only used if 16-bit TIFF genuinely isn't
  // available from this X RAW Studio version. sharp's 8-bit raw() path is
  // reliable (unlike its 16-bit path — see loadTiff16BitLinear's comment).
  const { data, info } = await sharp(path, { failOn: "none" }).rotate().removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const sampleCount = info.width * info.height * info.channels;
  const out = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) out[i] = srgbGammaToLinear(data[i] / 255);
  return { width: info.width, height: info.height, data: out, sourceBitDepth: 8 };
}

/**
 * Reads a 16-bit TIFF preserving full precision. sharp/libvips's `.raw()`
 * silently reduces to 8-bit for this pipeline shape no matter what `depth`
 * option is passed to `.raw()` (confirmed directly against a real X RAW
 * Studio Phase 3 export: `.raw({depth:'ushort'})` returned values capped
 * at 255, while `sharp().stats()` on the same file correctly reported the
 * full 0-65535 range — so the file itself has real 16-bit data, sharp's
 * raw-extraction path just doesn't preserve it here). Workaround, each
 * piece independently verified against the real file:
 *   1. `sips` (macOS, always present) converts TIFF -> PNG preserving the
 *      full 16-bit samples (verified: `sips -g bitsPerSample` on the
 *      output confirms 16).
 *   2. `pngjs` with `{ skipRescale: true }` is the one option that reads
 *      those samples back as true 0-65535 values — pngjs's default read
 *      path silently rescales 16-bit PNG data down to 8-bit for its
 *      output buffer (found by reading format-normaliser.js directly);
 *      skipRescale bypasses that.
 *   3. `sips`'s PNG output does NOT bake in the source's EXIF orientation
 *      (verified: output pixel dimensions matched the RAW, un-rotated
 *      TIFF storage, not the auto-oriented display dimensions) — so the
 *      rotation implied by the original TIFF's orientation tag is applied
 *      here manually, from the metadata already read by the caller.
 * This is macOS-specific (calls the `sips` binary) — acceptable since
 * every calibration script in this project already assumes macOS tooling
 * (e.g. disk-space handling, file symlinking conventions).
 */
function loadTiff16BitLinear(path, meta) {
  const tempPngPath = join(tmpdir(), `phase3-tiff-${process.pid}-${Date.now()}.png`);
  try {
    execFileSync("sips", ["-s", "format", "png", path, "--out", tempPngPath], { stdio: "pipe" });
    const png = PNG.sync.read(readFileSync(tempPngPath), { skipRescale: true });
    const rgba = new Uint16Array(png.data.buffer, png.data.byteOffset, png.data.length / 2);

    // Drop the alpha channel pngjs always includes, converting RGBA16 -> RGB16 linear.
    const rawWidth = png.width, rawHeight = png.height;
    const rgb = new Float32Array(rawWidth * rawHeight * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      rgb[j] = srgbGammaToLinear(rgba[i] / 65535);
      rgb[j + 1] = srgbGammaToLinear(rgba[i + 1] / 65535);
      rgb[j + 2] = srgbGammaToLinear(rgba[i + 2] / 65535);
    }

    const rotated = applyExifRotation(rgb, rawWidth, rawHeight, meta.orientation ?? 1);
    return { width: rotated.width, height: rotated.height, data: rotated.data, sourceBitDepth: 16 };
  } finally {
    try { unlinkSync(tempPngPath); } catch { /* best-effort cleanup */ }
  }
}

/** Applies the rotation implied by a standard EXIF orientation value (1-8)
 * to an interleaved RGB Float32Array. Only handles the four pure-rotation
 * cases (1, 3, 6, 8) — camera-shot photos overwhelmingly land in one of
 * these; the four mirrored cases (2, 4, 5, 7, from certain scan/software
 * pipelines, not typical direct camera capture) throw rather than silently
 * mis-orient a comparison. */
function applyExifRotation(data, width, height, orientation) {
  if (orientation === 1) return { data, width, height };
  if (orientation === 3) {
    const out = new Float32Array(data.length);
    const total = width * height;
    for (let i = 0; i < total; i++) {
      const srcBase = i * 3, dstBase = (total - 1 - i) * 3;
      out[dstBase] = data[srcBase]; out[dstBase + 1] = data[srcBase + 1]; out[dstBase + 2] = data[srcBase + 2];
    }
    return { data: out, width, height };
  }
  if (orientation === 6 || orientation === 8) {
    // 6 = rotate 90 CW, 8 = rotate 270 CW (90 CCW) — output dimensions swap.
    const outWidth = height, outHeight = width;
    const out = new Float32Array(data.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcBase = (y * width + x) * 3;
        const [dstX, dstY] = orientation === 6 ? [outWidth - 1 - y, x] : [y, outHeight - 1 - x];
        const dstBase = (dstY * outWidth + dstX) * 3;
        out[dstBase] = data[srcBase]; out[dstBase + 1] = data[srcBase + 1]; out[dstBase + 2] = data[srcBase + 2];
      }
    }
    return { data: out, width: outWidth, height: outHeight };
  }
  throw new Error(`Unsupported EXIF orientation ${orientation} (mirrored orientations 2/4/5/7 aren't handled — unusual for a direct camera capture; verify the source file).`);
}

/** Standard sRGB EOTF (gamma-encoded -> linear light), IEC 61966-2-1. */
function srgbGammaToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** Area-average resize of interleaved linear-RGB Float32Array data to an
 * exact target size — used to bring the X RAW Studio target (naturally
 * full sensor resolution) down to the browser export's resolution (the
 * .fjlrg container is already downsampled to a 1536px long edge at
 * export time), so every per-pixel comparison in the fitting pipeline
 * lines up on the same pixel grid. Averaging in LINEAR light (not on
 * gamma-encoded values) is the physically correct way to downsample —
 * mirrors the same area-average approach already used browser-side in
 * rawService.ts's downsampleLinearRgb, just parameterized by an exact
 * target size instead of a max-dimension cap. */
export function resizeLinearRgbTo(image, targetWidth, targetHeight) {
  const { width, height, data } = image;
  if (width === targetWidth && height === targetHeight) return image;
  const output = new Float32Array(targetWidth * targetHeight * 3);
  for (let outputY = 0; outputY < targetHeight; outputY++) {
    const sourceY0 = Math.floor((outputY * height) / targetHeight);
    const sourceY1 = Math.max(sourceY0 + 1, Math.floor(((outputY + 1) * height) / targetHeight));
    for (let outputX = 0; outputX < targetWidth; outputX++) {
      const sourceX0 = Math.floor((outputX * width) / targetWidth);
      const sourceX1 = Math.max(sourceX0 + 1, Math.floor(((outputX + 1) * width) / targetWidth));
      const count = (sourceX1 - sourceX0) * (sourceY1 - sourceY0);
      let sumR = 0, sumG = 0, sumB = 0;
      for (let sourceY = sourceY0; sourceY < sourceY1; sourceY++) {
        for (let sourceX = sourceX0; sourceX < sourceX1; sourceX++) {
          const offset = (sourceY * width + sourceX) * 3;
          sumR += data[offset]; sumG += data[offset + 1]; sumB += data[offset + 2];
        }
      }
      const destination = (outputY * targetWidth + outputX) * 3;
      output[destination] = sumR / count;
      output[destination + 1] = sumG / count;
      output[destination + 2] = sumB / count;
    }
  }
  return { width: targetWidth, height: targetHeight, data: output };
}
