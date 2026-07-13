/**
 * Fujifilm .RAF files embed a full-size JPEG preview alongside the raw
 * sensor data (every RAF, regardless of camera generation, carries one —
 * it's what the camera's own screen/software use for quick display).
 * extractRafPreviewJpeg extracts that preview and runs it through the
 * existing WebGL pipeline — a reasonable "RAW support" baseline on the web,
 * where full Bayer/X-Trans sensor demosaicing would need a WASM-compiled
 * LibRaw build (a large third-party dependency, and X-Trans demosaicing is
 * CPU/memory-heavy enough that it's a real risk on mobile browsers).
 *
 * But that preview is already rendered through the camera's JPEG engine —
 * whatever film simulation/grain was dialed in at capture is baked into its
 * pixels and can't be removed, only compensated for numerically (see
 * src/lib/recipes/neutralize.ts). decodeNeutralRaf below is the real fix,
 * available only inside the native iOS app: it hands the RAF's actual
 * sensor data to Apple's CIRAWFilter (ios/App/App/RawDecoder/
 * RawDecoderPlugin.swift), which demosaics it directly — no film simulation
 * or grain baked in at all, so any recipe can be applied to a genuinely
 * clean base.
 *
 * The RAF header stores the exact byte offset and length of that embedded
 * JPEG as two big-endian uint32 fields at fixed positions (0x54 and 0x58) —
 * confirmed against exiftool's FujiFilm.pm source (its own RAFHeader tag
 * table comments these exact offsets) and libopenraw's RAF format docs, both
 * independent, actively-maintained reverse-engineerings of the format.
 * Reading those fields directly is what exiftool itself does to pull EXIF
 * out of a RAF — it's the correct extraction, not a byte-marker scan (an
 * earlier version of this function scanned for JPEG SOI/EOI marker bytes
 * instead, which is unreliable: a RAF's embedded preview JPEG typically
 * carries its own nested EXIF thumbnail with its own SOI/EOI markers, so a
 * naive scan can grab that instead of the real preview, or run off the end
 * of a bounded scan window on cameras with a larger preview).
 */

import { Capacitor } from "@capacitor/core";
import { RawDecoder } from "@/lib/raw/rawDecoderPlugin";

const RAF_MAGIC = "FUJIFILMCCD-RAW";
const JPEG_OFFSET_FIELD = 0x54; // big-endian uint32
const JPEG_LENGTH_FIELD = 0x58; // big-endian uint32

// Encoded in chunks rather than one `String.fromCharCode(...bytes)` spread —
// a 26MB+ RAF blows the JS engine's argument-count limit on a single spread.
const BASE64_CHUNK_SIZE = 0x8000;

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += BASE64_CHUNK_SIZE) {
    binary += String.fromCharCode(...bytes.subarray(i, i + BASE64_CHUNK_SIZE));
  }
  return btoa(binary);
}

function base64ToBlob(base64: string, type: string): Blob {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type });
}

export async function extractRafPreviewJpeg(file: File): Promise<Blob> {
  const headerBytes = new Uint8Array(await file.slice(0, RAF_MAGIC.length).arrayBuffer());
  const magic = new TextDecoder().decode(headerBytes);
  if (magic !== RAF_MAGIC) {
    throw new Error("This doesn't look like a Fujifilm .RAF file.");
  }

  const fieldsBuffer = await file.slice(JPEG_OFFSET_FIELD, JPEG_LENGTH_FIELD + 4).arrayBuffer();
  const fieldsView = new DataView(fieldsBuffer);
  const jpegOffset = fieldsView.getUint32(0, false); // false = big-endian
  const jpegLength = fieldsView.getUint32(4, false);

  // A handful of older/multi-shot RAF variants (e.g. GFX "M-RAW" mode) store
  // the preview elsewhere and leave these fields zero — not something the
  // X100VI produces, but worth failing clearly rather than slicing garbage.
  if (jpegOffset === 0 || jpegLength === 0 || jpegOffset + jpegLength > file.size) {
    throw new Error("Couldn't find an embedded preview image in this .RAF file.");
  }

  const jpegBytes = await file.slice(jpegOffset, jpegOffset + jpegLength).arrayBuffer();
  return new Blob([jpegBytes], { type: "image/jpeg" });
}

export function isRafFile(file: File): boolean {
  return /\.raf$/i.test(file.name);
}

/**
 * True RAW demosaic via the native iOS RawDecoder plugin — returns null on
 * web (no native platform to decode with) or if the decode itself fails
 * (e.g. an unsupported/corrupt file), so callers can fall back to
 * extractRafPreviewJpeg without treating either case as fatal.
 */
export async function decodeNeutralRaf(file: File): Promise<Blob | null> {
  if (!Capacitor.isNativePlatform()) return null;
  try {
    const base64 = arrayBufferToBase64(await file.arrayBuffer());
    const result = await RawDecoder.decodeNeutral({ data: base64 });
    return base64ToBlob(result.data, "image/jpeg");
  } catch {
    return null;
  }
}
