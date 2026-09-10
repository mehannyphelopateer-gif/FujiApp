/**
 * Fujifilm .RAF files embed a full-size JPEG preview alongside the raw
 * sensor data (every RAF, regardless of camera generation, carries one —
 * it's what the camera's own screen/software use for quick display).
 * Historically, the web path extracted that preview and ran it through the
 * existing WebGL pipeline. That is useful as a last-resort fallback, but it
 * cannot truly replace a source recipe: film simulation, grain and the
 * source white balance are already baked into those pixels. The normal web
 * path now uses LibRaw compiled to WebAssembly to demosaic the sensor data in
 * a worker, entirely locally. That gives the recipe renderer a neutral RAW
 * base on macOS and browsers, just as CIRAWFilter does in the native app.
 *
 * But that preview is already rendered through the camera's JPEG engine —
 * whatever film simulation/grain was dialed in at capture is baked into its
 * pixels and can't be removed, only compensated for numerically (see
 * src/lib/recipes/neutralize.ts). decodeNeutralRaf below is the real fix:
 * LibRaw WebAssembly demosaics in a browser worker and native iOS uses
 * Apple's CIRAWFilter. Both use actual sensor data rather than the baked
 * preview, so recipes start from a genuinely clean base.
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

const WEB_RAW_MAX_DIMENSION = 4096;

export interface NeutralRafDecodeResult {
  blob: Blob | null;
  /** Present when the app had to fall back to the RAF's embedded JPEG. */
  error?: string;
}

/** Converts LibRaw's RGB/RGBA byte buffer into a JPEG-sized preview without sending the RAF off-device. */
async function libRawImageToBlob(
  image: { width: number; height: number; colors: number; bits: number; data: Uint8Array | Uint16Array },
): Promise<Blob> {
  if (image.bits !== 8 || (image.colors !== 3 && image.colors !== 4)) {
    throw new Error("The RAW decoder returned an unsupported pixel format.");
  }

  const rgba = new Uint8ClampedArray(image.width * image.height * 4);
  for (let source = 0, target = 0; target < rgba.length; source += image.colors, target += 4) {
    rgba[target] = image.data[source];
    rgba[target + 1] = image.data[source + 1];
    rgba[target + 2] = image.data[source + 2];
    rgba[target + 3] = image.colors === 4 ? image.data[source + 3] : 255;
  }

  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = image.width;
  sourceCanvas.height = image.height;
  const sourceContext = sourceCanvas.getContext("2d");
  if (!sourceContext) throw new Error("Canvas 2D context unavailable for RAW rendering.");
  sourceContext.putImageData(new ImageData(rgba, image.width, image.height), 0, 0);

  const scale = Math.min(1, WEB_RAW_MAX_DIMENSION / Math.max(image.width, image.height));
  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = Math.round(image.width * scale);
  outputCanvas.height = Math.round(image.height * scale);
  const outputContext = outputCanvas.getContext("2d");
  if (!outputContext) throw new Error("Canvas 2D context unavailable for RAW rendering.");
  outputContext.drawImage(sourceCanvas, 0, 0, outputCanvas.width, outputCanvas.height);

  return new Promise<Blob>((resolve, reject) => {
    outputCanvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Failed to encode RAW preview."))), "image/jpeg", 0.95);
  });
}

/**
 * Calibration-only override: `?rawCalibrationWb=camera` on the page URL
 * makes the browser LibRaw decode use the RAF's own as-shot white balance
 * instead of the shipped default (no WB at all — left entirely to the
 * app's own shader-level WB stage). This exists only so a calibration
 * capture session can produce a browser-baseline export whose WB matches
 * an X RAW Studio "As Shot" export with zero guessing on either side —
 * comparing two independent Auto-WB guesses was confirmed to blow up the
 * base-normalization fit on any scene the camera's real auto-WB handled
 * differently than the app's own gray-world approximation (see
 * scripts/derive-libraw-base-normalization.mjs's header comment). Never
 * surfaced in the UI and never read outside this file — a normal user has
 * no way to set it, and the shipped Preview decode is unaffected.
 */
function isCalibrationCameraWbRequested(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("rawCalibrationWb") === "camera";
}

/**
 * Calibration-only LibRaw highlight-recovery override. The normal Preview
 * path deliberately remains mode 0 (LibRaw's default hard clip); this lets
 * controlled RAF/X RAW Studio comparisons test whether a recovery mode can
 * account for errors that correlate with genuinely clipped highlights.
 */
function calibrationHighlightMode(): number {
  if (typeof window === "undefined") return 0;
  const params = new URLSearchParams(window.location.search);
  if (params.get("rawCalibrationWb") !== "camera") return 0;
  const raw = params.get("rawCalibrationHighlight");
  if (raw === null) return 0;
  const mode = Number(raw);
  return Number.isInteger(mode) && mode >= 0 && mode <= 9 ? mode : 0;
}

function isCalibrationMetadataInspectionRequested(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("rawCalibrationInspect") === "metadata";
}

function isCalibrationRawFeatureInspectionRequested(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("rawCalibrationInspect") === "raw-features";
}

function isCalibrationRawFeatureOnlyRequested(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("rawCalibrationFeatureOnly") === "1";
}

async function calibrationPlaceholderBlob(): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas 2D context unavailable for RAW calibration.");
  context.fillStyle = "black";
  context.fillRect(0, 0, 1, 1);
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Failed to create RAW calibration placeholder."))), "image/jpeg");
  });
}

/**
 * Summary of the visible, undemosaiced sensor mosaic. Values are normalized
 * against LibRaw's camera black/white levels and deliberately avoid any RGB
 * decode, camera matrix, film simulation, or output tone curve.
 */
export interface RawSensorFeatures {
  sampleCount: number;
  /** LibRaw's declared black level (zero for these Fuji RAFs). */
  blackLevel: number;
  /** Robust floor estimated from the raw mosaic when the declared level is zero/unusable. */
  effectiveBlackLevel: number;
  whiteLevel: number;
  shadowThreshold: number;
  nearBlackFraction: number;
  nearWhiteFraction98: number;
  nearWhiteFraction99: number;
  nearWhiteFraction995: number;
  percentiles: Record<"p01" | "p05" | "p50" | "p95" | "p99" | "p995" | "p999", number>;
}

function summarizeRawSensorData(
  raw: { raw_width: number; top_margin: number; left_margin: number; width: number; height: number; data: Uint16Array },
  colorData: { black?: number; maximum?: number; data_maximum?: number } | undefined,
): RawSensorFeatures {
  const blackLevel = colorData?.black ?? 0;
  // `maximum` is LibRaw's camera white level. Fall back to the actual raw
  // maximum only for an unusual file where the wrapper does not expose it.
  let observedMaximum = 0;
  const { data, raw_width: rawWidth, top_margin: top, left_margin: left, width, height } = raw;
  for (let y = 0; y < height; y++) {
    const row = (y + top) * rawWidth + left;
    for (let x = 0; x < width; x++) observedMaximum = Math.max(observedMaximum, data[row + x]);
  }
  const whiteLevel = Math.max(blackLevel + 1, colorData?.maximum ?? colorData?.data_maximum ?? observedMaximum);
  const range = whiteLevel - blackLevel;
  const histogram = new Uint32Array(1024);
  let sampleCount = 0;
  let nearWhite98 = 0;
  let nearWhite99 = 0;
  let nearWhite995 = 0;

  for (let y = 0; y < height; y++) {
    const row = (y + top) * rawWidth + left;
    for (let x = 0; x < width; x++) {
      const normalized = Math.min(1, Math.max(0, (data[row + x] - blackLevel) / range));
      histogram[Math.min(histogram.length - 1, Math.floor(normalized * histogram.length))]++;
      sampleCount++;
      if (normalized >= 0.98) nearWhite98++;
      if (normalized >= 0.99) nearWhite99++;
      if (normalized >= 0.995) nearWhite995++;
    }
  }

  function percentile(q: number): number {
    const target = Math.max(0, Math.ceil(sampleCount * q));
    let cumulative = 0;
    for (let index = 0; index < histogram.length; index++) {
      cumulative += histogram[index];
      if (cumulative >= target) return (index + 0.5) / histogram.length;
    }
    return 1;
  }

  // Some Fuji RAFs expose a zero declared black level even though their
  // untouched mosaic retains a stable offset around 1k DN. Using that zero
  // would make every "near black" measurement empty. Estimate a robust
  // sensor floor from p01, then classify the first 1% of usable range above
  // it as shadows. The declared level remains in the sidecar for auditing.
  const p01 = percentile(0.01);
  const effectiveBlackLevel = blackLevel > 0 ? blackLevel : blackLevel + p01 * range;
  const shadowThreshold = effectiveBlackLevel + (whiteLevel - effectiveBlackLevel) * 0.01;
  const shadowNormalized = Math.min(1, Math.max(0, (shadowThreshold - blackLevel) / range));
  let nearBlack = 0;
  const lastShadowBin = Math.min(histogram.length - 1, Math.floor(shadowNormalized * histogram.length));
  for (let index = 0; index <= lastShadowBin; index++) nearBlack += histogram[index];

  return {
    sampleCount,
    blackLevel,
    effectiveBlackLevel,
    whiteLevel,
    shadowThreshold,
    nearBlackFraction: nearBlack / sampleCount,
    nearWhiteFraction98: nearWhite98 / sampleCount,
    nearWhiteFraction99: nearWhite99 / sampleCount,
    nearWhiteFraction995: nearWhite995 / sampleCount,
    percentiles: {
      p01, p05: percentile(0.05), p50: percentile(0.5), p95: percentile(0.95),
      p99: percentile(0.99), p995: percentile(0.995), p999: percentile(0.999),
    },
  };
}

/** Browser-only RAW demosaic. LibRaw's worker keeps the CPU-heavy X-Trans work off the UI thread. */
async function decodeNeutralRafInBrowser(file: File): Promise<Blob> {
  const { default: LibRaw } = await import("libraw-wasm");
  const decoder = new LibRaw();
  try {
    await decoder.open(new Uint8Array(await file.arrayBuffer()), {
      // Half-size avoids allocating a 150MB+ RGBA canvas for a 40MP RAF;
      // it still yields roughly 3864px on the long edge for an X100VI, close
      // to the renderer's own 4096px texture ceiling. A full-size X-Trans
      // test on DSCF0752 took 24s and did not improve the X RAW Studio MAE.
      halfSize: true,
      outputBps: 8,
      outputColor: 1, // sRGB
      useCameraMatrix: 1,
      useCameraWb: isCalibrationCameraWbRequested(),
      useAutoWb: false,
      highlight: calibrationHighlightMode(),
      userQual: 3,
      useFujiRotate: -1,
      fbddNoiserd: 0,
    });
    if (isCalibrationMetadataInspectionRequested()) {
      // LibRaw maps Fuji's capture-time Dynamic Range tags into metadata.fuji.
      // This diagnostic is query-gated so it cannot affect normal decoding.
      console.info("[FujiApp calibration metadata]", (await decoder.metadata(true))?.fuji ?? null);
    }
    if (isCalibrationRawFeatureInspectionRequested()) {
      const [rawSensor, metadata] = await Promise.all([decoder.rawImageData(), decoder.metadata(true)]);
      if (!rawSensor) throw new Error("The RAW decoder returned no undemosaiced sensor data.");
      console.info(
        "[FujiApp calibration raw features]",
        summarizeRawSensorData(rawSensor, metadata?.color_data),
      );
      // Corpus scans need sensor features only. Avoiding imageData() here
      // skips demosaic/color processing while still exercising the exact
      // same LibRaw unpack/raw-data path used by Preview.
      if (isCalibrationRawFeatureOnlyRequested()) return calibrationPlaceholderBlob();
    }
    const image = await decoder.imageData();
    if (!image) throw new Error("The RAW decoder returned no pixel data.");
    return await libRawImageToBlob(image);
  } finally {
    decoder.dispose();
  }
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
 * True RAW demosaic: native CIRAWFilter on iOS, LibRaw WebAssembly on web.
 * Returns null only when neither decoder can process the file, letting the
 * caller retain its embedded-preview fallback for unsupported/corrupt RAFs.
 */
export async function decodeNeutralRafWithDiagnostics(file: File): Promise<NeutralRafDecodeResult> {
  try {
    if (Capacitor.isNativePlatform()) {
      const base64 = arrayBufferToBase64(await file.arrayBuffer());
      const result = await RawDecoder.decodeNeutral({ data: base64 });
      return { blob: base64ToBlob(result.data, "image/jpeg") };
    }
    return { blob: await decodeNeutralRafInBrowser(file) };
  } catch (error) {
    return {
      blob: null,
      error: error instanceof Error ? error.message : "The local RAW decoder failed.",
    };
  }
}

/** Backwards-compatible convenience wrapper for callers that only need a decoded blob. */
export async function decodeNeutralRaf(file: File): Promise<Blob | null> {
  return (await decodeNeutralRafWithDiagnostics(file)).blob;
}
