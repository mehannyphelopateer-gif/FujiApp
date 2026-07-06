/**
 * Fujifilm .RAF files embed a full-size JPEG preview alongside the raw
 * sensor data (every RAF, regardless of camera generation, carries one —
 * it's what the camera's own screen/software use for quick display). Since
 * this app's whole purpose is applying recipes as a *preview*, extracting
 * that embedded JPEG and running it through the existing WebGL pipeline is
 * the right-sized version of "RAW support" here — full Bayer/X-Trans sensor
 * demosaicing would need a WASM-compiled LibRaw build (a large third-party
 * dependency), which is out of scope for a recipe previewer.
 *
 * Extraction works by scanning for real JPEG marker bytes (SOI+APP0/APP1 ...
 * EOI) rather than trusting hardcoded field offsets, since RAF's header
 * layout has shifted across camera/firmware generations — a marker scan is
 * self-verifying and works across those variants without needing the exact
 * spec for each one.
 */

const RAF_MAGIC = "FUJIFILMCCD-RAW";
// Real embedded previews always sit near the start of the file, well before
// the (much larger) raw sensor data block — bounding the scan keeps this
// fast and avoids any theoretical false-positive marker match inside actual
// pixel data.
const SCAN_WINDOW_BYTES = 4 * 1024 * 1024;

export async function extractRafPreviewJpeg(file: File): Promise<Blob> {
  const headerBytes = new Uint8Array(await file.slice(0, RAF_MAGIC.length).arrayBuffer());
  const magic = new TextDecoder().decode(headerBytes);
  if (magic !== RAF_MAGIC) {
    throw new Error("This doesn't look like a Fujifilm .RAF file.");
  }

  const scanLength = Math.min(file.size, SCAN_WINDOW_BYTES);
  const bytes = new Uint8Array(await file.slice(0, scanLength).arrayBuffer());

  for (let i = 0; i < bytes.length - 4; i++) {
    const isSoiWithApp =
      bytes[i] === 0xff && bytes[i + 1] === 0xd8 && bytes[i + 2] === 0xff && (bytes[i + 3] === 0xe0 || bytes[i + 3] === 0xe1);
    if (!isSoiWithApp) continue;

    for (let j = i + 4; j < bytes.length - 1; j++) {
      if (bytes[j] === 0xff && bytes[j + 1] === 0xd9) {
        return new Blob([bytes.slice(i, j + 2)], { type: "image/jpeg" });
      }
    }
  }

  throw new Error("Couldn't find an embedded preview image in this .RAF file.");
}

export function isRafFile(file: File): boolean {
  return /\.raf$/i.test(file.name);
}
