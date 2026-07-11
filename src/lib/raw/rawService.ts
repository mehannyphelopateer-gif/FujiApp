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

const RAF_MAGIC = "FUJIFILMCCD-RAW";
const JPEG_OFFSET_FIELD = 0x54; // big-endian uint32
const JPEG_LENGTH_FIELD = 0x58; // big-endian uint32

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
