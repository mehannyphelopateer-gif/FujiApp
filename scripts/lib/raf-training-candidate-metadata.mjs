/**
 * Bounded, metadata-only field extraction for a Fujifilm RAF training *candidate*.
 *
 * FIELD EXTRACTION ONLY. This module does not admit, qualify, label, tier or score a
 * source, does not choose a WB policy or neutral kind, does not resolve capture DR,
 * and makes no claim about Fuji accuracy. Every value is the raw integer/string as
 * stored; a missing tag is an explicit `null`. Nothing is inferred, defaulted,
 * scaled or guessed. The caller (a separately frozen, scoped host) decides policy.
 *
 * What it reads (and nothing else):
 *   1. the 112-byte RAF header (magic, big-endian preview JPEG offset @84 / length @88);
 *   2. the preview JPEG's marker headers, bounded to MAX_JPEG_MARKERS markers, stopping
 *      AT the SOS marker (entropy-coded data is never read; SOS's own length is not read);
 *      for APP1 segments only the 6-byte identifier, and the full payload only of the
 *      single Exif APP1;
 *   3. via the UNCHANGED `readRafWbCoefficients` (./raf-wb-coefficients.mjs): the TIFF
 *      0xf00d / 0xf00e WB level records (its own bounded reads; pixels never touched).
 *
 * It never decodes an image, never loads pixels, never reads the whole RAF, never follows
 * strip/tile/RAFData pointers, and never dereferences unknown MakerNote fields.
 *
 * The single Exif APP1 is re-wrapped as a metadata-only JPEG (SOI + that APP1 + EOI) and
 * handed to ExifReader for the standard EXIF fields; the Fuji MakerNote IFD is parsed here
 * with explicit bounds, exact type/count checks and duplicate-conflict refusal.
 *
 * Input: Blob/File, or a duck-typed object with integer `size` and `slice(start, end)`
 * returning something with `arrayBuffer()`. No Node imports.
 *
 * Errors: structural/format problems throw CandidateMetadataError. Errors from the
 * unchanged WB helper propagate as its own RafWbError. TypeError for an invalid source.
 */
import ExifReader from "exifreader";
import { readRafWbCoefficients, RAF_MAGIC, RAF_HEADER_BYTES } from "./raf-wb-coefficients.mjs";

export const SCHEMA = "raf-training-candidate-metadata/v1";
export const JPEG_OFFSET_FIELD = 84;   // 0x54, big-endian uint32
export const JPEG_LENGTH_FIELD = 88;   // 0x58, big-endian uint32
export const MAX_JPEG_MARKERS = 256;   // markers scanned before SOS, SOS included
export const MAX_SEGMENT_BYTES = 65535; // JPEG segment length field maximum (includes the 2 length bytes)
export const MAX_READ_BYTES = MAX_SEGMENT_BYTES;
export const MAX_METADATA_BYTES = RAF_HEADER_BYTES + 2 + MAX_JPEG_MARKERS * (2 + 2 + 6) + MAX_SEGMENT_BYTES;
export const MAX_MAKERNOTE_ENTRIES = 512;
export const EXIF_IDENT = Object.freeze([0x45, 0x78, 0x69, 0x66, 0x00, 0x00]); // "Exif\0\0"
const FUJI_SIGNATURE = "FUJIFILM";

const T_SHORT = 3, T_LONG = 4, T_SLONG = 9;

/**
 * Fuji MakerNote fields extracted (tag ids per exiftool FujiFilm.pm, as already used by
 * scripts/dump-raf-exif.mjs and src/lib/exif/parseFujiMakerNotes.ts). `legacyName` is the
 * key the historical dump-raf-exif.mjs output used, so a host can compare field-for-field.
 * Allowed types are exact; any other type/count is refused, never coerced.
 */
export const FUJI_FIELDS = Object.freeze({
  whiteBalanceMode:        { tag: 0x1002, types: [T_SHORT],         count: 1, legacyName: "WhiteBalance" },
  colorTemperatureKelvin:  { tag: 0x1005, types: [T_SHORT, T_LONG], count: 1, legacyName: null },
  whiteBalanceFineTune:    { tag: 0x100a, types: [T_SLONG],         count: 2, legacyName: null },
  exposureCount:           { tag: 0x1032, types: [T_SHORT, T_LONG], count: 1, legacyName: "ExposureCount" },
  dynamicRange:            { tag: 0x1400, types: [T_SHORT, T_LONG], count: 1, legacyName: "DynamicRangeSetting" },
  dynamicRangeSetting:     { tag: 0x1402, types: [T_SHORT, T_LONG], count: 1, legacyName: "DynamicRangeSetting2" },
  developmentDynamicRange: { tag: 0x1403, types: [T_SHORT, T_LONG], count: 1, legacyName: "DevelopmentDynamicRange" },
  autoDynamicRange:        { tag: 0x140b, types: [T_SHORT, T_LONG], count: 1, legacyName: "AutoDynamicRange" },
  dRangePriority:          { tag: 0x1443, types: [T_SHORT, T_LONG], count: 1, legacyName: "DRangePriority" },
  dRangePriorityAuto:      { tag: 0x1444, types: [T_SHORT, T_LONG], count: 1, legacyName: "DRangePriorityAuto" },
  dRangePriorityFixed:     { tag: 0x1445, types: [T_SHORT, T_LONG], count: 1, legacyName: "DRangePriorityFixed" },
});

export class CandidateMetadataError extends Error {
  constructor(message) { super(message); this.name = "CandidateMetadataError"; }
}

function need(ok, message) {
  if (!ok) throw new CandidateMetadataError(message);
}

function checkSource(source) {
  if (!source || typeof source.size !== "number" || !Number.isSafeInteger(source.size) || source.size < 0 || typeof source.slice !== "function")
    throw new TypeError("Source must be a Blob/File or expose integer `size` and `slice(start, end)`");
}

/** Every read must lie inside the file and inside the caller-given window; per-read and total caps. */
class BoundedReader {
  constructor(source) {
    checkSource(source);
    this.source = source;
    this.size = source.size;
    this.bytesRead = 0;
    this.reads = 0;
  }

  async read(offset, length, lower = 0, upper = this.size) {
    need(Number.isSafeInteger(offset) && Number.isSafeInteger(length) && length >= 0, "Invalid read request");
    need(length <= MAX_READ_BYTES, `Read of ${length} bytes exceeds the ${MAX_READ_BYTES}-byte cap`);
    need(lower >= 0 && upper <= this.size && offset >= lower && offset + length <= upper,
      `Read [${offset}, ${offset + length}) is outside bounds [${lower}, ${upper})`);
    need(this.bytesRead + length <= MAX_METADATA_BYTES, `Metadata read budget of ${MAX_METADATA_BYTES} bytes exceeded`);
    const blob = this.source.slice(offset, offset + length);
    need(blob && typeof blob.arrayBuffer === "function", "slice() result has no arrayBuffer()");
    const buffer = await blob.arrayBuffer();
    need(buffer.byteLength === length, "Truncated read");
    this.bytesRead += length;
    this.reads += 1;
    return new Uint8Array(buffer);
  }
}

const be16 = (b, i) => (b[i] << 8) | b[i + 1];
const be32 = (b, i) => ((b[i] << 24) >>> 0) + (b[i + 1] << 16) + (b[i + 2] << 8) + b[i + 3];

/** Walk JPEG marker headers up to SOS; return segment table and the single Exif APP1. */
async function scanPreviewJpeg(r, jpegStart, jpegEnd) {
  const soi = await r.read(jpegStart, 2, jpegStart, jpegEnd);
  need(soi[0] === 0xff && soi[1] === 0xd8, "Preview JPEG does not start with SOI");
  let pos = jpegStart + 2;
  const segments = [];
  const exif = [];
  let sosOffset = null;
  let markers = 0;
  while (sosOffset === null) {
    need(markers < MAX_JPEG_MARKERS, `No SOS within ${MAX_JPEG_MARKERS} JPEG markers`);
    markers += 1;
    const m = await r.read(pos, 2, jpegStart, jpegEnd);
    need(m[0] === 0xff, `Expected a JPEG marker at offset ${pos}`);
    const marker = m[1];
    if (marker === 0xda) { sosOffset = pos; break; } // stop AT SOS: neither its length nor entropy data is read
    // Before SOS only length-bearing segments are legal here. Fill bytes, SOI, EOI, RSTn, TEM refuse.
    need(!(marker === 0x00 || marker === 0x01 || marker === 0xff || marker === 0xd8 || marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd7)),
      `Unexpected JPEG marker 0xff${marker.toString(16).padStart(2, "0")} before SOS at offset ${pos}`);
    const lenBytes = await r.read(pos + 2, 2, jpegStart, jpegEnd);
    const len = be16(lenBytes, 0);
    need(len >= 2, `JPEG segment at offset ${pos} has invalid length ${len}`);
    const end = pos + 2 + len;
    need(end <= jpegEnd, `JPEG segment at offset ${pos} (length ${len}) overruns the preview bounds`);
    const seg = { marker, offset: pos, length: len };
    if (marker === 0xe1 && len - 2 >= EXIF_IDENT.length) {
      const ident = await r.read(pos + 4, EXIF_IDENT.length, pos + 4, end);
      seg.exif = EXIF_IDENT.every((v, i) => ident[i] === v);
      if (seg.exif) exif.push(seg);
    }
    segments.push(seg);
    pos = end;
  }
  need(exif.length === 1, `Expected exactly one Exif APP1 segment, found ${exif.length}`);
  return { segments, exifSegment: exif[0], sosOffset, markersScanned: markers };
}

function asciiField(ex, key) {
  const t = ex[key];
  if (t === undefined) return null;
  need(t && typeof t.description === "string", `EXIF ${key} has an unexpected format`);
  return t.description;
}

function uintField(ex, key) {
  const t = ex[key];
  if (t === undefined) return null;
  let v = t && t.value;
  if (Array.isArray(v)) { need(v.length === 1, `EXIF ${key} has ${v.length} values (expected 1)`); v = v[0]; }
  need(Number.isSafeInteger(v) && v >= 0, `EXIF ${key} is not a single non-negative integer`);
  return v;
}

/** Parse the Fuji MakerNote IFD (little-endian, offsets relative to the "FUJIFILM" signature). */
function parseFujiMakerNote(bytes) {
  need(bytes.length >= 12, "Fuji MakerNote is shorter than its 12-byte header");
  for (let i = 0; i < FUJI_SIGNATURE.length; i++)
    need(bytes[i] === FUJI_SIGNATURE.charCodeAt(i), "MakerNote does not carry the FUJIFILM signature");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const ifdOffset = view.getUint32(8, true);
  need(ifdOffset >= 12 && ifdOffset + 2 <= bytes.length, `Fuji MakerNote IFD offset ${ifdOffset} is out of bounds`);
  const n = view.getUint16(ifdOffset, true);
  need(n <= MAX_MAKERNOTE_ENTRIES, `Fuji MakerNote entry count ${n} exceeds ${MAX_MAKERNOTE_ENTRIES}`);
  need(ifdOffset + 2 + n * 12 <= bytes.length, `Fuji MakerNote IFD (${n} entries) is truncated`);

  const byTag = new Map(Object.values(FUJI_FIELDS).map(f => [f.tag, []]));
  for (let i = 0; i < n; i++) {
    const e = ifdOffset + 2 + i * 12;
    const tag = view.getUint16(e, true);
    if (!byTag.has(tag)) continue; // unknown/other fields are skipped without dereferencing
    byTag.get(tag).push({ index: i, type: view.getUint16(e + 2, true), count: view.getUint32(e + 4, true), valueField: e + 8 });
  }

  const fields = {}, formats = {};
  for (const [name, spec] of Object.entries(FUJI_FIELDS)) {
    const hits = byTag.get(spec.tag);
    const hex = `0x${spec.tag.toString(16)}`;
    if (hits.length === 0) { fields[name] = null; formats[name] = null; continue; }
    const decoded = hits.map(h => {
      need(spec.types.includes(h.type) && h.count === spec.count,
        `Fuji ${hex} (${name}) has type ${h.type} x${h.count}; expected type ${spec.types.join("/")} x${spec.count}`);
      if (spec.count === 1) return h.type === T_SHORT ? view.getUint16(h.valueField, true) : view.getUint32(h.valueField, true);
      // SLONG x2 = 8 bytes: never inline, always an offset relative to the MakerNote start.
      const off = view.getUint32(h.valueField, true);
      need(off + 8 <= bytes.length, `Fuji ${hex} (${name}) value offset ${off} is out of bounds`);
      return { red: view.getInt32(off, true), blue: view.getInt32(off + 4, true) };
    });
    const key = JSON.stringify(decoded[0]);
    need(decoded.every(d => JSON.stringify(d) === key), `Conflicting duplicate Fuji ${hex} (${name}) entries`);
    fields[name] = decoded[0];
    formats[name] = { tag: hex, type: hits[0].type, count: hits[0].count, entries: hits.length, legacyName: spec.legacyName };
  }
  return { fields, formats, ifdOffset, entryCount: n, nextIfdRaw: ifdOffset + 2 + n * 12 + 4 <= bytes.length ? view.getUint32(ifdOffset + 2 + n * 12, true) : null };
}

/**
 * Extract raw candidate metadata fields. See module comment for scope.
 *
 * @returns {Promise<object>} JSON-serialisable record; missing tags are null.
 */
export async function readCandidateMetadata(source) {
  const r = new BoundedReader(source);
  const header = await r.read(0, RAF_HEADER_BYTES);
  for (let i = 0; i < RAF_MAGIC.length; i++)
    need(header[i] === RAF_MAGIC.charCodeAt(i), "Not a supported RAF header");
  const jpegOffset = be32(header, JPEG_OFFSET_FIELD);
  const jpegLength = be32(header, JPEG_LENGTH_FIELD);
  need(jpegOffset >= RAF_HEADER_BYTES && jpegLength >= 4 && jpegOffset + jpegLength <= r.size,
    `Preview JPEG [${jpegOffset}, +${jpegLength}) is out of bounds`);
  const jpegEnd = jpegOffset + jpegLength;

  const scan = await scanPreviewJpeg(r, jpegOffset, jpegEnd);
  const seg = scan.exifSegment;
  const payload = await r.read(seg.offset + 4, seg.length - 2, seg.offset + 4, seg.offset + 2 + seg.length);

  // Metadata-only JPEG: SOI + the single Exif APP1 (identical bytes) + EOI. No image data.
  const mini = new Uint8Array(2 + 4 + payload.length + 2);
  mini.set([0xff, 0xd8, 0xff, 0xe1, seg.length >> 8, seg.length & 0xff], 0);
  mini.set(payload, 6);
  mini.set([0xff, 0xd9], mini.length - 2);

  let tags;
  try { tags = ExifReader.load(mini.buffer, { expanded: true }); }
  catch (err) { throw new CandidateMetadataError(`ExifReader could not parse the Exif APP1: ${err && err.message}`); }
  const ex = (tags && tags.exif) || {};

  const mn = ex.MakerNote && ex.MakerNote.value;
  need(mn && typeof mn.length === "number" && mn.length > 0, "Exif APP1 has no MakerNote");
  const fuji = parseFujiMakerNote(new Uint8Array(mn));

  // Unchanged bounded TIFF WB helper; its reads are reported separately.
  const wb = await readRafWbCoefficients(source);

  return {
    schema: SCHEMA,
    exif: {
      model: asciiField(ex, "Model"),
      dateTimeOriginal: asciiField(ex, "DateTimeOriginal"),
      subSecTimeOriginal: asciiField(ex, "SubSecTimeOriginal"),
      offsetTimeOriginal: asciiField(ex, "OffsetTimeOriginal"),
      iso: uintField(ex, "ISOSpeedRatings"),
    },
    fuji: {
      ...fuji.fields,
      formats: fuji.formats,
      makerNoteBytes: mn.length,
      ifdOffset: fuji.ifdOffset,
      entryCount: fuji.entryCount,
      nextIfdRaw: fuji.nextIfdRaw,
    },
    wb: {
      whiteBalanceModeRaw: fuji.fields.whiteBalanceMode,
      colorTemperatureKelvinRaw: fuji.fields.colorTemperatureKelvin,
      whiteBalanceFineTuneRaw: fuji.fields.whiteBalanceFineTune,
      autoRgbg: wb.autoRgbg,
      asShotCamMul: wb.asShotCamMul,
      tiffRecords: wb.records,
      byteOrder: wb.byteOrder,
    },
    preview: {
      jpegOffset, jpegLength,
      exifSegment: { offset: seg.offset, length: seg.length },
      sosOffset: scan.sosOffset,
      markersScanned: scan.markersScanned,
      segments: scan.segments.map(s => ({ marker: `0xff${s.marker.toString(16)}`, offset: s.offset, length: s.length, ...(s.exif !== undefined ? { exif: s.exif } : {}) })),
    },
    reads: {
      metadataBytesRead: r.bytesRead, metadataReads: r.reads,
      wbBytesRead: wb.bytesRead, wbReads: wb.reads,
      totalBytesRead: r.bytesRead + wb.bytesRead,
    },
    limits: "Raw field extraction only. Not metadata admission, WB/neutral policy, capture-DR resolution, " +
      "tier assignment or Fuji-accuracy evidence. Fine-tune is the raw SLONG pair (no /20 scaling). " +
      "Missing tags are null; nothing is inferred.",
  };
}
