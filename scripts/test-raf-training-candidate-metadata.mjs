#!/usr/bin/env node
/**
 * Synthetic tests for scripts/lib/raf-training-candidate-metadata.mjs.
 * No real RAF, JPEG or candidate file is read: every fixture is built in memory.
 * Run: node scripts/test-raf-training-candidate-metadata.mjs
 */
import assert from "node:assert/strict";
import {
  readCandidateMetadata, CandidateMetadataError, MAX_READ_BYTES, MAX_METADATA_BYTES, MAX_JPEG_MARKERS,
} from "./lib/raf-training-candidate-metadata.mjs";
import { readRafWbCoefficients } from "./lib/raf-wb-coefficients.mjs";

const enc = s => new TextEncoder().encode(s);
const AUTO_GRB = [302, 518, 570];     // file order G, R, B
const AS_SHOT_GRB = [302, 628, 445];

// ---------- Fuji MakerNote (little-endian, offsets relative to "FUJIFILM") ----------
/** entries: { tag, type, count, inline } | { tag, type: 9, count: 2, slongs: [r, b] } | { ..., rawOffset } */
function buildMakerNote(entries, { signature = "FUJIFILM", ifdOffset = 12, declaredCount = null } = {}) {
  const ifdSize = 2 + entries.length * 12 + 4;
  let dataAt = ifdOffset + ifdSize;
  for (const e of entries) if (e.slongs) { e._at = dataAt; dataAt += e.slongs.length * 4; }
  const buf = new Uint8Array(Math.max(dataAt, 12)), v = new DataView(buf.buffer);
  buf.set(enc(signature.padEnd(8, " ").slice(0, 8)), 0);
  v.setUint32(8, ifdOffset, true);
  v.setUint16(ifdOffset, declaredCount ?? entries.length, true);
  entries.forEach((e, i) => {
    const p = ifdOffset + 2 + i * 12;
    v.setUint16(p, e.tag, true); v.setUint16(p + 2, e.type, true); v.setUint32(p + 4, e.count, true);
    if (e.rawOffset !== undefined) v.setUint32(p + 8, e.rawOffset, true);
    else if (e.slongs) v.setUint32(p + 8, e._at, true);
    else if (e.type === 3) v.setUint16(p + 8, e.inline, true);
    else if (e.type === 9) v.setInt32(p + 8, e.inline, true);
    else v.setUint32(p + 8, e.inline >>> 0, true);
  });
  for (const e of entries) if (e.slongs) e.slongs.forEach((x, k) => v.setInt32(e._at + k * 4, x, true));
  return buf;
}
const short = (tag, inline) => ({ tag, type: 3, count: 1, inline });
const long = (tag, inline) => ({ tag, type: 4, count: 1, inline });
const fineTune = (r, b) => ({ tag: 0x100a, type: 9, count: 2, slongs: [r, b] });

const autoFrameNote = () => [
  short(0x1002, 0x1), fineTune(0, 0), short(0x1032, 1), short(0x1400, 1), short(0x1402, 0),
  short(0x140b, 100), short(0x1443, 0),
  short(0x1401, 0), { tag: 0x1234, type: 4, count: 99, inline: 0x7ffffff0 }, // other/unknown fields: skipped, never dereferenced
];

// ---------- EXIF TIFF (II) with IFD0 -> ExifIFD -> MakerNote ----------
function buildExifTiff({ model = "X100VI", dateTimeOriginal = "2026:11:16 19:12:12", iso = 125, makerNote = buildMakerNote(autoFrameNote()) } = {}) {
  const ifd0 = [];
  if (model !== null) ifd0.push({ tag: 0x0110, type: 2, data: enc(model + "\0") });
  ifd0.push({ tag: 0x8769, type: 4, count: 1, exifPtr: true });
  const exif = [];
  if (iso !== null) exif.push({ tag: 0x8827, type: 3, count: 1, inline: iso });
  if (dateTimeOriginal !== null) exif.push({ tag: 0x9003, type: 2, data: enc(dateTimeOriginal + "\0") });
  if (makerNote !== null) exif.push({ tag: 0x927c, type: 7, data: makerNote });
  const ifdSize = n => 2 + n * 12 + 4;
  const ifd0At = 8, exifAt = ifd0At + ifdSize(ifd0.length);
  let dataAt = exifAt + ifdSize(exif.length);
  for (const e of [...ifd0, ...exif]) if (e.data && e.data.length > 4) { e._at = dataAt; dataAt += e.data.length + (e.data.length % 2); }
  const buf = new Uint8Array(dataAt), v = new DataView(buf.buffer);
  buf[0] = buf[1] = 0x49; v.setUint16(2, 42, true); v.setUint32(4, ifd0At, true);
  const writeIfd = (at, entries) => {
    v.setUint16(at, entries.length, true);
    entries.forEach((e, i) => {
      const p = at + 2 + i * 12;
      v.setUint16(p, e.tag, true); v.setUint16(p + 2, e.type, true);
      v.setUint32(p + 4, e.data ? e.data.length : e.count, true);
      if (e.data) { if (e.data.length <= 4) buf.set(e.data, p + 8); else v.setUint32(p + 8, e._at, true); }
      else if (e.exifPtr) v.setUint32(p + 8, exifAt, true);
      else if (e.type === 3) v.setUint16(p + 8, e.inline, true);
      else v.setUint32(p + 8, e.inline, true);
    });
    v.setUint32(at + 2 + entries.length * 12, 0, true);
  };
  writeIfd(ifd0At, ifd0); writeIfd(exifAt, exif);
  for (const e of [...ifd0, ...exif]) if (e._at !== undefined) buf.set(e.data, e._at);
  return buf;
}

// ---------- JPEG ----------
const seg = (marker, payload) => {
  const len = payload.length + 2;
  return Uint8Array.from([0xff, marker, len >> 8, len & 0xff, ...payload]);
};
const exifApp1 = tiff => seg(0xe1, [...enc("Exif\0\0"), ...tiff]);
const xmpApp1 = () => seg(0xe1, [...enc("http://ns.adobe.com/xap/1.0/\0"), ...enc("<x:xmpmeta/>")]);

/** segments: list of Uint8Array (already-built segments) placed between SOI and SOS. */
function buildJpeg({ segments = [seg(0xe0, enc("JFIF\0\x01\x02")), exifApp1(buildExifTiff()), seg(0xdb, new Array(65).fill(1))],
                     withSos = true, entropyBytes = 2048 } = {}) {
  const parts = [Uint8Array.from([0xff, 0xd8])];
  const offsets = [];
  let at = 2;
  for (const s of segments) { offsets.push(at); parts.push(s); at += s.length; }
  let sosAt = null;
  if (withSos) {
    sosAt = at;
    parts.push(seg(0xda, [3, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0]));
    parts.push(new Uint8Array(entropyBytes).fill(0xcd));
  }
  parts.push(Uint8Array.from([0xff, 0xd9]));
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(total);
  let p = 0; for (const x of parts) { out.set(x, p); p += x.length; }
  return { bytes: out, offsets, sosAt };
}

// ---------- CFA TIFF (WB records only, for the unchanged helper) ----------
function buildCfaTiff() {
  const entries = [
    { tag: 0xf007, type: 4, count: 1, raw: 0xfffffff0 },   // pixel pointer: must never be followed
    { tag: 0xf00d, type: 4, count: 3, values: AUTO_GRB },
    { tag: 0xf00e, type: 4, count: 3, values: AS_SHOT_GRB },
  ];
  let dataAt = 8 + 2 + entries.length * 12 + 4;
  for (const e of entries) if (e.values) { e._at = dataAt; dataAt += 12; }
  const buf = new Uint8Array(dataAt), v = new DataView(buf.buffer);
  buf[0] = buf[1] = 0x49; v.setUint16(2, 42, true); v.setUint32(4, 8, true);
  v.setUint16(8, entries.length, true);
  entries.forEach((e, i) => {
    const p = 10 + i * 12;
    v.setUint16(p, e.tag, true); v.setUint16(p + 2, e.type, true); v.setUint32(p + 4, e.count, true);
    v.setUint32(p + 8, e.values ? e._at : e.raw >>> 0, true);
  });
  for (const e of entries) if (e.values) e.values.forEach((x, k) => v.setUint32(e._at + k * 4, x, true));
  return buf;
}

// ---------- RAF ----------
const JPEG_AT = 160;
function buildRaf({ jpeg = buildJpeg(), magic = "FUJIFILMCCD-RAW ", jpegOffset = JPEG_AT, jpegLength = null, pixelBytes = 4096 } = {}) {
  const cfa = buildCfaTiff();
  const cfaAt = JPEG_AT + jpeg.bytes.length + 16;
  const pixelAt = cfaAt + cfa.length;
  const buf = new Uint8Array(pixelAt + pixelBytes), v = new DataView(buf.buffer);
  buf.set(enc(magic), 0);
  v.setUint32(84, jpegOffset, false);
  v.setUint32(88, jpegLength ?? jpeg.bytes.length, false);
  v.setUint32(0x64, cfaAt, false);
  v.setUint32(0x68, cfa.length, false);
  buf.set(jpeg.bytes, JPEG_AT);
  buf.set(cfa, cfaAt);
  buf.fill(0xab, pixelAt);
  return { bytes: buf, jpeg, cfaAt, pixelAt,
           sosAbs: jpeg.sosAt === null ? null : JPEG_AT + jpeg.sosAt,
           segAbs: jpeg.offsets.map(o => JPEG_AT + o), jpegEnd: JPEG_AT + jpeg.bytes.length };
}

function recordingSource(bytes) {
  const reads = [];
  return {
    reads, size: bytes.length,
    slice(start, end) {
      reads.push([start, end]);
      const part = bytes.slice(start, end);
      return { arrayBuffer: async () => part.buffer.slice(part.byteOffset, part.byteOffset + part.byteLength) };
    },
  };
}

const rafWith = opts => new Blob([buildRaf(opts).bytes]);
const rafWithNote = (noteEntries, noteOpts) =>
  rafWith({ jpeg: buildJpeg({ segments: [exifApp1(buildExifTiff({ makerNote: buildMakerNote(noteEntries, noteOpts) }))] }) });

async function rejects(promise, pattern) {
  await assert.rejects(promise, err => err instanceof CandidateMetadataError && pattern.test(err.message), `expected ${pattern}`);
}

const tests = {
  async "Auto-WB frame: raw fields, 0 fine-tune, AutoDR 100, explicit nulls"() {
    const out = await readCandidateMetadata(rafWith());
    assert.deepEqual(out.exif, { model: "X100VI", dateTimeOriginal: "2026:11:16 19:12:12",
      subSecTimeOriginal: null, offsetTimeOriginal: null, iso: 125 });
    assert.equal(out.fuji.whiteBalanceMode, 1);
    assert.equal(out.fuji.colorTemperatureKelvin, null);
    assert.deepEqual(out.fuji.whiteBalanceFineTune, { red: 0, blue: 0 });
    assert.equal(out.fuji.autoDynamicRange, 100);
    assert.equal(out.fuji.developmentDynamicRange, null);
    assert.equal(out.fuji.dRangePriority, 0);
    assert.equal(out.fuji.dRangePriorityAuto, null);
    assert.equal(out.fuji.dRangePriorityFixed, null);
    assert.equal(out.fuji.exposureCount, 1);
    assert.equal(out.fuji.dynamicRange, 1);
    assert.equal(out.fuji.dynamicRangeSetting, 0);
    assert.equal(out.fuji.formats.autoDynamicRange.legacyName, "AutoDynamicRange");
    assert.equal(out.fuji.formats.colorTemperatureKelvin, null);
    assert.deepEqual(out.wb.autoRgbg, [518, 302, 570, 302]);
    assert.deepEqual(out.wb.asShotCamMul, [628, 302, 445, 0]);
    assert.equal(out.wb.whiteBalanceModeRaw, 1);
    assert.deepEqual(out.wb.whiteBalanceFineTuneRaw, { red: 0, blue: 0 });
    JSON.stringify(out); // must be serialisable
  },

  async "Kelvin frame: 0xff0 mode, 6600 K as SHORT and as LONG, raw signed fine-tune"() {
    for (const kelvin of [short(0x1005, 6600), long(0x1005, 6600)]) {
      const out = await readCandidateMetadata(rafWithNote([short(0x1002, 0xff0), kelvin, fineTune(-40, 60), short(0x140b, 100)]));
      assert.equal(out.fuji.whiteBalanceMode, 4080);
      assert.equal(out.fuji.colorTemperatureKelvin, 6600);
      assert.equal(out.fuji.formats.colorTemperatureKelvin.type, kelvin.type);
      assert.deepEqual(out.fuji.whiteBalanceFineTune, { red: -40, blue: 60 }); // raw, not /20
    }
  },

  async "DR fields are reported raw and independently, never resolved"() {
    const out = await readCandidateMetadata(rafWithNote([short(0x1002, 1), short(0x1403, 200), short(0x140b, 100), long(0x1444, 3)]));
    assert.equal(out.fuji.developmentDynamicRange, 200);
    assert.equal(out.fuji.autoDynamicRange, 100);
    assert.equal(out.fuji.dRangePriorityAuto, 3);
    assert.equal("captureDynamicRange" in out.fuji, false);
  },

  async "missing tags are explicit nulls (MakerNote and EXIF)"() {
    const jpeg = buildJpeg({ segments: [exifApp1(buildExifTiff({ model: null, dateTimeOriginal: null, iso: null,
      makerNote: buildMakerNote([short(0x1401, 0)]) }))] });
    const out = await readCandidateMetadata(rafWith({ jpeg }));
    assert.deepEqual(out.exif, { model: null, dateTimeOriginal: null, subSecTimeOriginal: null, offsetTimeOriginal: null, iso: null });
    for (const k of ["whiteBalanceMode", "colorTemperatureKelvin", "whiteBalanceFineTune", "exposureCount", "dynamicRange",
                     "dynamicRangeSetting", "developmentDynamicRange", "autoDynamicRange", "dRangePriority",
                     "dRangePriorityAuto", "dRangePriorityFixed"]) {
      assert.equal(out.fuji[k], null, k);
      assert.equal(out.fuji.formats[k], null, k);
    }
    assert.equal(out.wb.whiteBalanceFineTuneRaw, null);
  },

  async "no entropy, pixel or XMP-payload bytes are read; reads are bounded"() {
    const xmp = xmpApp1();
    const jpeg = buildJpeg({ segments: [seg(0xe0, enc("JFIF\0")), xmp, exifApp1(buildExifTiff()), seg(0xdb, [1, 2, 3])], entropyBytes: 8192 });
    const raf = buildRaf({ jpeg });
    const src = recordingSource(raf.bytes);
    const out = await readCandidateMetadata(src);
    const xmpAbs = raf.segAbs[1];
    for (const [s, e] of src.reads) {
      const inMetadata = e <= raf.sosAbs + 2;              // up to and including the SOS marker, never its header/entropy
      const inCfaTiff = s >= raf.cfaAt && e <= raf.pixelAt; // unchanged WB helper, TIFF section only
      assert(inMetadata || inCfaTiff, `read [${s}, ${e}) outside the metadata windows`);
      assert(e - s <= MAX_READ_BYTES);
      assert(!(s < xmpAbs + xmp.length && e > xmpAbs + 4 + 6), `read [${s}, ${e}) went past the XMP identifier`);
    }
    assert.equal(out.preview.sosOffset, raf.sosAbs);
    assert.equal(out.preview.segments.find(x => x.offset === xmpAbs).exif, false);
    assert.equal(out.reads.totalBytesRead, src.reads.reduce((a, [s, e]) => a + e - s, 0));
    assert(out.reads.metadataBytesRead <= MAX_METADATA_BYTES);
    assert(out.reads.totalBytesRead < raf.bytes.length / 4, `read ${out.reads.totalBytesRead} of ${raf.bytes.length}`);
  },

  async "WB levels come from the unchanged helper"() {
    const blob = rafWith();
    const [out, direct] = await Promise.all([readCandidateMetadata(blob), readRafWbCoefficients(blob)]);
    assert.deepEqual(out.wb.autoRgbg, direct.autoRgbg);
    assert.deepEqual(out.wb.asShotCamMul, direct.asShotCamMul);
  },

  async "exactly one Exif APP1 required"() {
    await rejects(readCandidateMetadata(rafWith({ jpeg: buildJpeg({ segments: [xmpApp1()] }) })), /exactly one Exif APP1.*found 0/);
    const t = buildExifTiff();
    await rejects(readCandidateMetadata(rafWith({ jpeg: buildJpeg({ segments: [exifApp1(t), exifApp1(t)] }) })), /found 2/);
  },

  async "malformed RAF header and preview bounds are refused"() {
    await rejects(readCandidateMetadata(rafWith({ magic: "FUJIFILMCCD-RAX " })), /RAF header/);
    await rejects(readCandidateMetadata(rafWith({ jpegOffset: 64 })), /out of bounds/);
    await rejects(readCandidateMetadata(rafWith({ jpegLength: 10_000_000 })), /out of bounds/);
    const { bytes } = buildRaf();
    bytes[JPEG_AT + 1] = 0xd9;
    await rejects(readCandidateMetadata(new Blob([bytes])), /SOI/);
    await rejects(readCandidateMetadata(new Blob([bytes.slice(0, 50)])), /outside bounds/);
  },

  async "malformed JPEG segments are refused"() {
    const overrun = buildJpeg();
    overrun.bytes[4] = 0xff; overrun.bytes[5] = 0xf0; // APP0 length far past the preview end
    await rejects(readCandidateMetadata(rafWith({ jpeg: overrun })), /overruns/);
    const notMarker = buildJpeg();
    notMarker.bytes[2] = 0x00;
    await rejects(readCandidateMetadata(rafWith({ jpeg: notMarker })), /Expected a JPEG marker/);
    await rejects(readCandidateMetadata(rafWith({ jpeg: buildJpeg({ withSos: false }) })), /Unexpected JPEG marker 0xffd9/);
    const zeroLen = buildJpeg();
    zeroLen.bytes[4] = 0; zeroLen.bytes[5] = 1;
    await rejects(readCandidateMetadata(rafWith({ jpeg: zeroLen })), /invalid length/);
  },

  async "marker cap: SOS must appear within the bounded scan"() {
    const many = n => [exifApp1(buildExifTiff()), ...Array.from({ length: n }, () => seg(0xfe, []))];
    await readCandidateMetadata(rafWith({ jpeg: buildJpeg({ segments: many(MAX_JPEG_MARKERS - 2) }) })); // SOS is exactly the last allowed marker
    await rejects(readCandidateMetadata(rafWith({ jpeg: buildJpeg({ segments: many(MAX_JPEG_MARKERS) }) })), /No SOS within/);
  },

  async "malformed MakerNote is refused"() {
    await rejects(readCandidateMetadata(rafWithNote(autoFrameNote(), { signature: "OLYMPUS!" })), /FUJIFILM signature/);
    await rejects(readCandidateMetadata(rafWithNote(autoFrameNote(), { declaredCount: 400 })), /truncated/);
    await rejects(readCandidateMetadata(rafWithNote([short(0x1002, 1)], { ifdOffset: 8 })), /IFD offset/); // overlaps the 12-byte header
    await rejects(readCandidateMetadata(rafWithNote([{ tag: 0x100a, type: 9, count: 2, rawOffset: 0x7ffffff0 }])), /value offset .* out of bounds/);
    const jpeg = buildJpeg({ segments: [exifApp1(buildExifTiff({ makerNote: null }))] });
    await rejects(readCandidateMetadata(rafWith({ jpeg })), /no MakerNote/);
  },

  async "wrong type/count for critical fields is refused, never coerced"() {
    for (const bad of [long(0x1002, 1), { tag: 0x1005, type: 9, count: 1, inline: 6600 },
                       { tag: 0x100a, type: 9, count: 1, inline: 0 }, { tag: 0x100a, type: 3, count: 2, inline: 0 },
                       { tag: 0x140b, type: 3, count: 2, inline: 100 }, { tag: 0x1032, type: 9, count: 1, inline: 1 }]) {
      await rejects(readCandidateMetadata(rafWithNote([bad])), /expected type/);
    }
  },

  async "duplicate critical tags: identical accepted, conflicting refused"() {
    const same = await readCandidateMetadata(rafWithNote([short(0x140b, 100), short(0x140b, 100)]));
    assert.equal(same.fuji.autoDynamicRange, 100);
    assert.equal(same.fuji.formats.autoDynamicRange.entries, 2);
    await rejects(readCandidateMetadata(rafWithNote([short(0x140b, 100), short(0x140b, 200)])), /Conflicting duplicate Fuji 0x140b/);
    await rejects(readCandidateMetadata(rafWithNote([short(0x1005, 6600), long(0x1005, 6500)])), /Conflicting duplicate Fuji 0x1005/);
    await rejects(readCandidateMetadata(rafWithNote([fineTune(0, 0), fineTune(0, 20)])), /Conflicting duplicate Fuji 0x100a/);
  },

  async "non-Blob sources are rejected"() {
    await assert.rejects(readCandidateMetadata({}), TypeError);
    await assert.rejects(readCandidateMetadata(new Uint8Array(200)), TypeError);
  },
};

let failed = 0;
for (const [name, fn] of Object.entries(tests)) {
  try { await fn(); console.log(`PASS ${name}`); }
  catch (err) { failed++; console.error(`FAIL ${name}\n  ${err.stack}`); }
}
console.log(`${Object.keys(tests).length - failed}/${Object.keys(tests).length} passed`);
process.exitCode = failed ? 1 : 0;
