/**
 * Diagnostic: print the exposure-relevant EXIF + Fuji MakerNote tags of one
 * or more RAFs (camera model, exposure bias, shutter/aperture/ISO, Fuji
 * dynamic-range / film-mode / WB tags) from the RAF's embedded preview
 * JPEG, the same way scripts/extract-capture2-exif.mjs does. Used to check
 * whether capture batch 2 was shot or rendered under different
 * exposure/DR settings than the phase-3 corpus.
 *
 * Run: node scripts/dump-raf-exif.mjs "calibration-input/Shoot 84/DSCF0820.RAF" ...
 *   or: node scripts/dump-raf-exif.mjs --list paths.json --out corpus-exif.json
 * Only the first 2.5 MB of each RAF is read - the embedded preview JPEG
 * sits near the file start - so a whole-corpus pass is cheap.
 */
import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import ExifReader from "exifreader";

function readHead(path, bytes = 2_500_000) {
  const fd = openSync(path, "r");
  try {
    const buf = Buffer.alloc(bytes);
    const n = readSync(fd, buf, 0, bytes, 0);
    return buf.subarray(0, n);
  } finally {
    closeSync(fd);
  }
}

const FUJI_TAGS = {
  0x1002: "WhiteBalance", 0x1400: "DynamicRangeSetting", 0x1401: "FilmMode", 0x1402: "DynamicRangeSetting2",
  0x1403: "DevelopmentDynamicRange", 0x1404: "MinFocalLength", 0x1405: "MaxFocalLength", 0x1406: "MaxApertureAtMinFocal",
  0x1407: "MaxApertureAtMaxFocal", 0x140b: "AutoDynamicRange", 0x1443: "DRangePriority", 0x1444: "DRangePriorityAuto",
  0x1445: "DRangePriorityFixed", 0x1031: "PictureMode", 0x1032: "ExposureCount", 0x1033: "EXRAuto", 0x1034: "EXRMode",
  0x1040: "ShadowTone", 0x1041: "HighlightTone", 0x1047: "GrainEffect", 0x104d: "CropMode", 0x3820: "ExposureBias?",
};

function extractEmbeddedJpeg(buf, searchLimit = 2_000_000) {
  let start = -1;
  for (let i = 0; i < Math.min(buf.length - 1, searchLimit); i++) {
    if (buf[i] === 0xff && buf[i + 1] === 0xd8 && start === -1) start = i;
    if (buf[i] === 0xff && buf[i + 1] === 0xd9 && start !== -1) return buf.subarray(start, i + 2);
  }
  return null;
}

function fujiTags(mn) {
  const out = {};
  if (!mn || mn.length < 12 || Buffer.from(mn.slice(0, 8)).toString("latin1") !== "FUJIFILM") return out;
  const v = new DataView(mn.buffer, mn.byteOffset, mn.byteLength);
  const ifd = v.getUint32(8, true);
  const n = v.getUint16(ifd, true);
  for (let i = 0; i < n; i++) {
    const e = ifd + 2 + i * 12;
    if (e + 12 > mn.length) break;
    const tag = v.getUint16(e, true), type = v.getUint16(e + 2, true), count = v.getUint32(e + 4, true);
    if (!(tag in FUJI_TAGS)) continue;
    let val;
    if (type === 3 && count === 1) val = v.getUint16(e + 8, true);
    else if (type === 4 && count === 1) val = v.getUint32(e + 8, true);
    else if (type === 9 && count === 1) val = v.getInt32(e + 8, true);
    else if (type === 10 && count === 1) { const off = v.getUint32(e + 8, true); val = `${v.getInt32(off, true)}/${v.getInt32(off + 4, true)}`; }
    else val = `type${type}x${count}`;
    out[FUJI_TAGS[tag]] = typeof val === "number" ? `${val} (0x${val.toString(16)})` : val;
  }
  return out;
}

function dump(path) {
  const jpeg = extractEmbeddedJpeg(readHead(path));
  if (!jpeg) return { path, error: "no embedded JPEG" };
  const t = ExifReader.load(jpeg, { expanded: true });
  const ex = t.exif ?? {};
  const d = (k) => ex[k]?.description ?? null;
  const mn = ex.MakerNote?.value ? new Uint8Array(ex.MakerNote.value) : null;
  return {
    path, name: path.split("/").pop(), model: d("Model"), software: d("Software"), dateTimeOriginal: d("DateTimeOriginal"),
    exposureTime: d("ExposureTime"), fNumber: d("FNumber"), iso: d("ISOSpeedRatings"), exposureBias: d("ExposureBiasValue"),
    exposureMode: d("ExposureMode"), exposureProgram: d("ExposureProgram"), brightnessValue: d("BrightnessValue"),
    fuji: fujiTags(mn),
  };
}

const args = process.argv.slice(2);
const listIdx = args.indexOf("--list");
if (listIdx >= 0) {
  const outIdx = args.indexOf("--out");
  const paths = JSON.parse(readFileSync(args[listIdx + 1], "utf8"));
  const rows = [];
  for (const [i, p] of paths.entries()) {
    rows.push(dump(p));
    if ((i + 1) % 100 === 0) console.log(`  ${i + 1}/${paths.length}`);
  }
  const out = outIdx >= 0 ? args[outIdx + 1] : "corpus-exif.json";
  writeFileSync(out, JSON.stringify(rows, null, 1));
  console.log(`wrote ${rows.length} entries to ${out}`);
} else {
  for (const path of args) {
    const r = dump(path);
    console.log(`\n${path}`);
    if (r.error) { console.log(`  ${r.error}`); continue; }
    console.log(`  Model=${r.model}  Software=${r.software}  DateTimeOriginal=${r.dateTimeOriginal}`);
    console.log(`  ExposureTime=${r.exposureTime}  FNumber=${r.fNumber}  ISO=${r.iso}  ExposureBias=${r.exposureBias}  ExposureMode=${r.exposureMode}  ExposureProgram=${r.exposureProgram}  BrightnessValue=${r.brightnessValue}`);
    console.log(`  Fuji: ${JSON.stringify(r.fuji)}`);
  }
}
