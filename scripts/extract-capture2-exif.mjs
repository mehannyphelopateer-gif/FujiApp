/**
 * One-off diagnostic helper for capture batch 2 (docs/phase4-scene-adaptive-scope.md,
 * "Capture batch 2" section) - extracts capture timestamp, ISO, and Fuji WB mode
 * (Auto vs Manual/Kelvin) directly from each new RAF's EXIF/MakerNote, the same
 * tag (0x1002 WhiteBalance) and mapping src/lib/exif/parseFujiMakerNotes.ts uses,
 * reimplemented standalone here so this can run under plain node against a
 * Buffer instead of a browser File. Keep the WHITE_BALANCE_MAP in sync with that
 * file if Fuji's tag values are ever revisited.
 *
 * Run: node scripts/extract-capture2-exif.mjs
 * Reads calibration-input/phase4-capture2-raw/*.RAF, writes
 * calibration-input/phase4-capture2-exif.json.
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import ExifReader from "exifreader";

const RAW_DIR = resolve(process.cwd(), "calibration-input/phase4-capture2-raw");
const OUT_PATH = resolve(process.cwd(), "calibration-input/phase4-capture2-exif.json");

const WHITE_BALANCE_MAP = {
  0x0: "Auto", 0x1: "Auto", 0x2: "Auto",
  0x100: "Daylight", 0x200: "Shade",
  0x300: "Fluorescent1", 0x301: "Fluorescent1",
  0x302: "Fluorescent2", 0x303: "Fluorescent3", 0x304: "Fluorescent3",
  0x400: "Incandescent", 0x600: "Underwater", 0xff0: "Kelvin",
};

function parseFujiWbMode(makerNoteBytes) {
  if (!makerNoteBytes || makerNoteBytes.length < 12) return null;
  const sig = Buffer.from(makerNoteBytes.slice(0, 8)).toString("latin1");
  if (sig !== "FUJIFILM") return null;
  const view = new DataView(makerNoteBytes.buffer, makerNoteBytes.byteOffset, makerNoteBytes.byteLength);
  const ifdOffset = view.getUint32(8, true);
  if (ifdOffset + 2 > makerNoteBytes.length) return null;
  const numEntries = view.getUint16(ifdOffset, true);
  for (let i = 0; i < numEntries; i++) {
    const entryOffset = ifdOffset + 2 + i * 12;
    if (entryOffset + 12 > makerNoteBytes.length) break;
    const tag = view.getUint16(entryOffset, true);
    if (tag === 0x1002) {
      const type = view.getUint16(entryOffset + 2, true);
      const count = view.getUint32(entryOffset + 4, true);
      if (type === 3 && count === 1) {
        const raw = view.getUint16(entryOffset + 8, true);
        return { raw, mode: WHITE_BALANCE_MAP[raw] ?? "Unknown" };
      }
    }
  }
  return null;
}

/**
 * RAF containers embed a preview JPEG (with full standard EXIF + Fuji
 * MakerNote) ahead of the raw sensor data - this scans for its SOI/EOI
 * markers rather than hardcoding RAF-header offsets, which differ across
 * camera generations. Confirmed 2026-11-12 against this batch's actual
 * files: a single ~13KB embedded JPEG at a small fixed offset, with a
 * complete EXIF block including MakerNote.
 */
function extractEmbeddedJpeg(buf, searchLimit = 2_000_000) {
  let start = -1;
  for (let i = 0; i < Math.min(buf.length - 1, searchLimit); i++) {
    if (buf[i] === 0xff && buf[i + 1] === 0xd8 && start === -1) start = i;
    if (buf[i] === 0xff && buf[i + 1] === 0xd9 && start !== -1) return buf.subarray(start, i + 2);
  }
  return null;
}

const files = readdirSync(RAW_DIR).filter((f) => /\.raf$/i.test(f)).sort();
const results = [];
for (const f of files) {
  const raw = readFileSync(resolve(RAW_DIR, f));
  const jpeg = extractEmbeddedJpeg(raw);
  if (!jpeg) {
    results.push({ filename: f, error: "no embedded JPEG found" });
    continue;
  }
  const tags = ExifReader.load(jpeg, { expanded: true });
  const dateTimeOriginal = tags.exif?.DateTimeOriginal?.description ?? null;
  const iso = tags.exif?.ISOSpeedRatings?.description ?? tags.exif?.PhotographicSensitivity?.description ?? null;
  const makerNoteValue = tags.exif?.MakerNote?.value;
  const makerNoteBytes = makerNoteValue ? new Uint8Array(makerNoteValue) : null;
  const wb = parseFujiWbMode(makerNoteBytes);
  results.push({
    filename: f,
    dateTimeOriginal,
    iso,
    wbRaw: wb?.raw ?? null,
    wbMode: wb?.mode ?? "Unknown",
    wbManualFlag: wb ? (wb.mode === "Auto" ? 0 : 1) : null,
  });
}
writeFileSync(OUT_PATH, JSON.stringify(results, null, 2));
console.log(`wrote ${results.length} entries to ${OUT_PATH}`);
for (const r of results) {
  console.log(
    `${r.filename}  ${r.dateTimeOriginal}  ISO ${r.iso}  WB ${r.wbMode}` +
      (r.wbRaw !== null ? ` (raw 0x${r.wbRaw.toString(16)})` : "")
  );
}
