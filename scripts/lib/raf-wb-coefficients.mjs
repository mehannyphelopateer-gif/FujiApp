/**
 * Bounded reader for the two source WB records in a Fujifilm RAF's TIFF section.
 *
 *   0xf00d WB_GRBLevelsAuto  -> autoRgbg     [R, G, B, G]
 *   0xf00e WB_GRBLevels      -> asShotCamMul [R, G, B, 0]  (LibRaw cam_mul layout)
 *
 * Port of the TIFF half of the proven MetadataReader in
 * training/probe_provia_source_wb.py. Pure JS for browser and Node: no Node
 * imports and no globals beyond DataView / TextDecoder-free byte checks.
 *
 * What it does NOT do:
 *   - decide a WB policy or WB mode. The caller supplies an independently
 *     established mode (for example "source WB mode is Auto") before using autoRgbg;
 *   - parse the JPEG/EXIF maker notes, the RAF directory, RAFData (0xc000) or any
 *     WB/colour-temperature table;
 *   - follow strip/tile/0xf007/0xf008 pointers or any unknown field's offset;
 *   - claim anything about manual/Kelvin/preset neutralisation.
 *
 * Input: a Blob/File, or any duck-typed object with a numeric `size` and
 * `slice(start, end)` returning something with `arrayBuffer()`. Only small
 * bounded windows are read; the image payload is never loaded.
 */

export const RAF_MAGIC = "FUJIFILMCCD-RAW ";
export const RAF_HEADER_BYTES = 112;
export const CFA_OFFSET_FIELD = 0x64;
export const CFA_LENGTH_FIELD = 0x68;
export const MAX_IFDS = 16;
export const MAX_ENTRIES = 1024;
export const MAX_READ_BYTES = 2 + MAX_ENTRIES * 12 + 4; // one IFD: count, entries, next pointer
export const TAG_SUB_IFD = 0xf000;
export const TAG_WB_AUTO = 0xf00d;
export const TAG_WB_AS_SHOT = 0xf00e;
/** File order is G, R, B; output index i takes input index GRB_TO_RGB[i]. */
export const GRB_TO_RGB = Object.freeze([1, 0, 2]);
const U32_MAX = 0xffffffff;

export class RafWbError extends Error {
  constructor(message) { super(message); this.name = "RafWbError"; }
}

function require(ok, message) {
  if (!ok) throw new RafWbError(message);
}

function checkSource(source) {
  if (!source || typeof source.size !== "number" || !Number.isSafeInteger(source.size) || source.size < 0 || typeof source.slice !== "function")
    throw new TypeError("Source must be a Blob/File or expose integer `size` and `slice(start, end)`");
}

/** Bounded reader: every read must lie inside the file and the given section. */
class BoundedReader {
  constructor(source) {
    checkSource(source);
    this.source = source;
    this.size = source.size;
    this.bytesRead = 0;
    this.reads = 0;
  }

  async read(offset, length, lower = 0, upper = this.size) {
    require(Number.isSafeInteger(offset) && Number.isSafeInteger(length) && length >= 0, "Invalid read request");
    require(length <= MAX_READ_BYTES, `Read of ${length} bytes exceeds the ${MAX_READ_BYTES}-byte cap`);
    require(lower >= 0 && upper <= this.size && offset >= lower && offset + length <= upper,
      `Read [${offset}, ${offset + length}) is outside bounds [${lower}, ${upper})`);
    const blob = this.source.slice(offset, offset + length);
    require(blob && typeof blob.arrayBuffer === "function", "slice() result has no arrayBuffer()");
    const buffer = await blob.arrayBuffer();
    require(buffer.byteLength === length, "Truncated read");
    this.bytesRead += length;
    this.reads += 1;
    return new DataView(buffer);
  }
}

function parseWbRecord(view, entryPos, little, tag, sectionStart, sectionEnd) {
  const type = view.getUint16(entryPos + 2, little);
  const count = view.getUint32(entryPos + 4, little);
  require((type === 3 || type === 4) && count === 3,
    `Unsupported 0x${tag.toString(16)} record: type ${type}, count ${count} (expected SHORT/LONG x3)`);
  const width = type === 3 ? 2 : 4;
  return { type, count, width, valueOffsetRel: view.getUint32(entryPos + 8, little), sectionStart, sectionEnd };
}

/**
 * Read the 0xf00d and 0xf00e records.
 *
 * @returns {Promise<{autoRgbg:number[], asShotCamMul:number[], records:object, section:object,
 *                    byteOrder:string, ifds:object[], bytesRead:number, reads:number}>}
 */
export async function readRafWbCoefficients(source) {
  const r = new BoundedReader(source);
  const header = await r.read(0, RAF_HEADER_BYTES);
  for (let i = 0; i < RAF_MAGIC.length; i++)
    require(header.getUint8(i) === RAF_MAGIC.charCodeAt(i), "Not a supported RAF header");
  const start = header.getUint32(CFA_OFFSET_FIELD, false);
  const length = header.getUint32(CFA_LENGTH_FIELD, false);
  require(start >= RAF_HEADER_BYTES && length >= 8 && start + length <= r.size, "CFA/TIFF section out of bounds");
  const end = start + length;

  const tiffHeader = await r.read(start, 8, start, end);
  const order = String.fromCharCode(tiffHeader.getUint8(0), tiffHeader.getUint8(1));
  require(order === "II" || order === "MM", "Unsupported TIFF byte order");
  const little = order === "II";
  require(tiffHeader.getUint16(2, little) === 42, "Unsupported TIFF version");

  const pending = [tiffHeader.getUint32(4, little)];
  const seen = new Set();
  const ifds = [];
  const found = { [TAG_WB_AUTO]: [], [TAG_WB_AS_SHOT]: [] };

  while (pending.length) {
    const rel = pending.shift();
    if (rel === 0) continue;
    require(!seen.has(rel), `IFD cycle or repeated link at relative offset ${rel}`);
    seen.add(rel);
    require(seen.size <= MAX_IFDS, `More than ${MAX_IFDS} IFDs`);
    require(rel >= 8, "IFD overlaps TIFF header");
    const pos = start + rel;
    const countView = await r.read(pos, 2, start, end);
    const n = countView.getUint16(0, little);
    require(n <= MAX_ENTRIES, `IFD entry count ${n} exceeds ${MAX_ENTRIES}`);
    const entries = await r.read(pos + 2, n * 12 + 4, start, end);
    ifds.push({ offsetRelative: rel, entries: n });
    for (let i = 0; i < n; i++) {
      const e = i * 12;
      const tag = entries.getUint16(e, little);
      if (tag === TAG_SUB_IFD) {
        const type = entries.getUint16(e + 2, little);
        const count = entries.getUint32(e + 4, little);
        require((type === 4 || type === 13) && count === 1, "Unsupported Fuji sub-IFD pointer");
        pending.push(entries.getUint32(e + 8, little));
      } else if (tag === TAG_WB_AUTO || tag === TAG_WB_AS_SHOT) {
        const rec = parseWbRecord(entries, e, little, tag, start, end);
        const bytes = rec.width * 3;
        // 6 or 12 bytes never fit the 4-byte inline slot, so the value is always an offset.
        const valuePos = start + rec.valueOffsetRel;
        const data = await r.read(valuePos, bytes, start, end);
        const grb = [0, 1, 2].map(k => (rec.width === 2 ? data.getUint16(k * 2, little) : data.getUint32(k * 4, little)));
        require(grb.every(v => Number.isInteger(v) && v > 0 && v <= U32_MAX), `0x${tag.toString(16)} levels must be positive`);
        found[tag].push({ grb, type: rec.type, count: rec.count, ifdOffsetRelative: rel, entryIndex: i, valueOffsetRelative: rec.valueOffsetRel });
      }
      // Every other tag (strips, tiles, 0xf007/0xf008 pixel pointers, unknown fields) is skipped without dereferencing.
    }
    pending.push(entries.getUint32(n * 12, little));
  }

  const pick = tag => {
    const list = found[tag];
    require(list.length > 0, `Missing 0x${tag.toString(16)} record`);
    const key = JSON.stringify(list[0].grb);
    require(list.every(x => JSON.stringify(x.grb) === key), `Conflicting duplicate 0x${tag.toString(16)} records`);
    return list;
  };
  const auto = pick(TAG_WB_AUTO), asShot = pick(TAG_WB_AS_SHOT);
  const rgb = grb => GRB_TO_RGB.map(i => grb[i]);
  const autoRgb = rgb(auto[0].grb), asShotRgb = rgb(asShot[0].grb);
  return {
    autoRgbg: [autoRgb[0], autoRgb[1], autoRgb[2], autoRgb[1]],
    asShotCamMul: [asShotRgb[0], asShotRgb[1], asShotRgb[2], 0],
    records: { f00d: auto, f00e: asShot },
    section: { start, length },
    byteOrder: order,
    ifds,
    bytesRead: r.bytesRead,
    reads: r.reads,
    note: "Source metadata only. The caller decides WB mode/policy; nothing here is applied-parameter readback.",
  };
}
