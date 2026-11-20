#!/usr/bin/env node
/**
 * Synthetic tests for scripts/lib/raf-wb-coefficients.mjs. No real RAF is read.
 * Run: node scripts/test-raf-wb-coefficients.mjs
 *
 * Expected values are literals: the WB levels below are the DSCF0391 values
 * recorded by the independent Python/ExifTool source probe, used here only as
 * numbers inside synthetic files.
 */
import assert from "node:assert/strict";
import { readRafWbCoefficients, RafWbError, MAX_READ_BYTES } from "./lib/raf-wb-coefficients.mjs";

const AUTO_GRB = [302, 518, 570];    // file order G, R, B
const AS_SHOT_GRB = [302, 628, 445];
const CFA_START = 128;

/**
 * Build a synthetic RAF. `ifds` is a list of { entries, next } where entries are
 * { tag, type, count, values } (written out of line), { tag, type, count, raw } (inline
 * value) or { tag: 0xf000, ifdRef } (sub-IFD pointer). `next` is an IFD index, null,
 * or { raw } for an arbitrary relative offset.
 */
function buildRaf({ order = "II", ifds, magic = "FUJIFILMCCD-RAW ", cfaStart = CFA_START,
                    sectionLength = null, pixelBytes = 256 } = {}) {
  const little = order === "II";
  let rel = 8;
  const ifdOffsets = ifds.map(ifd => { const at = rel; rel += 2 + ifd.entries.length * 12 + 4; return at; });
  const blobs = [];
  for (const ifd of ifds) for (const e of ifd.entries) if (e.values) {
    const width = e.type === 3 ? 2 : 4;
    e._at = rel; rel += e.values.length * width + (e.values.length * width) % 2;
    blobs.push(e);
  }
  const tiffLength = rel;
  const pixelStart = cfaStart + tiffLength;           // "pixel" bytes live after the TIFF section
  const total = pixelStart + pixelBytes;
  const buf = new Uint8Array(total), view = new DataView(buf.buffer);
  for (let i = 0; i < 16; i++) buf[i] = magic.charCodeAt(i);
  view.setUint32(0x64, cfaStart, false);
  view.setUint32(0x68, sectionLength ?? tiffLength, false);
  const t = cfaStart;
  buf[t] = buf[t + 1] = order.charCodeAt(0);
  view.setUint16(t + 2, 42, little);
  view.setUint32(t + 4, ifdOffsets[0], little);
  ifds.forEach((ifd, k) => {
    let p = t + ifdOffsets[k];
    view.setUint16(p, ifd.entries.length, little); p += 2;
    for (const e of ifd.entries) {
      view.setUint16(p, e.tag, little);
      view.setUint16(p + 2, e.type ?? 4, little);
      view.setUint32(p + 4, e.count ?? 1, little);
      const value = e.ifdRef !== undefined ? ifdOffsets[e.ifdRef] : e.values ? e._at : e.raw ?? 0;
      view.setUint32(p + 8, value >>> 0, little);
      p += 12;
    }
    const next = ifd.next === null || ifd.next === undefined ? 0 : typeof ifd.next === "object" ? ifd.next.raw : ifdOffsets[ifd.next];
    view.setUint32(p, next >>> 0, little);
  });
  for (const e of blobs) e.values.forEach((v, i) => {
    if (e.type === 3) view.setUint16(t + e._at + i * 2, v, little);
    else view.setUint32(t + e._at + i * 4, v >>> 0, little);
  });
  buf.fill(0xab, pixelStart);
  return { bytes: buf, pixelStart, tiffLength };
}

/** Duck-typed source that records every slice, so tests can prove what was (not) read. */
function recordingSource(bytes) {
  const reads = [];
  return {
    reads,
    size: bytes.length,
    slice(start, end) {
      reads.push([start, end]);
      const part = bytes.slice(start, end);
      return { arrayBuffer: async () => part.buffer.slice(part.byteOffset, part.byteOffset + part.byteLength) };
    },
  };
}

const wb = (tag, values, type = 4, count = values.length) => ({ tag, type, count, values });
const standardIfds = (type = 4, extra = []) => [
  { entries: [{ tag: 0xf000, type: 4, count: 1, ifdRef: 1 }], next: null },
  { entries: [
    { tag: 0xf007, type: 4, count: 1, raw: 0xfffffff0 },   // pixel pointer far outside the file: must not be read
    { tag: 0xf008, type: 4, count: 1, raw: 0x7fffffff },
    wb(0xf00d, AUTO_GRB, type), wb(0xf00e, AS_SHOT_GRB, type),
    { tag: 0x1234, type: 4, count: 99, raw: 0x7ffffff0 },  // unknown field: skipped, not dereferenced
    ...extra], next: null },
];

async function rejects(promise, pattern) {
  await assert.rejects(promise, err => err instanceof RafWbError && pattern.test(err.message), `expected ${pattern}`);
}

const tests = {
  async "II LONG records give exact RGB order"() {
    const { bytes } = buildRaf({ ifds: standardIfds(4) });
    const out = await readRafWbCoefficients(new Blob([bytes]));
    assert.deepEqual(out.autoRgbg, [518, 302, 570, 302]);
    assert.deepEqual(out.asShotCamMul, [628, 302, 445, 0]);
    assert.deepEqual(out.records.f00d[0].grb, [302, 518, 570]);
    assert.deepEqual(out.records.f00e[0].grb, [302, 628, 445]);
    assert.equal(out.byteOrder, "II");
  },

  async "MM SHORT records give the same values"() {
    const { bytes } = buildRaf({ order: "MM", ifds: standardIfds(3) });
    const out = await readRafWbCoefficients(new Blob([bytes]));
    assert.deepEqual(out.autoRgbg, [518, 302, 570, 302]);
    assert.deepEqual(out.asShotCamMul, [628, 302, 445, 0]);
    assert.equal(out.records.f00d[0].type, 3);
  },

  async "full uint32 levels survive"() {
    const ifds = [{ entries: [wb(0xf00d, [70000, 90000, 80000]), wb(0xf00e, [302, 628, 445])], next: null }];
    const out = await readRafWbCoefficients(new Blob([buildRaf({ ifds }).bytes]));
    assert.deepEqual(out.autoRgbg, [90000, 70000, 80000, 70000]);
  },

  async "records split across IFD0, sub-IFD and a next-IFD chain"() {
    const ifds = [
      { entries: [wb(0xf00e, AS_SHOT_GRB), { tag: 0xf000, type: 13, count: 1, ifdRef: 2 }], next: 1 },
      { entries: [{ tag: 0x0100, type: 3, count: 1, raw: 7 }], next: null },
      { entries: [wb(0xf00d, AUTO_GRB, 3)], next: null },
    ];
    const out = await readRafWbCoefficients(new Blob([buildRaf({ ifds }).bytes]));
    assert.deepEqual(out.autoRgbg, [518, 302, 570, 302]);
    assert.deepEqual(out.asShotCamMul, [628, 302, 445, 0]);
    assert.equal(out.ifds.length, 3);
  },

  async "pixel and unknown pointers are never read; reads are small and bounded"() {
    const { bytes, pixelStart } = buildRaf({ ifds: standardIfds(4), pixelBytes: 4096 });
    const src = recordingSource(bytes);
    const out = await readRafWbCoefficients(src);
    for (const [s, e] of src.reads) {
      assert(e <= pixelStart, `read [${s}, ${e}) touched pixel bytes at ${pixelStart}`);
      assert(e - s <= MAX_READ_BYTES);
    }
    assert.equal(out.bytesRead, src.reads.reduce((a, [s, e]) => a + e - s, 0));
    assert(out.bytesRead < 400, `bytesRead ${out.bytesRead}`);
  },

  async "cycles and repeated links are refused"() {
    const ifds = [
      { entries: [{ tag: 0xf000, type: 4, count: 1, ifdRef: 1 }], next: null },
      { entries: [wb(0xf00d, AUTO_GRB), wb(0xf00e, AS_SHOT_GRB)], next: 0 },
    ];
    await rejects(readRafWbCoefficients(new Blob([buildRaf({ ifds }).bytes])), /cycle|repeated/);
  },

  async "IFD count cap enforced"() {
    const ifds = Array.from({ length: 17 }, (_, k) => ({ entries: [], next: k < 16 ? k + 1 : null }));
    ifds[0].entries = [wb(0xf00d, AUTO_GRB), wb(0xf00e, AS_SHOT_GRB)];
    await rejects(readRafWbCoefficients(new Blob([buildRaf({ ifds }).bytes])), /More than 16 IFDs/);
  },

  async "entry count cap enforced without a huge read"() {
    const { bytes } = buildRaf({ ifds: standardIfds(4) });
    new DataView(bytes.buffer).setUint16(CFA_START + 8, 1025, true);
    await rejects(readRafWbCoefficients(new Blob([bytes])), /exceeds 1024/);
  },

  async "pointers outside the section are refused"() {
    const outOfSection = [{ entries: [wb(0xf00d, AUTO_GRB), wb(0xf00e, AS_SHOT_GRB), { tag: 0xf000, type: 4, count: 1, raw: 0 }], next: { raw: 100000 } }];
    await rejects(readRafWbCoefficients(new Blob([buildRaf({ ifds: outOfSection }).bytes])), /outside bounds/);
    // A value offset that points into the (in-file) pixel area beyond the section end.
    const { bytes, tiffLength } = buildRaf({ ifds: standardIfds(4) });
    const view = new DataView(bytes.buffer);
    const subIfd = CFA_START + 8 + 2 + 12 + 4;          // IFD0 holds one entry
    view.setUint32(subIfd + 2 + 2 * 12 + 8, tiffLength + 8, true); // f00d value offset past section end
    await rejects(readRafWbCoefficients(new Blob([bytes])), /outside bounds/);
  },

  async "unsupported type or count is refused, never coerced"() {
    for (const rec of [wb(0xf00d, [302, 518, 570, 518]), wb(0xf00d, AUTO_GRB, 5), wb(0xf00d, [302, 518], 4)]) {
      const ifds = [{ entries: [rec, wb(0xf00e, AS_SHOT_GRB)], next: null }];
      await rejects(readRafWbCoefficients(new Blob([buildRaf({ ifds }).bytes])), /Unsupported 0xf00d/);
    }
  },

  async "missing records are refused"() {
    for (const entries of [[wb(0xf00d, AUTO_GRB)], [wb(0xf00e, AS_SHOT_GRB)], []]) {
      await rejects(readRafWbCoefficients(new Blob([buildRaf({ ifds: [{ entries, next: null }] }).bytes])), /Missing 0xf00/);
    }
  },

  async "zero WB levels are refused"() {
    const ifds = [{ entries: [wb(0xf00d, [302, 0, 570]), wb(0xf00e, AS_SHOT_GRB)], next: null }];
    await rejects(readRafWbCoefficients(new Blob([buildRaf({ ifds }).bytes])), /positive/);
  },

  async "duplicates: identical accepted, conflicting refused"() {
    const same = [{ entries: [wb(0xf00d, AUTO_GRB), wb(0xf00e, AS_SHOT_GRB)], next: 1 },
                  { entries: [wb(0xf00d, AUTO_GRB, 3)], next: null }];
    const out = await readRafWbCoefficients(new Blob([buildRaf({ ifds: same }).bytes]));
    assert.equal(out.records.f00d.length, 2);
    const conflict = [{ entries: [wb(0xf00d, AUTO_GRB), wb(0xf00e, AS_SHOT_GRB)], next: 1 },
                      { entries: [wb(0xf00d, [302, 519, 570])], next: null }];
    await rejects(readRafWbCoefficients(new Blob([buildRaf({ ifds: conflict }).bytes])), /Conflicting duplicate 0xf00d/);
  },

  async "bad magic, bad section and bad TIFF header are refused"() {
    await rejects(readRafWbCoefficients(new Blob([buildRaf({ ifds: standardIfds(), magic: "FUJIFILMCCD-RAX " }).bytes])), /RAF header/);
    await rejects(readRafWbCoefficients(new Blob([buildRaf({ ifds: standardIfds(), sectionLength: 1e6 }).bytes])), /section out of bounds/);
    const overlap = buildRaf({ ifds: standardIfds() }).bytes;
    new DataView(overlap.buffer).setUint32(0x64, 64, false); // section starting inside the 112-byte header
    await rejects(readRafWbCoefficients(new Blob([overlap])), /section out of bounds/);
    const { bytes } = buildRaf({ ifds: standardIfds() });
    bytes[CFA_START] = 0x58;
    await rejects(readRafWbCoefficients(new Blob([bytes])), /byte order/);
    await rejects(readRafWbCoefficients(new Blob([bytes.slice(0, 50)])), /outside bounds/);
  },

  async "non-Blob sources are rejected"() {
    await assert.rejects(readRafWbCoefficients({}), TypeError);
    await assert.rejects(readRafWbCoefficients(new Uint8Array(200)), TypeError);
  },
};

let failed = 0;
for (const [name, fn] of Object.entries(tests)) {
  try { await fn(); console.log(`PASS ${name}`); }
  catch (err) { failed++; console.error(`FAIL ${name}\n  ${err.stack}`); }
}
console.log(`${Object.keys(tests).length - failed}/${Object.keys(tests).length} passed`);
process.exitCode = failed ? 1 : 0;
