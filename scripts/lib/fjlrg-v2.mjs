/** Standalone research container. Never an input adapter for frozen Round 11. */
export const HEADER_BYTES = 64;
export const CONTRACTS = Object.freeze({
  "v2-linear-asshot": Object.freeze({ policy: 1, source: 1, kind: "cam_mul" }),
  "v2-linear-autowb": Object.freeze({ policy: 2, source: 2, kind: "fuji_f00d" }),
});
const MAGIC = "FJLRGB16";
const HEX = /^[0-9a-f]{64}$/;
const ROOT_KEYS = ["schema", "input_contract", "container_sha256", "source_raf_sha256", "wasm_sha256",
  "decoder_package", "decoder_version", "libraw_build_version", "options_json", "options_sha256", "declared_options",
  "wb_source", "source_wb_mode", "source_wb_fine_tune_raw", "source_kelvin", "capture_dynamic_range",
  "capture_dynamic_range_source", "decoded_native_size", "parameters_are_declared"];

function requireThat(ok, why) { if (!ok) throw new Error(`FJLRGB v2: ${why}`); }
function object(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
function keys(value, allowed, label) {
  requireThat(object(value) && Object.keys(value).sort().join("\0") === [...allowed].sort().join("\0"), `${label} fields differ`);
}
function integer(value, minimum, maximum) { return Number.isSafeInteger(value) && value >= minimum && value <= maximum; }
function same(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => same(v, b[i]));
  if (object(a) || object(b)) return object(a) && object(b) && Object.keys(a).sort().join("\0") === Object.keys(b).sort().join("\0") && Object.keys(a).every(k => same(a[k], b[k]));
  return typeof a === typeof b && a === b;
}
function dimensions(width, height) {
  requireThat(integer(width, 1, 1536) && integer(height, 1, 1536), "unsupported working dimensions");
}
function levels(values, contract) {
  requireThat(Array.isArray(values) && values.length === 4 && values.every(v => integer(v, 0, 0xffffffff)), "WB levels must be four uint32 integers");
  requireThat(values.slice(0, 3).every(v => v > 0), "RGB WB levels must be positive");
  requireThat(values[3] === values[1] || (contract.policy === 1 && values[3] === 0), "fourth WB level must match green or recorded zero");
}
function contractFor(name) {
  requireThat(typeof name === "string" && Object.hasOwn(CONTRACTS, name), "explicit supported input contract required");
  return CONTRACTS[name];
}

/** JSON.parse semantics plus duplicate-key, depth and non-finite refusal. */
export function parseStrictJson(text) {
  requireThat(typeof text === "string", "JSON must be text");
  let offset = 0;
  const space = () => { while (/[ \n\r\t]/.test(text[offset] ?? "") && offset < text.length) offset++; };
  function string() {
    const start = offset;
    requireThat(text[offset++] === '"', "JSON string expected");
    while (offset < text.length) {
      const ch = text[offset++];
      if (ch === '"') return JSON.parse(text.slice(start, offset));
      if (ch === "\\") offset++;
    }
    throw new Error("FJLRGB v2: unterminated JSON string");
  }
  function value(depth) {
    requireThat(depth <= 64, "JSON nesting exceeds limit");
    space();
    const ch = text[offset];
    if (ch === "{") {
      offset++; space();
      const seen = new Set();
      if (text[offset] === "}") { offset++; return; }
      while (offset < text.length) {
        space(); const key = string();
        requireThat(!seen.has(key), "duplicate JSON key"); seen.add(key);
        space(); requireThat(text[offset++] === ":", "JSON colon expected");
        value(depth + 1); space();
        const next = text[offset++];
        if (next === "}") return;
        requireThat(next === ",", "JSON comma expected");
      }
    } else if (ch === "[") {
      offset++; space();
      if (text[offset] === "]") { offset++; return; }
      while (offset < text.length) {
        value(depth + 1); space();
        const next = text[offset++];
        if (next === "]") return;
        requireThat(next === ",", "JSON array comma expected");
      }
    } else if (ch === '"') { string(); return; }
    else {
      const match = /^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)/.exec(text.slice(offset));
      requireThat(Boolean(match), "invalid JSON value"); offset += match[0].length; return;
    }
    throw new Error("FJLRGB v2: incomplete JSON");
  }
  value(0); space(); requireThat(offset === text.length, "trailing JSON data");
  return JSON.parse(text, (_key, item) => {
    requireThat(typeof item !== "number" || Number.isFinite(item), "non-finite JSON number"); return item;
  });
}

export async function sha256Hex(bytes) {
  requireThat(bytes instanceof Uint8Array, "hash input must be bytes");
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map(v => v.toString(16).padStart(2, "0")).join("");
}

/** Stable writer string. Verification hashes the saved string, never a rewrite. */
export function serializeOptions(options) {
  requireThat(object(options), "options must be an object");
  return JSON.stringify(Object.fromEntries(Object.keys(options).sort().map(k => [k, options[k]])));
}

export function encodeFjlrgV2(frame, { inputContract, wbLevelsRgbg, optionsSha256 }) {
  const c = contractFor(inputContract);
  dimensions(frame.width, frame.height);
  requireThat(frame.data instanceof Uint16Array && frame.data.length === frame.width * frame.height * 3, "uint16 RGB payload shape differs");
  levels(wbLevelsRgbg, c);
  requireThat(typeof optionsSha256 === "string" && HEX.test(optionsSha256), "invalid options SHA256");
  const bytes = new Uint8Array(HEADER_BYTES + frame.data.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < MAGIC.length; i++) bytes[i] = MAGIC.charCodeAt(i);
  for (const [offset, value] of [[8, 2], [10, 64], [20, 3], [22, 16], [32, 1], [34, 1], [36, c.policy], [38, c.source]]) view.setUint16(offset, value, true);
  for (const [offset, value] of [[12, frame.width], [16, frame.height], [24, frame.data.length * 2], [28, 3]]) view.setUint32(offset, value, true);
  for (let i = 0; i < 4; i++) view.setUint32(40 + 4 * i, wbLevelsRgbg[i], true);
  for (let i = 0; i < 8; i++) bytes[56 + i] = Number.parseInt(optionsSha256.slice(2 * i, 2 * i + 2), 16);
  for (let i = 0; i < frame.data.length; i++) view.setUint16(HEADER_BYTES + 2 * i, frame.data[i], true);
  return bytes;
}

function validateOptions(options, c, wb) {
  const expected = { halfSize: false, outputBps: 16, outputColor: 1, gamm: [1, 1, 0, 0, 0, 0], noAutoBright: true,
    adjustMaximumThr: 0, useCameraMatrix: 1, useCameraWb: c.policy === 1, useAutoWb: false,
    highlight: 0, userQual: 3, useFujiRotate: -1, fbddNoiserd: 0 };
  if (c.policy === 2) expected.userMul = wb;
  requireThat(same(options, expected), "decode options do not match the input contract");
}

/** Validates payload, sidecar and caller expectation before returning copied pixels. */
export async function decodeFjlrgV2(bytes, provenance, { expectedContract, expectedSourceSha256, expectedWasmSha256 } = {}) {
  const c = contractFor(expectedContract);
  requireThat(bytes instanceof Uint8Array && bytes.byteLength >= HEADER_BYTES, "incomplete container");
  bytes = new Uint8Array(bytes);
  provenance = structuredClone(provenance);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  requireThat(String.fromCharCode(...bytes.subarray(0, 8)) === MAGIC && view.getUint16(8, true) === 2 && view.getUint16(10, true) === 64, "wrong magic/version/header; legacy v1 is refused");
  const width = view.getUint32(12, true), height = view.getUint32(16, true);
  dimensions(width, height);
  const count = width * height * 3;
  requireThat(view.getUint16(20, true) === 3 && view.getUint16(22, true) === 16 && view.getUint32(24, true) === count * 2
    && bytes.byteLength === HEADER_BYTES + count * 2, "invalid payload size/channels/bits or trailing bytes");
  requireThat(view.getUint32(28, true) === 3 && view.getUint16(32, true) === 1 && view.getUint16(34, true) === 1, "unknown flags, transfer or primaries");
  requireThat(view.getUint16(36, true) === c.policy && view.getUint16(38, true) === c.source, "header WB policy/source mismatches requested contract");
  const wb = Array.from({ length: 4 }, (_, i) => view.getUint32(40 + 4 * i, true));
  levels(wb, c);
  keys(provenance, ROOT_KEYS, "provenance");
  requireThat(provenance.schema === "fujiapp.raw-input.v2" && provenance.input_contract === expectedContract, "provenance contract differs");
  for (const key of ["container_sha256", "source_raf_sha256", "wasm_sha256", "options_sha256"])
    requireThat(typeof provenance[key] === "string" && HEX.test(provenance[key]), `${key} is invalid`);
  requireThat(await sha256Hex(bytes) === provenance.container_sha256, "container digest differs");
  if (expectedSourceSha256 !== undefined) requireThat(HEX.test(expectedSourceSha256) && provenance.source_raf_sha256 === expectedSourceSha256, "source fingerprint differs");
  if (expectedWasmSha256 !== undefined) requireThat(HEX.test(expectedWasmSha256) && provenance.wasm_sha256 === expectedWasmSha256, "decoder fingerprint differs");
  requireThat(provenance.decoder_package === "libraw-wasm" && provenance.decoder_version === "1.6.0" && provenance.libraw_build_version === "0.22.1", "unsupported decoder provenance");
  requireThat(typeof provenance.options_json === "string" && await sha256Hex(new TextEncoder().encode(provenance.options_json)) === provenance.options_sha256, "options string digest differs");
  let parsed;
  try { parsed = parseStrictJson(provenance.options_json); } catch { throw new Error("FJLRGB v2: malformed or ambiguous options JSON"); }
  requireThat(same(parsed, provenance.declared_options), "declared options differ from saved JSON");
  validateOptions(parsed, c, wb);
  requireThat([...bytes.subarray(56, 64)].map(v => v.toString(16).padStart(2, "0")).join("") === provenance.options_sha256.slice(0, 16), "header options prefix differs");
  keys(provenance.wb_source, ["kind", "levels_rgbg"], "WB source");
  requireThat(provenance.wb_source.kind === c.kind && same(provenance.wb_source.levels_rgbg, wb), "source WB record differs");
  requireThat(integer(provenance.source_wb_mode, 0, 65535), "invalid source WB mode");
  if (c.policy === 2) requireThat(provenance.source_wb_mode === 0, "recorded Auto policy requires source WB mode0");
  requireThat(Array.isArray(provenance.source_wb_fine_tune_raw) && provenance.source_wb_fine_tune_raw.length === 2
    && provenance.source_wb_fine_tune_raw.every(v => integer(v, -180, 180) && v % 20 === 0), "invalid source WB fine tune");
  requireThat(provenance.source_kelvin === null || integer(provenance.source_kelvin, 1, 0xffffffff), "invalid Kelvin provenance");
  requireThat([100, 200, 400].includes(provenance.capture_dynamic_range) && typeof provenance.capture_dynamic_range === "number", "invalid capture DR");
  requireThat(typeof provenance.capture_dynamic_range_source === "string" && provenance.capture_dynamic_range_source.trim().length > 0, "missing capture DR provenance");
  const native = provenance.decoded_native_size;
  requireThat(Array.isArray(native) && native.length === 2 && native.every(v => integer(v, 1, 0xffffffff)) && native[0] >= width && native[1] >= height, "invalid decoded native size");
  const scale = Math.min(1, 1536 / Math.max(...native));
  requireThat(Math.max(1, Math.round(native[0] * scale)) === width && Math.max(1, Math.round(native[1] * scale)) === height, "working dimensions differ from canonical native resize");
  requireThat(provenance.parameters_are_declared === true, "parameter-readback claim unsupported");
  const data = new Uint16Array(count);
  for (let i = 0; i < count; i++) data[i] = view.getUint16(HEADER_BYTES + 2 * i, true);
  return { width, height, data, inputContract: expectedContract, provenance };
}

export function toInterleavedFloat32(frame) {
  requireThat(frame.data instanceof Uint16Array && frame.data.length === frame.width * frame.height * 3, "invalid decoded frame");
  return Float32Array.from(frame.data, v => v / 65535);
}
