/**
 * Strict research codec for ONE contract: fit-derived Kelvin 6600 neutral-WB profile inputs.
 * UNSELECTED RESEARCH CODE - never a runtime/live input adapter and never a v2 extension.
 * Mirrors training/fjnrgb_profile.py. Reuses only the public pure helpers of ./fjlrg-v2.mjs
 * (parseStrictJson, sha256Hex, serializeOptions); v2 constants and validators are not touched.
 *
 * Contract research-linear-fitprofile-k6600, schema fujiapp.raw-input.research-neutralwb.v1.
 * Header: magic FJNRGB16, version 1, 64-byte LE header with v2 field geometry, transfer 1,
 * primaries 1, wb_policy 2 (userMul), wb_coeff_source 3 (frozen fit profile), levels
 * [620,302,455,302], options SHA-256 prefix, uint16 RGB, long edge <= 1536. Levels are DECLARED
 * userMul (parameters_are_declared), never applied-parameter readback.
 */
import { parseStrictJson, sha256Hex, serializeOptions } from "./fjlrg-v2.mjs";

export const HEADER_BYTES = 64;
export const MAGIC = "FJNRGB16";
export const FORMAT_VERSION = 1;
export const SCHEMA = "fujiapp.raw-input.research-neutralwb.v1";
export const CONTRACT = "research-linear-fitprofile-k6600";
export const WB_POLICY_USER_MUL = 2;
export const COEFF_SOURCE_FIT_PROFILE = 3;
export const WB_SOURCE_KIND = "fit_profile";
export const PROFILE_LEVELS_RGBG = Object.freeze([620, 302, 455, 302]);
export const PROFILE_MANIFEST_SHA256 = "6f6dac1c4bdd619ae1daacac36581866b1b71b059fc970f23145b89ffbed8ffc";
export const PROFILE_KEY = "4080/6600";
export const POLICY_BASIS = "calibrated-kelvin-profile-candidate";
export const PRIMITIVE_REPORTS = Object.freeze({
  sparse_primitive_report_sha256: "4fcb31eb72b7e41f3959d360c14c868501e31923142fc025a001d5b3581f213f",
  fullnative_v2_report_sha256: "d320a1a656a7c16ab2324b4ac1046b469f4d0f57fe03dd3c0aa36cf27a458027",
  fullnative_v2_verification_sha256: "0900736496e9967d45d936e300946aa6a702427a572a6f91e9b310d40038de32",
});
export const SOURCE_WB_MODE = 4080;
export const SOURCE_KELVIN = 6600;
export const SOURCE_FINE_TUNE_RAW = Object.freeze([20, -60]);

const HEX = /^[0-9a-f]{64}$/;
const SIDECAR_KEYS = ["schema", "input_contract", "container_sha256", "source_raf_sha256", "wasm_sha256",
  "decoder_package", "decoder_version", "libraw_build_version", "options_json", "options_sha256", "declared_options",
  "wb_source", "source_wb_mode", "source_wb_fine_tune_raw", "source_kelvin", "capture_dynamic_range",
  "capture_dynamic_range_source", "decoded_native_size", "parameters_are_declared",
  "profile_manifest_sha256", "profile_key", "policy_basis", "primitive_reports"];

function requireThat(ok, why) { if (!ok) throw new Error(`FJNRGB profile: ${why}`); }
function object(v) { return v !== null && typeof v === "object" && !Array.isArray(v); }
function sameKeys(v, allowed) { return object(v) && Object.keys(v).sort().join("\0") === [...allowed].sort().join("\0"); }
function integer(v, lo, hi) { return Number.isSafeInteger(v) && v >= lo && v <= hi; }
function same(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => same(v, b[i]));
  if (object(a) || object(b)) return object(a) && object(b) && sameKeys(a, Object.keys(b)) && Object.keys(a).every(k => same(a[k], b[k]));
  return typeof a === typeof b && a === b;
}
function workingSize(nativeWidth, nativeHeight) {
  const scale = Math.min(1, 1536 / Math.max(nativeWidth, nativeHeight));
  return [Math.max(1, Math.round(nativeWidth * scale)), Math.max(1, Math.round(nativeHeight * scale))];
}

/** The only accepted decode options: v2 common options, camera WB off, userMul = profile. */
export function canonicalOptions() {
  return { halfSize: false, outputBps: 16, outputColor: 1, gamm: [1, 1, 0, 0, 0, 0], noAutoBright: true,
    adjustMaximumThr: 0, useCameraMatrix: 1, useCameraWb: false, useAutoWb: false, highlight: 0, userQual: 3,
    useFujiRotate: -1, fbddNoiserd: 0, userMul: [...PROFILE_LEVELS_RGBG] };
}

export function encodeFjnrgbProfile(frame, { optionsSha256 }) {
  requireThat(integer(frame?.width, 1, 1536) && integer(frame?.height, 1, 1536), "unsupported working dimensions");
  requireThat(frame.data instanceof Uint16Array && frame.data.length === frame.width * frame.height * 3, "uint16 RGB payload shape differs");
  requireThat(typeof optionsSha256 === "string" && HEX.test(optionsSha256), "invalid options SHA256");
  const bytes = new Uint8Array(HEADER_BYTES + frame.data.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < 8; i++) bytes[i] = MAGIC.charCodeAt(i);
  for (const [o, v] of [[8, FORMAT_VERSION], [10, 64], [20, 3], [22, 16], [32, 1], [34, 1], [36, WB_POLICY_USER_MUL], [38, COEFF_SOURCE_FIT_PROFILE]]) view.setUint16(o, v, true);
  for (const [o, v] of [[12, frame.width], [16, frame.height], [24, frame.data.length * 2], [28, 3]]) view.setUint32(o, v, true);
  for (let i = 0; i < 4; i++) view.setUint32(40 + 4 * i, PROFILE_LEVELS_RGBG[i], true);
  for (let i = 0; i < 8; i++) bytes[56 + i] = Number.parseInt(optionsSha256.slice(2 * i, 2 * i + 2), 16);
  for (let i = 0; i < frame.data.length; i++) view.setUint16(HEADER_BYTES + 2 * i, frame.data[i], true);
  return bytes;
}

/** Canonical sidecar for a container; validated before it is returned. */
export async function buildProvenance(container, { sourceRafSha256, wasmSha256, captureDynamicRange, captureDynamicRangeSource, decodedNativeSize }) {
  const optionsJson = serializeOptions(canonicalOptions());
  const provenance = {
    schema: SCHEMA, input_contract: CONTRACT, container_sha256: await sha256Hex(container),
    source_raf_sha256: sourceRafSha256, wasm_sha256: wasmSha256,
    decoder_package: "libraw-wasm", decoder_version: "1.6.0", libraw_build_version: "0.22.1",
    options_json: optionsJson, options_sha256: await sha256Hex(new TextEncoder().encode(optionsJson)),
    declared_options: parseStrictJson(optionsJson),
    wb_source: { kind: WB_SOURCE_KIND, levels_rgbg: [...PROFILE_LEVELS_RGBG] },
    source_wb_mode: SOURCE_WB_MODE, source_wb_fine_tune_raw: [...SOURCE_FINE_TUNE_RAW], source_kelvin: SOURCE_KELVIN,
    capture_dynamic_range: captureDynamicRange, capture_dynamic_range_source: captureDynamicRangeSource,
    decoded_native_size: [...decodedNativeSize], parameters_are_declared: true,
    profile_manifest_sha256: PROFILE_MANIFEST_SHA256, profile_key: PROFILE_KEY, policy_basis: POLICY_BASIS,
    primitive_reports: { ...PRIMITIVE_REPORTS },
  };
  await decodeFjnrgbProfile(container, provenance, { expectedSourceSha256: sourceRafSha256, expectedWasmSha256: wasmSha256 });
  return provenance;
}

/** Validates header, sidecar and REQUIRED caller expectations before returning copied pixels. */
export async function decodeFjnrgbProfile(bytes, provenance, { expectedSourceSha256, expectedWasmSha256 } = {}) {
  requireThat(typeof expectedSourceSha256 === "string" && HEX.test(expectedSourceSha256), "expected source RAF SHA256 is required");
  requireThat(typeof expectedWasmSha256 === "string" && HEX.test(expectedWasmSha256), "expected WASM SHA256 is required");
  requireThat(bytes instanceof Uint8Array && bytes.byteLength >= HEADER_BYTES, "incomplete container");
  bytes = new Uint8Array(bytes);
  provenance = structuredClone(provenance);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  requireThat(String.fromCharCode(...bytes.subarray(0, 8)) === MAGIC, "wrong magic (FJLRGB16 v1/v2 refused)");
  requireThat(view.getUint16(8, true) === FORMAT_VERSION && view.getUint16(10, true) === 64, "wrong version/header size");
  const width = view.getUint32(12, true), height = view.getUint32(16, true);
  requireThat(integer(width, 1, 1536) && integer(height, 1, 1536), "unsupported working dimensions");
  const count = width * height * 3;
  requireThat(view.getUint16(20, true) === 3 && view.getUint16(22, true) === 16 && view.getUint32(24, true) === count * 2
    && bytes.byteLength === HEADER_BYTES + count * 2, "invalid payload size/channels/bits or trailing bytes");
  requireThat(view.getUint32(28, true) === 3 && view.getUint16(32, true) === 1 && view.getUint16(34, true) === 1, "unknown flags, transfer or primaries");
  requireThat(view.getUint16(36, true) === WB_POLICY_USER_MUL && view.getUint16(38, true) === COEFF_SOURCE_FIT_PROFILE, "header WB policy/source differ");
  const wb = Array.from({ length: 4 }, (_, i) => view.getUint32(40 + 4 * i, true));
  requireThat(same(wb, [...PROFILE_LEVELS_RGBG]), "header levels are not the frozen profile");

  requireThat(sameKeys(provenance, SIDECAR_KEYS), "provenance fields differ");
  requireThat(provenance.schema === SCHEMA && provenance.input_contract === CONTRACT, "provenance contract differs");
  for (const key of ["container_sha256", "source_raf_sha256", "wasm_sha256", "options_sha256"])
    requireThat(typeof provenance[key] === "string" && HEX.test(provenance[key]), `${key} is invalid`);
  requireThat(await sha256Hex(bytes) === provenance.container_sha256, "container digest differs");
  requireThat(provenance.source_raf_sha256 === expectedSourceSha256, "source fingerprint differs");
  requireThat(provenance.wasm_sha256 === expectedWasmSha256, "decoder fingerprint differs");
  requireThat(provenance.decoder_package === "libraw-wasm" && provenance.decoder_version === "1.6.0" && provenance.libraw_build_version === "0.22.1", "unsupported decoder provenance");
  requireThat(provenance.parameters_are_declared === true, "parameter-readback claim unsupported");
  requireThat(typeof provenance.options_json === "string" && await sha256Hex(new TextEncoder().encode(provenance.options_json)) === provenance.options_sha256, "options string digest differs");
  requireThat([...bytes.subarray(56, 64)].map(v => v.toString(16).padStart(2, "0")).join("") === provenance.options_sha256.slice(0, 16), "header options prefix differs");
  let parsed;
  try { parsed = parseStrictJson(provenance.options_json); } catch { throw new Error("FJNRGB profile: malformed or ambiguous options JSON"); }
  requireThat(same(parsed, provenance.declared_options), "declared options differ from saved JSON");
  requireThat(same(parsed, canonicalOptions()), "decode options are not the canonical profile set");
  requireThat(sameKeys(provenance.wb_source, ["kind", "levels_rgbg"]) && provenance.wb_source.kind === WB_SOURCE_KIND
    && same(provenance.wb_source.levels_rgbg, wb), "WB source record differs");
  requireThat(provenance.source_wb_mode === SOURCE_WB_MODE && provenance.source_kelvin === SOURCE_KELVIN
    && same(provenance.source_wb_fine_tune_raw, [...SOURCE_FINE_TUNE_RAW]), "source WB parameters are not Kelvin 6600 / [20,-60]");
  requireThat([100, 200, 400].includes(provenance.capture_dynamic_range) && typeof provenance.capture_dynamic_range === "number", "invalid capture DR");
  requireThat(typeof provenance.capture_dynamic_range_source === "string" && provenance.capture_dynamic_range_source.trim().length > 0, "missing capture DR provenance");
  const native = provenance.decoded_native_size;
  requireThat(Array.isArray(native) && native.length === 2 && native.every(v => integer(v, 1, 0xffffffff)), "invalid decoded native size");
  requireThat(same(workingSize(native[0], native[1]), [width, height]), "working dimensions differ from canonical native resize");
  requireThat(provenance.profile_manifest_sha256 === PROFILE_MANIFEST_SHA256 && provenance.profile_key === PROFILE_KEY
    && provenance.policy_basis === POLICY_BASIS, "profile identity differs");
  requireThat(same(provenance.primitive_reports, { ...PRIMITIVE_REPORTS }), "primitive report hashes differ");
  const data = new Uint16Array(count);
  for (let i = 0; i < count; i++) data[i] = view.getUint16(HEADER_BYTES + 2 * i, true);
  return { width, height, data, inputContract: CONTRACT, provenance };
}
