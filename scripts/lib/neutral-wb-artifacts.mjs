/**
 * HOST-side (Node) strict normaliser for neutral-WB research decoder results. RESEARCH CODE ONLY.
 * No file I/O, no browser, no media. It validates results returned by the EXISTING, unmodified helpers:
 *   W rows: decodeResearchRaw (scripts/lib/decode-provia-v2-research.mjs)
 *           C = control (explicit as-shot userMul, raw working uint16 bytes, no provenance)
 *           N/R = v2-linear-autowb container (userMul = source auto_rgbg)
 *   P rows: decodeFjnrgbProfileResearch (scripts/lib/decode-fjnrgb-profile-research.mjs)
 *           C = control (explicit as-shot userMul, raw working bytes), N/R = FJNRGB16 profile container
 * Options are DECLARED, never applied-parameter readback. The W helper validates camera/WB/cam_mul
 * internally but does not return them: returned_metadata is null with an explicit note, never filled
 * from frozen expectations. Comparison against existing G files is the caller's job.
 */
import { createHash } from "node:crypto";
import * as V from "./fjlrg-v2.mjs";
import * as P from "./fjnrgb-profile.mjs";

export const ARMS = Object.freeze(["C", "N", "R"]);
export const KINDS = Object.freeze(["W", "P"]);
const HEX = /^[0-9a-f]{64}$/;
const B64 = /^[A-Za-z0-9+/]*={0,2}$/;
const W_KEYS = ["base64", "stripsBase64", "stripRows", "provenance", "native_sha256", "working_payload_sha256", "width", "height",
  "native_size", "source_coefficients", "options_json", "options_sha256"];
const P_KEYS = ["arm", "scene_name", "source_raf_sha256", "parameters_are_declared", "declared_options", "options_json", "options_sha256",
  "native_size", "working_size", "native_sha256", "working_payload_sha256", "source_coefficients", "decoder_readback", "strips", "note"];
const W_NOTE = "W helper validates camera model, WB preset and cam_mul internally but does not return them: not returned, not reconstructed.";

function need(ok, why) { if (!ok) throw new Error(`neutral-WB artifact: ${why}`); }
const sha = b => createHash("sha256").update(b).digest("hex");
const posInt = (v, max = 0xffffffff) => Number.isSafeInteger(v) && v > 0 && v <= max;
const isObj = v => v !== null && typeof v === "object" && !Array.isArray(v);
/** Deep equality: arrays ordered, object keys order-insensitive, primitives strict (no 1 == true). */
function same(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => same(v, b[i]));
  if (isObj(a) || isObj(b)) return isObj(a) && isObj(b) && Object.keys(a).length === Object.keys(b).length && Object.keys(a).every(k => Object.hasOwn(b, k) && same(a[k], b[k]));
  return typeof a === typeof b && a === b;
}
const sameKeys = (o, keys) => isObj(o) && Object.keys(o).sort().join("\0") === [...keys].sort().join("\0");

function workingSize(w, h) {
  const scale = Math.min(1, 1536 / Math.max(w, h));
  return [Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale))];
}
function stripRows(h) { const rows = []; for (let y = 64; y < h; y += 128) rows.push(y); return rows; }
function b64(text, label) {
  need(typeof text === "string" && text.length % 4 === 0 && B64.test(text), `${label} is not strict base64`);
  const buf = Buffer.from(text, "base64");
  need(buf.toString("base64") === text, `${label} base64 is not canonical`);
  return buf;
}

/** Strict frozen-source contract for a W or P fit row. Returns a defensive copy. */
export function validateScene(scene) {
  need(isObj(scene), "scene must be an object");
  need(KINDS.includes(scene.kind), `kind must be W or P (G/other refused), got ${scene.kind}`);
  need(scene.role === "fit", "fit rows only");
  need(typeof scene.name === "string" && /^DSCF\d{4}\.RAF$/.test(scene.name), "unsafe source name");
  need(typeof scene.raf_sha256 === "string" && HEX.test(scene.raf_sha256), "raf_sha256 must be 64 lowercase hex");
  const cam = scene.cam_mul, auto = scene.auto_rgbg, ft = scene.source_wb_fine_tune_raw;
  need(Array.isArray(cam) && cam.length === 4 && cam.slice(0, 3).every(v => posInt(v)) && cam[3] === 0, "cam_mul must be [R,G,B,0]");
  need(Array.isArray(auto) && auto.length === 4 && auto.every(v => posInt(v)) && auto[1] === auto[3], "auto_rgbg must be [R,G,B,G]");
  need(Array.isArray(ft) && ft.length === 2 && ft.every(v => Number.isSafeInteger(v) && v % 20 === 0 && Math.abs(v) <= 180), "invalid fine-tune");
  if (scene.kind === "W") {
    need(scene.source_wb_mode === 0 && scene.source_kelvin === null, "W requires source Auto (mode 0, Kelvin null)");
  } else {
    need(scene.source_wb_mode === P.SOURCE_WB_MODE && scene.source_kelvin === P.SOURCE_KELVIN && same(ft, [...P.SOURCE_FINE_TUNE_RAW]),
      "P requires 4080 / 6600 / [20,-60]");
  }
  need([100, 200, 400].includes(scene.capture_dynamic_range), "capture_dynamic_range must be 100/200/400");
  const nat = scene.expected_native_size, wk = scene.expected_working_size;
  need(Array.isArray(nat) && nat.length === 2 && nat.every(v => posInt(v)), "expected_native_size must be [w,h]");
  need(Array.isArray(wk) && wk.length === 2 && same(wk, workingSize(nat[0], nat[1])), "expected_working_size is not the canonical downsample");
  return structuredClone(scene);
}

/** Declared options per arm; all camera WB off. C keeps the source's own green. */
export function armOptions(scene, arm) {
  const s = validateScene(scene);
  need(ARMS.includes(arm), `unknown arm ${arm}`);
  const o = P.canonicalOptions();
  if (arm === "C") o.userMul = [s.cam_mul[0], s.cam_mul[1], s.cam_mul[2], s.cam_mul[1]];
  else if (s.kind === "W") o.userMul = [...s.auto_rgbg];
  return o;
}

function checkCommon(s, r, arm, options, nativeSize, workingSz, stripsText, rows, stripWidth) {
  need(same(nativeSize, s.expected_native_size), "native size differs from frozen");
  need(same(workingSz, s.expected_working_size), "working size differs from frozen");
  need(typeof r.native_sha256 === "string" && HEX.test(r.native_sha256), "invalid native_sha256");
  need(typeof r.working_payload_sha256 === "string" && HEX.test(r.working_payload_sha256), "invalid working_payload_sha256");
  need(r.options_json === V.serializeOptions(options), "options_json is not the declared arm options");
  need(r.options_sha256 === sha(Buffer.from(r.options_json, "utf8")), "options_sha256 does not hash options_json");
  need(same(rows, stripRows(nativeSize[1])), "strip rows are not 64, 192, ...");
  need(stripWidth === nativeSize[0], "strip width differs from native width");
  const strips = b64(stripsText, "strips");
  need(strips.length === rows.length * nativeSize[0] * 6, "strip byte count differs");
  return strips;
}

function normalized(s, arm, r, extra) {
  return { kind: s.kind, arm, name: s.name, parameters_are_declared: true, options_json: r.options_json, options_sha256: r.options_sha256,
    native_sha256: r.native_sha256, working_payload_sha256: r.working_payload_sha256, ...extra };
}

async function normalizeW(s, arm, r, wasmSha256) {
  const keys = Object.keys(r);
  need(keys.every(k => W_KEYS.includes(k)) && W_KEYS.filter(k => k !== "provenance").every(k => keys.includes(k)), "W result fields differ");
  const options = armOptions(s, arm);
  const sc = r.source_coefficients;
  need(isObj(sc) && same(sc.autoRgbg, s.auto_rgbg) && same(sc.asShotCamMul, s.cam_mul), "returned source coefficients differ from frozen source");
  const strips = checkCommon(s, r, arm, options, r.native_size, [r.width, r.height], r.stripsBase64, r.stripRows, r.native_size?.[0]);
  const bytes = b64(r.base64, "payload");
  let provenance = null, contract = null, containerSha = null;
  if (arm === "C") {
    need(r.provenance === undefined || r.provenance === null, "control must not carry provenance");
    need(bytes.length === s.expected_working_size[0] * s.expected_working_size[1] * 6, "control payload byte count differs");
    need(sha(bytes) === r.working_payload_sha256, "control payload SHA differs");
  } else {
    need(isObj(r.provenance), "N/R requires v2 provenance");
    const frame = await V.decodeFjlrgV2(new Uint8Array(bytes), r.provenance,
      { expectedContract: "v2-linear-autowb", expectedSourceSha256: s.raf_sha256, expectedWasmSha256: wasmSha256 });
    need(same([frame.width, frame.height], s.expected_working_size), "decoded container size differs");
    need(sha(Buffer.from(frame.data.buffer, frame.data.byteOffset, frame.data.byteLength)) === r.working_payload_sha256, "container payload SHA differs");
    need(same(r.provenance.declared_options, options) && r.provenance.options_json === r.options_json, "provenance options differ from declared arm options");
    need(same(r.provenance.wb_source, { kind: "fuji_f00d", levels_rgbg: s.auto_rgbg }), "provenance WB source differs");
    need(r.provenance.source_kelvin === null && same(r.provenance.source_wb_fine_tune_raw, s.source_wb_fine_tune_raw)
      && r.provenance.capture_dynamic_range === s.capture_dynamic_range && same(r.provenance.decoded_native_size, s.expected_native_size),
      "provenance source parameters differ");
    provenance = structuredClone(r.provenance); contract = "v2-linear-autowb"; containerSha = sha(bytes);
    need(provenance.container_sha256 === containerSha, "container SHA differs");
  }
  return normalized(s, arm, r, { bytes, strips, strip_rows: [...r.stripRows], provenance, contract, container_sha256: containerSha,
    source_coefficients: { autoRgbg: [...sc.autoRgbg], asShotCamMul: [...sc.asShotCamMul] }, returned_metadata: null, metadata_note: W_NOTE,
    declared_options: options, native_size: [...r.native_size], width: r.width, height: r.height });
}

async function normalizeP(s, arm, r, wasmSha256) {
  const payloadKey = arm === "C" ? "working_payload" : "container";
  need(sameKeys(r, [...P_KEYS, payloadKey]), "P result fields differ");
  need(r.arm === (arm === "C" ? "control" : "treatment"), "P helper arm differs");
  need(r.scene_name === s.name && r.source_raf_sha256 === s.raf_sha256 && r.parameters_are_declared === true, "P identity fields differ");
  const options = armOptions(s, arm);
  need(same(r.declared_options, options), "declared options differ from arm options");
  const sc = r.source_coefficients;
  need(sameKeys(sc, ["auto_rgbg", "as_shot_cam_mul"]) && same(sc.auto_rgbg, s.auto_rgbg) && same(sc.as_shot_cam_mul, s.cam_mul), "returned source coefficients differ");
  const md = r.decoder_readback;
  need(sameKeys(md, ["camera_model", "fujiWB", "cam_mul", "pre_mul"]) && md.camera_model === "X100VI" && md.fujiWB === P.SOURCE_WB_MODE
    && same(md.cam_mul, s.cam_mul) && (md.pre_mul === null || (Array.isArray(md.pre_mul) && md.pre_mul.length === 4 && md.pre_mul.every(Number.isFinite))),
    "returned decoder metadata differs (camera/WB/cam_mul)");
  const st = r.strips;
  need(sameKeys(st, ["rows", "width", "channels", "layout", "base64"]) && st.channels === 3 && st.layout === "uint16 LE, native rows interleaved RGB", "strip descriptor differs");
  const strips = checkCommon(s, r, arm, options, r.native_size, r.working_size, st.base64, st.rows, st.width);
  let bytes, provenance = null, contract = null, containerSha = null;
  if (arm === "C") {
    need(sameKeys(r.working_payload, ["layout", "base64"]) && r.working_payload.layout === "uint16 LE interleaved RGB, no container", "control payload descriptor differs");
    bytes = b64(r.working_payload.base64, "payload");
    need(bytes.length === s.expected_working_size[0] * s.expected_working_size[1] * 6, "control payload byte count differs");
    need(sha(bytes) === r.working_payload_sha256, "control payload SHA differs");
  } else {
    const c = r.container;
    need(sameKeys(c, ["contract", "container_sha256", "provenance", "base64"]) && c.contract === P.CONTRACT, "profile container descriptor differs");
    bytes = b64(c.base64, "container");
    containerSha = sha(bytes);
    need(c.container_sha256 === containerSha, "container SHA differs");
    const frame = await P.decodeFjnrgbProfile(new Uint8Array(bytes), c.provenance, { expectedSourceSha256: s.raf_sha256, expectedWasmSha256: wasmSha256 });
    need(same([frame.width, frame.height], s.expected_working_size), "decoded container size differs");
    need(sha(Buffer.from(frame.data.buffer, frame.data.byteOffset, frame.data.byteLength)) === r.working_payload_sha256, "container payload SHA differs");
    need(c.provenance.options_json === r.options_json && c.provenance.capture_dynamic_range === s.capture_dynamic_range
      && same(c.provenance.decoded_native_size, s.expected_native_size), "profile provenance differs");
    provenance = structuredClone(c.provenance); contract = P.CONTRACT;
  }
  return normalized(s, arm, r, { bytes, strips, strip_rows: [...st.rows], provenance, contract, container_sha256: containerSha,
    source_coefficients: { autoRgbg: [...sc.auto_rgbg], asShotCamMul: [...sc.as_shot_cam_mul] },
    returned_metadata: structuredClone(md), metadata_note: "Actual decoder readback returned by the P helper (pre_mul descriptive).",
    declared_options: options, native_size: [...r.native_size], width: r.working_size[0], height: r.working_size[1] });
}

/** Strictly validate one decoder result for (scene, arm) and return a normalised record. */
export async function normalizeDecoderResult(scene, arm, result, { wasmSha256 } = {}) {
  const s = validateScene(scene);
  need(ARMS.includes(arm), `unknown arm ${arm}`);
  need(typeof wasmSha256 === "string" && HEX.test(wasmSha256), "expected WASM SHA256 required");
  need(isObj(result), "decoder result must be an object");
  return s.kind === "W" ? normalizeW(s, arm, result, wasmSha256) : normalizeP(s, arm, result, wasmSha256);
}
