/**
 * Browser-only research decoder for the manual Kelvin 6600 fit-profile sentinels. UNSELECTED RESEARCH
 * CODE: not imported by the live app, not a v2 extension, and no old module/codec is modified.
 *
 * Sources: ONLY fit rows with WB mode 4080, 6600 K and raw fine-tune [20,-60] (host provides the frozen
 * source manifest entry). Two arms, nothing else:
 *   control    canonical profile options, but userMul = explicit as-shot [camR, camG, camB, camG]
 *              (camera WB off). Returns RAW working uint16 bytes, never a v2 container: its declared
 *              options differ from G's camera-WB decode, so the identity gate is native RGB SHA-256 +
 *              working payload SHA-256 against the existing G proof, never a container hash.
 *   treatment  canonical fixed profile options (userMul [620,302,455,302]); encoded with the new
 *              FJNRGB16 codec and its research sidecar. A repeat is a second call with arm "treatment";
 *              native, payload and container must then be exact.
 * No target data, model, exposure compensation, source-Auto path or preset fallback.
 *
 * Pure exports (Node-testable): canonicalFloorBoxDownsample, validateSourceScene, armOptions,
 * serializeOptions, stripRows, constants. The browser entry point is decodeFjnrgbProfileResearch and
 * imports /libraw/index.js, /profile-codec.mjs and /wb.mjs dynamically. The host must also serve
 * /fjlrg-v2.mjs, which /profile-codec.mjs imports for its public helpers.
 */

export const ARMS = Object.freeze(["control", "treatment"]);
export const MAX_WORKING_LONG_EDGE = 1536;
export const STRIP_FIRST_ROW = 64;
export const STRIP_ROW_STEP = 128;
export const SOURCE_WB_MODE = 4080;
export const SOURCE_KELVIN = 6600;
export const SOURCE_FINE_TUNE_RAW = Object.freeze([20, -60]);
export const PROFILE_LEVELS_RGBG = Object.freeze([620, 302, 455, 302]); // checked against the codec at runtime
const HEX = /^[0-9a-f]{64}$/;

function requireThat(ok, why) { if (!ok) throw new Error(`FJNRGB research decode: ${why}`); }
const posInt = (v, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(v) && v > 0 && v <= max;
const sameArray = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i]);

/** Exact transcription of the existing G/v2 research downsample: floor bins, per-bin mean, Math.round. */
export function canonicalFloorBoxDownsample(data, nativeWidth, nativeHeight, maxEdge = MAX_WORKING_LONG_EDGE) {
  requireThat(posInt(nativeWidth, 0xffffffff) && posInt(nativeHeight, 0xffffffff), "native dimensions must be positive safe integers");
  requireThat(posInt(maxEdge, MAX_WORKING_LONG_EDGE), "maxEdge must be an integer in 1..1536");
  requireThat(data instanceof Uint16Array && data.length === nativeWidth * nativeHeight * 3, "data must be Uint16Array of width*height*3");
  const scale = Math.min(1, maxEdge / Math.max(nativeWidth, nativeHeight));
  const width = Math.max(1, Math.round(nativeWidth * scale)), height = Math.max(1, Math.round(nativeHeight * scale));
  const out = new Uint16Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const y0 = Math.floor(y * nativeHeight / height), y1 = Math.max(y0 + 1, Math.floor((y + 1) * nativeHeight / height));
    for (let x = 0; x < width; x++) {
      const x0 = Math.floor(x * nativeWidth / width), x1 = Math.max(x0 + 1, Math.floor((x + 1) * nativeWidth / width));
      const sum = [0, 0, 0];
      for (let sy = y0; sy < y1; sy++) for (let sx = x0; sx < x1; sx++) {
        const i = (sy * nativeWidth + sx) * 3; sum[0] += data[i]; sum[1] += data[i + 1]; sum[2] += data[i + 2];
      }
      for (let c = 0; c < 3; c++) out[(y * width + x) * 3 + c] = Math.round(sum[c] / ((y1 - y0) * (x1 - x0)));
    }
  }
  return { width, height, data: out };
}

/** Rows saved as independent native strips: 64, 192, 320, ... (< height). */
export function stripRows(height) {
  requireThat(posInt(height, 0xffffffff), "height must be a positive safe integer");
  const rows = [];
  for (let y = STRIP_FIRST_ROW; y < height; y += STRIP_ROW_STEP) rows.push(y);
  return rows;
}

/** Same writer convention as fjlrg-v2.mjs serializeOptions (sorted top-level keys, compact). */
export function serializeOptions(options) {
  requireThat(options !== null && typeof options === "object" && !Array.isArray(options), "options must be an object");
  return JSON.stringify(Object.fromEntries(Object.keys(options).sort().map(k => [k, options[k]])));
}

/** Strict frozen-source contract. Returns a defensive copy. */
export function validateSourceScene(scene) {
  requireThat(scene !== null && typeof scene === "object" && !Array.isArray(scene), "scene must be an object");
  requireThat(typeof scene.name === "string" && /^DSCF\d{4}\.RAF$/.test(scene.name), "unsafe or missing source name");
  requireThat(scene.role === "fit", "fit sources only");
  requireThat(scene.source_wb_mode === SOURCE_WB_MODE, "source WB mode must be 4080 (Kelvin)");
  requireThat(scene.source_kelvin === SOURCE_KELVIN, "source Kelvin must be 6600");
  requireThat(sameArray(scene.source_wb_fine_tune_raw, SOURCE_FINE_TUNE_RAW), "source fine-tune must be [20,-60]");
  requireThat(typeof scene.raf_sha256 === "string" && HEX.test(scene.raf_sha256), "raf_sha256 must be 64 lowercase hex");
  const cam = scene.cam_mul, auto = scene.auto_rgbg, native = scene.g_native_size;
  requireThat(Array.isArray(cam) && cam.length === 4 && cam.slice(0, 3).every(v => posInt(v, 0xffffffff)) && cam[3] === 0, "cam_mul must be [R,G,B,0] positive integers");
  requireThat(Array.isArray(auto) && auto.length === 4 && auto.every(v => posInt(v, 0xffffffff)) && auto[1] === auto[3], "auto_rgbg must be [R,G,B,G] positive integers");
  requireThat(Array.isArray(native) && native.length === 2 && native.every(v => posInt(v, 0xffffffff)), "g_native_size must be [width,height] of the frozen G decode");
  requireThat([100, 200, 400].includes(scene.capture_dynamic_range), "capture_dynamic_range must be 100/200/400");
  requireThat(typeof scene.capture_dynamic_range_source === "string" && scene.capture_dynamic_range_source.trim() !== "", "capture_dynamic_range_source required");
  return structuredClone(scene);
}

function profileOptions() {
  return { halfSize: false, outputBps: 16, outputColor: 1, gamm: [1, 1, 0, 0, 0, 0], noAutoBright: true,
    adjustMaximumThr: 0, useCameraMatrix: 1, useCameraWb: false, useAutoWb: false, highlight: 0, userQual: 3,
    useFujiRotate: -1, fbddNoiserd: 0, userMul: [...PROFILE_LEVELS_RGBG] };
}

/** Fresh options for an arm. Control differs from treatment ONLY in userMul (explicit as-shot). */
export function armOptions(scene, arm) {
  const s = validateSourceScene(scene);
  requireThat(ARMS.includes(arm), `unknown arm ${String(arm)}`);
  const options = profileOptions();
  if (arm === "control") options.userMul = [s.cam_mul[0], s.cam_mul[1], s.cam_mul[2], s.cam_mul[1]];
  return options;
}

// ------------------------------------------------------------------ browser entry point
async function sha256Hex(bytes) {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map(v => v.toString(16).padStart(2, "0")).join("");
}
function base64(bytes) {
  return new Promise((ok, fail) => { const r = new FileReader(); r.onload = () => ok(String(r.result).split(",")[1]); r.onerror = fail; r.readAsDataURL(new Blob([bytes])); });
}
const u8 = arr => new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);

export async function decodeFjnrgbProfileResearch({ scene, arm, wasmSha256 }) {
  const s = validateSourceScene(scene);
  const options = armOptions(s, arm);
  requireThat(typeof wasmSha256 === "string" && HEX.test(wasmSha256), "expected WASM SHA256 required");
  const probe = new Uint8Array(new Uint16Array([0x0102]).buffer);
  requireThat(probe[0] === 0x02 && probe[1] === 0x01, "typed arrays are not little-endian");

  const { default: LibRaw } = await import("/libraw/index.js");
  const codec = await import("/profile-codec.mjs");
  const { readRafWbCoefficients } = await import("/wb.mjs");
  requireThat(sameArray([...codec.PROFILE_LEVELS_RGBG], [...PROFILE_LEVELS_RGBG]), "codec profile levels differ");
  requireThat(serializeOptions(codec.canonicalOptions()) === serializeOptions(profileOptions()), "codec canonical options differ");

  const response = await fetch(`/raw/${s.name}`);
  requireThat(response.ok, "missing approved RAF");
  const raf = new Uint8Array(await response.arrayBuffer());
  requireThat(await sha256Hex(raf) === s.raf_sha256, "whole fetched RAF SHA256 differs from frozen source");
  const source = await readRafWbCoefficients(new Blob([raf]));
  requireThat(sameArray(source.autoRgbg, s.auto_rgbg) && sameArray(source.asShotCamMul, s.cam_mul), "current RAF WB coefficients differ from frozen source");

  let decoder = new LibRaw();
  let native, nativeWidth, nativeHeight, readback, strips, rows;
  try {
    await decoder.open(raf, options);
    const image = await decoder.imageData(), meta = await decoder.metadata(true);
    readback = { camera_model: meta.camera_model == null ? null : String(meta.camera_model).trim(), fujiWB: meta.fuji?.WB_Preset ?? null,
      cam_mul: meta.color_data?.cam_mul == null ? null : Array.from(meta.color_data.cam_mul, Number),
      pre_mul: meta.color_data?.pre_mul == null ? null : Array.from(meta.color_data.pre_mul, Number) };
    requireThat(readback.camera_model === "X100VI" && readback.fujiWB === SOURCE_WB_MODE && sameArray(readback.cam_mul, s.cam_mul), "decoder metadata differs from frozen source");
    requireThat(image?.bits === 16 && image?.colors === 3 && image.data instanceof Uint16Array, "expected 16-bit 3-colour Uint16Array imageData");
    requireThat(image.data.buffer instanceof ArrayBuffer && !(typeof SharedArrayBuffer !== "undefined" && image.data.buffer instanceof SharedArrayBuffer), "imageData must be a plain main-thread ArrayBuffer");
    nativeWidth = image.width; nativeHeight = image.height;
    requireThat(sameArray([nativeWidth, nativeHeight], s.g_native_size), "native geometry differs from frozen G dimensions");
    requireThat(image.data.length === nativeWidth * nativeHeight * 3, "imageData length differs");
    native = image.data;
    rows = stripRows(nativeHeight);
    strips = new Uint16Array(rows.length * nativeWidth * 3);
    for (const [j, y] of rows.entries()) strips.set(native.subarray(y * nativeWidth * 3, (y + 1) * nativeWidth * 3), j * nativeWidth * 3);
  } finally { decoder.dispose(); decoder = null; } // free worker/WASM heap before hashing the native buffer
  requireThat(native.byteLength === nativeWidth * nativeHeight * 6, "native buffer changed after worker dispose");

  const working = canonicalFloorBoxDownsample(native, nativeWidth, nativeHeight, MAX_WORKING_LONG_EDGE);
  const optionsJson = serializeOptions(options);
  const result = { arm, scene_name: s.name, source_raf_sha256: s.raf_sha256, parameters_are_declared: true,
    declared_options: options, options_json: optionsJson, options_sha256: await sha256Hex(new TextEncoder().encode(optionsJson)),
    native_size: [nativeWidth, nativeHeight], working_size: [working.width, working.height],
    native_sha256: await sha256Hex(u8(native)), working_payload_sha256: await sha256Hex(u8(working.data)),
    source_coefficients: { auto_rgbg: [...source.autoRgbg], as_shot_cam_mul: [...source.asShotCamMul] }, decoder_readback: readback,
    strips: { rows, width: nativeWidth, channels: 3, layout: "uint16 LE, native rows interleaved RGB", base64: await base64(u8(strips)) },
    note: "Options are declared, not applied-parameter readback. Source Kelvin/fine-tune are bound by the whole-RAF SHA to frozen metadata; no separate readback is claimed." };
  if (arm === "control") {
    result.working_payload = { layout: "uint16 LE interleaved RGB, no container", base64: await base64(u8(working.data)) };
  } else {
    const container = codec.encodeFjnrgbProfile(working, { optionsSha256: result.options_sha256 });
    const provenance = await codec.buildProvenance(container, { sourceRafSha256: s.raf_sha256, wasmSha256,
      captureDynamicRange: s.capture_dynamic_range, captureDynamicRangeSource: s.capture_dynamic_range_source,
      decodedNativeSize: [nativeWidth, nativeHeight] });
    requireThat(provenance.options_json === optionsJson, "codec provenance options differ from declared options");
    result.container = { contract: codec.CONTRACT, container_sha256: provenance.container_sha256, provenance, base64: await base64(container) };
  }
  return result;
}
