/** Synthetic-only tests for neutral-wb-development-artifacts (real codecs, tiny frames). No files, browser or media. */
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import * as A from "./lib/neutral-wb-development-artifacts.mjs";
import * as V from "./lib/fjlrg-v2.mjs";
import * as P from "./lib/fjnrgb-profile.mjs";

const SRC = "a".repeat(64), WASM = "b".repeat(64), NATIVE_SHA = "e".repeat(64);
const W_ = 12, H_ = 260;
const sha = b => createHash("sha256").update(b).digest("hex");
const b64 = u8 => Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength).toString("base64");

function scene(kind = "W", over = {}) {
  const base = { kind, role: "development", name: "DSCF0001.RAF", raf_sha256: SRC, capture_dynamic_range: 400,
    cam_mul: kind === "W" ? [440, 302, 828, 0] : [636, 302, 414, 0], auto_rgbg: [461, 302, 995, 302],
    source_wb_mode: kind === "W" ? 0 : 4080, source_kelvin: kind === "W" ? null : 6600,
    source_wb_fine_tune_raw: kind === "W" ? [80, -100] : [20, -60], expected_native_size: [W_, H_], expected_working_size: [W_, H_] };
  return { ...base, ...over };
}
function pixels(seed) { const d = new Uint16Array(W_ * H_ * 3); for (let i = 0; i < d.length; i++) d[i] = (i * 37 + seed) % 65536; return d; }
const strips = () => new Uint16Array(2 * W_ * 3).fill(7);

async function wResult(arm, s = scene("W")) {
  const options = A.armOptions(s, arm), optionsJson = V.serializeOptions(options);
  const optionsSha = sha(Buffer.from(optionsJson));
  const data = pixels(arm === "C" ? 1 : 2);
  const common = { stripsBase64: b64(strips()), stripRows: [64, 192], native_sha256: NATIVE_SHA, working_payload_sha256: sha(Buffer.from(data.buffer)),
    width: W_, height: H_, native_size: [W_, H_], source_coefficients: { autoRgbg: [...s.auto_rgbg], asShotCamMul: [...s.cam_mul], records: {} },
    options_json: optionsJson, options_sha256: optionsSha };
  if (arm === "C") return { ...common, base64: b64(data), provenance: undefined };
  const container = V.encodeFjlrgV2({ width: W_, height: H_, data }, { inputContract: "v2-linear-autowb", wbLevelsRgbg: s.auto_rgbg, optionsSha256: optionsSha });
  const provenance = { schema: "fujiapp.raw-input.v2", input_contract: "v2-linear-autowb", container_sha256: sha(container), source_raf_sha256: SRC,
    wasm_sha256: WASM, decoder_package: "libraw-wasm", decoder_version: "1.6.0", libraw_build_version: "0.22.1", options_json: optionsJson,
    options_sha256: optionsSha, declared_options: options, wb_source: { kind: "fuji_f00d", levels_rgbg: [...s.auto_rgbg] }, source_wb_mode: 0,
    source_wb_fine_tune_raw: [...s.source_wb_fine_tune_raw], source_kelvin: null, capture_dynamic_range: 400,
    capture_dynamic_range_source: "synthetic", decoded_native_size: [W_, H_], parameters_are_declared: true };
  return { ...common, base64: b64(container), provenance };
}

async function pResult(arm, s = scene("P")) {
  const options = A.armOptions(s, arm), optionsJson = V.serializeOptions(options);
  const data = pixels(arm === "C" ? 3 : 4);
  const r = { arm: arm === "C" ? "control" : "treatment", role: "development", scene_name: s.name, source_raf_sha256: SRC, parameters_are_declared: true,
    declared_options: options, options_json: optionsJson, options_sha256: sha(Buffer.from(optionsJson)), native_size: [W_, H_], working_size: [W_, H_],
    native_sha256: NATIVE_SHA, working_payload_sha256: sha(Buffer.from(data.buffer)),
    source_coefficients: { auto_rgbg: [...s.auto_rgbg], as_shot_cam_mul: [...s.cam_mul] },
    decoder_readback: { camera_model: "X100VI", fujiWB: 4080, cam_mul: [...s.cam_mul], pre_mul: [1.5, 1, 2.1, 0] },
    strips: { rows: [64, 192], width: W_, channels: 3, layout: "uint16 LE, native rows interleaved RGB", base64: b64(strips()) }, note: "synthetic" };
  if (arm === "C") return { ...r, working_payload: { layout: "uint16 LE interleaved RGB, no container", base64: b64(data) } };
  const container = P.encodeFjnrgbProfile({ width: W_, height: H_, data }, { optionsSha256: r.options_sha256 });
  const provenance = await P.buildProvenance(container, { sourceRafSha256: SRC, wasmSha256: WASM, captureDynamicRange: 400,
    captureDynamicRangeSource: "synthetic", decodedNativeSize: [W_, H_] });
  return { ...r, container: { contract: P.CONTRACT, container_sha256: sha(container), provenance, base64: b64(container) } };
}

const norm = (s, arm, r) => A.normalizeDevelopmentDecoderResult(s, arm, r, { wasmSha256: WASM });
const rejects = (s, arm, r) => assert.rejects(() => norm(s, arm, r));

test("arm options: copied profile base, camera WB off, own green kept", () => {
  const w = scene("W"), p = scene("P", { cam_mul: [636, 325, 414, 0] });
  assert.deepEqual(A.armOptions(w, "C").userMul, [440, 302, 828, 302]);
  assert.deepEqual(A.armOptions(w, "N").userMul, [461, 302, 995, 302]);
  assert.deepEqual(A.armOptions(p, "C").userMul, [636, 325, 414, 325]);
  assert.deepEqual(A.armOptions(p, "R"), P.canonicalOptions());
  for (const s of [w, p]) for (const arm of A.ARMS) assert.equal(A.armOptions(s, arm).useCameraWb, false);
  const o = A.armOptions(w, "N"); o.userMul[0] = 1;
  assert.equal(A.armOptions(w, "N").userMul[0], 461);
  assert.throws(() => A.armOptions(w, "E"));
});

test("scene refusals", () => {
  const bad = [["W", { kind: "G" }], ["W", { role: "fit" }], ["W", { role: "monitor" }], ["W", { name: "../x.RAF" }],
    ["W", { raf_sha256: "A".repeat(64) }], ["W", { source_wb_mode: 4080 }], ["W", { source_kelvin: 6600 }],
    ["P", { source_kelvin: 5600 }], ["P", { source_wb_fine_tune_raw: [0, 0] }], ["P", { source_wb_mode: 0 }],
    ["W", { cam_mul: [440, 302, 828, 302] }], ["W", { auto_rgbg: [461, 302, 995, 300] }], ["W", { cam_mul: [440, true, 828, 0] }],
    ["W", { source_wb_fine_tune_raw: [10, 0] }], ["W", { capture_dynamic_range: 300 }], ["W", { expected_native_size: [W_] }],
    ["W", { expected_working_size: [W_, H_ + 1] }], ["W", { expected_native_size: [0, H_] }]];
  for (const [kind, over] of bad) assert.throws(() => A.validateDevelopmentScene(scene(kind, over)), undefined, JSON.stringify(over));
});

test("W C/N/R normalise; metadata not returned and never reconstructed", async () => {
  const s = scene("W");
  const c = await norm(s, "C", await wResult("C"));
  assert.equal(c.provenance, null); assert.equal(c.contract, null); assert.equal(c.container_sha256, null);
  assert.equal(c.returned_metadata, null); assert.match(c.metadata_note, /not returned/);
  assert.ok(Buffer.isBuffer(c.bytes) && Buffer.isBuffer(c.strips));
  assert.deepEqual(c.source_coefficients, { autoRgbg: s.auto_rgbg, asShotCamMul: s.cam_mul });
  for (const arm of ["N", "R"]) {
    const r = await wResult(arm), n = await norm(s, arm, r);
    assert.equal(n.contract, "v2-linear-autowb"); assert.equal(n.container_sha256, sha(n.bytes)); assert.equal(n.provenance.input_contract, "v2-linear-autowb");
    assert.equal(n.returned_metadata, null);
  }
  const noKey = await wResult("C"); delete noKey.provenance;
  await norm(s, "C", noKey);
});

test("P C/N/R normalise with actual returned readback", async () => {
  const s = scene("P");
  const c = await norm(s, "C", await pResult("C"));
  assert.equal(c.provenance, null); assert.equal(c.container_sha256, null);
  assert.deepEqual(c.returned_metadata, { camera_model: "X100VI", fujiWB: 4080, cam_mul: s.cam_mul, pre_mul: [1.5, 1, 2.1, 0] });
  const n = await norm(s, "N", await pResult("N"));
  assert.equal(n.contract, P.CONTRACT); assert.equal(n.container_sha256, sha(n.bytes)); assert.equal(n.provenance.profile_key, "4080/6600");
  assert.deepEqual(n.strip_rows, [64, 192]);
});

test("W tampering refused", async () => {
  const s = scene("W");
  const cases = [
    ["C", r => { r.provenance = {}; }], ["C", r => { r.working_payload_sha256 = "0".repeat(64); }],
    ["C", r => { r.base64 = b64(new Uint16Array(10)); }], ["C", r => { r.base64 = r.base64.slice(0, -4); }],
    ["C", r => { r.options_json = V.serializeOptions(A.armOptions(s, "N")); }], ["C", r => { r.options_sha256 = "f".repeat(64); }],
    ["C", r => { r.stripRows = [0, 128]; }], ["C", r => { r.stripsBase64 = b64(new Uint16Array(3)); }], ["C", r => { r.native_sha256 = "bad"; }],
    ["C", r => { r.width = W_ + 1; }], ["C", r => { r.native_size = [W_, H_ + 2]; }], ["C", r => { r.source_coefficients.autoRgbg = [1, 302, 1, 302]; }],
    ["C", r => { r.extra = 1; }], ["C", r => { delete r.stripRows; }], ["C", r => { r.base64 = "%%%%"; }],
    ["N", r => { r.provenance = undefined; }], ["N", r => { r.provenance.source_raf_sha256 = "c".repeat(64); }],
    ["N", r => { r.provenance.wasm_sha256 = "c".repeat(64); }], ["N", r => { r.working_payload_sha256 = "0".repeat(64); }],
    ["N", r => { r.provenance.capture_dynamic_range = 200; }], ["N", r => { r.provenance.container_sha256 = "0".repeat(64); }],
    ["N", r => { const b = Buffer.from(r.base64, "base64"); b[100] ^= 1; r.base64 = b.toString("base64"); }],
  ];
  for (const [arm, mutate] of cases) { const r = await wResult(arm); mutate(r); await rejects(s, arm, r); }
  await assert.rejects(async () => A.normalizeDevelopmentDecoderResult(s, "C", await wResult("C"), {}));
  await assert.rejects(async () => A.normalizeDevelopmentDecoderResult(s, "X", await wResult("C"), { wasmSha256: WASM }));
});

test("P tampering refused", async () => {
  const s = scene("P");
  const cases = [
    ["C", r => { r.arm = "treatment"; }], ["C", r => { r.role = "fit"; }], ["C", r => { delete r.role; }], ["C", r => { r.container = {}; }], ["C", r => { r.scene_name = "DSCF0002.RAF"; }],
    ["C", r => { r.parameters_are_declared = 1; }], ["C", r => { r.decoder_readback.camera_model = "X-T5"; }],
    ["C", r => { r.decoder_readback.fujiWB = 0; }], ["C", r => { r.decoder_readback.cam_mul = [636, 302, 414, 302]; }],
    ["C", r => { r.decoder_readback = null; }], ["C", r => { r.source_coefficients.as_shot_cam_mul = [1, 302, 1, 0]; }],
    ["C", r => { r.decoder_readback.pre_mul = []; }], ["C", r => { r.strips.layout = "float32"; }],
    ["C", r => { r.working_payload.layout = "float32"; }],
    ["C", r => { r.declared_options = A.armOptions(s, "N"); }], ["C", r => { r.strips.channels = 4; }], ["C", r => { r.strips.width = 11; }],
    ["C", r => { r.working_payload.base64 = b64(new Uint16Array(9)); }], ["C", r => { r.working_size = [W_, H_ - 1]; }],
    ["N", r => { r.container.container_sha256 = "0".repeat(64); }], ["N", r => { r.container.contract = "v2-linear-autowb"; }],
    ["N", r => { r.container.provenance.profile_key = "4080/5600"; }], ["N", r => { r.working_payload_sha256 = "0".repeat(64); }],
    ["N", r => { r.working_payload = { layout: "x", base64: "" }; }], ["R", r => { r.arm = "control"; }],
  ];
  for (const [arm, mutate] of cases) { const r = await pResult(arm); mutate(r); await rejects(s, arm, r); }
  await rejects(scene("W"), "N", await pResult("N"));   // P result under a W scene
  await rejects(s, "N", await wResult("N", scene("W"))); // W result under a P scene
});
