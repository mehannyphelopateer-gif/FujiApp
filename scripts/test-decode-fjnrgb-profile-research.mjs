/** Synthetic-only unit tests for the research decoder's pure helpers. No fetch, browser, RAF or media. */
import test from "node:test";
import assert from "node:assert/strict";
import * as R from "./lib/decode-fjnrgb-profile-research.mjs";
import * as N from "./lib/fjnrgb-profile.mjs";
import * as V from "./lib/fjlrg-v2.mjs";

const scene = () => ({ name: "DSCF0464.RAF", role: "fit", source_wb_mode: 4080, source_kelvin: 6600,
  source_wb_fine_tune_raw: [20, -60], raf_sha256: "a".repeat(64), cam_mul: [636, 302, 414, 0],
  auto_rgbg: [461, 302, 995, 302], g_native_size: [6240, 4160], capture_dynamic_range: 400,
  capture_dynamic_range_source: "frozen fit manifest captureDynamicRange" });

/** Independent transcription of the loop in scripts/lib/decode-provia-v2-research.mjs (lines 19-32). */
function oldReference(data, W, H) {
  const scale = Math.min(1, 1536 / Math.max(W, H));
  const width = Math.max(1, Math.round(W * scale)), height = Math.max(1, Math.round(H * scale));
  const out = new Uint16Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const y0 = Math.floor(y * H / height), y1 = Math.max(y0 + 1, Math.floor((y + 1) * H / height));
    for (let x = 0; x < width; x++) {
      const x0 = Math.floor(x * W / width), x1 = Math.max(x0 + 1, Math.floor((x + 1) * W / width));
      const sum = [0, 0, 0];
      for (let sy = y0; sy < y1; sy++) for (let sx = x0; sx < x1; sx++) { const i = (sy * W + sx) * 3; sum[0] += data[i]; sum[1] += data[i + 1]; sum[2] += data[i + 2]; }
      for (let c = 0; c < 3; c++) out[(y * width + x) * 3 + c] = Math.round(sum[c] / ((y1 - y0) * (x1 - x0)));
    }
  }
  return { width, height, data: out };
}
function pseudo(n, seed = 1) { const a = new Uint16Array(n); let s = seed; for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) >>> 0; a[i] = s >>> 16; } return a; }
function transpose(data, W, H) { const t = new Uint16Array(data.length); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) t[(x * H + y) * 3 + c] = data[(y * W + x) * 3 + c]; return t; }

test("downsample: identity when the long edge already fits", () => {
  const d = pseudo(7 * 5 * 3);
  const r = R.canonicalFloorBoxDownsample(d, 7, 5);
  assert.equal(r.width, 7); assert.equal(r.height, 5); assert.deepEqual(r.data, d);
});

test("downsample: exact floor-bin mean with Math.round half-up", () => {
  const W = 3072, H = 2; // scale 0.5 -> 1536 x 1, each output averages a 2x2 bin
  const d = new Uint16Array(W * H * 3);
  for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) for (let c = 0; c < 3; c++) d[(y * W + x) * 3 + c] = x % 2 === 0 ? 1 : 2 + c;
  const r = R.canonicalFloorBoxDownsample(d, W, H);
  assert.deepEqual([r.width, r.height], [1536, 1]);
  assert.deepEqual([...r.data.slice(0, 3)], [2, 2, 3]); // (1+1+2+2)/4 = 1.5 -> 2, (1+1+3+3)/4 = 2, (1+1+4+4)/4 = 2.5 -> 3
});

test("downsample: odd sizes and both orientations match the old loop exactly", () => {
  for (const [W, H] of [[3001, 7], [1537, 1], [2049, 1543], [6240, 41], [1, 1], [1, 4097]]) {
    const d = pseudo(W * H * 3, W + H);
    const r = R.canonicalFloorBoxDownsample(d, W, H);
    const ref = oldReference(d, W, H);
    assert.deepEqual([r.width, r.height], [ref.width, ref.height]);
    assert.deepEqual(r.data, ref.data);
    assert(Math.max(r.width, r.height) <= 1536);
    const rt = R.canonicalFloorBoxDownsample(transpose(d, W, H), H, W);
    assert.deepEqual([rt.width, rt.height], [r.height, r.width]);
    assert.deepEqual(rt.data, transpose(r.data, r.width, r.height));
  }
});

test("downsample: refusals", () => {
  const ok = new Uint16Array(4 * 3 * 3);
  for (const args of [[new Uint8Array(36), 4, 3], [new Float32Array(36), 4, 3], [ok, 4, 4], [ok, 0, 3], [ok, 4.5, 3], [ok, "4", 3],
    [ok, 4, 3, 0], [ok, 4, 3, 1537], [ok, 4, 3, 1.5], [ok, 4, 3, "1536"], [ok, 4, 3, -1]])
    assert.throws(() => R.canonicalFloorBoxDownsample(...args));
});

test("strip rows start at 64 every 128 rows", () => {
  assert.deepEqual(R.stripRows(64), []);
  assert.deepEqual(R.stripRows(65), [64]);
  assert.deepEqual(R.stripRows(400), [64, 192, 320]);
  assert.throws(() => R.stripRows(0));
});

test("source contract accepts only fit Kelvin 6600 [20,-60] sources", () => {
  assert.deepEqual(R.validateSourceScene(scene()), scene());
  const bad = [
    { role: "development" }, { role: "monitor" }, { source_wb_mode: 0 }, { source_wb_mode: 770 }, { source_kelvin: 5600 },
    { source_kelvin: null }, { source_wb_fine_tune_raw: [0, 0] }, { source_wb_fine_tune_raw: [20, -60, 0] },
    { cam_mul: [636, 302, 414, 302] }, { cam_mul: [636, true, 414, 0] }, { cam_mul: [636, 0, 414, 0] },
    { auto_rgbg: [461, 302, 995, 300] }, { name: "../x.RAF" }, { name: "DSCF0464.raf" }, { raf_sha256: "A".repeat(64) },
    { g_native_size: [6240] }, { g_native_size: [6240, 0] }, { capture_dynamic_range: 300 }, { capture_dynamic_range_source: " " },
  ];
  for (const change of bad) assert.throws(() => R.validateSourceScene({ ...scene(), ...change }), undefined, JSON.stringify(change));
  assert.throws(() => R.validateSourceScene(null));
});

test("arm options: treatment equals the codec canonical options; control differs only in explicit as-shot userMul", () => {
  const t = R.armOptions(scene(), "treatment"), c = R.armOptions(scene(), "control");
  assert.deepEqual(t, N.canonicalOptions());
  assert.deepEqual(c.userMul, [636, 302, 414, 302]);
  assert.equal(c.useCameraWb, false);
  const { userMul: _a, ...tRest } = t, { userMul: _b, ...cRest } = c;
  assert.deepEqual(cRest, tRest);
  for (const arm of ["C", "D", "E", "treatment-repeat", "", undefined]) assert.throws(() => R.armOptions(scene(), arm));
  t.userMul[0] = 1; t.gamm[0] = 9;
  assert.deepEqual(R.armOptions(scene(), "treatment"), N.canonicalOptions());
  assert.throws(() => R.armOptions({ ...scene(), source_kelvin: 5600 }, "treatment"));
});

test("local constants and serializer match the codecs", () => {
  assert.deepEqual([...R.PROFILE_LEVELS_RGBG], [...N.PROFILE_LEVELS_RGBG]);
  assert.equal(R.SOURCE_WB_MODE, N.SOURCE_WB_MODE);
  assert.equal(R.SOURCE_KELVIN, N.SOURCE_KELVIN);
  assert.deepEqual([...R.SOURCE_FINE_TUNE_RAW], [...N.SOURCE_FINE_TUNE_RAW]);
  for (const arm of R.ARMS) assert.equal(R.serializeOptions(R.armOptions(scene(), arm)), V.serializeOptions(R.armOptions(scene(), arm)));
});
