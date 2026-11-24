/** Synthetic-only tests for the DEVELOPMENT profile decoder's pure helpers. No fetch, browser, RAF or media. */
import test from "node:test";
import assert from "node:assert/strict";
import * as D from "./lib/decode-fjnrgb-profile-development.mjs";
import * as F from "./lib/decode-fjnrgb-profile-research.mjs";
import * as N from "./lib/fjnrgb-profile.mjs";

const devScene = () => ({ name: "DSCF0703.RAF", role: "development", source_wb_mode: 4080, source_kelvin: 6600,
  source_wb_fine_tune_raw: [20, -60], raf_sha256: "a".repeat(64), cam_mul: [636, 302, 414, 0],
  auto_rgbg: [461, 302, 995, 302], g_native_size: [5178, 7752], capture_dynamic_range: 400,
  capture_dynamic_range_source: "frozen development manifest captureDynamicRange" });
// SEPARATE fit fixture used only to compare pure functions with the unchanged fit module; never relabels a source.
const fitScene = () => ({ ...devScene(), name: "DSCF0464.RAF", role: "fit" });
function pseudo(n, seed = 1) { const a = new Uint16Array(n); let s = seed; for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) >>> 0; a[i] = s >>> 16; } return a; }

test("development role accepted; fit and protected roles refused", () => {
  assert.deepEqual(D.validateDevelopmentSourceScene(devScene()), devScene());
  for (const role of ["fit", "monitor", "finalHeldOut", "K", "T8", "dr100Eval", "", undefined, "Development"])
    assert.throws(() => D.validateDevelopmentSourceScene({ ...devScene(), role }), undefined, String(role));
  assert.throws(() => F.validateSourceScene(devScene()), undefined, "fit decoder must still refuse development rows");
});

test("other source guards identical to the fit decoder", () => {
  const bad = [{ source_wb_mode: 0 }, { source_kelvin: 5600 }, { source_wb_fine_tune_raw: [0, 0] }, { raf_sha256: "A".repeat(64) },
    { cam_mul: [636, 302, 414, 302] }, { auto_rgbg: [461, 302, 995, 300] }, { name: "../x.RAF" }, { g_native_size: [5178] },
    { capture_dynamic_range: 300 }, { capture_dynamic_range_source: " " }];
  for (const change of bad) {
    assert.throws(() => D.validateDevelopmentSourceScene({ ...devScene(), ...change }), undefined, JSON.stringify(change));
    assert.throws(() => F.validateSourceScene({ ...fitScene(), ...change }), undefined, JSON.stringify(change));
  }
});

test("arm options identical to the fit module for equal source parameters", () => {
  for (const arm of ["control", "treatment"]) {
    assert.deepEqual(D.armOptions(devScene(), arm), F.armOptions(fitScene(), arm));
    assert.equal(D.serializeOptions(D.armOptions(devScene(), arm)), F.serializeOptions(F.armOptions(fitScene(), arm)));
  }
  assert.deepEqual(D.armOptions(devScene(), "treatment"), N.canonicalOptions());
  assert.deepEqual(D.armOptions(devScene(), "control").userMul, [636, 302, 414, 302]);
  assert.throws(() => D.armOptions(devScene(), "treatment-repeat"));
  assert.throws(() => D.armOptions(fitScene(), "treatment"));
});

test("downsample, strips and constants bit-identical to the fit module", () => {
  for (const [W, H] of [[7, 5], [3001, 7], [1537, 1], [2049, 1543], [1, 4097]]) {
    const d = pseudo(W * H * 3, W + H);
    const a = D.canonicalFloorBoxDownsample(d, W, H), b = F.canonicalFloorBoxDownsample(d, W, H);
    assert.deepEqual([a.width, a.height], [b.width, b.height]);
    assert.deepEqual(a.data, b.data);
  }
  for (const h of [64, 65, 400, 7752]) assert.deepEqual(D.stripRows(h), F.stripRows(h));
  for (const k of ["MAX_WORKING_LONG_EDGE", "STRIP_FIRST_ROW", "STRIP_ROW_STEP", "SOURCE_WB_MODE", "SOURCE_KELVIN"]) assert.equal(D[k], F[k]);
  assert.deepEqual([...D.SOURCE_FINE_TUNE_RAW], [...F.SOURCE_FINE_TUNE_RAW]);
  assert.deepEqual([...D.PROFILE_LEVELS_RGBG], [...F.PROFILE_LEVELS_RGBG]);
  assert.deepEqual([...D.ARMS], [...F.ARMS]);
  assert.equal(D.ROLE, "development");
  assert.throws(() => D.canonicalFloorBoxDownsample(new Uint16Array(9), 3, 1, 1537));
});

test("browser entry refuses a fit row before any import or fetch", async () => {
  await assert.rejects(() => D.decodeFjnrgbProfileDevelopment({ scene: fitScene(), arm: "treatment", wasmSha256: "b".repeat(64) }), /development sources only/);
});
