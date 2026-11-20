import assert from "node:assert/strict";
import { decodeFjlrgV2, encodeFjlrgV2, parseStrictJson, sha256Hex, toInterleavedFloat32 } from "./lib/fjlrg-v2.mjs";
import { makeFixture, fixtureOptions } from "./lib/fjlrg-v2-fixtures.mjs";

let assertions = 0;
const verify = async f => decodeFjlrgV2(f.bytes, f.provenance, { expectedContract: f.contract });
for (const contract of ["v2-linear-asshot", "v2-linear-autowb"]) {
  const f = await makeFixture({ contract });
  const frame = await verify(f);
  assert.deepEqual(frame.data, f.pixels); assertions++;
  assert.deepEqual([...f.bytes.slice(64, 70)], [2, 1, 0, 0, 255, 255]); assertions++;
  const v = new DataView(f.bytes.buffer);
  for (const [off, value] of [[8, 2], [10, 64], [20, 3], [22, 16], [32, 1], [34, 1], [36, contract === "v2-linear-asshot" ? 1 : 2]]) {
    assert.equal(v.getUint16(off, true), value); assertions++;
  }
  assert.equal(toInterleavedFloat32(frame)[2], 1); assertions++;
  await assert.rejects(() => decodeFjlrgV2(f.bytes, f.provenance)); assertions++;
  const wrong = contract === "v2-linear-asshot" ? "v2-linear-autowb" : "v2-linear-asshot";
  await assert.rejects(() => decodeFjlrgV2(f.bytes, f.provenance, { expectedContract: wrong })); assertions++;
}

const base = await makeFixture();
for (const [offset, value, wide] of [[8, 1], [10, 32], [12, 0, true], [16, 2000, true], [20, 4], [22, 8],
  [24, 0, true], [28, 7, true], [32, 2], [34, 0], [36, 2], [38, 2], [40, 0, true], [52, 7, true], [56, 99]]) {
  const bytes = new Uint8Array(base.bytes), view = new DataView(bytes.buffer);
  if (wide) view.setUint32(offset, value, true); else view.setUint16(offset, value, true);
  const provenance = { ...base.provenance, container_sha256: await sha256Hex(bytes) };
  await assert.rejects(() => decodeFjlrgV2(bytes, provenance, { expectedContract: base.contract })); assertions++;
}
for (const mutate of [p => { p.extra = 1; }, p => { delete p.source_kelvin; }, p => { p.wasm_sha256 = "x"; },
  p => { p.wb_source.kind = "fuji_f00d"; }, p => { p.source_wb_mode = true; }, p => { p.source_wb_fine_tune_raw = [1, 2]; },
  p => { p.capture_dynamic_range = 800; }, p => { p.decoded_native_size = [100, 100]; }, p => { p.parameters_are_declared = false; }]) {
  const p = structuredClone(base.provenance); mutate(p);
  await assert.rejects(() => decodeFjlrgV2(base.bytes, p, { expectedContract: base.contract })); assertions++;
}
const badPixel = new Uint8Array(base.bytes); badPixel[65] ^= 1;
await assert.rejects(() => decodeFjlrgV2(badPixel, base.provenance, { expectedContract: base.contract })); assertions++;
await assert.rejects(() => decodeFjlrgV2(base.bytes, base.provenance, { expectedContract: base.contract, expectedWasmSha256: "ab".repeat(32) })); assertions++;
const originalJson = JSON.stringify(fixtureOptions(base.contract, base.provenance.wb_source.levels_rgbg));
for (const change of [o => { o.gamm = [1, 1]; }, o => { o.useCameraWb = 1; }, o => { o.userMul = [1, 1, 1, 1]; }, o => { o.noAutoBright = false; }]) {
  const o = JSON.parse(originalJson); change(o);
  const f = await makeFixture({ optionsJson: JSON.stringify(o) });
  await assert.rejects(() => verify(f)); assertions++;
}
for (const text of ['{"x":1,"x":2}', '{"x":1,"\\u0078":2}', '[1,]', 'NaN', '1e999', '{}junk']) {
  assert.throws(() => parseStrictJson(text)); assertions++;
}
const duplicate = originalJson.replace('"halfSize":false', '"halfSize":true,"halfSize":false');
const dupFixture = await makeFixture({ optionsJson: duplicate });
await assert.rejects(() => verify(dupFixture)); assertions++;
const decimal = originalJson.replace('[1,1,0,0,0,0]', '[1.0,1,0,0,0,0]');
assert.deepEqual((await verify(await makeFixture({ optionsJson: decimal }))).data, base.pixels); assertions++;
assert.deepEqual((await verify(await makeFixture({ optionsJson: ' \n' + originalJson + ' ' }))).data, base.pixels); assertions++;
const big = await makeFixture({ levels: [70000, 80000, 90000, 0] });
assert.equal((await verify(big)).provenance.wb_source.levels_rgbg[0], 70000); assertions++;
const snapshot = new Uint8Array(base.bytes), frame = await verify(base); frame.data[0] = 77;
assert.deepEqual(base.bytes, snapshot); assertions++;
assert.throws(() => encodeFjlrgV2({ width: 1, height: 1, data: new Uint16Array(3) }, { inputContract: "legacy", wbLevelsRgbg: [1, 1, 1, 0], optionsSha256: "ab".repeat(32) })); assertions++;
console.log(`JavaScript v2 codec: ${assertions} synthetic assertions PASS`);
