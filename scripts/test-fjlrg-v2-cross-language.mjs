/** Conformance on owned temporary synthetic files only, no camera media. */
import assert from "node:assert/strict";
import { mkdtemp, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { makeFixture, fixtureOptions } from "./lib/fjlrg-v2-fixtures.mjs";
import { decodeFjlrgV2, encodeFjlrgV2, parseStrictJson, sha256Hex, toInterleavedFloat32 } from "./lib/fjlrg-v2.mjs";
const root = await mkdtemp(join(tmpdir(), "fuji-v2-synthetic-"));
const cases = [];
try {
  await writeFile(join(root, "synthetic-marker.txt"), "FujiApp v2 synthetic conformance only\n");
  async function add(name, fixture, accepted, fullRange = false) {
    let ok;
    try { await decodeFjlrgV2(fixture.bytes, fixture.provenance, { expectedContract: fixture.contract }); ok = true; } catch { ok = false; }
    assert.equal(ok, accepted, `JS unexpected acceptance: ${name}`);
    await writeFile(join(root, name), fixture.bytes);
    await writeFile(join(root, name + ".json"), JSON.stringify(fixture.provenance));
    cases.push({ name, contract: fixture.contract, expected_accepted: accepted, full_range: fullRange });
    if (fullRange) {
      const frame = await decodeFjlrgV2(fixture.bytes, fixture.provenance, { expectedContract: fixture.contract });
      const f = toInterleavedFloat32(frame);
      await writeFile(join(root, name + ".js-float32"), new Uint8Array(f.buffer));
    }
  }
  await add("js-asshot.fjlrg", await makeFixture(), true);
  await add("js-autowb.fjlrg", await makeFixture({ contract: "v2-linear-autowb" }), true);
  await add("js-u32-levels.fjlrg", await makeFixture({ levels: [70000, 80000, 90000, 0] }), true);
  const text = JSON.stringify(fixtureOptions("v2-linear-asshot", [628, 302, 445, 0]));
  await add("js-decimal-options.fjlrg", await makeFixture({ optionsJson: text.replace("[1,1,0,0,0,0]", "[1.0,1,0,0,0,0]") }), true);
  await add("js-spaced-options.fjlrg", await makeFixture({ optionsJson: " \n" + text + "\t" }), true);
  const range = await makeFixture({ width: 256, height: 256 });
  for (let i = 0; i < 65536; i++) { range.pixels[3 * i] = i; range.pixels[3 * i + 1] = 65535 - i; range.pixels[3 * i + 2] = i ^ 0xaaaa; }
  range.bytes = encodeFjlrgV2({ width: 256, height: 256, data: range.pixels }, { inputContract: range.contract,
    wbLevelsRgbg: range.provenance.wb_source.levels_rgbg, optionsSha256: range.provenance.options_sha256 });
  range.provenance.container_sha256 = await sha256Hex(range.bytes);
  await add("js-full-range.fjlrg", range, true, true);
  for (const [index, mutate] of [p => { p.extra = 1; }, p => { p.source_wb_mode = true; }, p => { p.parameters_are_declared = 1; },
    p => { p.decoded_native_size = [100, 100]; }, p => { p.capture_dynamic_range = 300; }, p => { p.wb_source.levels_rgbg[1] = 0; }].entries()) {
    const f = await makeFixture(); mutate(f.provenance); await add(`js-invalid-sidecar-${index}.fjlrg`, f, false);
  }
  for (const [index, [off, value]] of [[8, 1], [10, 32], [20, 4], [32, 2], [34, 0], [38, 2]].entries()) {
    const f = await makeFixture(); new DataView(f.bytes.buffer).setUint16(off, value, true);
    f.provenance.container_sha256 = await sha256Hex(f.bytes); await add(`js-invalid-header-${index}.fjlrg`, f, false);
  }
  await add("js-duplicate-options.fjlrg", await makeFixture({ optionsJson: text.replace('"halfSize":false', '"halfSize":true,"halfSize":false') }), false);
  await writeFile(join(root, "cases.json"), JSON.stringify(cases));
  const output = execFileSync("python3", ["training/check_fjlrg_v2_conformance.py", root], { encoding: "utf8" });
  const py = parseStrictJson(await readFile(join(root, "python-result.json"), "utf8"));
  for (const row of cases.filter(c => c.expected_accepted)) assert.deepEqual(await readFile(join(root, row.name + ".python-reencoded")), await readFile(join(root, row.name)));
  for (const row of py.python_cases) {
    const bytes = new Uint8Array(await readFile(join(root, row.name)));
    const prov = parseStrictJson(await readFile(join(root, row.name + ".json"), "utf8"));
    const frame = await decodeFjlrgV2(bytes, prov, { expectedContract: row.contract });
    const rebuilt = encodeFjlrgV2(frame, { inputContract: row.contract, wbLevelsRgbg: prov.wb_source.levels_rgbg, optionsSha256: prov.options_sha256 });
    assert.deepEqual(rebuilt, bytes);
  }
  const code = ["scripts/lib/fjlrg-v2.mjs", "training/fjlrg_v2.py", "scripts/test-fjlrg-v2-cross-language.mjs", "training/check_fjlrg_v2_conformance.py"];
  const bindings = {};
  for (const p of code) bindings[p] = createHash("sha256").update(await readFile(p)).digest("hex");
  const report = { passed: true, scope: "Synthetic fixtures only; no media or model data", bindings,
    javascript_to_python: py.js_cases, python_to_javascript: py.python_cases,
    all_uint16_codes_float32_normalisation_exact: true, byte_reencoding_exact_both_directions: true };
  await writeFile("training/diagnostics/provia-v2-codec-conformance.json", JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log("V2 cross-language conformance PASS", output.trim());
} finally { await rm(root, { recursive: true, force: true }); }
