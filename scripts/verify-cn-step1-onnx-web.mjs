#!/usr/bin/env node
/** Hash-bound F-only Classic Negative parity gate; no app integration.
 * Uses the existing numerical browser harness without changing its limits.
 * Provia plumbing evidence requires --allow-provia-smoke and is never a
 * Classic Negative accuracy or export-acceptance result.
 */
import { readFile, writeFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { basename, resolve, sep } from "node:path";
import { spawnSync } from "node:child_process";
import assert from "node:assert/strict";

const args = process.argv.slice(2);
const value = key => args.includes(key) ? args[args.indexOf(key) + 1] : undefined;
assert(value("--vector-dir") && value("--ort-dir"), "Pass --vector-dir and --ort-dir");
const dir = resolve(value("--vector-dir"));
assert(dir.startsWith(resolve("training/checkpoints") + sep), "Vectors must be in the checkpoint workspace");
// A failed rerun must not leave an older successful gate stamp behind.
await rm(resolve(dir, "cn-export-gate-report.json"), { force: true });
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const fileSha = async p => sha(await readFile(p));
const reportPath = resolve(dir, "native-report.json");
const reportBytes = await readFile(reportPath);
const report = JSON.parse(reportBytes);
const smoke = report.purpose === "provia-plumbing-smoke";
assert(report.passed === true && report.accuracy_acceptance === false);
assert(report.target_output_dr === 100 && report.models.length === 1);
assert.deepEqual(report.native_tolerances, { max_abs: 1e-4, mean_abs: 1e-5 });
assert(smoke ? args.includes("--allow-provia-smoke") && report.target_film_simulation === "Provia"
  : report.purpose === "classic-negative-step1-parity" && report.target_film_simulation === "Classic Negative"
    && report.contract_version === 2 && report.candidate_provenance);
const cohortPath = resolve("training/diagnostics/cuban-cohort-proposal.json");
const cohortBytes = await readFile(cohortPath);
assert.equal(sha(cohortBytes), report.cohort_sha256);
const fit = new Map(JSON.parse(cohortBytes).scenes.filter(s => s.role === "fit" && s.tier === "train").map(s => [s.name, s]));
assert.equal(fit.size, 24);
const expectedNames = new Set(["DSCF0523", "DSCF0964", "DSCF0514", "synthetic-73x111", "synthetic-111x73"]);
assert.equal(report.cases.length, 5);
assert.deepEqual(new Set(report.cases.map(c => c.name)), expectedNames);
const localFile = name => {
  assert.equal(typeof name, "string");
  assert.equal(basename(name), name);
  assert(!name.includes("..") && !name.includes("\\"));
  return resolve(dir, name);
};
const withinRepo = name => {
  const p = resolve(name);
  assert(p.startsWith(resolve(".") + sep), "Source must be inside this repo");
  return p;
};
const finiteWithin = (n, max) => typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= max;
const hashes = new Map();
const bind = async (p, expected) => { assert.equal(await fileSha(p), expected, `Hash mismatch: ${p}`); hashes.set(p, expected); };
await bind(resolve("training/model.py"), report.model_source_sha256);
await bind(resolve("training/export_cn_step1_onnx.py"), report.export_code_sha256);
await bind(resolve("training/export_round11_onnx.py"), report.normalization_helper_sha256);
assert.equal(report.contract_source, smoke ? "training/cuban_contract.py" : "training/cuban_contract_v2.py");
await bind(resolve(report.contract_source), report.contract_source_sha256);
assert.equal(report.original_round11_sha256, "1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c");
if (!smoke) {
  assert.equal(report.contract_sha256, "9824b0799737960363c95bb028332fdd0e5118e748fc0139375f1d0e230099a0");
  assert.equal(report.candidate_provenance.contract_sha256, report.contract_sha256);
  for (const field of ["run_manifest", "fold_manifest"]) {
    const entry = report.candidate_provenance[field];
    const p = withinRepo(entry.path);
    assert(p.startsWith(resolve("training") + sep));
    await bind(p, entry.sha256);
  }
}
await bind(resolve("training/checkpoints/r11-drcond-seed42-best.pt"), report.original_round11_sha256);
for (const c of report.cases) {
  assert.equal(c.capture_dr, 400); assert.equal(c.condition, 1);
  assert.equal(c.shape.length, 4); assert.equal(c.shape[0], 1); assert.equal(c.shape[1], 3);
  assert(c.shape.every(n => Number.isSafeInteger(n) && n > 0));
  if (c.source === "F-fit") {
    const row = fit.get(c.source_name);
    assert(row && row.shoot === c.shoot && c.name === c.source_name.replace(/\.RAF$/, ""));
    assert.equal(c.source_path, `calibration-input/${row.shoot}/browser-phase3-linear.fjlrg`);
    await bind(withinRepo(c.source_path), c.source_sha256);
  } else assert(c.source === "synthetic" && c.name.startsWith("synthetic-"));
  const input = localFile(c.input);
  await bind(input, c.input_sha256);
  assert.equal((await readFile(input)).byteLength, c.shape.reduce((a, b) => a * b) * 4);
}
for (const model of report.models) {
  assert([42, 43, 44].includes(model.seed));
  const checkpoint = withinRepo(model.checkpoint);
  assert(checkpoint.startsWith(resolve("training/checkpoints") + sep));
  await bind(checkpoint, model.checkpoint_sha256);
  await bind(localFile(model.model), model.model_sha256);
  assert.equal(model.native_cases.length, expectedNames.size);
  assert.deepEqual(new Set(model.native_cases.map(c => c.name)), expectedNames);
  for (const c of model.native_cases) {
    assert(c.passed === true && finiteWithin(c.max_abs, 1e-4) && finiteWithin(c.mean_abs, 1e-5));
    const p = localFile(c.expected);
    await bind(p, c.expected_sha256);
    const shape = report.cases.find(v => v.name === c.name).shape;
    assert.equal((await readFile(p)).byteLength, shape.reduce((a, b) => a * b) * 4);
  }
}
await bind(reportPath, sha(reportBytes));
const harnessPath = resolve("scripts/verify-round11-onnx-web.mjs");
const harnessHash = await fileSha(harnessPath);
await bind(harnessPath, harnessHash);
const run = spawnSync(process.execPath, ["scripts/verify-round11-onnx-web.mjs", "--vector-dir", dir,
  "--ort-dir", resolve(value("--ort-dir")), "--webgpu-layout", "NCHW"], { stdio: "inherit" });
if (run.error) throw run.error;
assert.equal(run.status, 0, "Browser parity process failed");
const browser = JSON.parse(await readFile(resolve(dir, "browser-report.json")));
assert.equal(browser.runtimeVersion, "1.31.1");
assert.equal(browser.webgpuLayout, "NCHW");
assert.deepEqual(browser.tolerances, { max_abs: 5e-4, mean_abs: 5e-5 });
assert(browser.passed === true && browser.providers.length === 2);
assert.deepEqual(new Set(browser.providers.map(p => p.provider)), new Set(["wasm", "webgpu"]));
for (const p of browser.providers) {
  assert(!p.error && Array.isArray(p.consoleErrors) && p.consoleErrors.length === 0);
  if (p.provider === "webgpu") assert(p.adapterInfo && p.adapterInfo.isFallbackAdapter === false);
  assert.equal(p.models.length, report.models.length);
  for (const m of p.models) {
    assert(report.models.some(x => x.seed === m.seed));
    assert.equal(m.cases.length, expectedNames.size);
    assert.deepEqual(new Set(m.cases.map(c => c.name)), expectedNames);
    for (const c of m.cases) assert(c.passed === true && finiteWithin(c.max_abs, 5e-4) && finiteWithin(c.mean_abs, 5e-5));
  }
}
for (const [p, hash] of hashes) assert.equal(await fileSha(p), hash, `Changed during browser run: ${p}`);
const result = { purpose: report.purpose, numerical_parity_passed: true, accuracy_acceptance: false,
  target_film_simulation: report.target_film_simulation, target_output_dr: 100,
  primary_candidate: !smoke && report.models[0].seed === 42,
  native_report_sha256: sha(reportBytes), models: report.models.map(m => ({ seed: m.seed, sha256: m.model_sha256 })),
  browser_harness_sha256: harnessHash,
  browser, scope: "F-only export parity; no recipe-accuracy claim, app integration or protected-tier processing" };
await writeFile(resolve(dir, "cn-export-gate-report.json"), `${JSON.stringify(result, null, 2)}\n`);
console.log(`${report.purpose}: native + WASM + WebGPU numerical parity passed; accuracy_acceptance=false`);
