#!/usr/bin/env node
/**
 * Manual Kelvin 6600 WB primitive probe - research only, software decoder primitive.
 *
 * Question: does a fit-derived neutral Kelvin 6600 coefficient vector APPLY AS DECLARED in
 * LibRaw's unmixed native CFA output? Not Fuji colour accuracy, not sRGB/demosaic parity,
 * not runtime qualification, not a new input contract.
 *
 * Exactly 3 frozen fit sources (one per fit session). Four decodes per RAF, all with
 * six-element linear gamma, camera RGB (outputColor 0) and noInterpolation:
 *   C        base + useCameraWb:true, no userMul         (control: LibRaw as-shot)
 *   D        base + userMul: as_shot_rgbg                (control: declared as-shot)
 *   E        base + userMul: neutral_rgbg                (treatment)
 *   E-repeat identical to E                              (determinism)
 * C and D must agree exactly (native + samples), and so must E and E-repeat; any structural or
 * control failure stops immediately. Only structure statistics, metadata, full-native SHA-256
 * and a fixed native sample grid are saved (uint16 LE). No images, areas or FJLRG.
 * Python (training/qualify_manual_kelvin_wb.py) evaluates E vs D with the unchanged gates.
 * The source Auto estimate (0xf00d) is never used here.
 *
 * Usage: node scripts/probe-manual-kelvin-wb.mjs <committed-manifest-sha256>
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile, mkdir } from "node:fs/promises";
import { basename, resolve, relative, extname, sep } from "node:path";
import { chromium } from "playwright";

const root = process.cwd();
const MANIFEST_REL = "training/diagnostics/manual-kelvin-wb-primitive-manifest.json";
const OUT_REL = "training/checkpoints/manual-kelvin-wb-primitive";
const COMMITTED = [MANIFEST_REL, "scripts/probe-manual-kelvin-wb.mjs", "training/qualify_manual_kelvin_wb.py"];
const LIBRAW_DIR_REL = "node_modules/libraw-wasm/dist";
const LIBRAW_FILES = ["index.js", "worker.js", "libraw.js", "libraw.wasm"];
const ARMS = ["C", "D", "E", "E-repeat"];
const NEUTRAL_K6600 = [620, 302, 455]; // exact ratio family; common green may be rescaled by Codex
// Inherited exactly from the original source-Auto primitive manifest (provia-wb-primitive-v2).
const EXPECTED_BASE = {
  halfSize: false, outputBps: 16, outputColor: 0, gamm: [1, 1, 0, 0, 0, 0], noAutoBright: true,
  adjustMaximumThr: 0, useCameraMatrix: 1, useCameraWb: false, useAutoWb: false, highlight: 0,
  userQual: 3, useFujiRotate: -1, fbddNoiserd: 0, noInterpolation: true,
};

const args = process.argv.slice(2);
assert.equal(args.length, 1, "Pass exactly one argument: the committed manifest SHA256");
assert.match(args[0], /^[0-9a-f]{64}$/, "Manifest SHA256 must be 64 lowercase hex");

const sha = b => createHash("sha256").update(b).digest("hex");
const hashFile = async p => { const h = createHash("sha256"); for await (const b of createReadStream(p)) h.update(b); return h.digest("hex"); };
const isPosInt = v => Number.isSafeInteger(v) && v > 0;
const sameRatio = (a, b) => a[0] * b[1] === b[0] * a[1] && a[2] * b[1] === b[2] * a[1]; // exact integer cross-products

// ------------------------------------------------------------------ manifest + bindings
const manifestBytes = await readFile(resolve(root, MANIFEST_REL));
assert.equal(sha(manifestBytes), args[0], "Manifest drift");
const manifest = JSON.parse(manifestBytes);
assert(manifest.bindings && typeof manifest.bindings === "object", "Manifest bindings required");
for (const [p, h] of Object.entries(manifest.bindings)) assert.equal(await hashFile(resolve(root, p)), h, `Bound file drift: ${p}`);
for (const f of LIBRAW_FILES) assert(`${LIBRAW_DIR_REL}/${f}` in manifest.bindings, `LibRaw module not bound: ${f}`);
for (const p of COMMITTED)
  assert.equal(sha(execFileSync("git", ["show", `HEAD:${p}`], { cwd: root })), await hashFile(resolve(root, p)), `Not committed at HEAD: ${p}`);
assert.equal(typeof manifest.scope, "string", "Manifest scope required");
if ("output_dir" in manifest) assert.equal(manifest.output_dir, OUT_REL, "Output directory is fixed");

// ------------------------------------------------------------------ frozen option invariants
const base = manifest.base_options;
assert.deepEqual(base, EXPECTED_BASE, "Base options must equal the inherited primitive options exactly");
assert(!("userMul" in base), "userMul is per arm, never in base options");
const grid = manifest.sample_grid;
assert.deepEqual(grid, { offset: 15, stride: 31 }, "Frozen sample grid");

// ------------------------------------------------------------------ scene contract
const scenes = manifest.scenes;
assert(Array.isArray(scenes) && scenes.length === 3, "Exactly 3 scenes");
assert.equal(new Set(scenes.map(s => s.name)).size, 3, "Scene names must be unique");
assert.equal(new Set(scenes.map(s => s.session)).size, 3, "Scenes must come from 3 distinct sessions");
assert.equal(new Set(scenes.map(s => s.raf_sha256)).size, 3, "Source RAFs must be distinct");
const calibrationRoot = resolve(root, "calibration-input");
const allowed = new Map();
for (const s of scenes) {
  assert.match(s.name, /^DSCF\d{4}\.RAF$/, `Unsafe RAF name: ${s.name}`);
  assert.equal(s.name, basename(s.name));
  assert.match(s.shoot, /^Shoot \d+$/, `Unsafe shoot folder: ${s.shoot}`);
  assert.equal(s.role, "fit", `${s.name}: fit sources only`);
  assert(Number.isSafeInteger(s.session), `${s.name}: integer session required`);
  assert.equal(s.source_wb_mode, 4080, `${s.name}: Kelvin mode 4080 only`);
  assert.equal(s.source_kelvin, 6600, `${s.name}: 6600 K only`);
  assert.deepEqual(s.source_wb_fine_tune_raw, [20, -60], `${s.name}: frozen fine-tune only`);
  assert.deepEqual(s.shift, [1, -3], `${s.name}: shift must be fine-tune / 20`);
  assert.match(s.raf_sha256, /^[0-9a-f]{64}$/, `${s.name}: RAF SHA256 required`);
  const cam = s.cam_mul, asShot = s.as_shot_rgbg, neutral = s.neutral_rgbg, gain = s.expected_gain_RGB;
  assert(Array.isArray(cam) && cam.length === 4 && cam.slice(0, 3).every(isPosInt) && cam[3] === 0, `${s.name}: cam_mul [R,G,B,0]`);
  for (const [label, v] of [["as_shot_rgbg", asShot], ["neutral_rgbg", neutral]])
    assert(Array.isArray(v) && v.length === 4 && v.every(isPosInt) && v[1] === v[3], `${s.name}: ${label} must be positive ints [R,G,B,G]`);
  assert(sameRatio(asShot, cam), `${s.name}: as_shot_rgbg is not cam_mul's exact ratio vector`);
  assert(sameRatio(neutral, NEUTRAL_K6600), `${s.name}: neutral_rgbg is not the frozen Kelvin 6600 ratio vector`);
  assert.equal(asShot[1], neutral[1], `${s.name}: as-shot and neutral vectors must share one common green`);
  assert(Array.isArray(gain) && gain.length === 3 && gain.every(g => Number.isFinite(g) && g > 0), `${s.name}: expected_gain_RGB`);
  for (let c = 0; c < 3; c++)
    assert(Math.abs(gain[c] - neutral[c] / asShot[c]) <= 1e-12 * (neutral[c] / asShot[c]), `${s.name}: expected_gain_RGB[${c}] differs from neutral/as-shot`);
  assert.equal(gain[1], 1, `${s.name}: green gain must be exactly 1`);
  const path = resolve(calibrationRoot, s.shoot, s.name);
  assert(path.startsWith(calibrationRoot + sep), `${s.name}: path escapes calibration-input`);
  allowed.set(s.name, path);
}

const outDir = resolve(root, OUT_REL);
await mkdir(outDir); // refuses an existing directory: never overwrite evidence
const assets = resolve(root, LIBRAW_DIR_REL);

// ------------------------------------------------------------------ localhost-only server
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://127.0.0.1");
    if (url.pathname === "/") { res.writeHead(200, { "Content-Type": "text/html" }); res.end('<!doctype html><link rel="icon" href="data:,">'); return; }
    let path;
    if (url.pathname.startsWith("/libraw/")) {
      const n = decodeURIComponent(url.pathname.slice(8));
      assert.equal(n, basename(n));
      assert(LIBRAW_FILES.includes(n));
      path = resolve(assets, n);
    } else if (url.pathname.startsWith("/raw/")) {
      const n = decodeURIComponent(url.pathname.slice(5));
      assert.equal(n, basename(n));
      path = allowed.get(n);
    }
    assert(path, "Route is not allowlisted");
    const info = await stat(path);
    res.writeHead(200, { "Content-Length": info.size, "Content-Type": extname(path) === ".js" ? "text/javascript" : extname(path) === ".wasm" ? "application/wasm" : "application/octet-stream" });
    createReadStream(path).pipe(res);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));

const report = { manifest_sha256: args[0], scope: manifest.scope, scenes: [], completed: false,
  provenance_note: "Options are DECLARED, not internal applied-parameter readback. pre_mul is descriptive only. " +
    "Interpretation is limited to declared sensor-WB application on unmixed native CFA samples; " +
    "not Fuji image colour accuracy, sRGB/demosaic parity or runtime qualification." };
const progressPath = resolve(outDir, "decode-progress.json");
const saveProgress = () => writeFile(progressPath, JSON.stringify(report, null, 2) + "\n");

async function stop(row, scene, arm, reason) {
  row.stopped = { arm, reason };
  // Preserve source integrity evidence even on an early control/structure stop.
  try { row.raf_sha256_at_stop = await hashFile(allowed.get(scene.name)); }
  catch (error) { row.raf_hash_error_at_stop = String(error); }
  report.scenes.push(row);
  report.stopped = { scene: scene.name, arm, reason };
  await saveProgress();
  throw new Error(`STOP ${scene.name} ${arm}: ${reason}`);
}

function armOptions(scene, arm) {
  if (arm === "C") return { ...base, useCameraWb: true };
  if (arm === "D") return { ...base, useCameraWb: false, userMul: scene.as_shot_rgbg };
  return { ...base, useCameraWb: false, userMul: scene.neutral_rgbg }; // E and E-repeat: identical
}

// Runs inside the page. Structure, metadata, full-native SHA and samples are all computed
// from the same imageData before decoder.dispose().
async function decodeInPage({ name, options, grid }) {
  const { default: LibRaw } = await import("/libraw/index.js");
  const decoder = new LibRaw();
  try {
    const response = await fetch(`/raw/${name}`);
    if (!response.ok) throw new Error("Missing allowlisted RAF");
    await decoder.open(new Uint8Array(await response.arrayBuffer()), options);
    const image = await decoder.imageData();
    const shape = { width: image?.width, height: image?.height, colors: image?.colors, bits: image?.bits };
    if (!image || image.bits !== 16 || image.colors !== 3 || !(image.data instanceof Uint16Array))
      return { structure_error: "Expected 16-bit, 3-colour Uint16Array imageData", shape };
    const { width, height, data } = image;
    if (data.length !== width * height * 3) return { structure_error: "imageData length differs from width*height*3", shape };
    // Per CFA phase (row mod 6, col mod 6): single-active counts per channel, plus all-zero pixels.
    const phase = new Float64Array(36 * 4);
    let mixed = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const p = (y * width + x) * 3, k = ((y % 6) * 6 + (x % 6)) * 4;
        const nz = (data[p] !== 0) + (data[p + 1] !== 0) + (data[p + 2] !== 0);
        if (nz > 1) mixed++;
        else if (nz === 0) phase[k + 3]++;
        else phase[k + (data[p] !== 0 ? 0 : data[p + 1] !== 0 ? 1 : 2)]++;
      }
    }
    const digest = await crypto.subtle.digest("SHA-256", new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    const native_sha256 = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
    const samples = [];
    for (let y = grid.offset; y < height; y += grid.stride)
      for (let x = grid.offset; x < width; x += grid.stride) {
        const p = (y * width + x) * 3;
        samples.push(data[p], data[p + 1], data[p + 2]);
      }
    const bytes = new Uint8Array(samples.length * 2);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < samples.length; i++) view.setUint16(i * 2, samples[i], true);
    const samples_base64 = await new Promise((ok, fail) => {
      const r = new FileReader(); r.onload = () => ok(String(r.result).split(",")[1]); r.onerror = fail; r.readAsDataURL(new Blob([bytes]));
    });
    const metadata = await decoder.metadata(true);
    const toArray = v => (v == null ? null : Array.from(v, Number));
    return { shape, mixed_pixels: mixed, phase_counts: Array.from(phase), native_sha256, samples_base64,
      sample_count: samples.length / 3,
      returned_metadata: { cam_mul: toArray(metadata.color_data?.cam_mul), pre_mul: toArray(metadata.color_data?.pre_mul),
        camera_model: metadata.camera_model == null ? null : String(metadata.camera_model).trim(),
        fujiWB: metadata.fuji?.WB_Preset ?? null } };
  } finally { decoder.dispose(); }
}

function inferPattern(phaseCounts) {
  const pattern = [];
  for (let k = 0; k < 36; k++) {
    const active = [0, 1, 2].filter(c => phaseCounts[k * 4 + c] > 0);
    if (active.length !== 1) return { error: `CFA phase ${k} has ${active.length} active channels` };
    pattern.push(active[0]);
  }
  if (new Set(pattern).size !== 3) return { error: "CFA does not contain all RGB channels" };
  return { pattern };
}

const browser = await chromium.launch({ headless: true });
try {
  for (const scene of scenes) {
    const rafPath = allowed.get(scene.name);
    assert.equal(await hashFile(rafPath), scene.raf_sha256, `RAF changed before decode: ${scene.name}`);
    const row = { name: scene.name, shoot: scene.shoot, session: scene.session, arms: {} };
    for (const arm of ARMS) {
      const options = armOptions(scene, arm);
      const declared = { declared_options: options, declared_options_sha256: sha(JSON.stringify(options)) };
      const page = await browser.newPage();
      let result;
      try {
        await page.goto(`http://127.0.0.1:${server.address().port}/`);
        result = await page.evaluate(decodeInPage, { name: scene.name, options, grid });
      } finally { await page.close(); }

      if (result.structure_error) { row.arms[arm] = { ...declared, shape: result.shape }; await stop(row, scene, arm, result.structure_error); }
      if (result.mixed_pixels !== 0) { row.arms[arm] = { ...declared, shape: result.shape, mixed_pixels: result.mixed_pixels }; await stop(row, scene, arm, `Mixed pixels: ${result.mixed_pixels}`); }
      const inferred = inferPattern(result.phase_counts);
      const record = { shape: result.shape, mixed_pixels: 0, phase_counts: result.phase_counts, inferred_pattern: inferred.pattern ?? null,
        returned_metadata: result.returned_metadata, ...declared, native_sha256: result.native_sha256 };
      row.arms[arm] = record;
      if (inferred.error) await stop(row, scene, arm, inferred.error);
      const first = row.arms.C;
      if (arm !== "C" && (record.shape.width !== first.shape.width || record.shape.height !== first.shape.height))
        await stop(row, scene, arm, "Arm native dimensions differ");
      if (arm !== "C" && JSON.stringify(record.inferred_pattern) !== JSON.stringify(first.inferred_pattern))
        await stop(row, scene, arm, "Arm CFA patterns differ");
      const md = result.returned_metadata;
      if (md.camera_model !== "X100VI") await stop(row, scene, arm, `Camera model ${md.camera_model}`);
      if (md.fujiWB !== 4080) await stop(row, scene, arm, `Fuji WB preset ${md.fujiWB}`);
      if (JSON.stringify(md.cam_mul) !== JSON.stringify(scene.cam_mul)) await stop(row, scene, arm, "Returned cam_mul differs from frozen scene cam_mul");

      // Only the fixed-grid samples are persisted.
      const samples = Buffer.from(result.samples_base64, "base64");
      const dest = resolve(outDir, `${scene.name}-${arm}.samples.u16le`);
      await writeFile(dest, samples, { flag: "wx" });
      Object.assign(record, { sample_sites: result.sample_count, samples_file: relative(root, dest), samples_sha256: sha(samples),
        samples_layout: "uint16 little-endian, row-major native grid, 3 channels per site" });

      if (arm === "D" && (record.native_sha256 !== first.native_sha256 || record.samples_sha256 !== first.samples_sha256))
        await stop(row, scene, arm, "Control C (camera WB) and D (declared as-shot userMul) differ");
      if (arm === "E-repeat" && (record.native_sha256 !== row.arms.E.native_sha256 || record.samples_sha256 !== row.arms.E.samples_sha256))
        await stop(row, scene, arm, "E and E-repeat differ");
      console.log(`${report.scenes.length + 1}/3 ${scene.name} ${arm} ${record.shape.width}x${record.shape.height}`);
    }
    assert.equal(await hashFile(rafPath), scene.raf_sha256, `RAF changed during decode: ${scene.name}`);
    report.scenes.push(row);
    await saveProgress();
  }
  report.completed = true;
  await writeFile(resolve(outDir, "decode-complete.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
} finally {
  await browser.close();
  await new Promise(r => server.close(r));
}
