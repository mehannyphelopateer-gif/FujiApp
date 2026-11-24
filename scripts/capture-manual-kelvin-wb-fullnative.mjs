#!/usr/bin/env node
/**
 * Full-native capture for the manual Kelvin 6600 WB primitive follow-up - research only.
 *
 * Re-decodes ONLY arms D (declared as-shot userMul) and E (fixed fit-derived neutral Kelvin 6600
 * profile) of the EXACT same 3 fit sources as the frozen sparse primitive, with options identical
 * to that run. Every native buffer must hash to the ORIGINAL D/E full-native SHA-256 before it is
 * uploaded, so the original C/D and E/E-repeat controls cover these exact bytes. The full native
 * uint16 RGB arrays (unmixed CFA, camera RGB, linear) are streamed over a same-origin localhost
 * POST to six pre-allowlisted ignored diagnostic files. No rendered photos, FJLRG, targets, models,
 * development or protected data. Python (training/qualify_manual_kelvin_wb_fullnative.py) evaluates.
 *
 * A PASS would only qualify declared sensor-WB application on these 3 sources - not the semantic
 * correctness of the neutral target, full RGB matrix/demosaic/parity, a runtime contract or a model.
 * The original sparse NOT_QUALIFIED verdict stays as recorded.
 *
 * Usage: node scripts/capture-manual-kelvin-wb-fullnative.mjs <committed-manifest-sha256>
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, createWriteStream, constants } from "node:fs";
import { readFile, stat, writeFile, mkdir, copyFile, unlink, access } from "node:fs/promises";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { basename, resolve, relative, extname, sep } from "node:path";
import { chromium } from "playwright";

const root = process.cwd();
const MANIFEST_REL = "training/diagnostics/manual-kelvin-wb-fullnative-manifest.json";
const OUT_REL = "training/checkpoints/manual-kelvin-wb-fullnative";
const OLD_MANIFEST_REL = "training/diagnostics/manual-kelvin-wb-primitive-manifest.json";
const OLD_REPORT_REL = "training/diagnostics/manual-kelvin-wb-primitive-report.json";
const OLD_DECODE_REL = "training/checkpoints/manual-kelvin-wb-primitive/decode-complete.json";
const PY_REL = "training/qualify_manual_kelvin_wb_fullnative.py";
const OWN_REL = "scripts/capture-manual-kelvin-wb-fullnative.mjs";
const COMMITTED = [MANIFEST_REL, OWN_REL, PY_REL];
const LIBRAW_DIR_REL = "node_modules/libraw-wasm/dist";
const LIBRAW_FILES = ["index.js", "worker.js", "libraw.js", "libraw.wasm"];
const ARMS = ["D", "E"];
const DECODE_TIMEOUT_MS = 10 * 60 * 1000;
const SCENE_FIELDS = ["name", "shoot", "role", "session", "source_wb_mode", "source_kelvin", "source_wb_fine_tune_raw",
  "raf_sha256", "cam_mul", "as_shot_rgbg", "neutral_rgbg", "expected_gain_RGB"];

const args = process.argv.slice(2);
assert.equal(args.length, 1, "Pass exactly one argument: the committed manifest SHA256");
assert.match(args[0], /^[0-9a-f]{64}$/, "Manifest SHA256 must be 64 lowercase hex");

const sha = b => createHash("sha256").update(b).digest("hex");
const hashFile = async p => { const h = createHash("sha256"); for await (const b of createReadStream(p)) h.update(b); return h.digest("hex"); };
const isHex64 = v => typeof v === "string" && /^[0-9a-f]{64}$/.test(v);
const readJson = async rel => JSON.parse(await readFile(resolve(root, rel)));
const exists = async p => { try { await access(p); return true; } catch { return false; } };

// ------------------------------------------------------------------ manifest, bindings, commits
const manifestBytes = await readFile(resolve(root, MANIFEST_REL));
assert.equal(sha(manifestBytes), args[0], "Manifest drift");
const manifest = JSON.parse(manifestBytes);
assert(manifest.bindings && typeof manifest.bindings === "object", "Manifest bindings required");
for (const [p, h] of Object.entries(manifest.bindings)) assert.equal(await hashFile(resolve(root, p)), h, `Bound file drift: ${p}`);
for (const p of [OLD_MANIFEST_REL, OLD_REPORT_REL, OLD_DECODE_REL, PY_REL, ...LIBRAW_FILES.map(f => `${LIBRAW_DIR_REL}/${f}`)])
  assert(p in manifest.bindings, `Required binding missing: ${p}`);
for (const p of COMMITTED)
  assert.equal(sha(execFileSync("git", ["show", `HEAD:${p}`], { cwd: root })), await hashFile(resolve(root, p)), `Not committed at HEAD: ${p}`);
assert.equal(typeof manifest.scope, "string", "Manifest scope required");
if ("output_dir" in manifest) assert.equal(manifest.output_dir, OUT_REL, "Output directory is fixed");

// ------------------------------------------------------------------ inherited options and scenes
const oldManifest = await readJson(OLD_MANIFEST_REL);
const oldDecode = await readJson(OLD_DECODE_REL);
assert.equal(oldDecode.completed, true, "Original sparse decode must be complete");
assert.equal(oldDecode.manifest_sha256, sha(await readFile(resolve(root, OLD_MANIFEST_REL))), "Original decode belongs to another manifest");

const base = manifest.base_options;
assert.deepEqual(base, oldManifest.base_options, "Base options must equal the original primitive's exactly");
assert.deepEqual(base.gamm, [1, 1, 0, 0, 0, 0], "Six-element linear gamma required");
assert.equal(base.outputColor, 0, "Camera RGB required");
assert.equal(base.noInterpolation, true, "noInterpolation required");
assert.equal(base.noAutoBright, true, "noAutoBright required");
assert.equal(base.outputBps, 16, "16-bit output required");
assert.equal(base.halfSize, false, "Full size required");
assert.equal(base.useCameraWb, false, "Camera WB off in base");
assert.equal(base.useAutoWb, false, "Auto WB off");
assert.equal(base.highlight, 0, "Highlight clip mode required");
assert(!("userMul" in base), "userMul is per arm only");

const scenes = manifest.scenes;
assert(Array.isArray(scenes) && scenes.length === 3, "Exactly 3 scenes");
assert.deepEqual(scenes.map(s => s.name), oldManifest.scenes.map(s => s.name), "Scenes must be the exact original 3, same order");
assert.equal(new Set(scenes.map(s => s.raf_sha256)).size, 3, "Source RAFs must be distinct");
assert.equal(new Set(scenes.map(s => s.session)).size, 3, "Sessions must be distinct");
const oldDecodeByName = new Map(oldDecode.scenes.map(r => [r.name, r]));
const calibrationRoot = resolve(root, "calibration-input");
const allowedRaw = new Map();
const uploads = new Map(); // filename -> { scene, arm, path, partial, expectedBytes, expectedHash, state }
const outDir = resolve(root, OUT_REL);

function armOptions(scene, arm) {
  return { ...base, useCameraWb: false, userMul: arm === "D" ? scene.as_shot_rgbg : scene.neutral_rgbg };
}

scenes.forEach((s, i) => {
  const old = oldManifest.scenes[i];
  for (const k of SCENE_FIELDS) assert.deepEqual(s[k], old[k], `${s.name}: ${k} differs from original primitive`);
  assert.match(s.name, /^DSCF\d{4}\.RAF$/, `Unsafe RAF name: ${s.name}`);
  assert.match(s.shoot, /^Shoot \d+$/, `Unsafe shoot folder: ${s.shoot}`);
  assert.equal(s.role, "fit");
  assert.equal(s.source_wb_mode, 4080);
  assert.equal(s.source_kelvin, 6600);
  assert.deepEqual(s.source_wb_fine_tune_raw, [20, -60]);
  assert.match(s.raf_sha256, /^[0-9a-f]{64}$/);
  const shape = s.native_shape;
  assert(shape && Number.isSafeInteger(shape.width) && Number.isSafeInteger(shape.height) && shape.width > 0 && shape.height > 0
    && shape.colors === 3 && shape.bits === 16 && Object.keys(shape).length === 4, `${s.name}: native_shape {width,height,colors:3,bits:16}`);
  const oldRow = oldDecodeByName.get(s.name);
  assert(oldRow && !oldRow.stopped, `${s.name}: original decode row missing or stopped`);
  for (const arm of ARMS) {
    const oldArm = oldRow.arms?.[arm];
    assert(oldArm, `${s.name}: original arm ${arm} missing`);
    assert(isHex64(s.original_native_sha256?.[arm]), `${s.name}: original_native_sha256.${arm}`);
    assert.equal(s.original_native_sha256[arm], oldArm.native_sha256, `${s.name}: ${arm} hash differs from original decode`);
    assert.equal(oldArm.shape.width, shape.width, `${s.name}: ${arm} width differs from original`);
    assert.equal(oldArm.shape.height, shape.height, `${s.name}: ${arm} height differs from original`);
    assert.equal(sha(JSON.stringify(armOptions(s, arm))), oldArm.declared_options_sha256, `${s.name}: ${arm} options differ from original`);
    const filename = `${s.name}-${arm}.native.u16le`;
    const path = resolve(outDir, filename);
    uploads.set(filename, { scene: s.name, arm, path, partial: `${path}.partial`, expectedBytes: shape.width * shape.height * 3 * 2,
      expectedHash: s.original_native_sha256[arm], state: "pending" });
  }
  const rawPath = resolve(calibrationRoot, s.shoot, s.name);
  assert(rawPath.startsWith(calibrationRoot + sep), `${s.name}: path escapes calibration-input`);
  allowedRaw.set(s.name, rawPath);
});
assert.equal(uploads.size, 6, "Exactly six allowlisted uploads");

await mkdir(outDir); // refuses an existing directory: never overwrite evidence
const assets = resolve(root, LIBRAW_DIR_REL);

// ------------------------------------------------------------------ streaming upload (serial)
let uploadBusy = false;
let uploadFailure = null;

function sizeAndHash(expectedBytes) {
  const hash = createHash("sha256");
  let bytes = 0;
  const t = new Transform({
    transform(chunk, _enc, done) {
      bytes += chunk.length;
      if (bytes > expectedBytes) return done(new Error(`Upload exceeds ${expectedBytes} bytes`));
      hash.update(chunk);
      done(null, chunk);
    },
  });
  return { t, result: () => ({ bytes, sha256: hash.digest("hex") }) };
}

async function handleUpload(req, res, filename) {
  const entry = uploads.get(filename);
  const reply = (code, body) => { res.writeHead(code, { "Content-Type": "application/json" }); res.end(JSON.stringify(body)); };
  if (!entry) { req.resume(); return reply(404, { ok: false, error: "Upload not allowlisted" }); }
  if (uploadBusy) { req.resume(); return reply(409, { ok: false, error: "Another upload is in progress" }); }
  if (entry.state !== "pending") { req.resume(); return reply(409, { ok: false, error: `Upload already ${entry.state}` }); }
  uploadBusy = true;
  entry.state = "receiving";
  try {
    assert(!req.headers["transfer-encoding"], "Chunked transfer refused");
    assert.equal(req.headers["content-length"], String(entry.expectedBytes), "Content-Length differs from expected native size");
    assert(!(await exists(entry.path)), "Final file already exists");
    const counter = sizeAndHash(entry.expectedBytes);
    await pipeline(req, counter.t, createWriteStream(entry.partial, { flags: "wx" }));
    const got = counter.result();
    assert.equal(got.bytes, entry.expectedBytes, "Uploaded byte count differs");
    assert.equal(got.sha256, entry.expectedHash, "Uploaded native SHA256 differs from original");
    // exFAT cannot hard-link. Publish via an exclusive copy, then verify it
    // before recording completion. An interrupted copy is never certified.
    await copyFile(entry.partial, entry.path, constants.COPYFILE_EXCL);
    assert.equal((await stat(entry.path)).size, entry.expectedBytes, "Published file is short");
    assert.equal(await hashFile(entry.path), entry.expectedHash, "Published file hash differs");
    await unlink(entry.partial);
    entry.state = "done";
    entry.received = got;
    reply(200, { ok: true, ...got });
  } catch (error) {
    entry.state = "failed";
    entry.error = String(error?.message ?? error);
    uploadFailure ??= `${filename}: ${entry.error}`;
    req.resume(); // drain; the bad .partial (if any) is kept as evidence
    if (!res.headersSent) reply(500, { ok: false, error: entry.error });
  } finally { uploadBusy = false; }
}

// ------------------------------------------------------------------ localhost-only server
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://127.0.0.1");
    if (req.method === "POST" && url.pathname.startsWith("/upload/")) {
      const n = decodeURIComponent(url.pathname.slice(8));
      if (n !== basename(n)) { req.resume(); res.writeHead(404); res.end(); return; }
      return await handleUpload(req, res, n);
    }
    assert.equal(req.method, "GET");
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
      path = allowedRaw.get(n);
    }
    assert(path, "Route is not allowlisted");
    const info = await stat(path); // stat follows symlinks; the file is streamed, never copied
    res.writeHead(200, { "Content-Length": info.size, "Content-Type": extname(path) === ".js" ? "text/javascript" : extname(path) === ".wasm" ? "application/wasm" : "application/octet-stream" });
    createReadStream(path).pipe(res);
  } catch { if (!res.headersSent) { res.writeHead(404); res.end(); } }
});
server.requestTimeout = DECODE_TIMEOUT_MS;
await new Promise(r => server.listen(0, "127.0.0.1", r));

// ------------------------------------------------------------------ in-page decode + upload
async function decodeAndUpload({ name, options, expected, filename }) {
  const probe = new Uint8Array(new Uint16Array([0x0102]).buffer);
  if (probe[0] !== 0x02 || probe[1] !== 0x01) return { error: "Browser typed arrays are not little-endian" };
  const { default: LibRaw } = await import("/libraw/index.js");
  const decoder = new LibRaw();
  try {
    const response = await fetch(`/raw/${name}`);
    if (!response.ok) throw new Error("Missing allowlisted RAF");
    await decoder.open(new Uint8Array(await response.arrayBuffer()), options);
    const image = await decoder.imageData();
    const shape = { width: image?.width, height: image?.height, colors: image?.colors, bits: image?.bits };
    if (!image || image.bits !== 16 || image.colors !== 3 || !(image.data instanceof Uint16Array))
      return { error: "Expected 16-bit, 3-colour Uint16Array imageData", shape };
    if (image.width !== expected.width || image.height !== expected.height)
      return { error: "Native dimensions differ from frozen shape", shape };
    if (image.data.length !== image.width * image.height * 3) return { error: "imageData length differs", shape };
    const metadata = await decoder.metadata(true);
    const toArray = v => (v == null ? null : Array.from(v, Number));
    const returned_metadata = { cam_mul: toArray(metadata.color_data?.cam_mul), pre_mul: toArray(metadata.color_data?.pre_mul),
      camera_model: metadata.camera_model == null ? null : String(metadata.camera_model).trim(), fujiWB: metadata.fuji?.WB_Preset ?? null };
    const bytes = new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const native_sha256 = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
    if (native_sha256 !== expected.hash) return { error: "Native SHA256 differs from original decode", shape, native_sha256, returned_metadata };
    if (returned_metadata.camera_model !== "X100VI" || returned_metadata.fujiWB !== 4080
        || JSON.stringify(returned_metadata.cam_mul) !== JSON.stringify(expected.cam_mul))
      return { error: "Returned source metadata differs", shape, native_sha256, returned_metadata };
    const upload = await fetch(`/upload/${encodeURIComponent(filename)}`, { method: "POST", body: bytes,
      headers: { "Content-Type": "application/octet-stream" } });
    let body = null;
    try { body = await upload.json(); } catch { body = null; }
    if (!upload.ok || !body?.ok) return { error: `Upload failed: HTTP ${upload.status} ${body?.error ?? ""}`, shape, native_sha256, returned_metadata };
    return { shape, native_sha256, returned_metadata, upload: body };
  } finally { decoder.dispose(); }
}

function withTimeout(promise, ms, label) {
  let timer;
  return Promise.race([promise, new Promise((_, fail) => { timer = setTimeout(() => fail(new Error(`${label} timed out after ${ms} ms`)), ms); })])
    .finally(() => clearTimeout(timer));
}

const report = { manifest_sha256: args[0], scope: manifest.scope, scenes: [], completed: false,
  provenance_note: "Options are DECLARED (identical to the original sparse primitive), not internal applied-parameter readback; " +
    "pre_mul is descriptive only. Each native buffer equals the original D/E full-native SHA256, so original C/D and E/E-repeat " +
    "controls cover these exact bytes. Not Fuji colour accuracy, demosaic/matrix parity, runtime or model qualification." };
const progressPath = resolve(outDir, "decode-progress.json");
const saveProgress = () => writeFile(progressPath, JSON.stringify(report, null, 2) + "\n");

async function stop(row, sceneName, arm, reason) {
  row.stopped = { arm, reason };
  try { row.raf_sha256_at_stop = await hashFile(allowedRaw.get(sceneName)); }
  catch (error) { row.raf_hash_error_at_stop = String(error); }
  if (!report.scenes.includes(row)) report.scenes.push(row);
  report.stopped = { scene: sceneName, arm, reason };
  await saveProgress();
  throw new Error(`STOP ${sceneName} ${arm}: ${reason}`);
}

const browser = await chromium.launch({ headless: true });
try {
  for (const scene of scenes) {
    const rafPath = allowedRaw.get(scene.name);
    assert.equal(await hashFile(rafPath), scene.raf_sha256, `RAF changed before decode: ${scene.name}`);
    const row = { name: scene.name, arms: {} };
    report.scenes.push(row);
    for (const arm of ARMS) {
      const options = armOptions(scene, arm);
      const filename = `${scene.name}-${arm}.native.u16le`;
      const entry = uploads.get(filename);
      const declared = { declared_options: options, declared_options_sha256: sha(JSON.stringify(options)) };
      let result;
      const page = await browser.newPage();
      try {
        page.setDefaultTimeout(DECODE_TIMEOUT_MS);
        await page.goto(`http://127.0.0.1:${server.address().port}/`);
        result = await withTimeout(page.evaluate(decodeAndUpload, { name: scene.name, options, filename,
          expected: { width: scene.native_shape.width, height: scene.native_shape.height, hash: scene.original_native_sha256[arm], cam_mul: scene.cam_mul } }),
          DECODE_TIMEOUT_MS, `${scene.name} ${arm}`);
      } catch (error) {
        row.arms[arm] = { ...declared };
        await stop(row, scene.name, arm, `Decode/upload error: ${String(error?.message ?? error)}`);
      } finally { await page.close(); }

      row.arms[arm] = { ...declared, shape: result.shape ?? null, native_sha256: result.native_sha256 ?? null,
        returned_metadata: result.returned_metadata ?? null };
      if (result.error) await stop(row, scene.name, arm, result.error);
      if (uploadFailure) await stop(row, scene.name, arm, `Upload stream failure: ${uploadFailure}`);
      // Independent Node-side verification: never trust the page's report alone.
      if (entry.state !== "done") await stop(row, scene.name, arm, `Upload state ${entry.state}`);
      const onDisk = await hashFile(entry.path);
      const size = (await stat(entry.path)).size;
      if (onDisk !== entry.expectedHash || size !== entry.expectedBytes || result.upload.sha256 !== entry.expectedHash)
        await stop(row, scene.name, arm, "Saved native file hash/size differs from original");
      Object.assign(row.arms[arm], { file: relative(root, entry.path), sha256: onDisk, bytes: size,
        layout: "uint16 little-endian, row-major, 3 channels per native pixel (unmixed CFA)" });
      await saveProgress();
      console.log(`${scene.name} ${arm} ${result.shape.width}x${result.shape.height} saved ${size} bytes`);
    }
    const after = await hashFile(rafPath);
    if (after !== scene.raf_sha256) await stop(row, scene.name, "after", "RAF changed during decode");
    await saveProgress();
  }
  for (const [filename, entry] of uploads) assert.equal(entry.state, "done", `Upload incomplete: ${filename}`);
  assert.equal(uploadFailure, null, "An upload failed");
  report.completed = true;
  await writeFile(resolve(outDir, "decode-complete.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
} finally {
  await browser.close();
  await new Promise(r => server.close(r));
}
