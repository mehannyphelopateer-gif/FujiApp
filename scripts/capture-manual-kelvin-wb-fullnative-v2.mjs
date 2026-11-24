#!/usr/bin/env node
/**
 * Full-native capture, TRANSPORT REVISION 2 - research only.
 *
 * Scientific content is unchanged from the first full-native attempt (manifest 22f982...): the EXACT
 * same 3 fit sources, arms D (declared as-shot userMul) and E (fixed fit-derived neutral Kelvin 6600
 * profile), identical options, and every native buffer must equal the ORIGINAL sparse-primitive D/E
 * full-native SHA-256, so the original C/D and E/E-repeat controls cover these exact bytes.
 * The first attempt stopped on a transport timeout; it is preserved unchanged and labelled
 * INCOMPLETE_TRANSPORT_TIMEOUT (not a scientific failure). All 6 files are recaptured here; nothing
 * from the first attempt is reused.
 *
 * Transport changes only:
 *  - Completion is the HOST's verified upload, never a large page.evaluate return value. Each upload
 *    carries a small X-Native-Metadata header (shape + decoder metadata read in the page BEFORE upload);
 *    the server validates it before reading the body, streams size+SHA into an exclusive .partial,
 *    publishes via copyFile(COPYFILE_EXCL), re-hashes the published file, removes the .partial, and only
 *    then resolves that entry's completion promise.
 *  - No page-side digest; the worker is disposed BEFORE the HTTP upload.
 *  - A fresh BrowserServer (own child process) per arm; every cleanup step is time-bounded and, if a
 *    graceful close hangs, only that arm's own BrowserServer is killed via its public handle.
 *
 * A PASS (evaluated by training/qualify_manual_kelvin_wb_fullnative_v2.py) would only qualify declared
 * sensor-WB application on these 3 sources - not Fuji RGB, demosaic/matrix parity, an input arm or a
 * runtime contract. The original sparse NOT_QUALIFIED verdict stays as recorded.
 *
 * Usage: node scripts/capture-manual-kelvin-wb-fullnative-v2.mjs <committed-manifest-sha256>
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
const MANIFEST_REL = "training/diagnostics/manual-kelvin-wb-fullnative-v2-manifest.json";
const OUT_REL = "training/checkpoints/manual-kelvin-wb-fullnative-v2";
const OWN_REL = "scripts/capture-manual-kelvin-wb-fullnative-v2.mjs";
const PY_REL = "training/qualify_manual_kelvin_wb_fullnative_v2.py";
const COMMITTED = [MANIFEST_REL, OWN_REL, PY_REL];
const PARENT_MANIFEST_REL = "training/diagnostics/manual-kelvin-wb-fullnative-manifest.json";
const PARENT_PROGRESS_REL = "training/checkpoints/manual-kelvin-wb-fullnative/decode-progress.json";
const PARENT_COMPLETE_REL = "training/checkpoints/manual-kelvin-wb-fullnative/decode-complete.json";
const PARENT_SHA_PREFIX = "22f982";
const SPARSE_MANIFEST_REL = "training/diagnostics/manual-kelvin-wb-primitive-manifest.json";
const SPARSE_REPORT_REL = "training/diagnostics/manual-kelvin-wb-primitive-report.json";
const SPARSE_DECODE_REL = "training/checkpoints/manual-kelvin-wb-primitive/decode-complete.json";
const LIBRAW_DIR_REL = "node_modules/libraw-wasm/dist";
const LIBRAW_FILES = ["index.js", "worker.js", "libraw.js", "libraw.wasm"];
const ARMS = ["D", "E"];
const DECODE_TIMEOUT_MS = 10 * 60 * 1000; // per arm: decode + upload + host verification
const STEP_TIMEOUT_MS = 60 * 1000;        // launch / connect / newPage / goto
const CLEANUP_TIMEOUT_MS = 10 * 1000;     // each cleanup step
const MAX_METADATA_HEADER = 8192;
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
const log = msg => console.log(`[${new Date().toISOString()}] ${msg}`);

function withTimeout(promise, ms, label) {
  let timer;
  return Promise.race([promise, new Promise((_, fail) => { timer = setTimeout(() => fail(new Error(`${label} timed out after ${ms} ms`)), ms); })])
    .finally(() => clearTimeout(timer));
}

// ------------------------------------------------------------------ manifest, bindings, commits
const manifestBytes = await readFile(resolve(root, MANIFEST_REL));
assert.equal(sha(manifestBytes), args[0], "Manifest drift");
const manifest = JSON.parse(manifestBytes);
assert(manifest.bindings && typeof manifest.bindings === "object", "Manifest bindings required");
for (const [p, h] of Object.entries(manifest.bindings)) assert.equal(await hashFile(resolve(root, p)), h, `Bound file drift: ${p}`);
for (const p of [PARENT_MANIFEST_REL, PARENT_PROGRESS_REL, SPARSE_MANIFEST_REL, SPARSE_REPORT_REL, SPARSE_DECODE_REL, PY_REL,
  ...LIBRAW_FILES.map(f => `${LIBRAW_DIR_REL}/${f}`)])
  assert(p in manifest.bindings, `Required binding missing: ${p}`);
for (const p of COMMITTED)
  assert.equal(sha(execFileSync("git", ["show", `HEAD:${p}`], { cwd: root })), await hashFile(resolve(root, p)), `Not committed at HEAD: ${p}`);
assert.equal(typeof manifest.scope, "string", "Manifest scope required");
assert.equal(manifest.transport_revision, 2, "transport_revision must be 2");
if ("output_dir" in manifest) assert.equal(manifest.output_dir, OUT_REL, "Output directory is fixed");

// ------------------------------------------------------------------ parent (stopped) full-native attempt
const parentBytes = await readFile(resolve(root, PARENT_MANIFEST_REL));
const parentSha = sha(parentBytes);
assert(parentSha.startsWith(PARENT_SHA_PREFIX), "Unexpected parent full-native manifest");
assert.equal(manifest.parent_fullnative_manifest_sha256, parentSha, "parent_fullnative_manifest_sha256 differs");
const parent = JSON.parse(parentBytes);
const parentProgress = await readJson(PARENT_PROGRESS_REL);
assert.equal(parentProgress.manifest_sha256, parentSha, "Parent progress belongs to another manifest");
assert.equal(parentProgress.completed, false, "Parent attempt must be incomplete");
assert(parentProgress.stopped, "Parent attempt must have a recorded stop");
assert(!(await exists(resolve(root, PARENT_COMPLETE_REL))), "Parent attempt unexpectedly has a completion file");
assert.deepEqual(manifest.scenes, parent.scenes, "Scenes must be inherited from the parent full-native manifest exactly");
assert.deepEqual(manifest.base_options, parent.base_options, "Base options must be inherited from the parent exactly");
for (const k of ["primitive_limits"]) if (k in parent) assert.deepEqual(manifest[k], parent[k], `${k} must be inherited exactly`);

// ------------------------------------------------------------------ original sparse cross-checks (unchanged)
const sparseManifest = await readJson(SPARSE_MANIFEST_REL);
const sparseDecode = await readJson(SPARSE_DECODE_REL);
assert.equal(sparseDecode.completed, true, "Original sparse decode must be complete");
assert.equal(sparseDecode.manifest_sha256, sha(await readFile(resolve(root, SPARSE_MANIFEST_REL))), "Sparse decode belongs to another manifest");

const base = manifest.base_options;
assert.deepEqual(base, sparseManifest.base_options, "Base options must equal the original primitive's exactly");
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
assert.deepEqual(scenes.map(s => s.name), sparseManifest.scenes.map(s => s.name), "Scenes must be the exact original 3, same order");
assert.equal(new Set(scenes.map(s => s.raf_sha256)).size, 3, "Source RAFs must be distinct");
assert.equal(new Set(scenes.map(s => s.session)).size, 3, "Sessions must be distinct");
const sparseByName = new Map(sparseDecode.scenes.map(r => [r.name, r]));
const calibrationRoot = resolve(root, "calibration-input");
const allowedRaw = new Map();
const sceneByName = new Map(scenes.map(s => [s.name, s]));
const uploads = new Map(); // filename -> entry
const outDir = resolve(root, OUT_REL);

function armOptions(scene, arm) {
  return { ...base, useCameraWb: false, userMul: arm === "D" ? scene.as_shot_rgbg : scene.neutral_rgbg };
}

function newEntry(scene, arm, path, expectedBytes, expectedHash) {
  const entry = { scene: scene.name, arm, path, partial: `${path}.partial`, expectedBytes, expectedHash, state: "pending" };
  entry.completion = new Promise((ok, fail) => {
    entry.resolve = value => { if (!entry.settled) { entry.settled = true; ok(value); } };
    entry.reject = error => { if (!entry.settled) { entry.settled = true; fail(error); } };
  });
  entry.completion.catch(() => {}); // never an unhandled rejection; the arm loop awaits this same promise
  return entry;
}

scenes.forEach((s, i) => {
  const old = sparseManifest.scenes[i];
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
  const oldRow = sparseByName.get(s.name);
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
    uploads.set(filename, newEntry(s, arm, resolve(outDir, filename), shape.width * shape.height * 3 * 2, s.original_native_sha256[arm]));
  }
  const rawPath = resolve(calibrationRoot, s.shoot, s.name);
  assert(rawPath.startsWith(calibrationRoot + sep), `${s.name}: path escapes calibration-input`);
  allowedRaw.set(s.name, rawPath);
});
assert.equal(uploads.size, 6, "Exactly six allowlisted uploads");

await mkdir(outDir); // refuses an existing directory: never overwrite evidence
const assets = resolve(root, LIBRAW_DIR_REL);

// ------------------------------------------------------------------ upload validation (serial, host-authoritative)
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

/** Parse and validate the page's actual decoder readback BEFORE any body byte is accepted. */
function parseReadback(header, entry) {
  assert(typeof header === "string" && header.length > 0, "Missing X-Native-Metadata header");
  assert(header.length <= MAX_METADATA_HEADER, "X-Native-Metadata header too large");
  const value = JSON.parse(decodeURIComponent(header));
  const scene = sceneByName.get(entry.scene);
  assert(value && typeof value === "object" && value.shape && value.returned_metadata, "Malformed readback");
  assert.deepEqual(value.shape, scene.native_shape, "Readback shape differs from frozen native shape");
  const md = value.returned_metadata;
  assert.equal(md.camera_model, "X100VI", "Readback camera model differs");
  assert.equal(md.fujiWB, 4080, "Readback Fuji WB preset differs");
  assert.deepEqual(md.cam_mul, scene.cam_mul, "Readback cam_mul differs from frozen scene cam_mul");
  assert(md.pre_mul === null || (Array.isArray(md.pre_mul) && md.pre_mul.every(Number.isFinite)), "Readback pre_mul malformed (descriptive only)");
  return { shape: value.shape, returned_metadata: { cam_mul: md.cam_mul, pre_mul: md.pre_mul, camera_model: md.camera_model, fujiWB: md.fujiWB } };
}

async function handleUpload(req, res, filename) {
  const entry = uploads.get(filename);
  const reply = (code, body) => { if (!res.headersSent) { res.writeHead(code, { "Content-Type": "application/json" }); res.end(JSON.stringify(body)); } };
  if (!entry) { req.resume(); return reply(404, { ok: false, error: "Upload not allowlisted" }); }
  if (uploadBusy) { req.resume(); return reply(409, { ok: false, error: "Another upload is in progress" }); }
  if (entry.state !== "pending") { req.resume(); return reply(409, { ok: false, error: `Upload already ${entry.state}` }); }
  uploadBusy = true;
  entry.state = "receiving";
  try {
    const readback = parseReadback(req.headers["x-native-metadata"], entry);
    assert(!req.headers["transfer-encoding"], "Chunked transfer refused");
    assert.equal(req.headers["content-length"], String(entry.expectedBytes), "Content-Length differs from expected native size");
    assert(!(await exists(entry.path)) && !(await exists(entry.partial)), "Target or partial already exists");
    const counter = sizeAndHash(entry.expectedBytes);
    await pipeline(req, counter.t, createWriteStream(entry.partial, { flags: "wx" }));
    const got = counter.result();
    assert.equal(got.bytes, entry.expectedBytes, "Uploaded byte count differs");
    assert.equal(got.sha256, entry.expectedHash, "Uploaded native SHA256 differs from original");
    entry.state = "publishing";
    await copyFile(entry.partial, entry.path, constants.COPYFILE_EXCL); // exFAT: no hard links
    assert.equal((await stat(entry.path)).size, entry.expectedBytes, "Published file is short");
    assert.equal(await hashFile(entry.path), entry.expectedHash, "Published file hash differs");
    await unlink(entry.partial);
    entry.state = "done";
    const info = { file: relative(root, entry.path), sha256: got.sha256, bytes: got.bytes, ...readback };
    log(`upload published ${filename} ${got.bytes} bytes`);
    entry.resolve(info);
    reply(200, { ok: true, bytes: got.bytes, sha256: got.sha256 });
  } catch (error) {
    entry.state = "failed";
    entry.error = String(error?.message ?? error);
    uploadFailure ??= `${filename}: ${entry.error}`;
    entry.reject(new Error(`Upload ${filename}: ${entry.error}`));
    req.resume(); // drain; any bad .partial is kept as evidence
    reply(entry.error.includes("header") || entry.error.includes("Readback") ? 400 : 500, { ok: false, error: entry.error });
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
    const info = await stat(path); // follows symlinks; streamed, never copied
    res.writeHead(200, { "Content-Length": info.size, "Content-Type": extname(path) === ".js" ? "text/javascript" : extname(path) === ".wasm" ? "application/wasm" : "application/octet-stream" });
    createReadStream(path).pipe(res);
  } catch { if (!res.headersSent) { res.writeHead(404); res.end(); } }
});
server.requestTimeout = DECODE_TIMEOUT_MS;
await new Promise(r => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`;

// ------------------------------------------------------------------ in-page decode + upload (tiny return value)
async function decodeAndUpload({ name, options, expected, filename }) {
  const probe = new Uint8Array(new Uint16Array([0x0102]).buffer);
  if (probe[0] !== 0x02 || probe[1] !== 0x01) return { error: "Browser typed arrays are not little-endian" };
  const { default: LibRaw } = await import("/libraw/index.js");
  let decoder = new LibRaw();
  try {
    const response = await fetch(`/raw/${name}`);
    if (!response.ok) return { error: "Missing allowlisted RAF" };
    await decoder.open(new Uint8Array(await response.arrayBuffer()), options);
    const image = await decoder.imageData();
    const shape = { width: image?.width, height: image?.height, colors: image?.colors, bits: image?.bits };
    if (!image || image.bits !== 16 || image.colors !== 3 || !(image.data instanceof Uint16Array))
      return { error: "Expected 16-bit, 3-colour Uint16Array imageData" };
    if (image.width !== expected.width || image.height !== expected.height) return { error: "Native dimensions differ from frozen shape" };
    if (image.data.length !== image.width * image.height * 3) return { error: "imageData length differs" };
    const buffer = image.data.buffer;
    if (!(buffer instanceof ArrayBuffer) || (typeof SharedArrayBuffer !== "undefined" && buffer instanceof SharedArrayBuffer))
      return { error: "imageData is not backed by a plain main-thread ArrayBuffer" };
    const metadata = await decoder.metadata(true);
    const toArray = v => (v == null ? null : Array.from(v, Number));
    const returned_metadata = { cam_mul: toArray(metadata.color_data?.cam_mul), pre_mul: toArray(metadata.color_data?.pre_mul),
      camera_model: metadata.camera_model == null ? null : String(metadata.camera_model).trim(), fujiWB: metadata.fuji?.WB_Preset ?? null };
    const bytes = new Uint8Array(buffer, image.data.byteOffset, image.data.byteLength);
    const byteLength = bytes.byteLength;
    decoder.dispose(); // free the worker/WASM heap BEFORE the upload
    decoder = null;
    if (bytes.byteLength !== byteLength || byteLength !== expected.width * expected.height * 6)
      return { error: "Native buffer changed after worker dispose" };
    const header = encodeURIComponent(JSON.stringify({ shape, returned_metadata }));
    const upload = await fetch(`/upload/${encodeURIComponent(filename)}`, { method: "POST", body: bytes,
      headers: { "Content-Type": "application/octet-stream", "X-Native-Metadata": header } });
    let body = null;
    try { body = await upload.json(); } catch { body = null; }
    if (!upload.ok || !body?.ok) return { error: `Upload failed: HTTP ${upload.status} ${body?.error ?? ""}` };
    return { ok: true };
  } catch (error) {
    return { error: `Page error: ${String(error?.message ?? error)}` };
  } finally { if (decoder) decoder.dispose(); }
}

// ------------------------------------------------------------------ bounded per-arm browser lifecycle
async function bounded(label, fn, warnings) {
  try { await withTimeout(Promise.resolve().then(fn), CLEANUP_TIMEOUT_MS, label); return true; }
  catch (error) { warnings.push(`${label}: ${String(error?.message ?? error)}`); return false; }
}

const ownExited = proc => !proc || proc.exitCode !== null || proc.signalCode !== null;

/** Close page, client and this arm's OWN BrowserServer; kill only that child if graceful close hangs. */
async function cleanup(ctx, warnings) {
  if (ctx.page) await bounded("page.close", () => ctx.page.close(), warnings);
  if (ctx.browser) await bounded("browser.close", () => ctx.browser.close(), warnings);
  if (ctx.browserServer) {
    const proc = ctx.browserServer.process();
    const closed = await bounded("browserServer.close", () => ctx.browserServer.close(), warnings);
    if (!closed || !ownExited(proc)) {
      await bounded("browserServer.kill", () => ctx.browserServer.kill(), warnings);
      if (!ownExited(proc)) {
        await new Promise(r => setTimeout(r, 1000));
        if (!ownExited(proc)) return false; // isolation cannot be confirmed
      }
    }
  }
  return true;
}

// ------------------------------------------------------------------ report
const report = { manifest_sha256: args[0], scope: manifest.scope, transport_revision: 2, scenes: [], completed: false,
  parent_attempt: { manifest_sha256: parentSha, progress_sha256: sha(await readFile(resolve(root, PARENT_PROGRESS_REL))),
    status: "INCOMPLETE_TRANSPORT_TIMEOUT", note: "Transport stop, not a scientific failure; preserved unchanged, no files reused." },
  provenance_note: "Options are DECLARED (identical to the original sparse primitive), not internal applied-parameter readback; " +
    "returned metadata is the page's actual decoder readback, validated by the host before upload; pre_mul is descriptive only. " +
    "Each native buffer equals the original D/E full-native SHA256, so original C/D and E/E-repeat controls cover these exact bytes. " +
    "Not Fuji colour accuracy, demosaic/matrix parity, an input arm, runtime or model qualification." };
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

async function captureArm(scene, arm, row) {
  const filename = `${scene.name}-${arm}.native.u16le`;
  const entry = uploads.get(filename);
  const options = armOptions(scene, arm);
  const declared = { declared_options: options, declared_options_sha256: sha(JSON.stringify(options)) };
  row.arms[arm] = { ...declared, status: "started" };
  const warnings = [];
  const ctx = {};
  let failure = null;
  const rejectIfNotStarted = reason => () => { if (entry.state === "pending") entry.reject(new Error(reason)); };
  try {
    ctx.browserServer = await withTimeout(chromium.launchServer({ headless: true, host: "127.0.0.1" }), STEP_TIMEOUT_MS, "launchServer");
    ctx.browser = await withTimeout(chromium.connect(ctx.browserServer.wsEndpoint()), STEP_TIMEOUT_MS, "connect");
    ctx.browser.once("disconnected", rejectIfNotStarted("Browser disconnected before upload"));
    ctx.page = await withTimeout(ctx.browser.newPage(), STEP_TIMEOUT_MS, "newPage");
    ctx.page.once("crash", rejectIfNotStarted("Page crashed before upload"));
    ctx.page.once("close", rejectIfNotStarted("Page closed before upload"));
    await withTimeout(ctx.page.goto(`${origin}/`), STEP_TIMEOUT_MS, "goto");
    // Not awaited: completion is the host-verified upload, never this return value.
    const evaluation = ctx.page.evaluate(decodeAndUpload, { name: scene.name, options, filename,
      expected: { width: scene.native_shape.width, height: scene.native_shape.height } });
    evaluation.then(
      r => { if (r?.error) entry.reject(new Error(`Page: ${r.error}`)); else rejectIfNotStarted("Page finished without uploading")(); },
      e => rejectIfNotStarted(`Page evaluate failed before upload: ${String(e?.message ?? e)}`)(),
    ).catch(() => {});
    const info = await withTimeout(entry.completion, DECODE_TIMEOUT_MS, `${scene.name} ${arm} host-verified upload`);
    // Persist the host-validated readback BEFORE any browser cleanup.
    Object.assign(row.arms[arm], { status: "published", file: info.file, sha256: info.sha256, bytes: info.bytes,
      shape: info.shape, returned_metadata: info.returned_metadata,
      layout: "uint16 little-endian, row-major, 3 channels per native pixel (unmixed CFA)" });
    await saveProgress();
    // Independent Node-side verification of the published bytes.
    const onDisk = await hashFile(entry.path);
    const size = (await stat(entry.path)).size;
    if (onDisk !== entry.expectedHash || size !== entry.expectedBytes) failure = "Published native file hash/size differs from original";
    else { row.arms[arm].status = "verified"; log(`native verified ${filename}`); }
  } catch (error) {
    failure = String(error?.message ?? error);
  }
  const isolated = await cleanup(ctx, warnings);
  row.arms[arm].cleanup_warnings = warnings;
  if (!isolated && !failure) failure = "Own BrowserServer process could not be confirmed terminated";
  await saveProgress();
  log(`cleanup done ${filename}${warnings.length ? ` (warnings: ${warnings.length})` : ""}`);
  if (failure) await stop(row, scene.name, arm, failure);
  if (uploadFailure) await stop(row, scene.name, arm, `Upload failure: ${uploadFailure}`);
}

async function closeServer() {
  server.closeAllConnections?.();
  try { await withTimeout(new Promise(r => server.close(r)), CLEANUP_TIMEOUT_MS, "server.close"); }
  catch (error) { log(`warning: ${error.message}`); }
}

try {
  for (const scene of scenes) {
    const rafPath = allowedRaw.get(scene.name);
    assert.equal(await hashFile(rafPath), scene.raf_sha256, `RAF changed before decode: ${scene.name}`);
    const row = { name: scene.name, arms: {} };
    report.scenes.push(row);
    for (const arm of ARMS) await captureArm(scene, arm, row);
    const after = await hashFile(rafPath);
    if (after !== scene.raf_sha256) await stop(row, scene.name, "after", "RAF changed during decode");
    row.raf_sha256_after = after;
    await saveProgress();
  }
  for (const [filename, entry] of uploads) assert.equal(entry.state, "done", `Upload incomplete: ${filename}`);
  assert.equal(uploadFailure, null, "An upload failed");
  for (const row of report.scenes) for (const arm of ARMS) assert.equal(row.arms[arm]?.status, "verified", `${row.name} ${arm} not verified`);
  report.completed = true;
  await writeFile(resolve(outDir, "decode-complete.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  log("decode-complete written");
} finally {
  await closeServer();
}
