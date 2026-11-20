#!/usr/bin/env node
/** F24 full-native gamma-only paired export; separate sampling revision P1b. No target reads or model execution. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync } from "node:fs";
import { readFile, writeFile, stat, mkdir, rename } from "node:fs/promises";
import { resolve, basename, extname, relative } from "node:path";
import { chromium } from "playwright";
import { decodeFjlrgV2 } from "./lib/fjlrg-v2.mjs";
import { librawGammaLookup } from "./lib/libraw-gamma-lookup.mjs";

const root = process.cwd(), manifestPath = "training/diagnostics/provia-v2-fullnative-pilot-manifest.json";
const sha = b => createHash("sha256").update(b).digest("hex");
const fileHash = async p => { const h = createHash("sha256"); for await (const b of createReadStream(p)) h.update(b); return h.digest("hex"); };
const expected = process.argv[2], resume = process.argv[3] === "--resume";
assert(process.argv.length === 3 || (process.argv.length === 4 && resume), "Expected manifest SHA and optional --resume");
const bytes = await readFile(manifestPath); assert.equal(sha(bytes), expected);
const manifest = JSON.parse(bytes);
for (const [p, h] of Object.entries(manifest.bindings)) assert.equal(await fileHash(p), h, `Bound source differs: ${p}`);
for (const p of [manifestPath, "scripts/export-provia-input-v2-fullnative.mjs", "training/verify_provia_v2_fullnative.py"])
  assert.equal(sha(execFileSync("git", ["show", `HEAD:${p}`])), await fileHash(p), "Code/manifest must be committed");
assert.equal(manifest.scenes.length, 24);
const output = resolve(root, "training/checkpoints/provia-input-v2-pilot-fullnative");
assert.equal(resolve(root, manifest.output_dir), output);
if (resume) {
  assert.equal(JSON.parse(await readFile(resolve(output, "run-anchor.json"), "utf8")).manifest_sha256, expected, "Cannot resume a different manifest");
  assert(!existsSync(resolve(output, "complete.json")), "Pilot already complete; run independent verification instead");
} else {
  await mkdir(output); // Do not overwrite previous scientific evidence.
  await writeFile(resolve(output, "run-anchor.json"), JSON.stringify({ manifest_sha256: expected }) + "\n", { flag: "wx" });
}
const allowed = new Map(manifest.scenes.map(s => [s.name, resolve(root, "calibration-input", s.shoot, s.name)]));
assert.equal(allowed.size, 24);
const libDir = resolve(root, "node_modules/libraw-wasm/dist"), codec = resolve(root, "scripts/lib/fjlrg-v2.mjs");
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/") { res.writeHead(200, { "Content-Type": "text/html" }); res.end('<!doctype html><link rel="icon" href="data:,">'); return; }
    let p;
    if (url.pathname === "/codec.mjs") p = codec;
    else if (url.pathname === "/gamma.mjs") p = resolve(root, "scripts/lib/libraw-gamma-lookup.mjs");
    else if (url.pathname === "/transfer.mjs") p = resolve(root, "scripts/lib/provia-transfer-histogram.mjs");
    else if (url.pathname.startsWith("/libraw/")) {
      const n = decodeURIComponent(url.pathname.slice(8)); assert.equal(n, basename(n));
      assert(["index.js", "worker.js", "libraw.js", "libraw.wasm"].includes(n)); p = resolve(libDir, n);
    } else if (url.pathname.startsWith("/raw/")) p = allowed.get(decodeURIComponent(url.pathname.slice(5)));
    assert(p, "Not allowlisted"); const info = await stat(p);
    res.writeHead(200, { "Content-Length": info.size, "Content-Type": [".js", ".mjs"].includes(extname(p)) ? "text/javascript" : extname(p) === ".wasm" ? "application/wasm" : "application/octet-stream" });
    createReadStream(p).pipe(res);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));
assert.deepEqual(manifest.native_measurement.eligible_linear_codes_inclusive,[656,62258]);
const table=librawGammaLookup(); assert.deepEqual(Array.from(table.keys()).filter(i=>table[i]===65535), manifest.legacy_white_preimage_codes);
const browser = await chromium.launch({ headless: true });
const report = { manifest_sha256: expected, completed: false, scenes: [], scope: manifest.scope, resume_actions: [] };
try {
  for (const scene of manifest.scenes) {
    assert.equal(await fileHash(allowed.get(scene.name)), scene.raf_sha256, "RAF differs before decode");
    const folder = resolve(output, scene.shoot), sealed = resolve(folder, "scene-complete.json"), seal = sealed + ".sha256";
    if (resume && existsSync(sealed) && existsSync(seal)) {
      const raw = await readFile(sealed); assert.equal(sha(raw), (await readFile(seal, "utf8")).trim(), "Scene record checksum differs");
      const row = JSON.parse(raw); assert.equal(row.name, scene.name);
      const full = row.native_transfer, hist = new Uint32Array(65536);
      for (const [e,n] of full.histogram) { assert(Number.isInteger(e) && e>=0 && e<65536 && Number.isSafeInteger(n) && n>0 && hist[e]===0); hist[e]=n; }
      const { histogramSummary } = await import("./lib/provia-transfer-histogram.mjs");
      assert.deepEqual(histogramSummary(hist, full.per_channel_n, full.ineligible_count), full); assert(full.passed);
      assert.deepEqual(row.arms.legacy.nativeWhite, row.arms.G.nativeLegacyWhitePreimage);
      assert.deepEqual(Object.keys(row.arms).sort(), ["G", "G-repeat", "legacy"].sort());
      assert.equal(row.arms.legacy.sha256, scene.canonical_fjlrg_sha256);
      assert.equal(row.arms.G.sha256, row.arms["G-repeat"].sha256);
      for (const arm of ["legacy", "G", "G-repeat"]) {
        const record = row.arms[arm], p = resolve(root, record.file);
        assert.equal(resolve(p, ".."), folder); assert.equal(await fileHash(p), record.sha256);
        if (arm !== "G-repeat") {
          const strip = record.strips; assert.equal(resolve(root, strip.file, ".."), folder);
          assert.equal(await fileHash(resolve(root, strip.file)), strip.sha256);
        }
        if (arm !== "legacy") {
          const provenance = JSON.parse(await readFile(p + ".json", "utf8"));
          assert.deepEqual(provenance, record.provenance);
          await decodeFjlrgV2(new Uint8Array(await readFile(p)), provenance, { expectedContract: "v2-linear-asshot", expectedSourceSha256: scene.raf_sha256, expectedWasmSha256: manifest.wasm_sha256 });
        }
      }
      assert.equal(row.gates_G0_G1_G2_G5_passed, true);
      report.scenes.push(row); report.resume_actions.push({ name: scene.name, action: "verified sealed scene reused" });
      console.log(`resume verified ${scene.name} (${report.scenes.length}/24)`);
      continue;
    }
    if (existsSync(folder)) {
      assert(resume, "Existing scientific output folder");
      const partials = resolve(output, "partials"); await mkdir(partials, { recursive: true });
      const preserved = resolve(partials, `${scene.shoot}.${Date.now()}`); await rename(folder, preserved);
      report.resume_actions.push({ name: scene.name, action: "partial preserved before fresh scene replay", folder: relative(root, preserved) });
    }
    await mkdir(folder);
    const row = { name: scene.name, shoot: scene.shoot, arms: {} };
    let pairedPage;
    try {
    for (const arm of ["legacy", "G", "G-repeat"]) {
      const options = { ...manifest.base_options, gamm: arm === "legacy" ? [1, 1] : [1, 1, 0, 0, 0, 0] };
      const page = arm === "G" ? pairedPage : await browser.newPage(), started = Date.now();
      if (arm === "legacy") pairedPage = page;
      try {
        if (arm !== "G") await page.goto(`http://127.0.0.1:${server.address().port}`);
        const result = await page.evaluate(async ({ scene, arm, options, wasmSha, legacyWhiteMin }) => {
          const { default: LibRaw } = await import("/libraw/index.js");
          const C = await import("/codec.mjs");
          const decoder = new LibRaw();
          try {
            const response = await fetch(`/raw/${scene.name}`); if (!response.ok) throw new Error("Missing allowlisted RAF");
            await decoder.open(new Uint8Array(await response.arrayBuffer()), options);
            const image = await decoder.imageData();
            const metadata = await decoder.metadata(true);
            if (String(metadata.camera_model).trim() !== "X100VI" || metadata.fuji?.WB_Preset !== scene.source_wb_mode) throw new Error("Camera/WB metadata differs");
            const wb = metadata.color_data?.cam_mul;
            if (!Array.isArray(wb) || JSON.stringify(wb) !== JSON.stringify(scene.cam_mul)) throw new Error("Source WB coefficients differ");
            if (image.bits !== 16 || image.colors !== 3 || !(image.data instanceof Uint16Array)) throw new Error("Expected16bitRGB");
            const toBase64 = bytes => new Promise((ok, reject) => { const r = new FileReader(); r.onload = () => ok(String(r.result).split(",")[1]); r.onerror = reject; r.readAsDataURL(new Blob([bytes])); });
            const samples = [], nativeWhite = [0, 0, 0], nativeLegacyWhitePreimage = [0, 0, 0];
            for (let i = 0; i < image.data.length; i += 3) for (let c = 0; c < 3; c++) {
              if (image.data[i + c] === 65535) nativeWhite[c]++;
              if (image.data[i + c] >= legacyWhiteMin) nativeLegacyWhitePreimage[c]++;
            }
            for (let y = 64; y < image.height; y += 128) for (let x = 64; x < image.width; x += 128) {
              const p = (y * image.width + x) * 3; samples.push(image.data[p], image.data[p + 1], image.data[p + 2]);
            }
            let native_transfer, stripBase64, stripRows;
            if (arm !== "G-repeat") {
              stripRows = []; for (let y=64; y<image.height; y+=128) stripRows.push(y);
              const strips = new Uint16Array(stripRows.length * image.width * 3);
              for (const [j,y] of stripRows.entries()) strips.set(image.data.subarray(y*image.width*3,(y+1)*image.width*3),j*image.width*3);
              stripBase64 = await toBase64(strips);
            }
            if (arm === "legacy") {
              // Own this copy independently of the worker/Wasm lifetime.
              window.__legacyNative = {data:new Uint16Array(image.data), width:image.width, height:image.height};
            } else if (arm === "G") {
              const old=window.__legacyNative;
              if (!old || old.width!==image.width || old.height!==image.height) throw new Error("Legacy native geometry mismatch");
              const {measureNativeTransfer}=await import("/transfer.mjs");
              const {librawGammaLookup}=await import("/gamma.mjs");
              native_transfer=measureNativeTransfer(old.data,image.data,librawGammaLookup());
              delete window.__legacyNative;
            }
            const scale = Math.min(1, 1536 / Math.max(image.width, image.height));
            const width = Math.max(1, Math.round(image.width * scale)), height = Math.max(1, Math.round(image.height * scale));
            const data = new Uint16Array(width * height * 3), workingWhite = [0, 0, 0];
            for (let y = 0; y < height; y++) {
              const y0 = Math.floor(y * image.height / height), y1 = Math.max(y0 + 1, Math.floor((y + 1) * image.height / height));
              for (let x = 0; x < width; x++) {
                const x0 = Math.floor(x * image.width / width), x1 = Math.max(x0 + 1, Math.floor((x + 1) * image.width / width));
                const sum = [0, 0, 0];
                for (let sy = y0; sy < y1; sy++) for (let sx = x0; sx < x1; sx++) {
                  const p = (sy * image.width + sx) * 3;
                  sum[0] += image.data[p]; sum[1] += image.data[p + 1]; sum[2] += image.data[p + 2];
                }
                for (let c = 0; c < 3; c++) {
                  const v = Math.round(sum[c] / ((y1 - y0) * (x1 - x0))); data[(y * width + x) * 3 + c] = v;
                  if (v === 65535) workingWhite[c]++;
                }
              }
            }
            let container, provenance;
            if (arm === "legacy") {
              container = new Uint8Array(32 + data.byteLength); const v = new DataView(container.buffer);
              for (const [i, c] of [..."FJLRGB16"].entries()) container[i] = c.charCodeAt(0);
              for (const [o, n] of [[8, 1], [10, 32], [20, 3], [22, 16]]) v.setUint16(o, n, true);
              for (const [o, n] of [[12, width], [16, height], [24, data.byteLength], [28, 3]]) v.setUint32(o, n, true);
              new Uint16Array(container.buffer, 32).set(data);
            } else {
              const optionsJson = C.serializeOptions(options), optionsSha = await C.sha256Hex(new TextEncoder().encode(optionsJson));
              container = C.encodeFjlrgV2({ width, height, data }, { inputContract: "v2-linear-asshot", wbLevelsRgbg: wb, optionsSha256: optionsSha });
              provenance = { schema: "fujiapp.raw-input.v2", input_contract: "v2-linear-asshot", container_sha256: await C.sha256Hex(container),
                source_raf_sha256: scene.raf_sha256, wasm_sha256: wasmSha, decoder_package: "libraw-wasm", decoder_version: "1.6.0", libraw_build_version: "0.22.1",
                options_json: optionsJson, options_sha256: optionsSha, declared_options: options, wb_source: { kind: "cam_mul", levels_rgbg: wb },
                source_wb_mode: scene.source_wb_mode, source_wb_fine_tune_raw: scene.source_wb_fine_tune_raw, source_kelvin: scene.source_kelvin,
                capture_dynamic_range: scene.capture_dynamic_range, capture_dynamic_range_source: "frozen approved train manifest captureDynamicRange",
                decoded_native_size: [image.width, image.height], parameters_are_declared: true };
              await C.decodeFjlrgV2(container, provenance, { expectedContract: "v2-linear-asshot", expectedSourceSha256: scene.raf_sha256, expectedWasmSha256: wasmSha });
            }
            const base64 = await new Promise((ok, reject) => { const r = new FileReader(); r.onload = () => ok(String(r.result).split(",")[1]); r.onerror = reject; r.readAsDataURL(new Blob([container])); });
            return { base64, provenance, width, height, native_transfer, stripBase64, stripRows, native_size: [image.width, image.height], samples, nativeWhite, nativeLegacyWhitePreimage, workingWhite };
          } finally { decoder.dispose(); }
        }, { scene, arm, options, wasmSha: manifest.wasm_sha256, legacyWhiteMin: manifest.legacy_white_preimage_codes[0] });
        const container = Buffer.from(result.base64, "base64"), digest = sha(container);
        if (arm === "legacy") assert.equal(digest, scene.canonical_fjlrg_sha256, "G0 legacy anchor failed");
        if (arm === "G-repeat") assert.equal(digest, row.arms.G.sha256, "G2 independent decode not deterministic");
        if (arm !== "legacy") await decodeFjlrgV2(new Uint8Array(container), result.provenance, { expectedContract: "v2-linear-asshot", expectedSourceSha256: scene.raf_sha256, expectedWasmSha256: manifest.wasm_sha256 });
        const name = arm === "legacy" ? "legacy-replay.fjlrg" : arm === "G" ? "input-v2-linear-asshot.fjlrg" : "input-v2-linear-asshot-repeat.fjlrg";
        const file = resolve(folder, name); await writeFile(file, container, { flag: "wx" });
        if (result.provenance) await writeFile(file + ".json", JSON.stringify(result.provenance, null, 2) + "\n", { flag: "wx" });
        const { base64, stripBase64, stripRows, ...info } = result;
        if (arm !== "G-repeat") {
          const strips=Buffer.from(stripBase64,"base64"), stripFile=resolve(folder,`${arm}-native-strips.u16le`);
          await writeFile(stripFile,strips,{flag:"wx"});
          info.strips={file:relative(root,stripFile),sha256:sha(strips),rows:stripRows,width:info.native_size[0],channels:3,dtype:"uint16-le"};
        }

        row.arms[arm] = { ...info, file: relative(root, file), sha256: digest, seconds: (Date.now() - started) / 1000 };
        console.log(`${report.scenes.length + 1}/24 ${scene.name} ${arm} ${info.width}x${info.height} ${row.arms[arm].seconds.toFixed(1)}s`);
      } finally { if (arm !== "legacy") await page.close(); }
    }
    const old = row.arms.legacy, g = row.arms.G;
    row.native_transfer = g.native_transfer;
    // Persist all measurements before asserting scientific gates. Never seal a failed scene.
    await writeFile(resolve(folder,"scene-measured-unsealed.json"), JSON.stringify(row,null,2)+"\n",{flag:"wx"});
    assert.deepEqual(old.native_size, g.native_size);
    assert(row.native_transfer.n>=1000,"G1 full-native transfer unmeasurable");
    assert(row.native_transfer.passed,"G1 full-native transfer gate failed");
    assert.deepEqual(old.nativeWhite, g.nativeLegacyWhitePreimage, "G5 quantizer-aware native white counts differ");
    assert.equal(await fileHash(allowed.get(scene.name)), scene.raf_sha256, "RAF differs after decode");
    row.gates_G0_G1_G2_G5_passed = true;
    const sealedBytes = Buffer.from(JSON.stringify(row, null, 2) + "\n");
    await writeFile(sealed, sealedBytes, { flag: "wx" });
    await writeFile(seal, sha(sealedBytes) + "\n", { flag: "wx" });
    report.scenes.push(row);
    await writeFile(resolve(output, "progress.json"), JSON.stringify(report, null, 2) + "\n");
    } catch (e) {
      await writeFile(resolve(folder,"failure.json"),JSON.stringify({name:scene.name,shoot:scene.shoot,error:String(e),measurements:row},null,2)+"\n",{flag:"wx"});
      throw e;
    } finally { if (pairedPage && !pairedPage.isClosed()) await pairedPage.close(); }
  }
  report.completed = true;
  await writeFile(resolve(output, "complete.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
} finally { await browser.close(); await new Promise(r => server.close(r)); }
