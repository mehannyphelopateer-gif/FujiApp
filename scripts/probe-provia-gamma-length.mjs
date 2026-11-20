#!/usr/bin/env node
/** Research-only single-scene gamma-array-length binary probe; no model or targets. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile, mkdir } from "node:fs/promises";
import { basename, resolve, relative, extname } from "node:path";
import { chromium } from "playwright";

const root = process.cwd();
const manifestRel = "training/diagnostics/provia-gamma-length-probe-manifest.json";
const args = process.argv.slice(2);
assert.equal(args.length, 1, "Pass the committed manifest SHA256");
const sha = b => createHash("sha256").update(b).digest("hex");
const hashFile = async p => { const h = createHash("sha256"); for await (const b of createReadStream(p)) h.update(b); return h.digest("hex"); };
const manifestBytes = await readFile(manifestRel);
assert.equal(sha(manifestBytes), args[0], "Manifest drift");
const manifest = JSON.parse(manifestBytes);
for (const [p, h] of Object.entries(manifest.bindings)) assert.equal(await hashFile(p), h, `Bound file drift: ${p}`);
for (const p of [manifestRel, "scripts/probe-provia-gamma-length.mjs", "training/eval_provia_gamma_length.py"])
  assert.equal(sha(execFileSync("git", ["show", `HEAD:${p}`])), await hashFile(p), "Uncommitted research code");
const outDir = resolve(root, manifest.output_dir);
assert.equal(outDir, resolve(root, "training/checkpoints/provia-gamma-length-probe"));
await mkdir(outDir); // Existing directory is refused; never overwrite prior evidence.
const allowed = new Map(manifest.scenes.map(s => [s.name, resolve(root, "calibration-input", s.shoot, s.name)]));
assert.equal(allowed.size, 1);
const assets = resolve(root, "node_modules/libraw-wasm/dist");
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/") { res.writeHead(200, { "Content-Type": "text/html" }); res.end('<!doctype html><link rel="icon" href="data:,">'); return; }
    let path;
    if (url.pathname.startsWith("/libraw/")) {
      const n = decodeURIComponent(url.pathname.slice(8));
      assert.equal(n, basename(n));
      assert(["index.js", "worker.js", "libraw.js", "libraw.wasm"].includes(n));
      path = resolve(assets, n);
    } else if (url.pathname.startsWith("/raw/")) path = allowed.get(decodeURIComponent(url.pathname.slice(5)));
    assert(path, "Route is not allowlisted");
    const info = await stat(path);
    res.writeHead(200, { "Content-Length": info.size, "Content-Type": extname(path) === ".js" ? "text/javascript" : extname(path) === ".wasm" ? "application/wasm" : "application/octet-stream" });
    createReadStream(path).pipe(res);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ headless: true });
const report = { manifest_sha256: args[0], scenes: [], scope: manifest.scope };
try {
  for (const scene of manifest.scenes) {
    assert.equal(await hashFile(allowed.get(scene.name)), scene.raf_sha256);
    const row = { name: scene.name, arms: {} };
    for (const arm of ["D2", "D6", "E6"]) {
      const options = { ...manifest.base_options };
      options.userMul = arm === "E6" ? scene.auto_rgbg : scene.as_shot_rgbg;
      options.gamm = arm === "D2" ? [1, 1] : [1, 1, 0, 0, 0, 0];
      const page = await browser.newPage();
      try {
        await page.goto(`http://127.0.0.1:${server.address().port}`);
        const result = await page.evaluate(async ({ name, options }) => {
          const { default: LibRaw } = await import("/libraw/index.js");
          const decoder = new LibRaw();
          try {
            const response = await fetch(`/raw/${name}`);
            if (!response.ok) throw new Error("Missing allowlisted RAF");
            await decoder.open(new Uint8Array(await response.arrayBuffer()), options);
            const image = await decoder.imageData();
            if (!image || image.bits !== 16 || !(image.data instanceof Uint16Array) || ![3, 4].includes(image.colors)) throw new Error("Expected 16-bit RGB");
            // Fixed native samples BEFORE area averaging: gamma and averaging do not commute.
            const nativeSamples = { width: image.width, height: image.height, offset: 64, stride: 128, data: [] };
            for (let y = 64; y < image.height; y += 128) for (let x = 64; x < image.width; x += 128) {
              const p = (y * image.width + x) * image.colors;
              nativeSamples.data.push(image.data[p], image.data[p + 1], image.data[p + 2]);
            }
            // Exact canonical phase3 area-average and round-to-uint16 convention.
            const scale = Math.min(1, 1536 / Math.max(image.width, image.height));
            const width = Math.max(1, Math.round(image.width * scale));
            const height = Math.max(1, Math.round(image.height * scale));
            const data = new Uint16Array(width * height * 3);
            for (let y = 0; y < height; y++) {
              const y0 = Math.floor(y * image.height / height), y1 = Math.max(y0 + 1, Math.floor((y + 1) * image.height / height));
              for (let x = 0; x < width; x++) {
                const x0 = Math.floor(x * image.width / width), x1 = Math.max(x0 + 1, Math.floor((x + 1) * image.width / width));
                const sum = [0, 0, 0];
                for (let sy = y0; sy < y1; sy++) for (let sx = x0; sx < x1; sx++) {
                  const p = (sy * image.width + sx) * image.colors;
                  for (let c = 0; c < 3; c++) sum[c] += image.data[p + c];
                }
                for (let c = 0; c < 3; c++) data[(y * width + x) * 3 + c] = Math.round(sum[c] / ((y1 - y0) * (x1 - x0)));
              }
            }
            const bytes = new Uint8Array(32 + data.byteLength), view = new DataView(bytes.buffer);
            for (const [i, c] of [..."FJLRGB16"].entries()) bytes[i] = c.charCodeAt(0);
            for (const [off, val] of [[8, 1], [10, 32], [20, 3], [22, 16]]) view.setUint16(off, val, true);
            for (const [off, val] of [[12, width], [16, height], [24, data.byteLength], [28, options.outputColor === 1 ? 3 : 1]]) view.setUint32(off, val, true);
            new Uint16Array(bytes.buffer, 32).set(data);
            const metadata = await decoder.metadata(true);
            const base64 = await new Promise((resolveData, reject) => {
              const r = new FileReader(); r.onload = () => resolveData(String(r.result).split(",")[1]); r.onerror = reject; r.readAsDataURL(new Blob([bytes]));
            });
            return { base64, width, height, colors: image.colors, native_samples: nativeSamples,
              returned_metadata: { cam_mul: metadata.color_data?.cam_mul, pre_mul: metadata.color_data?.pre_mul, rgb_cam: metadata.color_data?.rgb_cam, fujiWB: metadata.fuji?.WB_Preset } };
          } finally { decoder.dispose(); }
        }, { name: scene.name, options });
        const payload = Buffer.from(result.base64, "base64");
        if (arm === "D2") {
          assert.equal(sha(payload), scene.old_camera_D_sha256, "Fresh old-gamma decode does not reproduce original D");
          assert.equal(sha(JSON.stringify(options)), scene.old_camera_D_declared_options_sha256, "Old-gamma declared options differ");
        }
        const dest = resolve(outDir, `${scene.name}-${arm}.fjlrg`);
        await writeFile(dest, payload, { flag: "wx" });
        const { base64, ...info } = result;
        row.arms[arm] = { ...info, file: relative(root, dest), sha256: sha(payload), declared_options: options,
          declared_options_sha256: sha(JSON.stringify(options)), note: "Declared options, not internal applied-parameter readback" };
        console.log(`${report.scenes.length + 1}/1 ${scene.name} ${arm} ${result.width}x${result.height}`, { hash: sha(payload) });
      } finally { await page.close(); }
    }
    assert.equal(await hashFile(allowed.get(scene.name)), scene.raf_sha256);
    report.scenes.push(row);
    // Incremental evidence; completion marker only after all three arms.
    await writeFile(resolve(outDir, "decode-progress.json"), JSON.stringify(report, null, 2) + "\n");
  }
  report.completed = true;
  await writeFile(resolve(outDir, "decode-complete.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
} finally { await browser.close(); await new Promise(r => server.close(r)); }
