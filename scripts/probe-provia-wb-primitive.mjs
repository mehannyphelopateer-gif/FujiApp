#!/usr/bin/env node
/**
 * WB primitive gate v2 — research-only decode of the frozen F11 source-Auto subset.
 *
 * For each scene, two decodes that differ ONLY in userMul (source 0xf00e as-shot
 * vs source 0xf00d Auto), both with six-element linear gamma, camera RGB
 * (outputColor 0) and noInterpolation, so imageData() should be an unmixed
 * native CFA mosaic. No demosaic, no area average, no image export. Only
 * structure statistics and a fixed native sample grid are saved, to the
 * ignored output directory. Python (training/eval_provia_wb_primitive.py)
 * evaluates. A structural failure stops the run immediately.
 *
 * Usage: node scripts/probe-provia-wb-primitive.mjs <committed-manifest-sha256>
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile, mkdir } from "node:fs/promises";
import { basename, resolve, relative, extname } from "node:path";
import { chromium } from "playwright";

const root = process.cwd();
const manifestRel = "training/diagnostics/provia-wb-primitive-v2-manifest.json";
const ownCode = ["scripts/probe-provia-wb-primitive.mjs", "training/eval_provia_wb_primitive.py"];
const args = process.argv.slice(2);
assert.equal(args.length, 1, "Pass the committed manifest SHA256");

const sha = b => createHash("sha256").update(b).digest("hex");
const hashFile = async p => { const h = createHash("sha256"); for await (const b of createReadStream(p)) h.update(b); return h.digest("hex"); };

const manifestBytes = await readFile(manifestRel);
assert.equal(sha(manifestBytes), args[0], "Manifest drift");
const manifest = JSON.parse(manifestBytes);
for (const [p, h] of Object.entries(manifest.bindings)) assert.equal(await hashFile(p), h, `Bound file drift: ${p}`);
for (const p of [manifestRel, ...ownCode])
  assert.equal(sha(execFileSync("git", ["show", `HEAD:${p}`])), await hashFile(p), `Uncommitted research code: ${p}`);

// Frozen option invariants: only userMul differs between arms.
const base = manifest.base_options;
assert.deepEqual(base.gamm, [1, 1, 0, 0, 0, 0], "Six-element linear gamma required");
assert.equal(base.outputColor, 0, "Camera RGB required");
assert.equal(base.noInterpolation, true, "noInterpolation required");
assert.equal(base.useCameraWb, false, "Camera WB must be off");
assert.equal(base.useAutoWb, false, "Auto WB must be off");
assert.equal(base.highlight, 0, "Highlight clip mode required");
assert(!("userMul" in base), "userMul is per arm, not in base options");

const grid = manifest.sample_grid;
assert.deepEqual([grid.offset, grid.stride], [15, 31], "Frozen sample grid");

const outDir = resolve(root, manifest.output_dir);
assert.equal(outDir, resolve(root, "training/checkpoints/provia-wb-primitive-v2"));
await mkdir(outDir); // Refuses an existing directory: never overwrite evidence.

const allowed = new Map(manifest.scenes.map(s => [s.name, resolve(root, "calibration-input", s.shoot, s.name)]));
assert.equal(allowed.size, 11, "Exactly the frozen F11 source-Auto subset");
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
const report = { manifest_sha256: args[0], scope: manifest.scope, scenes: [], completed: false };
const progressPath = resolve(outDir, "decode-progress.json");

try {
  for (const scene of manifest.scenes) {
    assert.equal(await hashFile(allowed.get(scene.name)), scene.raf_sha256, `RAF changed before decode: ${scene.name}`);
    const row = { name: scene.name, arms: {} };
    for (const arm of ["D", "E"]) {
      const options = { ...base, userMul: arm === "E" ? scene.auto_rgbg : scene.as_shot_rgbg };
      const page = await browser.newPage();
      try {
        await page.goto(`http://127.0.0.1:${server.address().port}`);
        const result = await page.evaluate(async ({ name, options, grid }) => {
          const { default: LibRaw } = await import("/libraw/index.js");
          const decoder = new LibRaw();
          try {
            const response = await fetch(`/raw/${name}`);
            if (!response.ok) throw new Error("Missing allowlisted RAF");
            await decoder.open(new Uint8Array(await response.arrayBuffer()), options);
            const image = await decoder.imageData();
            const shape = { width: image?.width, height: image?.height, colors: image?.colors, bits: image?.bits };
            if (!image || image.bits !== 16 || image.colors !== 3 || !(image.data instanceof Uint16Array))
              return { structure_error: "Expected 16-bit, 3-colour imageData", shape };
            const { width, height, data } = image;
            // Per CFA phase (row mod 6, col mod 6): counts of single-active pixels by channel, plus all-zero pixels.
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
            // Fixed native grid, all three channels, BEFORE anything else touches the data.
            const samples = [];
            for (let y = grid.offset; y < height; y += grid.stride)
              for (let x = grid.offset; x < width; x += grid.stride) {
                const p = (y * width + x) * 3;
                samples.push(data[p], data[p + 1], data[p + 2]);
              }
            const bytes = new Uint8Array(new Uint16Array(samples).buffer);
            const base64 = await new Promise((ok, fail) => {
              const r = new FileReader(); r.onload = () => ok(String(r.result).split(",")[1]); r.onerror = fail; r.readAsDataURL(new Blob([bytes]));
            });
            const metadata = await decoder.metadata(true);
            return { shape, mixed_pixels: mixed, phase_counts: Array.from(phase), samples_base64: base64,
              returned_metadata: { cam_mul: metadata.color_data?.cam_mul, pre_mul: metadata.color_data?.pre_mul } };
          } finally { decoder.dispose(); }
        }, { name: scene.name, options, grid });

        const info = { shape: result.shape, declared_options: options, declared_options_sha256: sha(JSON.stringify(options)),
          note: "Declared options, not internal applied-parameter readback" };
        if (!result.structure_error && result.mixed_pixels === 0) {
          const pattern = [];
          for (let k = 0; k < 36; k++) {
            const active = [0, 1, 2].filter(c => result.phase_counts[k * 4 + c] > 0);
            if (active.length !== 1) { result.structure_error = `CFA phase ${k} has ${active.length} active channels`; break; }
            pattern.push(active[0]);
          }
          if (!result.structure_error && new Set(pattern).size !== 3) result.structure_error = "CFA does not contain all RGB channels";
          if (arm === "E" && !result.structure_error) {
            if (result.shape.width !== row.arms.D.shape.width || result.shape.height !== row.arms.D.shape.height)
              result.structure_error = "Arm native dimensions differ";
            if (JSON.stringify(pattern) !== JSON.stringify(row.arms.D.inferred_pattern)) result.structure_error = "Arm CFA patterns differ";
          }
          result.inferred_pattern = pattern;
        }
        if (result.structure_error || result.mixed_pixels !== 0) {
          row.arms[arm] = { ...info, structure_error: result.structure_error ?? `Mixed pixels: ${result.mixed_pixels}` };
          report.scenes.push(row);
          report.stopped = { scene: scene.name, arm, reason: row.arms[arm].structure_error };
          await writeFile(progressPath, JSON.stringify(report, null, 2) + "\n");
          throw new Error(`STRUCTURE STOP ${scene.name} ${arm}: ${row.arms[arm].structure_error}`);
        }
        const samples = Buffer.from(result.samples_base64, "base64");
        const dest = resolve(outDir, `${scene.name}-${arm}.samples.u16le`);
        await writeFile(dest, samples, { flag: "wx" });
        row.arms[arm] = { ...info, mixed_pixels: 0, phase_counts: result.phase_counts, inferred_pattern: result.inferred_pattern, returned_metadata: result.returned_metadata,
          samples_file: relative(root, dest), samples_sha256: sha(samples), samples_layout: "uint16-le, row-major grid, 3 channels per site" };
        console.log(`${report.scenes.length + 1}/11 ${scene.name} ${arm} ${result.shape.width}x${result.shape.height}`);
      } finally { await page.close(); }
    }
    assert.equal(await hashFile(allowed.get(scene.name)), scene.raf_sha256, `RAF changed during decode: ${scene.name}`);
    report.scenes.push(row);
    await writeFile(progressPath, JSON.stringify(report, null, 2) + "\n");
  }
  report.completed = true;
  await writeFile(resolve(outDir, "decode-complete.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
} finally {
  await browser.close();
  await new Promise(r => server.close(r));
}
