#!/usr/bin/env node
/** Integration smoke/parity on explicit train/monitor fixtures only. No training/held-out access.
 * Start Vite, then: FUJIAPP_URL=http://127.0.0.1:5182 node scripts/verify-round11-app.mjs
 * Native reference vectors: python3 training/export_round11_onnx.py
 */
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { PNG } from "pngjs";

const root = process.cwd();
const manifest = JSON.parse(await readFile(resolve(root, "calibration-input/phase4-corpus-manifest.json"), "utf8"));
const approved = [...manifest.tiers.train.scenes, ...manifest.tiers.monitor.scenes];
const names = ["DSCF1419.RAF", "DSCF0878.RAF", "DSCF0538.RAF"];
const native = JSON.parse(await readFile(resolve(root, "training/checkpoints/round11-onnx/native-report.json"), "utf8"));
const vectors = native.models.find(m => m.seed === 42);
assert(vectors && native.passed);
const appUrl = process.env.FUJIAPP_URL ?? "http://127.0.0.1:5182";
const built = process.env.FUJIAPP_BUILT === "1";
const forceWasm = process.env.FUJIAPP_FORCE_WASM === "1";
const browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", ...(process.platform === "darwin" ? ["--use-angle=metal"] : [])] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
if (forceWasm) await page.addInitScript(() => { Object.defineProperty(navigator, "gpu", { get: () => undefined }); });
page.setDefaultTimeout(180_000);
const pageErrors = [];
page.on("pageerror", error => pageErrors.push(error.message));
const results = [];
try {
  await page.goto(appUrl, { waitUntil: "networkidle", timeout: 120_000 });
  await page.getByRole("button", { name: "Experimental Provia", exact: true }).click();
  await page.getByRole("heading", { name: "Experimental RAW processor" }).waitFor();
  await page.evaluate(() => { const input = document.createElement("input"); input.type = "file"; input.multiple = true; input.id = "integration-fixtures"; document.body.append(input); });

  for (const [index, name] of (built ? [] : names).entries()) {
    const scene = approved.find(s => s.name === name);
    assert(scene, `${name} must be in train/monitor`);
    const folder = resolve(root, "calibration-input", scene.shoot);
    const expected = vectors.native_cases.find(c => c.name === name.replace(/\.RAF$/i, ""));
    await page.locator("#integration-fixtures").setInputFiles([
      resolve(folder, name), resolve(folder, "browser-phase3-linear.fjlrg"),
      resolve(root, "training/checkpoints/round11-onnx", expected.expected),
    ]);
    const result = await page.evaluate(async ({ expectedDr, decode }) => {
      const files = document.querySelector("#integration-fixtures").files;
      const raw = await import("/src/lib/raw/rawService.ts");
      const metadata = await import("/src/lib/exif/parseFujiMakerNotes.ts");
      const processor = await import("/src/lib/raw/round11Processor.ts");
      const capture = await metadata.extractProcessorCaptureMetadata(await raw.extractRafPreviewJpeg(files[0]));
      if (capture.captureDynamicRange !== expectedDr) throw new Error("Capture DR differs from frozen manifest");
      const frame = await processor.readProcessorLinearFrame(files[1]);
      let rawDecodeMatches = null;
      if (decode) {
        // Calibration query flags deliberately attempt to change recovery;
        // the real app entry must still use the frozen mode 0.
        history.replaceState(null, "", "?rawCalibrationWb=camera&rawCalibrationHighlight=5");
        const decoded = await raw.decodeProcessorLinearRaf(files[0]);
        const [actual, original] = await Promise.all([decoded.blob.arrayBuffer(), files[1].arrayBuffer()]);
        const hash = async value => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", value)), x => x.toString(16).padStart(2, "0")).join("");
        rawDecodeMatches = await hash(actual) === await hash(original);
        if (!rawDecodeMatches) throw new Error("App RAW decode differs from calibration container");
        history.replaceState(null, "", "/");
      }
      const expected = new Float32Array(await files[2].arrayBuffer());
      const providers = [];
      for (const backend of ["wasm", "webgpu"]) {
        const output = await processor.renderRound11Frame(frame, capture.captureDynamicRange, backend);
        let max = 0, sum = 0;
        for (let i = 0; i < expected.length; i++) { const e = Math.abs(expected[i] - output.data[i]); max = Math.max(max, e); sum += e; }
        providers.push({ backend: output.backend, max_abs: max, mean_abs: sum / expected.length,
          passed: Number.isFinite(max) && Number.isFinite(sum) && max <= 5e-4 && sum / expected.length <= 5e-5 });
      }
      let invalidDrRejected = false;
      try { await processor.renderRound11Frame(frame, 0); } catch { invalidDrRejected = true; }
      if (!invalidDrRejected) throw new Error("Missing DR was silently accepted");
      return { capture, width: frame.width, height: frame.height, rawDecodeMatches, invalidDrRejected, providers };
    }, { expectedDr: scene.captureDynamicRange, decode: index === 0 });
    assert(result.providers.every(p => p.passed));
    results.push({ name, ...result });
    console.log(JSON.stringify(results.at(-1)));
  }

  const first = approved.find(s => s.name === names[0]);
  await page.getByLabel("Upload RAF for experimental processor").setInputFiles(resolve(root, "calibration-input", first.shoot, first.name));
  await page.getByRole("status").filter({ hasText: "Rendered locally." }).waitFor({ timeout: 180_000 });
  if (forceWasm) assert(await page.getByText(/Capture DR100 → Output DR100.*WASM/).isVisible());
  const canvas = page.getByLabel("Experimental Provia rendering");
  assert(await canvas.isVisible());
  assert(await page.getByRole("button", { name: "Save preview PNG" }).isEnabled());
  const downloadReady = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save preview PNG" }).click();
  const download = await downloadReady;
  assert.match(download.suggestedFilename(), /-experimental-provia-dr100\.png$/);
  const png = PNG.sync.read(await readFile(await download.path()));
  const firstReference = vectors.native_cases.find(c => c.name === names[0].replace(/\.RAF$/i, ""));
  const buffer = await readFile(resolve(root, "training/checkpoints/round11-onnx", firstReference.expected));
  const expected = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  const pixels = png.width * png.height;
  assert.equal(pixels * 3, expected.length);
  let pngMax = 0, pngSum = 0;
  for (let i = 0; i < pixels; i++) {
    assert.equal(png.data[i * 4 + 3], 255);
    for (let c = 0; c < 3; c++) {
      const linear = Math.min(1, Math.max(0, expected[c * pixels + i]));
      const srgb = linear <= .0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - .055;
      const error = Math.abs(png.data[i * 4 + c] - Math.round(255 * srgb));
      pngMax = Math.max(pngMax, error); pngSum += error;
    }
  }
  assert(pngMax <= 1 && pngSum / (pixels * 3) <= .01, "Displayed/exported PNG differs from frozen model");
  await page.screenshot({ path: resolve(root, "training/checkpoints/round11-app-preview.png") });
  await page.getByLabel("Upload RAF for experimental processor").setInputFiles({ name: "unsupported.jpg", mimeType: "image/jpeg", buffer: Buffer.from("not a RAF") });
  await page.getByRole("alert").filter({ hasText: "Choose an original X100VI RAF" }).waitFor();
  assert(!(await canvas.isVisible()), "Old render must disappear after invalid replacement");
  assert.equal(await page.getByRole("button", { name: "Save preview PNG" }).count(), 0);
  await page.getByRole("button", { name: "Standard Preview", exact: true }).click();
  assert(await page.getByText("Drop a JPEG or .RAF here", { exact: true }).isVisible());
  assert.deepEqual(pageErrors, []);
  await writeFile(resolve(root, `training/checkpoints/round11-app${built ? "-built" : ""}${forceWasm ? "-wasm" : ""}-report.json`), JSON.stringify({ passed: true, built, forceWasm, results,
    pngExport: { width: png.width, height: png.height, max_byte_error: pngMax, mean_byte_error: pngSum / (pixels * 3) },
    uiUploadPassed: true, invalidReplacementCleared: true, pageErrors }, null, 2));
  console.log("App integration: native-reference parity, metadata, frozen RAW decode, UI upload, invalid replacement, and standard-mode switch passed.");
} finally { await browser.close(); }
