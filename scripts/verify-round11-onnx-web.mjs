#!/usr/bin/env node
/** Run the trained Round 11 graphs in Chromium WASM and WebGPU.
 * First: python3 training/export_round11_onnx.py
 * Then: node scripts/verify-round11-onnx-web.mjs --ort-dir /path/to/onnxruntime-web/dist
 * Generated weights/test vectors remain under gitignored training/checkpoints.
 * No app integration, calibration targets, training, or held-out access.
 */
import { createServer } from "node:http";
import { readFile, writeFile, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { resolve, extname, sep } from "node:path";
import { chromium } from "playwright";

const args = process.argv.slice(2);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const vectorDir = resolve(value("--vector-dir", "training/checkpoints/round11-onnx"));
const ortDirArg = value("--ort-dir", process.env.FUJIAPP_ORT_DIST);
if (!ortDirArg) throw new Error("Pass --ort-dir with an installed onnxruntime-web/dist directory.");
const ortDir = resolve(ortDirArg);
const report = JSON.parse(await readFile(resolve(vectorDir, "native-report.json"), "utf8"));
if (!report.passed) throw new Error("Native export parity gate must pass first.");
const runtimeVersion = JSON.parse(await readFile(resolve(ortDir, "../package.json"), "utf8")).version;
const webgpuLayout = value("--webgpu-layout", "NCHW");
if (!["NCHW", "NHWC"].includes(webgpuLayout)) throw new Error("Invalid WebGPU layout");
const MAX_ABS = 5e-4, MEAN_ABS = 5e-5;
const allowedVectors = new Set(["native-report.json", ...report.models.map(m => m.model),
  ...report.cases.map(c => c.input), ...report.models.flatMap(m => m.native_cases.flatMap(c => [c.expected, ...Object.values(c.stages ?? {})]))]);
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>Round 11 export verification</title>"); return;
    }
    const isRuntime = url.pathname.startsWith("/ort/");
    const name = decodeURIComponent(url.pathname.slice(isRuntime ? 5 : 6));
    if ((!isRuntime && (!url.pathname.startsWith("/data/") || !allowedVectors.has(name))) || name.includes(sep) || name.includes("..")) {
      res.writeHead(404); res.end(); return;
    }
    const file = resolve(isRuntime ? ortDir : vectorDir, name);
    const mime = { ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" }[extname(file)] ?? "application/octet-stream";
    const info = await stat(file);
    res.writeHead(200, { "Content-Type": mime, "Content-Length": info.size });
    createReadStream(file).pipe(res);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));
const baseUrl = `http://127.0.0.1:${server.address().port}`;
const chromiumArgs = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
  ...(process.platform === "darwin" ? ["--use-angle=metal"] : [])];
const browser = await chromium.launch({ headless: true, args: chromiumArgs });
const results = { runtimeVersion, browserVersion: browser.version(), chromiumArgs, webgpuLayout,
  tolerances: { max_abs: MAX_ABS, mean_abs: MEAN_ABS }, providers: [] };
try {
  for (const provider of ["wasm", "webgpu"]) {
    const page = await browser.newPage();
    page.setDefaultTimeout(300_000);
    const errors = [], warnings = [];
    const deadline = setTimeout(() => { void page.close().catch(() => {}); }, 180_000);
    page.on("pageerror", e => errors.push(e.message));
    page.on("console", msg => {
      // ORT's WASM logger writes severity-W messages through console.error.
      // Preserve them as warnings rather than misreporting successful shape-op
      // CPU assignments as runtime errors. Actual errors remain fatal upstream.
      const clean = msg.text().replace(/\u001b\[[0-9;]*m/g, "");
      if (msg.type() === "error") {
        if (/\[W:onnxruntime:/.test(clean)) warnings.push(clean);
        else errors.push(msg.text());
      } else if (msg.type() === "warning") warnings.push(msg.text());
      if (msg.text().startsWith("GATE ")) console.log(msg.text());
    });
    await page.goto(baseUrl);
    try {
      const result = await page.evaluate(async ({ provider, report, maxAbs, meanAbs, webgpuLayout }) => {
        const ort = await import("/ort/ort.webgpu.bundle.min.mjs");
        ort.env.wasm.wasmPaths = `${location.origin}/ort/`;
        ort.env.wasm.numThreads = 1;
        const getFloats = async filename => {
          const response = await fetch(`/data/${filename}`);
          if (!response.ok) throw new Error(`Failed to load ${filename}`);
          return new Float32Array(await response.arrayBuffer());
        };
        let adapterInfo = null;
        if (provider === "webgpu") {
          const adapter = await navigator.gpu?.requestAdapter();
          if (!adapter) throw new Error("No WebGPU adapter available");
          const info = adapter.info;
          adapterInfo = { vendor: info?.vendor, architecture: info?.architecture, device: info?.device,
                          description: info?.description,
                          isFallbackAdapter: info?.isFallbackAdapter ?? adapter.isFallbackAdapter };
        }
        const output = { provider, adapterInfo, models: [] };
        for (const model of report.models) {
          const start = performance.now();
          const session = await ort.InferenceSession.create(`/data/${model.model}`, {
            executionProviders: provider === "webgpu" ? [{ name: "webgpu", preferredLayout: webgpuLayout }] : [provider],
          });
          const entries = [];
          for (const vector of report.cases) {
            const input = await getFloats(vector.input);
            const expected = await getFloats(model.native_cases.find(c => c.name === vector.name).expected);
            const condition = vector.condition;
            if (![0, 0.5, 1].includes(condition)) throw new Error("Unknown capture DR; refusing to guess");
            const begin = performance.now();
            const result = await session.run({ linear_rgb: new ort.Tensor("float32", input, vector.shape),
              capture_dr_condition: new ort.Tensor("float32", new Float32Array([condition]), [1, 1]) });
            const actual = result.corrected_linear_rgb;
            if (actual.dims.some((n, i) => n !== vector.shape[i]) || actual.data.length !== expected.length) throw new Error("Output shape mismatch");
            let max = 0, sum = 0, finite = true;
            for (let i = 0; i < expected.length; i++) {
              const e = Math.abs(actual.data[i] - expected[i]);
              if (!Number.isFinite(e)) { finite = false; break; }
              max = Math.max(max, e); sum += e;
            }
            const mean = sum / expected.length;
            entries.push({ name: vector.name, shape: vector.shape, capture_dr: vector.capture_dr,
              max_abs: max, mean_abs: mean, elapsed_ms: performance.now() - begin,
              passed: finite && max <= maxAbs && mean <= meanAbs });
            const stages = model.native_cases.find(c => c.name === vector.name).stages;
            if (stages) {
              entries.at(-1).stages = {};
              for (const [name, filename] of Object.entries(stages)) {
                const target = await getFloats(filename);
                const data = result[name].data;
                let max = 0, sum = 0;
                for (let i = 0; i < data.length; i++) { const e = Math.abs(data[i] - target[i]); max = Math.max(max, e); sum += e; }
                entries.at(-1).stages[name] = { max_abs: max, mean_abs: sum / data.length };
              }
              console.log(`GATE stages ${provider} seed ${model.seed} ${vector.name}: ${JSON.stringify(entries.at(-1).stages)}`);
            }
            console.log(`GATE ${provider} seed ${model.seed} ${vector.name}: max=${max.toExponential(2)} mean=${mean.toExponential(2)}`);
          }
          output.models.push({ seed: model.seed, total_ms: performance.now() - start, cases: entries });
          await session.release();
        }
        return output;
      }, { provider, report, maxAbs: MAX_ABS, meanAbs: MEAN_ABS, webgpuLayout });
      results.providers.push({ ...result, consoleErrors: errors, consoleWarnings: warnings });
      for (const m of result.models) for (const c of m.cases)
        console.log(`${provider} seed ${m.seed} ${c.name}: max=${c.max_abs.toExponential(2)} mean=${c.mean_abs.toExponential(2)} pass=${c.passed}`);
    } catch (error) {
      results.providers.push({ provider, error: error.message, consoleErrors: errors, consoleWarnings: warnings });
      console.error(`${provider} failed: ${error.message}`);
    } finally { clearTimeout(deadline); await page.close(); }
  }
} finally {
  await browser.close();
  await new Promise(r => server.close(r));
}
results.passed = results.providers.length === 2 && results.providers.every(p => !p.error && p.models.every(m => m.cases.every(c => c.passed)));
await writeFile(resolve(vectorDir, "browser-report.json"), `${JSON.stringify(results, null, 2)}\n`);
if (!results.passed) process.exitCode = 1;
