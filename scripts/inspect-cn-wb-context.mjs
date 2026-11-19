#!/usr/bin/env node
/** F-only LibRaw camera-WB metadata probe. No demosaic, fitting or targets.
 * Serves three predeclared F RAFs (or the fixed F24 with --all-fit) and libraw assets on
 * loopback, then closes Chromium and the server. Source files stay read-only.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { chromium } from "playwright";

const cohortPath = "training/diagnostics/cuban-cohort-proposal.json";
const cohortBytes = await readFile(cohortPath);
const cohort = JSON.parse(cohortBytes).scenes;
assert(process.argv.slice(2).every(arg => arg === "--all-fit"), "Unknown argument");
const allFit = process.argv.includes("--all-fit");
const names = allFit
  ? cohort.filter(s => s.role === "fit" && s.tier === "train").map(s => s.name).sort()
  : ["DSCF0523.RAF", "DSCF0964.RAF", "DSCF0514.RAF"];
assert.equal(names.length, allFit ? 24 : 3);
const allowed = new Map(names.map(name => {
  const row = cohort.find(s => s.name === name && s.role === "fit" && s.tier === "train");
  assert(row, "Only approved F scenes may be opened");
  return [name, resolve("calibration-input", row.shoot, name)];
}));
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const sourceHashes = {};
for (const [name, path] of allowed) {
  const h = createHash("sha256");
  for await (const chunk of createReadStream(path)) h.update(chunk);
  sourceHashes[name] = h.digest("hex");
}
const lib = resolve("node_modules/libraw-wasm/dist");
const version = JSON.parse(await readFile("node_modules/libraw-wasm/package.json", "utf8")).version;
assert.equal(version, "1.6.0");
const server = createServer(async (req, res) => {
  try {
    const u = new URL(req.url, "http://localhost");
    if (u.pathname === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end('<!doctype html><link rel="icon" href="data:,">');
      return;
    }
    let path;
    if (u.pathname.startsWith("/libraw/")) {
      const n = decodeURIComponent(u.pathname.slice(8));
      assert(n === basename(n) && !n.includes("..") && !n.includes("\\"));
      path = resolve(lib, n);
    } else if (u.pathname.startsWith("/raw/")) {
      path = allowed.get(decodeURIComponent(u.pathname.slice(5)));
      assert(path, "RAF is not in the F fixture allowlist");
    } else throw new Error("Unknown route");
    const info = await stat(path);
    const type = extname(path) === ".js" ? "text/javascript"
      : extname(path) === ".wasm" ? "application/wasm" : "application/octet-stream";
    res.writeHead(200, { "Content-Length": info.size, "Content-Type": type });
    createReadStream(path).pipe(res);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  page.on("console", msg => { if (msg.text().startsWith("WBPROBE ")) console.log(msg.text()); });
  const scenes = await page.evaluate(async names => {
    const { default: LibRaw } = await import("/libraw/index.js");
    const out = [];
    for (const name of names) {
      const decoder = new LibRaw();
      try {
        const response = await fetch(`/raw/${name}`);
        if (!response.ok) throw new Error("Missing F RAF");
        const bytes = new Uint8Array(await response.arrayBuffer());
        await decoder.open(bytes, {
          halfSize: false, outputBps: 16, outputColor: 1, gamm: [1, 1],
          noAutoBright: true, adjustMaximumThr: 0, useCameraMatrix: 1,
          useCameraWb: true, useAutoWb: false, highlight: 0,
          userQual: 3, useFujiRotate: -1, fbddNoiserd: 0,
        });
        const m = await decoder.metadata(true);
        out.push({ name, cam_mul: m.color_data?.cam_mul,
          pre_mul: m.color_data?.pre_mul, fujiWB: m.fuji?.WB_Preset });
        console.log(`WBPROBE ${out.length}/${names.length} ${name}`);
        // Intentionally never call imageData() or rawImageData().
      } finally { decoder.dispose(); }
    }
    return out;
  }, names);
  for (const scene of scenes) {
    assert(Array.isArray(scene.cam_mul) && scene.cam_mul.length === 4);
    scene.source_raf_sha256 = sourceHashes[scene.name];
  }
  const record = {
    scope: "F-only LibRaw WB metadata; no demosaic, target scoring or fitting",
    decode: "canonical phase3 options; imageData never called",
    libraw_wasm_version: version, cohort_sha256: sha(cohortBytes), scenes,
  };
  const output = allFit ? "training/diagnostics/cn-step1-wb-input-probe-all-F.json"
    : "training/diagnostics/cn-step1-wb-input-probe.json";
  await writeFile(output, `${JSON.stringify(record, null, 2)}\n`);
  console.log(JSON.stringify(record, null, 2));
} finally {
  await browser.close();
  await new Promise(r => server.close(r));
}
