#!/usr/bin/env node
/**
 * Writes Phase 3 browser calibration inputs (`browser-phase3-linear.fjlrg`)
 * for the training + monitoring range only. The final-test shoot range is
 * intentionally excluded from the default and guarded against accidentally
 * being requested here.
 *
 * Start the Vite app first, then run:
 *   node scripts/export-phase3-linear.mjs
 * Optionally set FUJIAPP_URL (default http://127.0.0.1:5173).
 *
 * For a separately approved calibration batch, set both:
 *   FUJIAPP_EXPORT_MANIFEST=calibration-input/phase4-capture2-manifest.json
 *   FUJIAPP_EXPORT_FILENAME=browser-phase4-capture2-linear.fjlrg
 *
 * The manifest may use either the original { shoot, name, tier } schema or
 * the capture-batch { shoot_folder, filename } schema.  Final-held-out rows
 * are rejected in either form.
 */
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";

const inputDir = resolve(process.cwd(), "calibration-input");
const appUrl = process.env.FUJIAPP_URL ?? "http://127.0.0.1:5173";
const manifestPath = resolve(process.cwd(), process.env.FUJIAPP_EXPORT_MANIFEST ?? "calibration-input/phase4-needs-browser-export.json");
const outputFilename = process.env.FUJIAPP_EXPORT_FILENAME ?? "browser-phase3-linear.fjlrg";
if (!/^[A-Za-z0-9._-]+\.fjlrg$/.test(outputFilename)) {
  throw new Error(`Invalid output filename: ${outputFilename}`);
}
const rawManifest = JSON.parse(readFileSync(manifestPath, "utf8"));
if (!Array.isArray(rawManifest) || rawManifest.length === 0) {
  throw new Error(`Expected a non-empty approved export list: ${manifestPath}`);
}
const approved = rawManifest.map((scene) => ({
  shoot: scene?.shoot ?? scene?.shoot_folder,
  name: scene?.name ?? scene?.filename,
  tier: scene?.tier,
}));
if (approved.some((scene) => !scene || !/^Shoot \d+$/.test(scene.shoot) || !/^[^/]+\.RAF$/i.test(scene.name) || scene.tier === "finalHeldOut")) {
  throw new Error("Approved export list is invalid or includes a locked final-held-out scene.");
}
const requested = new Set(process.argv.slice(2));
const scenes = requested.size === 0
  ? approved
  : approved.filter((scene) => requested.has(String(scene.shoot).replace("Shoot ", "")) || requested.has(scene.shoot));
if (requested.size > 0 && scenes.length !== requested.size) {
  throw new Error("A requested shoot is not in the approved Phase 4 train/monitor export list.");
}

const calibrationUrl = new URL(appUrl);
calibrationUrl.searchParams.set("rawCalibrationWb", "camera");
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
try {
  await page.goto(calibrationUrl.href, { waitUntil: "networkidle" });
  await page.evaluate(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.id = "phase3-linear-export-input";
    document.body.append(input);
  });

  for (const [index, scene] of scenes.entries()) {
    const folder = resolve(inputDir, scene.shoot);
    if (!existsSync(folder)) throw new Error(`Missing folder: ${folder}`);
    const rafs = readdirSync(folder).filter((name) => name === scene.name);
    if (rafs.length !== 1) throw new Error(`Expected ${scene.name} in ${scene.shoot}; found ${rafs.length}.`);
    const outputPath = resolve(folder, outputFilename);
    if (existsSync(outputPath)) {
      console.log(`skip ${scene.shoot} (${index + 1}/${scenes.length}): output already exists`);
      continue;
    }

    await page.locator("#phase3-linear-export-input").setInputFiles(resolve(folder, rafs[0]));
    const result = await page.evaluate(async () => {
      const input = document.querySelector("#phase3-linear-export-input");
      const file = input?.files?.[0];
      if (!file) throw new Error("Calibration RAF was not attached.");
      const { decodePhase3LinearRaf } = await import("/src/lib/raw/rawService.ts");
      const decoded = await decodePhase3LinearRaf(file);
      const dataUrl = await new Promise((resolveDataUrl, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolveDataUrl(reader.result);
        reader.onerror = () => reject(reader.error ?? new Error("Failed to read linear calibration export."));
        reader.readAsDataURL(decoded.blob);
      });
      return { width: decoded.width, height: decoded.height, base64: String(dataUrl).split(",")[1] };
    });
    if (!result.base64) throw new Error(`Phase 3 decode returned no binary payload for ${scene.shoot}.`);
    writeFileSync(outputPath, Buffer.from(result.base64, "base64"));
    console.log(`wrote ${scene.shoot} (${index + 1}/${scenes.length}): ${result.width}x${result.height}`);
  }
} finally {
  await browser.close();
}
