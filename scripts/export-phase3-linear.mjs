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
 */
import { existsSync, readdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";

const inputDir = resolve(process.cwd(), "calibration-input");
const appUrl = process.env.FUJIAPP_URL ?? "http://127.0.0.1:5173";
const firstTrainingShoot = 127;
const lastMonitoringShoot = 386;
const firstHeldOutShoot = 387;
const lastHeldOutShoot = 426;

const requested = process.argv.slice(2).map(Number).filter(Number.isInteger);
const shoots = requested.length > 0
  ? requested
  : Array.from({ length: lastMonitoringShoot - firstTrainingShoot + 1 }, (_, index) => firstTrainingShoot + index);

if (shoots.some((shoot) => shoot >= firstHeldOutShoot && shoot <= lastHeldOutShoot)) {
  throw new Error(`Refusing to export locked final-test Shoots ${firstHeldOutShoot}-${lastHeldOutShoot}.`);
}
if (shoots.some((shoot) => shoot < firstTrainingShoot || shoot > lastMonitoringShoot)) {
  throw new Error(`Only Phase 3 training/monitoring Shoots ${firstTrainingShoot}-${lastMonitoringShoot} are valid here.`);
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
try {
  await page.goto(appUrl, { waitUntil: "networkidle" });
  await page.evaluate(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.id = "phase3-linear-export-input";
    document.body.append(input);
  });

  for (const [index, shoot] of shoots.entries()) {
    const folder = resolve(inputDir, `Shoot ${shoot}`);
    if (!existsSync(folder)) throw new Error(`Missing folder: ${folder}`);
    const rafs = readdirSync(folder).filter((name) => /\.raf$/i.test(name));
    if (rafs.length !== 1) throw new Error(`Expected one RAF in Shoot ${shoot}; found ${rafs.length}.`);
    const outputPath = resolve(folder, "browser-phase3-linear.fjlrg");
    if (existsSync(outputPath)) {
      console.log(`skip ${shoot} (${index + 1}/${shoots.length}): output already exists`);
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
    if (!result.base64) throw new Error(`Phase 3 decode returned no binary payload for Shoot ${shoot}.`);
    writeFileSync(outputPath, Buffer.from(result.base64, "base64"));
    console.log(`wrote ${shoot} (${index + 1}/${shoots.length}): ${result.width}x${result.height}`);
  }
} finally {
  await browser.close();
}
