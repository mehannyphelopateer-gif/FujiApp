#!/usr/bin/env node
// Bakes fragmentShader.ts's OLD parametric applyColorChrome formula into 4
// placeholder Hald CLUTs (color-chrome/{weak,strong}.png,
// fx-blue/{weak,strong}.png) — the exact same bootstrap-before-real-data
// role scripts/generate-placeholder-luts.mjs played for the film-simulation
// LUTs before Phase 1/2's real camera calibration existed. Run this ONCE,
// before the fragmentShader.ts/useWebGLRenderer.ts Phase 3 wiring lands, so
// switching from the parametric formula to a LUT-based Color Chrome/FX Blue
// pipeline is a zero-behavior-change no-op. Run
// scripts/derive-color-chrome-luts.mjs afterward (once a real Phase 3
// calibration shoot exists) to overwrite these with real camera-calibrated
// versions — see ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section.
//
// This does NOT need to be rerun as part of normal development — it's a
// one-time bootstrap, kept only so the exact formula being replaced stays
// reproducible/auditable rather than living only in a deleted git diff.

import { mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { writeLutPng } from "./lib/hald-clut-fitting.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const outputDir = process.argv[2] ?? join(__dirname, "..", "public", "luts");

function luma(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// Verbatim port of fragmentShader.ts's old applyColorChrome(color, strength, hueWeight).
function applyColorChrome(r, g, b, strength, hueWeight) {
  const lum = luma(r, g, b);
  const maxC = Math.max(r, g, b);
  const minC = Math.min(r, g, b);
  const sat = maxC - minC;

  const amount = sat * hueWeight * strength;
  const lumaCompressed = lum - amount * 0.15;
  const factor = 1 + amount * 0.35;
  const boosted = [r, g, b].map((c) => lum + (c - lum) * factor);
  const lumaBoosted = Math.max(luma(boosted[0], boosted[1], boosted[2]), 0.0001);
  const scale = lumaCompressed / lumaBoosted;
  return boosted.map((c) => Math.min(1, Math.max(0, c * scale)));
}

// Verbatim port of fragmentShader.ts's warm/blue hue-weight formulas from main().
function warmWeight(r, g, b) {
  return Math.min(1, Math.max(0, r + g * 0.5 - b));
}
function blueWeight(r, g, b) {
  return Math.min(1, Math.max(0, b - Math.max(r, g) * 0.5));
}

const TARGETS = [
  { outPath: ["color-chrome", "weak.png"], strength: 0.5, weightFn: warmWeight },
  { outPath: ["color-chrome", "strong.png"], strength: 1, weightFn: warmWeight },
  { outPath: ["fx-blue", "weak.png"], strength: 0.5, weightFn: blueWeight },
  { outPath: ["fx-blue", "strong.png"], strength: 1, weightFn: blueWeight },
];

for (const { outPath, strength, weightFn } of TARGETS) {
  const outFile = join(outputDir, ...outPath);
  mkdirSync(dirname(outFile), { recursive: true });
  writeLutPng((r, g, b) => applyColorChrome(r, g, b, strength, weightFn(r, g, b)), outFile);
  console.log(`wrote ${outFile}`);
}
