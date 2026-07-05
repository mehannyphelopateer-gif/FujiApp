#!/usr/bin/env node
// Generates placeholder Hald CLUT PNGs into public/luts/ so the WebGL LUT
// pipeline (load -> texture bind -> shader sample -> render) can be
// exercised end-to-end before real film-simulation LUTs are sourced.
//
// Pixel layout matches src/engine/webgl/shaders/fragmentShader.ts's
// apply3DLut(): a standard (non-tile-atlas) Hald CLUT, level 8, 64 levels
// per channel, 512x512 total. See the plan doc for the derivation.
//
// Usage: node scripts/generate-placeholder-luts.mjs

import { PNG } from "pngjs";
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = join(__dirname, "..", "public", "luts");

const LEVELS = 64; // levels per channel
const N = Math.sqrt(LEVELS); // 8 — rows per blue-slice band
const SIDE = LEVELS * N; // 512

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

/** @typedef {(r: number, g: number, b: number) => [number, number, number]} TintFn */

/** @type {Record<string, TintFn>} */
const PRESETS = {
  identity: (r, g, b) => [r, g, b],

  "classic-chrome": (r, g, b) => {
    // Mild fade toward luma + a faint yellow push, flattened highlights.
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const fade = 0.06;
    return [
      r * (1 - fade) + luma * fade + 0.01,
      g * (1 - fade) + luma * fade + 0.01,
      Math.max(0, b * (1 - fade) + luma * fade - 0.015),
    ];
  },

  "classic-negative": (r, g, b) => {
    // Slight contrast boost with a faint cool-shadow / warm-highlight split.
    const contrast = (v) => clamp01((v - 0.5) * 1.12 + 0.5);
    return [contrast(r) + 0.01, contrast(g), Math.max(0, contrast(b) - 0.015)];
  },

  "pro-neg-std": (r, g, b) => {
    // Gentle highlight roll-off, warm cast.
    const soften = (v) => Math.pow(v, 0.95);
    return [Math.min(1, soften(r) + 0.02), soften(g), Math.max(0, soften(b) - 0.02)];
  },

  velvia: (r, g, b) => {
    // Saturation + contrast boost.
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const sat = 1.25;
    const contrast = (v) => clamp01((v - 0.5) * 1.15 + 0.5);
    return [
      contrast(luma + (r - luma) * sat),
      contrast(luma + (g - luma) * sat),
      contrast(luma + (b - luma) * sat),
    ];
  },

  acros: (r, g, b) => {
    // Desaturated toward luma (monochrome-leaning placeholder).
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return [luma, luma, luma];
  },

  "nostalgic-neg": (r, g, b) => {
    // Amber/warm cast with slightly lifted shadows.
    const lift = 0.03;
    return [Math.min(1, r + lift + 0.02), Math.min(1, g + lift), Math.max(0, b + lift * 0.3 - 0.02)];
  },

  provia: (r, g, b) => {
    // Neutral reference profile — only a whisper of contrast, near-identity.
    const contrast = (v) => clamp01((v - 0.5) * 1.04 + 0.5);
    return [contrast(r), contrast(g), contrast(b)];
  },

  astia: (r, g, b) => {
    // Soft, low-contrast, flattering — gentle desaturation with a warm mid-tone push.
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const desat = 0.12;
    const contrast = (v) => clamp01((v - 0.5) * 0.94 + 0.5);
    return [
      contrast(r * (1 - desat) + luma * desat + 0.008),
      contrast(g * (1 - desat) + luma * desat),
      contrast(b * (1 - desat) + luma * desat),
    ];
  },

  "pro-neg-hi": (r, g, b) => {
    // Portrait profile, more contrast than Pro Neg Std, natural warm skin tones.
    const contrast = (v) => clamp01((v - 0.5) * 1.1 + 0.5);
    return [Math.min(1, contrast(r) + 0.015), contrast(g), Math.max(0, contrast(b) - 0.01)];
  },

  eterna: (r, g, b) => {
    // Flat, desaturated, cool cinema profile.
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const desat = 0.22;
    const contrast = (v) => clamp01((v - 0.5) * 0.85 + 0.5);
    return [
      contrast(r * (1 - desat) + luma * desat),
      contrast(g * (1 - desat) + luma * desat),
      Math.min(1, contrast(b * (1 - desat) + luma * desat) + 0.01),
    ];
  },

  "eterna-bleach-bypass": (r, g, b) => {
    // Heavily desaturated with a strong contrast boost — classic bleach-bypass look.
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const desat = 0.55;
    const contrast = (v) => clamp01((v - 0.5) * 1.3 + 0.5);
    return [
      contrast(r * (1 - desat) + luma * desat),
      contrast(g * (1 - desat) + luma * desat),
      contrast(b * (1 - desat) + luma * desat),
    ];
  },

  monochrome: (r, g, b) => {
    // Plain grayscale conversion.
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return [luma, luma, luma];
  },

  sepia: (r, g, b) => {
    // Grayscale then a warm brown tint.
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return [Math.min(1, luma * 1.07 + 0.05), Math.min(1, luma * 0.94 + 0.02), Math.max(0, luma * 0.7 - 0.02)];
  },

  "reala-ace": (r, g, b) => {
    // Fujifilm's newer "true color" standard profile — a touch punchier than Provia.
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const sat = 1.08;
    const contrast = (v) => clamp01((v - 0.5) * 1.07 + 0.5);
    return [
      contrast(luma + (r - luma) * sat),
      contrast(luma + (g - luma) * sat),
      contrast(luma + (b - luma) * sat),
    ];
  },
};

function generateLut(tint) {
  const png = new PNG({ width: SIDE, height: SIDE });

  for (let y = 0; y < SIDE; y++) {
    for (let x = 0; x < SIDE; x++) {
      // Invert the pixel position back to (r, g, b) level indices — see
      // the fragment shader's haldUV() for the forward mapping this mirrors.
      const b = Math.floor(y / N);
      const gHi = y % N;
      const gLo = Math.floor(x / LEVELS);
      const r = x % LEVELS;
      const g = gHi * N + gLo;

      const [tr, tg, tb] = tint(r / (LEVELS - 1), g / (LEVELS - 1), b / (LEVELS - 1));

      const idx = (SIDE * y + x) << 2;
      png.data[idx] = Math.round(clamp01(tr) * 255);
      png.data[idx + 1] = Math.round(clamp01(tg) * 255);
      png.data[idx + 2] = Math.round(clamp01(tb) * 255);
      png.data[idx + 3] = 255;
    }
  }

  return png;
}

mkdirSync(OUTPUT_DIR, { recursive: true });

for (const [name, tint] of Object.entries(PRESETS)) {
  const png = generateLut(tint);
  const outPath = join(OUTPUT_DIR, `${name}.png`);
  writeFileSync(outPath, PNG.sync.write(png));
  console.log(`Generated ${outPath}`);
}
