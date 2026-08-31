#!/usr/bin/env node
// Converts third-party Hald CLUT PNGs (any level) into this app's exact
// format (level 8, 64 levels/channel, 512x512 — see
// src/engine/webgl/shaders/fragmentShader.ts's haldUV()/apply3DLut() and
// scripts/generate-placeholder-luts.mjs) via trilinear resampling.
//
// Used for scripts/../public/luts/NOTICE.md's abpy/FujifilmCameraProfiles
// import (CC-BY-NC-SA 4.0) — those ship as level-6 (36 levels/channel,
// 216x216) Hald CLUTs, a different resolution than this app uses.
//
// Usage: node scripts/convert-third-party-luts.mjs <source.png> <dest.png>

import { PNG } from "pngjs";
import { readFileSync, writeFileSync } from "node:fs";

const TARGET_LEVELS = 64;
const TARGET_N = Math.sqrt(TARGET_LEVELS); // 8
const TARGET_SIDE = TARGET_LEVELS * TARGET_N; // 512

function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}

/** Reads an arbitrary-resolution Hald CLUT PNG, returning a sampler `(r,g,b) => [r,g,b]` (inputs/outputs 0..1). */
function loadHaldSampler(path) {
  const png = PNG.sync.read(readFileSync(path));
  const side = png.width;
  // side = levels^1.5 for a standard (non-tiled) Hald CLUT.
  const levels = Math.round(Math.cbrt(side) ** 2);
  const n = Math.round(Math.sqrt(levels));
  if (n * n !== levels || levels * n !== side) {
    throw new Error(`${path}: ${side}x${side} doesn't look like a standard Hald CLUT (expected side = levels^1.5).`);
  }

  function at(rL, gL, bL) {
    const gLo = gL % n;
    const gHi = Math.floor(gL / n);
    const x = rL + levels * gLo;
    const y = bL * n + gHi;
    const idx = (png.width * y + x) << 2;
    return [png.data[idx] / 255, png.data[idx + 1] / 255, png.data[idx + 2] / 255];
  }

  return function sample(r, g, b) {
    const pos = [r, g, b].map((v) => clamp01(v) * (levels - 1));
    const lo = pos.map(Math.floor);
    const frac = pos.map((v, i) => v - lo[i]);
    const hi = lo.map((v) => Math.min(levels - 1, v + 1));

    function lerp3(a, b2, t) {
      return [0, 1, 2].map((i) => a[i] * (1 - t) + b2[i] * t);
    }

    const c000 = at(lo[0], lo[1], lo[2]);
    const c100 = at(hi[0], lo[1], lo[2]);
    const c010 = at(lo[0], hi[1], lo[2]);
    const c110 = at(hi[0], hi[1], lo[2]);
    const c001 = at(lo[0], lo[1], hi[2]);
    const c101 = at(hi[0], lo[1], hi[2]);
    const c011 = at(lo[0], hi[1], hi[2]);
    const c111 = at(hi[0], hi[1], hi[2]);

    const c00 = lerp3(c000, c100, frac[0]);
    const c10 = lerp3(c010, c110, frac[0]);
    const c01 = lerp3(c001, c101, frac[0]);
    const c11 = lerp3(c011, c111, frac[0]);
    const c0 = lerp3(c00, c10, frac[1]);
    const c1 = lerp3(c01, c11, frac[1]);
    return lerp3(c0, c1, frac[2]);
  };
}

function writeLutPng(lookup, outPath) {
  const png = new PNG({ width: TARGET_SIDE, height: TARGET_SIDE });
  for (let y = 0; y < TARGET_SIDE; y++) {
    for (let x = 0; x < TARGET_SIDE; x++) {
      const b = Math.floor(y / TARGET_N);
      const gHi = y % TARGET_N;
      const gLo = Math.floor(x / TARGET_LEVELS);
      const r = x % TARGET_LEVELS;
      const g = gHi * TARGET_N + gLo;

      const [tr, tg, tb] = lookup(r / (TARGET_LEVELS - 1), g / (TARGET_LEVELS - 1), b / (TARGET_LEVELS - 1));

      const idx = (TARGET_SIDE * y + x) << 2;
      png.data[idx] = Math.round(clamp01(tr) * 255);
      png.data[idx + 1] = Math.round(clamp01(tg) * 255);
      png.data[idx + 2] = Math.round(clamp01(tb) * 255);
      png.data[idx + 3] = 255;
    }
  }
  writeFileSync(outPath, PNG.sync.write(png));
}

const [, , srcPath, destPath] = process.argv;
if (!srcPath || !destPath) {
  console.error("Usage: node scripts/convert-third-party-luts.mjs <source.png> <dest.png>");
  process.exit(1);
}

const sampler = loadHaldSampler(srcPath);
writeLutPng(sampler, destPath);
console.log(`Wrote ${destPath} (${TARGET_SIDE}x${TARGET_SIDE}, ${TARGET_LEVELS} levels/channel) from ${srcPath}.`);
