#!/usr/bin/env node
// Generates the PWA icon set (standard + maskable + apple-touch-icon) as a
// simple programmatic aperture/lens mark, since there's no design asset yet.
// Swap these files for real branding whenever it's sourced — same paths,
// no manifest/code changes needed as long as sizes stay the same.

import { PNG } from "pngjs";
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = join(__dirname, "..", "public", "icons");
mkdirSync(OUTPUT_DIR, { recursive: true });

const BG = [10, 10, 10]; // neutral-950
const ACCENT = [16, 185, 129]; // emerald-500

function setPixel(png, x, y, [r, g, b], alpha = 255) {
  if (x < 0 || y < 0 || x >= png.width || y >= png.height) return;
  const idx = (png.width * y + x) << 2;
  png.data[idx] = r;
  png.data[idx + 1] = g;
  png.data[idx + 2] = b;
  png.data[idx + 3] = alpha;
}

/** Simple aperture/lens mark: an accent ring with 6 blade notches around a dark center. */
function drawApertureIcon(size, { safeZoneScale = 1 } = {}) {
  const png = new PNG({ width: size, height: size });
  const cx = size / 2;
  const cy = size / 2;
  const outerR = size * 0.38 * safeZoneScale;
  const innerR = size * 0.27 * safeZoneScale;
  const bladeCount = 6;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const angle = Math.atan2(dy, dx);

      let color = BG;
      if (dist <= outerR && dist >= innerR) {
        // Cut blade-shaped notches out of the ring for an aperture look.
        const bladeAngle = ((angle + Math.PI) / (2 * Math.PI)) * bladeCount;
        const withinBlade = bladeAngle - Math.floor(bladeAngle) < 0.55;
        color = withinBlade ? ACCENT : BG;
      }
      setPixel(png, x, y, color);
    }
  }
  return png;
}

const targets = [
  { name: "icon-192.png", size: 192 },
  { name: "icon-512.png", size: 512 },
  { name: "icon-512-maskable.png", size: 512, safeZoneScale: 0.8 }, // keep content in the safe zone
  { name: "apple-touch-icon.png", size: 180 },
];

for (const { name, size, safeZoneScale } of targets) {
  const png = drawApertureIcon(size, { safeZoneScale });
  const outPath = join(OUTPUT_DIR, name);
  writeFileSync(outPath, PNG.sync.write(png));
  console.log(`Generated ${outPath}`);
}
