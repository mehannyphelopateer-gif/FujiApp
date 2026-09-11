#!/usr/bin/env node
// Manifest-driven staging guard, added after a real incident (2026-09-10):
// 40 locked held-out RAFs (Shoots 387-426) were manually added to
// phase3-batch-3/ and its missing-28/ subfolder — outside of any script,
// while "filling up" a batch to 100 — and got opened/reviewed in X RAW
// Studio during export. The browser exporter and fitting pipeline both
// correctly hard-excluded that range in code; this incident happened at
// the manual staging-folder step, which no script guarded at all.
//
// This script checks EVERY file currently sitting in any
// calibration-input/phase3-batch-* directory (recursively, including
// subfolders like the missing-28 one from the incident) against the
// locked corpus manifest, and fails loudly if it finds anything from the
// held-out range. Run this BEFORE handing a staging folder to X RAW
// Studio or to the browser exporter — it catches a manual mistake at any
// point, not just at the moment a folder-creation script runs (which is
// exactly where the incident's guard would have been too late).
//
// Usage: node scripts/verify-phase3-staging.mjs [input-dir]

import { readdirSync, statSync } from "node:fs";
import { join, basename, extname } from "node:path";
import { readFileSync } from "node:fs";

const inputDir = process.argv[2] ?? join(new URL(".", import.meta.url).pathname, "..", "calibration-input");
const manifestPath = join(inputDir, "phase3-corpus-manifest.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));

const heldOutStems = new Set(manifest.heldOutShoots.map((i) => basename(i.fileName, extname(i.fileName))));
const trainValStems = new Set(
  [...manifest.trainShoots, ...manifest.validationShoots].map((i) => basename(i.fileName, extname(i.fileName))),
);

function walk(dir, out) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else out.push(full);
  }
}

const stagingDirs = readdirSync(inputDir, { withFileTypes: true })
  .filter((d) => d.isDirectory() && /^phase3-batch-/.test(d.name))
  .map((d) => join(inputDir, d.name));

if (stagingDirs.length === 0) {
  console.log("No phase3-batch-* staging directories found — nothing to check.");
  process.exit(0);
}

let violations = [];
let checked = 0;
for (const dir of stagingDirs) {
  const files = [];
  walk(dir, files);
  for (const file of files) {
    const stem = basename(file, extname(file));
    if (heldOutStems.has(stem)) {
      violations.push(file);
    } else if (!trainValStems.has(stem) && /\.(raf|tif|tiff|jpg|jpeg|fjlrg)$/i.test(file)) {
      // Not held-out, but also not a recognized training/validation scene
      // for a calibration-relevant file type — worth knowing about even
      // though it's not the specific held-out violation this script exists
      // to catch, since it means the staging folder has something the
      // corpus selection never intended.
      console.warn(`NOTE: ${file} is not in the training/validation corpus and not held-out either — unexpected file, not a held-out violation.`);
    }
    checked++;
  }
}

console.log(`Checked ${checked} files across ${stagingDirs.length} staging director${stagingDirs.length === 1 ? "y" : "ies"}.`);

if (violations.length > 0) {
  console.error(`\n🛑 FINAL HOLD-OUT — DO NOT EXPORT OR REVIEW 🛑`);
  console.error(`${violations.length} file(s) in staging belong to the LOCKED held-out range and must not be exported or opened:`);
  for (const v of violations) console.error(`  ${v}`);
  console.error(`\nRemove these before proceeding. See docs/phase3-export-protocol.md.`);
  process.exit(1);
}

console.log("PASSED — no held-out files found in any staging directory. Safe to proceed with export.");
