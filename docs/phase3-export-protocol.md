# Phase 3 export protocol — LOCKED

This is the exact, versioned settings protocol for every RAF exported in
Phase 3, on both sides (browser and X RAW Studio). Locked means: if a
setting here needs to change partway through a 300-scene export effort,
that's a new protocol version, and every scene already exported under the
old version gets re-exported — not silently mixed. Drift risk is much
higher at 300 scenes than the ~80-scene batches in Phase 2, so this is
written down explicitly rather than repeated informally per-batch.

**Protocol version: 3.0** (first Phase 3 protocol — supersedes Phase 2's
as-shot protocol for all new Phase 3 exports; Phase 2's existing 119
scenes are NOT re-exported under this protocol, since Phase 3 is a new,
separate corpus, not an extension of the Phase 2 dataset in place).

## X RAW Studio side (per RAF)

| Setting | Value |
|---|---|
| Film Simulation | Provia |
| White Balance | **As Shot** / Camera Settings — NOT Auto |
| Dynamic Range | DR100 |
| Highlight tone | 0 |
| Shadow tone | 0 |
| Color / Saturation | 0 |
| Sharpness | 0 |
| Color Chrome Effect | Off |
| Color Chrome FX Blue | Off |
| Grain Effect | Off |
| Clarity | 0 |
| Noise Reduction | 0 (default) |
| Export format | **16-bit TIFF if available** (check this specific X RAW Studio version supports it — see `docs/phase3-raw-aware-processor-plan.md` §3). Fallback: 8-bit JPEG, maximum quality, only if 16-bit TIFF is confirmed unavailable. |
| Export bit-depth/format used | **Record which one was actually used** — write it down (e.g. in the batch folder's own notes) so the fitting pipeline knows which precision it's working with per scene. Don't assume; confirm per export session in case the app's default changes. |

## Browser side (per RAF) — Codex's responsibility to confirm once the high-bit-depth path exists

| Setting | Value |
|---|---|
| URL flag | `?rawCalibrationWb=camera` (as-shot WB override — same as Phase 2) |
| Film Simulation (recipe) | Provia |
| Dynamic Range (recipe) | DR100 |
| White Balance mode (recipe) | Any non-Auto mode (e.g. Kelvin) — value irrelevant, only its non-Auto-ness matters, to keep the shader's gray-world AWB from also kicking in (same reasoning as Phase 2) |
| All other recipe controls | zero/off, matching the X RAW Studio side exactly |
| Output format | High-bit-depth linear RGB per the §3 data contract in `docs/phase3-raw-aware-processor-plan.md` — **NOT** the 8-bit `browser-as-shot-baseline.jpg` convention from Phase 2. New filename convention below. |

## File naming and folder convention

Each scene gets its own `Shoot N` folder under `calibration-input/`,
continuing the numbering from Phase 2's last folder (currently through
`Shoot 126` — Phase 3 scenes start at `Shoot 127`). Per folder:

- `<original-filename>.RAF` (or a symlink to it, per the disk-space
  lesson from Phase 2 — see below)
- `xraw-phase3.tiff` (or `.jpg` if TIFF wasn't available — extension
  reflects the actual format used, and which one was used gets recorded
  per the table above)
- `browser-phase3-linear.<ext>` (extension per Codex's chosen format from
  §3) — Codex's output, once the high-bit-depth path exists

Do not reuse Phase 2's `xraw-as-shot-baseline.jpg` /
`browser-as-shot-baseline.jpg` names for Phase 3 files, even for the same
RAF if one happens to overlap (it won't — Phase 3's corpus is selected to
exclude every RAF already used in Phase 1/2, see the corpus selection
list) — the different name makes it immediately obvious which protocol
version and precision a given file was produced under, without needing to
open it.

## Batch export mechanics (X RAW Studio side)

Same approach proven in Phase 2: RAFs get symlinked into one flat batch
folder (`calibration-input/phase3-batch/`) so X RAW Studio's batch-select
can process them all in one settings pass, then Claude sorts and renames
the outputs back into their per-scene `Shoot N` folders automatically —
no need to track which RAF belongs to which folder while exporting.
Symlinks, not copies, per the Phase 2 disk-space incident — RAFs are
large (~85-90MB each) and 300 of them copied would be ~25-27GB, which is
avoidable since the originals already sit on the same disk.
