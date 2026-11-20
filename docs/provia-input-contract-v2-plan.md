# Provia input contract v2 — plan (design only, nothing launched)

Status: proposal, 2026-11-20. Written by Claude for joint review with Codex under the user's
standing go-ahead ("start working on whatever is needed … work together continuously").
That go-ahead covers local, reversible scoping and research. It does **not** lift the
protected-tier locks (K, monitor incl. the 12 Cuban monitor scenes, finalHeldOut, T8,
dr100Eval), allow cloud spend, change the selected model, or change the app. Each stage below
still uses the project's usual discipline: runner and manifest committed and hash-bound
before any media read, endpoints frozen before scoring, and every step dated in
`docs/phase4-scene-adaptive-scope.md`.

## 1. What is established, and what is not

**Confirmed: the legacy input transfer is not linear.**
- libraw-wasm 1.6.0 copies `gamm` only when the array has exactly 6 elements.
  - Wrapper commit 32fd36a.
  - The installed wasm SHA 8947f7e… matches the npm 1.6.0 tarball integrity.
- Every decode that sends `gamm: [1, 1]` therefore gets LibRaw 0.22.1's default curve.
  - Power 0.45, toe 4.5, full-scale white 65536, ushort truncation.
  - Join 0.0180540, shoulder 0.0992968.
- Behavioural proof, DSCF0391:
  - Legacy native output equals `LUT[six-element linear native]` on 5,793 channel samples, with max error 0.
  - The unedited pinned C++ `gamma_curve` reproduces all 65,536 table entries of the independent NumPy table.
  - Evidence: `training/diagnostics/provia-gamma-length-{probe,verification}.json`.
- The app's linear path (`src/lib/raw/rawService.ts:852`) sends the same two-element request. So canonical training inputs, the experimental Round 11 runtime and the app are consistent with each other. Only the "linear" flag, the comments and the input-physics interpretation are wrong.

**Confirmed: the source WB includes the camera fine-tune, and an Auto-only source candidate exists.**
- On all 11 source-Auto F24 scenes, the FujiIFD record 0xf00d (`WB_GRBLevelsAuto`) equals the WB levels recorded in the zero-fine-tune Provia target, exactly.
  - This holds across two fine-tune groups and five sessions (1, 7, 8, 11, 13).
  - Scope: Fujifilm X100VI only (`RAFDataVersion` 0x0369). Nothing in this plan transfers to other bodies without its own evidence.
  - Evidence: `provia-source-wb-probe.json`.
- This is a metadata agreement only.
  - The semantics are inferred.
  - Untested: white-priority or ambience-priority Auto, zero-shift controls, other bodies.

**Not established: the joint WB-application check FAILED.**
- `provia-gamma-length-probe.json` records `behavioural_confirmation: false`.
- The working-resolution ratio screen passes: medians within 0.005%, central 98% within 0.65%, mask coverage 8.3%.
- The descriptive affine gate fails:
  - red slope 0.8454 against 0.8248 expected, intercept −0.0015;
  - blue slope 1.2593 against 1.2809 expected, intercept +0.0019.
- The leading hypothesis is clipping mixed into averaged blocks (see §8). It is not diagnosed.

**Unresolved: manual, Kelvin and preset WB.**
- No parsed source field matches the target levels for any of the 13 non-Auto F24 scenes.
- See §7.

**No claim of Fuji parity**, and no claim that any v2 input improves model quality.

## 2. Three independent axes — keep them separate

| Axis | Legacy (v1) | What v2 may change | Where it lives |
|---|---|---|---|
| **Input transfer** | LibRaw default Rec.709-like (0.45/4.5), mislabelled linear | Truly linear (six-element `gamm` = `[1,1,0,0,0,0]`) | Decode options |
| **WB policy** | Camera as-shot (`cam_mul` = 0xf00e), fine-tune included | Auto scenes: source 0xf00d; non-Auto: unchanged as-shot, labelled | Decode `userMul` |
| **Capture-DR condition** | Metadata-derived plane (all F24 = DR400) | **Unchanged.** Not computed from pixels, so neither change above touches it | Model conditioning |

Model *output* stays as it is: Round 11 outputs linear values trained against linearised
Fuji targets (`target_tiff.py`, which doesn't use LibRaw), and display applies the sRGB OETF.
Nothing here alters targets, the output transfer or the metrics.

## 3. Legacy preservation (v1 stays bit-for-bit)

- No existing `.fjlrg`, cache, manifest, checkpoint or diagnostic is rewritten or reinterpreted. Old tensors are never "corrected" in place.
- Round 11 seed 42 (`r11-drcond-seed42-best.pt`, sha 1cfd2682…) stays selected and unwritten. Its runtime keeps legacy decode options, because a v2 decode underneath it would *create* a mismatch between training and runtime.
- The legacy contract gets a written name: **`input-v1-legacy-rec709-camwb`**. Wherever a doc or report says "linear input" about v1, read it as this name. Fix the wording in docs only. The `.fjlrg` flag bit and the `rawService.ts` comment are app or format edits, deferred to an app scope.
- The v1 reproduction gate stays the anchor: any v2 tooling must first reproduce the frozen canonical v1 hash for the same scene (already shown on DSCF0391 and one other allowed scene), or stop.

## 4. v2 container, filenames and provenance

**Format.** Keep the `FJLRGB16` magic, set **format version = 2**, header 64 bytes, all little-endian:

| Offset | Size | Field | v2 value |
|---|---|---|---|
| 0–31 | 32 | Legacy geometry block: magic, version, header size, width, height, channels, bits, payload bytes, flags | Same layout as v1, except version = 2, header size = 64, flags = 3 |
| 32 | u16 | `transfer` | 1 = linear (six-element `gamm`). Legacy Rec.709 is never written in v2 |
| 34 | u16 | `primaries` | 1 = sRGB via LibRaw `rgb_cam` (camera-RGB arrays are diagnostics only and never exported in v2) |
| 36 | u16 | `wb_policy` | 1 = camera WB (`useCameraWb`, no `userMul`); 2 = explicit `userMul` from source 0xf00d |
| 38 | u16 | `wb_coeff_source` | 1 = `cam_mul` / 0xf00e metadata; 2 = 0xf00d metadata |
| 40 | 4 × u32 | Source WB levels, RGBG | Metadata values as read: RAW-IFD LONGs, so u32 avoids truncation. For policy 1 these are metadata, not values passed to the decoder; for policy 2 they are the declared `userMul` |
| 56 | 8 | Options SHA256 prefix | First 8 bytes of the SHA256 of the exact UTF-8 options JSON string |

- **Separate codec.** The v2 codec is new standalone research code (JS exporter plus Python reader). `training/fjlrg.py` and the live Round 11 code are unchanged. The existing loader is expected to refuse v2 files on the version check; show that on a synthetic file before any export.
- **Strict reading.** The v2 reader refuses unknown or invalid field values, a header size other than 64, and a payload length mismatch.
- **Sidecar JSON per file:**
  - wasm SHA, wrapper commit, LibRaw version, RAF SHA256, and the full raw-payload SHA256;
  - WB mode: "camera WB, no explicit `userMul`" (arm G) or "`userMul` = 0xf00d RGBG" (arm W);
  - the declared options as the **exact serialised UTF-8 JSON string** plus its SHA256, so Python never re-serialises numbers ambiguously. The reader checks that the parsed string equals the recorded options object and that its hash prefix matches header offset 56;
  - a note that these are declared options, not internal parameter readback.

**Filenames**, next to the legacy file and never replacing it:
- `input-v2-linear-asshot.fjlrg`: the gamma-only arm.
- `input-v2-linear-autowb.fjlrg`: the WB arm, Auto scenes only (see §5).
- Caches use new directory names (`provia-color-target-cache` stays, because targets don't change; inputs go under `calibration-input-v2/<arm>/…` or a hash-named cache). The pool manifest gets a v2 sibling and is never edited.

**Training config and checkpoints** record `input_contract` (`v1-legacy-rec709-camwb`,
`v2-linear-asshot`, or `v2-linear-autowb`). Loaders refuse a checkpoint/input contract
mismatch. That makes it impossible to feed v2 inputs to Round 11 by accident.

## 5. Two arms, never merged

**Arm G: gamma-only (`v2-linear-asshot`).**
- Only the transfer changes; WB stays camera as-shot, exactly as legacy.
- Qualified by the encoding gate (§6, G1). No WB question involved, so it is ready to pilot.

**Arm W: gamma plus WB neutralisation (`v2-linear-autowb`).**
- Builds on arm G and adds the 0xf00d substitution for source-Auto scenes only.
- **Blocked** until the WB-application gate (§6, G3) passes under a gate version preregistered before its data is read. The current failure stays recorded.
- Non-Auto scenes: two options, chosen and frozen before any W run.
  - (a) Exclude non-Auto scenes from arm W comparisons. Recommended: the W-vs-G contrast is then clean on Auto scenes.
  - (b) Keep them with as-shot WB and `wb_policy=1`, accepting a mixed-policy input.
- Never claim W handles manual or Kelvin scenes.

Comparisons are always paired with the same recipe, seed, folds and scenes: legacy versus G,
then G versus W. W is never compared directly with legacy as a single "fix", because that
would confound the two changes.

## 6. Numerical gates (all must pass before a stage advances)

**G0 — legacy anchor.**
- The v2 tooling's v1 mode reproduces the frozen canonical v1 `.fjlrg` hash byte for byte, on every pilot scene.

**G1 — transfer.**
- On each pilot scene's fixed native grid (offset 64, stride 128, sampled before averaging), legacy native must equal the pinned 0.22.1 lookup table applied to arm G native.
- Gate: mean ≤ 2 codes and p99 ≤ 3 codes on ≥ 1000 channel samples with linear values between 1% and 95%. Report the exact-match fraction (expected 1.0, as on DSCF0391).
- Never compare encoded-then-averaged with averaged-then-encoded.

**G2 — determinism and export parity.**
- Two independent decodes of the same scene and arm give identical `.fjlrg` bytes.
- The working buffer equals the area average of the native image under the frozen canonical rounding.
- The Python v2 loader reads back exactly what the exporter wrote, with header fields asserted.

**G3 — WB application (arm W only).**
- Version to be preregistered after the clipping diagnostic design (§8) is frozen and **before** its data is read.
- The original joint gate stays FAILED as recorded. A new gate is a new, labelled protocol version, never a relaxation of the old one.

**G4 — loader and contract refusal.**
- v1 loaders refuse v2 files, and v2-contract training refuses v1 inputs and checkpoints. Shown on synthetic files.

**G5 — LUT-aware clipping accounting.**
- Changing the transfer doesn't move the physical clip point, but code counts are **not** comparable.
  - The pinned default table is many-to-one near the top: linear codes 65534 and 65535 both map to 65535.
  - Worked check: x = 65534/65536; x^0.45 × 1.0992968 − 0.0992968 ≈ 1 − 1.51×10⁻⁵; × 65536 ≈ 65535.01, which truncates to 65535. Code 65533 maps to 65534.
  - Above roughly 28% of linear scale the table's slope is below 1, so many linear codes share one encoded code.
  - So gamma-only can raise the encoded white count without any change in physical clipping.
- The gate is therefore the G1 identity on the native grid, legacy = table[G], applied to clipped samples too.
- Record arm G's linear saturation (value 65535 per channel, native and averaged) as a separate descriptive quantity. Use no data-dependent thresholds.

## 7. Manual, Kelvin and preset WB: separate exact-node inventory (unresolved)

- None of the 13 non-Auto F24 scenes is matched by 0xf00d, 0xf00c (fixed A/D65 calibration points, identical in all 24 files) or the as-shot record.
- In all 11 Kelvin 6600K scenes the target levels are constant at R620/B455, against as-shot R636/B414.
- The only untested source route is the RAFData (0xc000) WB block that LibRaw parses into `WBCT_Coeffs` and the preset records via `Fuji_wb_list1`.
- Follow-up scope, separate and metadata-only:
  - A bounded 0xc000 parser, cross-checked against a native LibRaw `WB_Coeffs`/`WBCT_Coeffs` dump.
  - Pinned to 0.22.1, noting the X100VI 34-record versus 31-record read in `fuji.cpp:834–851`.
  - Score a scene only where its exact Kelvin value or preset appears as a table node. No interpolation and no extrapolation.
- Static findings from LibRaw **0.22.1** (`src/metadata/fuji.cpp`, `src/tables/wblists.cpp`), not `master`:
  - `FujiCCT_K` has 31 nodes: 2500, 2550, 2650, 2700, 2800, 2850, 2950, 3000, 3100, 3200, 3300, 3400, 3600, 3700, 3800, 4000, 4200, 4300, 4500, 4800, 5000, 5300, 5600, 5900, 6300, 6700, 7100, 7700, 8300, 9100, 10000.
  - **6600K is not a node**: it falls between 6300 and 6700. So under LibRaw's labels, the 11 Kelvin scenes can't be scored on exact nodes at all.
  - For X100VI, `is34` is set (`RAFDataVersion` 0x0369, offset 0x0c5a). The table start is computed for **34** records, but only 31 are read and labelled with the 31-entry list. The table's real node temperatures, and so LibRaw's labels for this body, are unverified. Don't trust `WBCT_Coeffs` for X100VI without an independent label source.
  - Presets (`Fuji_wb_list1`, 6-byte records): FineWeather, Shade, FL_D, FL_N, FL_W, Tungsten. Per `tag2wbtable`, Shade matches Cloudy (0x2200) and FL_W matches White Fluorescent (0x2302). So DSCF0511 (Cloudy) and DSCF0514 (White Fluorescent) each have one candidate record. Each is a singleton, so inventory only, never a pass.
  - The RAFData Auto read (`wb_section_offset − 6`) only runs when `RAFDataVersion` is 0, so not on X100VI. The `TSNERDTS` scan is a separate path.
- Until that passes, arm W stays Auto-only. Inverting the fine-tune from the as-shot record is not ruled out in general (only the simple per-step model failed), but it is also not a validated runtime path.

## 8. Next diagnostic: the WB-application failure (Codex-owned)

Codex's clipping hypothesis predicts both observed bias signs:
- Red clips first in D, so E/D is above the gain.
- Blue clips first in E, so E/D is below the gain.

The design must decide what the original 5–80% working mask can't: whether the OLS bias
comes from blocks that contain clipped or clip-adjacent native pixels. Rules, frozen before
any data is read:

- **Flags.** Footprints are taken after demosaicing, not from sensor clipping; the source `RawSensorData` is not read.
  - Primary: a native pixel is clipped if any channel is 65535 in fresh D6 or E6. Dilate by radius 6 on the native grid, and flag a working block if any dilated native pixel lies in its averaging window.
  - Sensitivity variant, reported alongside: a radius-2 dilation on the working grid.
- **Decomposition.** Split the global OLS slope bias exactly and additively into contributions from flagged and clean blocks, inside the unchanged mask. Keep full sparse per-working-point leverage and covariance maps so a few extreme points can't be hidden.
- **Frequency doesn't matter.** Rare high-intensity blocks can carry almost all the covariance; that is the hypothesis. There is no minimum flagged fraction.
- **Support criterion, for both R and B:**
  - the flagged contribution has the predicted sign and is at least 80% of the global bias;
  - clean coverage is at least 2% of the frame, with at least 1000 points;
  - the clean-block affine fit is within the existing limits (slope within 0.1%, |intercept| ≤ 0.001).
- **Anything else is "not supported".** In particular, a clean fit that still fails means clipping doesn't explain the bias.
- **Labelling.** This is a prospective test of a hypothesis formed after seeing the failure. The original joint gate stays FAILED. A pass can only feed a newly versioned G3; it never relabels the old result.
- **Determinism.** The fresh D6/E6 must reproduce the existing D6/E6 hashes, or the run stops.

Its outcome only qualifies or keeps blocking arm W. It changes nothing about arm G.

## 9. Staged pilot before any bulk export

**P0 — tooling on synthetic files.** G4 and the header round-trip, with no media.

**P1 — F24 decode pilot.**
- 24 train scenes (all DR400, all already approved), arm G plus a v1 replay.
- Gates G0, G1, G2 and G5 on every scene.
- Arm W only if G3 has passed, and then only on the 11 Auto scenes.
- Cost:
  - about 30 s per full-resolution decode, so 24 × 3 decodes take about 36 minutes;
  - about 9.4 MB per `.fjlrg` at working resolution (≈1536 × 1024 × 3 × 2 bytes), under 1 GB including sidecars.

**P2 — training pool decode, only after P1 passes and is reviewed.**
- The 583-scene Provia pool, arm G (and arm W for its Auto scenes once G3 passes).
- About 583 × 30 s ≈ 5 h per arm and about 5.5 GB per arm. Fits the 307 GiB free with a wide margin.
- Protected tiers are never decoded in v2 at this stage.

**P3 — paired training comparison: a separate scope, not launched by this plan.**
- Same recipe, seed and frozen folds:
  - a v1 legacy retrain from the same initialisation, so legacy is compared with a matched re-run and not with the selected Round 11;
  - arm G;
  - arm W.
- Fine-tuning from Round 11 onto v2 inputs starts from a model trained on a different input distribution. That is a confound to name or avoid: prefer the same initialisation for all three arms.
- Local MPS only.
- Protected-tier evaluation stays locked until a later, explicitly authorised step.

**App.** No app change and no v2 decode under Round 11. A v2 runtime path would need a model
trained on v2 plus a browser-versus-export byte-parity gate, which is a later app scope.

## 10. Stop rules

Any of these stops the stage, keeps every output as recorded, and returns to review before anything else runs:
- a G0, G1, G2, G4 or G5 failure;
- a RAF hash change;
- an unexpected header or contract field;
- any request touching a protected tier.

No thresholds are loosened after their data is seen.


## 11. Implementation status (2026-11-20; supersedes proposal status above)

The v2 codecs and synthetic cross-language gates are implemented. The native
clipping screen did not explain the remaining red OLS residual; that old gate
stays failed. A new predeclared unmixed-photosite primitive test passed11/11
source-Auto scenes and33channels, qualifying source coefficient application
for research only. The bounded browser-safe RAF reader passed its15 synthetic
tests and matched the independent Python/ExifTool0xf00d and0xf00e records
on24/24 F24 sources, reading408 metadata bytes per RAF. No app integration.

The original P1 sparse transfer test stopped SHORT at DSCF0514 after two
complete scenes. It is not a pass. Separate full-native P1b is frozen with
unchanged error limits, independent saved strips, new output directory and
prospective manifest61c1efdbc2d8439cc9e016c02f35fcbe45e495882b47926b0f3f113f19547a8b.
No model receives v2 data yet. Manual/Kelvin remains unresolved.
