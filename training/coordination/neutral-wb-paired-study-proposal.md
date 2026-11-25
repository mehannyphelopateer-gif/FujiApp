# Approved bounded study design: fresh paired G vs neutral-WB, 455 fit / 116 development

Status (2026-11-25): Codex reviewed the implemented design under the user's standing instruction to continue local research with Claude. The design below is fixed; the executable manifest is not yet created. Both full input receipts must pass and be published before manifest creation. A committed manifest, full571-pair preflight and disposable smoke proofs are mandatory before production training. No protected-tier access, model selection or app integration is authorized by this study.

## 1. Question and scope

Does replacing the as-shot-WB input (G) with the frozen, source-only neutral-WB input domain change
development colour error, with every other factor held fixed?

Single variable: input WB domain. Held fixed in both arms:

- Geometry: the existing G working geometry (no common-rectangle resampling of inputs or targets; that
  is a separate future single-variable study).
- Architecture/config: the Round 11 Hybrid config frozen in the matched-gamma parent manifest
  (`training/diagnostics/matched-gamma-training-manifest-v2.json`, `model_config`: grid 8x8x9,
  base_ch 16, refinement 8 / 0.08, low_res 256, `luma_mode` rec709, `condition_channels` 1).
- Guide: the standard `model.compute_luma_guide(..., "rec709")` (NO encoded guide; that trial is closed).
- Loss/optimisation: linear L1 mean, Adam lr 1e-3 (betas/eps/weight decay exactly as
  `run_matched_gamma_training.make_optimizer`), cosine to 1e-5, batch 1, 20 fixed epochs, no early stop,
  no checkpoint selection, no schedule change.
- Seeds: 42, 43, 44. Fresh initial weights from `matched_gamma_data.fresh_model(config, seed)`; the
  initial-state digest must equal the parent's `initial_state_sha256[seed]` (model init is data-independent).
- No R11 warm start, no R11 inference, no comparison to Round 11 numbers (R11 trained on development).

## 2. Cohort and eligibility (source-only, already frozen upstream)

| Role | Total | Neutral-domain kind | Notes |
|---|---|---|---|
| fit | 455 | W 288, P 166, G 1 | 12 presets (770 x9, 512 x2, 256 x1) excluded from BOTH arms, no fallback |
| development | 116 | W 5, P 74, G 37 | G37 = 25 Auto-White-Priority zero-shift + 12 Kelvin zero-shift |

Provenance: `training/neutral_wb_domain_plan.py` (`plan_rows`, `fit_rows`, `summarize`) over the
target-WB-header-audited inventory (`training/audit_target_wb_headers.py`, header manifest 16fcb23a…,
report 571/571 supported exact, 12 fit unsupported). Profile 4080/6600 frozen in wb-alignment manifest
6f6dac1c…. Kind selection is metadata-only; no target values, errors or pixels select rows.

Subgroups defined from metadata before any training:

- POLICY-TREATED-79 (source-metadata-defined W5 + P74, regardless of the observed pixel effect).
- UNCHANGED-37 (development rows whose neutral-domain input IS the G bytes): G37.
- DR100-12: development rows captured at DR100, outside fit support (no DR100 in fit). Reported only,
  never qualified. Metadata cross-tab is resolved below.

Development metadata cross-tab: all79policy-treated rows are DR400 (W5/P74);
all25DR200 rows and12DR100 rows are G identity. Fit has DR400 W256/P166
and DR200 W32/G1. Thus the development subgroup split is confounded with
capture DR: it tests treatment benefit at DR400 and collateral retention
at DR200/DR100, not WB neutralisation benefit at every DR mode.

## 3. Inputs per arm (both arms read the SAME targets)

Targets: the parent manifest target caches (`target_path`, `npy_sha256` per row), unchanged, CPU float32,
bit-exact, loaded exactly as `matched_gamma_data.Pairs.get` does (hash-checked, no clamp/rewrite).
Condition: `log2(captureDynamicRange/100)/2`, identical in both arms.

| Arm | Fit inputs | Development inputs |
|---|---|---|
| G (control) | existing G `v2-linear-asshot` containers for the 455 fit rows (parent `inputs.linear`) | existing G containers for all 116 |
| N (neutral WB) | W: `v2-linear-autowb` (fit export); P: `research-linear-fitprofile-k6600` FJNRGB16; G1: the same G container | W5 + P74 from a NEW development export (below); G37: the same G containers |

Fit neutral inputs come only from the fit export (`scripts/export-neutral-wb-fit-v2.mjs`,
`training/prepare_neutral_wb_fit_export_v2.py`, `training/verify_neutral_wb_fit_export_v2.py`, scope v2
bea38e7e…). Canary verification passed16/16, including two novel W/P cases; full455-row verification passed and was published in99d2133 (receipt7042fbf8…). The fit export
qualifies source/input identity, repeatability and provenance only; it does not qualify any model.

### 3.1 Separate development scope (implemented; full verification pending)

- 116 dev rows: 37 G identity (existing G bytes, no decode), W5 + P74 changed. Of the 5 dev W rows, 2
  are old W-pilot development artifacts; reuse them only if fully revalidated against their frozen
  proofs. Otherwise all 79 are new. Planned: 77 new decodes (3 W + 74 P) plus 2 revalidated W reuses.
  Confirmed from frozen proof metadata: DSCF0727 and DSCF0730; stored-byte revalidation passed in the41-row development canary receipt.
- The fit tooling is deliberately `role == "fit"` only (`scripts/lib/neutral-wb-artifacts.mjs`
  `validateScene`, `neutral_wb_domain_plan.fit_rows`, the decode helpers' `validateSourceScene`).
  It must NOT be weakened, and development rows must NEVER be relabelled `fit` to pass it.
- Required: a NEW development-only adapter and guard (separate module and manifest) that:
  - preserves the unchanged codecs and decode mathematics/options with the same C/N/R gates; the
    existing P decoder itself refuses development, so a separately scoped development-only decoder
    must honestly validate that role rather than call it with a fabricated fit role
    (C payload, strips and native hash equal to G; N/R exact);
  - carries the true `role: "development"`;
  - uses actual decoder metadata, never values filled from frozen expectations;
  - reads no development target pixels during export.
- Freeze the development-only scope and pass its synthetic/preflight checks before reading any
  development input pixels. Verify captured input artifacts afterward, before training or scoring.
  Target pixels remain unread during input export; this ordering avoids a circular prerequisite.

## 4. Loader and training mechanics (implemented and synthetic-tested)

- New per-arm, per-row loader, separate from `matched_gamma_data.Pairs`, which stays unchanged:
  - G rows: `fjlrg_v2.load_fjlrg_v2(..., expected_contract=CONTRACT_ASSHOT, expected_source_sha256, expected_wasm_sha256)`.
  - W rows: the same loader with `expected_contract=CONTRACT_AUTOWB`.
  - P rows: `fjnrgb_profile.load(path, expected_source_sha256, expected_wasm_sha256)`.
  - Each row checks its container and sidecar SHA, finite values and unit-range input, and geometry
    equal to the frozen working size (same checks as `Pairs.get`).
- Kind-to-contract map frozen per row in the study manifest. A row whose kind, contract or hash differs
  is refused, with no fallback.
- Orders: `matched_gamma_data.make_orders(fit_names_455, seed)`, regenerated for 455 and identical for
  both arms of a seed (old 467-row orders are not reusable). Order digest goes in the identity.
- Updates per run: 20 x 455 = 9100.
- Training loop and checkpoints: the new `run_neutral_wb_study.py` copies the existing
  `run_matched_gamma_training.run`, `update`, `make_optimizer`, `make_scheduler` and
  `seed_everything` algorithms without editing or patching those frozen modules:
  - atomic checkpoints every 50 updates and at each epoch end;
  - exact identity-checked resume (manifest/seed/arm/initial hash/order hash/config/trainer SHA);
  - non-finite loss or gradient stops before the step;
  - global lock; R11 hash check before and after.
- Endpoints: write-once `endpoint.pt` plus `endpoint.json`, validated like `validate_endpoint`.
- Evaluation: CPU, the unchanged `provia_color_pilot.dev_metrics` via `eval_matched_gamma_training`
  (`summarise`, per-scene rows), on all 455 fit and 116 development rows of each endpoint.
- Reporting: all six endpoints (2 arms x 3 seeds) are reported regardless of outcome. Only an
  integrity failure (hash/identity/non-finite) voids a run, and it is reported as voided, never dropped.

## 5. Predeclared internal screen (from the matched-gamma study; unchanged thresholds)

Deltas are neutral minus G, per development scene, same seed.

1. Pooled development mean ΔE00 delta (mean over seeds of per-seed development mean delta, all 116)
   is at most −0.25.
2. Every seed's development mean ΔE00 is lower for neutral than for G.
3. Pooled mean per-scene p95 ΔE00 delta is at most 0.

Additional predeclared subgroup conditions (metadata subgroups from section 2):

4. TREATED-79: pooled mean ΔE00 delta < 0 (direction of benefit where bytes changed). Reported per seed.
5. UNCHANGED-37 retention: for every seed, the G37 mean ΔE00 delta is at most +0.10.
   +0.10 is a fixed engineering margin chosen before any outcome. It is NOT derived from noise
   estimates or pixel results, and it exists only to catch collateral damage to rows whose inputs did
   not change.

Pass requires 1 to 5 together. DR100-12 and excluding-DR100 numbers are reported as context and
never qualified. Fit-set metrics are reported, not gated.

A pass would mean only that the neutral-WB input domain improved internal development error at this
fixed budget and architecture. It would not show Fuji parity, independent acceptance, runtime
readiness, or anything about geometry or the guide. A failure is recorded as negative, with no
retuning, extra arms or post-hoc subgroup gates.

## 6. Interpretation limits

- Development rows are an internal, previously-seen diagnostic split (sessions 7/12/14 clustered;
  P74/116 with nonzero-shift Kelvin6600), not independent acceptance. Protected tiers (K, monitor, finalHeldOut, T8,
  dr100Eval) are never read.
- The measured geometry difference (~0.3–0.5% scale on the sampled5178 class) is held fixed in
  both arms. Its DeltaE impact and WB interactions are unmeasured; this estimates WB benefit
  conditional on the current geometry, not benefit under a corrected geometry.
- The WB change also moves the Rec709 guide (luma mixes R/B), so guide shift is part of the treatment.
- Pairing controls initial state and order, but does not guarantee identical MPS numerical noise.
  All three seed results and spread are reported; no threshold is adjusted from those outcomes.

## 7. PENDING before freeze

- [x] Fit canary verification16/16.
- [x] Full455FIT export verification (455/455).
- [x] Development-only scope, adapter and guard reviewed and tested;41/41canary input verification passed.
- [ ] Full116development input verification (bulk in progress;37identity,2reuse,77new).
- [x] DR100 x kind cross-tab (metadata only), with subgroup/DR confounding recorded.
- [ ] Study manifest: per-row kind/contract/paths/SHAs, orders digest, initial-state digests, config, trainer SHA, screen JSON.
- [x] Separate exporter code/scopes reviewed and synthetic-tested before input capture.
- [x] Claude authored trainer/evaluator; Codex reviewed the loader/trainer/evaluator and ran42synthetic checks, all passing. Claude also reviewed the development exporter with no blocker; a further data-API review was usage-limited and has NOT occurred.
- [x] Fixed local work bound: six ×9100 =54,600optimizer updates plus six571-scene CPU evaluations and independent canaries. Disposable smoke reports actual current-device step time; production logs update throughput and ETA. No cloud work or spend; no deadline inferred from synthetic tests.

## 8. One-shot continuation and failure handling

The reviewed local `continue_neutral_wb_study.py --run` waits for the
full116-row input qualification, publishes only its allowlisted evidence,
prepares and commits the study manifest, runs the full paired preflight
and disposable smoke, publishes both proofs, then runs all six fixed
endpoints. Every result is reported even if earlier performance is negative.
Integrity/nonfinite/process failures stop the job; they never loosen a gate.
Checkpoint resume requires the identical frozen run identity. A separate
verifier reproduces aggregates/report arithmetic and re-infers6–7
metadata-selected CPU canaries per endpoint, not every571-scene prediction.
Completion requests a real read-only Claude review through the focused CLI
session; quota failure is recorded as review pending, never claimed as review.
This is a bounded local job, not a recurring automation or deployment.
