# Scene-adaptive browser-LibRaw-to-X-RAW-Studio processor — architecture

Goal: close the color/tone gap between the browser's LibRaw-based neutral RAF
decode (Preview tab) and X RAW Studio's own rendering of the same RAF, for
the X100VI, in a way that generalizes across scenes rather than helping some
and regressing others. This document is the design agreed before any more
data collection or fitting — per the explicit rule that got us here: no
correction ships until it passes held-out, per-subgroup acceptance tests.

Owners: Claude leads this architecture, the feature/fitting/validation
tooling, and data-collection targeting. Codex owns the browser/WebGL
implementation — exposing RAW-domain features, generating calibration
renders, integrating an approved model, and independently verifying it
against real X RAW Studio exports. See role split agreed 2026-09-10.

## What's already been ruled out, and why this design avoids repeating it

Three global (scene-independent) corrections were fit and failed leave-one-out
validation on a 9-scene, then 39-scene, as-shot-WB dataset:
- A global 3x4 color affine (`scripts/derive-libraw-base-normalization.mjs`).
- LibRaw's `highlight` (clip-recovery) decode modes 1/2/3/5/9.
- A global WebGL output-power/gamma adjustment (0.9) — this one is
  informative, not just a failure: it *helped* roughly half the scenes and
  *regressed* the other half, proving the needed correction amount varies
  per scene, not that no correction helps at all.

Two scene-adaptive attempts were then tried against real data:
- **Blown-highlight fraction in the decoded browser image** as a blend
  weight for the affine correction — failed. The signal that actually
  predicts correction benefit (r=-0.43, clean threshold) only exists when
  measured on the **X RAW Studio reference** image (the thing being
  predicted). Measured on the **browser's own decoded output** instead —
  the only thing ever available at runtime — the correlation collapses
  (r=0.28, unreliable sign). Mechanism: the browser decode under-renders
  exactly the high-dynamic-range regions the signal needs to detect, so the
  flawed output doesn't preserve evidence of its own flaw.
- **Exposure bias (EXIF metadata)** — a real signal (r=-0.43) that, unlike
  the above, *is* available at runtime without touching decoded pixels. But
  it only discriminates at the extremes (≥+0.67 EV: 6/6 scenes improved;
  ≤-1 EV: 7/9 regressed) — the bulk of the dataset sits at bias=0 and splits
  evenly, uninformative on its own.

**The organizing principle this leaves us with:** any feature computed from
the *decoded* (demosaiced, color-matrixed, tone-mapped) image is
untrustworthy, because it's corrupted by the exact flaw being corrected.
Features must come from EXIF/MakerNotes metadata or from RAW sensor data
examined *before* color processing. This is the core constraint the feature
list below is built around.

## 1. Runtime feature space

All of these must be computable at the moment Preview decodes a RAF, without
needing X RAW Studio's output (which will never be available at runtime) and
without depending on the already-color-processed/demosaiced pixel output.

| Feature | Source | Status |
|---|---|---|
| `exposureBiasEV` | EXIF `ExposureBiasValue` | Confirmed real signal (r=-0.43), noisy alone — primary candidate |
| `iso` | EXIF `ISOSpeedRatings` | No marginal correlation found alone; keep for interaction effects |
| `cameraModel` | EXIF `Model` | Only X100VI in the dataset so far — this whole effort is scoped per-camera-model; do not assume transfer to other bodies without their own calibration set |
| `asShotWbGains` (red/blue relative to green) | LibRaw `metadata(true).fuji` / `cam_mul` (Codex already exposed this — see commit history around the as-shot WB work) | Not yet tested for correlation — how far the scene's illuminant is from neutral could interact with the tone-curve mismatch |
| `rawHighlightClipFraction` | RAW sensor data (pre-demosaic, pre-color-matrix), fraction of pixels at/near the sensor's saturation level | **Not yet built.** This is the RAW-domain analog of the decoded-image blown-fraction that failed — needs to be measured before the color-processing flaw is introduced, not after. Highest-priority thing for Codex to expose next. |
| `rawShadowFraction` | Same RAW-domain source, near black level | Not yet built, same priority as above |
| `rawHistogramPercentiles` (p5/p50/p95, pre-color-processing) | RAW-domain | Optional richer alternative/supplement to the two fraction features |
| `dynamicRangeModeAtCapture` (DR100/200/400) | LibRaw `metadata(true).fuji.DevelopmentDynamicRange` (already exposed, commit eea7f8e) | Confirmed NOT correlated in the current 39-scene set, but only because it doesn't vary yet (all DR400) — keep as a feature since future data may vary it, and it's a plausible interaction term with the RAW clip features |

Deliberately excluded: any mean/stdev/percentile/fraction computed from
`browser-as-shot-baseline.jpg` or any other decoded output. Tested this
session (near-black fraction, mean luma, contrast, peakedness, decoded
blown-fraction) — none of it is trustworthy, for the reason above.

## 2. Correction representation

Given three fixed-parameter corrections already failed, the representation
needs real per-scene adaptivity — but with a dataset in the dozens-to-~100
range, it must stay low-parameter and auditable, not a black box.

Split the problem into two parts, each independently reviewable:

1. **A fixed, already-understood correction *shape***: reuse the per-luma-zone
   lift curve already proven out in `scripts/derive-shadow-chroma-recovery.mjs`
   (a small number of luma-bucket control points, each holding a per-channel
   lift amount, nearest-bucket or interpolated lookup at render time) rather
   than inventing a new functional form. This shape is mechanically simple
   and already validated as sound for a *different*, narrower axis (shadow
   blue recovery) — reusing it here means the only new risk is in part 2, not
   in whether this kind of curve can represent Fuji tone response at all.
2. **A small, regularized mapping from runtime features to that shape's
   control-point values** — e.g. ridge-regularized linear regression, each
   control point's lift predicted from a short feature vector
   (`exposureBiasEV`, `rawHighlightClipFraction`, `rawShadowFraction` to
   start; add others only if they show real marginal signal). Regularization
   strength chosen by cross-validation, not hand-picked — with ~100 scenes
   this needs to lean hard toward simplicity by default.

This keeps the "adaptive" part small and inspectable (a handful of linear
coefficients per control point) rather than jumping straight to something
expressive enough to memorize the training set, which is exactly the failure
mode already seen twice this investigation (the single-scene Hald CLUT, and
the concern flagged repeatedly about small-dataset overfitting).

## 3. Held-out acceptance tests

1. **Reserve a held-out test set before any fitting or design iteration** —
   roughly 20-25% of scenes, chosen to include real representation across the
   key feature ranges (especially exposure-bias extremes and whatever the RAW
   clip features turn out to look like), locked and untouched by model
   selection or tuning. Only used for the final accept/reject call.
2. **Cross-validation on the remaining training set** for fitting and
   regularization-strength selection — leave-one-out or k-fold, reusing the
   pattern already built in `derive-libraw-base-normalization.mjs`.
3. **Acceptance gate** (generalizes the existing per-scene gate):
   - No individual held-out scene regresses beyond a small tolerance (e.g.
     +2 MAE) — same spirit as the current script's hard per-scene check.
   - No regression in mean MAE for any defined **subgroup** — e.g.
     bias ≤ -1 / bias ≈ 0 / bias ≥ +0.67 tiers, and separately by RAW
     clip-fraction tier once that feature exists. This is the explicit
     requirement from the 2026-09-10 role-split discussion: an average win
     across the whole set is not sufficient if it's bought by regressing a
     recognizable subgroup.
   - A minimum overall improvement worth the complexity — if the model only
     matches the status quo, it isn't worth shipping the added complexity.
4. If any of the above fails, the model does not ship. Full stop, same
   discipline as everything else in this investigation — see
   `scripts/derive-libraw-base-normalization.mjs`'s existing hard-fail gate
   for the precedent this generalizes.

## 4. Sequencing — validate the new features cheaply before collecting more data

**Phase 1 (no new photography or X RAW Studio exports needed):** Codex builds
RAW-domain feature extraction (`rawHighlightClipFraction`, `rawShadowFraction`,
ideally the percentile histogram too) and exports it for the **39 scenes
already on disk** (`calibration-input/Shoot 8` through `Shoot 46` all already
have their RAFs). Claude then correlates these RAW-domain features against
the already-known before/after affine deltas from this session, the same way
the decoded-blown-fraction and exposure-bias correlations were tested — to
confirm or rule out whether a RAW-domain (uncorrupted) version of the
highlight-fraction signal actually survives, before asking for any more
manual X RAW Studio work.

**Phase 2 (only if Phase 1 finds a real, runtime-usable signal):** design a
deliberately-stratified data collection target — informed by whichever
features prove predictive — covering the full range of those features (not
just brightness/ISO like the 30-scene expansion this session), sized to
reach roughly 100+ total scenes with real held-out reserve. Claude will
specify the exact shot list from the existing 717-photo trip library (or
flag if new photography is actually needed for a range the trip library
doesn't cover), and the user batch-exports through X RAW Studio as before.

Nothing in Phase 2 is requested yet — it depends on what Phase 1 finds.
