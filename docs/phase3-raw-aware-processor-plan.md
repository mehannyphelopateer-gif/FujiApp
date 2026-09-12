# Phase 3 — high-bit-depth, spatial RAW-aware processor (X100VI)

## Status (2026-09-11): FROZEN — not shippable for pixel-match mode

The bilateral-grid processor described below is **not** approved to ship
as a guaranteed-match replacement for X RAW Studio. It is a real,
substantial improvement over Phase 2 (training-pool mean MAE 26.24 → 7.85
across 230 scenes, most individual scenes well under 10) but it does
**not** pass the per-photo zero-regression gate: 4 of 230 training scenes
still regress (get made *worse*, not just insufficiently better) —
Shoots 180, 229, 319, 427. These are NOT excluded from the gate and the
processor is NOT to be described as solved or complete while they fail.

What was tried, in order, each with a real evaluation before moving on
(no hyperparameter tuning or hackery kept anything that didn't clear its
own gate):

1. **Local color-cast (chroma ratio) features** per spatial-tile/luma-zone
   bin — real win, closed most of the gap (this is what took the pool
   from consistently regressing to 4/230).
2. **10 more real training examples** of the same rare subject (gilded
   palace-ceiling frescoes with dominant highlights), added as a
   manifest-tagged group with leave-whole-group-out validation (so
   near-duplicate frames couldn't inflate confidence) — fixed 9 of the 13
   palace-cohort scenes cleanly. Did not fix the remaining 4.
3. **RAW-domain highlight-topology features** (connected-component
   analysis distinguishing one dominant glow from scattered specular
   highlights, per-tile near-saturation fraction, WB-corrected highlight
   chroma) — no measurable effect. Ruled out.
4. **Residual bucketing by luma × saturation × warm-hue** — showed the
   failures' error is uniform across hue/saturation in the bright zones,
   not concentrated in saturated warm highlights specifically. Ruled out
   hue/chroma-conditioning as the missing signal.
5. **RAW-domain exposure-regime audit** (tail percentiles, clipped
   fraction at multiple thresholds, full Fuji capture metadata) — found
   the 4 failures actually have *less* real sensor clipping than several
   successful scenes, ruling out "these are just more overexposed."
   Found Shoot 427 specifically was shot with Manual/Kelvin white balance
   (2.11× red gain) while every other scene in the series used Auto WB —
   a genuine, isolated capture-provenance difference for that one scene.
6. **Signed (not just absolute) residual by luma zone** — for Shoots 180/
   229/319, error is near-zero in shadows/midtones and explodes to +186
   to +241 (of 255) in bright zones, uniformly across R/G/B (neutral, not
   a color cast) — X RAW Studio renders these highlight regions far
   brighter than the corrected output reaches. For Shoot 427, the signed
   residual is NOT neutral — it shows a zone-dependent color-cast
   crossover (warm bias low-zone, cool bias top-zone), confirming it's a
   different failure mode (WB) from the other three (tone-curve).
7. **Constrained highlight-shoulder correction** — 3 predeclared,
   monotonic smoothstep basis functions at the existing zone 3/4/5/6
   boundaries, fit as a single global luminance-only correction (chroma
   preserved) on honest out-of-fold residuals, strong ridge, negative
   weights clipped to zero. Essentially no effect (e.g. Shoot 180:
   28.64 → 28.64 unchanged) — the fitted weights are tiny (max ~5/255)
   because a pooled, regularized fit correctly learns only the small
   correction most scenes need. The 3 failures need roughly 100-240/255
   in their brightest zones — 20-50× larger than what the rest of the
   corpus ever requires. Ruled out: this is not "mild curvature everyone
   needs a bit more of," it's a different magnitude of problem entirely.

**Conclusion, per team agreement**: exact offline matching cannot be
guaranteed without Fujifilm's own proprietary rendering engine — true
before this phase, still true now. For genuinely guaranteed Fuji output
today, the app must use the connected-camera/X RAW Studio path; the
offline processor is a best-effort improvement, not a guarantee, and must
not claim otherwise until it passes the untouched held-out acceptance
data (which it cannot even attempt yet, since the training-pool gate
itself isn't clean).

**Forward work split into two tracks; Track 1 is now closed:**

- **Track 1 — Manual/Kelvin WB parity (Shoot 427). CLOSED (2026-09-11),
  no decoder defect found.** Ran the direct controlled-gain test: decoded
  with a genuinely neutral gain (no camera WB, no auto WB), measured the
  unit-gain decode's own per-zone R/G and B/G ratio in non-clipped pixels,
  then solved for the exact gain each zone would need to match a
  second, independently-confirmed X RAW Studio reference (an explicit
  6600K export, verified pixel-identical to the original As-Shot export —
  ruling out any inconsistency on X RAW Studio's side too). The required
  gain dropped monotonically and substantially across zones (R gain 4.79
  in shadows down to 2.15 near highlights, a 2.23× spread) — a real
  decoder/WB-coefficient bug would need the same correction at every
  brightness level, since a wrong fixed multiplier is wrong by a constant
  factor everywhere. A steadily shrinking correction as the scene
  brightens is the signature of Fuji's tone curve desaturating a very
  warm, high-chroma capture as luminance increases, not a wrong gain
  value. No single gain aligns all zones simultaneously. Conclusion:
  Shoot 427's failure is the same underlying category of problem as
  180/229/319 — a corpus-diversity gap the learned correction has to
  extrapolate past — just along a different axis (extreme WB/chroma
  rather than extreme highlight magnitude). It is not a decoder defect
  and does not need app-side code changes. Folded into Phase 4's corpus
  scope (`docs/phase4-scene-adaptive-scope.md`) as a required
  stratification axis, not tracked further here.
- **Track 2 — scene-adaptive Fuji rendering (all four scenes now share
  this track).** Do not keep tuning the current tile/luma regression —
  the evidence above shows its correction magnitude is structurally too
  small for these cases, no matter what local features feed it. The next
  architecture needs richer spatial image context end-to-end, trained on
  a much larger and genuinely diverse paired corpus — specifically more
  bright interiors, chandeliers, windows, stage lighting, and other
  strong-highlight-rolloff scenes, AND scenes with extreme/manual white
  balance, with capture-sequence group holdouts enforced from the start.
  A handful of near-duplicate frames from one palace visit is not that
  corpus; this needs deliberate, broad collection across many distinct
  locations and lighting setups. See `docs/phase4-scene-adaptive-scope.md`
  for the full scope — corpus discovery is now underway there.

The diagnostics and code built along the way (train/validation split,
grouped leave-one-out, chroma/topology/shoulder feature infrastructure)
stay as-is, available for reuse by Phase 4 or any future revisit of this
architecture.

---

Owners: Claude owns this document, the export protocol, the calibration
corpus selection, and the offline fitting/evaluation pipeline. Codex owns
replacing the app's 8-bit JPEG RAW handoff with a high-bit-depth linear
path, exposing the RAW data this pipeline needs, integrating an approved
model into Preview, and independently validating browser renders — see
role split reaffirmed 2026-09-10 (previous Phase 2 split) and confirmed
for Phase 3.

Goal, stated precisely: a processor that is **consistently correct
per-photo** against X RAW Studio, not one with a better average error.
Every acceptance rule below is a per-photo gate, never an average.
Permanent caveat, carried over from the Phase 2 freeze and still true
here: without Fujifilm's own proprietary rendering engine, pixel-perfect
equivalence cannot be guaranteed. This phase aims to get substantially
closer than the frozen Phase 2 work, not to promise exact matching.

## 1. Why Phase 2 hit a ceiling, and what Phase 3 changes

Phase 2 (see `docs/scene-adaptive-processor-architecture.md`) conditioned
one correction *shape* (a 7-point luma lift curve) on *scalar summaries*
of a whole scene (a percentile, a fraction, a ratio) computed from 8-bit,
already-demosaiced, gamma-encoded JPEG samples. Three different ways of
using those summaries — a continuous model, a benefit-detector gate, a
risk-detector gate — all hit the same wall: real, measured improvement,
never a clean pass. Phase 3 changes both halves of that formula:

- **Input**: high-bit-depth **linear** RGB (not 8-bit gamma-encoded JPEG).
  The tone/highlight relationship this whole effort is chasing is much
  better-behaved in linear light — working from an already gamma-encoded,
  quantized 8-bit sample throws away precision before any correction is
  even computed.
- **Structure**: a **spatial** grid, not a single per-scene scalar. The
  recurring Phase 2 failure pattern (sparse specular/flash highlights vs.
  broad bright regions, structured vs. uniform shadow content) is
  fundamentally about *where* tonal extremes sit in the frame — something
  a whole-image percentile can never represent, no matter how many of them
  you add.

## 2. Model architecture

Deliberately **not** a deep neural network. The corpus size this phase can
realistically reach (see §4) is enough to support a genuinely local model
with real spatial capacity, but not enough to train an end-to-end deep net
without either massive overfitting or heavy pretraining infrastructure
this project doesn't have. Instead: a **bilateral-grid-style local
correction**, fit with the same ridge-regression machinery already proven
out in Phase 2, extended with spatial structure and a smoothness prior —
the standard, production-proven middle ground between "one global curve"
(Phase 2, failed) and "a full pixel-level deep net" (needs far more data
and infrastructure than this project has).

Concretely:

1. **Spatial grid**: divide each frame into an S×S grid (start at 4×4=16
   tiles — finer than Phase 2's 3×3 occupancy grid, coarse enough to fit
   with a few hundred scenes). Each tile is a genuine local correction
   target, not just a descriptive statistic like Phase 2's grid was.
2. **Luma zones within each tile**: reuse the 7-zone luma binning already
   validated in Phase 2 (`derive-shadow-chroma-recovery.mjs`,
   `derive-scene-adaptive-tonecurve.mjs`). Total bins per image: 16 spatial
   × 7 luma = 112.
3. **Per-bin local features**: the same feature families that showed real
   signal in Phase 2 (RAW percentiles, per-CFA-color ratios, near-black/
   near-white fractions) computed **within each spatial tile**, not just
   once for the whole scene — this is the concrete fix for "a bright patch
   in the corner needs different treatment than a bright patch filling the
   frame," which no whole-scene scalar could ever express.
4. **Fitting**: ridge regression per bin, as in Phase 2, but with an added
   **spatial smoothness penalty** — a discrete Laplacian regularizer
   pulling adjacent spatial tiles' coefficients toward each other. This is
   the standard bilateral-grid regularization technique (the non-deep-
   learning ancestor of HDRNet-style local tone mapping): it lets the grid
   genuinely adapt where the data supports it, while preventing any one
   sparsely-populated tile from fitting pure noise. Smoothness strength is
   a hyperparameter selected by cross-validation, exactly like every ridge
   lambda in Phase 2.
5. **Application at render time**: for a given pixel, find its spatial
   tile and luma zone, bilinearly interpolate the correction from
   neighboring grid cells (standard bilateral-grid slicing — smooth
   output, no visible tile boundaries), apply in linear space, then run
   the existing output tone curve/gamma for display.

This is a real increase in capacity and a real fix for the diagnosed
failure mode (spatial blindness), while staying inside tooling the team
already has working (Node.js ridge regression, no new ML framework to
stand up) and inside a corpus size this project can realistically reach.

## 3. Data contract — exactly what Codex needs to expose

This is the precise interface between Codex's app-side work and this
pipeline. Ambiguity here wastes both sides' effort, so it's written as a
contract, not a suggestion:

- **Format**: 16-bit-per-channel linear RGB (float32 acceptable and
  preferred if convenient on the LibRaw side). NOT sRGB-gamma-encoded, NOT
  8-bit, NOT run through the existing WebGL tone-curve/film-sim shader —
  this must be the neutral, linear-light sensor-to-RGB output, upstream of
  any of that.
- **Container**: 16-bit PNG or TIFF if a straightforward browser/Node-
  compatible encode path exists; a raw binary dump (documented byte
  layout: width, height, channel order, endianness) is an acceptable
  fallback if 16-bit image encoding isn't readily available in the browser
  — Codex's call on which is less engineering effort, as long as the
  format is exactly documented.
- **Resolution**: doesn't need to be full sensor resolution — downsampled
  after decode (in linear space, before any gamma) to something in the
  ~1024–2048px long-edge range is enough to support a 4×4–8×8 spatial
  grid meaningfully, while keeping file sizes and fitting time reasonable
  across a few hundred scenes.
- **Color space**: the same camera-matrix-corrected linear RGB LibRaw
  already produces internally (before the existing 8-bit sRGB gamma
  encode step in `rawService.ts`) — no new color science, just stop
  discarding precision before export.
- **As-shot WB**: same as Phase 2 — `useCameraWb: true` via the existing
  `?rawCalibrationWb=camera` calibration-only override, so this remains
  apples-to-apples with the As-Shot X RAW Studio protocol below.
- **X RAW Studio side**: check whether this version of X RAW Studio can
  export 16-bit TIFF (most professional RAW converters do). If yes, use it
  as the ground truth instead of 8-bit JPEG — comparing an 8-bit target
  against a 16-bit candidate would reintroduce exactly the precision loss
  this phase exists to remove. If 16-bit TIFF export isn't available,
  8-bit JPEG at maximum quality is the fallback, and that limitation
  should be written down, not silently accepted.

Nothing in the fitting pipeline (§5) can be finalized until this contract
is confirmed and Codex delivers at least one real sample pair in this
format — building fitting logic against a guessed format risks having to
redo it.

## 4. Calibration corpus — size, split, and selection method

**Target: 300 total scenes** — a real step up from Phase 2's 119 (103
train + 16 held-out), sized for the added spatial/local model capacity in
§2 without inviting the same overfitting risk that sank the single-scene
Hald CLUT and every "just add one more scalar feature" attempt in Phase 2.

- **220 training** — used for fitting and cross-validated hyperparameter
  selection (spatial smoothness strength, ridge lambda), same LOO/k-fold
  discipline as Phase 2.
- **40 validation** — a SECOND reserved set, new to this phase: used for
  monitoring/model-selection decisions *during* development (comparing
  candidate architectures, catching overfitting) so the training set's own
  cross-validation isn't the only signal guiding iteration — but still
  never used for the final accept/reject call.
- **40 held-out (final acceptance only)** — reserved before any fitting
  starts, exactly like Phase 2's 16-scene set, and under the same rule:
  zero access, zero inspection, until model architecture and training are
  fully frozen. Used exactly once.

**Selection method**: reuse the whole-library RAW-domain scan
(`calibration-input/phase2-raw-sensor-corpus.json`, 723 candidate RAFs
from the Egypt/Spain trip library, already excludes nothing — the 123
RAFs used in Phase 1/2 need excluding fresh). Stratify across the two
confirmed-predictive scalar features from Phase 2 (`percentiles.p99`,
`nearBlackFraction`) as a starting proxy for diversity, same method as the
Phase 2 batch-2 selection. **Caveat, disclosed rather than hidden:** the
existing library scan only has whole-image scalar stats, not spatial
ones — it can't directly select for spatial diversity (where highlights
sit in the frame, not just how much of the frame they cover), which is
the actual thing Phase 3's model newly cares about. Two ways to close this
gap, Codex's/the group's call: (a) proceed with scalar stratification now,
accepting it's an imperfect proxy for spatial diversity, since scalar
diversity at least correlates with scene-type diversity; or (b) have
Codex extend the whole-library scanner with a cheap spatial descriptor
(e.g. each candidate's existing-style 3×3 grid) before final selection,
which would take some scan time but directly targets what actually
matters for this phase. Flagging this now rather than picking silently.

## 5. Offline fitting/evaluation pipeline

Cannot be fully built until §3's data contract is confirmed and at least
one real sample exists — the core algorithm (spatial-grid assembly,
Laplacian-regularized ridge fit, cross-validation, gating) will be built
against an injectable image-loading function, so the loader is the only
piece that needs to change once Codex's real export format is confirmed.
Structurally mirrors Phase 2's proven pattern
(`derive-scene-adaptive-tonecurve.mjs`'s LOO + hard-fail gate), extended
for the spatial grid and the new validation-set role in §4:

1. Load each training scene's high-bit-depth linear input + X RAW Studio
   target (16-bit TIFF preferred per §3).
2. Assemble per-bin (spatial tile × luma zone) training data across all
   training scenes, exactly as Phase 2's `assembleZoneData` does but
   indexed by (tile, zone) instead of just zone.
3. Fit per-bin ridge regression with a spatial-Laplacian smoothness
   penalty between adjacent tiles, at each candidate (ridge lambda,
   smoothness lambda) pair.
4. Select hyperparameters via cross-validation on the training set, using
   the validation set as an independent check before finalizing (not for
   fitting itself).
5. **Acceptance gate, per-photo, exactly as strict as Phase 2's**: every
   training-pool scene must not regress beyond tolerance under leave-one-
   out. No exceptions, no averaging across scenes to justify a pass.
6. Only once that passes cleanly: the ONE-SHOT held-out test against the
   40 locked scenes. If it passes there too, the model is a candidate for
   Codex to integrate into Preview (still gated behind the existing
   calibration-only query-flag pattern before ever touching shipped
   behavior, consistent with every Phase 2 experiment).

## 6. Sequencing — what happens now vs. what's blocked

**Can start now, no dependency on Codex's high-bit-depth work:**
- Corpus selection (§4) — the RAF list and export folder/naming
  convention, so the user can start the X RAW Studio side in parallel.
- The X RAW Studio export protocol (separate locked-protocol document) —
  needs deciding regardless of which RAW format Codex ends up using on the
  app side.

**Blocked on Codex confirming the §3 data contract:**
- The high-bit-depth browser-side capture (Codex can't build against an
  unconfirmed spec either — the contract needs a quick sanity check on
  format feasibility before either side commits to it).
- The actual fitting pipeline's image-loading code (the algorithmic core
  can be written and smoke-tested against Phase 2's existing 8-bit data as
  a structural stand-in, clearly labeled as not real Phase 3 data, exactly
  as was done for Phase 2's scripts before real pairs existed).
