# Phase 4 — scene-adaptive Fuji rendering: scope (not started)

This is a scoping document only, written per explicit instruction before
any Phase 4 data collection, training, or code begins. Phase 3
(`docs/phase3-raw-aware-processor-plan.md`) is frozen, not shippable, and
stays that way until whatever comes out of this phase passes the same
per-photo zero-regression gate — Phase 3's 3 remaining Auto-WB failures
(Shoots 180, 229, 319) motivate this phase directly: their required
correction is 20-50× larger than anything the rest of a 230-scene corpus
ever needs, which Phase 3's own shoulder-correction experiment showed is
structurally impossible for a small, pooled, hand-featured regression to
supply without overfitting to just those scenes. Phase 4 is not a bigger
version of Phase 3's tuning — it's a different architecture and a
different scale of corpus, aimed specifically at that gap.

## 1. Why this needs richer spatial context, not more hand-designed features

Every Phase 3 feature (luma, near-black/white fraction, chroma ratio,
RAW-domain highlight topology, connected-component concentration) is a
**hand-picked scalar summary** of a pixel or a tile. Each one was tried,
measured, and either kept or ruled out — the process worked, but it has
hit a ceiling: no single hand-picked scalar (or small combination of
them) captures what Fuji's actual rendering engine is responding to for
scenes with an extreme, dominant highlight. The natural next lever isn't
"one more feature" — it's letting a model see actual local image
*structure* (edges, gradients, the shape and context of a bright region)
rather than a handful of numbers summarizing it.

The concrete architecture direction: a small convolutional network that
consumes a downsampled version of the full linear image and predicts a
**bilateral grid** of local affine color transforms (a low-resolution
grid over space × luma, each cell holding a 3×4 affine matrix) — the same
HDRNet-style trick Phase 3's spatial-tile/luma-zone model already
approximates by hand-fitting. The difference is that a small CNN
conditions its grid predictions on real image content the network learns
to attend to, not just the ~6-14 scalars per bin Phase 3 could specify in
advance. Applying the predicted grid at full resolution stays cheap (a
lookup + affine transform per pixel, not a full-resolution network
forward pass), which matters for browser inference — this preserves
Phase 3's core efficiency property while replacing the fitting mechanism.

This is Phase 4's central open technical decision and needs sign-off
before data collection starts in earnest: it's the first departure from
the "no deep learning framework" approach used through Phases 1-3, and
introduces new tooling (a training framework, likely a browser-inference
runtime) that doesn't exist in this project yet.

## 2. Data volume and diversity

Phase 3 needed 230 training scenes to go from "the approach doesn't work
at all" to "4 failures, all sharing one specific hard category." A model
with real spatial capacity (many more effective parameters than Phase
3's ridge-regression bins) needs more data to avoid overfitting, and
specifically needs enough independent examples of the failure category to
learn it as a general pattern rather than memorize a handful of scenes.

Proposed targets (to be revisited once real collection starts surfacing
practical constraints):

- **Overall corpus**: 800-1,500 scenes, roughly 3-6× Phase 3's size.
- **Hard-category minimum**: at least 60-100 scenes with a large, dominant
  bright highlight (chandeliers, direct light fixtures in frame, bright
  windows/skies filling a significant frame fraction, stage/performance
  lighting, strong sunset/sunset-adjacent highlight rolloff) — and
  critically, drawn from **at least 15-20 independent capture sessions**
  (different locations, different days, different lighting setups), not
  one location shot many times. Phase 3's own palace-cohort experiment is
  the direct cautionary example here: 10 more frames from the *same*
  session fixed 9 of 13 near-duplicate photos but didn't touch the
  qualitatively hardest 3 — more of the same session is not the same as
  more diversity.
- **Everything else**: broad, general coverage matching how the app is
  actually used (varied subjects, lighting, times of day), similar in
  spirit to Phase 3's existing 220+ ordinary scenes, just more of them.

## 3. Independent scene groups (holdout discipline)

Phase 3 only grouped the obviously near-duplicate palace frames for
leave-whole-group-out validation. Phase 4 should make grouping the
**default**, not an exception: every scene is tagged with a capture-
session ID (same location + day + lighting setup), and no split
(train/validation/held-out) may divide a single session across two tiers.
This is the direct fix for the failure mode Phase 3 already found once
(near-duplicate frames inflating confidence) — applying it universally
from the start avoids re-discovering the same problem per corpus.

Concretely: extend the existing manifest schema's `group` field (already
built and proven in Phase 3) to be populated for every scene, not just
the flagged ones, before any model-selection work begins.

## 4. Model input/output contract

- **Input**: downsampled linear RGB (same convention as Phase 3's
  `.fjlrg` container — high-bit-depth, linear light, sRGB primaries,
  as-shot WB), likely at a fixed small resolution for the network's
  low-res path (e.g. 256×256 or similar, resized independently of the
  full working resolution) plus the full-resolution image for applying
  the predicted grid.
- **Output**: a bilateral grid of per-cell affine transforms (grid
  dimensions TBD by experiment — Phase 3's 4×4 spatial × 7 luma zones is
  a reasonable starting point, but a learned model may benefit from finer
  resolution than hand-fitting could support).
- **Application**: full-resolution per-pixel affine transform via
  trilinear interpolation into the predicted grid (standard bilateral-
  grid slicing) — this is what keeps inference cheap enough for
  interactive Preview use regardless of network training cost.
- **Training target**: X RAW Studio's rendered TIFF, exactly as Phase 3's
  export protocol already produces it — no change needed to the target
  side of the pipeline, only the model consuming the pairs.

## 5. Hardware / memory requirements

Two separate concerns: training compute (offline, one-time-ish) and
inference compute (must run acceptably in a browser, every time a user
opens Preview).

- **Training**: a corpus of 800-1,500 full-resolution linear image pairs
  is substantially heavier than Phase 3's already-slow TIFF-loading step
  (which alone took 9-60+ minutes for 230-270 scenes on this Mac, using
  only CPU-bound `sips` conversion). Training a CNN, even a small one,
  on this volume of image data on CPU alone is likely impractical for
  reasonable iteration speed. This needs an explicit decision before
  committing to the approach: local Metal-accelerated training (if the
  chosen framework supports it), or renting cloud GPU time for training
  runs. Either way, expect the offline fitting pipeline's turnaround time
  to change qualitatively from Phase 3's (minutes-to-an-hour per
  experiment) — plan iteration cycles accordingly.
- **Inference**: must run client-side in the browser at Preview-
  interactive speed. This means the trained network needs to be small
  enough and exported to a format a browser runtime can execute
  efficiently (e.g. TensorFlow.js or ONNX Runtime Web) — a new dependency
  and a new export/conversion step that doesn't exist anywhere in this
  project yet. Model size and inference latency become real constraints
  on the architecture search, not an afterthought.

## 6. Untouched acceptance protocol

Same discipline as Phase 3, made explicit up front rather than
reconstructed under pressure later:

- A final held-out test set, sized proportionally to the larger corpus
  (Phase 3 used 40 of 300; a similar ~10-15% fraction here), selected and
  locked **before** any model architecture or training decision is made,
  respecting capture-session group boundaries (a session is entirely in
  training, entirely in validation, or entirely held out — never split).
- Zero access or inspection of the held-out set until the architecture
  and training procedure are fully frozen — the same rule Phase 3 upheld
  (and the same rule a real incident tested during Phase 3's corpus
  staging, resolved without any actual model leakage).
- The same per-photo zero-regression gate, not an average-error target —
  every training scene must be at least as good as no correction at all,
  scored under proper leave-one-out (or leave-one-session-out, given
  point 3 above) — before the held-out test is ever run, and the held-out
  test is run exactly once.
- Phase 3 stays frozen and unshipped regardless of Phase 4's progress
  until Phase 4's own held-out test passes this gate. No partial credit,
  no "close enough," and no claim of solved status before that.

## 7. Status

Not started. This document exists to be reviewed and revised before any
Phase 4 corpus collection, architecture prototyping, or training begins —
per instruction, Track 1 (Shoot 427 Manual/Kelvin WB parity) is the only
active work right now.
