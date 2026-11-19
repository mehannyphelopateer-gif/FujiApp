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
- **WB-stratification minimum** (added after closing Phase 3's Track 1):
  at least 30-50 scenes shot with manual/Kelvin-set or otherwise extreme
  white balance (not Auto WB), across multiple independent sessions, not
  just one photographer's habitual settings. Phase 3's Shoot 427 was
  originally suspected as a decoder/WB-parity defect; a direct controlled-
  gain test proved otherwise — no single gain value could align its
  shadow-through-highlight color response to X RAW Studio's rendering,
  ruling out a coefficient bug and confirming Fuji's tone curve responds
  to extreme WB/chroma content in a way this corpus never had another
  example of. That is a **missing training signal**, not a bug, exactly
  like the highlight-magnitude gap the other three scenes exposed — Phase
  4 must treat "how extreme is this scene's white balance" as its own
  stratification axis alongside highlight content, not fold it into the
  same "bright highlights" bucket, since the two failure modes involve
  different underlying color-science behavior (tone-curve response to
  saturated highlights vs. tone-curve response to extreme chroma).
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

## 6. Collection protocol — what's actually needed next

The 304-scene pilot inventory (§8) proved the needed regimes exist and
are findable, but it's all one continuous ~2-week trip with one
photographer and one camera body. For an X100VI-specific processor, more
camera bodies are **optional** — this project has never needed to
generalize across hardware, and doesn't need to now. What's **required**
is more independent *shooting conditions*: different trips, different
seasons, different locations, different lighting environments. A second
week in the same two countries with the same camera would not satisfy
this — the sessions need to be genuinely unrelated to each other and to
the existing corpus.

Target categories for new material (RAF originals only — no export or
processing needed at collection time, that follows Phase 3's locked
protocol once a corpus is actually selected):

- **Manual/Kelvin white balance** — deliberately set, not Auto. Vary the
  actual Kelvin value across sessions; don't repeat one setting.
- **Mixed lighting** — multiple light sources of different color
  temperature in one frame (e.g. daylight through a window mixed with
  interior tungsten/LED).
- **Bright windows / chandeliers / direct light fixtures in frame** — the
  specific failure category from Phase 3's 3 remaining scenes.
- **Night scenes** — low ambient light, likely high ISO, often with
  isolated bright point sources (street lights, signage) rather than
  broad highlights — a meaningfully different highlight *shape* than
  daytime chandeliers.
- **Skin / portraits** — Fuji's rendering is well known to treat skin
  tones with particular care; this corpus has none yet.
- **Foliage** — greens are another color-science area with dedicated
  handling in most film simulations.
- **Blue skies** — a large, saturated, spatially broad color region,
  structurally different from anything in the current corpus.
- **Ordinary scenes** — plain control cases, as many as the above
  combined, so the model has to stay good at what already works, not
  just get better at the hard cases.

No specific count is fixed here — that depends on how much independent
material actually becomes available. What matters more than volume is
that each category is represented across **multiple unrelated sessions**,
not concentrated in one outing (exactly the lesson from Phase 3's palace
cohort and this pilot inventory's single trip).

**Do not select or freeze a training corpus from what exists today.**
This section exists so that whenever new material arrives (another
existing RAF library, or a newly-shot session), it's immediately clear
what to classify it against — not so classification restarts from
scratch next time.

## 7. Untouched acceptance protocol

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

## 8. Status

**Superseded 2026-09-14 — see the dated decisions below.** A
session-grouped train/monitor/final-held-out split is frozen
(`calibration-input/phase4-corpus-manifest.json`), all 637 train+monitor
scenes have verified TIFF/`.fjlrg` pairs, and a local PyTorch/MPS training
pipeline (`training/`) is built and smoke-tested against real data — see
"Decision (2026-09-14): architecture, framework, compute" below. Track 1
(Shoot 427 Manual/Kelvin WB parity) is closed — see
`docs/phase3-raw-aware-processor-plan.md` — with no decoder defect found,
which is what motivated adding the WB-stratification axis in §2 above.

Discovery scope (as understood on 2026-09-11): the existing calibration
corpus (Phases 1-3) was believed to draw entirely from the "Spain" leg of
one trip, with a second, entirely unused "Egypt" leg (~392 RAFs) as the
natural first source of independent capture sessions. **Corrected
2026-09-13**: a full-library capture-timestamp sweep (see "Decision
(2026-09-13)" below) shows this was wrong — Phase 2/3's existing corpus
actually draws from RAFs physically stored in *both* the `Egypt/` and
`Spain/` library folders, and real capture sessions straddle the old
Phase 2/3 train/validation/held-out boundaries (that split was never
session-clean, since Phase 3 only grouped the one flagged palace cohort,
per §3). This doesn't affect Phase 3 itself, which is closed and already
ran its own held-out test — it only means Phase 4's own split can't be
built by reusing old per-file tier labels; it needed a fresh, unified,
whole-library session clustering instead. The locked Phase 3 held-out set
(Shoots 387-426) was excluded from the 2026-09-11 discovery *scanning*
pass by construction (cross-referenced against already-used filenames)
and stayed untouched through that pass — a separate matter from the fresh
session-boundary reclustering, which necessarily computed capture
timestamps for every file, including those, to find session edges at all
(see §8 note on what "untouched" means for that step).

### First discovery pass results (2026-09-11)

Scanned all 296 not-yet-used RAFs from the Egypt+Spain library (lightweight
capture metadata + a RAW tail-statistics pass per scene — no fitting, no
model touched). Full results in `calibration-input/phase4-discovery-manifest.json`
(untracked, same convention as `phase3-corpus-manifest.json`).

- **296 candidates**, clustered by capture timestamp (>3h gap = new
  session, trip legs never merged) into **14 independent sessions** across
  both trip legs — Egypt-1 through Egypt-9 (2026-07-11 through 07-18) and
  Spain-10 through Spain-14 (2026-07-20 through 07-22).
- **153 scenes shot with manual/non-Auto white balance** (`WB_Preset != 0`),
  spanning 8-9 of those independent sessions — not a near-duplicate
  cluster. Two distinct manual presets appear (R/B gain ≈2.11/1.37, the
  same 6600K-equivalent setting as Shoot 427; and ≈1.39/2.18, a different,
  cooler preset never seen anywhere in the Phase 3 corpus) — already more
  WB diversity than the single-scene evidence Track 1 worked from.
- **25 scenes classified bright-highlight** (RAW p99 > 0.5, or ≥1% of
  pixels within 1% of raw saturation) and **25 mixed-lighting** (p99 >
  0.25), against 246 ordinary-content scenes — a real, usable spread
  across the highlight-magnitude axis too.
- **14 scenes are BOTH manual-WB and bright-highlight** — the exact
  combination that caused Shoot 427's failure — spanning **6 independent
  sessions** (Egypt-3, 5, 6, 7, 8, 9), not one repeated visit. This is
  the highest-priority candidate set for the WB-stratification axis in
  §2, and it already satisfies the "multiple independent sessions" bar
  without needing to look beyond the existing library.

This first pass covers only the two trip legs already on this machine —
it does not yet reach §2's overall volume target (800-1,500 scenes) or
confirm whether session diversity is enough breadth on its own (different
lighting *within* one trip is not the same as different climates,
architectures, and cultures a broader collection would add).

### Broader machine-wide search (2026-09-11) — local sources exhausted

Read-only search across this Mac and its attached volumes for any other
Fuji RAF material: home directory (all locations, full depth), Photos.app
library, Lightroom library, Documents, Downloads, iCloud Drive, and every
mounted volume (`/Volumes`) — no copying, exporting, fitting, or training,
per instruction.

Result: **no additional independent trip or camera source exists on this
machine.** Every RAF found traces back to the same Egypt+Spain 2026 trip
already discovered, or to this project's own calibration-input staging
(symlinks to those same files, re-encountered as separate paths). One
attached volume is a stray single file already known from Phase 1/2
(`DSCF0752.RAF` in iCloud Drive). No external drives with photo content
are attached. No other trips, no other cameras.

One genuine addition: found 8 previously-missed unique frames in an
`Egypt:Spain 2026/All` subfolder, timestamped 2026-07-10 — a full day
*before* the earliest session already in the manifest, making it a
distinct 15th session (`Egypt-0`). Notably, one of these 8
(`DSCF0352.RAF`) carries **`WB_Preset=256`, a third distinct manual WB
value** not seen in either of the first two presets already found. 3 of
the 8 also classify as bright-highlight (p99 0.59-0.71). Folded into
`calibration-input/phase4-discovery-manifest.json` (now 304 scenes / 15
sessions total).

**Honest conclusion**: this machine's local discovery is complete for
now — 15 independent sessions, 3 distinct manual-WB presets, and a real
highlight-content spread, all from one continuous two-country trip. That
is genuine diversity in lighting, location, and WB *within* that trip,
but it is not diversity across separate trips, climates, or camera
bodies — everything found so far is the same X100VI, the same
photographer, the same ~2-week window. Reaching the volume and breadth
this phase's goal actually implies (real "broad Fuji parity," not just
"parity within this one trip") requires material this machine doesn't
have: either the user's other RAF sources not present here (phone,
another computer, a camera SD card, cloud storage not synced locally),
or genuinely new photography sessions. This is a scope decision for the
user/Codex, not something further local searching can resolve.

### Decision (2026-09-11): frozen as pilot inventory, awaiting new material

Per team review: this 304-scene, 15-session inventory is retained as-is —
a **pilot inventory**, not a corpus. No smaller train/validation/held-out
split will be carved out of it, and no model work starts from it alone.
More camera bodies are not required for an X100VI-specific processor;
more genuinely independent shooting conditions (other trips, seasons,
locations, lighting) are. §6 lists the target categories. Next action is
external, not further work on this repository: wait for the user to
provide additional RAF libraries or shoot new sessions, then repeat this
discovery process against the new material before any corpus selection
is proposed.

### Decision (2026-09-13): reversal — pilot corpus frozen, narrower claim

**This reverses the 2026-09-11 decision above.** The user explicitly
chose to proceed with the existing Egypt/Spain library now rather than
wait for additional trips — no new independent material has been added
since 2026-09-11. This is an intentional, authorized override, scoped
narrowly: **the resulting corpus is a Phase 4 pilot for the current
X100VI / Egypt-Spain-2026 trip only. It does not and cannot claim
universal Fuji-parity** — §2's 800-1,500 scene / 15-20 independent-session
targets for that broader claim are unchanged and still unmet (this
library has only 14 real sessions total, see below). Everything in §2's
data-volume discussion and §7's acceptance protocol still applies in full
to whatever claim gets made from this pilot's results — only the "wait
for more material before selecting anything" part of the 09-11 call is
reversed.

**What changed technically**: freezing a real train/monitor/held-out
split required re-clustering the *entire* 731-RAF library (not just the
304 previously-unused scenes) into sessions from scratch, because — per
the correction above — the old Phase 2/3 corpus and the new discovery
pilot turn out to share real capture sessions once actual timestamps are
compared directly. Re-running the same `>3h gap = new session` rule
across all 731 RAFs uniformly (lightweight `mdls` capture-timestamp / ISO
/ WB-manual-flag sweep only — no pixel decode, no RAW-domain pass, exactly
the discipline the 09-11 pass used) collapses this to **14 independent
sessions**, not the 15 the 09-11 discovery pass counted for its narrower
296+8 subset — the two adjacent 2026-07-10 pre-trip frames the 09-11 pass
grouped as one 8-frame "Egypt-0" session split into two 4-frame sessions
under the same >3h rule applied strictly. This is a correction to the
count, not a new decision to weaken the rule.

**Tier assignment** (whole sessions only, never split, per §3 and §7):

| Tier | Sessions | Scenes | % of library |
|---|---|---|---|
| train | 0,1,2,3,7,8,11,12,13 | 561 | 76.7% |
| monitor | 4,5,9,10 | 76 | 10.4% |
| finalHeldOut | 6 (2026-07-14, Egypt, 94 scenes) | 94 | 12.9% |

Held-out lands inside §7's ~10-15% target as a single clean whole-session
pick. Both the held-out session and the monitor sessions include a real
mix of manual-WB and bright-highlight/mixed-lighting scenes (not just
ordinary content) so the eventual zero-regression test is a meaningful
one, not vacuous. Full manifest:
`calibration-input/phase4-corpus-manifest.json` (untracked, same
convention as the other phase*-manifest.json files).

**Held-out discipline applied**: the `finalHeldOut` tier's manifest
entries carry only filename, session index, trip folder, capture
timestamp, and (if one already exists) which `Shoot N` folder holds it —
no pixel-derived stats (p99, highlight class, WB gains), no export, no
batch-folder entry, even where those stats already happen to exist from
the old, closed Phase 2/3 experiment for a scene that lands in this tier.
Nothing further will touch this tier until train/monitor model selection
is frozen, per the user's explicit instruction not to export, inspect,
fit, or compare it before then.

**Reused existing pairs**: 229 of the 637 train+monitor scenes already
have a valid Phase 3-protocol export pair (`xraw-phase3.tiff` +
`browser-phase3-linear.fjlrg`) sitting in their `Shoot N` folder from the
earlier Phase 2/3 work — these are reused as-is, no re-export. The
remaining **408 train+monitor scenes need a fresh export**; their RAFs
are symlinked into `calibration-input/phase4-batch/` (flat batch folder,
same symlink-not-copy convention as `phase3-batch/` in
`docs/phase3-export-protocol.md`) ready for the X RAW Studio + browser
export pass. That export itself has not been run — this phase produced
the manifest and the batch folder only, per instruction. 33 of the 408
are RAFs the 2026-09-11 discovery pass missed entirely (see the "37
uncatalogued" note — 4 of the 37 fall in the held-out session and are
therefore excluded even from this count); they have no highlight-class
stats yet since computing those would require a fresh RAW pixel decode
this pass didn't do — a gap to backfill later, not a blocker for export.

Both export sides completed 2026-09-14: 408 X RAW Studio TIFFs (4 batches
of 102, split per the staging contract below) and the matching 408
browser `.fjlrg` exports (Codex, `scripts/export-phase3-linear.mjs`
updated to read `calibration-input/phase4-needs-browser-export.json`
directly and refuse `finalHeldOut` scenes). All 637 train+monitor scenes
now have a verified `xraw-phase3.tiff` + `browser-phase3-linear.fjlrg`
pair. `finalHeldOut` (94 scenes) confirmed untouched throughout — no
export, no pixel-level inspection beyond the session-clustering timestamp
sweep.

**Staging contract**: every train/monitor scene needing export got an
explicit, unique `Shoot N` folder before any export ran — 139 reused from
earlier discovery work, 269 newly created (`Shoot 437`–`Shoot 705`), each
holding just that scene's RAF symlink. RAFs were split into 4 batch
folders (`phase4-batch-1`–`4`, 102 each) with matching manifests
(`phase4-batch-N-manifest.json`) mapping filename → `Shoot N`, so sorting
X RAW Studio's output back was a lookup, not a guess.

### Decision (2026-09-14): architecture, framework, compute

Per user authorization, following the options laid out for review
(`training/README.md` has the implementation-level detail): **train
locally only**, in PyTorch over Apple's MPS backend, export the frozen
model to ONNX, and run browser inference via ONNX Runtime Web (WebGPU
backend, WASM fallback). No cloud GPU, no paid compute, unless a later
explicit decision changes this.

A local training pipeline (`training/`) is built and validated end-to-end
against real data — `.fjlrg` + target-TIFF loaders, the bilateral-grid
model and its slicing op (§4), and a training loop, smoke-tested on 4 real
train + 2 real monitor scenes (loss 0.287→0.187→0.107 over 3 epochs on
that tiny overfit check). Two Apple MPS backend gaps were found and worked
around along the way (documented in `training/README.md`): 5D
`grid_sample`'s backward pass isn't implemented on MPS yet (falls back to
CPU for that one op only, negligible cost), and `F.interpolate`'s `"area"`
mode requires evenly-divisible sizes on MPS (switched to antialiased
bilinear, which doesn't have that restriction).

**This is a smoke test, not a trained model.** Full-corpus target
precompute (637 scenes, ~50 min one-time cost) and real
architecture/hyperparameter iteration against `train`/`monitor` haven't
started. `finalHeldOut` remains completely off-limits until that iteration
converges and model selection is frozen, per §7.

### Decision (2026-09-15): first model-selection pass frozen

Per user authorization, ran the first real model-selection pass: three
candidates (chosen to test §4's two open axes - grid resolution and
encoder capacity), each 15 epochs on the full 561-scene train / 76-scene
monitor split, local PyTorch/MPS, batch size 1 (forced - scenes are not
uniformly sized; `.fjlrg` export fixes the long edge at 1536px but the
short edge varies a few px by aspect ratio, e.g. 1026×1536 / 1536×1026 /
1023×1536 all appear in this corpus, and PyTorch's default batch collate
can't stack mismatched sizes - documented in `training/train.py`).
`training/run_model_selection.py` selects by best-epoch monitor L1, never
train L1, and has no code path to `finalHeldOut` at all.

| Candidate | Grid | Params | Best epoch | Best monitor L1 | Regressed | Wall time |
|---|---|---|---|---|---|---|
| a-baseline | 4×4×7 | 103,380 | 14 | 0.0390 | 5/76 | 103.5 min |
| **b-finer-grid** | 8×8×9 | 104,940 | 8 | **0.0381** | 5/76 | 103.1 min |
| c-higher-capacity | 4×4×7, base_ch=32 | 400,212 | 13 | 0.0388 | 5/76 | 104.0 min |

No-correction baseline monitor L1: 0.1023 - every candidate landed at
roughly 2.5-2.7× better than doing nothing, on a first, non-exhaustive
attempt. **Selected: `cand-b-finer-grid`** (finer grid resolution won at
essentially the same parameter count and runtime as the baseline - the
extra grid cells cost almost nothing since the encoder backbone is
identical; higher encoder capacity alone, at 4x baseline's params, did not
beat it). Checkpoint: `training/checkpoints/cand-b-finer-grid-best.pt`
(epoch 8 weights - its best-epoch checkpoint, not its final-epoch one;
monitor L1 drifted slightly worse from epoch 8 to 15, a mild overfitting
signal worth a learning-rate schedule or early stopping next time).
MPS driver memory stayed flat per candidate (~1.1-1.2GB, well under the
16GB budget) regardless of encoder width.

This is a first pass, not a converged final architecture - three
candidates on one seed is a reasoned start, not an exhaustive search.
**`finalHeldOut` was not accessed.** Per §7, the next gate is explicit
permission for any held-out evaluation, which has not been requested or
granted.

### Decision (2026-09-15): round 2 — predeclared, targeted at the 5 regressions

Per instruction: round 1's winner is not frozen yet (one seed, 5 monitor
regressions, visible epoch-8→15 drift are not "stable"). A second,
small, predeclared round - not open-ended tuning, loss stays plain
linear-RGB L1 throughout, no perceptual loss - tested two single-change
variants of `cand-b-finer-grid` (8×8×9 grid), then reran the winner at
2 more seeds for a stability check:

| Run | Change | Best monitor L1 | Regressed | Worst-case |
|---|---|---|---|---|
| Round 1 baseline (seed 42) | — | 0.0381 | 5/76 | not tracked |
| r2-b-lrsched | cosine LR schedule + early stop (patience 5) | 0.0387 | 5/76 | 0.0532 |
| r2-b-smooth | grid TV regularization (weight 0.005) | 0.0397 | 5/76 | 0.0648 |

Neither single change clearly beat the baseline; TV regularization made
the worst-case regression *worse*, not better - the smoothness penalty
isn't the actual fix for whatever's failing on those scenes. Per the
predeclared selection rule (fewest regressions, then lowest worst-case,
then lowest monitor L1), `r2-b-lrsched` won the tiebreak against
`r2-b-smooth` - not a confirmed win, the less-bad of two roughly-neutral
changes.

**Stability check** (`r2-b-lrsched`'s exact config, 3 seeds):

| Seed | Best monitor L1 | Regressed | Worst-case |
|---|---|---|---|
| 42 | 0.0387 | 5/76 | 0.0532 |
| 43 | 0.0355 | 3/76 | 0.0535 |
| 44 | 0.0390 | 4/76 | 0.0628 |
| **mean / stdev** | **0.0377 / 0.0020** | 4 / — | — |

Real seed-to-seed variance (~5% relative stdev on monitor L1, regression
count ranging 3-5): seed 43 clearly beats round 1's baseline, seed 44 is
about the same or slightly worse. Mean monitor L1 (0.0377) is roughly on
par with round 1's single-seed baseline (0.0381); mean regression count
(4) is a modest improvement over round 1's 5. **This is not an
unambiguous, stable win** - it's within the range where seed variance
alone could explain the difference from round 1. Per instruction ("do not
access `finalHeldOut` unless the selected configuration is stable and
clearly improves or matches B without trading in new monitor failures"),
whether this clears that bar is a judgment call for the user/Codex, not
decided here.

Also logged: three unrelated external-drive I/O incidents during this
round (a device-busy error, an extended outage that exceeded the first
retry budget, a numpy partial-read `ValueError`, and a stale
working-directory handle after a remount) each exposed a real gap in the
training scripts' resilience, now fixed - see `training/README.md` and
the commit history for `training/dataset.py` / `training/run_resilient.sh`.
None of these affected data integrity; each was caught, logged, and
recovered from automatically or by inspection before any run's result was
trusted.

**`finalHeldOut` was not accessed** - independently verified by scanning
every round-2 run log and the final comparison file for any held-out
scene name (`grep`, not just code inspection): zero matches.

### Diagnosis (2026-09-15): regression mechanism found - Auto WB, not manual

Per instruction: no further blind hyperparameter sweeps until the 3-5
monitor regressions were actually understood. Ran all 3 round-2 seed
checkpoints against every monitor scene (`training/diagnose_regressions.py`,
inference only - no training, no `finalHeldOut` access, independently
verified: zero held-out scene names anywhere in the output), took the
union of scenes that regressed under at least one seed (5 of 76), and
compared them against the 71 that never regressed under any seed.

**The result is clean and points the opposite direction from Phase 3's
motivating case.** Phase 3's Shoot 427 (the scene that motivated adding
the WB-stratification axis in §2) was a *manual/Kelvin* WB failure. Every
one of these 5 regressions is the opposite:

| | Regressing (n=5) | Control (n=71) |
|---|---|---|
| Manual WB fraction | **0%** | 45% |
| Mean RAW p99 (highlight magnitude) | **0.547** | 0.234 |
| bright-highlight / mixed-lighting share | 80% (4/5) | 30% (21/71) |

All 5 regressing scenes shoot **Auto White Balance**, none manual - a
100%-vs-45% split that isn't plausibly chance at this sample size. They
also skew sharply toward elevated highlight content (p99 more than double
the control mean). Session membership is spread across all 4 monitor
sessions (not one bad batch): sessions 4, 5, 9, 10.

**Seed consistency**: 4 of the 5 scenes (`DSCF0878`, `DSCF0873`,
`DSCF0879`, `DSCF0590`) regressed under **all three seeds** - a real,
reproducible property of those scenes, not training noise. Only
`DSCF0553` is seed-dependent (regressed under seeds 42/44, not 43),
exactly matching why seed 43 had one fewer regression (3 vs 4-5) than the
other two runs.

**Spatial pattern** (residual-vs-luma correlation + bright-pixel error
share, seed43 checkpoint, `training/diagnostics/residual_heatmaps/`):
4 of 5 scenes show moderate-to-strong positive correlation (+0.51 to
+0.66) between per-pixel error and local luma, with the brightest 5% of
pixels holding 2-3.4× their fair share of total error mass - the error
concentrates in the bright regions themselves, consistent with the
highlight-magnitude failure category Phase 3 already knew about. The
exception, `DSCF0878` (the single most extreme scene by p99, 0.964),
shows weak correlation (+0.19) and its brightest pixels hold *less* than
their fair share (1.6% vs 5%) - its failure looks structurally different
from the other four, worth separate attention rather than assuming one
mechanism explains all 5.

**Working hypothesis**: the bilateral-grid model's luma-indexed grid axis
conditions correction on luma computed from the (as-shot-WB) linear
input; an Auto-WB scene's white balance can shift where highlight
content actually sits in RGB-balance space relative to how the grid was
trained to expect it, compounding with elevated highlight magnitude to
push the grid's luma-zone lookup into a region it fits poorly. This is a
hypothesis from the data pattern, not confirmed by a controlled
intervention.

**Checked, not just assumed**: it is *not* simple training-data scarcity
- the train tier has 304 Auto-WB scenes (54.2% of 561) and 52 that are
both Auto-WB and bright-highlight/mixed-lighting, a reasonable-sized
group, not a handful. The model sees this combination during training and
still fails on these 5 monitor scenes specifically, which argues for
something about these particular scenes' extremity (or a genuine
architectural limitation in how the grid's luma axis handles this
regime) over "needs more examples of the category." The next real test
is targeted at that distinction - e.g. checking where these 5 scenes'
p99/WB values actually sit relative to the 52 similar-category train
scenes (tail of the distribution, or a genuinely different regime?) -
rather than tuning grid resolution or regularization again.

Full data: `training/diagnostics/regression_diagnosis.json`.
`finalHeldOut` untouched throughout - still not requested.

### Round 3 (2026-09-15/16): WB-stable luma guide - hypothesis falsified

Per instruction: one targeted intervention, architecture otherwise
unchanged. `model.compute_luma_guide` gained a `"green_channel"` mode -
this project's own WB convention (`src/lib/raw/rawService.ts`) normalizes
gains relative to green=1.0, so the G channel of the already-cached
as-shot-WB `.fjlrg` data is already WB-gain-invariant, no new per-scene
gain extraction needed. Only the bilateral grid's luma-axis lookup
coordinate changed; the as-shot-WB color image stayed both the model's
input and output throughout. Trained the same 3 seeds (42/43/44) at the
identical architecture/schedule as round 2's winner, then evaluated every
monitor scene against both the frozen baseline and the new checkpoints
with the same evaluation code for a fair comparison
(`training/run_round3_wbstable.py`).

**Result: exactly the same 5 scenes regressed, under exactly the same
seeds, with zero exceptions** - `DSCF0553` (42,44 only, as before),
`DSCF0590`/`DSCF0873`/`DSCF0879`/`DSCF0878` (all 3 seeds, as before).
Zero resolved, zero new regressions. `predeclared_success_met: false`.
Per-seed aggregate monitor L1 moved in both directions (seed 42 improved
0.0387→0.0371, seeds 43/44 got slightly worse) but that movement had no
relationship to which scenes actually fail - pure training noise, not
progress on the mechanism.

**This falsifies the working hypothesis from the diagnosis.** The
WB-shift-via-luma-axis theory does not explain these 5 failures - a
WB-invariant luma guide changed nothing about them. The actual mechanism
is still unknown. Two honest directions from here, not yet decided:
inspect what's actually spatially/structurally unusual about these exact
5 scenes beyond the WB/highlight metadata pattern (the diagnosis's
metadata correlation was real but apparently not causal), or treat this
as the pilot's floor and move toward `finalHeldOut` with 5/76 as a known,
documented limitation rather than keep spending targeted-intervention
budget on an increasingly narrow mechanism search.

Full data: `training/diagnostics/round3-wbstable-comparison.json`.
`finalHeldOut` untouched throughout - still not requested.

### Structural diagnosis (2026-09-16): compact highlight shape, prediction not slicing

Per instruction: 5 reproducible monitor regressions aren't an acceptable
floor, so kept investigating instead of requesting `finalHeldOut`.
Compared the 4 reproducible failures (`DSCF0590`/`DSCF0873`/`DSCF0879`/
`DSCF0553`, excluding the already-known-distinct `DSCF0878`) against all
21 matched controls (every other monitor scene in the same bright-
highlight/mixed-lighting categories that never regressed) on highlight
region shape, local luminance distribution, and a prediction-vs-slicing
split (`training/diagnose_structural.py`, CPU-only - 5D `grid_sample`'s
`nearest` mode isn't implemented on MPS).

**Highlight shape, not overall brightness, is what stands out.** Luma
distribution stats (near-black fraction, bright fraction, std) are
essentially the same between the regressing 4 and control - these aren't
unusually bright or contrasty scenes overall. What differs is the
*shape* of the brightest region itself:

| | Regressing 4 | Control (n=21) |
|---|---|---|
| Highlight components (count) | 467 | 861 |
| Largest blob area fraction | 0.0022 | 0.0016 |
| Largest blob bbox aspect ratio | 1.54 | 2.21 |
| Largest blob compactness | 0.570 | 0.488 |

Fewer, larger, more compact (closer to square, less elongated) highlight
blobs - one dominant roughly-round bright region rather than several
smaller or more elongated ones (a window's strip shape, a line of
lights). Sample size is small (4 vs 21) - suggestive, not proven.

**Prediction, not slicing, looks like the actual bottleneck.** The
interpolation-sensitivity test (trilinear vs. nearest-neighbor grid
sampling on the identical predicted grid - a big gap means the grid
varies rapidly where the scene's pixels land, a resolution/slicing
problem; a small gap means the grid is locally smooth there and the
predicted *coefficients* themselves must be wrong) showed **no elevation
for the regressing 4** (0.0336) versus control (0.0345) - if anything
marginally lower. The bilateral grid isn't unusually rough in the region
these scenes occupy; the coefficients the CNN encoder predicts for these
scenes are the more likely fault, not the coarseness of the 8×8×9 grid
or its trilinear interpolation.

**`DSCF0878` stays the outlier it already looked like** - far fewer
highlight components (125) but the *least* compact shape (0.326,
scattered rather than one solid blob), much higher bright-pixel fraction
(0.038 vs ~0.0086 for everyone else) and near-black fraction less than
half the others' (0.198) - a broadly, evenly lit scene, not a small hard
highlight in a dark frame. It also showed the highest interpolation
sensitivity (0.0457) of any scene checked, the opposite pattern from the
other 4 - a genuinely different failure mode, not tuned to or folded into
the same read on the other four.

Full data: `training/diagnostics/structural_diagnosis.json`. Next step
not yet decided: this points toward architecture-side ideas (a CNN
encoder that better resolves compact, high-contrast highlight shapes -
e.g. a higher-resolution downsample path, or an auxiliary loss/input
tied to highlight-region shape) over more grid-resolution or
regularization tuning, but no intervention has been designed or tried
yet. `finalHeldOut` untouched - still not requested.

### Round 4 (2026-09-16): highlight-detail branch - no full resolution, one partial signal

Per instruction: one targeted encoder intervention, grid/slicing
unchanged. `BilateralGridPredictor` gained `use_detail_branch` - a
shallow, separate `[luma, local_highpass]` branch (2 stride-2 layers vs
the main encoder's 5, so far more spatial detail survives), max-pooled
(not averaged - preserves "is there a compact bright feature here"
rather than diluting it) and fused into the grid-coefficient head. +1.6%
params (104,940 → 106,572). `luma_mode` back to `rec709` (round 3's
WB-stable guide isn't part of this intervention). Same 3 seeds, same
schedule as every prior round; evaluated against the frozen baseline with
identical code (`training/run_round4_detailbranch.py`).

**`predeclared_success_met: false`** - the exact same 5-scene union
regressed before and after, zero new regressions. But this is not
identical to round 3's flat result: at the **per-seed** level, `DSCF0553`
(already the one scene that wasn't fully seed-reproducible) improved from
regressing under 2 of 3 seeds (42, 44) to just 1 (44 only) - genuine
movement, not noise, since it's the same content and target, only the
architecture changed. The other three fully-reproducible failures
(`DSCF0590`, `DSCF0873`, `DSCF0879`) showed **zero response** - identical
regression pattern, all 3 seeds, before and after. `DSCF0878` also
unchanged, as expected (not tuned to).

**Read**: the detail branch produced a real but partial and inconclusive
signal - it nudged the one already-marginal case in the right direction
but didn't touch the three scenes with the strongest, most reproducible
failure signature. That's a weaker result than the diagnosis's clean
correlation pattern predicted, and doesn't clear the predeclared bar.
Whether this means the detail-branch idea needs more capacity/a different
fusion point to actually work, or the highlight-shape correlation - like
the WB correlation before it - isn't the real causal lever either, is not
decided here.

Full data: `training/diagnostics/round4-detailbranch-comparison.json`.
`finalHeldOut` untouched - still not requested. Three targeted rounds
in (WB-axis: falsified; structural diagnosis: real correlation, unclear
causation; detail branch: partial, inconclusive) - worth an explicit
decision on whether a fourth targeted attempt is the right use of time
next, versus treating 3-5/76 as this pilot's documented floor.

### Visual inspection (2026-09-16): grid-seam artifacts, not just shape

Per instruction: diagnostic only, no training. Rendered side-by-side
[as-shot input | model output | Fuji target] comparisons - both full-frame
and a crop zoomed on each scene's highlight-region centroid (from the
structural diagnosis) - for the 3 fully-reproducible failures, the
seed-dependent scene, the `DSCF0878` outlier, and 4 matched controls,
using the frozen baseline's best checkpoint (`r2-b-lrsched-seed43-best.pt`).
Actually looked at all of them (`training/diagnostics/visual_inspection/`).

**A consistent, visible artifact pattern showed up in 3 of the 4
reproducible/near-reproducible scenes that neither prior round's metrics
directly measured:**

- **`DSCF0590`** (string-light night scene): a visible **dark ring/halo**
  around the compact bright bulb highlight in the model output - absent
  in the Fuji target, which has a clean, smooth falloff.
- **`DSCF0873`** (sunlit building facade): a distinct **greenish/olive
  color-cast blotch** in the upper-left sky in the model output, against
  an otherwise smooth, uniform blue gradient in the target.
- **`DSCF0553`** (the seed-dependent scene, and under this specific
  checkpoint it doesn't even register as a quantitative regression): the
  *same* greenish-cast blotch appears on a smooth white car fender - the
  artifact is there even when the aggregate L1 metric doesn't flag it,
  because the rest of the frame still nets an improvement over baseline.
- **`DSCF0879`**: less a color-cast artifact, more a **desaturation/
  color-recovery failure** in foliage sitting right next to a blown-out
  bright backdrop - the target shows rich saturated green, the model
  output is washed pale.
- **Matched control (`DSCF0542`)**: clean, accurate, artifact-free -
  confirms these aren't just generic model imperfections.

**Common thread**: every artifact sits at or near a strong *local*
luminance/color transition - a compact highlight's edge, a large smooth
sky gradient, a dark-subject-against-blown-highlight boundary. That
pattern looks like **bilateral-grid discretization/seam artifacts**
(the 8×8 spatial and/or 9-zone luma grid too coarse to represent a smooth
transition, producing a visible seam or ring at a cell boundary) - a
different, more specific mechanism than either round 3's WB-axis theory
(falsified) or round 4's framing (that the *encoder* can't resolve
highlight shape). Round 4's interpolation-sensitivity test only checked
the luma axis in isolation; it never tested spatial (x,y) grid-sampling
sensitivity, which this visual pattern implicates directly.

**`DSCF0878` confirmed as a genuinely different failure**, not degree but
kind: its crop shows the model **overexposing a large region to
near-white**, losing real detail (a hanging lamp, foliage, people) that
both the input and the Fuji target clearly retain - wholesale clipping,
not a localized seam. Consistent with its already-known outlier stats
(highest bright-fraction, highest interpolation-sensitivity of any scene
checked). Kept separate, as instructed.

**Not yet tested**: a spatial-grid-resolution/seam-targeted intervention
(finer spatial grid cells, or a spatial-smoothness prior specifically
tied to detected sharp transitions, as opposed to round 2's blanket TV
regularization which didn't target transitions specifically and made
things worse). No training was run for this pass - purely diagnostic, per
instruction. `finalHeldOut` untouched throughout - independently
verified nothing in this pass touched it (evaluation restricted to the
same `Phase4PairDataset("monitor")` loader every other round used).

### Round 5 (2026-09-16/17): finer spatial grid - clean negative, worse on the targeted metric

Per instruction: tested the visual seam diagnosis directly. Grid
8×8→16×16, luma axis/loss/encoder all held fixed (identical param count,
104,940, either way - grid_spatial only changes the head's output
spatial size via pooling, not capacity). Added crop-level L1 (error
within each scene's highlight-region crop) and a seam score (residual-
error gradient at grid-cell boundaries vs frame average, computed at
each checkpoint's own resolution) specifically to test this hypothesis
beyond whole-frame metrics (`training/run_round5_finegrid.py`).

**`predeclared_success_met: false`** - same 5-scene union regressed, zero
resolved. Worse than flat, though:

- **Crop-level L1 got worse on every single flagged scene** - not noise:
  `DSCF0873` +29%, `DSCF0878` +46%, smaller increases on the rest. The
  finer grid made the specific defect regions worse, not better.
- **`DSCF0553` actively regressed further** - now fails under all 3 seeds
  (up from 2/3 in the baseline), the wrong direction from round 4's
  partial improvement on this exact scene.
- **Seam scores were mixed, not a clean win**: `DSCF0590` dropped
  meaningfully (1.06→0.94, genuinely less boundary-elevated error) but
  `DSCF0553` moved the wrong way (0.94→0.97) and the rest barely changed.
  No consistent signal that finer resolution reduces boundary artifacts.
- Aggregate training also looked qualitatively different from every prior
  round: seed 43 converged unusually fast and shallow (best epoch 2,
  early-stopped at 7) then plateaued - some sign the finer grid is harder
  to fit well under the same training budget/schedule, not just a
  different local optimum.

**Read**: this doesn't just fail to confirm the seam-discretization
theory, it argues against "finer grid resolution alone" as the fix - the
visually-identified defect regions got measurably worse, most sharply for
the two hardest-known scenes (`DSCF0873`, `DSCF0878`). If grid coarseness
is part of the real mechanism, more cells with the same encoder/training
budget isn't the right lever - possibly needs more training capacity/time
to actually fit the finer grid well, or the mechanism is something else
entirely (a fourth falsified-or-inconclusive avenue, after WB-axis and
detail-branch).

Full data: `training/diagnostics/round5-finegrid-comparison.json`.
`finalHeldOut` untouched - still not requested. Four targeted
interventions in now (WB-axis: falsified; detail branch: partial;
finer grid: worse on the metric designed to test it) plus one structural
diagnosis and one visual inspection - worth a frank decision on whether
a fifth attempt is justified, or this is genuinely the pilot's floor.

### Round 6 (2026-09-17): hybrid grid + bounded refinement net

Per instruction: round 5 closed the bilateral-grid-only tuning branch
(WB guide, TV smoothness, encoder detail, grid resolution all tested
without a reliable fix). This is a genuinely different component, not
another grid tweak - `model.HybridPredictor` composes the unchanged
8×8×9 grid with `model.RefinementNet`, a small (10,635 param) multiscale
U-Net at full working resolution with a high-resolution skip connection,
trained jointly end-to-end. Bounded and identity-safe by construction:
zero-initialized final layer (exact no-op at training start, verified),
residual hard-clamped to ±0.08 via tanh regardless of what the network
learns - it cannot freely rewrite an already-correct photo.

Cleared Codex's full ONNX/ONNX Runtime Web export gate before training
(dynamic-shape export, checker, native-runtime parity to 2.09e-6,
real-browser WASM/WebGPU) - two real export blockers found and fixed
along the way: antialiased resize doesn't export to ONNX opset 20
(dropped antialiasing from `resize_for_network` entirely, verified still
MPS-safe), and `adaptive_avg_pool2d` can't export under dynamic input
shapes even though it's a true no-op in the approved config (resolved
once at construction time from static ints, not runtime tensor shape -
verified bit-identical output). Same 3 seeds, same schedule, same
evaluation code as every prior round (`training/run_round6_hybrid.py`).

**`predeclared_success_met: false`** by the strict criterion (fully
resolve a scene across all 3 seeds) - same 5-scene union regressed, zero
new regressions. But the pattern underneath is different from every
prior round, and the most promising yet:

- **`DSCF0553` improved substantially**: regressed under 2/3 seeds before
  (42, 44) → only 1/3 after (43 - a *different* seed than either that
  failed before). Real movement, not noise.
- **Crop-level L1 (the actual flagged defect region) improved on 4 of 5
  scenes**, including the outlier: `DSCF0553` −8%, `DSCF0590` −6%,
  `DSCF0879` −11%, `DSCF0878` **−31%**. `DSCF0873` was the one exception,
  +4% worse. Nothing like round 5, which made every single scene's
  crop-level error worse.
- **Seam scores stayed roughly flat** - no clear reduction either
  direction - suggesting the refinement net is doing general local
  correction rather than specifically un-seaming grid boundaries, but
  it's helping the actual defect regions regardless of mechanism.
- **Applied correction stayed modest**: mean |delta| ~0.01-0.02 across
  flagged scenes, well under the 0.08 hard bound - not saturating, a
  good sign the identity-safe design is working as intended rather than
  the network just maxing out its allowed range everywhere.

**Read**: doesn't clear the predeclared bar, but this is the first
intervention across six rounds that improved the *targeted* defect
metric on most flagged scenes without making any of them worse in
aggregate, and moved the closest-to-resolved scene meaningfully closer.
Whether that's enough to justify iterating on this same architecture
(more capacity, longer training, a larger max_delta) rather than treating
it as inconclusive like the detail-branch round, is not decided here.

Full data: `training/diagnostics/round6-hybrid-comparison.json`.
`finalHeldOut` untouched - still not requested. Per instruction, any
selected hybrid checkpoint repeats the full export gate before app
integration - not done, since nothing has been selected as final.

### Round 7 (2026-09-17): refinement capacity 8→16 - one controlled
follow-up, comparison target round 6 itself

Per instruction: round 6 was the first architecture with a real positive
signal, so before requesting `finalHeldOut`, one isolated follow-up -
double the refinement branch's capacity only (`refinement_base_ch` 8→16,
115,575 → 145,791 total params). The 0.08 residual bound and the
training schedule were both left unchanged per instruction, since
round 6's applied delta wasn't saturating (~0.01–0.02 vs the 0.08 cap) -
increasing the bound wasn't justified yet, and this round was scoped to
test capacity alone, not combined with a bound or schedule change. Same
3 seeds, same evaluation code. Predeclared success: fewer persistent
union-of-seed regressions than round 6's own result, AND no loss of
round 6's crop-level gains on the 4 reproducible scenes (`DSCF0878`
tracked separately, excluded from the criterion, per instruction).
(`training/run_round7_hybridcapacity.py`.)

**`predeclared_success_met: false`.** `fewer_persistent_regressions:
false` - the union-of-seed regressed set is *identical* to round 6's,
scene for scene: `DSCF0553`, `DSCF0590`, `DSCF0873`, `DSCF0878`,
`DSCF0879`. Doubling refinement capacity resolved nothing and introduced
no new regressions either. `no_lost_crop_gains: true` - crop-level L1 on
all 4 reproducible scenes continued to improve slightly versus round 6
(not just held steady):

- `DSCF0553`: 0.1381 → 0.1371 (small further gain)
- `DSCF0590`: 0.1902 → 0.1843 (further gain)
- `DSCF0873`: 0.2001 → 0.1970 (further gain, but still worse than the
  pre-hybrid baseline's 0.1921 - this scene has resisted every round to
  date, hybrid included)
- `DSCF0879`: 0.1790 → 0.1779 (further gain)
- `DSCF0878` (outlier, not counted): 0.0746 → 0.0807, *worse* than
  round 6 though still far better than the pre-hybrid baseline's 0.1075
  - `crop_l1_lost_gain_vs_round6: true`, reported per instruction, not
  counted toward the verdict.

Applied correction stayed modest and did not saturate with the added
capacity: mean |delta| 0.017–0.035 across flagged scenes, still well
under the 0.08 bound.

**Read**: capacity was not the bottleneck. The refinement branch already
had enough representational power at `base_ch=8` to do what it's doing;
doubling it bought a small further crop-level improvement on the scenes
it was already helping, but didn't touch the scenes it wasn't - the
persistent-regression union is bit-for-bit the same 5 scenes as round 6.
Per instruction, this closes the capacity branch and hands the decision
to schedule/loss as the next isolated variable to test, rather than
combining that with another capacity change. Not started without
explicit authorization.

Full data: `training/diagnostics/round7-hybridcapacity-comparison.json`.
`finalHeldOut` untouched - still not requested; this script has no
`finalHeldOut` code path (only `Phase4PairDataset("monitor")` is
constructed), independently verified by grep on the run log and script.

### Round 8 (2026-09-17/18): schedule intervention - regressed, closes
the tuning branch per instruction

Per instruction: round 7 closed the capacity branch cleanly (identical
5-scene union, capacity wasn't the bottleneck). Before trying a loss
change, one isolated schedule intervention - keep round 6's architecture
exactly (`refinement_base_ch=8`, reverted from round 7's 16; `max_delta`
0.08 unchanged), only `epochs` 20→40. Because `CosineAnnealingLR`'s
`T_max` tracks `args.epochs`, this also halves the LR decay rate per
epoch; early-stopping patience (5, on monitor L1) was left unchanged.
Rationale going in: round 7's logs showed training using the full
20-epoch budget with train L1 still falling when patience forced a stop,
with LR already decayed to ~1e-5 by then - the schedule appeared to be
truncating training before genuine convergence. (`training/run_round8_longschedule.py`.)

**This was wrong, and the result is a clean negative - not just
inconclusive, actively worse than round 6 on every flagged scene.**
`predeclared_success_met: false`, `no_lost_crop_gains: false` - all four
reproducible scenes lost ground versus round 6:

- `DSCF0553`: 0.1381 → 0.1451 crop L1, *and* moved backward on seed
  count - regressed under 2/3 seeds (42, 43) versus round 6's 1/3 (43
  only). This was the one scene showing real improvement through round
  6/7; round 8 undid it.
- `DSCF0590`: 0.1902 → 0.1929 (worse)
- `DSCF0873`: 0.2001 → **0.2477** (worse by ~24% relative, its worst
  result across every round to date, well past the pre-hybrid
  baseline's 0.1921)
- `DSCF0879`: 0.1790 → 0.1883 (worse)
- `DSCF0878` (outlier, excluded from the criterion): 0.0746 → 0.0940
  (worse)

Union-of-seed regressed scenes stayed the identical 5 (no resolutions,
no new regressions in that binary sense) - but the schedule change is
unambiguously net-negative on the metric that actually matters here.

**Mechanism, found in the per-seed epoch histories** (not left as an
unexplained negative): the premise was backwards. Doubling `T_max`
without touching patience doesn't buy the model more effective
optimization time - it extends the noisy high-LR phase without
extending the patience budget to match, so two of three seeds ran out of
patience *while still in that noisy phase*, before ever reaching the
lower-LR regime that produced round 6's actual gains:

- **seed 42**: stopped at epoch 8 (best epoch 3, LR still ~9.9e-4,
  essentially undecayed) - best monitor L1 0.0393 versus round 6 seed
  42's 0.0361 at its best epoch 14 (LR 2.1e-4).
- **seed 43**: stopped at epoch 15 (best epoch 10, LR ~8.6e-4) - best
  monitor L1 0.0388 versus round 6 seed 43's 0.0366 at best epoch 18
  (LR 3.4e-5).
- **seed 44**: the one seed that ran long enough (33 epochs, best epoch
  28) to land close to round 6's own result (0.0361 vs round 6's
  0.0361) - confirming the mechanism: this seed got the outcome the
  intervention intended only because it happened to keep finding new
  bests long enough to outlast where the other two seeds' patience ran
  out.

In round 6/7's `T_max=20` schedule, LR was already down to ~6e-4 to
2e-4 by epochs 8-14, i.e. seeds 42/43 previously did their real
converging inside the window where round 8's slower schedule kept LR
near its initial value. Patience counts epochs, not schedule progress -
doubling the schedule without doubling (or otherwise scaling) patience
systematically shortchanges exactly the runs that would have needed the
extra time.

**Read**: this is a design mistake in the intervention, not evidence
against "more training time" as a hypothesis per se - a version that
scaled patience alongside `T_max` might behave differently. But per
instruction, this is a stop point regardless: the five-scene union is
still unresolved, and this round leaves the hybrid strictly worse off
than round 6 on the metric being tracked, not just flat. Continuing to
tune architecture/schedule blind is exactly what the instruction called
out to stop doing once this outcome landed. Next step per instruction is
a new diagnostic or corpus-level decision, not a loss-function change -
not started without explicit authorization. Round 6's checkpoints
remain the best result to date and are unaffected by this round.

Full data: `training/diagnostics/round8-longschedule-comparison.json`.
`finalHeldOut` untouched - still not requested; this script has no
`finalHeldOut` code path, independently verified by grep on the run log
and script.

### Decision (2026-09-18): stop point reached - round 6 is the current
best, corpus-level decision next

Per instruction, following round 8's regression: **round 6's hybrid
checkpoints (`r6-hybrid-seed{42,43,44}-best.pt`) are the current best
result and stay selected.** Round 8's longer-schedule result is not
eligible for selection - its early-stopping patience was mismatched to
its doubled schedule length (see round 8 above for the mechanism), and
it regressed the crop-level metric on every one of the 4 reproducible
hard scenes versus round 6, not just failed to improve on them. (Round
7, the capacity-only round, is separately closed as neutral - identical
5-scene union, small further crop-level gains, no regressions - it
simply isn't the round that regressed; that was round 8.)

**Blind architecture/schedule tuning stops here**, per instruction. No
further rounds without a new diagnostic or corpus-level decision behind
them. `finalHeldOut` is not being requested. Instead: document the five
persistent failure-pattern scenes precisely, and use that to scope a
targeted future capture plan - new material exported through the same
locked Fuji/X RAW Studio/browser protocol used for the existing corpus,
not ad hoc images, before any further model round is attempted.

**The five persistent-regression scenes, precisely** (from
`training/diagnostics/structural_diagnosis.json` and
`training/diagnostics/regression_diagnosis.json` - every round 2-8 has
regressed on the union of exactly these five, regardless of
architecture):

| scene | session | highlightClass | p99 brightness | largest-highlight compactness | largest-highlight area | near-black fraction |
|---|---|---|---|---|---|---|
| `DSCF0590.RAF` | 5 | bright-highlight | 0.804 | 0.75 (very compact) | 0.11% of frame | 66.0% |
| `DSCF0878.RAF` (outlier) | 10 | bright-highlight | 0.964 (near-clipped) | 0.33 (least compact) | 0.22% of frame | 19.8% |
| `DSCF0879.RAF` | 10 | mixed-lighting | 0.361 | 0.62 (compact) | 0.21% of frame, near frame edge | 39.3% |
| `DSCF0553.RAF` | 4 | mixed-lighting | 0.429 | 0.37 (elongated, aspect 1.61) | 0.41% of frame | 10.2% |
| `DSCF0873.RAF` | 9 | ordinary (not a highlight scene) | 0.176 | 0.54 | 0.13% of frame | 39.3% |

All five are Auto-WB (`wbManualFlag: 0`), confirming the 2026-09-15
diagnosis holds through every subsequent round - manual/Kelvin-WB scenes
have never been the problem. They span only 4 of the corpus's 14 real
capture sessions (5, 9, 10, and 4), all inside the monitor tier - not
because those sessions are special, but because that's the slice these
particular scenes happen to sit in; nothing rules out the same failure
mode existing, unflagged, somewhere in the 561-scene train tier too.

This maps onto three genuine patterns plus one open anomaly, not four
equally-weighted categories:

1. **Compact bright highlight against a smooth, mostly dark/mid
   surface** - `DSCF0590` (very compact, 66% near-black frame) and
   `DSCF0879` (compact, highlight near the frame edge) fit this
   cleanly. This is the pattern round 4's detail branch and round 5's
   finer grid both targeted without a reliable fix, and where the
   hybrid's bounded refinement net (round 6) made its clearest genuine
   progress on `DSCF0553` before round 8 undid it.
2. **Broad, low-compactness clipped bright region** - `DSCF0878` is the
   distinct case: least-compact of the five, highest near-clipped p99
   (0.964) and highest bright-pixel fraction. Consistently the single
   worst-behaved scene by raw magnitude across every round, though also
   the scene the hybrid helped most in relative terms (round 6: -31%
   crop L1 vs the pre-hybrid baseline) - reported separately throughout
   per instruction, and should stay a separate reporting bucket in any
   future round, not folded into the other four.
3. **Auto-WB lighting generally, across a wide brightness range** - the
   cross-cutting factor tying all five together; not just bright scenes
   (see the open anomaly below) but the WB estimation path itself,
   which the round 3 WB-stable-guide experiment falsified as the direct
   mechanism (the luma-guide coordinate wasn't it) without ruling out
   Auto-WB as a *correlated* corpus-composition signal - i.e. whatever
   is actually wrong is more common in, or harder in, Auto-WB captures,
   even though it isn't the WB guide computation per se.
4. **Open anomaly - `DSCF0873` does not fit the highlight story at
   all.** `highlightClass: ordinary`, p99 0.176 (not remotely clipped or
   bright), `bright_fraction` effectively zero. It's a low-key,
   Auto-WB, mid-brightness frame that has gotten *worse*, not better,
   across every hybrid round to date (crop L1: baseline 0.192 → round 6
   0.200 → round 7 0.197 → round 8 0.248, its worst result of any
   round). None of the "bright highlight" framing that explains the
   other four applies here. This scene should not be assumed to belong
   to the same failure family as the other four just because it's in
   the same persistent-regression union - it may need its own
   diagnostic pass, not just more examples of its type in the capture
   plan below (its type isn't even well-characterized yet).

**Targeted future capture plan** - once approved, new examples should
specifically add:

- Compact bright highlights against smooth surfaces (pattern 1) -
  varied surface types/textures, not just repeats of the existing two.
- Broad clipped bright regions (pattern 2) - more examples like
  `DSCF0878`'s near-clipped, low-compactness highlight shape, since it's
  currently a single-scene category.
- Varied Auto-WB lighting conditions generally (pattern 3) - broader
  color-temperature and mixed-source lighting coverage under Auto-WB,
  not concentrated in one or two sessions.
- More locations/sessions overall - the existing corpus (14 sessions,
  Egypt+Spain only) is the entire known Fuji-RAF library; genuinely new
  material means new capture days/locations, not a re-split of what
  already exists.
- A still-open question, not yet a capture target: what `DSCF0873`
  actually represents. Needs its own look before it can be turned into
  a capture spec.

Any new material must go through the same locked export protocol as the
existing 731-file corpus (`.fjlrg` container, as-shot WB, sRGB
primaries, X RAW Studio/browser export pipeline - see
`docs/phase3-export-protocol.md` and `docs/phase3-linear-rgb-format.md`)
before it's usable for another model round. Which tier(s) new material
would join (train/monitor, or held to the same finalHeldOut discipline)
is not decided here - a decision for whoever scopes the actual capture
work.

**Honest framing, per instruction**: the round 6 hybrid pilot is a real,
meaningfully better result overall than Phase 3's hand-fit regression -
substantially lower L1 across train and monitor, and genuine (if
partial) progress on the previously-intractable highlight-defect
scenes. It is explicitly **not** being represented as pixel-perfect Fuji
parity - five scenes in the 76-scene monitor tier still regress under
every architecture tried, `DSCF0878` remains the single worst-behaved
scene by magnitude, and `DSCF0873` isn't even understood yet.
`finalHeldOut` (94 scenes) remains completely untouched throughout all
eight rounds - independently verified after every round, most recently
after round 8.

## Capture batch 2 (2026-11-12): first targeted material against the
## capture plan above - export complete, not yet in any tier

38 new scenes captured and exported through the locked protocol,
targeting the capture plan's patterns 1-3 (compact bright highlights,
broad clipped bright regions, varied Auto-WB lighting) - 35 were
deliberately planned from a shot list built against those patterns; 3
extra frames ended up in the batch and were kept rather than pruned
(more real examples is fine, no reason to discard). Two planned shots
(a café-interior Auto/manual-WB pair, and a snow-covered-ground daylight
shot) were not taken and are not expected - the batch is treated as
complete at 38, not pending those two.

- Folders: `calibration-input/Shoot 706` through `Shoot 743`, continuing
  the corpus's existing numbering (previously ended at `Shoot 705`).
  Manifest: `calibration-input/phase4-capture2-manifest.json` (filename
  -> Shoot folder).
- X RAW Studio side: 16-bit TIFF export (confirmed available this
  session), locked settings table unchanged -
  `xraw-phase4-capture2.tiff` in each folder.
- Browser side: `browser-phase4-capture2-linear.fjlrg` in each folder,
  `?rawCalibrationWb=camera`, 1536px long edge - same container as the
  existing 731-file corpus. Verified 38/38 present, all the expected
  fixed byte size for this resolution class, zero missing.
- `scripts/export-phase3-linear.mjs` was generalized to take an
  approved-export-list manifest via `FUJIAPP_EXPORT_MANIFEST` and a
  custom output filename via `FUJIAPP_EXPORT_FILENAME`, replacing the
  old hardcoded Shoot-number-range logic. It accepts either the
  original `{shoot, name, tier}` schema or the capture-batch
  `{shoot_folder, filename}` schema, and still hard-refuses any row
  tagged `finalHeldOut` - the lockout isn't weakened by the schema
  change. Not yet committed as of this note (was sitting in the shared
  working tree with other in-progress Phase 4 edits); syntax-checked and
  proven working against this real 38-scene batch.
- Not yet decided: which tier(s) (train/monitor, or a finalHeldOut-style
  hold) this batch joins, and whether/when a model round runs against
  it. Per the stop-point decision above, no training round is implied
  by this capture work being done - that's a separate decision for
  later, once it's clear these 38 scenes actually cover the intended
  patterns rather than just assumed to.

## Capture batch 2 structural diagnostic (2026-11-12) - diagnostic only,
## no training, no finalHeldOut access

Checked whether the 38 scenes actually cover the capture plan's intended
patterns, rather than assuming they do. Method: `scripts/extract-capture2-exif.mjs`
pulls capture timestamp/ISO/WB-mode straight from each RAF's embedded-preview
EXIF (the RAF container's own embedded JPEG, found by scanning for its
SOI/EOI markers rather than hardcoding RAF-header offsets); `training/diagnose_capture2.py`
reuses the exact same highlight-shape/luma-distribution feature code
(`highlight_shape_stats`/`luma_distribution_stats` from `diagnose_structural.py`)
already used to characterize the 5 persistent-regression scenes, so the
numbers below are directly comparable to that table. Output:
`training/diagnostics/capture2-structural-diagnosis.json`.

**Sessions: 2, not "more locations/sessions."** Applying the corpus's own
rule (`>3h` capture-timestamp gap = new session) gives **37 scenes in one
continuous session** (2026-11-11 18:56 - 2026-11-12 01:03, one evening)
plus **1 isolated scene** (`DSCF1423`, shot ~15h later). The capture
plan's fourth point - "more locations/sessions overall... new capture
days/locations" - is **not met** by this batch. Patterns 1-3 may still be
served (see below), but this is effectively one new session, not several,
and that should be weighed honestly against the plan's own stated goal,
not glossed over because the file count (38) looks substantial.

**Pattern 2 (broad, low-compactness clipped bright region) - the
strongest result.** `DSCF1413/1414/1415/1416` (23:33-23:35, one tight
cluster) all show near-total clipping (p99 0.999-1.000) with low
compactness (0.15-0.37) and bright-pixel fractions of **13-44%** -
3 to 12x higher than the single existing example of this pattern
(`DSCF0878`, bright_fraction 0.038). `DSCF1387/1388` (p99 ~0.998,
compactness 0.24-0.31, bright_fraction ~0.02) are a more moderate second
match. This directly answers the capture plan's ask for more examples of
a "currently single-scene category" - assuming visual inspection confirms
these are genuinely the intended blown-highlight/bright-region shots and
not, say, a mis-exposed accident.

**Pattern 1 (compact bright highlight against a smooth surface) - plausible
but weaker.** `DSCF1379` (p99 0.70, compactness 0.74, 5 components),
`DSCF1412` (p99 0.92, compactness 0.80), and `DSCF1417` (p99 0.78,
compactness 0.71, small area fraction) are the best candidates - high
compactness and high brightness together, few connected components. Less
unambiguous than pattern 2's cluster; worth a visual check before relying
on them.

**Pattern 3 (varied Auto-WB lighting) - partially supported, not fully
verified.** 26 Auto-WB / 12 Manual-Kelvin-WB split, with ISO spanning
125-10000 across the session (implying real exposure/lighting variety).
But WB *mode* alone doesn't confirm varied *color temperature* within the
Auto-WB shots - that would need the camera's resolved WB gain values,
which weren't extracted here. Treat this pattern as directionally
supported, not confirmed.

**WB-contrast pairs: ~12-14 candidate transitions detected, not a clean
12.** Adjacent-in-time, opposite-WB-mode transitions: `1379/1381`,
`1384/1385`, `1386/1387`, `1388/1389`, `1393/1394`, `1401/1402/1403`
(a 3-frame Auto-Kelvin-Auto chain, not a clean pair - one of the two
Auto frames here is probably an extra, not part of a pair),
`1405/1406`, `1409/1410`, `1413/1414/1415/1416` (overlaps with the
pattern-2 cluster above), `1417/1418`, `1419/1420`. 12 Kelvin shots
exist against the plan's 10 intended non-cafe pairs, consistent with the
"3 extra frames" already noted in the batch-1 entry above.

**No accidental duplicates detected** under a strict same-WB/<20s-gap/
near-identical-structure test. Two tight clusters are worth the
photographer's own quick visual glance, not flagged as defects outright
since the structural numbers alone can't fully rule out legitimate
distinct framing: `1391/1392/1395` (19:26-19:32, moderate/unremarkable
stats, no clear pattern-1/2 match) and `1396/1397/1398` (19:35-19:38,
similarly unremarkable, low p99). `DSCF1423` (the lone session-2 frame)
is also unremarkable on its own (p99 0.30) and adds little by itself
regardless of where it ends up.

**Proposed tier assignment, for approval - not yet applied:** add **all
38 scenes to the `train` tier**, leaving `monitor` and `finalHeldOut`
exactly as they are. Reasoning: every round-2-through-8 comparison in
this document was computed against the *same* monitor composition (76
scenes) - changing that composition now would break comparability with
every L1 number already recorded, including the round 6/7/8 decision
above. Adding this material to `train` instead feeds the genuinely novel
pattern-2/pattern-1 examples into fitting directly, without touching the
yardstick used to judge whether a future round actually helps. The
tradeoff: this means no round can be evaluated specifically *against*
these new hard cases via the monitor mechanism - a deliberate choice,
not an oversight, and reversible later if there's a reason to carve out
a different split.

**Approved and applied (2026-11-12).** All 38 scenes added to the `train`
tier only: `calibration-input/phase4-corpus-manifest.json`'s train tier
went from 561 to 599 scenes, with session indices 14 (37 scenes) and 15
(1 scene, `DSCF1423`) added to `train.sessionIndices`. `monitor` (76) and
`finalHeldOut` (94) are unchanged - confirmed by direct count, not just
by not having touched that code path. Rationale: this batch's 2-session
structure (see above) isn't strong enough to justify a new, trustworthy
evaluation tier - approved explicitly on that basis, not merely to avoid
disturbing comparability.

To make the new scenes actually loadable without any training-code
change, each of the 38 `Shoot N` folders got two symlinks pointing at its
already-exported files: `xraw-phase3.tiff` -> `xraw-phase4-capture2.tiff`
and `browser-phase3-linear.fjlrg` -> `browser-phase4-capture2-linear.fjlrg`
(both `dataset.py` and `precompute_targets.py` hardcode those literal
filenames). Verified end-to-end with `training/dataset.py`:
`list_scenes('train')` now returns 599 scenes including all 38, and
`Phase4PairDataset` successfully loads one of them (`DSCF1379.RAF`,
input/target both `3x1536x1026`) - a read-only sanity check, not a
training run. `precompute_targets.py` has **not** been run for these 38
scenes yet - that's a real prerequisite before any training round
actually uses them (`Phase4PairDataset`'s default
`require_cached_target=True` will raise `FileNotFoundError` on them
otherwise), left undone deliberately since no round is approved yet.
`finalHeldOut` was not read by any script in this diagnostic or
manifest-update pass.

Note: `calibration-input/` (including the manifest) is entirely
gitignored - this update lives only on the shared external drive, not in
a commit. Since Codex's working copy is the same physical drive/files
(confirmed during the exFAT `core.fileMode` fix above), this is already
visible there too, not something that needs separate syncing.

## Proposed Round 9 scope (2026-11-12) - NOT launched, awaiting approval

Per instruction, this is a proposal only - no training has run.

**Isolated change: train-tier composition only, nothing else.** Round 6's
exact hybrid config, unchanged: `grid_spatial=8, grid_luma=9, base_ch=16,
lr=1e-3, epochs=20, cosine schedule, early_stopping_patience=5,
use_hybrid=True, refinement_base_ch=8, refinement_max_delta=0.08`. Same 3
seeds (42, 43, 44). The only thing that differs from round 6 is that
`train` now has 599 scenes instead of 561 (the 38 capture-batch-2
additions) - no architecture, schedule, or loss change, consistent with
this project's one-variable-per-round discipline. This tests the most
direct hypothesis available: does the model improve on the known hard
patterns simply by seeing real examples of them during fitting, now that
such examples exist, without changing the model itself.

**Comparison target: round 6, on the unchanged monitor tier (76 scenes)** -
not round 7 or 8, both closed/unsuccessful branches, and not the
pre-hybrid baseline. `monitor` membership is exactly what it was for
every prior round, so this stays comparable to the existing round 2-8
numbers.

**Predeclared success criteria (checked before run, verified after,
exactly as every prior round):**
1. Fewer persistent union-of-seed regressions on `monitor` than round 6's
   5-scene union, AND no loss of round 6's crop-level gains on the 4
   reproducible scenes (`DSCF0553`, `DSCF0590`, `DSCF0873`, `DSCF0879`) -
   the same bar rounds 7 and 8 were held to.
2. `DSCF0878` reported separately, as always, not counted toward
   success/failure.
3. **Secondary, non-blocking metric to watch, not a gate:** `DSCF0878`'s
   crop-level L1, specifically because capture batch 2's strongest
   contribution (`DSCF1413-1416`) is more extreme examples of exactly
   `DSCF0878`'s failure pattern (broad, low-compactness, near-clipped
   highlights). If round 9 helps `DSCF0878` noticeably despite not being
   scored on it, that's meaningful signal about *why* it helped, even if
   criterion 1 doesn't clear on its own.

**If this doesn't clear criterion 1**: per the project's own established
practice (see round 7/8 above), the next step is back to diagnosis - not
immediately trying a different model/schedule change on top of the
expanded data, and not expanding the capture plan further without first
understanding why more relevant training data didn't help.

**Scope notes:** `precompute_targets.py` needs to run for the 38 new
scenes before any training loop can actually use them (see note above) -
that's a mechanical prerequisite, not itself a training step, and would
happen at launch time, not before approval. No code change to
`train.py`/`model.py`/`dataset.py` is needed. `finalHeldOut` has no code
path in this round, as always.

Approved and launched 2026-11-12 (relayed via Codex/user). Results below.

## Round 9 results (2026-11-12)

Ran exactly as proposed: `training/run_round9_expandeddata.py`, round 6's
hybrid config completely unchanged, same 3 seeds (42, 43, 44), trained on
the expanded 599-scene `train` tier, evaluated on the unchanged 76-scene
`monitor` tier. `finalHeldOut` was not accessed. Per-seed training:
seed 42 - 20/20 epochs, 10761s, best monitor L1 0.04074; seed 43 - 20/20
epochs, 10381s, best monitor L1 0.03930; seed 44 - 18/20 epochs (early
stop), 9356s, best monitor L1 0.03780. Full comparison output:
`training/diagnostics/round9-expandeddata-comparison.json`.

**Criterion 1 (fewer persistent union-of-seed regressions + no lost
crop-level gains): met.**
- Round 6's union-of-seed regression set on `monitor` was `DSCF0553,
  DSCF0590, DSCF0873, DSCF0878, DSCF0879` (5 scenes). Round 9's union is
  `DSCF0590, DSCF0873, DSCF0878, DSCF0879` (4 scenes) -
  **`DSCF0553` is resolved**, no new regressions appeared anywhere else
  in `monitor`.
- Crop-level L1 on all 4 reproducible scenes (`DSCF0553`, `DSCF0590`,
  `DSCF0873`, `DSCF0879`) improved vs round 6, none regressed:
  `DSCF0553` 0.1381->0.1078, `DSCF0590` 0.1902->0.1693, `DSCF0873`
  0.2001->0.1539, `DSCF0879` 0.1790->0.1566.

**Criterion 2/3 (`DSCF0878`, secondary/non-blocking):** still regressed
on all 3 seeds (not counted toward success, as predeclared), but its
crop-level L1 also improved vs round 6: 0.0746->0.0689. This is
consistent with the hypothesis in the proposal above - capture batch 2's
strongest cluster (`DSCF1413-1416`) is more extreme examples of
`DSCF0878`'s own failure pattern, and the expanded data seems to have
helped it somewhat even though it wasn't enough to fully resolve it.

**Verdict: `selected_best_model = "round9"`.** Per the predeclared
retention rule, round 9 replaces round 6 as the selected-best model -
expanding `train` with real examples of the known hard categories (no
architecture/schedule change) measurably helped, resolving one of round
6's 5 persistent regressions outright and improving crop-level accuracy
on the rest, with zero new regressions introduced. Round 6's checkpoints
remain on disk but are no longer the selection.

**Next step:** none launched yet - this closes out capture batch 2's
intended use. Any further round should be scoped and proposed
separately, following the same one-variable, predeclared-criteria
discipline as every round above.

## Round 9 remaining-regression diagnosis (2026-11-12) - diagnostic only, no training, no finalHeldOut

Per instruction: keep round 9 as selected-best; before proposing round
10, characterize what round 9 actually changed on the 4 scenes still in
its union-of-seed regression set (`DSCF0590`, `DSCF0873`, `DSCF0878`,
`DSCF0879` - `DSCF0553` is excluded, round 9 resolved it). Inference-only
on the unchanged monitor tier, comparing round 6 and round 9's existing
checkpoints; no new training, no checkpoint selection change, no
`finalHeldOut` access. New script: `training/diagnose_round9_remaining.py`.
Full data: `training/diagnostics/round9-remaining-diagnosis.json`.

**What round 9 improved:** the mean residual (|pred-target|) dropped on
all 4 scenes (`DSCF0590` 0.0616->0.0561, `DSCF0873` 0.0901->0.0747,
`DSCF0878` 0.1132->0.0945, `DSCF0879` 0.0772->0.0705) - consistent with
the crop-L1 gains already reported above, now confirmed over the full
frame, not just the flagged crop. On `DSCF0879` specifically, the
worst-error region also got far more spatially concentrated and moved
right onto the scene's own known highlight location (compactness
0.210->0.091, distance-to-known-highlight 0.205->0.089) - the broader
surrounding error round 6 had is gone, leaving a small, sharp residual
exactly at the highlight boundary. `DSCF0590` shows the same direction,
more mildly.

**What remains wrong - one consistent systematic bias across all 4
scenes, in both round 6 and round 9:** every scene's *predicted* image
has a smaller near-black fraction and a larger bright fraction than its
*target*, i.e. the model isn't pushing shadows as dark or highlights as
bright as Fuji's actual rendering - it under-compresses dynamic range at
both tails. Concretely (pred -> target, round 6, round 9 moves slightly
toward target but never closes the gap):
- `DSCF0590`: near-black 0.636/0.639 vs target 0.718; bright 0.103/0.090 vs target 0.051
- `DSCF0873`: near-black 0.219/0.234 vs target 0.731; bright 0.052/0.016 vs target 0.001
- `DSCF0878`: near-black 0.132/0.158 vs target 0.336; bright 0.120/0.101 vs target 0.076
- `DSCF0879`: near-black 0.319/0.332 vs target 0.616; bright 0.053/0.047 vs target 0.025

`pred_luma_std` is likewise higher than the target's own std in every
case - the model's output has a wider dynamic range than Fuji's actual
output, which is the same effect stated differently: not enough
highlight/shadow compression toward the midtones. This is the same
direction of error on all 4 scenes and on both checkpoints; round 9
reduced its magnitude (via more training data of similar scenes) without
changing its direction - consistent with a representational limit rather
than something more same-shaped data alone fixes.

`DSCF0878` is the exception on location, as in every prior round: its
worst-error region sits far from its own "known highlight" centroid
(normalized distance ~0.87-0.90, round 6 and round 9 alike, barely
moved) - whatever is wrong with it is still not the same localized
highlight-shape mechanism as the other 3.

**One evidence-based next intervention (proposal only - NOT launched,
awaiting approval):** increase `grid_luma` (currently 9) - the bilateral
grid's luma axis resolution, which is what determines how finely the
model can apply *different* corrections to true shadows vs true
highlights vs midtones, rather than one smoothed-out correction across
the whole brightness range. This is the one isolated grid axis never yet
tested alone: round 5 varied `grid_spatial` (8->16) with `grid_luma`
explicitly held fixed at 9, and found a *spatial*-resolution increase
made things worse, not better - but that is evidence about the spatial
axis, not the luma axis, and the defect found here (a brightness-axis
compression bias, present even in full-frame stats away from any
boundary) was never the thing round 5's seam-score test was built to
catch. Raising `refinement_max_delta` (currently 0.08) is explicitly
*not* recommended instead: round 6's own data already showed the
refinement net's actual correction magnitude (mean |delta| ~0.01-0.02)
is well under that bound, i.e. it isn't saturating against its clamp, so
loosening the clamp has no mechanism to help this specific bias.

Proposed round 10 (scope only, not approved): `grid_luma: 9 -> 17` (or
another single step up), everything else identical to round 9 (599-scene
train tier, hybrid architecture, same 3 seeds, same schedule), comparison
target round 9 (the current selected-best) on the unchanged monitor tier.
Predeclared success would need to include the near-black/bright-fraction
gap narrowing on these 4 scenes specifically, not just union-regression
count, since that's the mechanism actually being tested here - union
count alone could stay flat while this systematic bias still improved,
or vice versa. Awaiting approval before specifying exact criteria and
launching.

## Round 10 (2026-11-12/13): grid_luma 9→13 - clean negative, closes this branch too

Approved per instruction: isolated change `grid_luma: 9 -> 13`, round 9's
architecture/data/seeds/loss/schedule/monitor set otherwise completely
unchanged, `refinement_max_delta` untouched. Comparison target round 9
(current selected-best) on the unchanged 76-scene monitor tier.
Predeclared retention: keep round 10 only if it had fewer persistent
union-of-seed regressions than round 9's 4-scene union, OR clearly
improved crop-level L1 on all 4 of `DSCF0590`/`DSCF0873`/`DSCF0878`/
`DSCF0879` (`DSCF0878` counted this time, unlike round 9) - AND no new
monitor regressions beyond round 9's existing union either way. Script:
`training/run_round10_lumaxis.py`. Full data:
`training/diagnostics/round10-lumaxis-comparison.json`.

**`predeclared_success_met: false`.** Worse than round 9 on every axis
checked, not just inconclusive:

- **A brand new regression appeared**: `DSCF0553` - the one scene round 9
  resolved - regressed again under all 3 seeds. Round 10's union is 5
  scenes, not fewer than round 9's 4.
- **Crop-level L1 got worse on all 4 problem scenes**, none improved:
  `DSCF0590` 0.1693->0.1803, `DSCF0873` 0.1539->0.2323 (+51%), `DSCF0878`
  0.0689->0.0993 (+44%), `DSCF0879` 0.1566->0.1787. `DSCF0873` and
  `DSCF0878` took the largest hits.
- **The hypothesized mechanism moved the wrong way, not just stalled**:
  `pred_bright_fraction` (already too high vs target in round 9) got
  higher still on every one of the 4 scenes under round 10 - e.g.
  `DSCF0873` 0.0163->0.0693 bright fraction against a target of ~0.0007.
  Finer luma-axis resolution did not sharpen shadow/highlight separation
  as hypothesized; it made the over-brightening bias worse.
- **All 3 seeds early-stopped well before the 20-epoch budget, at
  unusually early best-epochs**: seed 42 stopped at epoch 8 (best epoch
  3), seed 43 at epoch 8 (best epoch 3), seed 44 at epoch 15 (best epoch
  10) - converging fast and shallow then degrading, the same signature
  round 5 saw when it increased `grid_spatial` under a fixed training
  budget. Total wall time (~4.5h combined) was well under the ~8-10h
  estimate given beforehand, specifically because of this early
  degradation, not because training was cheaper per epoch.

**Read**: this is the second time (after round 5's `grid_spatial`
increase) that making a grid axis finer, with everything else held
fixed, produces this exact fast-shallow-then-worse training signature and
makes the targeted defect regions worse rather than better. Taken
together, both of the bilateral grid's resolution axes have now been
isolated-tested and both failed the same way - the floor isn't grid
resolution on either axis under the current training budget/encoder; if
grid resolution is part of the real mechanism at all, it needs
materially more training capacity/time to fit than either axis-finening
attempt has been given, not just a finer grid. **Round 9 remains the
selected-best model** - round 10's checkpoints stay on disk, unselected.

Full data: `training/diagnostics/round10-lumaxis-comparison.json`.
`finalHeldOut` untouched - still not requested. No further round is
proposed here; next step is open for instruction.

## Signed-error reconciliation (2026-11-13): root cause found - Fuji DR mode, not the model

Per instruction (before designing round 11): reconcile the residual
direction with signed per-pixel errors and visual comparisons on the 4
remaining failures, confirm exactly what is too bright or too dark, then
propose one loss-function change. Diagnostic only: inference on
train/monitor with existing checkpoints, no training, no selection
change, `finalHeldOut` not accessed. Round 9 remains selected-best.

**The loss-function proposal is withdrawn** - the reconciliation found
the actual cause, and it is not something any loss can fix. Details in
order of discovery.

### 1. Signed errors (monitor): a uniform global over-lift, not a tail defect

`training/diagnose_signed_errors.py` - signed luma error
`pred - target`, binned by the *target's* luma band, round 9 (3 seeds)
and round 6, all 76 monitor scenes, the 72 non-failing scenes as
control. Output: `training/diagnostics/signed-error-diagnosis.json`;
visuals (input | round 9 | target | signed error map, red = model too
bright) in `training/diagnostics/signed_errors/`.

- On all 4 failures, round 9 is **too bright in every band from deep
  shadow through light (target luma 0 to 0.7)**: 92-99% of pixels too
  bright in the shadow/dark/mid/light bands, mid-band signed error +0.13
  to +0.24 linear (+0.09 to +0.17 in sRGB-encoded terms - visibly
  washed out). Highlights (target > 0.7) are roughly right (+0.02 to
  +0.05; `DSCF0873` -0.07). The error maps are red across the entire
  frame; the only blue is edge misregistration.
- Against what Fuji actually does to these scenes (input vs target):
  Fuji darkens their shadows slightly and lifts their mids by only
  0.08-0.13; the model lifts the mids by 2-4x that and lifts the
  shadows instead of darkening them.
- Sign is identical on every seed (within +-0.01) and on round 6; round
  9 shrank every band's error 10-30% without changing its sign.
- The control group is slightly too *dark* (mid-band mean -0.02;
  58/72 scenes). The 4 failures are a discrete cluster, not a tail:
  the next-most-over-lifted monitor scene is `DSCF0553` at +0.095 (the
  "resolved" round-9 scene - still the same signature, just under the
  L1 bar), and after it +0.017.
- Per channel the error is neutral (R~G~B) - brightness, not colour.

This **corrects the 2026-11-12 "remaining-regression" diagnosis above**:
"highlights too bright" was wrong (midtones pushed above 0.7 inflated
`bright_fraction`; the true highlights are fine), and "shadows not dark
enough" was true but a minor part. The dominant error is a scene-wide
exposure over-lift of roughly half to one stop. `DSCF0878` has exactly
the same signature as the other three; it was never structurally
different in the way that matters.

### 2. Train tier: the same under-fit inside round 9's own training set

`training/diagnose_signed_errors_train.py` (round 9, all 599 train
scenes; `training/diagnostics/signed-error-train-survey.json`). Train
overall is fit well (frame-weighted signed error mean +0.004), but
capture batch 2 is not: mean mid-band over-lift +0.11, 20 of 38 scenes
over +0.10, and the four worst-fit scenes in the entire train tier are
all batch 2 - `DSCF1419` +0.41 frame-wide (mean luma: input 0.24,
target 0.16, round 9 output 0.59 - on a scene it trained on). Visuals:
`training/diagnostics/signed_errors/*_train_round9_full.png`. Round 9 is
not overfitting (train L1 0.030-0.033 vs monitor 0.038-0.041, two seeds
still improving at epoch 20): it is being asked to map near-identical
thumbnails to different exposures and settling on a compromise.

### 3. Data survey: the target/input exposure ratio is set by the shot's DR mode

`training/diagnose_target_exposure.py` (no model; all 675 train+monitor
pairs; `training/diagnostics/target-exposure-survey.json`) measured
`log2(mean target luma / mean input luma)` per scene. Corpus median
+0.84 stops (Fuji's rendering is ~1.8x brighter than the raw-normalized
`.fjlrg` input), stable across Egypt (+0.90) and Spain (+0.88) - but
capture batch 2 sits at +0.25, and at *matched input brightness* batch 2
is 1.5 stops below batch 1 (input luma 0.18-0.26: batch 1 median +1.22,
n=21; batch 2 -0.35, n=12). No fixed tone curve produces that. Within
batch 1 the ratio tracks ISO: ISO 250 scenes +0.28, ISO >= 500 scenes
~+0.9.

`scripts/dump-raf-exif.mjs` then read the Fuji MakerNote DR tags of all
675 RAFs (`calibration-input/corpus-exif.json`, gitignored with the rest
of `calibration-input/`; the joined per-scene table is committed as
`training/diagnostics/target-exposure-survey-with-exif.json`). Ratio by
capture DR mode:

| capture DR (MakerNote) | n (train / monitor) | median log2 ratio | monitor failures |
|---|---|---|---|
| DR400 (`DevelopmentDynamicRange=400`) | 600 (529 / 71) | **+0.90** | **0** |
| DR200 (`DevelopmentDynamicRange=200`) | 37 (32 / **5**) | **+0.13** | **5 of 5** |
| batch 2, DR-Auto resolved to 200 | 26 (26 / 0) | +0.35 | - |
| batch 2, DR-Auto resolved to 100 | 12 (12 / 0) | **-0.35** | - |

The 5 monitor scenes shot at DR200 are exactly `DSCF0553, DSCF0590,
DSCF0873, DSCF0878, DSCF0879` - the complete persistent-regression
family, outlier included. The other 71 monitor scenes are all DR400 and
none has ever regressed. The 12 DR100 batch-2 scenes are the worst-fit
train scenes. (X100VI: DR200 needs ISO >= 250, DR400 ISO >= 500 - hence
the ISO dependence and the 358 train scenes at exactly ISO 500.)

**Mechanism (corrected 2026-11-13 after Codex's independent check).**
Codex verified that all 599 train and 76 monitor TIFFs record
`DevelopmentDynamicRange=100` in their own EXIF - the targets ARE
output-DR100 renders, as the protocol specifies. The first draft of this
section claimed the TIFFs "embed the capture DR"; that was wrong as
stated and is withdrawn. What the data shows is this: Fuji's DR200/DR400
*capture* under-exposes the sensor by 1/2 stops, and Fuji's converter
restores that exposure (x2 / x4, with the headroom compressed) *before*
rendering at whatever output DR is selected - so even an output-DR100
TIFF of a DR400 capture is at the shot's nominal exposure. The browser
`.fjlrg` input is LibRaw black/white-level normalized sensor data with
no such compensation. The target therefore contains a per-shot exposure
offset, set by the *capture* DR, that is absent from the input and not
recoverable from a 256 px thumbnail (ratio steps ~0.6 stop per DR
doubling in mean-luma terms, highlights converging - exactly the
"everything but the highlights is too bright" signature). A model
trained on 94% DR400 captures learns the DR400 offset and applies it to
DR200/DR100 captures - a ~0.6-1.2 stop over-lift of everything below
the highlights. Every prior observation fits: `DSCF0878` is not an
outlier; round 9 helped a little because batch 2 added 38 non-DR400
captures and hurt its own DR100 captures for the same reason; round
10's finer luma grid could not help because the information is not in
the pixels; rounds 3-8 could not either.

Capture DR and TIFF output DR are now kept as **separate fields** in the
corpus manifest (train + monitor only; `finalHeldOut` untouched):
`captureDynamicRange` / `captureDynamicRangeSource` from the RAF
MakerNote (`DevelopmentDynamicRange` for manual-DR captures,
`AutoDynamicRange`'s recorded effective value for DR-Auto captures;
missing or conflicting tags are flagged, never guessed - 0 flagged of
675), and `tiffOutputDynamicRange = 100` with a source string naming
Codex's verification rather than a measurement
(`training/annotate_capture_dr.py`). Annotation result: monitor 71 x
DR400 + 5 x DR200; train 529 x DR400 + 32 x DR200 (manual) + 26 x DR200
+ 12 x DR100 (batch 2, DR-Auto). The protocol doc is correct as written;
the earlier "protocol discrepancy" flag is withdrawn.

### 4. Round 11: condition the model on capture DR - approved and launched 2026-11-13

Not a loss change. The verified error is a missing input variable, and
reweighting the loss (gamma-space, log-space, band-weighted, exposure
term) cannot supply it - it would only move the compromise between DR
groups around. The one isolated change: **give the model the shot's
capture DR mode.** Simplest form that keeps everything else identical to
round 9: a constant fourth input plane on the 256 px encoder input,
value `log2(DR/100)/2` in {0, 0.5, 1} (encoder first conv `in_ch`
3 -> 4, +144 parameters at `base_ch=16`, 115,575 -> 115,719; grid,
refinement net, loss, schedule, seeds, data, monitor set unchanged;
TIFF targets output-DR100 throughout). Pre-scaling the input by `2^stops`
was considered and rejected: the luma guide clamps at 1, so a scaled
input would lose the highlight information the DR curve depends on; a
conditioning plane keeps the raw input intact. At inference the app
already surfaces the tag (`decoder.metadata(true).fuji`,
`src/lib/raw/rawService.ts:785`), so this is deployable; the ONNX
export gate would be re-run on the selected checkpoint as for round 6.

Mechanical prerequisites (not training): add `developmentDynamicRange`
per scene to the corpus manifest for train+monitor from
`corpus-exif.json` (batch 2: use the DR-Auto resolved value);
`dataset.py` returns it; `train.py`/`model.py` accept the extra plane.
`finalHeldOut`'s RAFs were not read - its DR composition is unknown. If
it is all DR400 the final gate cannot test the fix, and that is worth
knowing before relying on it; checking its MakerNote tags only (no
pixels) would need explicit authorization.

**Approved (Codex, relayed 2026-11-13)** as a controlled test of
capture-DR conditioning, output DR100 and everything else matched to
round 9, DR-Auto resolved from the recorded effective value,
missing/ambiguous metadata flagged not guessed, results reported per
capture-DR group with signed exposure errors and new regressions, round
9 preserved unless the predeclared criteria pass, `finalHeldOut` locked.

**Predeclared criteria (fixed before launch, encoded in
`training/run_round11_drconditioning.py`):** comparison target round 9,
same 3 seeds (42/43/44), unchanged 76-scene monitor tier. Success
requires ALL of: (1) the DR200 union-of-seed regression set strictly
smaller than round 9's (round 9: 4 of 5 - `DSCF0553` already passes
L1); (2) mean signed mid-band luma error over the 5 DR200 monitor scenes
below +0.05 (round 9: ~+0.16); (3) no new monitor regressions vs round
9 anywhere, DR400 or DR200; (4) crop-level L1 on every DR200 scene not
worse than round 9. "All 5 DR200 scenes non-regressed under all 3
seeds" is reported as the stronger outcome but not required. The
signed-band errors are reported for every monitor scene either way (the
mechanism check), and the 12 DR100 train captures' signed errors are
reported as a train-tier observation only. **Monitor contains no DR100
captures, so nothing in this round can establish DR100 accuracy** -
that claim needs separate evaluation examples; a future capture batch
should be planned by capture DR mode, not by highlight shape.

Code (committed before launch, all default-off so rounds 2-10 are
reproducible unchanged): `model.py` `condition_channels` on
`BilateralGridPredictor`/`HybridPredictor` (0 = identical network and
state_dict; round 9 checkpoints verified to load), `dataset.py` returns
`capture_dr_stops` (NaN when flagged), `train.py`
`--capture-dr-conditioning` / `capture_dr_condition()` (raises on NaN).
Smoke-tested end to end before launch. Run names
`r11-drcond-seed{42,43,44}`; output
`training/diagnostics/round11-drconditioning-comparison.json`.

Scripts added this pass (all diagnostic, train+monitor only):
`training/diagnose_signed_errors.py`,
`training/diagnose_signed_errors_train.py`,
`training/diagnose_target_exposure.py`, `training/render_signed_errors.py`,
`scripts/dump-raf-exif.mjs`.

## Round 11 results (2026-11-13): capture-DR conditioning passes - new selected-best

All four predeclared criteria met; **round 11 replaces round 9 as the
selected-best model.** Round 9 is preserved (checkpoints, run logs,
docs) as the comparison baseline. `finalHeldOut` was not accessed.
Evidence: `training/runs/r11-drcond-seed{42,43,44}.json`,
`training/diagnostics/round11-drconditioning-comparison.json`.

**Per-seed training (vs round 9, same 76-scene monitor, baseline L1
0.1023):**

| seed | round 9 best monitor L1 (epoch) | round 11 best monitor L1 (epoch) | round 11 regressed / 76 | train L1 r9 -> r11 |
|---|---|---|---|---|
| 42 | 0.0407 (20) | **0.0301** (20) | 0 | 0.0334 -> 0.0229 |
| 43 | 0.0393 (20) | **0.0302** (20) | 0 | 0.0299 -> 0.0228 |
| 44 | 0.0378 (13, early-stop) | 0.0374 (3, early-stop at 8) | 1 (`DSCF0873`) | 0.0316 -> 0.0334 |

Seeds 42/43: ~25% lower monitor L1 than any prior round, zero
regressions, still improving at epoch 20. Seed 44 is the honest caveat:
its monitor L1 oscillated (0.037 -> 0.043 -> 0.039 -> 0.038 -> 0.041 ->
0.038 over epochs 3-8) and early stopping fired at epoch 8 with best
epoch 3, so its reported checkpoint is a barely-trained one - it is
not evidence that conditioning fails on that seed, only that patience 5
cut it off during the noisy early phase. It matches round 9's seed 44 L1
rather than beating it. The group numbers below are means over all
three seeds, so they include this weaker checkpoint.

**Monitor DR200 captures (n=5) - the group the fix targets:**
union-of-seed regression set 4 -> 1 (resolved `DSCF0590`, `DSCF0878`,
`DSCF0879`; `DSCF0873` persists under seed 44 only; no new). Mean
mid-band signed luma error **+0.160 -> +0.003**; mean model L1 0.0756 ->
0.0329.

| scene | model L1 r9 -> r11 (baseline) | regressed seeds r9 -> r11 | mid signed r9 -> r11 | frame signed r9 -> r11 | crop L1 r9 -> r11 |
|---|---|---|---|---|---|
| DSCF0553 | 0.0820 -> 0.0423 (0.1118) | [] -> [] | +0.095 -> +0.013 | +0.078 -> +0.009 | 0.1078 -> 0.0566 |
| DSCF0590 | 0.0561 -> 0.0292 (0.0397) | [42,43,44] -> [] | +0.240 -> +0.038 | +0.049 -> +0.004 | 0.1693 -> 0.1141 |
| DSCF0873 | 0.0747 -> 0.0314 (0.0423) | [42,43,44] -> [44] | +0.129 -> -0.001 | +0.064 -> +0.006 | 0.1539 -> 0.0704 |
| DSCF0878 | 0.0945 -> 0.0386 (0.0575) | [42,43,44] -> [] | +0.162 -> -0.015 | +0.088 -> -0.009 | 0.0689 -> 0.0563 |
| DSCF0879 | 0.0705 -> 0.0229 (0.0397) | [42,43,44] -> [] | +0.176 -> -0.021 | +0.068 -> -0.006 | 0.1566 -> 0.0563 |

The signed exposure error - the verified mechanism - went to ~zero on
every DR200 scene, including `DSCF0878`, the one previously labelled a
"structural outlier": it was the same exposure error, not structure.
Crop L1 improved on all five (criterion 4 required "not worse").

**Monitor DR400 captures (n=71):** union regressed 0 -> 0, no new
regressions, mean mid-band signed -0.022 -> -0.023 (unchanged, slightly
dark as before), mean model L1 0.0367 -> 0.0325. The conditioning did
not cost the majority group anything and the L1 gain there is real.

**Train-tier DR100 captures (n=12, observation only, not evaluation -
these were trained on):** mean mid-band signed +0.176 -> +0.007; mean
model L1 0.153 -> 0.042. The catastrophic round 9 fits (`DSCF1417`
0.321, `DSCF1419` 0.390, `DSCF1420` 0.252, `DSCF1392` 0.234) all drop
to ~0.04-0.05. Union "regressed vs per-scene baseline" 10 -> 9 looks
flat but is misleading: 7 of the 9 are seed-44-only (the weak
checkpoint) and the remaining one, `DSCF1395`, has a very low baseline
(0.0251) that round 11 at 0.037 still doesn't beat; two "new" entries
(`DSCF1391`, `DSCF1397`) are seed 44 only with L1 0.039/0.034 vs
baselines 0.068/0.060, i.e. marginal. This says the model *can* use the
DR100 value, nothing more.

**Criteria:** (1) fewer DR200 regressions: 4 -> 1, pass. (2) DR200 mean
mid-band signed < +0.05: +0.003, pass. (3) no new monitor regressions vs
round 9: none, pass. (4) DR200 crop L1 not worse: better on all 5,
pass. Stronger outcome (all 5 DR200 clear under all 3 seeds): **not
met** (`DSCF0873` under seed 44).

**What this does and does not establish.** It establishes that the
per-shot exposure offset the model could not predict from pixels is
explained by capture DR and that supplying it fixes the DR200 monitor
failures without hurting DR400. It does **not** establish DR100
accuracy (no DR100 captures in monitor; the 12 train DR100 scenes are
not held out) and it does not say anything about `finalHeldOut`, whose
DR composition is unknown. Both still need separate, authorized
evaluation examples before the DR100 path is relied on in the app.

**Next steps (not started):** (a) ONNX export gate re-run on a round 11
checkpoint (the conditioned model has a fourth input plane; the app
already reads `DevelopmentDynamicRange`/`AutoDynamicRange` via
`rawService.ts`, but the inference wrapper must build and pass the
plane); (b) decide seed-44 handling - either accept seeds 42/43 as the
candidates for export (both best at epoch 20, 0 regressions) or re-run
seed 44 with the early-stop patience it evidently needed, as a
predeclared follow-up, not a silent re-roll; (c) plan a DR100 capture
set for monitor-tier evaluation.

## Round 11 browser export gate (2026-11-14, Codex)

Seed 42 and 43 best checkpoints passed reproducible native ONNX and
Chromium WASM/WebGPU parity gates. Seed 44's original early-stopped
result is preserved; no seed rerun or new training was launched.

The complete inference graph accepts full-resolution linear RGB plus
one capture-DR scalar and constructs the fourth encoder plane internally.
Both portrait/landscape working resolutions and a smaller dynamic shape
were exercised, across DR100/200/400 conditioning. Reference comparisons
use the original trained PyTorch models, unchanged. Model files, hashes,
reports and instructions are versioned under `training/exports/round11/`;
scripts are `training/export_round11_onnx.py` and
`scripts/verify-round11-onnx-web.mjs`.

Testing trained refinement weights at working resolution exposed two
export/runtime problems that an untrained zero-residual smoke test could
hide. Default ONNX InstanceNormalization lost full-image WASM precision;
export-only centered GroupNorm with separate spatial/channel reductions
fixed it. WebGPU's default layout produced substantial refinement drift;
explicit `preferredLayout: "NCHW"` fixed it. Neither fix changes trained
weights or the learning objective. Preprocessing, DR conditioning, and
bilateral-grid slicing matched before these fixes.

Final maximum absolute error vs PyTorch: native 2.21e-6, WASM 2.51e-6,
WebGPU 9.54e-7. Maximum browser mean absolute error: 9.80e-8. Both seeds
passed all cases under tolerances declared before testing. WebGPU was
tested on Apple's Metal adapter in Chromium, not just software WebGPU.
This is conversion accuracy, not a new Fuji-parity or performance score.

Independent DR100 evaluation is still needed. Plan a separate set of
12-20 DR100 RAFs from at least three new capture sessions, spanning
daylight, low light, and bright-window/light-source scenes, using the
same output-DR100/Provia/as-shot/16-bit TIFF and browser-linear protocol.
Keep these examples outside training and predeclare the evaluation before
using their results. Capture DR should be explicitly 100 on the camera,
with DR Priority off, and verified afterward in metadata.

No app integration/deployment or final-held-out access occurred. The
existing final-held-out tier remains locked. Passing this export gate
does not establish full-resolution 40MP behavior, DR100 generalization,
or matching of every film simulation/recipe.

## DR100 evaluation batch - scope and predeclared evaluation (2026-11-14, not captured yet)

Why: round 11's selection evidence is DR400 (71) + DR200 (5) monitor
scenes. The only DR100 pixels the model has ever seen are the 12 batch-2
train scenes (all DR-Auto, resolved from `AutoDynamicRange=100`, ISO
125-400). Nothing held out tests DR100, and the capture-DR plane value 0
has never been evaluated. This batch exists to make exactly one claim
testable: *round 11 renders unseen DR100 captures without the exposure
error that round 9 showed on DR100, and without regressing below the
identity baseline.* It is an evaluation set, not training data. Its
first scoring is an independent acceptance test. If its results ever
inform a model change, every later scoring on it is a **regression
check only** - independent acceptance of that later model needs fresh
evaluation data (Codex correction 2026-11-14 to an earlier "single-use"
wording).

**Confound worth designing out.** In the current corpus capture DR and
ISO are entangled: DR100 only at ISO 125-400, DR200 only at ISO >= 250,
DR400 only at ISO >= 500 (the X100VI's DR200/400 ISO floors). A model
could have partly learned "low ISO => no exposure restore" from noise
statistics instead of from the plane. DR100 is selectable at every ISO,
so the eval batch must include DR100 frames at ISO >= 1600 - if those
score like the low-ISO ones, the plane is doing the work.

**Capture spec (user shoots; >= 3 new sessions, 12-20 frames, target
18):**

- Sessions: at least three, each satisfying the corpus's `>3h gap = new
  session` rule and not overlapping any existing session (last existing
  capture: 2026-11-12 00:58, batch 2). Different days or different
  locations preferred; ~6 frames per session.
- Scene classes, each present in at least two sessions: (a) daylight
  exterior; (b) low light / night interior or street; (c) bright window
  or in-frame light source against a darker interior (the highlight
  class the DR modes exist for - on DR100 these will clip, which is the
  point).
- ISO spread inside DR100: at least 4 frames at base ISO 125-200, at
  least 4 at ISO >= 1600, the rest wherever the scene puts them. **Each
  ISO group must span at least two sessions and at least two of the
  lighting classes (a)-(c)** - high ISO must not become a synonym for
  night scenes, or the confound check just swaps one confound for
  another (stop down / use the ND filter / shorter shutter to put
  ISO >= 1600 on a daylight or window scene; use a tripod or wide
  aperture to put base ISO on a low-light one).
- White balance: mix of Auto WB and one manual preset/Kelvin, as in the
  rest of the corpus (as-shot is what the protocol exports either way).
- Camera menu, verified before each session: **Dynamic Range = DR100
  (not Auto)**; **D Range Priority = Off** (DRP overrides DR and tone and
  would silently invalidate the frame); film simulation Provia; single
  exposure (no multiple exposure, no bracketing, no HDR mode); exposure
  compensation may vary - record nothing by hand, the RAF carries it.
- No deliberate duplicates of existing corpus scenes/locations.

**Metadata verification (before any export; exclusion rules fixed here,
before scoring; a frame that fails is dropped, not corrected, and every
excluded frame is listed with its reason in the evaluation report):**
`node scripts/dump-raf-exif.mjs` over the batch, then
`training/annotate_capture_dr.py`'s `resolve()` must return 100 with
source `DevelopmentDynamicRange` for every frame, zero flagged.
Additionally per frame: `AutoDynamicRange` absent (it was present on
every DR-Auto frame in batch 2 - its presence here means the camera was
left on Auto), `DRangePriority`/`DRangePriorityAuto`/`DRangePriorityFixed`
**absent or 0** (either is acceptable - the tag may simply not be
written when DRP is off), `ExposureCount` 1, model X100VI. If exclusions
leave fewer than 12 frames, or fewer than 4 in either ISO group, the
batch is short and the user shoots the gap before scoring - the
criteria are not relaxed to fit. Note this exercises the
resolver's manual-DR path; the 12 train DR100 scenes exercised only the
Auto path. The resolver treats both as the same plane value 0 - that
equivalence is one of the things this batch tests.

**Export:** protocol 3.0 unchanged (`docs/phase3-export-protocol.md`:
X RAW Studio Provia / as-shot WB / output DR100 / all tone and effects
zero or off / 16-bit TIFF, record the format used; browser
`?rawCalibrationWb=camera`, 1536 px long edge, via
`scripts/export-phase3-linear.mjs` with `FUJIAPP_EXPORT_MANIFEST`). Shoot
folders continue from `Shoot 744`; filenames
`xraw-phase4-dr100eval.tiff` and `browser-phase4-dr100eval-linear.fjlrg`
so the batch is identifiable without opening files. Batch manifest
`calibration-input/phase4-dr100eval-manifest.json` (`{shoot_folder,
filename}` schema). After export, Codex-style verification that every
TIFF records `DevelopmentDynamicRange=100` (output DR), as for the main
corpus.

**Tier handling:** a new manifest tier `dr100Eval`, scenes carrying the
same fields as train/monitor plus `captureDynamicRange=100`.
`dataset.list_scenes` keeps refusing everything but `train`/`monitor`, so
training code cannot load it by construction; the evaluation script gets
its own loader that accepts only `dr100Eval`. `precompute_targets.py` is
not run over it through the training path. `finalHeldOut` is untouched by
all of this.

**Predeclared evaluation (fixed now, before a frame exists):**

- Models: round 11 seed 42 best checkpoint (primary export candidate),
  seed 43 (backup); round 9 seeds 42/43/44 as the comparison; seed 44 of
  round 11 reported but not a candidate (early-stop caveat above).
- Per scene (same `diagnose_signed_errors.py` machinery): model L1,
  identity-baseline L1, regressed flag (model L1 > identity L1),
  signed mid/shadow/highlight/frame luma errors, mid-band pixel
  fraction, data-only exposure ratio
  `log2(mean target luma / mean input luma)`, and the seed-matched
  per-scene L1 difference round 11 minus round 9. Reported in three
  groups: all, ISO <= 400, ISO >= 1600, and every per-scene value is
  listed - no group number stands without its scenes.
- **Midtone coverage:** a scene's mid-band (target luma 0.2-0.5)
  signed error is *measurable* only if that band holds **>= 5% of the
  frame's pixels**. Scenes below that are reported as "mid-band not
  measurable", excluded from the mid-band means and the mid-band count,
  and **never counted as zero**; they still count fully for L1,
  regression, and frame-signed criteria. At least 10 measurable scenes
  overall and at least 3 per ISO group are required for the mid-band
  criteria to be evaluable at all - otherwise the batch is short, not
  passed.
- Pass requires ALL of, for **both** seed 42 and seed 43 independently,
  each compared with the **same-seed** round 9 checkpoint (r11-s42 vs
  r9-s42, r11-s43 vs r9-s43):
  (1) mean **signed** mid-band luma error over the measurable scenes
  within **+-0.05**, AND mean **absolute** per-scene mid-band signed
  error **<= 0.05** (so over-bright and over-dark scenes cannot cancel);
  (2) **zero** regressed scenes vs identity (round 11 seeds 42/43 are
  0/76 on monitor; DR100 is held to the same bar);
  (3) round 11's mean L1 over the set **below** the matching round 9
  seed's mean L1, with every per-scene difference reported (the
  expectation is that conditioning helps on most scenes; a few scenes
  where round 9 happens to be closer are reported, not hidden, and do
  not fail the criterion on their own);
  (4) the ISO >= 1600 group meets both parts of (1) on its own (the
  confound check), and no single scene anywhere has |frame signed
  error| > 0.10;
  (5) **browser parity gated, not informational:** the round 11 ONNX
  seed 42 export (and seed 43 if it is the candidate) run on these
  exact inputs in WASM and WebGPU must pass the tolerances already
  declared for the 2026-11-14 export gate (`training/exports/round11/`
  README and reports), via `scripts/verify-round11-onnx-web.mjs`. A
  parity failure here blocks app integration regardless of (1)-(4).
- Reported, not gated: the exposure-ratio distribution (hypothesis: DR100
  median near 0 stops; batch 2's DR100 group showed -0.35, which is
  unexplained and may be a scene-content effect - this batch is the
  first chance to see it on independent material); round 11 seed 44 and
  round 9 seed 44 for completeness.
- Outcomes: pass => round 11 seed 42 is cleared for DR100 on this
  evidence and app integration can be scoped. Fail => no retraining on
  or tuning against these scenes; diagnose on the 12 train DR100 scenes,
  change one thing, re-run train/monitor, and only then look here again -
  at which point this set is a regression check for the changed model,
  and independent acceptance of that model needs new DR100 frames.

**What a pass would not establish:** anything about `finalHeldOut`
(still locked, DR composition unknown); 40 MP full-resolution behaviour
(the export gate ran at working resolution); film simulations other than
Provia; output DR other than 100. (An earlier draft of this paragraph
claimed monitor's DR200 group was all ISO >= 500; checked against
`corpus-exif.json`, it is not - DSCF0553 and DSCF0873 are ISO 250, 0590
is 500, 0878/0879 are 4000/2000 - so DR200's ISO range is already
covered and is not a gap.)

**Codex review (2026-11-14), applied above before any capture:** ISO
groups spread across >= 2 sessions and >= 2 lighting classes; mean
absolute per-scene mid-band bias <= 0.05 added alongside the signed
mean, with a minimum midtone-coverage rule so unmeasurable scenes are
never counted as zero; round 9 comparison is seed-matched, requires
overall improvement, reports every per-scene difference, and no longer
claims round 9 must lose on every photo; browser parity is a gate, not
a report; the "single-use" rule is replaced by acceptance-then-regression
with fresh data required for independent acceptance; metadata exclusions
fixed before scoring with every excluded frame and reason reported; DRP
rule stays "absent or 0". Planned frame count unchanged at 12-20.

Status: scoped and approved for capture. Nothing captured, no manifest,
no tier, no script. Next action is the user's: shoot the three sessions.

### DR100 eval batch, capture attempt 1 (2026-11-16): all 18 frames excluded by the fixed metadata rules - reshoot needed

18 RAFs (`DSCF1424`-`DSCF1441`) dropped in
`calibration-input/phase4-dr100eval-raw/`; MakerNote dump only
(`phase4-dr100eval-exif.json`), no pixel decode, no export. Applying the
exclusion rules exactly as fixed on 2026-11-14:

- **`AutoDynamicRange` present on 18/18, `DevelopmentDynamicRange`
  absent on 18/18** - the camera's Dynamic Range menu was on **AUTO**,
  not DR100. Every frame is excluded under the rule written for exactly
  this case.
- Under Auto the camera chose **DR100 for the 9 base-ISO frames (ISO
  125/160) and DR200 for all 9 ISO-1600 frames** - so the entire
  ISO >= 1600 group is not DR100 at all, and the confound check cannot be
  run on this material under any reading of the rules.
- **Sessions: 1, not >= 3.** All 18 were taken 2026-11-16 19:12-23:31;
  the largest gap is 2h01 (21:26 -> 23:27), under the corpus's `>3h`
  rule.
- Everything else was right: X100VI, `ExposureCount` 1, DRP tags
  absent, DR100/ISO-1600 pairs of the same scene across visibly varied
  brightness (BV -0.6 to 9.2), two WB modes, exposure comp 0/+0.33.

Not relaxed: the 9 Auto-resolved DR100 frames would exercise only the
Auto path the 12 train DR100 scenes already cover, the high-ISO half is
DR200, and the session count fails regardless - a partial acceptance
here would be a different, weaker test than the one approved. Decision:
**reshoot** with the camera's Dynamic Range set to **DR100 (fixed), not
AUTO**, across >= 3 sessions separated by > 3 h, same ISO/lighting
spread. These 18 frames are kept on disk, listed here as excluded with
reasons, and are not used for anything; the folder is cleared before the
reshoot lands so the two attempts cannot mix. `finalHeldOut` untouched.

### DR100 eval batch, capture attempt 2 (2026-11-17): metadata clean, session spec not met - held, not scored

20 RAFs (`DSCF1443`-`DSCF1458`, `DSCF1461`-`DSCF1464`; `DSCF1442`,
`DSCF1459`, `DSCF1460` not supplied - asked about) in
`calibration-input/phase4-dr100eval-raw/`. Tag-only gate
(`training/check_dr100eval_metadata.py`): **20/20 accepted** -
`DevelopmentDynamicRange=100`, `AutoDynamicRange` absent, DRP tags
absent, X100VI, single exposure. The camera-menu fault from attempt 1
is fixed. ISO split 10 x ISO 125 / 10 x ISO 1600 as base/high pairs;
WB 16 Auto (white priority) + 4 Kelvin. (The gate script first
mislabelled Fuji WB code 1, "Auto white priority", as manual; corrected
before this report - it affects labelling only, not any exclusion rule.)

**Not yet a valid batch: 1 session, not >= 3.** All 20 frames fall
2026-11-17 15:57-18:22 (longest gap 49 min), so neither ISO group spans
two sessions either. Nothing exported or scored. The test pair Codex
asked for was not gated before the full shoot (the user shot the set in
one go); the frames happened to pass, so no harm, but the step was
skipped, not performed.

Open question for Codex before more shooting: two more sessions at ~6
frames each would bring the batch to ~32, over the planned 12-20. The
choice - accept ~32, or a subset rule fixed now from metadata only (no
pixels seen) - is a protocol decision, not mine to make after the fact.

**User decision (2026-11-17): use attempt 2's 20 frames as the DR100
eval batch, with the session rule waived.** The user's reasons: every
pair was shot at a different place, under different lighting and
shadows, and the three missing frame numbers (`DSCF1442`, `1459`,
`1460`) are frames deleted in camera before the transfer, not frames
dropped after anyone looked at results. The metadata is consistent with
that: brightness value (BV) spans 0.6-8.7 across the pairs: 1 pair at
~0.7 (dim interior), 4 pairs at ~1.7-4.0, 3 pairs at 7.4-8.7 (bright
daylight). Each base/high-ISO pair is its own scene/location.

Recorded as a **protocol deviation, not a pass of the original spec**.
The "≥3 sessions >3h apart" rule and "each ISO group / lighting class
in ≥2 sessions" cannot be met by one 2h25 afternoon. What the deviation
costs: same day, same weather, same sun path, one photographer's
outing - session-level independence (the reason the rule exists) is not
tested. Every result from this batch carries that label. Everything else
is unchanged and still gated: metadata exclusions (20/20 pass), ISO
groups (10/10), midtone coverage, all five pass criteria, browser
parity, and the acceptance-then-regression rule. Codex is informed
before scoring. If Codex objects, the set is still unscored, so it can
be downgraded to a pre-check and fresh multi-session data shot for
acceptance.

Prepared (no pixels decoded): `Shoot 744`-`Shoot 763` (RAF symlinks),
`calibration-input/phase4-dr100eval-manifest.json` (tier `dr100Eval`,
`{shoot_folder, filename}` schema), flat X RAW Studio batch folder
`calibration-input/phase4-dr100eval-batch/` (symlinks). Next: the
protocol 3.0 exports on both sides.

### DR100 eval attempt 2: browser export complete (2026-11-17, Codex)

All 20 approved attempt-2 RAFs in `Shoot 744`-`Shoot 763` have now been
exported through the existing browser-only `decodePhase3LinearRaf`
pipeline, using `phase4-dr100eval-manifest.json`,
`FUJIAPP_EXPORT_FILENAME=browser-phase4-dr100eval-linear.fjlrg`, and
`rawCalibrationWb=camera`. The approved single-outing/session waiver
above is retained; this work does not satisfy the original multi-session
capture requirement.

Independent metadata check before export: 20/20 frames satisfy the
unchanged per-frame rules, with 10 ISO-125 and 10 ISO-1600 frames. Export
completed with zero failures. Integrity checks passed for all 20 outputs:
FJLRGB16 v1, 32-byte header, linear/sRGB-primary flags, 3 channels at 16
bits, exact payload sizes, and a 1536px long edge (8 at 1023x1536, 12 at
1026x1536). Per-file SHA-256 checksums and source/manifest hashes are
recorded locally in
`calibration-input/phase4-dr100eval-browser-export-report.json`.

No model inference, target comparison, model selection, training, or
final-held-out access occurred. The batch's first model evaluation is
still pending. At verification, none of the 20
`xraw-phase4-dr100eval.tiff` targets were present, so completing and
checking the Fuji-side exports remains the next data-preparation step.
Run the frozen evaluation/parity criteria only after those targets and
the dedicated `dr100Eval` evaluation script are ready.

**Browser side done; scoring code committed before targets exist
(2026-11-17).** Codex exported and audited all 20
`browser-phase4-dr100eval-linear.fjlrg` (format, payload size,
dimensions, checksum; `calibration-input/phase4-dr100eval-browser-export-report.json`).
The 20 X RAW Studio TIFFs did not exist yet when the scoring script
`training/eval_dr100.py` was written and committed, so the scoring code
predates every target. Stages: `check-targets` (TIFF present, and its
own Fuji MakerNote records output `DevelopmentDynamicRange=100` - reader
checked against two existing corpus TIFFs, both 100), `score` (criteria
1-4 exactly as written, round 11 seeds 42/43 candidates with seed 44
report-only, seed-matched round 9, midtone coverage >= 5%, unmeasurable
scenes never counted as zero), `parity-vectors` (native ONNX parity on
these exact 20 inputs against the committed `exports/round11` models,
SHA-checked against the export gate) followed by the existing
`scripts/verify-round11-onnx-web.mjs --vector-dir
training/checkpoints/round11-onnx-dr100eval` (WASM + WebGPU, unchanged
tolerances), then `verdict` (criteria 1-5). Every output carries the
single-session deviation label.

### DR100 evaluation results (2026-11-17): round 11 seeds 42 and 43 pass all five predeclared criteria

**Label carried by every number below:** single-session batch
(2026-11-17 15:57-18:22, user-approved deviation). Session-level
independence is not tested. 20 frames, 0 excluded, 10 x ISO 125 /
10 x ISO 1600. All 20 TIFFs record output `DevelopmentDynamicRange=100`
and FilmMode 0 (Provia) in their own MakerNote. All 20 scenes are
mid-band measurable (>= 5% coverage), so no measurement was dropped.
`finalHeldOut` not accessed. Evidence:
`training/diagnostics/dr100eval-results.json`,
`training/exports/round11/dr100eval-{native,browser}-report.json`.

| criterion | seed 42 | seed 43 |
|---|---|---|
| (1) mean signed mid, all (limit +-0.05) | +0.020 | +0.003 |
| (1) mean abs per-scene mid (limit 0.05) | 0.021 | 0.013 |
| (2) regressions vs identity | 0 / 20 | 0 / 20 |
| (3) mean L1 vs same-seed round 9 | 0.0338 vs 0.1783 | 0.0283 vs 0.1504 |
| (4) ISO 1600 mid signed / mean abs | +0.019 / 0.021 | +0.003 / 0.013 |
| (4) max scene abs(frame signed) (limit 0.10) | 0.024 | 0.016 |
| (5) native ONNX parity, 20 inputs | pass | pass |
| (5) Chromium WASM / WebGPU (Apple Metal, NCHW), 20 inputs | pass / pass (max abs 5.4e-7 / 6.0e-7, tol 5e-4) | pass / pass |

Round 11 beat its same-seed round 9 on **every** one of the 20 scenes
for both candidates. That was not required (criterion 3 asks only for
the overall mean), but it is reported as asked. Baseline (identity) L1
is 0.045-0.098 per scene, and round 11 is 0.020-0.050. The ISO 125 and
ISO 1600 groups score alike (seed 42 mid +0.021 vs +0.019, seed 43
+0.003 vs +0.003), so the ISO/DR confound check found nothing: the
capture-DR plane, not ISO noise, carries the decision.

**Honest caveats.**
- Seed 42, the primary export candidate, has a small **systematic
  positive** mid-band bias. Its mean signed (+0.020) is nearly equal to
  its mean abs (0.021) because almost every scene is slightly bright.
  That is within the limit, not noise around zero. Seed 43 is closer
  (+0.003) and lower in L1 on this set. Codex picked seed 42 as primary
  before these results, and this evaluation does not by itself justify
  swapping. Swapping would be selecting on this set, which would turn
  it into a regression check for the swapped choice. Recorded for
  Codex's decision.
- **Seed 44 (report-only) fails badly:** 14/20 regressed vs identity,
  mean mid +0.068, two scenes frame-signed > 0.10, L1 0.088. That
  confirms the round 11 early-stop caveat: seed 44's best checkpoint
  (epoch 3) is under-trained. It is not a candidate and was never one.
- **Hypothesis falsified:** the exposure ratio log2(target/input luma)
  for DR100 is **not** near 0 on this batch. Median -0.50 stops, range
  -0.74 to +0.01; here Fuji's DR100 render is darker than the linear
  input, the same direction as batch 2's -0.35. *(Corrected 2026-11-17
  per Codex: an earlier wording called this "how DR100 behaves". It is
  this batch's observed ratio, one afternoon of scenes, and not a
  universal DR100 correction. The ratio depends on scene content via the
  tone curve, and nothing should hard-code it.)* Round 11 handles it on
  these scenes, and round 9 did not, which is why round 9 is 3-5x worse
  here.

**Status:** seed 42 (and backup seed 43) is cleared for DR100 on this
evidence, so app integration can be scoped. This was the batch's first
scoring, so it counts as independent acceptance under the deviation
label. Any model change informed by it makes later scores here
regression checks only.

What this does not establish: multi-session/day robustness (the
deviation); `finalHeldOut` (locked); 40 MP full resolution; film
simulations other than Provia; output DR other than 100.

**Codex verification and follow-ups (2026-11-17).** Codex independently
verified the DR100 results: both candidates meet the numerical criteria,
both browser backends cover all 20 inputs, and TIFF headers and input
checksums match. **Seed 42 stays frozen as primary**; seed 43 remains
backup. The -0.50 stop ratio is this batch's observation only (corrected
above). `finalHeldOut` stays locked.

Eval-tooling gaps fixed before any further acceptance run (no threshold
or criterion changed):
- **Failure exits:** `score` exits 1 when the batch is short or a
  candidate fails criteria 1-4. `verdict` exits 1 unless the overall
  pass holds, and a missing or malformed report is a reported failure,
  not a crash. `check_dr100eval_metadata.py` exits 1 on a SHORT batch.
  (On the current batch it now exits 1 because of the single session.
  That is correct and consistent with the user-approved deviation
  recorded above.)
- **Cache fingerprinting:** the target cache is keyed by the source
  TIFF's SHA-256 plus the cached array's own SHA-256. A re-exported TIFF
  or a corrupted cache is rebuilt, never reused. Per-scene input and TIFF
  SHA-256 values are recorded in the results. `parity-vectors` clears
  all old vectors and reports first (skipping exFAT `._` sidecars), and
  records each input's source and vector SHA-256.
- **Report validation (`verdict` -> `validate_reports`):** results
  scene set matches the manifest; thresholds match the constants;
  inputs and TIFFs are unchanged since scoring. Native report: passed,
  every case, candidate seeds only, model SHA and served ONNX match the
  export gate, tolerances match the gate, and parity inputs are
  hash-identical to the scored inputs. Browser report: newer than the
  native report, passed, exactly wasm + webgpu with no errors,
  non-fallback WebGPU adapter, NCHW, tolerances identical to the export
  gate's browser report, all candidate seeds and all 20 cases passed.

Verified by a tooling regression run on the same 20 inputs (not a new
acceptance run). Every metric was identical to the first scoring (max
difference 0.0), and all stages exited 0. Tamper tests, each restored
afterward: loosened browser tolerance, a dropped browser case, a stale
browser report, a deleted browser report, a tightened mid limit
(in-process only), and a corrupted cache / TIFF-hash mismatch all made
`verdict`/`score` exit 1 or forced a cache rebuild. The SHORT metadata
gate exited 1.

**Codex follow-up fixes (2026-11-17), implementation only.** Thresholds,
seed selection (42 primary, 43 backup) and the session waiver are
unchanged.
1. **Numbers rechecked, not flags:** `validate_reports` now re-tests
   every native and browser case's recorded `max_abs`/`mean_abs`. Both
   must be real finite numbers within the frozen export-gate limits
   (native 1e-4/1e-5, browser 5e-4/5e-5), in addition to `passed`.
2. **Target check bound to TIFF bytes:** `check-targets` records each
   TIFF's SHA-256. `score` refuses to run unless the passed check covers
   exactly the accepted scenes and every hash still matches the file on
   disk. `verdict` re-verifies this and also requires each scored TIFF
   hash to equal the hash that passed the check.
3. **Results file handled like every other report:** `verdict` reads it
   inside the failure handler, so a missing or malformed results file
   exits 1 with a reason instead of crashing.

Verified: regenerated hash-bound target check (20/20, exit 0). The
untampered verdict passed (exit 0, seeds 42/43 true). Each tamper test
was rejected with exit 1 and a specific reason, then restored:
(A) native case `passed=true` with max_abs 0.5; (B) browser case
`passed=true` with mean_abs NaN; (C) browser case `passed=true` with
max_abs as a string; (D) a TIFF "replaced" after its check - both
`score` and `verdict` refuse; (E) results file missing; (F) results
file malformed. After restoring, all four reports were byte-identical to
before and the verdict passed again.

**Stage closed (2026-11-17, Codex).** Tooling fixes reviewed and
accepted. Final state of this stage: round 11 **seed 42 frozen as
primary**, seed 43 backup. The DR100 result stands with its
**single-session limitation** attached. No further tooling changes or
training for this stage. `finalHeldOut` remains locked and has never
been accessed. Open for a later stage, not started: app integration of
the seed 42 ONNX model, multi-session DR100 data for an unqualified
DR100 claim, and the `finalHeldOut` gate (needs explicit authorization).
## Local experimental Preview integration (2026-11-17, Codex)

Frozen Round 11 seed 42 is integrated into a separate **Experimental
Provia** mode for local/development builds. Exact input is the calibrated
LibRaw/as-shot/16-bit linear decode (including in native WebViews),
followed by capture-DR conditioning and the hash-pinned ONNX model.
The legacy recipe shader and AWB/DR/film-simulation corrections do not
run over model output. Supported output is Provia/DR100 at 1536px long
edge; JPEG input, other simulations/recipes, and full-resolution output
remain outside this mode. The original Preview workflow is preserved.

App-level checks pass on existing train/monitor fixtures: metadata for
capture DR100/200/400, exact RAW-input equality, original-PyTorch parity
in both browser backends, UI upload/render, PNG saving, invalid-upload
clearing, and switching back to standard mode. Packaged UI also passes
with WebGPU unavailable using WASM fallback. Maximum linear numerical
drift: 2.51e-6; PNG display rounding: at most one RGB byte level.

Ordinary release builds exclude experimental assets. The experimental
bundle is synced for Xcode testing; actual native WKWebView validation
is still pending. Instructions and limits are in
`docs/round11-local-preview.md`, reproducible checks in
`scripts/verify-round11-app.mjs`, evidence in
`training/exports/round11/app-*-report.json`.

No model change, training, public deployment, or finalHeldOut access.
Seed 42 remains primary, seed 43 backup, with the single-session DR100
acceptance limitation retained.

## Classic Cuban Neg validation stage (2026-11-17) - scoping, no fitting

Started per instruction. Round 11 **seed 42 stays frozen** (no
retraining, no re-selection). `finalHeldOut` is locked and was not read.
Nothing new is fitted until the RAW-input and recipe-processing plan is
agreed with Codex.

**Exact recipe (`src/data/recipes.json`, `fsr-classic-cuban-neg`, by
Osan Bilgi, X100VI / X-Trans V):** Classic Negative; DR400; WB Auto,
shift R+4 / B-5; Highlight -2; Shadow +1; Color +4; Sharpness 0; Color
Chrome Effect Strong; Color Chrome FX Blue Strong; Grain Strong / Large;
High ISO NR -4; Clarity -4; ISO Auto 500-6400.

**Existing references, audited from each file's own Fuji MakerNote:**
- `Shoot 10/xraw-cuban-neg.jpg` (DSCF0752): the **only genuine
  full-recipe X RAW Studio reference**. Every setting above matches,
  including WB Auto (0x0) and fine-tune [80,-100] = R+4/B-5. Limits: an
  8-bit JPEG, not 16-bit TIFF; one scene; DSCF0752 is **train tier**
  (round 11 trained on its Provia target), so it is diagnostic only.
- `Shoot 7/xraw-cuban-neg.jpg`: byte-identical duplicate of the above,
  filed in a folder with no RAF. Not a second reference.
- `Egypt:Spain 2026/Egypt/xraw-classic-cuban-neg.jpg`: **mislabelled**.
  It is byte-identical to `Shoot 10/xraw-provia-auto-neutral.jpg`
  (Provia, DR100, all zero). Must not be used as a Cuban Neg reference.
  Left in place, not renamed.
- `Shoot 10/calib-classic-cuban-neg-reference.jpg` and
  `...-auto-wb-probe.jpg`: camera raw-profile API outputs. Every recipe
  setting matches **except WB, recorded as Kelvin (0xff0), not Auto**.
  The API ignores requested WB modes (as `calibrationRecipes.ts`
  already notes), so the "auto-wb-probe" did not actually probe Auto
  WB. These are valid for isolated same-source deltas only, not as
  whole-recipe ground truth.
- 14 `app-cuban-negtest*.jpg` files (Shoot 7/10) are earlier app
  renders, not references.

**Validation set (metadata-only, `training/select_cuban_validation.py`,
`calibration-input/cuban-validation-manifest.json`):** 12 **monitor-tier**
scenes, so round 11 never trained on them, restricted to the recipe's
own capture conditions (captureDynamicRange 400, ISO 500-6400; pool 70).
Stratified across all 4 monitor sessions (4/3/3/2), with bright-highlight
x3, mixed-lighting x2, ISO 500-2000, and 8 manual + 4 auto as-shot WB.
DSCF0752 is added as **diagnostic-only** to tie back to the old JPEG.
No pixels were read to choose them.

**X RAW Studio exports needed (13 RAFs x 4 variants, 16-bit TIFF):**
- **E1 full recipe**, exactly as above. This is the ground truth.
- **E2 = E1 with Grain Off.** Grain is stochastic, so pixel scores use
  E2, and E1 is for visual / grain-statistics checks.
- **E3 = E2 with WB As Shot** (shift R+4/B-5 kept). Isolates Fuji's
  Auto WB from every other stage (E2 vs E3).
- **E4 Classic Negative only**: as-shot WB, DR100, every other control
  0/Off. Isolates the film-simulation base from the round 11
  Provia/DR100 output.
Batch folder `calibration-input/cuban-validation-batch/` (RAF symlinks,
one output subfolder per variant, so X RAW Studio's same-named outputs
can't collide).

**Open questions for Codex before any fitting:**
1. **RAW input:** does the recipe path start from round 11 seed 42's
   linear output (Fuji Provia, output DR100) and then convert to Classic
   Neg, or from the browser-linear `.fjlrg` directly? Round 11 targets
   only Provia/output-DR100, and the recipe's **output DR400** curve is
   outside what it learned.
2. **Output DR400:** the existing `applyDynamicRange` stage was
   calibrated from one camera-API scene (`calib-dr-400.jpg`). E4 vs E2/E3
   can test it, but only together with Classic Neg + tone.
3. **Auto WB:** the app's gray-world estimate versus Fuji's AWB. E2 vs
   E3 measures that gap directly on 13 scenes.
4. **Scoring:** proposed: predeclare per-stage and whole-recipe
   criteria on E2 (pixel), with E1 for grain statistics, before any
   stage is refitted. The 12 monitor scenes are validation, and
   anything fitted must come from train-tier data or a separate set.

### Codex plan review (2026-11-17) and export pilot setup

Codex's decisions, adopted as written:
1. **RAW input = `.fjlrg`**, not seed 42's output. Seed 42 already applies
   Fuji-like Provia tone and colour, so feeding it to a Classic Neg path
   would stack transformations and may lose information. Seed 42 stays
   frozen as the Provia baseline.
2. **E1-E4 kept, E5 added:** E4 with output DR changed to DR400 (as-shot
   WB, zero shift, all else 0/Off). E4 vs E5 isolates output DR. E3 vs E4
   changes several controls at once and cannot isolate DR, clarity or NR.
3. **The 12 monitor scenes stay outside all fitting** (film-sim, WB,
   control calibration). They are development-validation, not a fresh
   independent acceptance set. DSCF0752 stays diagnostic-only. A
   separate train-tier reference cohort is selected before any fitting.
4. **Two-scene export pilot first** (train tier, different lighting and
   as-shot WB, one Kelvin), E1-E5, verifying settings, orientation, colour
   space and genuine 16-bit data before the larger batch. Stage-distinct
   filenames; the Provia corpus files are never overwritten.
5. **AWB qualification:** E2 vs E3 measures Fuji's output change between
   Auto and As-Shot WB (R+4/B-5 kept). It does not by itself measure the
   app's AWB error or establish equivalent RAW gains. The app is compared
   against each corresponding reference, without re-applying WB to
   already-balanced input.
6. **Scoring predeclared before examining monitor results,** with limits
   set from training-pilot evidence and then frozen: E4 = simulation
   base, E5 = DR400, E3 = remaining recipe stack, E2 = complete grain-free
   recipe. Every scene is reported, with signed exposure/colour errors and
   visible artifacts. E1's grain is judged statistically and visually,
   not pixel-for-pixel. Existing control stages must be validated on this
   RAW path before they are kept; any refit uses train-tier references
   only.

**Pilot (no fitting):** `training/cuban_exports.py prepare-pilot` picked,
by fixed rule from the 456-scene train pool at recipe capture conditions
(capture DR400, ISO 500-6400; first by filename):
`DSCF0523` (Shoot 340, session 3, as-shot **Kelvin**, mixed-lighting) and
`DSCF0964` (Shoot 101, session 11, as-shot **Auto**, bright-highlight).
Batch: `calibration-input/cuban-pilot-batch/` with `E1`-`E5` subfolders.
`cuban_exports.py verify pilot` checks every TIFF against per-variant
expected MakerNote encodings (WB "as-shot" = the RAF's own code),
DateTimeOriginal vs the RAF, orientation and shape vs the scene's Provia
corpus TIFF, EXIF ColorSpace = sRGB plus an ICC profile byte-identical
to the corpus TIFF, and genuine 16-bit samples (not 8-bit stretched; the
multiples-of-257 test). Smoke-tested on `Shoot 10/xraw-phase3.tiff`
(E4 expectations): 16-bit, orientation and colour all pass, and the only
reported problem is `FilmMode=0 expected 2048`, as it should be.
`install pilot` copies into Shoot folders as
`xraw-cuban-<variant>.tiff` only after verification passes, and refuses
to overwrite anything. For `DSCF0964` (as-shot Auto), E3's WB tag
equals E2's by construction, so only pixels can show whether X RAW
Studio treats Auto and As-Shot differently on an Auto capture. The
`DSCF0523` Kelvin capture is the case where the tags must differ. The
validation batch gained the E5 folder too, but it is not exported yet.

### Codex verifier review fixes (2026-11-17)

Implementation-only; no thresholds for scoring set, no fitting, no export
batch, seed 42 frozen, finalHeldOut untouched.

1. **16-bit test no longer counts black/clipped samples.** 0 and 65535 are
   multiples of 257 in genuine 16-bit data, so an image with >5% clipped
   samples would have been falsely rejected. The test now uses only
   samples strictly between 0 and 65535; with fewer than 10,000 usable
   samples (of ~2.4M sampled) it reports `inconclusive`, which blocks
   install. Black/clipped fractions are recorded. Synthetic checks:
   genuine 16-bit with 40% clipped or 30% black -> pass; 8-bit stretched
   -> fail; fully clipped -> inconclusive. Both pilot scenes' Provia
   corpus TIFFs -> pass.
2. **Install no longer trusts a possibly stale report.** `verify` records
   each source TIFF's SHA-256 (re-hashed after checking, to catch a file
   changing mid-verify), the manifest SHA-256 and a hash of the variant
   definitions. `install` preflights everything before copying anything:
   report readable and passed; manifest and variant hashes unchanged;
   report covers exactly manifest scenes x E1-E5 with no duplicates or
   extras and no listed problems; every source TIFF re-hashed and
   unchanged; every destination folder present; no destination (or
   `.partial`) already existing; no destination used twice. Copies go to
   `.partial`, are hash-checked, then renamed. Mock-tree tests: valid ->
   10 installed; stale TIFF, edited manifest, dropped entry, duplicated
   entry, passed=false, changed variants, existing last destination,
   missing destination folder, missing or malformed report -> each
   refused with nothing copied.
3. **WB mode tag is not treated as proof of the same WB.** The report now
   keeps, for both the export and the RAF, WhiteBalance, ColorTemperature
   (MakerNote 0x1005, present only for Kelvin), WBFineTune and the
   0x144a/0x144b/0x144c levels (read as R/G/B gains; interpretation
   inferred from values, not documented). For as-shot Kelvin captures
   the exported Kelvin must equal the RAF's (gated). Gains are
   report-only: they differ between a RAF and every export because the
   WB shift is baked into them (e.g. DSCF0523 RAF R/G/B 636/302/414 with
   shift R+1/B-3 vs its Provia TIFF 2480/1208/1820 with shift 0). The
   planned E2-vs-E3 pixel comparison stays in the plan.

### Pilot comparison prepared (2026-11-18)

Codex: on arrival of the 10 pilot TIFFs, verify, install only on pass,
then inspect E2 vs E3 (WB) and E4 vs E5 (output DR400) and report actual
results before expanding the batch. `training/cuban_pilot_compare.py`
is written in advance and is **descriptive only** (no limits, no
scoring, no fitting). It refuses unless the pilot report passed and every
installed file re-hashes to its verified source. E2 vs E3: byte identity,
mean/p99 absolute difference, per-channel mean/median log2(E3/E2) over
pixels unclipped in both, mean-RGB chromaticity, alongside the recorded
WB metadata; this is Fuji's Auto-vs-As-Shot output change, not the app's
AWB error. E4 vs E5: byte identity, clipped/black fractions, luma
percentiles, and mean log2(E5/E4) in shadow/mid/highlight/top-1% bands
of E4 luma. Self-tests: identical input -> 0; blue x0.9 -> blue median
-0.152; halved -> -1.0 in every band. Pilot TIFFs not yet exported.

### Pilot exports verified and installed (2026-11-18)

All ten pilot TIFFs now pass `cuban_exports.py verify pilot`: exact
variant settings, source timestamp, orientation/shape, sRGB/ICC and the
unclipped-sample 16-bit check. The initial DSCF0964 outputs were JPEGs
with incorrect settings in the later variants; the user re-exported that
scene. All five replacement TIFFs pass. The verified ten files were
installed into Shoot 340 and Shoot 101 using the hash-bound installer;
the original Provia corpus files were preserved.

The descriptive comparison ran successfully after installation. Evidence:
`training/diagnostics/cuban-pilot-compare.json`. These comparisons sample
every fourth pixel in each direction and use sRGB-encoded values; their
log2 ratios are not physical exposure stops or inferred RAW WB gains.

- **DSCF0523, Kelvin capture:** E2 Auto versus E3 As-Shot gives a sampled
  mean absolute difference of 7.831 in 8-bit units. The export metadata
  distinguishes Auto from the original 6600 K setting; both retain recipe
  shift R+4/B-5. This confirms that the manual-capture pilot exercises a
  real WB-mode change.
- **DSCF0964, Auto capture:** E2 and E3 record matching WB levels and
  shifts, with a sampled mean absolute difference of 0.405 in 8-bit
  units. There is no meaningful mean chromaticity change in this pilot;
  the small pixel differences do not establish a different AWB algorithm
  or its cause.
- **Output DR400 is distinguishable from DR100:** E4/E5 sampled mean
  absolute differences are 1.265 (DSCF0523) and 2.535 (DSCF0964) in
  8-bit units. For DSCF0964, the fraction of sampled pixels with any
  encoded channel >=0.999 drops from 20.512% to 1.147%. This is an output
  rendering difference, separate from capture-DR conditioning.

This completes reference-export verification, not app recipe accuracy
acceptance. No new processor, fitting, monitor export or larger batch was
started. Next: select the separate train-tier reference cohort and freeze
stage/whole-recipe scoring limits before examining monitor results.
Seed 42 remains frozen and finalHeldOut stays locked and untouched.

### Cuban cohort, export list and criteria - PROPOSAL (2026-11-18)

Status: **proposed to Codex, not approved.** No exports requested, no
fitting, no app-vs-reference scoring run. Seed 42 frozen; the 12 monitor
scenes stay outside fitting; finalHeldOut not read.

**Pilot effect sizes (train-tier, Fuji vs Fuji, CIEDE2000 on sRGB-encoded
values via new `training/color_metrics.py`, Sharma-2005 self-test max
error 4e-5; every 4th pixel):**

| Fuji effect | DSCF0523 mean / p95 | DSCF0964 mean / p95 |
|---|---|---|
| Provia -> Classic Neg (E4 vs `xraw-phase3`) | 4.25 / 7.56 | 3.58 / 8.87 |
| DR400, whole frame (E4 vs E5) | 0.62 / 1.85 | 1.16 / 3.82 |
| DR400, E4 L*>=70 or clipped (5.9% / 27.9% of frame) | 4.04 / 12.43 | 2.99 / 4.48 |
| rest of stack (E5 vs E3) | 4.18 / 9.97 | 6.32 / 10.97 |
| Auto vs As-Shot WB (E3 vs E2) | 6.37 / 18.60 | 0.29 / 1.20 |
| grain (E2 vs E1) | 1.55 / 4.25 | 1.64 / 4.09 |

Consequence: whole-frame ΔE limits cannot detect DR400 or grain (their
frame effects are below any sensible frame limit), so stages are scored
against Fuji's own effect inside the region where it acts.

Also found: `cuban_pilot_compare.py`'s percentile bands break on ties.
With 20% of DSCF0964 E4 clipped, its "top1pct" band holds 431,211 pixels.
The mean differences and clipped fractions in b30f891 are unaffected; the
band rows for DSCF0964 should not be read as percentile bands. Not fixed
yet (would change committed evidence) - pending Codex.

**Cohort (`training/select_cuban_cohort.py` ->
`calibration-input/cuban-cohort-proposal.json`, copy committed at
`training/diagnostics/cuban-cohort-proposal.json`; metadata only).** Pool:
457-scene train pool minus DSCF0752 and Shoot 10 (456). Train and monitor
share no session (monitor 4/5/9/10) or shoot. One scene per shoot.
- **F fit, 24:** both pilots; sessions 1/3/7/8/11/12/13; WB 11 Auto,
  11 Kelvin, 1 White Fluorescent, 1 Cloudy; 6 bright-highlight,
  6 mixed-lighting; 7 at ISO>=1600 (2 at >=5000).
- **K train-check, 8:** shoot-disjoint from F; sessions 2/3/7/8/11/12/13;
  3 Auto, 4 Kelvin, 1 White Fluorescent; 1 bright, 2 mixed; 5 at
  ISO>=1600. Used to test the frozen criteria before any monitor run.
- **C isolated-control subset of F, 8** (both pilots plus one per
  session 7/8/12/13, the fluorescent scene, and an ISO>=5000 scene).
- Rule revisions, all before any pixel of these scenes was read: the
  first draft gave F 16 Kelvin / 6 Auto and C only 7 scenes; WB slot
  balancing then dropped bright-highlight to 2, so the bright-highlight
  minimum is now forced in like the rare WB presets. Fluorescent and
  Cloudy exist only in session 3; session 12 is Kelvin-only.

**Export list (each a separate X RAW Studio pass, own folder, verified
and hash-installed like the pilot):**
1. *Repeatability:* E1 and E2 re-exported for both pilots (4). Sets the
   X RAW Studio repeat floor and shows whether grain is deterministic;
   explains or not the 0.29 ΔE E2/E3 gap on DSCF0964 with identical
   metadata.
2. *Isolated controls on C* (8 scenes x 8 = 64), each = E5 (Classic Neg,
   DR400, As-Shot, all else 0/Off) plus exactly one setting: C1 Highlight
   -2, C2 Shadow +1, C3 Color +4, C4 CCE Strong, C5 FX Blue Strong,
   C6 NR -4, C7 Clarity -4, C8 WB shift R+4/B-5. Grain is isolated by
   E1 vs E2.
3. *E1-E5* for the 22 non-pilot F scenes and 8 K scenes (150).
Total 218 TIFFs. Proposed order: batch 1 = items 1 and 2 plus E1-E5 for
the 6 non-pilot C scenes (98); batch 2 = the remaining 24 scenes x E1-E5
(120).

**Predeclared criteria (to freeze on Codex approval, before any
app-vs-reference run on F, K or monitor).** Space: CIELAB D65 from the
sRGB-encoded 16-bit TIFFs, every 4th pixel. Per scene report: mean and
p95 ΔE00; signed ΔL* in reference L* bands <25 / 25-75 / >75; signed
ΔC*ab where reference C*>10; signed hue error where C*>20.

*Effect residual* for stage s over base b in region R:
rho = mean_R ΔE00(Lab(ref_b) + Lab(app_s) - Lab(app_b), Lab(ref_s)) /
mean_R ΔE00(Lab(ref_b), Lab(ref_s)). 0 = exact, 1 = no better than
doing nothing. Measurable only where Fuji's effect over R is >=0.5 ΔE00;
a stage with fewer than 3/4 of scenes measurable is reported as
unmeasurable, not passed. Regions: DR400 = ref E4 L*>=70 or clipped;
Highlight = base L*>=60; Shadow = base L*<=35; Color = C*>=20; CCE =
C*>=30; FX Blue = hue 200-290 deg and C*>=15; WB shift, NR, Clarity =
whole frame (NR/Clarity also report band-pass L* detail-energy change,
app vs Fuji).

Per-scene limits (chosen so each is at most about half the pilot effect
it guards):

| Stage | mean ΔE00 | p95 ΔE00 | other |
|---|---|---|---|
| E4 simulation base | <=2.0 | <=5.0 | abs mid ΔL* <=2.0; rho(Provia->E4) <=0.5 |
| E5 DR400 | <=2.5 | <=6.0 | rho_DR <=0.5; clipped-fraction gap <=0.02 |
| E3 rest of stack | <=3.0 | <=7.5 | abs mid ΔL* <=2.5; abs mean ΔC* <=3.0; rho(E5->E3) <=0.5 |
| E2 whole grain-free recipe (app AWB) | <=4.0 | <=10.0 | abs mid ΔL* <=3.0; rho_WB(E3->E2) <=0.6 where measurable |
| C1-C8 controls | - | - | rho <=0.5 where measurable; sign of the region's mean signed effect matches Fuji |

E1 grain (statistical, no pixelwise ΔE): residual g = E1 - E2 in L*,
app vs Fuji; per L* quintile std ratio in [0.75, 1.33]; radial power-
spectrum centroid ratio in [0.80, 1.25]; a*/b* residual std ratio in
[0.5, 2.0] (or both <0.3); abs mean-residual gap <=0.5 L*; plus a
recorded visual review of 100% crops (shadow/mid/highlight per scene).

Aggregate, per stage and set: every scene reported with signed errors;
scene pass rate >=7/8 on K (>=11/12 later on monitor); median mean ΔE00
E4 <=1.5, E3 <=2.5, E2 <=3.0; no systematic bias: abs cross-scene mean
mid ΔL* <=1.0, abs mean ΔC* <=1.5, abs mean hue error <=3 deg. A visible
artifact (banding, halos, colour blotches, hard clipping onset) fails a
scene regardless of numbers. F results may drive fitting but may never
change a limit. Only permitted hold: if the repeat floor exceeds 0.25 x
the smallest limit it would affect, stop and review with Codex - no
automatic loosening.

**Open design point for Codex:** which app path is scored. Seed 42
targets Provia, and the app has a Classic Negative LUT; the plan says
start from `.fjlrg`, so the processor that maps `.fjlrg` to E4 (new head
vs Provia-then-LUT) needs a decision before any fitting.

### Codex proposal review and corrected descriptive report (2026-11-18)

**Decision:** the F/K/C cohort is acceptable for this development pilot.
K is shoot-disjoint from F but shares capture sessions; it is an internal
development check, not independent session-level acceptance. Monitor
remains excluded from fitting; finalHeldOut stays locked and unread.

**RAW path:** use a separate Classic Negative variant/head starting from
the canonical `.fjlrg`, with the same capture-DR interpretation. A
warm-start from a separate copy of Round 11 weights may be proposed in
the fitting plan, without modifying or reselecting frozen seed 42.
Seed-42 Provia followed by the existing Classic Negative LUT may be a
diagnostic baseline; it is not the approved recipe-processing path.
No model or fitting implementation is started by this review.

**Next export authorization is the four repeatability files only:**
E1 and E2 once more for each pilot, under new names/folders preserving
the first exports. Verify settings and hashes, measure the repeat floor,
and apply the proposed hold rule before requesting the remaining 214
exports. The 98/120 batching proposal is not yet released wholesale.

Before freezing the scoring implementation and requesting that larger
batch, make the following evaluation contract explicit:

- Use the existing orientation-corrected, 16-bit target-loader path to
  convert each TIFF to linear sRGB and area-downsample to the exact
  corresponding `.fjlrg` dimensions. Then convert the matched app and
  reference outputs to encoded sRGB/Lab for colour metrics. Comparing a
  full-resolution TIFF sampled every fourth pixel to the 1536px model
  output is not a valid correspondence. Dense working-resolution
  colour comparisons are the default; subsampling cannot replace
  alignment or downsampling. Detail/grain tests need a declared scale
  and matched sampling; this stage cannot establish 40MP readiness.
- Keep effect residual rho as a supplemental contrast test alongside
  absolute stage errors and visual artifact checks. Define minimum
  region occupancy, empty-region handling and the signed quantity for
  each control. NR/clarity need a repeat-floor-aware detail statistic;
  colour-difference rho alone must not decide their accuracy. If a
  control lacks enough measurable examples, it is unvalidated and
  requires appropriate train-only examples, not an automatic pass.
- Compute the whole-recipe pass from the conjunction of its applicable
  per-scene requirements. Independent 7/8 pass counts at different
  stages can otherwise conceal several photos that fail the recipe.
  Report every failing scene and the visual review outcome.
- The colour-metric self-test currently covers nine selected Sharma
  reference pairs. Validate the complete published reference set,
  including hue-wrap cases, before using it as a gate.

The proposed numerical limits remain development targets, not evidence
of universal or perceptually indistinguishable Fuji parity. Do not tune
them from K/monitor outcomes. No app-vs-reference evaluation or fitting
has been authorized here; stage implementation and the final metric
contract still need to be recorded before those start.

**Percentile-label correction:** `cuban_pilot_compare.py` now writes
`training/diagnostics/cuban-pilot-compare-v2.json`; the original committed
JSON remains unchanged. Bands explicitly describe intervals between
percentile thresholds, keep ties together, and report actual occupancy
before/after the near-black exclusion. They do not claim fixed-size rank
quantiles. DSCF0964's >=p99 band contains 431,211 sampled pixels (17.3287%
of the sampled frame), rather than 1%. Whole-frame differences and
clipped fractions match the original report exactly. Empty bands yield
null statistics. The sampled-pixel equality field is named explicitly.
Verified on tied-highlight and constant/black synthetic images, followed
by regeneration against both hash-verified installed train-tier pilots.

### Repeat exports set up; evaluation contract draft (2026-11-18)

Following Codex's d9d791e review. Only the four repeat exports are
requested. No app-vs-reference run, no model or fitting code, no further
exports. Seed 42 frozen; monitor excluded from fitting; finalHeldOut
unread.

**Repeat exports.** `cuban_exports.py prepare-repeat` made
`calibration-input/cuban-repeat-batch/` (both pilot RAFs;
`R1-full-recipe-repeat` = E1 settings, `R2-no-grain-repeat` = E2
settings). `verify repeat` applies the full E1/E2 checks and also requires
the first export to still exist and the repeat's TIFF DateTime (export
time) to be present and different from the first export's, so a copied
file cannot pass. Byte identity with the first export is recorded, not
rejected, because a deterministic renderer would produce it. Installs go
to `xraw-cuban-R1-full-recipe-repeat.tiff` and
`xraw-cuban-R2-no-grain-repeat.tiff`; the first exports are untouched.
Tests: copies of the installed originals put in a temporary batch were
all rejected (byte-identical flag True); a pilot re-verify written to the
scratchpad still passed 10/10.

**Repeat floor** (`training/cuban_repeat_floor.py`, run after verify and
install). Pilot and repeat files are re-hashed against both reports.
Dense CIEDE2000 between first and repeat for E1 and E2 in the evaluation
space below; detail-RMS floors for both detail bands; grain determinism
as the correlation of the E1-E2 L* residual between runs. Hold rule
declared before any repeat exists: HOLD if on either pilot the E2-vs-R2
floor has mean ΔE00 > 0.375 (0.25 x the E4 median limit 1.5) or p95 >
1.25 (0.25 x the E4 p95 limit 5.0). A hold means review with Codex,
never automatic loosening.

**CIEDE2000 validation.** The self-test now uses the complete published
set: Sharma's `ciede2000testdata.txt`, 34 pairs including the hue-wrap
and achromatic cases. It is fetched from the author's site, committed as
`training/data/ciede2000-sharma-testdata.txt` and pinned by SHA-256. Its
values are identical to scikit-image v0.24.0's copy. All 34 pairs match
exactly to 4 decimal places in both argument orders, with max abs error
4.95e-5.

**Evaluation contract (draft, to freeze with Codex before any
app-vs-reference run):**
1. *Space.* Reference: the existing `target_tiff` loader, which reads the
   16-bit TIFF, decodes to linear sRGB and applies EXIF orientation, then
   `INTER_AREA` downsamples in linear light to the scene's exact
   `browser-phase3-linear.fjlrg` dimensions (1026x1536 for both pilots).
   App: its linear output at the same working resolution. Both go to
   CIELAB D65 via `linear_srgb_to_lab`, clipped to [0,1]. This matches the
   encoded path to 1e-13. Every pixel is used; no subsampling. The
   every-4th-pixel full-resolution figures in the pilot reports are
   descriptive only and are not comparable with the limits.
2. *Scale.* All colour, tone, region, detail and grain metrics are
   computed at that working scale. Nothing here establishes
   full-resolution (40MP) behaviour. Native-resolution grain crops may be
   reported as diagnostics only. Observed on DSCF0523: at working scale
   grain is 0.71 mean ΔE00, against 1.55 at native every-4th-pixel
   sampling, and it moves fine-band detail RMS by only 0.1%. So detail
   RMS is not a grain test. Grain keeps its residual-std, power-spectrum
   and chroma statistics, all computed at working scale.
3. *Regions.* A region R is scored on a scene only if it covers at least
   2% of the working frame (about 31.5k pixels at 1026x1536). It must also
   carry a Fuji effect of at least 0.5 mean ΔE00 and at least 4x the E2
   repeat-floor mean. Otherwise that scene is *unmeasurable* for that
   check, which counts as neither pass nor fail. An empty region gives
   null statistics.
4. *Signed quantity per control* (app sign must match Fuji's measured
   sign; no sign is assumed in advance):

   | Control | Region | Signed quantity |
   |---|---|---|
   | C1 Highlight -2 | base L* >= 60 | mean ΔL* |
   | C2 Shadow +1 | base L* <= 35 | mean ΔL* |
   | C3 Color +4 | C* >= 20 | mean ΔC* |
   | C4 CCE Strong | C* >= 30 | mean ΔL* (ΔC* also reported) |
   | C5 FX Blue Strong | hue 200-290 deg, C* >= 15 | mean ΔL* (ΔC* also reported) |
   | C6 NR -4 | frame | relative change of fine-band (s1-s3) L* detail RMS |
   | C7 Clarity -4 | frame | relative change of mid-band (s4-s16) L* detail RMS |
   | C8 WB shift | frame | mean (Δa*, Δb*); direction cosine with Fuji's >= 0.8 |

5. *NR and Clarity are repeat-aware.* A scene is measurable only if
   Fuji's relative detail change is at least 3x that band's repeat floor.
   They pass if the absolute difference between app and Fuji relative
   change is at most max(0.3 x Fuji's change, 3 x floor). ΔE-based rho
   never decides them alone.
6. *Unmeasurable controls.* A control needs at least 6 of 8 C scenes
   measurable. Otherwise it is *unvalidated*. It then needs extra
   train-tier examples, chosen by metadata or region occupancy before
   their pixels are scored, and it is never an automatic pass. A recipe
   with an unvalidated control cannot be reported as validated for that
   setting.
7. *rho stays supplemental.* The effect residual rho is a contrast test
   alongside the absolute stage limits and the visual artifact review. It
   is never a substitute for them.
8. *Whole-recipe pass is per scene and conjunctive.* A scene passes the
   recipe only if it passes every applicable per-scene requirement: E2
   limits, plus E4, E5 and E3 per-scene limits, plus the E1 grain
   statistics, plus no visible artifact. The set-level rule (at least 7/8
   on K, later at least 11/12 on monitor) counts scenes passing that
   conjunction. The median and bias aggregates apply on top. Every
   failing scene is reported with each failed requirement and its visual
   review outcome. Separate stage pass rates are reported for diagnosis
   only.
9. The numerical limits are development targets, not a claim of
   universal or perceptually indistinguishable parity. They are never
   tuned from K or monitor outcomes.

### Repeat exports verified and repeat floor measured (2026-11-18)

All four user-exported repeat TIFFs passed `cuban_exports.py verify
repeat`, including E1/E2 settings, 16-bit/sRGB/orientation/source checks
and the distinct export timestamps. The hash-bound installer saved R1/R2
into Shoot 340 and Shoot 101 without overwriting the original pilot or
Provia files. `cuban_repeat_floor.py` completed successfully, using both
verification reports and re-hashing the installed files. Evidence:
`training/diagnostics/cuban-repeat-floor.json`.

At the exact 1026x1536 working scale, the grain-off E2/R2 comparisons are:

| Pilot | Mean Delta E00 | p95 Delta E00 |
|---|---|---|
| DSCF0523 | 0.1093 | 0.2738 |
| DSCF0964 | 0.1120 | 0.2442 |

Both are below the predeclared hold limits (mean 0.375, p95 1.25), so
`hold=false`, no hold reasons, exit 0. Limits were not changed.

For the grain-on E1/R1 comparisons, means are 0.9304 and 1.0268 Delta
E00; these are descriptive, not held against the grain-off limits.
First-versus-repeat grain-residual L* correlations are 0.0188 and 0.0094,
while residual standard deviations are stable (1.0047/1.0053 and
1.1632/1.1631). On these two pilots this supports assessing the grain
statistically rather than requiring the same pixel pattern. It does not
establish a universal repeat floor across other settings or scenes.

The four-repeat step is complete. No additional export batch, fitting,
model change or app-vs-reference scoring was started. Next: finalize the
draft evaluation contract and prepare the remaining train-tier reference
export batches for a concrete user checklist. Seed 42 remains frozen;
monitor stays outside fitting and finalHeldOut was not accessed.

### Evaluation contract finalized; export batch 1 prepared (2026-11-18)

**Contract frozen** as `training/cuban_contract.py` (version 1),
**sha256 `c3f840ef9fad9d48de83b4edf183f2434523f352f358305b91eb7764d22d0067`**.
It is the draft above with every open item fixed, written before any
app-vs-reference run. Future scoring code must import it and refuse to
run on a hash mismatch. Any revision needs Codex approval and a dated
entry here, and may never be driven by K or monitor outcomes. Additions
to the draft:
- *Repeat floor* constants are the larger value of the two pilots from
  58816af: E2/R2 mean 0.1120, p95 0.2738 ΔE00; fine and mid detail floor
  3e-5 relative each; grain residual-std floor 6e-4 relative. The grain
  pixel pattern is not repeatable (correlation 0.0188 / 0.0094), so E1 is
  never scored pixelwise. The floor term 4 x 0.112 = 0.448 sits below the
  0.5 ΔE00 minimum effect, so the 0.5 minimum binds. The NR and Clarity
  tolerance max(0.3 x |Fuji|, 3 x floor) is effectively 0.3 x |Fuji|.
- *Clipped pixel*: any linear channel >= 0.999 at working scale. *Tone
  bands*: reference L* <25 / 25-75 / >75. *Hue*: signed degrees wrapped
  to +-180 where reference C* > 20.
- *rho bases*: Provia -> E4 (base is `xraw-phase3`), E4 -> E5, E5 -> E3,
  E3 -> E2. Regions are taken on the reference base image.
- *Grain*: bands are reference E2 L* quintile thresholds, with ties kept
  together and occupancy reported. The power spectrum is a Hann-windowed
  radial power spectrum of the L* residual, using the frequency-weighted
  centroid over 0.02-0.5 cycles/px. Visual review uses shadow, mid and
  high working-scale crops, recorded per scene.
- Unchanged from the draft: the stage limits, the per-scene conjunctive
  whole-recipe pass, the K and monitor set rules, the aggregates, the
  measurability rules, the unvalidated-control handling, and rho as
  supplemental only.

**Batch 1 (94 TIFFs), prepared by `cuban_exports.py prepare-batch1`.**
The cohort proposal in `calibration-input` is first checked byte-equal to
the committed copy.
- `calibration-input/cuban-controls-batch/`: the 8 C scenes (DSCF0523,
  0964, 0514, 1057, 0706, 0753, 0825, 1013) x C1-C8 = 64. Each control is
  E5 plus exactly one setting. Smoke test: the verified pilot E5 standing
  in for each control was flagged for exactly that control's one setting
  and nothing else.
- `calibration-input/cuban-cohort1-batch/`: the 6 non-pilot C scenes x
  E1-E5 = 30.
- Both are verified and hash-installed like the pilot, as
  `xraw-cuban-<variant>.tiff`, with no overwrites. DSCF0514 was captured
  with White Fluorescent (0x302), so its "As Shot" must stay White
  Fluorescent. Batch 2 (the remaining 24 scenes x E1-E5 = 120) is not
  released. Nothing is fitted.

### C1-C8 exports verified and installed (2026-11-18)

The user reported that X RAW Studio exports into the working folder.
At their request, all eight control RAFs were physically copied into
each C1-C8 subfolder (64 real RAF copies, SHA-256 checked against their
train-tier sources; existing TIFFs preserved). Copy evidence is at
`calibration-input/cuban-controls-raf-copies.json`.

All **64 control TIFFs** now pass `cuban_exports.py verify controls`:
exact per-control settings, source timestamps/WB (including DSCF0514's
White Fluorescent), orientation/dimensions, sRGB/ICC and 16-bit checks.
`install controls` completed with exit 0, checking each copied TIFF's
SHA-256 before its final rename. The permanent Shoot folders contain
all 64 expected files and no unfinished `.partial` copies. The user's
staging/export folders remain intact. No re-exports are needed for C1-C8.

Evidence: `training/diagnostics/cuban-controls-verification.json`, with
the exact scene/variant coverage, installed paths, verified file hashes,
and manifest/variant/report fingerprints. Contract v1 remains unchanged
at `c3f840ef9fad9d48de83b4edf183f2434523f352f358305b91eb7764d22d0067`.

Before C1, the user also exported eight BASE-setting TIFFs into the root
of `cuban-controls-batch`. These were verified against E5 settings using
the same verifier (report: `calibration-input/cuban-controls-base-export-check.json`).
They are potential E5 references for the six non-pilot cohort1 scenes;
check their recorded hashes and use the verified staging/install path
before treating them as installed E5 targets. Avoid requesting six
unnecessary E5 re-exports if those files remain unchanged.

At this checkpoint the cohort1 E1-E5 output subfolders are still empty.
The next reference step is the six non-pilot scenes' recipe variants,
with the reusable E5 files accounted for. Batch 2 remains unreleased;
no fitting, app-vs-reference scoring, or model change was run. Seed 42
is frozen, monitor remains excluded from fitting, and finalHeldOut was
not accessed.

### Cohort1 working folders prepared; E5 reused (2026-11-18)

To support X RAW Studio exporting into its working folder, each cohort1
E1-E4 subfolder now contains real copies of the six train-tier RAFs
(24 copies total), checked against source SHA-256 hashes. The six earlier
BASE TIFFs still match `cuban-controls-base-export-check.json`; copies
were placed in `cuban-cohort1-batch/E5-classic-neg-dr400/` and hash-checked.
The standard export verifier was then run scoped to E5 on the cohort1
manifest: all six pass their settings, 16-bit, colour-space, orientation
and source checks. Originals and existing pilot targets were preserved.

Evidence: `calibration-input/cuban-cohort1-stage-check.json` and
`calibration-input/cuban-cohort1-e5-reuse-export-check.json`. E5 is staged
and verified, not yet installed into the permanent Shoot folders; the
normal cohort1 verification/install follows when E1-E4 also arrive.

Only **24 new user conversions** remain for this part of batch 1:
six photos each at E1, E2, E3 and E4. Do not request six redundant E5
conversions. The next user pass is E1 full recipe, with Auto WB, recipe
shift R+4/B-5 and Strong/Large grain, saved as 16-bit sRGB TIFFs in its
working folder. No fitting, model change or app scoring; the contract,
seed 42 and protected-tier restrictions are unchanged.

### Cohort1 E1-E5 verified and installed; batch 1 complete (2026-11-18)

The user completed E1-E4 for all six cohort1 scenes. E1, E2 and E3 were
checked individually before the following pass. After E4 arrived, the
standard full `verify cohort1` checked all 30 scene/variant pairs,
including the six reused E5 TIFFs: **30/30 passed** source/settings,
WB/shift, orientation/shape, sRGB/ICC and 16-bit checks. `install cohort1`
completed with exit 0; each permanent copy was SHA-256 checked before
its final rename. No existing targets were overwritten and staging
folders remain intact.

Evidence: `training/diagnostics/cuban-cohort1-verification.json` records
exact coverage, permanent paths, verified hashes and report/manifest/
variant fingerprints. All 30 permanent targets are present, with no
unfinished `.partial` files. Combined with the 64 control references,
export batch 1 is complete: **94/94 permanent targets**. The six existing
BASE TIFFs supplied E5, saving the user six repeat conversions.

Contract v1's fingerprint is unchanged:
`c3f840ef9fad9d48de83b4edf183f2434523f352f358305b91eb7764d22d0067`.
No new model, fitting, app-vs-reference scoring or larger export batch
was started. Next is review of reference-only effect/measurability
coverage under the contract and a concrete RAW-input Classic Negative
fitting plan before further work is released. Batch 2's 120 exports
remain on hold; seed 42 remains frozen, monitor remains excluded from
fitting, and finalHeldOut was not accessed. This milestone verifies
calibration references, not recipe accuracy in the live app.

### Reference-only effect/measurability audit under contract v1 (2026-11-18)

`training/cuban_reference_audit.py` imports contract v1 and refuses
unless the hash is `c3f840ef...`. It runs the full Sharma self-test and
re-hashes every one of the 8 C scenes' E1-E5 and C1-C8 TIFFs against the
pilot, controls and cohort1 verification reports. Everything is Fuji vs
Fuji at working scale. There is no app output, model or fitting, and no
monitor or finalHeldOut data. Evidence:
`training/diagnostics/cuban-reference-audit.json`. Each control is
measured against its own scene's E5 base. Minimum effect = max(0.5,
4 x 0.112) = 0.5 ΔE00; minimum region 2%; detail threshold 3 x 3e-5.

| Control | Measurable / 8 | Status | Fuji signed effect (range over measurable scenes) |
|---|---|---|---|
| C1 Highlight -2 | 8 | sufficient | ΔL* -1.78 to -3.34 in L*>=60 |
| C2 Shadow +1 | 7 | sufficient | ΔL* -0.45 to -2.47 in L*<=35 (DSCF0706 effect 0.39 ΔE00, below 0.5) |
| C3 Color +4 | 5 | **UNVALIDATED** | C*>=20 occupancy <2% on DSCF0964, 0706, 0825 |
| C4 CCE Strong | 4 | **UNVALIDATED** | C*>=30 occupancy <2% on DSCF0964, 1057, 0706, 0825 |
| C5 FX Blue Strong | 0 | **UNVALIDATED** | blue region occupancy <=0.7% on every scene (0 on four) |
| C6 NR -4 | 8 | sufficient | fine detail +0.29% to +2.98% |
| C7 Clarity -4 | 8 | sufficient | mid-band detail -5.9% to -10.9% |
| C8 WB shift R+4/B-5 | 8 | sufficient | (Δa*, Δb*) all positive, e.g. (2.03, 2.68) on DSCF0523 |

Stages over the same 8 scenes:
- Provia->E4: 8/8 measurable, 1.30-4.77 mean ΔE00.
- E4->E5 DR400: 7/8. On DSCF1013 the DR region is below 2%.
- E5->E3: 8/8.
- E3->E2 WB: 5/8. The 3 Auto captures (DSCF0964, 1057, 0825) show
  0.11 ΔE00, which equals the repeat floor. So X RAW Studio Auto
  reproduces the camera's Auto there. The contract already scores
  rho_WB only where measurable.
- Fuji grain residual std: L* 0.65-1.29; a* and b* 0.15-0.72.

Observation, no criterion change: NR -4's Fuji effect is small (fine
detail +0.3% on five scenes). Its contract tolerance, max(0.3 x
|Fuji|, 3 x floor), is therefore about 0.1% absolute on those scenes.
It is well defined but strict.

**Gap remedy, proposed and not requested.**
`training/select_cuban_control_extras.py` measures region occupancy on
each candidate's existing *Provia* reference (`xraw-phase3.tiff`) at
working scale. That honours contract rule 6: extras are chosen by
occupancy before any of their Classic Neg pixels exist. The pool is the
456-scene train pool minus every F/K shoot, one scene per shoot, giving
424 candidates. Rule fixed before running: Provia blue occupancy >=0.06
and C*>=30 occupancy >=0.06 (margins, since Classic Neg is less
saturated), ranked by blue occupancy, at most 2 per session, take 8.
**Result: only 40 eligible, concentrated in sessions 3 (31), 11 (8) and
1 (1), so the rule yields 5, not 8:** DSCF0479, 0470 (session 3), 0914,
0929 (session 11), 0392 (session 1). Proposal at
`training/diagnostics/cuban-control-extras-proposal.json`. Exports if
approved: E5 + C3 + C4 + C5 per scene, so 20 for the 5, or 32 for 8.
Open for Codex:
1. Five extras cannot bring C5 to 6 measurable. Raising the session cap
   to 4 would give 8 scenes (sessions 3:4, 11:3 or 4, 1:1). That is a
   rule change after seeing occupancy counts, though not Cuban pixels,
   so it needs your decision.
2. With extras, the contract's "6 of 8 C scenes" denominator needs
   restating, for example ">= 6 measurable scenes across C + extras".
### Proposed .fjlrg -> Classic Negative fitting plan (2026-11-18)

**Proposed `.fjlrg` -> Classic Negative fitting plan (for Codex review; nothing
implemented or run).** One isolated change per step. Each step has its own
predeclared check under contract v1 (sha `c3f840ef...`). A step only
proceeds when the previous one passed and Codex approved it.

*Architecture.* Step 1 uses a separate `HybridPredictor` instance with the
same config as Round 11: grid 8x9, base_ch 16, refinement base_ch 8 with
max_delta 0.08, and the capture-DR condition plane (all cohort scenes are
capture DR400, so the plane is constant 1.0). It is saved under new
`cn-*` checkpoint names and never writes to `r11-*` files. Seed 42 stays
frozen and selected.

*Initialization.* The weights are a copy of `r11-drcond-seed42-best.pt`.
The file's SHA-256 is recorded before and after training, and it must
not change. Rationale: Round 11 already maps `.fjlrg` to a Fuji render
from 599 train scenes. Classic Negative versus Provia is mostly a
colour/tone transform of that render: 3.6-4.3 mean ΔE00 on the pilots.
Twenty-four target scenes are too few to learn the RAW mapping from
scratch.

*Trainable components (step 1).* The grid encoder stays frozen (98,064
params; scene-content features learned on 599 scenes). The grid head
(7,020) and the refinement net (10,635) are trained. Total trainable:
17,655 of 115,719. Training the full model is not a step-1 variant. It
becomes a later, separate change only if step 1 fails in a way that
implicates the encoder.

*Data.* Fit only on F (24 scenes). This needs batch 2's E1-E5 for the 16
F scenes that don't have them yet, plus the 8 K scenes for the check.
Batch 2 stays held until this plan is approved. Inputs are
`browser-phase3-linear.fjlrg`. Targets are the E4 TIFFs, loaded by the
contract's target loader at the `.fjlrg` working size. The loss is
Round 11's per-scene L1 in linear light, with the schedule unchanged
except for LR and epochs, which are set by CV. K, monitor and
finalHeldOut have no code path in training.

*Model selection.* Use 4-fold CV on F, grouped by shoot (one scene per
shoot already), with folds stratified by session and WB. CV picks only
the epoch count and an LR from {1e-4, 3e-4}, by mean fold-validation L1.
The final model is then trained on all 24 F scenes with that fixed
setting. Seeds 42/43/44 are trained. Seed 42 is primary and predeclared;
43 and 44 are reported, never used to pick a winner.

*Step 1 check (E4 simulation base), run once on K after training.* Per
scene, apply the E4 limits from contract v1: mean <=2.0, p95 <=5.0,
abs mid ΔL* <=2.0, and rho(Provia->E4) <=0.5, with the app's Provia
render from seed 42 as `app_b`. It passes if at least 7/8 K scenes pass
the conjunction, median mean ΔE00 <=1.5, and the bias aggregates hold.
Every scene is reported. Two baselines are scored the same way, for
context only: (a) Round 11 seed 42 untouched, meaning no Classic Neg;
(b) seed 42 followed by the app's existing Classic Negative LUT, the
diagnostic baseline. Monitor scoring happens only after K passes and
with Codex approval, as development-validation.

*Later steps (each separately approved; outlines only).*
1. **Step 2, output DR400.** First score the app's existing shader DR
   stage applied to the step-1 output against E5, using rho_dr400 and E5
   limits on F (train). If it fails, the isolated change is a second
   condition plane for output DR, trained on E4 and E5 targets.
2. **Step 3, recipe controls C1-C8.** Validate each existing shader stage
   in isolation by applying it to Fuji's own E5 reference and comparing
   with Fuji's C_k reference. This avoids any head error. Any refit uses
   the C scenes (train) only, and unvalidated controls follow contract
   rule 6.
3. **Step 4, stack and AWB.** Score E3 (As-Shot) and then E2 (app
   gray-world AWB) end to end on K under the conjunctive whole-recipe
   rule.
4. **Step 5, grain.** Compare the shader grain with Fuji's statistics
   under the contract's grain rules, plus the visual review.

### Codex review: targeted support and E4-first fitting sequence (2026-11-18)

**Reference audit reviewed.** The control gaps are informative-region
coverage gaps, not demonstrated app failures: C3, C4 and C5 remain
unvalidated. The three Auto-capture E3/E2 pairs show no distinguishable
mode effect beyond the measured repeat floor on these inputs; this does
not establish a universal equivalence of Fuji's WB modes.

**Control-support selection approved, export request deferred.** Raising
the per-session cap based only on existing train-tier Provia occupancy
counts is approved, with explicit coverage of all three eligible
sessions: session 1 takes 1, session 3 takes 4, session 11 takes 3. Within
each quota, rank by descending blue occupancy, then filename. This is a
documented post-count rule revision, before any selected Classic
Negative exports or app outcomes. It does not create independent test
data. The original proposal remains unchanged; the resulting eight
scenes and provenance are frozen in
`training/diagnostics/cuban-control-support-selection.json`.

The selected scenes are DSCF0392, DSCF0479, DSCF0470, DSCF0478,
DSCF0473, DSCF0914, DSCF0929 and DSCF1007. They are an additional T8
support set for C3-C5, not automatic additions to F or K. Their future
E5+C3+C4+C5 exports total 32, but are not the next user request.

**Approved contract-v2 coverage clarification to implement before new
scoring:** retain the original C8 and its results. C1/C2/C6/C7/C8 keep
their original >=6/8 measurable-scene requirement. For each of C3/C4/C5,
the preselected T8 support set must independently provide >=6/8
measurable scenes. All measurable original C3-C5 cases remain scored and
reported; an original failure cannot be discarded by adding successful
support scenes. Do not redefine the pooled requirement as six of sixteen.
Effect/region/detail thresholds, accuracy limits and recipe conjunction
rules are unchanged. Preserve contract v1 and its existing evidence;
record and pin v2's fingerprint before any new scoring. V1 has not been
edited by this review.

**Release the smaller E4-only step next:** step 1 requires only E4 from
the remaining 16 F and 8 K scenes, so prepare 24 exports, not 120. The
exact rows and source-cohort hash are in
`training/diagnostics/cuban-base-step1-export-plan.json`. Keep F/K roles
distinct. The other 96 batch-2 recipe exports and 32 support exports
stay deferred. Prepare real RAF copies in the working output folder,
matching the user's X RAW Studio workflow, and a single E4-settings
checklist. No new photography is required.

**Step-1 design approved with these changes:**
- Keep the separate RAW-input HybridPredictor, frozen encoder and
  trainable head/refinement, copied Round-11 initialization, capture-DR
  condition and protected original checkpoint exactly as proposed.
- Use four folds grouped by whole capture session, not shoot. A single
  image per Shoot folder does not remove adjacent/session-related
  dependence. Balance session sizes/WB where possible without splitting
  any session; freeze the fold manifest before training. Rare WB modes
  confined to one session cannot be independently stratified.
- Predeclare one local search: LR in {1e-4, 3e-4}, maximum 20 epochs,
  CV seed 42. Select LR and epoch by mean out-of-fold per-scene linear
  L1 over F. Keep the schedule horizon fixed at 20 for both CV and the
  final fits, stopping at the chosen epoch without rescaling the cosine
  schedule. No new capacity/loss/regularization search is authorized.
- Fit final seeds 42/43/44 at that fixed setting; 42 stays primary,
  other seeds are report-only. Record initialization, optimizer/schedule,
  fold and data hashes, trainable parameter names and source checkpoint
  hash before/after. Check training inputs are F-only by explicit
  allowlist. Implementation and F-only smoke tests may proceed; full
  fits wait for verified E4 targets, the pinned revised contract and
  the predeclared run/fold manifests.
- Once those conditions hold, the authorized step-1 run is the above
  CV/final-fit sequence followed by one frozen K evaluation of E4-only
  requirements and its applicable aggregates. This is not a whole-recipe
  pass. No choosing a backup because it scores better on K. K shares
  sessions with F, and Round 11's warm-start already used their Provia
  targets: label it an internal simulation-adaptation check, not wholly
  unseen-RAW acceptance.
- Initial evidence is restricted to capture DR400 and this cohort's
  ISO range. Keeping a capture-DR input plane does not establish Classic
  Negative accuracy for DR100/200 captures without corresponding
  references. Monitor scoring still needs separate review; finalHeldOut
  remains locked and unread. Subsequent DR/control/stack/AWB/grain steps
  remain separately scoped.

This review produced decisions and metadata plans only. No export,
training, app score or model change was run.

### E4-only 24-scene working folder ready (2026-11-18)

The next user export folder is
`calibration-input/cuban-base-step1-batch/E4-classic-neg-only/`.
It contains 24 real RAF copies from the approved step-1 export plan,
each checked against its source SHA-256. The manifest
`calibration-input/cuban-base-step1-manifest.json` preserves the 16 F
fit roles and 8 K train-check roles; it does not merge them for fitting.
Staging evidence is `calibration-input/cuban-base-step1-stage-check.json`.

`cuban_exports.py verify base-step1` and `install base-step1` now use
this manifest and only the existing E4 settings. The user should open
the working folder and export all 24 as full-resolution, 16-bit sRGB
TIFFs: Classic Negative, output DR100, As-Shot WB, shift 0/0, other
controls zero/off. No E1/E2/E3/E5 conversions are requested for this
batch yet. No TIFF export, training, app scoring or model change was
run during folder preparation. Existing targets and the contract are
unchanged; finalHeldOut remains locked and unread.

### E4-only step-1 exports verified and installed (2026-11-18)

The user completed the 24-scene E4-only batch. The standard
`cuban_exports.py verify base-step1` passed **24/24**: no missing TIFFs,
correct E4 settings (Classic Negative / output DR100 / As-Shot WB /
shift 0/0 / controls zero-off), source identity, orientation/dimensions,
sRGB/ICC and 16-bit checks. `install base-step1` completed with exit 0,
checking each copied TIFF's SHA-256 before its final rename. All 24
permanent targets are present with no unfinished `.partial` files.
Staging files were preserved and no existing target was overwritten.

Evidence: `training/diagnostics/cuban-base-step1-verification.json`,
including exact source/target coverage, file hashes, manifest/report/
variant fingerprints and F/K roles. The new data is **16 F fit scenes
plus 8 K train-check scenes**. With the eight existing F targets, E4
availability is now **F 24/24, K 8/8**, with the canonical `.fjlrg`
inputs present. Keep the sets separate in the fitting loader.

No repeat exports are needed for this batch. No fitting, target-cache
precompute, app-vs-reference scoring or model change was run here.
Contract v1 is still unchanged; the approved v2 coverage amendment and
frozen session-grouped folds/run manifests must be implemented before
the previously authorized step-1 fitting sequence starts. Original
seed 42 remains frozen, further recipe/control exports remain deferred,
monitor is excluded from fitting and finalHeldOut was not accessed.

### Contract v2 pinned; step-1 folds, run manifest and K evaluation frozen before training (2026-11-19)

**Contract v2** (`training/cuban_contract_v2.py`) implements only the
approved coverage clarification. Contract v1 (`training/cuban_contract.py`)
is byte-identical, still hashes to `c3f840ef...`, and its evidence (repeat
floor, reference audit) stays under v1. v2 deep-copies v1 and adds three
things: `controls.coverage` (C1/C2/C6/C7/C8: >= 6 of the original 8 C
scenes; C3/C4/C5: >= 6 of the T8 support set, evaluated independently;
measurable original C3-C5 cases still scored; pooling as 6 of 16
forbidden), a pin of `cuban-control-support-selection.json` (sha
`fdd3e3e8...`) and the T8 set role. A self-check asserts that every other
field equals v1. **v2 sha256
`9824b0799737960363c95bb028332fdd0e5118e748fc0139375f1d0e230099a0`.**

**Frozen session-grouped folds**
(`training/diagnostics/cuban-step1-folds.json`, sha `76b55d91...`).
The rule, fixed in `cuban_step1_prepare.py` before it ran, is: whole
sessions only; minimise the fold-size range, then the squared Auto-WB
fraction deviation, then lexical order. Result: fold 0 = sessions {1, 7,
12} (8 scenes), fold 1 = {3, 13} (8), fold 2 = {8} (4), fold 3 = {11}
(4). With session sizes 4/4/4/4/4/3/1, a range of 4 is the minimum
possible without splitting a session. White Fluorescent and Cloudy occur
only in session 3 and cannot be stratified.

**Frozen run manifest**
(`training/diagnostics/cuban-step1-run-manifest.json`, sha `27dc5918...`).
It binds contract v1/v2, the cohort proposal (calibration-input copy
byte-equal to the committed copy), the base-step1 manifest roles, the
folds, and the source checkpoint `r11-drcond-seed42-best.pt` (sha
`1cfd2682...`, read-only, re-hashed before and after every run). It also
lists the F allowlist (24 scenes with `.fjlrg` and E4 hashes; E4 hashes
cross-checked against the passed pilot/cohort1/base-step1 reports) and K
(8, evaluation only). The remaining settings are as approved:

- trainable `grid.head.*` and `refinement.*` (17,655 of 115,719);
  `grid.encoder.*` frozen
- per-scene linear L1, batch 1
- Adam (0.9, 0.999), fresh state
- cosine T_max = 20, eta_min 0.01*lr
- LR {1e-4, 3e-4}, epochs 1-20, CV seed 42
- selection by mean out-of-fold per-scene L1 over all 24 F; tie-break
  fewer epochs, then smaller LR
- epoch 0 (warm start) reported but not selectable
- final seeds 42/43/44 on all F, stopped at the chosen epoch without
  rescaling the schedule

The seeds differ only in data order, because all three start from the
same copied weights.

**Code.**
- `training/run_cuban_step1.py` (smoke/cv/select/final/freeze) refuses on
  any fingerprint mismatch. Its loader accepts only F-allowlist names and
  re-hashes every file. It asserts that the frozen encoder tensors are
  unchanged after each run. Checkpoints are written as `cn-step1-seed*.pt`
  only.
- `training/eval_cuban_step1_k.py` is the single K evaluation, written now
  before any result exists. It refuses unless
  `cuban-step1-freeze.json` exists and every frozen hash matches, and it
  refuses to run twice. Seed 42 decides; 43/44 are report-only.
- The K-evaluation criteria are the E4 per-scene limits, >= 7/8, and the
  E4 aggregates, as in the run manifest. The untouched Round 11 seed 42
  output is a context baseline. The seed 42 + app-LUT baseline is **not
  run**: it would need a Python reimplementation of the app's LUT pipeline,
  which is out of scope for this step.

**F-only smoke test.** It used 3 F train scenes and 1 F validation scene,
2 epochs, output to scratch only. The allowlist refused a K name and an
unknown name. The encoder was unchanged, and the source hash was
unchanged after the run. About 0.5 s per scene per epoch on MPS.
DSCF0882's high L1 (0.12) reflects scene brightness (mean linear 0.74),
not misalignment: the E4 vs Provia target correlation is 0.99. The
metric functions were checked on synthetic data (perfect app -> rho 0;
no change -> rho 1).

No K, monitor or finalHeldOut pixels were read. T8 is untouched. No
cloud, no app change and no extra exports.

### Step-1 CV, final fits and the single K evaluation: FAILED (2026-11-19)

The approved sequence ran exactly as frozen in 75f7a71. No integrity
check failed, and the source checkpoint `r11-drcond-seed42-best.pt` is
unchanged (`1cfd2682...`) after every run.

**CV and selection** (`training/runs/cn-step1-cv-*.json`,
`training/diagnostics/cuban-step1-selection.json`). Eight runs: 2 LR x
4 session folds x 20 epochs, seed 42. The predeclared rule chose
**LR 3e-4, epoch 16**, with mean OOF per-scene L1 0.03259. The warm
start (epoch 0, not selectable) scored 0.04594; the LR 1e-4 best was
0.03356 at epoch 20. Session 11 (fold 3) remained the weakest fold
(about 0.061).

**Final fits** (`training/runs/cn-step1-final-seed{42,43,44}.json`).
All 24 F scenes, LR 3e-4, stopped at epoch 16 with T_max 20. The
checkpoints were frozen in `training/diagnostics/cuban-step1-freeze.json`
(seed 42 `27a87df5...`, 43 `9cd5fd16...`, 44 `6bda9e43...`) and pushed
as b0f04c2 **before** K was read.

**Single K evaluation** (`training/diagnostics/cuban-step1-k-eval.json`,
contract v2, E4-only internal simulation-adaptation check). **Step 1
fails.** Seed 42 passes **0/8** K scenes against a required 7, and fails
all four aggregates:

| | median mean ΔE00 (≤1.5) | mean mid ΔL* (≤1.0) | mean ΔC* (≤1.5) | mean hue (≤3°) |
|---|---|---|---|---|
| Round 11 seed 42 untouched (context) | 5.95 | −2.49 | −0.30 | −5.6 |
| **step 1 seed 42 (decision)** | **5.14** | −1.35 | −2.16 | −9.1 |
| seed 43 (report-only) | 5.41 | −1.43 | −2.14 | −9.5 |
| seed 44 (report-only) | 5.25 | −1.51 | −1.92 | −9.0 |

Seed 42 per scene. "Fuji P→E4" is Fuji's own Provia-to-E4 difference.
"R11 mean" is the untouched Round 11 seed 42 output against E4.

| scene | Fuji P→E4 | R11 mean | mean | p95 | mid ΔL* | rho | failed |
|---|---|---|---|---|---|---|---|
| DSCF0415 | 4.78 | 5.91 | 5.33 | 11.77 | +0.93 | 0.78 | mean, p95, rho |
| DSCF0515 | 5.30 | 8.05 | 7.83 | 23.13 | −4.99 | 1.01 | all four |
| DSCF0690 | 5.58 | 6.00 | 3.94 | 6.47 | −9.01 | 0.83 | all four |
| DSCF0740 | 3.04 | 4.73 | 4.11 | 8.47 | +0.55 | 1.10 | mean, p95, rho |
| DSCF0754 | 5.22 | 5.75 | 4.94 | 10.53 | −0.25 | 0.77 | mean, p95, rho |
| DSCF0888 | 3.58 | 5.38 | 3.67 | 10.05 | +0.77 | 0.95 | mean, p95, rho |
| DSCF1029 | 5.62 | 7.78 | 6.94 | 10.41 | +0.40 | 0.65 | mean, p95, rho |
| DSCF1061 | 3.75 | 6.15 | 5.34 | 10.61 | +0.82 | 0.83 | mean, p95, rho |

**Observations (descriptive, not tuning).**
- Step 1 lowered the error on every K scene relative to the untouched
  render (median 5.95 → 5.14), but every rho is ≥ 0.65 against the
  required 0.5: most of Fuji's Provia→E4 change is not reproduced.
- For 7 of 8 scenes, the untouched Round 11 output is further from E4
  than Fuji's own Provia is. This is consistent with the inherited
  Round 11 render error on these scenes being of the same order as the
  whole Classic Negative effect; that inherited error was not measured
  here.
- The step-1 model also shifted mean hue further negative (−5.6° →
  −9.1°) and pulled chroma down (ΔC* −0.3 → −2.2).
- DSCF0515 (White Fluorescent, ISO 6400) and DSCF0690 (−9 mid ΔL*) are
  the largest failures.
- Seeds 43 and 44 agree with seed 42, so the result is not a seed
  artifact.

These are hypotheses for Codex's review, not a selected next change. Per
instruction, no further tuning, re-runs or alternative selection
followed the K result; the K evaluation refuses to run twice. Round 11
seed 42 stays frozen and selected. The `cn-step1-*` checkpoints are
retained but not selected and not integrated into the app. Monitor,
finalHeldOut and T8 were not accessed. No cloud spend, no extra exports.
Visual review was not performed.

### Codex review after Step-1 failure: F-only input audit (2026-11-19)

**Decision:** keep the Step-1 checkpoints unselected and Round 11 seed 42
unchanged. The next isolated step is a baseline-versus-simulation-error
diagnosis on **F24 only**, before another fit. No architecture, loss,
contract, export settings or thresholds are changed by this decision.
K's existing result stands. Monitor, finalHeldOut and T8 remain unread.

The candidate's export is not the current blocker. The bounded overnight
gate completed against the frozen Step-1 seed-42 checkpoint, with native
ORT, browser WASM and real WebGPU passing the existing numerical limits
(maximum browser absolute difference 9.54e-7). This is numerical parity,
not Fuji accuracy or permission to integrate it into the app. The report
is saved as `training/diagnostics/cn-step1-export-gate.json`; the vectors,
ONNX and full native report remain in the ignored
`training/checkpoints/codex-cn-overnight-20261119/parity/`.

**New measurement: source-WB context, F metadata only.**
`scripts/inspect-cn-wb-context.mjs --all-fit` opened the fixed F24 RAFs
with the canonical LibRaw options and read `metadata(true)` only. It
never called `imageData()` or `rawImageData()`. The loopback server can
serve only the explicit F allowlist and the installed LibRaw assets.
`training/audit_cn_step1_wb_context.py` joined that result to the source
preview MakerNotes and the existing Provia and E4 TIFF metadata:

- All 24 Provia references have the expected neutral Provia settings.
- Provia and E4 match in WB mode, Kelvin, zero fine-tune shift, and
  normalized recorded R/G and B/G WB levels on all 24 scenes.
- All 24 source RAFs have nonzero WB fine-tune shifts. LibRaw's `cam_mul`
  R/G/B values exactly equal the source preview's recorded WB levels
  on all 24, and their normalized ratios differ from the zero-shift
  reference ratios on all 24.
- Example DSCF0964: source shift R+5/B-6, source ratios R/G 1.9834,
  B/G 1.5861; both zero-shift references record 1.7517 and 1.9073.

Evidence is in `training/diagnostics/cn-step1-wb-input-probe-all-F.json`
and `cn-step1-wb-reference-audit.json`; the initial three-scene probe is
retained separately. Probe records contain source RAF hashes; the joined
audit records its probe/run hashes and the frozen input fingerprints.
Interpreting the MakerNote levels as gains is inferred, with agreement
to actual LibRaw metadata measured here. This audit does **not** measure
how much WB context contributes to the model's error, prove a decoder
bug, or establish that adding WB planes would fix it. Target-derived
WB ratios must never become runtime input features. The existing model
may already infer some WB context from pixels, and the frozen encoder
or small adaptation corpus may also limit the learned simulation delta.

**Next diagnostic, owned by the training side (no new fit).** On F24,
using the frozen Round 11 and Step-1 seed-42 checkpoints, the same
working-scale TIFF loaders and the frozen Lab/DeltaE implementation:

1. Measure Round 11 versus its actual Fuji **Provia** reference. The
   existing Round 11-versus-E4 numbers do not measure Provia accuracy.
2. Measure Fuji Provia-to-E4, and Step-1-versus-E4, on the same pixels.
3. Form the learned Lab delta `D = CN_prediction - R11_prediction` and
   required Fuji delta `D_ref = E4_reference - Provia_reference`.
   Report delta-vector error and the diagnostic oracle-anchored render
   `Provia_reference + D` against E4. Lab vector errors decompose exactly
   as `CN_prediction - E4 = (R11_prediction - Provia) + (D - D_ref)`;
   scalar DeltaE values do not add or subtract that way. Report any
   cancellation between components rather than calling one component
   a floor from its magnitude alone.
4. Report frame and tone-band signed residuals, mean/p95 DeltaE and
   the above components by the already-recorded source shift/WB gains.
   These are descriptive, potentially confounded groups, not a causal
   test. Inspect fixed F examples DSCF0523, DSCF0964, DSCF0514 and
   DSCF0511 for visible artifacts using input, both predictions and
   both Fuji references. No K visual inspection is requested here.

Fingerprint the sources, references, checkpoints and contract in a new
diagnostic report; preserve all existing CV/K evidence. The final fits
have seen F, so their scores are **in-sample diagnostics**, never a new
validation result. If making an OOF claim, use the corresponding stored
session-held-out fold checkpoint for each scene and label the distinction.
No LR/epoch search, WB adjustment, decoder change or replacement reference
is authorized as part of this diagnostic. Return this decomposition before
selecting a model intervention: source-WB conditioning is a candidate only
if the measurements support it. No more user exports are needed for this
step. Reusing K after it informs a change would be a development regression
check, not fresh independent acceptance; the locked finalHeldOut is not
substituted for that work.

### F24 decomposition: inherited Round 11 error vs learned Provia→CN delta (2026-11-19)

Scope as set in df2b052. The run is F24 only and **in-sample**: Round 11
trained on these Provia references, and step 1 trained on these E4
references. It involved no fitting and no K, monitor, finalHeldOut or T8
reads, and made no app change. Round 11, contracts v1/v2 and all CV/K
evidence are untouched. The CV fold checkpoints were never stored, so
**no out-of-fold pixel decomposition exists**. Only the per-scene OOF
linear L1 from the CV logs is attached, labelled as context.

Script: `training/diagnose_cn_step1_f24.py`. Report:
`training/diagnostics/cn-step1-f24-decomposition.json`. Fingerprinted:
contract v1/v2, run manifest, folds, freeze, selection, WB audit, both
checkpoints, and every `.fjlrg`/E4/Provia hash. Visual panels:
`training/diagnostics/cn-step1-f24-visual/` for DSCF0523, 0964, 0514 and
0511 (input, R11, Fuji Provia, step 1, Fuji E4, plus ΔE heatmaps). Lab
identity `CN − E4 = (R11 − Provia) + (D − D_ref)` holds to 1.4e-14.

**Medians over F24 (mean ΔE00 per scene):**

| comparison | median |
|---|---|
| (1) R11 vs its own Fuji Provia (in-sample) | **4.94** |
| (2) Fuji Provia vs E4 (the whole CN effect) | 4.25 |
| (2) step 1 vs E4 (in-sample) | 4.81 |
| (3) oracle-anchored Provia + D vs E4 | **3.61** (in-sample rho 0.81) |

**Delta and energy split:**
- D recovers about 54% of D_ref along its direction (median projection
  gain 0.54), and the median weighted cosine is 0.63.
- Mean squared Lab error: total 64.3 = base 79.7 + delta 38.6 + cross
  term −54.0. The cross term equals −40% of the two parts, so the two
  components cancel strongly. Step 1's D partly undoes R11's base error
  rather than acting as a pure Classic Negative delta.
- About 21% of the base error energy is a frame-wide colour offset
  (median; maximum 69%). Most of it varies spatially.

**Reading.**
- Both components are independently too large for the E4 limits, even
  in-sample:
  - The inherited R11 baseline error alone (4.94) exceeds the whole Fuji
    Classic Negative change and is 2.5x the 2.0 E4 mean limit.
  - Even on a perfect Provia base, the learned delta would score 3.61
    with rho 0.81, against limits of 2.0 and 0.5.
- Fixing only one component cannot pass step 1.

**Source WB (descriptive, n = 24, confounded).**
- R11's frame-mean chroma bias tracks the source WB fine-tune shift
  strongly. Spearman ρ against the R shift and B shift: e_base Δa* +0.82
  and −0.85; Δb* +0.76 and −0.78. Against the source/reference log gain
  ratios: |ρ| 0.72–0.82.
- The sign flips by group. Auto scenes (shift +5/−6 and +4/−5) render
  too warm, with Δa*/Δb* around +1.7/+3.0 for +5/−6. Kelvin scenes
  (+1/−3, n = 11) render too cool, around −1.6/−3.2.
- e_delta correlates the opposite way (|ρ| 0.49–0.66), consistent with
  step 1 compensating part of that bias.
- Scene ΔE00 itself correlates only weakly to moderately (|ρ| ≤ 0.48),
  as the 21% offset share would suggest.
- **Confound:** in F24, the shift is fully determined by the WB mode
  (every Auto scene is +5/−6 or +4/−5, every Kelvin scene is +1/−3;
  Cloudy and White Fluorescent are single scenes), and modes cluster by
  session. These data cannot separate the shift from the WB mode,
  session or scene content.
- This supports source WB context as a *candidate* for part of the
  base-error colour bias (about 21% of base energy). It does not show
  that WB conditioning would close the gap. Target-derived gains were
  used only as diagnostic groupings.

**Visual review (F examples, working scale).** No banding, blotches or
halo artifacts are visible in either prediction.
- DSCF0514 (neon, White Fluorescent) is a large colour failure. R11
  renders the magenta and green neon as lavender and mint, and step 1 is
  paler still. Fuji Provia keeps saturated magenta/green, and Fuji E4
  renders pink/teal. Highlight-band base error is Δa* +14, Δb* −19.
- DSCF0964 and DSCF0523: overall colour is close. Error concentrates on
  bright foliage and sky edges and on the light sources.
- DSCF0511 (Cloudy interior): R11 is slightly less warm than Provia, and
  step 1 is pinker than E4's orange.

No intervention is selected. Per instruction, this returns the
decomposition for Codex's decision.

### Codex decision: controlled Provia WB-context assay, F24 only (2026-11-19)

Reviewed 3458241 and all four saved F visual panels. The inherited base
error and the poor learned simulation delta are both real; the existing
candidate remains failed and unselected. Strong cancellation means the
step-1 head is not a clean simulation module. Do not train another CN head
against that moving mixture yet. The next isolated intervention is a
**source-WB-context test on Provia only**, with a matched constant-context
control. This is a small adaptation experiment, not an app/model release.

**Corrections to interpretation and statistics.** The original report
and visual evidence remain unchanged. `diagnose_cn_step1_f24.py` used
`argsort().argsort()` for ranks, which incorrectly breaks ties between
repeated WB settings. `rank_metrics.py` now assigns average tied ranks;
`audit_cn_f24_correlations.py` recalculates only the committed scalar data
and saves `cn-step1-f24-correlations-tie-corrected.json`. No images or
checkpoints were re-read for that correction. Source R/B shift vs base
frame-mean a*: +0.866 / -0.853; vs b*: +0.848 / -0.845. The qualitative
association persists, but the corrected numbers supersede the original
rank correlations. Mode/session confounding remains.

The 21% median frame-offset share is **not** an estimate of energy caused
by WB, and it is not a demonstrated upper bound on what WB metadata could
help: its effects after nonlinear processing can vary spatially. Likewise,
oracle substitution with the current fixed delta demonstrates that fixing
the base alone while preserving that delta does not pass; it does not
prove that one intervention cannot improve both components after retraining.
Do not subtract scalar DeltaE values to attribute causes.

**Training-side scope (Claude):** implement and smoke-test the following
paired assay, commit its separate manifest/code before fitting, then run
the fixed F-only experiment if the integrity/smoke checks pass. No further
hyperparameter search or final fit is included in this authorization.

- Input/targets: the existing F24 `.fjlrg` and genuine neutral **Provia**
  references only. No E4 targets or CN loss during this assay. Bind all
  input/reference/RAF/metadata hashes and reject missing, malformed or
  out-of-range fine-tune tags. Read source R/B fine-tune only, in camera
  steps (the recorded tags divided by 20), normalized by 9. Never read
  zero-shift TIFF WB gains into a feature. Keep capture-DR conditioning.
- Two arms, identical architecture/parameter count/initialization: (A)
  per-scene source R/9 and B/9 constant input planes; (B) the same two
  planes filled with the training-fold mean of those two source features
  for every train and validation image. Compute means on the training
  partition only. A zero-plane-only control would confound useful context
  with the ability to learn a global offset; use the matched-mean control.
- Both arms start from a read-only copy of frozen Round 11 seed 42,
  **not** the failed CN checkpoint. Expand only the first encoder
  convolution for the two extra input planes, with new columns zeroed.
  Keep the original RGB/DR encoder weights and all other encoder tensors
  frozen. Train the new input columns plus the existing head/refinement;
  verify old encoder tensors/columns unchanged after every run. Gradient
  masking may be used with fresh Adam state and zero weight decay, but
  report both effective trainable coefficients and whole-tensor
  `requires_grad` counts. No WB mode embedding or new gain feature family.
- Smoke checks: no-op warm start against frozen R11 for both arms within
  the existing native numerical tolerances; only approved coefficient
  slices can change; source checkpoint hash unchanged. Check metadata
  refusal paths. This input expansion is a diagnostic model, not an
  alteration of `training/model.py` defaults or the shipped decode.
- Use the existing four whole-session F folds, seed 42, per-scene linear
  RGB L1, batch 1, fresh Adam with the Step-1 betas/epsilon, LR 3e-4,
  cosine T_max 20 / eta_min 0.01*LR, and **exactly 16 epochs**. Same data
  order in each paired fold; do not select an epoch or LR from outcomes.
  Eight runs total (two arms x four folds), no new seeds or early stopping.
  Save each epoch-16 fold checkpoint, its config, feature means and hashes
  so pixel-level OOF comparisons can be reproduced this time.
- Primary diagnostic: equal-scene-weighted OOF mean DeltaE00 versus the
  genuine Provia target, paired A-minus-B; report both arms and untouched
  R11. Also report median mean and p95 DeltaE, tone-band signed Lab,
  frame-mean chroma-offset energy, every per-scene difference, and
  session/shift groups. Do not hide regressions in an aggregate. Measure
  OOF adaptation predictions, but clearly state that the R11 warm start
  previously saw these Provia scenes: this is not independent unseen-RAW
  acceptance or proof that WB shift rather than its correlated mode is
  causal. No checkpoint selection or ship/pass declaration follows from
  these diagnostic metrics.

Return that one report regardless of outcome. A benefit over the matched
control would justify scoping a broader Provia correction; it would not
prove that the remaining Classic Negative delta is solved. A negative
result ends this narrow conditioning test, not every possible WB method.
Do not rerun K, score monitor, access finalHeldOut/T8, tune on the earlier
K failures, overwrite old evidence, or request user exports for this assay.
The current contracts and selected Round 11 files stay untouched. Any
candidate proposed later for the app still needs its own numerical export
gate and the applicable accuracy evaluation.

### Paired Provia WB-context assay: implementation and fingerprints frozen before fitting (2026-11-19)

The assay implements the f707a72 scope in `training/run_cn_wb_assay.py`.
Its manifest is `training/diagnostics/cn-wb-assay-manifest.json`, sha
`9f82541c...`, pinned in the script. The manifest binds:

- contract v1/v2
- the step-1 run manifest and the frozen folds
- the source `r11-drcond-seed42-best.pt` (`1cfd2682...`)
- the WB audit and the F24 decomposition
- every `.fjlrg`, genuine Provia TIFF (cross-checked against the
  decomposition record) and source RAF hash (cross-checked against the
  WB audit)

**Features.** The per-scene source WBFineTune tag is re-read from each
RAF preview and must equal the audit. Values are tag / 20 camera steps
(int, multiple of 20, |steps| ≤ 9, else refused), then divided by 9. No
TIFF WB gain is read. The condition vector is [captureDR/2 = 1.0, R/9,
B/9].

**Arms.** A uses the per-scene features. B uses the training-fold mean
for every image in that fold. The fold means, computed on training
partitions only, are (0.250, −0.465), (0.299, −0.472), (0.244, −0.456)
and (0.250, −0.461).

**Model.** Round 11 config with 3 condition channels. `grid.encoder.0.0`
is widened from 4 to 6 input channels, with the new columns zero. A
gradient mask keeps the original 4 columns fixed; the bias and every
other encoder tensor are frozen. Effective trainable coefficients:
17,943 (288 new + 17,655 head/refinement). Whole tensors with
requires_grad: 18,519. Total parameters: 116,007.

**Fixed settings, no search.** Four frozen session folds, seed 42,
per-scene linear L1 against Provia, batch 1, fresh Adam (0.9, 0.999,
1e-8, wd 0), LR 3e-4, cosine T_max 20 / eta_min 0.01·LR, **exactly 16
epochs**, no early stopping and no selection. Eight runs, with identical
seeded data order per fold across arms, verified after the run. The
epoch-16 checkpoint of every fold, with config, fold means and hashes,
is saved as `cn-wbassay-{source,meanctl}-fold{0-3}.pt`, and logs go to
`training/runs/`.

**Smoke checks passed:**
- Malformed, non-multiple-of-20 and out-of-range tags are refused, as
  are a RAF hash mismatch and a K scene name.
- The warm start is a bit-exact no-op against frozen R11 for both arms
  (max abs difference 0.0).
- After masked Adam steps, only first-conv new columns, head and
  refinement tensors changed; the fixed slices are identical and the
  source hash is unchanged.

**Report plan.**
- Equal-scene adaptation-OOF mean ΔE00 against Provia, paired A − B,
  with both arms and untouched R11.
- Median, p95, tone-band signed Lab and frame-mean chroma-offset energy.
- Every per-scene difference, regressions against R11, and session and
  shift groups.

The R11 warm start saw these Provia scenes, so this is not
unseen-RAW acceptance and not a causal shift-vs-mode test. Nothing is
selected, shipped or passed from it. No E4/CN loss, and no K, monitor,
finalHeldOut or T8 access.

### Paired Provia WB-context assay: result, NULL (2026-11-19)

All 8 fixed runs completed (2 arms × 4 session folds, 16 epochs, LR
3e-4, seed 42). There was no integrity failure. Paired data orders were
identical across arms, the source R11 hash is unchanged, and the fixed
encoder slices were verified after every run. Report:
`training/diagnostics/cn-wb-assay-report.json`. Logs:
`training/runs/cn-wbassay-*.json`. The epoch-16 fold checkpoints are
saved (gitignored) and hash-recorded in the report.

**Primary (paired A − B, adaptation-OOF mean ΔE00 vs genuine Provia,
equal scene weight):**
- Mean −0.017, median −0.017.
- Scene-resampled bootstrap 95% interval [−0.048, +0.016]. This ignores
  session clustering, so the real uncertainty is wider.
- A better on 14 scenes, B better on 10.
- Per-scene A − B ranges from −0.150 (DSCF0882) to +0.176 (DSCF0514).

**Source-shift conditioning gives no measurable benefit over the matched
fold-mean control.** The arm effect (≈0.02) is about 2% of the per-scene
spread and an order of magnitude below the ~0.5 ΔE00 minimum Fuji effect
used elsewhere in the contract.

**Secondary.**

| model | mean / median ΔE00 | mean p95 | chroma-offset energy |
|---|---|---|---|
| untouched R11 (**in-sample**) | 5.02 / 4.94 | 11.04 | 16.2 |
| A, source | 5.89 / 5.82 | 12.03 | 24.7 |
| B, mean control | 5.90 / 5.86 | 12.05 | 24.9 |

- Tone-band signed Lab is nearly identical between arms. Both reduce the
  shadow/mid a* bias relative to R11 but raise the shadow L*.
- By shift group, A − B is +0.020 for [5, −6] (n = 7), −0.059 for
  [1, −3] (n = 11), −0.019 for [4, −5] (n = 4), and +0.176 for the
  single [−6, −4] scene. By session, every A − B lies within ±0.06.
- The learned new first-conv columns have similar L2 norms in both arms
  (A 0.098–0.137, B 0.098–0.119), against 2.52 for the original columns.
  The per-scene variation in arm A was not exploited beyond what a
  constant offset gives.

**Regressions, not hidden.** Both arms are worse than untouched R11 on
the same 20 of 24 scenes. The largest is DSCF1013 at 4.51 → 9.37 / 9.49.
Only DSCF0511, 0703, 0751 and 0753 improve. This comparison is
**unequal**: R11 trained on all 24 of these Provia references, so its
numbers are in-sample, while the arms are evaluated on scenes held out
from the 18-scene adaptation. The gap therefore reflects how badly a
short head/refinement adaptation on about 18 scenes generalises across
sessions from an in-sample warm start. It is not a WB effect, and it is
not evidence that R11 generalises at 5.0. The same mechanism probably
also affected step 1.

**Conclusion.** Within the scoped test this ends the narrow
source-shift-conditioning hypothesis: the metadata did not help the
Provia baseline over a matched constant. That does not rule out other WB
methods, does not separate shift from WB mode (they are confounded in
F24), and says nothing about Classic Negative accuracy. No checkpoint is
selected. Round 11, contracts and earlier evidence are unchanged. No K,
monitor, finalHeldOut or T8 access, no E4/CN loss, no app change, no
exports.
