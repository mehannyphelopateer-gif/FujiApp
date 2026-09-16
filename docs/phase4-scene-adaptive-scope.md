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
