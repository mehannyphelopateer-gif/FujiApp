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
