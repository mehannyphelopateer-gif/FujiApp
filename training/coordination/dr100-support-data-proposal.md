# PROPOSED / UNAPPROVED: DR100 training-support data qualification and reference-data plan

Status: actual Claude's2026-11-26proposal, amended by Codex2026-11-27. Root approves ONLY the separately frozen PhaseA hash/fresh-legacy-tag check described in the addendum. No TRAINreuse, admission, copying, tierassignment, reference/pixel export, scoring, fitting, protected access or training is approved. Broader proposals below remain unapproved unless narrowed by that addendum.

The original frozen screen **stays FAIL (4/5)**. The condition-probe results, their verification and
the old sensitivity evidence stay unchanged. Nothing below re-scores or reinterprets any of them.

Inputs read for this review: `training/diagnostics/neutral-wb-condition-support-audit.json`,
`training/audit_neutral_wb_condition_support.py`, `training/neutral_wb_study_data.py`,
`training/check_dr100eval_metadata.py` (the original acceptance rules) and
`training/coordination/neutral-wb-condition-probe-proposal.md`. No media, arrays, RAFs, TIFFs,
checkpoint tensors, protected manifests or dr100Eval material were opened.

---

## 1. Review of the condition-support audit

**Headline numbers: no errors found.** 455 FIT = 33 DR200 (c=0.5) + 422 DR400 (c=1), with zero
DR100 (c=0). Over 20 epochs that is 660 vs 8440 updates per arm, and 660 + 8440 = 9100 = 455 × 20,
which matches the endpoint `global_updates` check. Name and session disjointness between FIT and
development, and zero session-14 FIT rows, are enforced by `need()` before output. The
synthetic-hook claim is correctly limited: it uses fresh random weights, uniform RGB, the actual
`encoder[0][0]` pre-hook, unchanged `D.forward`, and a weight digest that does not change. ISO and
lighting are correctly left UNKNOWN.

**Concrete issues** (none of them changes a reported number, but they matter for reuse):

1. **The `per_run.unique_condition_counts` value is not measured per run.** Line 48 copies the global
   FIT `support` counter into every run. Only `condition_update_counts` is derived per seed, and it
   comes from the manifest's planned `orders`, not from training logs. Since each order is a
   permutation of the 455 names, 660/8440 is fixed by construction. The endpoint check confirms the
   budget (epoch 20, 9100 updates), not that each update consumed the planned row. A correct label
   would be "planned per-seed order counts; budget confirmed by the endpoint meta".
2. **Several values in the output JSON are hard-coded constants.** `leakage_checks` (empty overlap
   lists, session14 count 0) and `DR100_FIT_count: 0` are literals. They are true only because
   `need()` would have aborted otherwise (lines 35, 36 and 38). That is acceptable, but a reader
   should know they are assertion outcomes, not computed fields.
3. **The DR grouping trusts one tag and would crash rather than refuse if it were missing.** Line 79
   uses `.get('AutoDynamicRange','').split()[0]`. If the tag were absent, this raises `IndexError`
   instead of a clean `need()` refusal. It also selects on `AutoDynamicRange` alone. It does not
   re-derive the original exclusion reasons (absent `DevelopmentDynamicRange`, DR-priority tags,
   `ExposureCount`, model), so the audit records *which* DR each frame resolved to, but not *why*
   it was excluded.
4. **The time range spans all 18 frames, not the nine DR100 candidates.** The 19:12–23:31 span
   (4 h 19 min) also exceeds the 3 h session-gap rule's window, so the nine could form one session
   or two, depending on the gaps between frames. That is not computed anywhere yet.
5. **Disjointness was checked by file name against the current 571 only.** Fuji `DSCFnnnn` counters
   repeat across cards and resets, so name disjointness is not identity disjointness. There is no
   check by RAF SHA-256 against the 571 `source_raf_sha256` values, and no check (by name, hash or
   time) against any protected tier. Most importantly, nothing checks against dr100Eval, whose
   accepted frames start at DSCF1443, immediately after these candidates in the counter sequence
   (see §4.4).
6. **The stored `exif.json` is not tied to the RAF bytes.** `metadata_json_sha256` pins the dump, not
   the originals. **Hazard:** `check_dr100eval_metadata.py --dir <excluded folder>` writes
   `exif.json` and `paths.json` *into that folder*, overwriting the audited dump and invalidating
   `0898a25b…437a`. Nobody should re-run it on that folder. A new qualification must write to a new
   write-once path (§4.1).

## 2. Why the original exclusion stays valid while TRAIN reuse is a separate question

The 18 frames were rejected by the dr100Eval **acceptance protocol** (`exclusion_reasons`). That
protocol required a deliberately **fixed** DR100 (`DevelopmentDynamicRange == 100`, no
`AutoDynamicRange`), DR priority off, a single exposure, at least 3 sessions under the >3 h rule, and
at least 4 low-ISO plus at least 4 high-ISO frames across sessions. These frames were shot on DR
AUTO in one outing, so they fail it. **That verdict is correct and stays in force.** They are not
dr100Eval frames, not acceptance evidence, and must never be scored as if they were.

TRAIN support asks a different question: does a frame carry a **truthful condition label** and a
**correct target** under the current domain contract? A DR-AUTO frame that resolved to 100 might
meet that bar. Whether it does is exactly what §4 has to show. Two rules follow from this:

- **No automatic reinterpretation.** Passing §4 would not turn "excluded from acceptance" into
  "accepted". The frames would get a new, separately named status (for example
  `qualified_train_candidate_autoDR100`), and their old `EXCLUDED` status would be recorded
  alongside it, not replaced.
- **No automatic reuse.** Even after §4 and §5 pass, using the frames in any fit set needs its own
  explicit approval and a new study manifest (§6).

## 3. Candidate allowlist (to freeze before any byte is read)

Exactly nine names, held in a new write-once allowlist file with its own SHA:

`DSCF1424, DSCF1426, DSCF1428, DSCF1430, DSCF1432, DSCF1434, DSCF1436, DSCF1438, DSCF1440` (`.RAF`,
under `calibration-input/phase4-dr100eval-attempt1-excluded/`).

The nine odd-numbered DR200/ISO1600 frames (`DSCF1425 … DSCF1441`) are **out of scope** and must be
refused by name. The allowlist also pins the stored-dump SHA `0898a25b…437a`, so a changed dump
refuses.

## 4. Required pre-pixel qualification (metadata and source hash only)

Every step is refuse-on-failure: no imputation, no fallback, no "closest" value. Steps 4.1–4.3
can run locally. Step 4.4 needs a decision on the protected boundary (§9).

### 4.1 Source identity and fresh metadata
- Hash each of the nine RAFs (SHA-256 of the bytes; no decode).
- Re-extract tags from the originals with the same `dump-raf-exif.mjs` into a **new** write-once
  file, never the old `exif.json`. Require field-for-field equality with the stored dump on: model,
  `dateTimeOriginal`, ISO, exposure, aperture, bias, `WhiteBalance`, `AutoDynamicRange`,
  `DevelopmentDynamicRange`, DR-priority tags and `ExposureCount`. Any mismatch refuses that frame
  and is reported. The stored dump is never silently corrected.
- Record, from the fresh extract only: model == X100VI, `ExposureCount` ∈ {absent, 1}, DR-priority
  off.

### 4.2 Capture-DR resolver, labelled truthfully
- Require `AutoDynamicRange` = 100 and `DevelopmentDynamicRange` absent. Record
  `capture_dr_resolver: "auto"`, `capture_dr_resolved: 100`. Never write `fixed` or relabel these
  frames as DR100-fixed.
- If they are ever used, the condition code would be `log2(100/100)/2 = 0.0` *under the assumption*
  that the resolved DR is the one the network should condition on. This is a **labelling decision
  for root**, not a fact. Record it as an assumption, not a qualification result.
- Caveat to record: the frames are ISO 125 (×8) and ISO 160 (×1). On X-Trans bodies, DR200/400
  normally need ISO at least 2×/4× base, so DR AUTO at these ISOs most likely *could only* resolve to
  100. Check this against the X100VI manual before stating it. If it holds, DR100 here is
  inseparable from low ISO.

### 4.3 White balance: resolve, do not guess
- Stored `WhiteBalance` is `0x1` (Auto, white priority) on 7 frames and `0xff0` (Kelvin) on 2.
- **Kelvin frames:** the Kelvin value itself must be read from a tag in the fresh extract. If no
  tag yields it unambiguously, refuse the frame. The value is never inferred from pixels or
  neighbouring frames.
- **All nine:** read the WB fine-tune (red/blue shift) tags. If the tag is absent or ambiguous, HOLD the frame. For a known shift, use the existing qualified source-only policy unchanged: nonzero shift is not automatically invalid because qualified W/P policies already neutralise specific source families. Unsupported families remain HOLD/no fallback. A target0shift does not require a source0shift when an already-qualified policy handles it.
- **Neutral-input policy:** map each frame's resolved `source_wb_mode` to a neutral kind using the
  **existing, already-qualified rule**, unchanged (the same mapping that produced `neutral_kind`
  G/W/P for the current 571). Any mode that rule does not cover refuses. No new rule is written for
  these nine.

### 4.4 Disjointness, time and session
- **Against the current 571:** RAF SHA-256 disjoint from every `source_raf_sha256`, and names
  disjoint (already shown). Using the 571 rows' capture times, assign a session with the same >3 h
  rule. Two outcomes refuse: a candidate falling inside any **development** session (7, 12 or 14),
  because adding it to FIT would leak a session; or one falling inside an existing FIT session,
  which would need a decision because it would no longer be "new session" support.
- **Within the nine:** compute their own session split under the >3 h rule and report it.
- **Against protected tiers (K, monitor, finalHeldOut, T8, dr100Eval):** this is the boundary
  problem. dr100Eval's accepted frames (DSCF1443+) are next in the counter sequence and may have
  been shot the same evening. If any of them falls within 3 h of a candidate, using the candidates
  in TRAIN would put a protected-eval session into FIT. **No reading of dr100Eval or its manifest
  is proposed here.** Root must choose one of:
  (a) a separately approved guard, sealed so it can only output two booleans per protected tier,
  "SHA-disjoint" and "min time gap > 3 h", and nothing else; or
  (b) the user states the dr100Eval capture dates from memory or notes; or
  (c) the candidates are refused outright because adjacency cannot be ruled out.
  Until one of these is done, the candidates stay **not qualifiable**.

**Output of §4:** one new write-once receipt with per-frame pass/refuse, reasons, hashes and the
fresh-extract SHA. No tier and no role are assigned.

## 5. Reference data needed later (requires a camera/user export; not now)

Only for frames that pass §4, and only after root explicitly approves a TRAIN-support purpose:

- **Targets:** genuine 16-bit TIFF, Provia, WB **as shot** (the Auto result as recorded, or the
  recorded Kelvin), WB shift **0**, **output DR100**. All other parameters must match the existing
  contract behind `provia-color-target-cache`, read from its own record and applied unchanged. The
  converter, version and settings are recorded per frame. Each TIFF must go through the same target
  check the current targets passed (bit depth, dimensions vs the source, no 8-bit or JPEG route).
  If a TIFF does not meet the contract, the frame is refused, never adjusted after export.
- **Inputs:** the **current true-linear Gv2 contract** only (`fjlrg_v2` as-shot for G; the existing
  qualified W/P contracts for neutral, chosen by the §4.3 rule), with source-RAF and WASM SHA
  bindings as in `Pairs.get`. **Never** reuse the historical `export-phase3-linear` domain, which was
  gamma-mislabelled, and never weaken or bypass any guard written because of it (contract checks,
  DR-provenance check, unit-range check). Neutral inputs need the same control-matches-G and
  neutral-repeat-exact qualification the 571 passed.
- **Until a purpose is approved:** no export, no target cache entry, no `.npy`, no metric on these
  frames.

## 6. Limits of the existing nine (stated before anyone hopes otherwise)

- **One outing**, at most one or two new sessions. Session confounding remains, as with session 14.
  Scene count is not session count.
- **Low ISO only** (125/160). These nine provide no high-ISO DR100 support. Whether DR AUTO can choose100 at other ISO values is not established here. Separately shooting fixedDR100 avoids Auto-selection ambiguity for future high-ISO support.
- **Small:** nine rows would add 9 × 20 = 180 updates per arm, against 8440 at c=1.
- **Not independent acceptance.** The frames are internal TRAIN candidates at best and say nothing
  about Fuji parity.
- **Auto-resolved, not fixed, DR100.** Whether it matches fixed DR100 at capture is an assumption
  (§4.2).
- **The DR200 partners are interleaved.** The odd-numbered frames look like alternating re-shots of
  the same scenes at ISO 1600. They stay out of scope, but any later use of them would create
  near-duplicate content across conditions.
- **No promise.** Nothing here predicts that adding these frames would fix the DR100 identity-37
  harm. The probe established only sensitivity, not cause.

## 7. Future inference canaries, reproductions and provenance boundaries

- **Frozen artefacts stay frozen.** Using any candidate would mean a **new** study manifest with a new
  SHA. The old manifest `d416942f…8d0`, its six endpoints, the report, its verification, the probe
  and R11 `1cfd2682…291c` are never edited, rebound or re-scored. No result from a new study may be
  compared against the +0.10 identity margin as if it rescued the old screen.
- **Canaries.** A new study needs its own independently selected canaries (salted metadata-hash
  selection frozen before the run), its own reproduction-first gate, and the unchanged numeric limits
  (1e-5 / 1e-4 / 1e-7). Canaries must include at least one of the new DR100 rows. The old
  shared-forward limitation (canaries reproduce numerically through the same `D.forward`) also
  applies to them and must be stated.
- **Provenance chain per frame:** RAF SHA → fresh-extract SHA → §4 receipt SHA → (later) TIFF SHA
  plus converter record → input SHA plus sidecar → manifest row. The stored `exif.json` is
  historical context only. It is never the authority for a label.
- **Boundaries.** The nine never enter development or any protected tier. dr100Eval is never read
  except through the guard root approves in §4.4. The untracked
  `training/runs/neutral-wb-study-progress.json` is not consumed.

## 8. Handoff

**Can be checked locally first (metadata and hashes only, after root approves §4):**
1. Freeze the nine-name allowlist and the stored-dump SHA.
2. Hash the RAFs and run a fresh tag extract to a new path. Compare it with the stored dump.
3. Check the resolver tags, model, exposure count and DR priority. Resolve Kelvin and WB fine-tune
   from tags, or refuse.
4. Check SHA, name and session disjointness against the 571 and against development sessions 7, 12
   and 14. Compute the session split within the nine.
5. Write one §4 receipt. Do not touch the old `exif.json`, and do not re-run
   `check_dr100eval_metadata.py --dir` there.

**Needs the user or a camera export later:**
- Confirmation of the dr100Eval capture dates, or the root-approved boolean guard (§4.4).
- 16-bit Provia / as-shot / 0-shift / output-DR100 TIFFs for frames that pass, using the existing
  target contract.
- Any genuine high-ISO DR100 support, which needs new fixed-DR100 shooting in multiple sessions.

**Decisions that remain with root:**
1. Approve, amend or reject §4 as a metadata-only qualification.
2. Choose how to handle the protected-boundary check: (a) a sealed boolean guard, (b) the user's
   statement, or (c) refuse.
3. Whether an auto-resolved DR100 is labelled c=0 or held out of conditioning.
4. Whether to refuse or hold frames with a non-zero or unknown WB fine-tune, or a Kelvin value
   that can't be resolved.
5. Whether a TRAIN-support purpose is approved at all. Until it is, there is no export, no scoring
   and no manifest.
6. Whether a candidate falling inside an existing FIT session is acceptable or refused.


## Root approval: Phase A only (2026-11-27)

The user's continuing local-research instruction authorises this reversible
read-only check. Freeze/commit the exact nine paths, olddumpSHA, existing
reader/helper code and new host/tests before any candidate bytes are read.
Then hash nine originals, re-extract legacy tags with the unchanged
dump-raf-exif.mjs into a NEWexclusive scratch directory, compare every
fresh record with the historical dump, hash originals again, check aliasing
against all583nonprotected parent sources, and compute sessions within
these nine from actual fresh timestamps. Persist the fresh extract and
receipt exclusively under new training/diagnostics names. Never invoke
check_dr100eval_metadata.py --dir on the old folder; original dumps remain
byte-identical. No source is assigned a new role or tier.

PhaseA cannot claim full qualification: WBfineTune/Kelvin/source-policy
resolution, complete current/protected-boundary checks and TIFF/Gv2input
qualification remain HOLD. The receipt must keep training_admission=false
and preserve all original acceptance exclusions regardless of success.
A sourcehash records current identity; historicaldump did not pin original
RAFbytes, so unchanged-since-original-dump is not established.

Existing public-in-repo notes already record the accepted dr100Eval
camera-time interval as2026-11-17 15:57–18:22 (scope doc§attempt2). This is
more than16hours after candidate olddump ends2026-11-16 23:31. Reading
that DOCSrecord does not open protected media/manifests; it reduces the
adjacency concern without a new protected-data query. It is not claimed
as a fresh protected-SHA comparison or complete admission clearance.

Corrections before further work: per-run support in the old audit means
planned-order counts plus endpoint budget confirmation, not independent
per-update tracing. Literal empty leakage/zeroDR100fields are successful
assertion outcomes; original code/evidence stay frozen. A missing tag
would abort the old stored-dump grouping; new fresh extracts must refuse
cleanly. Existing source-only WBpolicy, not a new blanket zero-shift rule,
governs any later policy classification. No claim that DR-AUTO cannot
ever resolve100at high ISO is made.

Actual Claude is implementing a separately bounded supplementary metadata
reader and synthetic fixtures only; no call on real sources is authorised
for that new reader until a new reviewed scope pins its code and PhaseA
source hashes. Root owns/executed PhaseA checks. No new training is running.
