# Reviewed design: inference-only condition-response probe on six frozen G/neutral endpoints

Status (2026-11-26): drafted by actual Claude, reviewed and amended by Codex under the user's standing local-research instruction. Fixed three-point design approved for implementation. Codex completed the runner and separate verifier after Claude's implementation attempt returned a usage-limit error; no Claude implementation/review is claimed. All17synthetic checks pass. The executable JSON scope and code must be committed BEFORE any forward pass. Training, model selection, app changes and protected-tier access remain prohibited.

The original frozen screen **remains FAIL (4/5)**. This probe cannot rescue, re-score, re-gate or
re-interpret that result. No dropping DR100, no selecting seed 43, no "corrected" screen.

---

## 0. Corrections to earlier statements (recorded first, so they are not repeated)

1. **Seed 44 passed the identity-37 margin DESPITE DR100, not because of it.** From
   `training/diagnostics/neutral-wb-dr100-seed-spread.json`: seed 44 DR100-12 contribution to the
   identity-37 mean was **+0.429** (12/12 scenes worse), offset by DR200-25 **−0.393**, giving
   **+0.037 ≤ +0.10**. DR100 hurt seed 44 too. It just hurt less than in seed 42 (+2.046 contribution,
   total +1.106).
2. **Three observed seeds are not a noise distribution.** The observed seed ranges (DR100 mean of
   per-scene ranges: G 4.25, neutral 8.46 ΔE00) are three draws. They are not a variance estimate, a CI
   or a significance basis. They cannot show that two arms are "equivalent within noise", and they
   cannot tell an arm effect from seed sensitivity.

## 1. Established facts this probe starts from (not re-derived)

- All six fresh endpoints (seeds 42/43/44 × G/neutral) completed. Report SHA
  `3e7e2194…a6a4`, independent verification `e5b7cdbf…f059` (`verified: true`, canaries exact).
- Pooled development 7.845 → 6.343. Treated-79 gain holds on every seed. The screen fails only on
  criterion 5: identity-37 on seed 42 is +1.106.
- Identity-37 = 25 DR200 (c=0.5) + 12 DR100 (c=0). All 12 DR100 scenes are from **one session (14)**,
  and fit has no DR100 (c=0) row at all. Fit support for the condition is c=1 (DR400, 422 rows) and
  c=0.5 (DR200, 33 rows: W32 + G1).
- DR200-25 improves on all seeds. DR100-12 is 12/12 worse on seeds 42 and 44 and 12/12 better on
  seed 43. Per-seed DR100 mean deltas are +6.309 / −3.644 / +1.324.
- Identity inputs are byte-identical across arms (`assemble_rows`: `n = dict(g)`). Identity deltas
  therefore come from weight differences, which come from the changed fit inputs.
- R11, app and protected tiers are unchanged. No training is running.

## 2. Question (sensitivity, not cause)

**Primary (descriptive sensitivity):** Hold the endpoint weights, input frame and target fixed and vary
only the scalar condition fed to the network. How does each frozen endpoint's development error on the
37 identity sources change, and how does the neutral−G arm gap change with it?

The one thing that differs between conditions is the condition scalar, so within an endpoint this
probe identifies the **trained function's response to its condition input**. That is a legitimate
counterfactual about the network's input-output map, and the only causal statement it can support.

**It does NOT identify:**
- why training produced that response. The arms differ on454of455fit inputs, and the training trajectory
  is not manipulated here;
- whether DR100 harm is "caused by extrapolation". Section 6 explains why a lower error at a wrong
  condition is not evidence of that;
- any production fix, relabelling rule or condition remap.

## 3. Scope (proposed for freeze)

| Item | Proposed value |
|---|---|
| Endpoints | all six: {42,43,44} × {G, neutral}, equally weighted, none dropped or selected |
| Sources | exactly the 37 development rows with `policy_subgroup == "identity"` and `neutral_kind == "G"` (25 DR200 + 12 DR100). No fit rows, no treated rows, nothing protected |
| Protected | K, monitor, finalHeldOut, T8, dr100Eval: never opened, never listed, never path-resolved |
| Frames / targets | unchanged: frozen manifest paths, SHA-checked by the unchanged `neutral_wb_study_data.Pairs.get` |
| Weights | the frozen epoch-20 `endpoint.pt` files, validated by the unchanged `run_neutral_wb_study.validate_endpoint` |
| Device | CPU only (as in the original eval). Thread count fixed and recorded |
| Metric | unchanged `provia_color_pilot.dev_metrics` (mean_dE00, p95_dE00, linear_l1_unclamped), plus the predeclared signed-brightness measures in §5.3 (new, labelled supplementary) |
| Condition grid | see §4. Recommended primary grid: **{0, 0.5, 1}** |
| Outputs | scalars only. No prediction arrays, images, tensors or checkpoints written |

The actual conditions are **0.0 for DR100** and **0.5 for DR200**. Both are exactly representable in
float32 and must equal `log2(DR/100)/2` bit-for-bit.

## 4. Grid choice: five points (1110 cases) vs three points (666)

Counts (sources × conditions × endpoints):

| Grid | Total forwards | Actual-condition (reproduction) | Counterfactual |
|---|---|---|---|
| {0, .25, .5, .75, 1} | 37×5×6 = **1110** | 37×6 = 222 (12×6 at c=0, 25×6 at c=.5) | 888 |
| {0, .5, 1} | 37×3×6 = **666** | 222 | 444 |

Both grids contain both actual values, so every source has exactly one reproduction cell per endpoint.

For scale, the original six full evaluations were 6 × 571 = 3426 CPU forwards. 1110 forwards ≈ 0.32× of
that, and 666 ≈ 0.19×. **Cost does not decide this.** Interpretability does.

**Recommendation: predeclare only {0, 0.5, 1}.** Reasons:
- 0, 0.5 and 1 encode the three capture DR modes (DR100/200/400). Each counterfactual supplies a code normally representing a capture DR mode.
  It does not recreate another capture or request another Fuji output-DR setting.
- 0.25 and 0.75 correspond to no capture mode (≈DR141, ≈DR283). 0.25is below the fit interval[0.5,1];0.75is interpolation inside that interval,
  although neither exact value occurs in training. No corresponding counterfactual target exists. The error measured there is distance to a target
  rendered under a different DR, at a condition the network never saw. That can only be read as curve
  shape.
- A deterministic function can be probed for sampled nonlinearity with more points.
  Three seeds limit generalisation of that observation, rather than preventing within-function description. Two extra points per source × 6 endpoints add 444 cells whose main risk is
  inviting post-hoc "sweet spot" readings, which §7 forbids anyway.

If root prefers five points, freeze all five in advance and report every one. Treat 0.25 and 0.75 as a
**descriptive shape tier only**: no summary statistic may be computed from them alone, and none may be
highlighted. Root decides; this proposal does not choose for them.

## 5. Measures (all defined before any inference)

Notation: endpoint e = (seed s, arm a); source i; condition c; actual condition c*ᵢ ∈ {0, 0.5}.
M(e,i,c) is any of the three `dev_metrics` scalars, computed on the CPU prediction at condition c
against the unchanged target.

### 5.1 Reproduction gate (must pass before any counterfactual is interpreted)
- For all 222 cells (e, i, c*ᵢ): |M_probe − M_frozen| ≤ the frozen canary tolerances (mean_dE00 1e-5,
  p95_dE00 1e-4, linear_l1_unclamped 1e-7). M_frozen is the per-scene value in that endpoint's frozen
  eval JSON, located by its SHA from the verification fingerprints.
- Also required: the condition tensor at c*ᵢ equals the one `Pairs.get` returns, and for every identity
  source `inputs["G"] == inputs["neutral"]`.
- **Run-order gate:** the probe runs the 222 reproduction cells first. If any cell fails, it stops
  before running a single counterfactual forward and writes a FAILED receipt. No partial counterfactual
  results are produced or reported.

### 5.2 Descriptive condition response (per endpoint)
- R(e,i,c) = M(e,i,c) − M(e,i,c*ᵢ), the error change from moving away from the actual condition.
  R = 0 at c*ᵢ by construction.
- Report per endpoint, per DR group (DR100-12, DR200-25) and per c: mean R, median R, min, max, and the
  counts R>0 and R<0. Report also the raw M(e,i,c) group means.
- Fit-support label on every c: DR200 sources have in-support counterfactual c=1 and out-of-support
  c=0. DR100 sources have in-support counterfactuals c=0.5 and c=1, but their actual c=0 is out of
  support.

### 5.3 Signed brightness (new, supplementary, defined now)
Rec709 linear luma Y = 0.2126 R + 0.7152 G + 0.0722 B, on unclamped linear tensors, mean over all
pixels:
- B_err(e,i,c) = mean Y_pred − mean Y_target (signed, linear).
- B_stops(e,i,c) = log2(mean Y_pred / mean Y_target).
- B_shift(e,i,c) = log2(mean Y_pred(c) / mean Y_pred(c*ᵢ)). This is target-free: the pure output
  brightness response of the network to the condition.

Coverage: B_stops is undefined when mean Y_target ≤ 1e-6 or mean Y_pred ≤ 1e-6. B_shift is undefined
when either mean Y_pred ≤ 1e-6. Undefined cells are recorded as `null` with a reason and counted. Group
statistics report `n_defined` and are computed only over defined cells. A group with fewer defined cells
than its full n is flagged `partial_coverage`, never silently pooled. Brightness answers "did the
condition mostly move exposure?". It is not an error metric, and nothing may be ranked by it.

### 5.4 Paired arm gap (per seed)
- Gap(s,i,c) = M((s,neutral),i,c) − M((s,G),i,c), the same sign convention as the study.
  At c = c*ᵢ this must reproduce the frozen per-scene delta. It is covered by 5.1, because both arms'
  cells reproduce.
- Report per seed, per DR group and per c: mean Gap, the counts neutral-worse and neutral-better, and
  the group contribution to a 37-source mean, (n_g/37) × mean Gap_g(c).
- A hypothetical "all 37 at condition c" mean gap may be listed **only** with the label
  `HYPOTHETICAL_UNIFORM_CONDITION_NOT_A_SCREEN`. It is never compared to the +0.10 margin and never put
  in a screen-shaped table.
- Report the observed seed range of Gap at each (group, c) as **three observed values**. Never call it
  a SD, a CI or noise.

### 5.5 Coverage rules (all measures)
- The expected cell set is fixed by the frozen scope. Each endpoint must yield exactly 37 × |grid|
  finite metric triples.
- A non-finite prediction or metric, or any hash, identity or role failure, is an **integrity failure**:
  the run stops and is reported as voided. Nothing is imputed, skipped or retried with changed settings.
- An aggregate is emitted only when its group has full coverage. The only exception is the §5.3
  brightness nulls, which are reported as counts.

## 6. Interpretation limits (binding on whoever writes up results)

1. **A wrong-DR condition changes the intended render.** On the camera, DR200 and DR400 change exposure
   and tone (base underexposure with shadow lift). The network learned capture-underexposure compensation as part of its tone/exposure mapping.
   Feeding c=0.5 to a DR100frame supplies inaccurate capture metadata; all targets remain outputDR100.
   This is a network-sensitivity probe, not a different desired look or a simulated capture. **Lower error at a wrong condition does NOT show that the DR100 harm is causally
   "extrapolation".** It does not show the network "should" treat DR100 as DR200, and it is not a
   production fix. It may simply mean an incorrect compensation changes brightness toward this fixed target.
2. **Sensitivity is not cause.** The probe identifies how each frozen function responds to c. It does
   not identify why the neutral-arm training made the c=0 region behave as it does on seeds 42/44 vs 43.
3. **Session confound.** All 12 DR100 sources are session 14. Anything specific to DR100 is
   inseparable from anything specific to session 14. Scene counts are not independent sessions.
4. **DR200→c=0 is the closest thing to a content-decoupling check.** Moving the 25 DR200 sources to c=0
   puts different scene content from the SAME session14 into the same out-of-support region. If the seed-42/44-worse and
   seed-43-better arm-gap pattern also shows up there, that is consistent with the region itself being
   seed-unstable. It is still sensitivity evidence, not identification, and three seeds cannot separate
   "arm effect" from "seed sensitivity".
5. **Three seeds per arm** limit everything to an observed pattern. No p-values, intervals,
   equivalence claims or "noise floor" from these runs.
6. **Internal development split only.** These sources were seen in prior development analyses. This is
   not acceptance, independent evaluation or Fuji parity.
7. **The original FAIL stands** whatever the probe shows.

## 7. Explicit prohibitions

- No target-optimised condition, per-image condition or gain, condition search, or argmin-over-c
  reporting.
- No thresholds, pass/fail or new gates on any probe quantity.
- No condition, remap or rule selected for the app, R11 or any future model.
- No dropping DR100, no selecting or favouring seed 43, no re-scored or "corrected" screen, no
  DR100-excluded verdict.
- No grid change, extra source, extra endpoint or rerun with changed settings after any result is seen.
- No training, fine-tuning, MPS inference, protected-tier access, or writing of media, arrays or
  checkpoints.
- No reuse or consumption of the untracked `training/runs/neutral-wb-study-progress.json`.

## 8. File, hash and role guards

**Bound inputs (read-only, SHA-checked before any forward pass):**

| File | Required SHA-256 |
|---|---|
| `training/diagnostics/neutral-wb-study-manifest.json` | `d416942f282ec8d1fbe65fad6ad6a4e3c30d612d7938cc6052bc364885b018d0` (via unchanged `D.load_manifest`, which also enforces committed bindings, R11, env and role checks) |
| `training/runs/neutral-wb-study-report.json` | `3e7e219463f245b86f1dd5da7437eaceb10ca267ae532f9de36ac974b3ffa6a4` |
| `training/diagnostics/neutral-wb-study-verification.json` | `e5b7cdbf82fb17e36e92c136e57d09c813af1c23696a09ef177ac1163665f059` |
| `training/diagnostics/neutral-wb-dr100-seed-spread.json` | SHA to be recorded at freeze (context only, not consumed numerically) |
| six eval JSONs | 42G `bde894ac…6666`, 42N `748b6d54…5e60`, 43G `9f03fb7d…6635`, 43N `3b93d2fc…efa9f`, 44G `b6eb52f1…6e59`, 44N `7b07428a…a9365` (full values as in the verification JSON) |
| six endpoint checkpoints | 42G `e99b0848…eb22`, 42N `76541dcf…c2f6`, 43G `0fb4f8f4…4fdf`, 43N `61ed544c…a7`, 44G `c7110f20…4717`, 44N `4be72c11…074` (full values from the verification `endpoint_fingerprints`) |
| R11 | `1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c`, checked before and after the run, unchanged |

**Role and source guards:**
- `D.Pairs(m, arm, role="development")`, then a frozen 37-name allowlist. The allowlist is stored in the
  scope manifest and must equal the manifest's identity subgroup (25 × DR200, 12 × DR100). Any other
  name is refused.
- Every one of the 37 must satisfy `neutral_kind == "G"`, `policy_subgroup == "identity"`,
  `inputs["G"] == inputs["neutral"]` and `captureDynamicRange ∈ {100, 200}`.
- The condition override is the only deviation from `Pairs.get`. The probe replaces the returned tensor
  with `torch.tensor([c], dtype=float32)` for c in the frozen grid tuple, asserts c ∈ grid, and asserts
  the returned tensor equals `log2(DR/100)/2` before replacing it.
- The device is asserted to be CPU. Weights are loaded strictly and their state digest checked against
  the endpoint meta (as `evaluate` does).

**New outputs (write-once, `open("x")`, no overwrite):**
- `training/diagnostics/neutral-wb-condition-probe-scope.json`: the frozen scope (grid, allowlist,
  bound SHAs, probe code SHA, thread count, environment). Committed before any run.
- `training/runs/neutral-wb-condition-probe-results.json`: per-cell scalars plus aggregates, or a
  FAILED or VOIDED receipt.
- `training/diagnostics/neutral-wb-condition-probe-verification.json`: the independent check.
- Code: `training/probe_neutral_wb_condition.py`, `training/test_probe_neutral_wb_condition.py`,
  `training/verify_neutral_wb_condition_probe.py`. All are new files, and no existing module is edited.

## 9. Expected arithmetic (for the verifier)

With the recommended grid {0, 0.5, 1}:
- 6 endpoints × 37 sources =222unique endpoint/source pairs. The two phases can reload them, for up to444pair loads with input/target SHA checks each.
- 222 reproduction forwards, then 444 counterfactual forwards (per endpoint: 12×2 for DR100 at
  {0.5, 1} + 25×2 for DR200 at {0, 1} = 74; × 6 = 444). Total **666**.
- Metric cells: 666 × 3 `dev_metrics` + 666 × 2 target-referenced brightness + 444 non-trivial B_shift
  (B_shift ≡ 0 at actual).
- Arm-gap cells: 3 seeds × 37 × 3 = 333. Of these, 111 at actual condition must equal the frozen
  per-scene deltas.
- Group contributions must sum to the 37-mean. At actual conditions per seed they must reproduce
  +1.106 / −2.268 / +0.037 (within summed tolerance).

With the five-point grid: 1110 forwards (222 + 888), 555 arm-gap cells, and the same reproduction
subset.

## 10. Minimal implementation split (proposed)

**Claude (authoring, no execution against real data):**
1. Write `probe_neutral_wb_condition.py`. It contains pure functions for the grid and allowlist
   construction, condition override, the measures in §5 and aggregation/report building, plus a thin
   runner that imports only the unchanged `D`, `T`, `dev_metrics`, `forward` and `validate_endpoint`.
   The runner enforces the reproduction-first order and the stop-on-integrity rule.
2. Write `test_probe_neutral_wb_condition.py`. Its synthetic fixtures only (a tiny fake model and fake
   rows) test: grid freeze, allowlist refusal of fit, treated and unknown names, the condition-equality
   assertion, the reproduction gate halting before counterfactuals, brightness null coverage, write-once
   refusal, and the `HYPOTHETICAL…` labelling.
3. Do a read-only review of Codex's verifier and the final results write-up for overclaiming against §6
   and §7.

**Codex (review, freeze, execution, verification):**
1. Review the probe and tests, then run the synthetic tests. Write `verify_neutral_wb_condition_probe.py`
   independently: recompute all aggregates from the per-cell scalars, check §9 arithmetic, and
   independently re-infer a small metadata-selected canary set per endpoint. Approved:2reproduction+3counterfactual cells per endpoint,30total; names selected by the frozen verifier's salted metadata hash rule before the run.
2. After root freezes the scope: write and commit the scope JSON and code, run the CPU probe, run the
   verifier, commit the results and push per CLAUDE.md.
3. If the reproduction gate fails, publish the FAILED receipt and stop. Do not debug by changing tolerances.

**Root:** review this proposal, choose the 3- or 5-point grid, approve or amend the measures and canary
count, and freeze before any inference.

---

## Root handoff (concise)

- **Ask:** approve, amend or reject an inference-only probe. It covers 6 frozen endpoints × 37 identity
  sources × condition grid, CPU, unchanged frames, targets, weights and `dev_metrics`, and is bound to
  manifest `d416942f…8d0` and the saved endpoint and eval SHAs.
- **Recommend grid {0, 0.5, 1} (666 forwards)** over five points (1110). The inner points match no
  capture mode:0.25is extrapolation below0.5, while0.75interpolates between0.5and1. Function shape can be described, but it does not identify the training mechanism. If you choose five
  points, freeze them as a descriptive shape tier only.
- **Gate:** all 222 actual-condition cells must reproduce the frozen evals within the canary tolerances
  before any counterfactual runs.
- **What it can say:** how each frozen network's error, arm gap and output brightness respond to the
  condition input. That is sensitivity. **What it cannot say:** why training produced it, that DR100
  harm is "extrapolation", or that any remap is a fix. A lower error at a wrong DR is expected, because
  the condition changes the intended exposure.
- **Unchanged:** the original screen stays FAIL 4/5. There is no DR100 dropping, no seed-43 selection,
  and no thresholds, app/R11 change or protected access.
- **Corrections on record:** seed 44 passed despite DR100 (+0.430 offset by DR200 −0.393), and three
  seeds are not a noise distribution.


## Root implementation decision (2026-11-26; still no inference permission)

Proceed to implementation with the fixed three-point grid[0,0.5,1]:666
forwards, with all222actual-condition reproductions completed before any
counterfactual. Root corrected the earlier wording:0.75is interpolation,
not extrapolation; all37sources are session14; captureDRconditioning is
distinct from the outputDR100target protocol. Wrong-condition improvement
is not a production fix or causal mechanism identification.

Independent canaries: five per endpoint (30total), using one deterministic
metadata-hash-selected source from each DR group. Two actual-condition
reproductions plus DR100at0.5and1, and DR200at0. This covers all three
codes and the main swapped-code comparisons. Numerical metric limits
remain unchanged. Thread count is frozen from the observed current CPU
default at preparation. Reproduction and counterfactual phases may load
inputs twice rather than cache full frames:222unique endpoint/source
pairs and up to444pair-load operations; the forward counts remain666.

No fitting, new sources, app change, threshold adjustment, best-condition
selection or protected-tier access is authorised. The code and executable
scope must be committed and independently checked before any forward.


## Final executable implementation notes (Codex, before freeze)

The runner writes raw scalar cells only. The separate verifier recomputes
all group response statistics, paired gaps, signed contributions,
brightness coverage and observed three-seed ranges from those cells, then
independently re-infers30canary cells. It does not use a second copy of the
runner's summary function: there is no runner summary function. This avoids
claiming independent arithmetic when both paths call the same helper.
Primary numeric reproducibility limits remain1e-5/1e-4/1e-7. Supplementary
brightness verification limits are1e-6for linear means/difference and1e-5
for stop/response ratios; these are implementation tolerances, not colour
acceptance limits. All are frozen in code/scope before inference.

Preparation reads only parent JSON metadata and byte hashes. Invalid
scopes or pre-inference metadata bindings refuse before any forward and
write no result. Once a valid run begins, reproduction or later integrity
errors write an exclusive FAILED_REPRODUCTION or VOIDED receipt and stop.
No metric, source or setting is changed to obtain a successful run.

The observed current CPU default is6threads, stored by prepare and restored
in both runner/verifier. Full phase ordering is222actual reproductions,
then444counterfactuals. Full source loads may occur twice; no full prediction
arrays or checkpoints are saved. Original result remains FAIL regardless.
