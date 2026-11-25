"""Evaluate neutral-WB study endpoints and write the paired neutral-minus-G screening report.

evaluate(manifest, seed, arm, checkpoint_path) -> per-arm eval JSON path
    Validates the fixed epoch-20 endpoint (manifest, rows, seed, arm, initial hash, order, config, trainer,
    data API, checkpoint SHA, weight digest), refuses incomplete runs, runs a CPU forward on all 571 scenes
    (455 fit + 116 development; explicit role=None access) and scores each with provia_color_pilot.dev_metrics,
    imported unchanged. Split/session/DR summaries use the unchanged eval_matched_gamma_training.summarise.
report(manifest) -> report path
    Pairs neutral minus G per seed with no checkpoint selection. Until all six evals exist it rewrites a
    progress report; then it writes the write-once final report with the frozen D.SCREEN criteria:
      pooled-across-seeds development mean delta dE00 <= -0.25;
      every seed's development mean delta < 0;
      pooled development mean per-scene p95 delta <= 0;
      treated-79 pooled mean delta < 0;
      every seed's identity-37 mean delta <= +0.10 (predeclared engineering margin).
    PASS requires all five. Internal screening only: no automatic selection, no app change, no final gate.
    Subgroups are frozen manifest metadata (policy_subgroup), never regrouped from pixel results.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

import neutral_wb_study_data as D
import run_neutral_wb_study as T
from provia_color_pilot import dev_metrics  # verbatim, unchanged metric definition
from eval_matched_gamma_training import summarise, grouped, _mean, _check_finite  # unchanged summary algorithm

CODE = Path(__file__).resolve()
ROLES = ("fit", "development")
SUBGROUPS = ("treated", "identity")
IDENTITY_FIELDS = ("role", "shoot", "session", "captureDynamicRange", "neutral_kind", "policy_subgroup")
DR100_LABEL = "development only, outside fit support: OUT OF SUPPORT, NEVER QUALIFIED"
CONFOUND_LABEL = ("METADATA CONFOUND: all 79 treated development scenes are DR400, while the 37 identity development "
                  "scenes are 25 DR200 + 12 DR100. Treated-vs-identity differences are confounded with capture DR "
                  "groups and cannot be attributed to the WB input change alone.")
NOTES = ["Internal development screening only; not acceptance, not app or model selection, no final gate.",
         "Fixed epoch-20 endpoints from fresh seeded weights; no checkpoint selection, no early stopping, no retuning.",
         "Identity scenes feed the same G input in both arms; their delta measures training-mix effects only.",
         "All 12 DR100 scenes are development identity scenes outside fit support: reported, never qualified.",
         "Round 11 is context only (trained on a different set); no Round 11 warm start, inference or change.",
         CONFOUND_LABEL]


def eval_path(seed, arm):
    return T.RUNS / f"neutral-wb-study-seed{seed}-{arm}-eval.json"


def final_path():
    return T.RUNS / "neutral-wb-study-report.json"


def progress_path():
    return T.RUNS / "neutral-wb-study-progress.json"


def log(msg):
    print(msg, flush=True)


def _is_dr100(s):
    return s["captureDynamicRange"] == 100


# ------------------------------------------------------------------ per-arm evaluation
def validate_eval(m, seed, arm, r, checkpoint_sha):
    D.need(r.get("kind") == "neutral-wb-study-endpoint-eval" and r.get("manifest_sha256") == m["_manifest_sha256"]
           and r.get("rows_sha256") == m["_rows_sha256"] and r.get("seed") == seed and r.get("arm") == arm
           and r.get("checkpoint_sha256") == checkpoint_sha and r.get("device") == "cpu",
           "Existing evaluation belongs to another endpoint/manifest or was not a CPU evaluation")
    D.need(isinstance(r.get("evaluator_sha256"), str) and len(r["evaluator_sha256"]) == 64, "Evaluator hash missing")
    wanted = {s["name"]: s for s in m["scenes"]}
    D.need(set(r.get("scenes", {})) == set(wanted) and len(wanted) == T.ALL_PAIRS, "Evaluation scene coverage differs")
    for name, row in r["scenes"].items():
        D.need(all(row.get(k) == wanted[name][k] for k in IDENTITY_FIELDS), "Evaluation scene identity differs")
        D.need(all(type(row["metrics"].get(k)) in (int, float) for k in ("mean_dE00", "p95_dE00", "linear_l1_unclamped")),
               "Missing or non-numeric evaluation metric")
    _check_finite(r)


def evaluate(manifest, seed, arm, checkpoint_path):
    m = manifest
    dev = torch.device("cpu")  # fixed CPU scoring for every endpoint
    D.need(m["screen"] == D.SCREEN, "Manifest screen differs from the frozen study screen")
    ep, ck, ck_sha = T.validate_endpoint(m, seed, arm, checkpoint_path)
    T.validate_split(m)
    out = eval_path(seed, arm)
    if out.exists():
        validate_eval(m, seed, arm, json.loads(out.read_text()), ck_sha)
        return out
    T.check_r11()
    meta = ck["meta"]
    model = D.initialise(m, seed, dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    D.need(D.state_digest(T._to_cpu(model.state_dict())) == meta["state_sha256"], "Loaded weights differ from endpoint digest")
    pairs = D.Pairs(m, arm, role=None)  # explicit all-role access for the frozen 571-pair evaluation
    rows, t0 = {}, time.time()
    with torch.no_grad():
        for i, s in enumerate(m["scenes"]):
            x, y, cond = pairs.get(s["name"])
            pred = D.forward(model, m, x, cond, dev)
            D.need(bool(torch.isfinite(pred).all()), f"Non-finite prediction for {s['name']}")
            rows[s["name"]] = {**{k: s[k] for k in IDENTITY_FIELDS},
                               "dr100_out_of_support": _is_dr100(s), "metrics": dev_metrics(pred.cpu(), y.cpu())}
            _check_finite(rows[s["name"]], s["name"])
            if (i + 1) % 50 == 0:
                log(f"[eval seed{seed} {arm}] {i + 1}/{len(m['scenes'])} scenes ({time.time() - t0:.0f} s)")
    D.need(len(rows) == T.ALL_PAIRS, "Did not score every scene")
    rl = list(rows.values())
    dev_rows = [r for r in rl if r["role"] == "development"]
    obj = {"kind": "neutral-wb-study-endpoint-eval", "manifest_sha256": m["_manifest_sha256"],
           "rows_sha256": m["_rows_sha256"], "seed": seed, "arm": arm,
           "checkpoint": T._rel(ep), "checkpoint_sha256": ck_sha, "state_sha256": meta["state_sha256"],
           "initial_state_sha256": meta["initial_state_sha256"], "epoch": meta["epoch"], "device": str(dev),
           "evaluator_sha256": D.sha(CODE), "trainer_sha256": meta["trainer_sha256"],
           "metric_function": "provia_color_pilot.dev_metrics (unchanged)",
           "summary_function": "eval_matched_gamma_training.summarise (unchanged)",
           "dr100": {"names": sorted(n for n, r in rows.items() if r["dr100_out_of_support"]), "status": DR100_LABEL},
           "split_means": {role: summarise([r for r in rl if r["role"] == role]) for role in ROLES},
           "development_subgroups": {sub: summarise([r for r in dev_rows if r["policy_subgroup"] == sub]) for sub in SUBGROUPS},
           "per_session": {role: grouped([r for r in rl if r["role"] == role], "session") for role in ROLES},
           "per_dr": {role: grouped([r for r in rl if r["role"] == role], "captureDynamicRange") for role in ROLES},
           "scenes": rows, "notes": NOTES}
    _check_finite(obj)
    D.write_once_json(out, obj)
    T.check_r11()
    return out


# ------------------------------------------------------------------ paired report (pure given validated evals)
def _paired(G, N, names):
    """Neutral minus G over `names` (the old paired algorithm with the arms renamed)."""
    if not names:
        return {"n": 0}
    gm = [G["scenes"][n]["metrics"]["mean_dE00"] for n in names]
    nm = [N["scenes"][n]["metrics"]["mean_dE00"] for n in names]
    gp = [G["scenes"][n]["metrics"]["p95_dE00"] for n in names]
    np_ = [N["scenes"][n]["metrics"]["p95_dE00"] for n in names]
    d = [b - a for a, b in zip(gm, nm)]
    return {"n": len(names), "G_mean_dE00": _mean(gm), "neutral_mean_dE00": _mean(nm), "delta_mean_dE00": _mean(d),
            "G_mean_p95_dE00": _mean(gp), "neutral_mean_p95_dE00": _mean(np_),
            "G_median_scene_dE00": float(np.median(gm)), "neutral_median_scene_dE00": float(np.median(nm)),
            "delta_mean_p95_dE00": _mean([b - a for a, b in zip(gp, np_)]),
            "scenes_neutral_worse": sum(x > 0 for x in d), "scenes_neutral_better": sum(x < 0 for x in d),
            "worst_scene_delta_dE00": max(d), "best_scene_delta_dE00": min(d)}


def _pool(blocks):
    """Equal-weight mean of per-seed paired blocks (identical scene sets across seeds)."""
    keys = [k for k in blocks[0] if k not in ("n", "scenes_neutral_worse", "scenes_neutral_better")]
    out = {"n_per_seed": blocks[0]["n"], "seeds": len(blocks)}
    if blocks[0]["n"]:
        out.update({k: _mean([b[k] for b in blocks]) for k in keys})
    return out


def _groups(m):
    by = {s["name"]: s for s in m["scenes"]}
    fit = [s["name"] for s in m["scenes"] if s["role"] == "fit"]
    dev = [s["name"] for s in m["scenes"] if s["role"] == "development"]
    g = {"development": dev, "fit": fit,
         "development_treated79": [n for n in dev if by[n]["policy_subgroup"] == "treated"],
         "development_identity37": [n for n in dev if by[n]["policy_subgroup"] == "identity"],
         "development_excluding_DR100": [n for n in dev if not _is_dr100(by[n])],
         "development_DR100_out_of_support": [n for n in dev if _is_dr100(by[n])],
         "development_identity37_DR200": [n for n in dev if by[n]["policy_subgroup"] == "identity"
                                          and by[n]["captureDynamicRange"] == 200],
         "fit_treated": [n for n in fit if by[n]["policy_subgroup"] == "treated"],
         "fit_identity": [n for n in fit if by[n]["policy_subgroup"] == "identity"]}
    return by, g


def screen(m, per_seed, pooled, seeds):
    S = D.SCREEN
    D.need(m["screen"] == S, "Manifest screen differs from the frozen study screen")
    pd, pt = pooled["development"], pooled["development_treated79"]
    seed_dev = {str(s): per_seed[str(s)]["development"]["delta_mean_dE00"] for s in seeds}
    seed_id = {str(s): per_seed[str(s)]["development_identity37"]["delta_mean_dE00"] for s in seeds}
    c = {
        "pooled_development_mean_delta_dE00_le": {
            "value": pd["delta_mean_dE00"], "op": "<=", "threshold": S["pooled_dev_mean_delta_max"],
            "pass": pd["delta_mean_dE00"] <= S["pooled_dev_mean_delta_max"]},
        "every_seed_development_mean_delta_dE00_lt": {
            "values": seed_dev, "op": "<", "threshold": S["each_seed_dev_mean_delta_strict_max"],
            "pass": all(v < S["each_seed_dev_mean_delta_strict_max"] for v in seed_dev.values())},
        "pooled_development_mean_scene_p95_delta_dE00_le": {
            "value": pd["delta_mean_p95_dE00"], "op": "<=", "threshold": S["pooled_dev_mean_scene_p95_delta_max"],
            "pass": pd["delta_mean_p95_dE00"] <= S["pooled_dev_mean_scene_p95_delta_max"]},
        "treated79_pooled_mean_delta_dE00_lt": {
            "value": pt["delta_mean_dE00"], "op": "<", "threshold": S["treated79_pooled_mean_delta_strict_max"],
            "pass": pt["delta_mean_dE00"] < S["treated79_pooled_mean_delta_strict_max"]},
        "every_seed_identity37_mean_delta_dE00_le_engineering_margin": {
            "values": seed_id, "op": "<=", "threshold": S["identity37_each_seed_mean_delta_max"],
            "pass": all(v <= S["identity37_each_seed_mean_delta_max"] for v in seed_id.values())}}
    D.need(len(c) == 5, "Screen must have exactly five frozen criteria")
    return {"scope": "internal development screening only; all 116 development scenes as frozen (DR100 included in "
                     "pooled and identity means but reported separately as out of support, never qualified); no "
                     "automatic selection, app change or final gate",
            "criteria": c, "passed": all(v["pass"] for v in c.values())}


def build_report(m, evals):
    """evals: {(seed, arm): validated eval dict or None}. Pure; no file access."""
    seeds = m["training"]["seeds"]
    facts = T.metadata_facts(m)
    by, g = _groups(m)
    dev = g["development"]
    sessions = {role: sorted({str(by[n]["session"]) for n in g[role]}) for role in ("development", "fit")}
    drs = {role: sorted({by[n]["captureDynamicRange"] for n in g[role]}) for role in ("development", "fit")}
    complete = [s for s in seeds if evals.get((s, "G")) and evals.get((s, "neutral"))]
    present = [r for r in evals.values() if r]
    D.need(len({r["evaluator_sha256"] for r in present}) <= 1, "Evaluations were produced by different evaluator versions")
    per_seed = {}
    for s in complete:
        G, N = evals[(s, "G")], evals[(s, "neutral")]
        block = {"G_checkpoint_sha256": G["checkpoint_sha256"], "neutral_checkpoint_sha256": N["checkpoint_sha256"]}
        block.update({k: _paired(G, N, names) for k, names in g.items()})
        for role in ("development", "fit"):
            block[f"per_session_{role}"] = {x: _paired(G, N, [n for n in g[role] if str(by[n]["session"]) == x])
                                            for x in sessions[role]}
            block[f"per_dr_{role}"] = {str(x): _paired(G, N, [n for n in g[role] if by[n]["captureDynamicRange"] == x])
                                       for x in drs[role]}
        regress = [{"name": n, "role": by[n]["role"], "policy_subgroup": by[n]["policy_subgroup"],
                    "session": by[n]["session"], "captureDynamicRange": by[n]["captureDynamicRange"],
                    "G_mean_dE00": G["scenes"][n]["metrics"]["mean_dE00"],
                    "neutral_mean_dE00": N["scenes"][n]["metrics"]["mean_dE00"],
                    "delta_mean_dE00": N["scenes"][n]["metrics"]["mean_dE00"] - G["scenes"][n]["metrics"]["mean_dE00"],
                    "delta_p95_dE00": N["scenes"][n]["metrics"]["p95_dE00"] - G["scenes"][n]["metrics"]["p95_dE00"]}
                   for n in by]
        block["regressions_development"] = sorted([r for r in regress if r["role"] == "development" and r["delta_mean_dE00"] > 0],
                                                  key=lambda r: -r["delta_mean_dE00"])
        block["regressions_fit"] = sorted([r for r in regress if r["role"] == "fit" and r["delta_mean_dE00"] > 0],
                                          key=lambda r: -r["delta_mean_dE00"])
        per_seed[str(s)] = block
    out = {"kind": "neutral-wb-study-paired-report", "manifest_sha256": m["_manifest_sha256"],
           "rows_sha256": m["_rows_sha256"], "seeds": seeds, "seeds_complete": complete,
           "comparison": "neutral minus G, per seed, fixed epoch-20 endpoints, CPU scoring",
           "evals_present": sorted(f"seed{s}-{a}" for (s, a), r in evals.items() if r),
           "evaluator_sha256": present[0]["evaluator_sha256"] if present else None,
           "subgroups": {"source": "frozen manifest policy_subgroup metadata; never regrouped from pixel results",
                         "treated": {k: facts["treated"][k] for k in ("n", "dr_counts")},
                         "identity": {k: facts["identity"][k] for k in ("n", "dr_counts")}},
           "confound": CONFOUND_LABEL,
           "dr100": {"names": facts["dr100_development"], "status": DR100_LABEL},
           "per_seed": per_seed, "notes": NOTES}
    if len(complete) != len(seeds):
        out.update(status="incomplete", screening=None)
        _check_finite(out)
        return out
    blocks = [per_seed[str(s)] for s in seeds]
    pooled = {k: _pool([b[k] for b in blocks]) for k in g}
    for role in ("development", "fit"):
        pooled[f"per_session_{role}"] = {x: _pool([b[f"per_session_{role}"][x] for b in blocks]) for x in sessions[role]}
        pooled[f"per_dr_{role}"] = {str(x): _pool([b[f"per_dr_{role}"][str(x)] for b in blocks]) for x in drs[role]}
    across = []
    for n, row in by.items():
        d = [evals[(s, "neutral")]["scenes"][n]["metrics"]["mean_dE00"] - evals[(s, "G")]["scenes"][n]["metrics"]["mean_dE00"]
             for s in seeds]
        p = [evals[(s, "neutral")]["scenes"][n]["metrics"]["p95_dE00"] - evals[(s, "G")]["scenes"][n]["metrics"]["p95_dE00"]
             for s in seeds]
        across.append({"name": n, "role": row["role"], "policy_subgroup": row["policy_subgroup"],
                       "neutral_kind": row["neutral_kind"], "session": row["session"],
                       "captureDynamicRange": row["captureDynamicRange"], "dr100_out_of_support": _is_dr100(row),
                       "seed_deltas_mean_dE00": d, "mean_delta_mean_dE00": _mean(d),
                       "seed_deltas_p95_dE00": p, "mean_delta_p95_dE00": _mean(p),
                       "seeds_neutral_worse": sum(x > 0 for x in d)})
    out.update(status="complete", pooled_across_seeds=pooled, per_scene_all=across,
               regressions_development_across_seeds=sorted(
                   [a for a in across if a["role"] == "development" and a["mean_delta_mean_dE00"] > 0],
                   key=lambda a: -a["mean_delta_mean_dE00"]),
               regressions_fit_across_seeds=sorted(
                   [a for a in across if a["role"] == "fit" and a["mean_delta_mean_dE00"] > 0],
                   key=lambda a: -a["mean_delta_mean_dE00"]),
               development_scenes_with_any_seed_regression=sum(a["seeds_neutral_worse"] > 0 for a in across
                                                               if a["role"] == "development"),
               screening=screen(m, per_seed, pooled, seeds))
    _check_finite(out)
    return out


def _load_eval(m, seed, arm):
    p = eval_path(seed, arm)
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    _, _, sha = T.validate_endpoint(m, seed, arm)
    validate_eval(m, seed, arm, r, sha)
    return r


def report(manifest):
    m = manifest
    T.validate_split(m)
    evals = {(s, a): _load_eval(m, s, a) for s in m["training"]["seeds"] for a in T.ARMS}
    out = build_report(m, evals)
    if out["status"] != "complete":
        T.atomic_json(progress_path(), out)
        return progress_path()
    fp = final_path()
    if fp.exists():
        D.need(json.loads(fp.read_text()) == json.loads(json.dumps(out, allow_nan=False)),
               f"Existing final report {fp} differs from recomputation; refusing")
        return fp
    D.write_once_json(fp, out)
    return fp
