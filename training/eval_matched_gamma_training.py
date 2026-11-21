"""Evaluate matched gamma-only endpoints and write the paired legacy-vs-linear screening report.

evaluate(manifest, seed, arm, checkpoint_path, device=None) -> per-arm eval JSON path
    Validates the endpoint (manifest, seed, arm, epoch 20, initial hash, checkpoint SHA, weight digest),
    refuses incomplete runs, runs a CPU forward on all 583 scenes and scores each with
    provia_color_pilot.dev_metrics, imported unchanged. Writes split means, per-session and per-DR
    summaries per role, and per-scene rows with role labels. DR100 (development only, outside fit
    support) is labelled NOT VALIDATED.
report(manifest) -> report path
    Pairs linear (G) minus legacy per seed with no checkpoint selection. Until all six evals exist it
    rewrites a progress report; once they do it writes the write-once final report with the internal
    development screen:
      pooled-across-seeds mean development dE00 (G - legacy) <= -0.25;
      each seed's development mean G < legacy;
      pooled mean per-scene p95, G <= legacy.
    Every regression is reported with its magnitude. This is internal screening only: not acceptance,
    not app or model selection. Round 11 is context only (it trained on all 583); no Round 11 inference.
"""
import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

import matched_gamma_data as D
import run_matched_gamma_training as T
from provia_color_pilot import dev_metrics  # verbatim, unchanged metric definition

SCREEN_DELTA = -0.25
ROLES = ("fit", "development")
SCALARS = ("mean_dE00", "p95_dE00", "linear_l1_unclamped", "signed_dC_ref_C_gt_10", "signed_hue_deg_ref_C_gt_20",
           "frac_any_channel_below_0", "frac_any_channel_above_1")
DR100_LABEL = "development only, outside fit support: NOT VALIDATED"
NOTES = ["Internal development screening only; not acceptance, not app or model selection.",
         "Fixed epoch-20 endpoints; no checkpoint selection, no early stopping.",
         "All 12 DR100 scenes are development-only and outside fit support; reported, not validated.",
         "Round 11 trained on all 583 scenes, development included: context only, not a control. No Round 11 inference."]


def eval_path(seed, arm):
    return D.RUNS / f"matched-gamma-seed{seed}-{arm}-eval.json"


def final_path():
    return D.RUNS / "matched-gamma-report.json"


def progress_path():
    return D.RUNS / "matched-gamma-progress.json"


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ helpers
def _check_finite(obj, where="report"):
    if isinstance(obj, float):
        D.need(bool(np.isfinite(obj)), f"Non-finite value at {where}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_finite(v, f"{where}[{i}]")


def _mean(vals):
    v = [float(x) for x in vals if x is not None]
    return float(np.mean(v)) if v else None   # empty -> null, never NaN


def summarise(rows):
    out = {"n": len(rows)}
    if not rows:
        return out
    for k in SCALARS:
        vals = [r["metrics"][k] for r in rows]
        out[k] = _mean(vals)
        defined = sum(v is not None for v in vals)
        if defined != len(rows):
            out[k + "_n_defined"] = defined
    bands = {}
    for b in rows[0]["metrics"]["tone_band_signed_Lab"]:
        vecs = [r["metrics"]["tone_band_signed_Lab"][b] for r in rows if r["metrics"]["tone_band_signed_Lab"][b] is not None]
        bands[b] = [float(v) for v in np.mean(np.array(vecs, dtype=np.float64), 0)] if vecs else None
    out["tone_band_signed_Lab"] = bands
    out["median_scene_mean_dE00"] = float(np.median([r["metrics"]["mean_dE00"] for r in rows]))
    return out


def grouped(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(r)
    return {k: summarise(v) for k, v in sorted(groups.items())}


def _is_dr100(s):
    return s["captureDynamicRange"] == 100


def validate_eval(m, seed, arm, r, checkpoint_sha):
    D.need(r.get("manifest_sha256") == m["_manifest_sha256"] and r.get("seed") == seed
           and r.get("arm") == arm and r.get("checkpoint_sha256") == checkpoint_sha,
           "Existing evaluation belongs to another endpoint")
    wanted = {s["name"]: s for s in m["scenes"]}
    D.need(set(r.get("scenes", {})) == set(wanted), "Evaluation scene coverage differs")
    for name, row in r["scenes"].items():
        D.need(all(row[k] == wanted[name][k] for k in ("role", "shoot", "session", "captureDynamicRange")),
               "Evaluation scene identity differs")
        D.need(all(type(row["metrics"][k]) in (int, float) for k in ("mean_dE00", "p95_dE00", "linear_l1_unclamped")),
               "Missing or non-numeric evaluation metric")
    _check_finite(r)


# ------------------------------------------------------------------ per-arm evaluation
def evaluate(manifest, seed, arm, checkpoint_path, device=None):
    m = manifest
    dev = torch.device(device or "cpu")
    ep, ck, ck_sha = T.validate_endpoint(m, seed, arm, checkpoint_path)
    out = eval_path(seed, arm)
    if out.exists():
        r = json.loads(out.read_text())
        validate_eval(m, seed, arm, r, ck_sha)
        return out
    T.check_r11()
    T.validate_split(m)
    meta = ck["meta"]
    model = D.initialise(m, seed, dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    D.need(D.state_digest(T._to_cpu(model.state_dict())) == meta["state_sha256"], "Loaded weights differ from endpoint digest")
    pairs = D.Pairs(m, arm, role=None)
    rows, t0 = {}, time.time()
    with torch.no_grad():
        for i, s in enumerate(m["scenes"]):
            x, y, cond = pairs.get(s["name"])
            pred = D.forward(model, m, x, cond, dev)
            D.need(bool(torch.isfinite(pred).all()), f"Non-finite prediction for {s['name']}")
            rows[s["name"]] = {"role": s["role"], "shoot": s["shoot"], "session": s["session"],
                               "captureDynamicRange": s["captureDynamicRange"],
                               "dr100_not_validated": _is_dr100(s), "metrics": dev_metrics(pred.cpu(), y.cpu())}
            _check_finite(rows[s["name"]], s["name"])
            if (i + 1) % 50 == 0:
                log(f"[eval seed{seed} {arm}] {i + 1}/{len(m['scenes'])} scenes ({time.time() - t0:.0f} s)")
    D.need(len(rows) == T.FIT_N + T.DEV_N, "Did not score every scene")
    rl = list(rows.values())
    dr100 = sorted(n for n, r in rows.items() if r["captureDynamicRange"] == 100)
    obj = {"kind": "matched-gamma-endpoint-eval", "manifest_sha256": m["_manifest_sha256"], "seed": seed, "arm": arm,
           "checkpoint": T._rel(ep), "checkpoint_sha256": ck_sha, "state_sha256": meta["state_sha256"],
           "initial_state_sha256": meta["initial_state_sha256"], "epoch": meta["epoch"], "device": str(dev),
           "metric_function": "provia_color_pilot.dev_metrics (unchanged)",
           "dr100": {"names": dr100, "status": DR100_LABEL},
           "split_means": {role: summarise([r for r in rl if r["role"] == role]) for role in ROLES},
           "per_session": {role: grouped([r for r in rl if r["role"] == role], "session") for role in ROLES},
           "per_dr": {role: grouped([r for r in rl if r["role"] == role], "captureDynamicRange") for role in ROLES},
           "scenes": rows, "notes": NOTES}
    _check_finite(obj)
    D.write_once_json(out, obj)
    T.check_r11()
    return out


# ------------------------------------------------------------------ paired report
def _load_eval(m, seed, arm):
    p = eval_path(seed, arm)
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    _, _, sha = T.validate_endpoint(m, seed, arm)
    validate_eval(m, seed, arm, r, sha)
    return r


def _paired(L, G, names):
    if not names:
        return {"n": 0}
    lm = [L["scenes"][n]["metrics"]["mean_dE00"] for n in names]
    gm = [G["scenes"][n]["metrics"]["mean_dE00"] for n in names]
    lp = [L["scenes"][n]["metrics"]["p95_dE00"] for n in names]
    gp = [G["scenes"][n]["metrics"]["p95_dE00"] for n in names]
    d = [g - l for g, l in zip(gm, lm)]
    return {"n": len(names), "legacy_mean_dE00": _mean(lm), "linear_mean_dE00": _mean(gm), "delta_mean_dE00": _mean(d),
            "legacy_mean_p95_dE00": _mean(lp), "linear_mean_p95_dE00": _mean(gp),
            "legacy_median_scene_dE00": float(np.median(lm)), "linear_median_scene_dE00": float(np.median(gm)),
            "delta_mean_p95_dE00": _mean([g - l for g, l in zip(gp, lp)]),
            "scenes_linear_worse": sum(x > 0 for x in d), "scenes_linear_better": sum(x < 0 for x in d),
            "worst_scene_delta_dE00": max(d), "best_scene_delta_dE00": min(d)}


def _pool(blocks):
    """Equal-weight mean of per-seed paired blocks (identical scene sets across seeds)."""
    keys = [k for k in blocks[0] if k not in ("n", "scenes_linear_worse", "scenes_linear_better")]
    out = {"n_per_seed": blocks[0]["n"], "seeds": len(blocks)}
    if blocks[0]["n"]:
        out.update({k: _mean([b[k] for b in blocks]) for k in keys})
    return out


def report(manifest):
    m = manifest
    seeds = m["training"]["seeds"]
    by = {s["name"]: s for s in m["scenes"]}
    fit = [s["name"] for s in m["scenes"] if s["role"] == "fit"]
    dev = [s["name"] for s in m["scenes"] if s["role"] == "development"]
    dev_dr100 = [n for n in dev if _is_dr100(by[n])]
    dev_other = [n for n in dev if not _is_dr100(by[n])]
    sessions = sorted({str(by[n]["session"]) for n in dev})
    drs = sorted({by[n]["captureDynamicRange"] for n in dev})
    evals = {(s, a): _load_eval(m, s, a) for s in seeds for a in T.ARMS}
    complete = [s for s in seeds if evals[(s, "legacy")] and evals[(s, "linear")]]
    per_seed = {}
    for s in complete:
        L, G = evals[(s, "legacy")], evals[(s, "linear")]
        regress = [{"name": n, "session": by[n]["session"], "captureDynamicRange": by[n]["captureDynamicRange"],
                    "legacy_mean_dE00": L["scenes"][n]["metrics"]["mean_dE00"],
                    "linear_mean_dE00": G["scenes"][n]["metrics"]["mean_dE00"],
                    "delta_mean_dE00": G["scenes"][n]["metrics"]["mean_dE00"] - L["scenes"][n]["metrics"]["mean_dE00"],
                    "delta_p95_dE00": G["scenes"][n]["metrics"]["p95_dE00"] - L["scenes"][n]["metrics"]["p95_dE00"]}
                   for n in dev]
        per_seed[str(s)] = {
            "legacy_checkpoint_sha256": L["checkpoint_sha256"], "linear_checkpoint_sha256": G["checkpoint_sha256"],
            "development": _paired(L, G, dev),
            "development_excluding_DR100": _paired(L, G, dev_other),
            "development_DR100_not_validated": _paired(L, G, dev_dr100),
            "fit": _paired(L, G, fit),
            "per_session_development": {x: _paired(L, G, [n for n in dev if str(by[n]["session"]) == x]) for x in sessions},
            "per_dr_development": {str(x): _paired(L, G, [n for n in dev if by[n]["captureDynamicRange"] == x]) for x in drs},
            "per_scene_development": regress,
            "regressions_development": sorted([r for r in regress if r["delta_mean_dE00"] > 0],
                                              key=lambda r: -r["delta_mean_dE00"])}
    out = {"kind": "matched-gamma-paired-report", "manifest_sha256": m["_manifest_sha256"], "seeds": seeds,
           "seeds_complete": complete, "comparison": "linear (G) minus legacy, per seed, fixed epoch-20 endpoints",
           "evals_present": sorted(f"seed{s}-{a}" for (s, a), r in evals.items() if r),
           "dr100": {"names": sorted(dev_dr100), "status": DR100_LABEL}, "per_seed": per_seed, "notes": NOTES}
    if len(complete) != len(seeds):
        out.update(status="incomplete", screening=None)
        _check_finite(out)
        T.atomic_json(progress_path(), out)
        return progress_path()
    blocks = [per_seed[str(s)] for s in seeds]
    pooled = {k: _pool([b[k] for b in blocks]) for k in
              ("development", "development_excluding_DR100", "development_DR100_not_validated", "fit")}
    pooled["per_session_development"] = {x: _pool([b["per_session_development"][x] for b in blocks]) for x in sessions}
    pooled["per_dr_development"] = {str(x): _pool([b["per_dr_development"][str(x)] for b in blocks]) for x in drs}
    across = []
    for n in dev:
        d = [evals[(s, "linear")]["scenes"][n]["metrics"]["mean_dE00"] - evals[(s, "legacy")]["scenes"][n]["metrics"]["mean_dE00"]
             for s in seeds]
        across.append({"name": n, "session": by[n]["session"], "captureDynamicRange": by[n]["captureDynamicRange"],
                       "seed_deltas_mean_dE00": d, "mean_delta_mean_dE00": _mean(d), "seeds_linear_worse": sum(x > 0 for x in d)})
    pd = pooled["development"]
    seed_means = {str(s): {"legacy": per_seed[str(s)]["development"]["legacy_mean_dE00"],
                           "linear": per_seed[str(s)]["development"]["linear_mean_dE00"]} for s in seeds}
    criteria = {
        "pooled_development_delta_mean_dE00_le_-0.25": {"value": pd["delta_mean_dE00"], "threshold": SCREEN_DELTA,
                                                        "pass": pd["delta_mean_dE00"] <= SCREEN_DELTA},
        "every_seed_development_mean_linear_lt_legacy": {"seed_means": seed_means,
                                                         "pass": all(v["linear"] < v["legacy"] for v in seed_means.values())},
        "pooled_mean_per_scene_p95_linear_le_legacy": {"legacy": pd["legacy_mean_p95_dE00"], "linear": pd["linear_mean_p95_dE00"],
                                                       "pass": pd["linear_mean_p95_dE00"] <= pd["legacy_mean_p95_dE00"]}}
    out.update(status="complete", pooled_across_seeds=pooled,
               regressions_development_across_seeds=sorted([a for a in across if a["mean_delta_mean_dE00"] > 0],
                                                           key=lambda a: -a["mean_delta_mean_dE00"]),
               scenes_with_any_seed_regression=sum(a["seeds_linear_worse"] > 0 for a in across),
               screening={"scope": "internal development screening only; all 116 development scenes as frozen "
                                   "(DR100 included in the pooled mean but separately reported as not validated); "
                                   "not acceptance, not app or model selection",
                          "criteria": criteria, "passed": all(c["pass"] for c in criteria.values())})
    _check_finite(out)
    fp = final_path()
    if fp.exists():
        D.need(json.loads(fp.read_text()) == json.loads(json.dumps(out, allow_nan=False)),
               f"Existing final report {fp} differs from recomputation; refusing")
        return fp
    D.write_once_json(fp, out)
    return fp
