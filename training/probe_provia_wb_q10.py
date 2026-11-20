"""Source-only Q10 lookup diagnostic for Fuji WB fine-tune (source metadata only).

Hypothesis (formed AFTER seeing the full-data intervals, so post hoc): for each fine-tune
axis setting, the camera's fine-tuned level is y = (x * n + 512) // 1024, with x the source
0xf00d Auto level, y the source 0xf00e as-shot level, and n an integer that depends only on
that channel's own axis setting (red on the red step, blue on the blue step).

What this module does, on the frozen 293 source-Auto rows only:
  * builds per-axis observations (red: step r with R levels; blue: step b with B levels),
    pooling across the other axis. That separability is an assumption, tested only where an
    axis value occurs under two or more other-axis values;
  * reports all six quantisation variants' pooled intersections (descriptive);
  * scans binary denominators 256..4096 for exact integer candidates (descriptive) and the
    coarsest tested grid that covers all qualified axes (not a minimum over all denominators);
  * for the primary fixed hypothesis D = 1024 under exact-source round-half-up: a full-data
    lookup (unique candidate only; qualified axes need >= 2 rows and >= 2 capture sessions),
    an algebraic resubstitution check (not evidence by itself), and leave-one-capture-session-out
    fold predictions using only the other sessions.

Unseen, ambiguous, empty or sparse axes are unmeasurable with a reason, never an identity
default or a pass. LOSO is a post-hoc diagnostic fold prediction, not independent
confirmation. No runtime policy, no manual-WB interpolation of unseen axis values, no
targets, images, models or protected tiers.
"""
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import subprocess

import probe_provia_wb_shift_intervals as I

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
INVENTORY = I.INVENTORY
POOL = I.POOL
POOL_SHA = I.POOL_SHA
MANIFEST = DIAG / "provia-wb-q10-manifest.json"
REPORT = DIAG / "provia-wb-q10-report.json"
OWN_CODE = ["training/probe_provia_wb_q10.py", "training/test_provia_wb_q10.py"]
INTERVAL_CODE = "training/probe_provia_wb_shift_intervals.py"
EXPECTED_AUTO_ROWS = 293
PRIMARY_VARIANT = "source_half_up"
PRIMARY_DEN = 1024
DENOMINATORS = (256, 512, 1024, 2048, 4096)
MIN_ROWS, MIN_SESSIONS = 2, 2
MAX_LISTED_CANDIDATES = 32
CHANNEL_AXIS = {"R": 0, "B": 1}       # which fine-tune component drives which channel
CHANNEL_INDEX = {"R": 0, "B": 2}      # index into RGBG levels

require = I.require
sha = I.sha
dump = I.dump
committed = I.committed


# --- Pure functions -------------------------------------------------------------

def integer_candidates(iv, den):
    """All integers n with n/den inside interval iv, respecting open/closed endpoints."""
    lo, hi = iv.lo * den, iv.hi * den
    n_min = math.ceil(lo)
    if n_min == lo and not iv.lo_closed:
        n_min += 1
    n_max = math.floor(hi)
    if n_max == hi and not iv.hi_closed:
        n_max -= 1
    return list(range(n_min, n_max + 1)) if n_min <= n_max else []


def candidate_summary(cands):
    return {"count": len(cands), "values": cands[:MAX_LISTED_CANDIDATES], "truncated": len(cands) > MAX_LISTED_CANDIDATES}


def predict_level(x, n, den=PRIMARY_DEN):
    """Integer round-half-up of x * n / den for positive ints (den even): (x*n + den//2) // den."""
    I.check_level(x, "x")
    require(type(n) is int and n > 0 and type(den) is int and den > 0 and den % 2 == 0, "n/den must be positive ints, den even")
    return (x * n + den // 2) // den


def axis_observations(rows):
    """rows: validated source-Auto rows -> {channel: {axis_step: [obs, ...]}}.

    Each obs = {name, session, x, y, other_step}. Requires a capture session on every row.
    """
    out = {c: {} for c in CHANNEL_AXIS}
    for row in rows:
        require(row.get("session") is not None, f"Capture session required: {row['name']}")
        steps = [v // I.FINE_TUNE_STEP for v in row["source_wb_fine_tune_raw"]]
        for c, axis in CHANNEL_AXIS.items():
            k = CHANNEL_INDEX[c]
            out[c].setdefault(steps[axis], []).append({
                "name": row["name"], "session": row["session"],
                "x": I.check_level(row["auto_rgbg"][k], "auto level"), "y": I.check_level(row["cam_mul"][k], "as-shot level"),
                "other_step": steps[1 - axis]})
    return out


def validated_auto_rows(inventory_rows):
    """Source-Auto rows passing the interval module's checks (G=302 both, positive ints, fine-tune multiples of 20)."""
    groups = I.source_pairs(inventory_rows)
    names = {m["name"] for members in groups.values() for m in members}
    return [r for r in inventory_rows if r["source_wb_mode"] == 0 and r["name"] in names]


def axis_lookup(obs, den=PRIMARY_DEN, variant=PRIMARY_VARIANT):
    """Pooled interval and integer candidates for one axis setting; qualification rules applied."""
    sessions = sorted({o["session"] for o in obs})
    if not obs:
        return {"status": "unseen", "n_rows": 0, "sessions": [], "n": None}
    iv, lo_idx, hi_idx = I.intersect([I.row_interval(variant, o["x"], o["y"]) for o in obs])
    cands = integer_candidates(iv, den) if not iv.is_empty() else []
    rec = {"n_rows": len(obs), "sessions": sessions, "interval_empty": iv.is_empty(),
           "lo": I.rational(iv.lo), "lo_closed": iv.lo_closed, "hi": I.rational(iv.hi), "hi_closed": iv.hi_closed,
           "candidates": candidate_summary(cands), "n": None}
    if len(obs) < MIN_ROWS or len(sessions) < MIN_SESSIONS:
        rec["status"] = "sparse"
    elif iv.is_empty():
        rec["status"] = "empty"
    elif len(cands) == 0:
        rec["status"] = "no_candidate"
    elif len(cands) > 1:
        rec["status"] = "ambiguous"
    else:
        rec["status"], rec["n"] = "qualified_unique", cands[0]
    return rec


def separability(obs):
    """Descriptive: per other-axis value, does each sub-group's interval contain the pooled D1024 candidate set?"""
    by_other = {}
    for o in obs:
        by_other.setdefault(o["other_step"], []).append(o)
    if len(by_other) < 2:
        return None
    return {str(k): axis_lookup(v)["candidates"] for k, v in sorted(by_other.items())}


def coarsest_covering_denominator(axis_records_by_den):
    """Smallest TESTED denominator whose grid has >= 1 candidate on every qualified-size axis."""
    for den in DENOMINATORS:
        recs = axis_records_by_den[den]
        eligible = [r for r in recs.values() if r["n_rows"] >= MIN_ROWS and len(r["sessions"]) >= MIN_SESSIONS]
        if eligible and all(r["candidates"]["count"] >= 1 for r in eligible):
            return den
    return None


def loso(obs_by_channel):
    """Leave-one-capture-session-out: each held session's rows predicted from other sessions only."""
    sessions = sorted({o["session"] for axes in obs_by_channel.values() for obs in axes.values() for o in obs})
    folds, rows = [], {}
    for held in sessions:
        fold = {"held_session": held, "channels": {}}
        for c, axes in obs_by_channel.items():
            fold["channels"][c] = {}
            for step, obs in sorted(axes.items()):
                held_obs = [o for o in obs if o["session"] == held]
                if not held_obs:
                    continue
                train = [o for o in obs if o["session"] != held]
                lookup = axis_lookup(train)
                measurable = lookup["status"] == "qualified_unique"
                preds = []
                for o in held_obs:
                    pred = predict_level(o["x"], lookup["n"]) if measurable else None
                    preds.append({"name": o["name"], "x": o["x"], "y": o["y"], "predicted": pred,
                                  "exact": (pred == o["y"]) if measurable else None,
                                  "code_error": (pred - o["y"]) if measurable else None})
                    r = rows.setdefault(o["name"], {"name": o["name"], "session": held})
                    r[c] = {"measurable": measurable, "exact": (pred == o["y"]) if measurable else None,
                            "reason": None if measurable else lookup["status"], "fold_n": lookup["n"]}
                fold["channels"][c][str(step)] = {"status": lookup["status"], "fold_n": lookup["n"], "train_rows": lookup["n_rows"],
                                                  "train_sessions": lookup["sessions"], "held": preds}
        folds.append(fold)
    for r in rows.values():
        both = all(r.get(c, {}).get("measurable") for c in CHANNEL_AXIS)
        r["both_channels_measurable"] = both
        r["both_channels_exact"] = all(r[c]["exact"] for c in CHANNEL_AXIS) if both else None
    return folds, sorted(rows.values(), key=lambda r: r["name"])


def loso_summary(folds, rows):
    per_axis = {}
    for f in folds:
        for c, axes in f["channels"].items():
            for step, rec in axes.items():
                s = per_axis.setdefault(f"{c}{step}", {"held_rows": 0, "measurable_rows": 0, "exact_rows": 0, "max_abs_code_error": 0,
                                                       "unmeasurable_reasons": {}})
                for p in rec["held"]:
                    s["held_rows"] += 1
                    if p["predicted"] is None:
                        s["unmeasurable_reasons"][rec["status"]] = s["unmeasurable_reasons"].get(rec["status"], 0) + 1
                    else:
                        s["measurable_rows"] += 1
                        s["exact_rows"] += int(p["exact"])
                        s["max_abs_code_error"] = max(s["max_abs_code_error"], abs(p["code_error"]))
    measurable = [r for r in rows if r["both_channels_measurable"]]
    return {"per_axis": per_axis, "rows_total": len(rows), "rows_both_measurable": len(measurable),
            "rows_both_exact": sum(1 for r in measurable if r["both_channels_exact"])}


def analyse(rows):
    obs = axis_observations(rows)
    by_den = {den: {c: {step: axis_lookup(o, den) for step, o in sorted(axes.items())} for c, axes in obs.items()} for den in DENOMINATORS}
    flat_by_den = {den: {f"{c}{s}": r for c, axes in recs.items() for s, r in axes.items()} for den, recs in by_den.items()}
    primary = by_den[PRIMARY_DEN]
    resub = {}
    for c, axes in primary.items():
        for step, rec in axes.items():
            if rec["status"] == "qualified_unique":
                ok = all(predict_level(o["x"], rec["n"]) == o["y"] for o in obs[c][step])
                resub[f"{c}{step}"] = {"n": rec["n"], "all_rows_reproduced": ok,
                                       "label": "algebraic resubstitution of the full-data interval; not evidence by itself"}
    folds, loso_rows = loso(obs)
    summary = loso_summary(folds, loso_rows)
    variants = {v: {c: {str(s): {"interval_empty": axis_lookup(o, PRIMARY_DEN, v)["interval_empty"],
                                 "d1024_candidates": axis_lookup(o, PRIMARY_DEN, v)["candidates"]}
                        for s, o in sorted(axes.items())} for c, axes in obs.items()} for v in I.VARIANTS}
    proposal = []
    for key, r in resub.items():
        loso_axis = summary["per_axis"].get(key)
        if (r["all_rows_reproduced"] and loso_axis and loso_axis["measurable_rows"] > 0
                and loso_axis["measurable_rows"] == loso_axis["held_rows"]
                and loso_axis["exact_rows"] == loso_axis["measurable_rows"]):
            proposal.append({"axis": key, "n": r["n"], "denominator": PRIMARY_DEN, "loso_measurable_rows": loso_axis["measurable_rows"]})
    return {
        "primary": {"variant": PRIMARY_VARIANT, "denominator": PRIMARY_DEN,
                    "full_data_lookup": {c: {str(s): r for s, r in axes.items()} for c, axes in primary.items()},
                    "resubstitution": resub},
        "denominator_scan": {str(den): {k: r["candidates"] for k, r in flat.items()} for den, flat in flat_by_den.items()},
        "coarsest_tested_covering_denominator": coarsest_covering_denominator(flat_by_den),
        "all_variants_descriptive": variants,
        "separability_descriptive": {c: {str(s): separability(o) for s, o in sorted(axes.items()) if separability(o) is not None} for c, axes in obs.items()},
        "loso": {"folds": folds, "rows": loso_rows, "summary": summary,
                 "label": "post-hoc diagnostic fold prediction; the Q10 family was formed after seeing the full-data intervals"},
        "future_transfer_proposal": {"axes": proposal,
                                     "note": "Axes with a unique full-data Q10 factor, exact resubstitution, complete LOSO coverage and every held row exact. Candidates for a later, separately frozen metadata-only test; not a runtime policy."},
    }


# --- CLI ----------------------------------------------------------------------

def prepare():
    require(sha(POOL) == POOL_SHA, "Frozen 583-source train allowlist changed")
    pool = {(s["name"], s["shoot"]) for s in json.loads(POOL.read_text())["scenes"]}
    inventory = json.loads(INVENTORY.read_text())
    require(inventory["completed"] and len(inventory["rows"]) == 583, "Source inventory incomplete")
    require({(r["name"], r["shoot"]) for r in inventory["rows"]} == pool, "Inventory differs from the train allowlist")
    rows = validated_auto_rows(inventory["rows"])
    require(len(rows) == EXPECTED_AUTO_ROWS, f"Expected {EXPECTED_AUTO_ROWS} source-Auto rows, found {len(rows)}")
    axis_observations(rows)  # requires sessions on every row
    files = OWN_CODE + [INTERVAL_CODE, str(INVENTORY.relative_to(ROOT)), str(POOL.relative_to(ROOT))]
    dump(MANIFEST, {
        "scope": "Source-metadata-only Q10 WB fine-tune lookup diagnostic on source-Auto rows. No targets, images, models, RAW pixels or protected tiers. No runtime policy.",
        "bindings": {f: sha(ROOT / f) for f in files}, "n_source_auto_rows": len(rows),
        "primary": {"variant": PRIMARY_VARIANT, "denominator": PRIMARY_DEN, "formula": "y = (x*n + 512) // 1024",
                    "qualification": f">= {MIN_ROWS} rows and >= {MIN_SESSIONS} capture sessions; unique integer candidate"},
        "denominators_descriptive": list(DENOMINATORS),
        "loso": "Leave one capture session out; fold factor from other sessions only; unseen/ambiguous/empty/sparse axes unmeasurable, never identity or pass.",
        "limits": "Q10 family formed after seeing full-data intervals (post hoc). Separability is an assumption tested only where an axis value repeats across other-axis values. No manual-WB interpolation of unseen axis values; no G/W/app/Round 11 change.",
    })
    print("Q10 manifest SHA", sha(MANIFEST))


def run(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    m = json.loads(MANIFEST.read_text())
    for p in [MANIFEST] + [ROOT / f for f in OWN_CODE]:
        require(committed(p), "Code/manifest must be committed before analysis: " + str(p))
    for f, h in m["bindings"].items():
        require(sha(ROOT / f) == h, "Bound file changed: " + f)
    rows = validated_auto_rows(json.loads(INVENTORY.read_text())["rows"])
    require(len(rows) == m["n_source_auto_rows"], "Source-Auto row count changed")
    result = analyse(rows)
    dump(REPORT, {"scope": m["scope"], "manifest_sha256": expected, **result, "limits": m["limits"]})
    print(json.dumps({"coarsest_tested_covering_denominator": result["coarsest_tested_covering_denominator"],
                      "resubstitution": result["primary"]["resubstitution"], "loso_summary": result["loso"]["summary"],
                      "future_transfer_proposal": result["future_transfer_proposal"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--manifest-sha")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        require(bool(args.manifest_sha), "Frozen manifest SHA required")
        run(args.manifest_sha)
