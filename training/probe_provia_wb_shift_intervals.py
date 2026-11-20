"""Exact interval test: is each source fine-tune group consistent with one constant per-channel gain?

Source metadata only. For every source-Auto (WB mode 0) row of the frozen 583-scene
inventory, x = source 0xf00d Auto level (zero-shift candidate) and y = source 0xf00e
as-shot level (fine-tuned), for R and B separately. G must be 302 in both. No target,
image, model or RAW pixel is read; no gain is inferred from targets.

For each fine-tune group (exact raw pair) and channel, six preregistered quantisation
variants each give one exact rational interval of gains g per row; a group is compatible
with a constant gain under a variant iff the exact intersection of its rows' intervals is
non-empty. Singletons are trivially compatible and are labelled non-evaluable.

  source_half_up : y = round_half_up(g*x)            -> [(y-1/2)/x, (y+1/2)/x)
  source_floor   : y = floor(g*x)                    -> [y/x, (y+1)/x)
  source_ceil    : y = ceil(g*x)                     -> ((y-1)/x, y/x]
  both_half_up   : x = rhu(X), y = rhu(g*X), X real  -> ((y-1/2)/(x+1/2), (y+1/2)/(x-1/2))
  both_floor     : x = floor(X), y = floor(g*X)      -> (y/(x+1), (y+1)/x)
  both_ceil      : x = ceil(X), y = ceil(g*X)        -> ((y-1)/x, y/(x-1)), needs x > 1

The both-quantised variants matter: exact-source failure alone does not rule out a
constant gain applied to an unrounded internal value. Intervals are never loosened after
results. This is diagnostic only: no variant or model is selected automatically, and no
manual-WB inverse policy follows from it. The hypothesis was informed by prior F24
observations; there is no independent camera or app qualification here.
"""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
INVENTORY = DIAG / "provia-v2-source-inventory.json"
POOL = DIAG / "provia-color-run-manifest.json"
POOL_SHA = "d6c04efd70df63aa38a9f2f08e0b291f858d5085a9670ffb7d3f1b3f4c6cf254"
MANIFEST = DIAG / "provia-wb-shift-interval-manifest.json"
REPORT = DIAG / "provia-wb-shift-interval-report.json"
OWN_CODE = ["training/probe_provia_wb_shift_intervals.py", "training/test_provia_wb_shift_intervals.py"]
EXPECTED_AUTO_ROWS = 293
SOURCE_G = 302
FINE_TUNE_STEP = 20
CHANNELS = {"R": 0, "B": 2}
HALF = Fraction(1, 2)
VARIANTS = ("source_half_up", "source_floor", "source_ceil", "both_half_up", "both_floor", "both_ceil")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    with open(p, "x") as f:  # write-once
        f.write(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def committed(path):
    blob = subprocess.check_output(["git", "show", "HEAD:" + str(Path(path).resolve().relative_to(ROOT))], cwd=ROOT)
    return hashlib.sha256(blob).hexdigest() == sha(path)


# --- Exact interval arithmetic (pure, unit-tested) ------------------------------

class Interval:
    """Real interval with exact Fraction endpoints and explicit closure flags."""
    __slots__ = ("lo", "lo_closed", "hi", "hi_closed")

    def __init__(self, lo, lo_closed, hi, hi_closed):
        self.lo, self.lo_closed, self.hi, self.hi_closed = Fraction(lo), bool(lo_closed), Fraction(hi), bool(hi_closed)

    def is_empty(self):
        return self.lo > self.hi or (self.lo == self.hi and not (self.lo_closed and self.hi_closed))

    def contains(self, g):
        g = Fraction(g)
        above = g > self.lo or (g == self.lo and self.lo_closed)
        below = g < self.hi or (g == self.hi and self.hi_closed)
        return above and below

    def as_tuple(self):
        return (self.lo, self.lo_closed, self.hi, self.hi_closed)


def check_level(v, name):
    require(type(v) is int and v > 0, f"{name} must be a positive int (not bool/float)")
    return v


def row_interval(variant, x, y):
    """Exact gain interval for one (x, y) source pair under a quantisation variant."""
    x, y = check_level(x, "x"), check_level(y, "y")
    if variant == "source_half_up":
        return Interval((y - HALF) / x, True, (y + HALF) / x, False)
    if variant == "source_floor":
        return Interval(Fraction(y, x), True, Fraction(y + 1, x), False)
    if variant == "source_ceil":
        return Interval(Fraction(y - 1, x), False, Fraction(y, x), True)
    if variant == "both_half_up":
        return Interval((y - HALF) / (x + HALF), False, (y + HALF) / (x - HALF), False)
    if variant == "both_floor":
        return Interval(Fraction(y, x + 1), False, Fraction(y + 1, x), False)
    if variant == "both_ceil":
        require(x > 1, "both_ceil requires x > 1")
        return Interval(Fraction(y - 1, x), False, Fraction(y, x - 1), False)
    raise ValueError(f"Unknown variant {variant!r}")


def intersect(intervals):
    """Exact intersection. At a tied bound the result is closed only if every tied interval is closed there.

    Returns (Interval, lower_binding_indices, upper_binding_indices).
    """
    intervals = list(intervals)
    require(len(intervals) > 0, "Empty interval list")
    lo = max(iv.lo for iv in intervals)
    hi = min(iv.hi for iv in intervals)
    lo_idx = [i for i, iv in enumerate(intervals) if iv.lo == lo]
    hi_idx = [i for i, iv in enumerate(intervals) if iv.hi == hi]
    lo_closed = all(intervals[i].lo_closed for i in lo_idx)
    hi_closed = all(intervals[i].hi_closed for i in hi_idx)
    return Interval(lo, lo_closed, hi, hi_closed), lo_idx, hi_idx


def representative(iv):
    """Midpoint of a non-empty interval, or its unique closed point; None if empty."""
    if iv.is_empty():
        return None
    return iv.lo if iv.lo == iv.hi else (iv.lo + iv.hi) / 2


def rational(f):
    f = Fraction(f)
    return {"num": f.numerator, "den": f.denominator, "float": float(f)}


def analyse_channel(variant, rows):
    """rows: list of (name, x, y). Exact compatibility record for one group/channel/variant."""
    intervals = [row_interval(variant, x, y) for _, x, y in rows]
    iv, lo_idx, hi_idx = intersect(intervals)
    empty = iv.is_empty()
    rep = representative(iv)
    out = {"variant": variant, "n": len(rows), "evaluable": len(rows) > 1,
           "compatible": not empty,
           "lo": rational(iv.lo), "lo_closed": iv.lo_closed, "hi": rational(iv.hi), "hi_closed": iv.hi_closed,
           "width_or_negative_gap": rational(iv.hi - iv.lo),
           "lower_binding_rows": [rows[i][0] for i in lo_idx], "upper_binding_rows": [rows[i][0] for i in hi_idx],
           "representative_gain": rational(rep) if rep is not None else None}
    if rep is not None:
        out["row_gain_interval_slack"] = [
            {"name": rows[i][0], "slack": rational(min(rep - interval.lo, interval.hi - rep)), "contains": interval.contains(rep)}
            for i, interval in enumerate(intervals)]
        require(all(s["contains"] for s in out["row_gain_interval_slack"]), "Representative outside a row interval")
    return out


def pearson(a, b):
    """Descriptive Pearson r; None when degenerate (n < 3 or zero variance)."""
    if len(a) < 3:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    da, db = [v - ma for v in a], [v - mb for v in b]
    sa, sb = sum(v * v for v in da), sum(v * v for v in db)
    if sa == 0 or sb == 0:
        return None
    return sum(p * q for p, q in zip(da, db)) / (sa * sb) ** 0.5


def source_pairs(inventory_rows):
    """Validate and group source-Auto rows by exact fine-tune raw pair."""
    groups = {}
    for row in inventory_rows:
        if row["source_wb_mode"] != 0:
            continue
        auto, cam = row["auto_rgbg"], row["cam_mul"]
        require(isinstance(auto, list) and len(auto) == 4 and isinstance(cam, list) and len(cam) == 4, "WB arrays must have 4 values")
        require(auto[1] == auto[3] == cam[1] == SOURCE_G and cam[3] == 0, f"Source G must be {SOURCE_G} in both records: {row['name']}")
        for c in CHANNELS.values():
            check_level(auto[c], "auto level")
            check_level(cam[c], "as-shot level")
        fine = row["source_wb_fine_tune_raw"]
        require(isinstance(fine, list) and len(fine) == 2 and all(type(v) is int and v % FINE_TUNE_STEP == 0 for v in fine),
                f"Fine-tune must be two multiples of 20: {row['name']}")
        groups.setdefault(tuple(fine), []).append({"name": row["name"], "shoot": row["shoot"], "session": row.get("session"),
                                                   "auto": auto, "cam": cam})
    return groups


def analyse_groups(groups):
    out = []
    for fine in sorted(groups):
        members = sorted(groups[fine], key=lambda m: m["name"])
        record = {"fine_tune_raw": list(fine), "steps": [fine[0] // FINE_TUNE_STEP, fine[1] // FINE_TUNE_STEP],
                  "n": len(members), "evaluable": len(members) > 1,
                  "sessions": sorted({m["session"] for m in members if m["session"] is not None}),
                  "channels": {}}
        for label, c in CHANNELS.items():
            rows = [(m["name"], m["auto"][c], m["cam"][c]) for m in members]
            ratios = [y / x for _, x, y in rows]
            white_gr = [SOURCE_G / m["auto"][0] for m in members]
            white_gb = [SOURCE_G / m["auto"][2] for m in members]
            record["channels"][label] = {
                "variants": {v: analyse_channel(v, rows) for v in VARIANTS},
                "descriptive": {"ratio_median": statistics.median(ratios), "ratio_min": min(ratios), "ratio_max": max(ratios),
                                "pearson_ratio_vs_zero_white_G_over_R": pearson(ratios, white_gr),
                                "pearson_ratio_vs_zero_white_G_over_B": pearson(ratios, white_gb),
                                "note": "Descriptive only; integer quantisation can itself induce correlation. No causal claim."}}
        out.append(record)
    return out


def aggregate(group_records):
    multi = [g for g in group_records if g["evaluable"]]
    return {"multi_scene_groups": [g["fine_tune_raw"] for g in multi],
            "both_channels_compatible": {v: {str(g["fine_tune_raw"]): all(g["channels"][c]["variants"][v]["compatible"] for c in CHANNELS)
                                             for g in multi} for v in VARIANTS},
            "singletons_non_evaluable": [g["fine_tune_raw"] for g in group_records if not g["evaluable"]],
            "note": "No automatic best-variant or model selection; no manual-WB inverse policy."}


# --- CLI ----------------------------------------------------------------------

def prepare():
    require(sha(POOL) == POOL_SHA, "Frozen 583-source train allowlist changed")
    pool = {(s["name"], s["shoot"]) for s in json.loads(POOL.read_text())["scenes"]}
    inventory = json.loads(INVENTORY.read_text())
    require(inventory["completed"] and len(inventory["rows"]) == 583, "Source inventory incomplete")
    require({(r["name"], r["shoot"]) for r in inventory["rows"]} == pool, "Inventory differs from the train allowlist")
    groups = source_pairs(inventory["rows"])
    n_auto = sum(len(v) for v in groups.values())
    require(n_auto == EXPECTED_AUTO_ROWS, f"Expected {EXPECTED_AUTO_ROWS} source-Auto rows, found {n_auto}")
    files = OWN_CODE + [str(INVENTORY.relative_to(ROOT)), str(POOL.relative_to(ROOT))]
    dump(MANIFEST, {
        "scope": "Source-metadata-only exact interval test of constant per-channel fine-tune gains on source-Auto rows. No targets, images, models, RAW pixels or protected tiers. Diagnostic only.",
        "bindings": {f: sha(ROOT / f) for f in files},
        "inputs": "x = source 0xf00d auto_rgbg (zero-shift candidate), y = source 0xf00e cam_mul (fine-tuned); channels R and B; G = 302 required in both",
        "n_source_auto_rows": n_auto, "groups": {str(list(k)): len(v) for k, v in sorted(groups.items())},
        "variants": {"source_half_up": "[(y-1/2)/x, (y+1/2)/x)", "source_floor": "[y/x, (y+1)/x)", "source_ceil": "((y-1)/x, y/x]",
                     "both_half_up": "((y-1/2)/(x+1/2), (y+1/2)/(x-1/2))", "both_floor": "(y/(x+1), (y+1)/x)",
                     "both_ceil": "((y-1)/x, y/(x-1)), x > 1"},
        "rule": "Compatible iff the exact Fraction intersection of all row intervals in a group is non-empty, with closure flags respected. Singletons non-evaluable. Intervals never loosened after results.",
        "limits": "Hypothesis informed by prior F24 observations. No independent camera/app qualification. No variant or model selection, no manual-WB inverse policy, no runtime or Round 11 change.",
    })
    print("WB shift interval manifest SHA", sha(MANIFEST))


def run(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    m = json.loads(MANIFEST.read_text())
    for p in [MANIFEST] + [ROOT / f for f in OWN_CODE]:
        require(committed(p), "Code/manifest must be committed before analysis: " + str(p))
    for f, h in m["bindings"].items():
        require(sha(ROOT / f) == h, "Bound file changed: " + f)
    groups = source_pairs(json.loads(INVENTORY.read_text())["rows"])
    require(sum(len(v) for v in groups.values()) == m["n_source_auto_rows"], "Source-Auto row count changed")
    records = analyse_groups(groups)
    dump(REPORT, {"scope": m["scope"], "manifest_sha256": expected, "groups": records, "aggregate": aggregate(records),
                  "limits": m["limits"]})
    print(json.dumps(aggregate(records), indent=2))


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
