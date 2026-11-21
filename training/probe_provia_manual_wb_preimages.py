"""Source-only identifiability of a candidate zero-fine-tune base for manual/preset captures.

For each manual/preset source capture in the frozen 583-scene train inventory, the camera's
as-shot level y (source 0xf00e cam_mul, R and B) is assumed, under the post-hoc Q10 hypothesis,
to be y = floor((x*n + 512) / 1024) of an unknown zero-fine-tune base level x, with n the
source-only Q10 factor for that channel's fine-tune axis. This module reports which captures
admit such an inverse and how many integer x satisfy it (unique / ambiguous / empty).

It is NOT a qualification of a manual input arm, a zero-shift manual WB claim or a runtime
policy: no target is compared (that is a separate future test). Source 0xf00d is never used as
a manual neutral base. Auto modes 0 (normal) and 1 (white priority) are reported separately and
are not manual. A zero fine-tune step uses n = 1024, labelled as an identity ASSUMPTION; any
non-zero step is allowed only if the source-only Q10 report qualified it uniquely with complete,
exact leave-one-session-out coverage. Unsupported axes stay unqualified: no extrapolation.
"""
import argparse
import json
from pathlib import Path

import probe_provia_wb_shift_intervals as I

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
Q10_REPORT = DIAG / "provia-wb-q10-report.json"
QUALIFICATION = DIAG / "provia-v2-source-qualification.json"
MANIFEST = DIAG / "provia-manual-wb-preimages-manifest.json"
REPORT = DIAG / "provia-manual-wb-preimages-report.json"
OWN_CODE = ["training/probe_provia_manual_wb_preimages.py", "training/test_provia_manual_wb_preimages.py"]
INTERVAL_CODE = "training/probe_provia_wb_shift_intervals.py"
AUTO_MODES = {0: "auto_normal", 1: "auto_white_priority"}
MANUAL_MODES = {256: "daylight", 512: "cloudy", 770: "white_fluorescent", 4080: "kelvin"}
CHANNELS = {"R": (0, 0), "B": (2, 1)}   # channel -> (RGBG index, fine-tune component)
IDENTITY_N, DEN, HALF_DEN = 1024, 1024, 512
U32_MAX = 0xFFFFFFFF

require, sha, dump, committed = I.require, I.sha, I.dump, I.committed


# --- Pure helpers ---------------------------------------------------------------

def _pos_int(v, name, upper=U32_MAX):
    require(type(v) is int and 0 < v <= upper, f"{name} must be a positive int <= {upper}")
    return v


def q10_forward(x, n):
    """y = floor((x*n + 512) / 1024) for positive ints."""
    return (_pos_int(x, "x") * _pos_int(n, "n") + HALF_DEN) // DEN


def q10_preimages(y, n):
    """All positive integers x with floor((x*n + 512) / 1024) == y. No candidate is chosen.

    1024*y - 512 <= x*n < 1024*y + 512  =>  x in [ceil((1024y - 512)/n), ceil((1024y + 512)/n) - 1].
    """
    _pos_int(y, "y")
    _pos_int(n, "n")
    lo = max(1, -(-(DEN * y - HALF_DEN) // n))
    hi = min(U32_MAX, -(-(DEN * y + HALF_DEN) // n) - 1)
    out = list(range(lo, hi + 1)) if lo <= hi else []
    require(all(q10_forward(x, n) == y for x in out), "Preimage self-check failed")
    return out


def classify_mode(mode):
    require(type(mode) is int, "WB mode must be an int")
    if mode in AUTO_MODES:
        return "auto", AUTO_MODES[mode]
    if mode in MANUAL_MODES:
        return "manual", MANUAL_MODES[mode]
    raise ValueError(f"Unknown source WB mode {mode}; refusing")


def qualified_factors(q10):
    """Axis key (e.g. 'R4') -> n, only for uniquely qualified axes with complete exact LOSO."""
    out = {}
    per_axis = q10["loso"]["summary"]["per_axis"]
    for a in q10["future_transfer_proposal"]["axes"]:
        loso = per_axis.get(a["axis"])
        if loso and loso["held_rows"] == loso["measurable_rows"] == loso["exact_rows"] > 0 and type(a["n"]) is int:
            _pos_int(a["n"], "qualified factor")
            require(a["axis"] not in out, "Duplicate qualified axis")
            out[a["axis"]] = a["n"]
    return out


def channel_inverse(y, step, channel, factors):
    if step == 0:
        n, basis = IDENTITY_N, "zero step: n=1024 identity ASSUMPTION"
    elif f"{channel}{step}" in factors:
        n, basis = factors[f"{channel}{step}"], "source-only Q10 qualified axis"
    else:
        return {"axis": f"{channel}{step}", "status": "unqualified", "n": None, "candidates": None, "basis": "no qualified factor; not extrapolated"}
    cands = q10_preimages(y, n)
    status = "unique" if len(cands) == 1 else "ambiguous" if cands else "empty"
    return {"axis": f"{channel}{step}", "status": status, "n": n, "candidates": cands, "basis": basis}


def analyse_row(row, factors):
    kind, label = classify_mode(row["source_wb_mode"])
    out = {"name": row["name"], "shoot": row["shoot"], "mode": row["source_wb_mode"], "mode_label": label, "kind": kind}
    if kind == "auto":
        return out
    cam = row["cam_mul"]
    require(isinstance(cam, list) and len(cam) == 4 and cam[1] == I.SOURCE_G and cam[3] == 0, f"Source as-shot must be [R,302,B,0]: {row['name']}")
    fine = row["source_wb_fine_tune_raw"]
    require(isinstance(fine, list) and len(fine) == 2 and all(type(v) is int and v % I.FINE_TUNE_STEP == 0 for v in fine),
            f"Fine-tune must be two multiples of 20: {row['name']}")
    steps = [v // I.FINE_TUNE_STEP for v in fine]
    out["fine_tune_steps"] = steps
    out["channels"] = {c: channel_inverse(_pos_int(cam[idx], "as-shot level"), steps[axis], c, factors) for c, (idx, axis) in CHANNELS.items()}
    st = [out["channels"][c]["status"] for c in CHANNELS]
    out["both_axes_zero_or_qualified"] = "unqualified" not in st
    out["both_unique"] = all(s == "unique" for s in st)
    return out


def summarise(rows):
    manual = [r for r in rows if r["kind"] == "manual"]
    per_mode = {}
    for r in rows:
        per_mode[r["mode_label"]] = per_mode.get(r["mode_label"], 0) + 1
    per_channel = {c: {} for c in CHANNELS}
    for r in manual:
        for c in CHANNELS:
            s = r["channels"][c]["status"]
            per_channel[c][s] = per_channel[c].get(s, 0) + 1
    unq = {}
    for r in manual:
        for c in CHANNELS:
            if r["channels"][c]["status"] == "unqualified":
                unq[r["channels"][c]["axis"]] = unq.get(r["channels"][c]["axis"], 0) + 1
    return {"rows": len(rows), "per_mode": per_mode, "manual_rows": len(manual),
            "manual_both_axes_zero_or_qualified": sum(r["both_axes_zero_or_qualified"] for r in manual),
            "manual_both_unique": sum(r["both_unique"] for r in manual),
            "per_channel_status": per_channel, "unqualified_axis_counts": dict(sorted(unq.items())),
            "note": "Identifiability of a candidate metadata inverse only; no target comparison, no manual input arm or runtime policy."}


# --- CLI ----------------------------------------------------------------------

def load_inventory():
    require(sha(I.POOL) == I.POOL_SHA, "Frozen 583-source train allowlist changed")
    pool = [(s["name"], s["shoot"]) for s in json.loads(I.POOL.read_text())["scenes"]]
    inv = json.loads(I.INVENTORY.read_text())
    rows = inv["rows"]
    keys = [(r["name"], r["shoot"]) for r in rows]
    require(inv["completed"] and len(rows) == 583 and len(set(keys)) == 583 and set(keys) == set(pool), "Inventory must cover exactly the 583 allowlisted scenes")
    require(len({r["name"] for r in rows}) == len({r["shoot"] for r in rows}) == 583, "Duplicate source name or shoot")
    return rows


def prepare():
    rows = load_inventory()
    qual = json.loads(QUALIFICATION.read_text())
    require(qual.get("passed") is True and qual.get("camera_model_counts") == {"X100VI": 583},
            "Source camera/WB qualification not passed")
    require(len(qual["rows"]) == 583
            and {(r["name"], r["shoot"]) for r in qual["rows"]}
                == {(r["name"], r["shoot"]) for r in rows},
            "Source qualification coverage differs")
    mode_counts = {}
    for r in rows:
        label = classify_mode(r["source_wb_mode"])[1]  # unknown modes refused before freezing
        mode_counts[label] = mode_counts.get(label, 0) + 1
    factors = qualified_factors(json.loads(Q10_REPORT.read_text()))
    files = OWN_CODE + [INTERVAL_CODE] + [str(p.relative_to(ROOT)) for p in (I.INVENTORY, I.POOL, Q10_REPORT, QUALIFICATION)]
    dump(MANIFEST, {"scope": __doc__.strip(), "bindings": {f: sha(ROOT / f) for f in files},
                    "allowlist_sha256": I.POOL_SHA, "qualified_factors": factors, "mode_counts": mode_counts})
    print("Manual WB preimage manifest SHA", sha(MANIFEST))


def report(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    for p in [MANIFEST] + [ROOT / f for f in OWN_CODE]:
        require(committed(p), f"{p} must be committed")
    m = json.loads(MANIFEST.read_text())
    for f, h in m["bindings"].items():
        require(sha(ROOT / f) == h, "Bound file changed: " + f)
    require(m["allowlist_sha256"] == I.POOL_SHA, "Allowlist binding differs")
    factors = qualified_factors(json.loads(Q10_REPORT.read_text()))
    require(factors == m["qualified_factors"], "Qualified factors changed")
    rows = [analyse_row(r, factors) for r in load_inventory()]
    dump(REPORT, {"scope": m["scope"], "manifest_sha256": expected, "qualified_factors": factors,
                  "summary": summarise(rows), "rows": sorted(rows, key=lambda r: r["name"])})
    print(json.dumps(summarise(rows), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "report"))
    ap.add_argument("--manifest-sha")
    a = ap.parse_args()
    if a.command == "prepare":
        prepare()
    else:
        require(bool(a.manifest_sha), "Frozen manifest SHA required")
        report(a.manifest_sha)
