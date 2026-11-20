"""Prepare and independently verify the P1b full-native gamma-only F24 pilot.

P1b is a sampling revision of the stopped P1 pilot (training/verify_provia_v2_pilot.py),
not a pass of it: P1 stays recorded as stopped SHORT at DSCF0514 (sparse native grid
N < 1000, exact N not retained). P1b keeps every option, gate and tolerance and only
changes G1 sampling from a sparse native grid to ALL native channel samples, with
full-width native strips saved for an independent Python recomputation.

No target decoding, fitting, model loading or protected-tier access. This module never
reads RAF files; it reads only the exporter's research outputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

import fjlrg_v2 as V
from fjlrg import load_fjlrg_linear
from eval_provia_gamma_length import libraw_forward_table

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
OLD_MANIFEST = DIAG / "provia-v2-pilot-manifest.json"
MANIFEST = DIAG / "provia-v2-fullnative-pilot-manifest.json"
REPORT = DIAG / "provia-v2-fullnative-pilot-verification.json"
OUT = ROOT / "training/checkpoints/provia-input-v2-pilot-fullnative"
RUNNER = "scripts/export-provia-input-v2-fullnative.mjs"
VERIFIER = "training/verify_provia_v2_fullnative.py"
OLD_RUNNER = "scripts/export-provia-input-v2-pilot.mjs"
OLD_VERIFIER = "training/verify_provia_v2_pilot.py"
# Bound in P1 but not used by the P1b runner and likely to change with later app work.
UNBOUND_IN_P1B = {"src/lib/raw/rawService.ts"}

FULL_SCALE = 65535
ELIGIBLE_LO = math.ceil(0.01 * FULL_SCALE)    # 656
ELIGIBLE_HI = math.floor(0.95 * FULL_SCALE)   # 62258
assert (ELIGIBLE_LO, ELIGIBLE_HI) == (656, 62258)
MIN_N, MEAN_LIMIT, P99_LIMIT = 1000, 2.0, 3.0
STRIP_OFFSET, STRIP_STEP = 64, 128
FLOAT_TOL = 1e-9


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
    require(not p.exists(), "Refusing overwrite")
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def committed(path):
    """True if the working-tree file equals its committed HEAD version."""
    blob = subprocess.check_output(["git", "show", "HEAD:" + str(Path(path).resolve().relative_to(ROOT))], cwd=ROOT)
    return hashlib.sha256(blob).hexdigest() == sha(path)


# --- Pure histogram helpers (synthetic-testable) ------------------------------

def sparse_histogram(errors):
    """Integer absolute errors -> sorted sparse [[error, count], ...] with no zero counts."""
    e = np.asarray(errors)
    if e.size == 0:
        return []
    require(np.issubdtype(e.dtype, np.integer) and e.min() >= 0 and e.max() <= FULL_SCALE, "Errors must be integers in 0..65535")
    values, counts = np.unique(e, return_counts=True)
    return [[int(v), int(c)] for v, c in zip(values, counts)]


def validate_histogram(hist):
    """Check sparse histogram form: list of [error, count], strictly increasing errors, positive int counts."""
    require(isinstance(hist, list), "Histogram must be a list")
    previous = -1
    for item in hist:
        require(isinstance(item, list) and len(item) == 2, "Histogram entries must be [error, count]")
        error, count = item
        require(type(error) is int and type(count) is int, "Histogram values must be ints")
        require(previous < error <= FULL_SCALE, "Histogram errors must be strictly increasing within 0..65535")
        require(count > 0, "Histogram counts must be positive")
        previous = error
    return hist


def value_at_rank(hist, k):
    """Error at 0-based rank k of the implicit sorted error list: smallest error with cumulative count > k."""
    cumulative = 0
    for error, count in hist:
        cumulative += count
        if cumulative > k:
            return error
    raise ValueError("Rank beyond histogram size")


def histogram_stats(hist):
    """n, mean, p99 (NumPy default 'linear'), max, exact fraction from a sparse histogram.

    p99: q = 0.99 * (n - 1), lo = floor(q), hi = ceil(q),
    p99 = e[lo] + (e[hi] - e[lo]) * (q - lo), with e the sorted errors.
    Returns None for an empty histogram.
    """
    validate_histogram(hist)
    n = sum(c for _, c in hist)
    if n == 0:
        return None
    total = sum(e * c for e, c in hist)
    q = 0.99 * (n - 1)
    lo, hi = math.floor(q), math.ceil(q)
    a, b = value_at_rank(hist, lo), value_at_rank(hist, hi)
    exact = next((c for e, c in hist if e == 0), 0)
    return {"n": n, "mean_abs_codes": total / n, "p99_abs_codes": a + (b - a) * (q - lo),
            "max_abs_codes": hist[-1][0], "exact_fraction": exact / n}


def is_subset_histogram(sub, full):
    """Every bin of `sub` has count <= the same bin in `full` (missing bins count as 0)."""
    full_counts = dict((e, c) for e, c in full)
    return all(c <= full_counts.get(e, 0) for e, c in sub)


def transfer_errors(old, linear, table):
    """Errors |old - table[linear]| over eligible linear codes 656..62258 (any channel)."""
    old = np.asarray(old, dtype=np.int64)
    linear = np.asarray(linear, dtype=np.int64)
    require(old.shape == linear.shape, "Old/linear shape mismatch")
    mask = (linear >= ELIGIBLE_LO) & (linear <= ELIGIBLE_HI)
    return np.abs(old[mask] - table[linear[mask]].astype(np.int64)), mask


def strip_rows(native_height):
    return list(range(STRIP_OFFSET, native_height, STRIP_STEP))


# --- CLI ----------------------------------------------------------------------

def prepare():
    require(committed(OLD_MANIFEST), "Original P1 manifest must be committed and unchanged")
    old = json.loads(OLD_MANIFEST.read_text())
    table, _ = libraw_forward_table()
    white_preimage = np.flatnonzero(table == 65535).tolist()
    require(white_preimage == old["legacy_white_preimage_codes"] == [65534, 65535], "Pinned transfer white preimage changed")
    # Every file carried over from P1 must still have its P1 hash.
    carried = {f: h for f, h in old["bindings"].items() if f not in {OLD_RUNNER, OLD_VERIFIER} | UNBOUND_IN_P1B}
    for f, h in carried.items():
        require(sha(ROOT / f) == h, "File drifted since P1 freeze: " + f)
    bindings = dict(carried)
    bindings.update({f: sha(ROOT / f) for f in (RUNNER, VERIFIER, "scripts/lib/provia-transfer-histogram.mjs", str(OLD_MANIFEST.relative_to(ROOT)))})
    dump(MANIFEST, {
        "scope": "P1b: F24 gamma-only input-contract pilot with full-native G1 sampling. As-shot WB preserved. No target/media fitting, model selection, app replacement, W arm or protected-tier access.",
        "supersedes": {"manifest_sha256": sha(OLD_MANIFEST),
                       "status": "P1 stopped SHORT at DSCF0514 (sparse native grid N < 1000; exact N not retained). Not a pass; P1b is a sampling revision with unchanged tolerances."},
        "bindings": bindings, "scenes": old["scenes"], "base_options": old["base_options"],
        "wasm_sha256": old["wasm_sha256"], "legacy_white_preimage_codes": white_preimage,
        "output_dir": str(OUT.relative_to(ROOT)),
        "arms": old["arms"],
        "native_measurement": {"eligible_linear_codes_inclusive": [ELIGIBLE_LO, ELIGIBLE_HI], "samples": "ALL native channel samples",
                               "min_n": MIN_N, "mean_abs_codes_max": MEAN_LIMIT, "p99_abs_codes_max": P99_LIMIT,
                               "p99": "NumPy default linear: q=0.99*(n-1), e[floor(q)] + (e[ceil(q)]-e[floor(q)])*(q-floor(q))",
                               "histogram": "sparse [error, count], strictly increasing error in 0..65535"},
        "independent_strips": {"rows": f"native rows y = {STRIP_OFFSET} + {STRIP_STEP}k, all x, RGB", "arms": ["legacy", "G"],
                               "dtype": "uint16-le", "gate": "no minimum strip N; strip histogram must be a subset of the full histogram"},
        "gates": dict(old["gates"], G1="ALL native channel samples with G linear code 656..62258: old == pinned default LUT[G], n>=1000, mean<=2, p99<=3 codes"),
        "limits": old["limits"],
    })
    print("P1b full-native manifest SHA", sha(MANIFEST))


def verify_transfer(row, legacy, g, width, height, table):
    """Independent checks of the recorded full-native transfer and of the saved strips."""
    nt = row["native_transfer"]
    hist = validate_histogram(nt["histogram"])
    stats = histogram_stats(hist)
    require(stats is not None, "Empty full-native histogram")
    require(stats["n"] == nt["n"] and stats["max_abs_codes"] == nt["max_abs_codes"], "Recorded n/max differ from histogram")
    for key in ("mean_abs_codes", "p99_abs_codes", "exact_fraction"):
        require(type(nt[key]) in (int,float) and math.isfinite(nt[key]) and abs(stats[key] - nt[key]) <= FLOAT_TOL, f"Recorded {key} differs from histogram")
    per_channel = nt["per_channel_n"]
    require(isinstance(per_channel, list) and len(per_channel) == 3 and all(type(n) is int and 0 <= n <= width*height for n in per_channel) and sum(per_channel) == nt["n"], "Per-channel N inconsistent")
    require(type(nt["ineligible_count"]) is int and nt["ineligible_count"] >= 0, "Invalid ineligible count")
    require(nt["n"] + nt["ineligible_count"] == width * height * 3, "Eligible + ineligible != all native channel samples")
    passed = stats["n"] >= MIN_N and stats["mean_abs_codes"] <= MEAN_LIMIT and stats["p99_abs_codes"] <= P99_LIMIT
    require(passed and nt["passed"] is True, "Full-native G1 gate failed")

    rows = strip_rows(height)
    arrays = {}
    for arm, record in (("legacy", legacy), ("G", g)):
        s = record["strips"]
        p = ROOT / s["file"]
        require(p.parent == OUT / row["shoot"] and sha(p) == s["sha256"], f"{arm} strip file changed")
        require(s["width"] == width and s["channels"] == 3 and s["dtype"] == "uint16-le", f"{arm} strip layout")
        require(s["rows"] == rows, f"{arm} strip rows differ from y = 64 + 128k")
        data = np.fromfile(p, dtype="<u2")
        require(data.size == len(rows) * width * 3, f"{arm} strip size")
        arrays[arm] = data.reshape(len(rows), width, 3)
    errors, mask = transfer_errors(arrays["legacy"], arrays["G"], table)
    strip_hist = sparse_histogram(errors)
    require(is_subset_histogram(strip_hist, hist), "Strip histogram is not a subset of the full-native histogram")
    strip_stats = histogram_stats(strip_hist)
    if strip_stats is not None:
        require(strip_stats["n"] <= nt["n"] and strip_stats["max_abs_codes"] <= nt["max_abs_codes"], "Strip exceeds full-native totals")
        require(abs(float(np.percentile(errors, 99)) - strip_stats["p99_abs_codes"]) <= FLOAT_TOL, "Histogram p99 disagrees with np.percentile")
        per_channel_strip = mask.reshape(-1, 3).sum(axis=0).tolist()
        require(all(a <= b for a, b in zip(per_channel_strip, per_channel)), "Strip per-channel N exceeds full-native")
    return stats, strip_stats


def verify(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    m = json.loads(MANIFEST.read_text())
    for p in (MANIFEST, Path(__file__).resolve()):
        require(committed(p), "Uncommitted runner or manifest")
    for p, h in m["bindings"].items():
        require(sha(ROOT / p) == h, "Bound source changed: " + p)
    path = OUT / "complete.json"
    complete = json.loads(path.read_text())
    require(complete["completed"] and complete["manifest_sha256"] == expected, "Incomplete or stale pilot")
    rows = complete["scenes"]
    require(len(rows) == 24 and [r["name"] for r in rows] == [s["name"] for s in m["scenes"]], "Scene coverage/order differs")
    table, _ = libraw_forward_table()
    checked = []
    for scene, row in zip(m["scenes"], rows):
        arms = row["arms"]
        require(row["shoot"] == scene["shoot"] and set(arms) == {"legacy", "G", "G-repeat"}, "Wrong scene/arm coverage")
        for record in arms.values():
            p = ROOT / record["file"]
            require(p.parent == OUT / scene["shoot"] and sha(p) == record["sha256"], "Changed export file")
        legacy, g, repeat = arms["legacy"], arms["G"], arms["G-repeat"]
        require(legacy["sha256"] == scene["canonical_fjlrg_sha256"], "G0 legacy anchor differs")
        _, width, height = load_fjlrg_linear(ROOT / legacy["file"])
        decoded = [V.load_fjlrg_v2(ROOT / r["file"], expected_contract=V.CONTRACT_ASSHOT,
                                   expected_source_sha256=scene["raf_sha256"], expected_wasm_sha256=m["wasm_sha256"]) for r in (g, repeat)]
        require(all(d.width == width and d.height == height for d in decoded), "Working dimensions changed")
        require(g["sha256"] == repeat["sha256"] and np.array_equal(decoded[0].rgb_uint16, decoded[1].rgb_uint16), "G2 repeat differs")
        require(legacy["native_size"] == g["native_size"] == repeat["native_size"], "Native size differs")
        native_w, native_h = g["native_size"]
        require(legacy["nativeWhite"] == g["nativeLegacyWhitePreimage"], "G5 LUT-aware native white counts differ")
        full, strip = verify_transfer(row, legacy, g, native_w, native_h, table)
        checked.append({"name": scene["name"], "shoot": scene["shoot"], "native_size": [native_w, native_h], "working_size": [width, height],
                        "legacy_sha256": legacy["sha256"], "linear_v2_sha256": g["sha256"],
                        "linear_sidecar_sha256": sha(V.sidecar_path(ROOT / g["file"])),
                        "full_native_transfer": full, "per_channel_n": row["native_transfer"]["per_channel_n"],
                        "ineligible_count": row["native_transfer"]["ineligible_count"],
                        "independent_strip_transfer": strip,
                        "processed_native_white_counts": {"legacy_65535": legacy["nativeWhite"], "linear_ge65534": g["nativeLegacyWhitePreimage"],
                                                          "linear_65535": g.get("nativeWhite")},
                        "passed": True})
    dump(REPORT, {"passed": True, "manifest_sha256": expected, "complete_report_sha256": sha(path), "scenes": checked,
                  "supersedes": m["supersedes"], "scope": m["scope"], "limits": m["limits"],
                  "note": "Full-native G1 is computed by the browser exporter; Python re-derives its statistics from the histogram and independently recomputes the transfer on saved native strips."})
    print("Independent P1b full-native verification PASS", len(checked))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--manifest-sha")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        require(bool(args.manifest_sha), "Frozen manifest SHA required")
        verify(args.manifest_sha)
