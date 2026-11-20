"""Prepare and independently verify the bulk-G legacy/transfer sentinels.

Sentinels are every 20th scene (indices 0, 20, ..., 580; exactly 30) of the frozen
583-scene bulk G manifest, in its declared order. Each runs the proven P1b three-arm
full-native protocol: legacy gamma2 v1 replay (G0: canonical v1 SHA), G gamma6, and an
independent G repeat (G2), with full-native transfer (G1) and LUT-aware whites (G5).
They must pass before bulk G data are released for training.

No target decoding, fitting, model loading, RAF reading or protected-tier access. This
module reads only manifests, reports and the exporter's research outputs. Pure histogram
helpers are reused from training/verify_provia_v2_fullnative.py (hash-bound), but none of
its output paths or globals.
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
from verify_provia_v2_fullnative import (ELIGIBLE_HI, ELIGIBLE_LO, FLOAT_TOL, MEAN_LIMIT, MIN_N, P99_LIMIT,
                                         histogram_stats, is_subset_histogram, sparse_histogram, strip_rows,
                                         transfer_errors, validate_histogram)

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
BULK_MANIFEST = DIAG / "provia-v2-train-export-manifest.json"
P1B_MANIFEST = DIAG / "provia-v2-fullnative-pilot-manifest.json"
POOL = DIAG / "provia-color-run-manifest.json"
POOL_SHA = "d6c04efd70df63aa38a9f2f08e0b291f858d5085a9670ffb7d3f1b3f4c6cf254"
QUALIFICATION = DIAG / "provia-v2-source-qualification.json"
MANIFEST = DIAG / "provia-v2-train-sentinel-manifest.json"
REPORT = DIAG / "provia-v2-train-sentinel-verification.json"
OUT = ROOT / "training/checkpoints/provia-input-v2-train-sentinels"
RUNNER = "scripts/export-provia-v2-train-sentinels.mjs"
VERIFIER = "training/verify_provia_v2_train_sentinels.py"
BULK_OWN_CODE = {"scripts/export-provia-input-v2-train.mjs", "training/verify_provia_v2_train_exports.py"}
EXTRA_BINDINGS = [RUNNER, VERIFIER, "training/verify_provia_v2_fullnative.py", "training/eval_provia_gamma_length.py",
                  "training/fjlrg.py", "training/fjlrg_v2.py", "scripts/lib/fjlrg-v2.mjs",
                  "scripts/lib/libraw-gamma-lookup.mjs", "scripts/lib/provia-transfer-histogram.mjs",
                  str(BULK_MANIFEST.relative_to(ROOT)), str(P1B_MANIFEST.relative_to(ROOT))]
STEP, N_BULK, N_SENTINELS = 20, 583, 30


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


def sentinel_indices(n=N_BULK, step=STEP):
    """Fixed prospective indices 0, 20, ..., in declared manifest order."""
    return list(range(0, n, step))


# --- CLI ----------------------------------------------------------------------

def prepare():
    require(committed(BULK_MANIFEST), "Bulk G manifest must be frozen and committed first")
    bulk = json.loads(BULK_MANIFEST.read_text())
    require(len(bulk["scenes"]) == N_BULK, "Bulk manifest must hold exactly 583 scenes")
    bulk_bindings = bulk["bindings"]
    pool_rel = str(POOL.relative_to(ROOT))
    require(bulk_bindings.get(pool_rel) == POOL_SHA == sha(POOL), "Train allowlist signature differs from the bulk preparer's")
    qual_rel = str(QUALIFICATION.relative_to(ROOT))
    require(qual_rel in bulk_bindings and sha(QUALIFICATION) == bulk_bindings[qual_rel], "Source qualification not bound by bulk manifest")
    qualification = json.loads(QUALIFICATION.read_text())
    require(qualification["passed"] and qualification["camera_model_counts"] == {"X100VI": N_BULK}, "Source camera qualification not passed")
    # Every carried bulk binding must still hold.
    carried = {f: h for f, h in bulk_bindings.items() if f not in BULK_OWN_CODE}
    for f, h in carried.items():
        require(sha(ROOT / f) == h, "File drifted since bulk freeze: " + f)

    pool = {s["name"]: s for s in json.loads(POOL.read_text())["scenes"]}
    p1b = json.loads(P1B_MANIFEST.read_text())
    require(p1b["native_measurement"]["eligible_linear_codes_inclusive"] == [ELIGIBLE_LO, ELIGIBLE_HI] == [656, 62258], "Eligible range changed")
    table, _ = libraw_forward_table()
    white_preimage = np.flatnonzero(table == 65535).tolist()
    require(white_preimage == p1b["legacy_white_preimage_codes"] == [65534, 65535], "Pinned transfer white preimage changed")
    base = bulk["base_options"]
    require(base.get("gamm") == [1, 1, 0, 0, 0, 0] and {k: v for k, v in base.items() if k != "gamm"} == {k: v for k, v in p1b["base_options"].items() if k != "gamm"},
            "Bulk base options differ from P1b apart from gamma")

    indices = sentinel_indices()
    require(len(indices) == N_SENTINELS and indices[-1] == 580, "Sentinel index rule")
    scenes = []
    for i in indices:
        row = bulk["scenes"][i]
        name = row["name"]
        require(name in pool and pool[name]["shoot"] == row["shoot"], "Sentinel outside the train allowlist: " + name)
        require(row["fjlrg_sha256"] == pool[name]["fjlrg_sha256"], "Bulk canonical v1 SHA differs from the allowlist: " + name)
        require(row["capture_dynamic_range"] == row["captureDynamicRange"], "Capture-DR alias mismatch: " + name)
        scenes.append({"bulk_index": i, "name": name, "shoot": row["shoot"], "raf_sha256": row["raf_sha256"],
                       "canonical_fjlrg_sha256": row["fjlrg_sha256"], "cam_mul": row["cam_mul"],
                       "source_wb_mode": row["source_wb_mode"], "source_wb_fine_tune_raw": row["source_wb_fine_tune_raw"],
                       "source_kelvin": row["source_kelvin"], "capture_dynamic_range": row["capture_dynamic_range"],
                       "captureDynamicRange": row["captureDynamicRange"]})
    require(len({s["shoot"] for s in scenes}) == len({s["name"] for s in scenes}) == N_SENTINELS, "Duplicate sentinel identity")

    bindings = dict(carried)
    bindings.update({f: sha(ROOT / f) for f in EXTRA_BINDINGS})
    dump(MANIFEST, {
        "scope": "Bulk G legacy/transfer sentinels: every 20th bulk scene (0..580, n=30), three-arm full-native P1b protocol. No targets, models, fitting, app change or protected tiers. Bulk G is not released for training until these pass.",
        "bulk_manifest_sha256": sha(BULK_MANIFEST), "sentinel_rule": "indices 0, 20, ..., 580 of the frozen bulk manifest's declared order",
        "bindings": bindings, "scenes": scenes, "base_options": base, "wasm_sha256": bulk["wasm_sha256"],
        "legacy_white_preimage_codes": white_preimage, "output_dir": str(OUT.relative_to(ROOT)),
        "arms": ["legacy replay, gamma2 and original v1", "G gamma6 v2-linear-asshot", "G-repeat independent gamma6 decode"],
        "native_measurement": p1b["native_measurement"], "independent_strips": p1b["independent_strips"],
        "gates": p1b["gates"],
        "limits": "Sentinels sample 30 of 583; they do not prove every frame. G only; source as-shot WB and fine-tune retained. No model receives these arrays.",
    })
    print("Bulk G sentinel manifest SHA", sha(MANIFEST))


def verify_transfer(row, legacy, g, width, height, table):
    """Re-derive the full-native statistics from the histogram and recompute strips independently."""
    nt = row["native_transfer"]
    hist = validate_histogram(nt["histogram"])
    stats = histogram_stats(hist)
    require(stats is not None, "Empty full-native histogram")
    require(stats["n"] == nt["n"] and stats["max_abs_codes"] == nt["max_abs_codes"], "Recorded n/max differ from histogram")
    for key in ("mean_abs_codes", "p99_abs_codes", "exact_fraction"):
        require(type(nt[key]) in (int, float) and math.isfinite(nt[key]) and abs(stats[key] - nt[key]) <= FLOAT_TOL, f"Recorded {key} differs from histogram")
    per_channel = nt["per_channel_n"]
    require(isinstance(per_channel, list) and len(per_channel) == 3 and all(type(n) is int and 0 <= n <= width * height for n in per_channel)
            and sum(per_channel) == nt["n"], "Per-channel N inconsistent")
    require(type(nt["ineligible_count"]) is int and nt["ineligible_count"] >= 0, "Invalid ineligible count")
    require(nt["n"] + nt["ineligible_count"] == width * height * 3, "Eligible + ineligible != all native channel samples")
    require(stats["n"] >= MIN_N and stats["mean_abs_codes"] <= MEAN_LIMIT and stats["p99_abs_codes"] <= P99_LIMIT and nt["passed"] is True,
            "Full-native G1 gate failed")
    rows = strip_rows(height)
    arrays = {}
    for arm, record in (("legacy", legacy), ("G", g)):
        s = record["strips"]
        p = ROOT / s["file"]
        require(p.parent == OUT / row["shoot"] and sha(p) == s["sha256"], f"{arm} strip file changed")
        require(s["width"] == width and s["channels"] == 3 and s["dtype"] == "uint16-le" and s["rows"] == rows, f"{arm} strip layout")
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
        require(all(a <= b for a, b in zip(mask.reshape(-1, 3).sum(axis=0).tolist(), per_channel)), "Strip per-channel N exceeds full-native")
    return stats, strip_stats


def verify(expected, check_only=False):
    require(sha(MANIFEST) == expected, "Manifest drift")
    m = json.loads(MANIFEST.read_text())
    for p in (MANIFEST, Path(__file__).resolve(), ROOT / RUNNER):
        require(committed(p), "Uncommitted runner, verifier or manifest")
    for p, h in m["bindings"].items():
        require(sha(ROOT / p) == h, "Bound source changed: " + p)
    require(sha(BULK_MANIFEST) == m["bulk_manifest_sha256"], "Bulk manifest changed since sentinel freeze")
    complete_path = OUT / "complete.json"
    complete = json.loads(complete_path.read_text())
    require(complete["completed"] and complete["manifest_sha256"] == expected, "Incomplete or stale sentinel run")
    rows = complete["scenes"]
    require(len(rows) == N_SENTINELS and [r["name"] for r in rows] == [s["name"] for s in m["scenes"]], "Sentinel coverage/order differs")
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
        for d in decoded:
            pr = d.provenance
            require(pr["source_wb_mode"] == scene["source_wb_mode"] and pr["source_wb_fine_tune_raw"] == scene["source_wb_fine_tune_raw"]
                    and pr["source_kelvin"] == scene["source_kelvin"] and pr["capture_dynamic_range"] == scene["capture_dynamic_range"],
                    "Source WB/DR provenance differs from manifest")
            require(pr["wb_source"]["levels_rgbg"] == scene["cam_mul"] and pr["declared_options"] == dict(m["base_options"]), "Source gains/options differ")
        require(legacy["native_size"] == g["native_size"] == repeat["native_size"], "Native size differs")
        native_w, native_h = g["native_size"]
        require(decoded[0].provenance["decoded_native_size"] == [native_w, native_h], "Native size provenance differs")
        require(legacy["nativeWhite"] == g["nativeLegacyWhitePreimage"], "G5 LUT-aware native white counts differ")
        full, strip = verify_transfer(row, legacy, g, native_w, native_h, table)
        checked.append({"bulk_index": scene["bulk_index"], "name": scene["name"], "shoot": scene["shoot"],
                        "native_size": [native_w, native_h], "working_size": [width, height],
                        "legacy_sha256": legacy["sha256"], "linear_v2_sha256": g["sha256"],
                        "linear_sidecar_sha256": sha(V.sidecar_path(ROOT / g["file"])),
                        "source_wb_mode": scene["source_wb_mode"], "full_native_transfer": full,
                        "per_channel_n": row["native_transfer"]["per_channel_n"], "ineligible_count": row["native_transfer"]["ineligible_count"],
                        "independent_strip_transfer": strip,
                        "processed_native_white_counts": {"legacy_65535": legacy["nativeWhite"], "linear_ge65534": g["nativeLegacyWhitePreimage"],
                                                          "linear_65535": g.get("nativeWhite")},
                        "passed": True})
    result = {"passed": True, "manifest_sha256": expected, "bulk_manifest_sha256": m["bulk_manifest_sha256"],
                  "complete_report_sha256": sha(complete_path), "scenes": checked, "scope": m["scope"], "limits": m["limits"],
                  "note": "Sentinel G containers are not required to equal later bulk outputs (generated separately); the bulk run's own canaries and verifier cover that."}
    if check_only:
        require(json.loads(REPORT.read_text()) == result, "Saved sentinel report differs from independently recomputed evidence")
    else:
        dump(REPORT, result)
    print("Independent bulk G sentinel verification PASS", len(checked))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "check"))
    parser.add_argument("--manifest-sha")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        require(bool(args.manifest_sha), "Frozen manifest SHA required")
        verify(args.manifest_sha, check_only=args.command == "check")
