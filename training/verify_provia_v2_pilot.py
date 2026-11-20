"""Prepare and independently verify a gamma-only F24 export pilot.

No target decoding, fitting, model loading or protected-tier access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import fjlrg_v2 as V
from fjlrg import load_fjlrg_linear
from eval_provia_gamma_length import libraw_forward_table

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
MANIFEST = DIAG / "provia-v2-pilot-manifest.json"
REPORT = DIAG / "provia-v2-pilot-verification.json"
OUT = ROOT / "training/checkpoints/provia-input-v2-pilot"


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


def prepare():
    cohort = json.loads((DIAG / "cuban-cohort-proposal.json").read_text())
    pool_path = DIAG / "provia-color-run-manifest.json"
    require(sha(pool_path) == "d6c04efd70df63aa38a9f2f08e0b291f858d5085a9670ffb7d3f1b3f4c6cf254", "Frozen train allowlist changed")
    pool = {r["name"]: r for r in json.loads(pool_path.read_text())["scenes"]}
    wb = {r["name"]: r for r in json.loads((DIAG / "cn-step1-wb-reference-audit.json").read_text())["scenes"]}
    fits = [s for s in cohort["scenes"] if s["role"] == "fit"]
    require(len(fits) == len({s["name"] for s in fits}) == 24, "F24 identity mismatch")
    require(len({s["shoot"] for s in fits}) == 24, "Pilot output folders require unique shoots")
    scenes = []
    for s in fits:
        n = s["name"]
        require(s["tier"] == "train" and n in pool and s["shoot"] == pool[n]["shoot"] == wb[n]["shoot"], "Non-allowlisted source")
        raw = wb[n]["source"]
        scenes.append({"name": n, "shoot": s["shoot"], "session": s["session"],
                       "raf_sha256": wb[n]["source_raf_sha256_at_probe"], "canonical_fjlrg_sha256": pool[n]["fjlrg_sha256"],
                       "cam_mul": wb[n]["libraw_cam_mul"], "source_wb_mode": raw["WhiteBalance"],
                       "source_wb_fine_tune_raw": raw["WBFineTune"], "source_kelvin": raw["ColorTemperature"],
                       "capture_dynamic_range": pool[n]["captureDynamicRange"]})
    canonical = json.loads((DIAG / "provia-neutral-wb-decode-manifest.json").read_text())["base_options"]
    require(canonical["useCameraWb"] is True and "userMul" not in canonical, "Not the canonical camera-WB options")
    files = ["scripts/export-provia-input-v2-pilot.mjs", "scripts/lib/fjlrg-v2.mjs", "scripts/lib/libraw-gamma-lookup.mjs",
             "training/verify_provia_v2_pilot.py", "training/fjlrg_v2.py", "training/fjlrg.py", "training/eval_provia_gamma_length.py",
             "training/diagnostics/cuban-cohort-proposal.json", "training/diagnostics/provia-color-run-manifest.json",
             "training/diagnostics/cn-step1-wb-reference-audit.json", "training/diagnostics/provia-gamma-length-verification.json",
             "training/diagnostics/provia-v2-codec-conformance.json", "src/lib/raw/rawService.ts", "node_modules/libraw-wasm/package.json"]
    files += ["node_modules/libraw-wasm/dist/" + f for f in ("index.js", "worker.js", "libraw.js", "libraw.wasm")]
    require(json.loads((DIAG / "provia-v2-codec-conformance.json").read_text())["passed"], "Synthetic codec gate not passed")
    table, _ = libraw_forward_table()
    white_preimage = np.flatnonzero(table == 65535).tolist()
    require(white_preimage == [65534, 65535], "Pinned transfer white preimage changed")
    dump(MANIFEST, {"scope": "F24 gamma-only input-contract pilot. As-shot WB preserved. No target/media fitting, model selection, app replacement, W arm or protected-tier access.",
        "bindings": {f: sha(ROOT / f) for f in files}, "scenes": scenes, "base_options": canonical,
        "wasm_sha256": sha(ROOT / "node_modules/libraw-wasm/dist/libraw.wasm"),
        "legacy_white_preimage_codes": white_preimage,
        "output_dir": "training/checkpoints/provia-input-v2-pilot",
        "arms": ["legacy replay, gamma2 and originalv1", "G gamma6 v2-linear-asshot", "G-repeat independent gamma6 decode"],
        "gates": {"G0": "Every legacy container byte-exact canonical SHA", "G1": "Every scene nativegrid(offset64,stride128), .01..95%linear channel mask, >=1000samples, old==pinneddefaultLUT[G] mean<=2codes,p99<=3codes",
                  "G2": "Two independent G decodes byte-exact; browser exporter -> Node codec -> independent Python codec exact uint16 pixels and source/decoder provenance",
                  "G4": "Synthetic v1/v2 cross-refusal and requested-contract guards passed before pilot",
                  "G5": "Post-conversion native whites: legacycode65535 count ==Glinearcode>=65534 count (LUT quantisation). Linear65535 saturation recorded separately; working white counts descriptive only"},
        "limits": "Gamma-only, still source as-shot fine-tune. Does not fix WB mismatch or prove better Fuji matching. Native is processed full-resolution RGB, not sensor mosaic. No model is fed these v2 arrays; old selected R11 remains on legacy input."})
    print("V2 pilot manifest SHA", sha(MANIFEST))


def verify(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    m = json.loads(MANIFEST.read_text())
    for p in (MANIFEST, Path(__file__).resolve()):
        b = subprocess.check_output(["git", "show", "HEAD:" + str(p.relative_to(ROOT))], cwd=ROOT)
        require(hashlib.sha256(b).hexdigest() == sha(p), "Uncommitted runner or manifest")
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
        require(set(arms) == {"legacy", "G", "G-repeat"}, "Wrong arm coverage")
        for arm, record in arms.items():
            p = ROOT / record["file"]
            require(p.parent == OUT / scene["shoot"] and sha(p) == record["sha256"], "Changed export file")
        legacy = arms["legacy"]
        require(legacy["sha256"] == scene["canonical_fjlrg_sha256"], "Legacy anchor differs")
        x, width, height = load_fjlrg_linear(ROOT / legacy["file"])
        g = []
        for arm in ("G", "G-repeat"):
            record = arms[arm]
            p = ROOT / record["file"]
            f = V.load_fjlrg_v2(p, expected_contract=V.CONTRACT_ASSHOT, expected_source_sha256=scene["raf_sha256"], expected_wasm_sha256=m["wasm_sha256"])
            require(f.width == width and f.height == height, "Working dimensions changed")
            g.append(f)
        require(arms["G"]["sha256"] == arms["G-repeat"]["sha256"] and np.array_equal(g[0].rgb_uint16, g[1].rgb_uint16), "Repeat decode differs")
        old = np.asarray(legacy["samples"], dtype=np.int64)
        linear = np.asarray(arms["G"]["samples"], dtype=np.int64)
        require(old.shape == linear.shape and legacy["native_size"] == arms["G"]["native_size"], "Native sampling geometry differs")
        mask = (linear >= .01 * 65535) & (linear <= .95 * 65535)
        errors = np.abs(old[mask] - table[linear[mask]].astype(np.int64))
        require(errors.size >= 1000 and errors.mean() <= 2 and np.percentile(errors, 99) <= 3, "Native transfer check failed")
        require(legacy["nativeWhite"] == arms["G"]["nativeLegacyWhitePreimage"], "Quantizer-aware white counts differ")
        checked.append({"name": scene["name"], "shoot": scene["shoot"], "legacy_sha256": legacy["sha256"], "linear_v2_sha256": arms["G"]["sha256"],
                        "linear_sidecar_sha256": sha(V.sidecar_path(ROOT / arms["G"]["file"])), "working_size": [width, height],
                        "native_transfer_samples": int(errors.size), "native_transfer_mean_codes": float(errors.mean()), "native_transfer_max_codes": int(errors.max()),
                        "native_linear_white_counts": arms["G"]["nativeWhite"], "native_legacy_white_counts": legacy["nativeWhite"], "passed": True})
    dump(REPORT, {"passed": True, "manifest_sha256": expected, "complete_report_sha256": sha(path), "scenes": checked,
                  "scope": m["scope"], "limits": m["limits"]})
    print("Independent Python pilot verification PASS", len(checked))


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
