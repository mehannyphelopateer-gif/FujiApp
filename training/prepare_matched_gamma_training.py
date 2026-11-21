"""Prepare a prospective, paired input-encoding experiment; no optimisation.

Metadata-only prepare freezes contracts/orders/config; preflight subsequently
reads only allowed train inputs and audited target caches and verifies hashes.
No original TIFF or RAF is decoded, and no protected evaluation tier is read.
"""
import argparse
from collections import Counter
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

import matched_gamma_data as D

G_MANIFEST = D.DIAG / "provia-v2-train-export-manifest.json"
G_SHA = "5655c9b3918c3428e741d02b31425ba2368d36812f1b2eee5a08621748a1af37"
G_REPORT = D.DIAG / "provia-v2-train-export-verification.json"
R11_LOG = D.RUNS / "r11-drcond-seed42.json"
PREFLIGHT = D.DIAG / "matched-gamma-training-preflight.json"
CODE = ["matched_gamma_data.py", "prepare_matched_gamma_training.py", "run_matched_gamma_training.py",
        "eval_matched_gamma_training.py", "test_matched_gamma_training.py", "test_fjlrg_v2.py", "model.py", "fjlrg.py", "fjlrg_v2.py",
        "provia_color_pilot.py", "color_metrics.py", "color_loss.py", "dataset.py", "cuban_contract.py", "cuban_contract_v2.py"]


def prepare():
    D.need(D.sha(D.POOL) == D.POOL_SHA and D.sha(G_MANIFEST) == G_SHA, "Source manifests changed")
    old = json.loads(D.POOL.read_text())
    g = json.loads(G_MANIFEST.read_text())
    verified = json.loads(G_REPORT.read_text())
    D.need(verified["passed"] is True and verified["manifest_sha256"] == G_SHA
           and len(verified["scenes"]) == 583, "Independent G verification incomplete")
    by_g = {r["name"]: r for r in verified["scenes"]}
    inventory = {r["name"]: r for r in g["scenes"]}
    D.need(set(by_g) == set(inventory) == {r["name"] for r in old["scenes"]}, "G/legacy coverage differs")
    config = json.loads(R11_LOG.read_text())["config"]
    D.need(config["batch_size"] == 1 and config["lr"] == 0.001 and config["epochs"] == 20
           and config["grid_tv_weight"] == 0 and config["use_hybrid"] and config["capture_dr_conditioning"],
           "Original recipe differs from proposed experiment")
    keys = ("grid_spatial", "grid_luma", "base_ch", "refinement_base_ch", "refinement_max_delta", "low_res", "luma_mode")
    mc = {k: config[k] for k in keys}
    mc["condition_channels"] = 1
    rows = []
    for prior in old["scenes"]:
        v, source = by_g[prior["name"]], inventory[prior["name"]]
        D.need(v["passed"] is True and v["shoot"] == source["shoot"] == prior["shoot"], "G identity/verification mismatch")
        D.need(v["source_raf_sha256"] == source["raf_sha256"], "G source binding differs")
        rows.append({**prior, "target_path": f"calibration-input/provia-color-target-cache/{prior['shoot']}.npy",
                     "inputs": {
                         "legacy": {"path": f"calibration-input/{prior['shoot']}/browser-phase3-linear.fjlrg", "sha256": prior["fjlrg_sha256"]},
                         "linear": {"path": v["file"], "sha256": v["sha256"], "sidecar_sha256": v["sidecar_sha256"],
                                    "source_raf_sha256": source["raf_sha256"], "wasm_sha256": g["wasm_sha256"]}}})
    fit = [r["name"] for r in rows if r["role"] == "fit"]
    paths = [D.TRAIN / f for f in CODE] + [D.POOL, G_MANIFEST, G_REPORT, R11_LOG]
    # Existing audit/helper dependencies are bound without reading their media.
    paths += [D.TRAIN / f for f in old["bindings"]["code"]]
    paths += [D.DIAG / f for f in old["bindings"]["evidence"]]
    m = {
        "scope": "Prospective matched gamma-only development experiment; no selection, app update or independent acceptance.",
        "pool_sha256": D.POOL_SHA,
        "bindings": {str(p.relative_to(D.ROOT)): D.sha(p) for p in paths},
        "model_config": mc,
        "training": {"seeds": list(D.SEEDS), "epochs": 20, "lr": 0.001, "eta_min": 1e-5, "batch_size": 1},
        "optimisation": "Fresh identical seeded models; Adam betas .9/.999 eps1e-8 wd0; linear L1 only; cosine20; no early stopping, AMP, augmentation, clipping, TV or warm start.",
        "orders": {str(seed): D.make_orders(fit, seed) for seed in D.SEEDS},
        "initial_state_sha256": {str(seed): D.state_digest(D.fresh_model(mc, seed).state_dict()) for seed in D.SEEDS},
        "scenes": rows,
        "counts": {role: dict(Counter(str(r["captureDynamicRange"]) for r in rows if r["role"] == role))
                   for role in ("fit", "development")},
        "evaluation": {"endpoint": 20, "all_six_arms": "Run all predeclared seeds regardless of first-seed outcome, unless integrity/non-finite failure stops the job.",
                       "screen": "Internal development screen only: paired seed-average mean dev DE00 improvement >=0.25, all three seeds lower dev mean than their matched legacy, seed-average mean per-scene dev p95 no worse. Every scene/group/regression reported; no automatic selection.",
                       "primary_set": "116 development scenes, whole-session disjoint from467 fit; not independent acceptance after previous research.",
                       "DR100": "All12 DR100 sources are development-only; no fit support. Report explicitly as out-of-support/extrapolation, never DR100 qualification.",
                       "guide_caveat": "The unchanged rec709 luma guide is computed from each arm's own input; linear G changes guide-bin occupancy. A negative result does not refute the corrected decode or prove absence of information.",
                       "R11": "Preserved selected checkpoint; not an endpoint control or unseen-dev comparator. New models never load its weights."},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "mps_available": torch.backends.mps.is_available()},
    }
    D.validate_manifest(m)
    D.write_once_json(D.MANIFEST, m)
    print("Matched gamma manifest SHA", D.sha(D.MANIFEST), flush=True)


def preflight(expected):
    m = D.load_manifest(expected)
    by_arm = {a: D.Pairs(m, a) for a in D.ARMS}
    count = 0
    for row in m["scenes"]:
        x0, y0, c0 = by_arm["legacy"].get(row["name"])
        x1, y1, c1 = by_arm["linear"].get(row["name"])
        D.need(torch.equal(y0, y1) and torch.equal(c0, c1) and x0.shape == x1.shape,
               "Targets/DR/geometry differ between arms")
        count += 1
        if count % 25 == 0:
            print("Preflight", count, "/ 583", flush=True)
    D.write_once_json(PREFLIGHT, {"passed": True, "manifest_sha256": expected, "pairs": count,
                                 "scope": "Both input contracts and hashes, finite float32 unit inputs and unchanged audited float32 targets, same targets/DR/geometry; no training."})
    print("Matched gamma pair preflight PASS", count, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "preflight"))
    p.add_argument("--manifest-sha")
    a = p.parse_args()
    if a.command == "prepare":
        prepare()
    else:
        D.need(bool(a.manifest_sha), "Frozen manifest SHA required")
        preflight(a.manifest_sha)
