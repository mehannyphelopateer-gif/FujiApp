"""F11 source-only WB substitution diagnostic; no fitting or model selection."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
MANIFEST = DIAG / "provia-neutral-wb-decode-manifest.json"
OUTPUT = ROOT / "training/checkpoints/provia-neutral-wb-decode"
CONTROLS = DIAG / "provia-neutral-wb-decode-controls.json"
RESULTS = DIAG / "provia-neutral-wb-frozen-model.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, data):
    require(not path.exists(), "Refusing overwrite: " + str(path))
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def ratio_screen(d, e, expected):
    require(d.shape == e.shape and d.ndim == 3 and d.shape[-1] == 3, "Camera RGB shape mismatch")
    mask = np.all((d >= .05) & (d <= .8) & (e >= .05) & (e <= .8), axis=-1)
    coverage = float(mask.mean())
    out = {"coverage": coverage, "measurable": coverage >= .02, "channels": [], "passed": False}
    if not out["measurable"]:
        return out
    observed = np.percentile(e[mask].astype(np.float64) / d[mask], [1, 50, 99], axis=0)
    for c in range(3):
        err = np.abs(observed[:, c] / expected[c] - 1)
        out["channels"].append({"expected": expected[c], "p01": observed[0, c], "median": observed[1, c], "p99": observed[2, c],
                               "median_relative_error": err[1], "max_central98_relative_error": float(max(err[0], err[2])),
                               "passed": bool(err[1] <= .001 and max(err[0], err[2]) <= .01)})
    out["passed"] = all(v["passed"] for v in out["channels"])
    return out


def prepare():
    source = DIAG / "provia-source-wb-probe.json"
    evidence = json.loads(source.read_text())
    pool_path = DIAG / "provia-color-run-manifest.json"
    require(sha(pool_path) == "d6c04efd70df63aa38a9f2f08e0b291f858d5085a9670ffb7d3f1b3f4c6cf254", "Original allowlist changed")
    pool = {s["name"]: s for s in json.loads(pool_path.read_text())["scenes"]}
    require(evidence["crosschecks_passed"] and evidence["screens"]["0xf00d"]["passed"], "No source metadata qualification")
    scenes = []
    for r in evidence["rows"]:
        if r["source_wb_mode"] != 0:
            continue
        n = r["name"]
        require(n in pool and pool[n]["shoot"] == r["shoot"], "Source outside approved training pool")
        by = {v["tag"]: v for v in r["records"]}
        # Only source-side coefficients, not the TIFF levels in candidate comparisons.
        as_shot = by["0xf00e"]["rgbg"][:3]
        auto = by["0xf00d"]["rgbg"][:3]
        require(as_shot[1] == min(as_shot) and auto[1] == min(auto), "Green is not smallest WB multiplier")
        expected = [(auto[c] / auto[1]) / (as_shot[c] / as_shot[1]) for c in range(3)]
        scenes.append({"name": n, "shoot": r["shoot"], "session": r["session"], "shift": r["shift_steps"],
                       "raf_sha256": r["raf_sha256"], "canonical_fjlrg_sha256": pool[n]["fjlrg_sha256"],
                       "as_shot_rgbg": as_shot + [as_shot[1]], "auto_rgbg": auto + [auto[1]],
                       "expected_camera_ratio_RGB": expected, "target_cache_sha256": pool[n]["npy_sha256"]})
    require(len(scenes) == 11, "Exact F11 coverage required")
    files = ["scripts/probe-provia-neutral-wb.mjs", "training/eval_provia_neutral_wb.py", "training/fjlrg.py",
             "training/provia_color_pilot.py", "training/run_provia_distill.py", "training/color_metrics.py", "training/color_loss.py", "training/model.py",
             "training/diagnostics/provia-source-wb-probe.json", "training/diagnostics/provia-color-run-manifest.json", "src/lib/raw/rawService.ts",
             "node_modules/libraw-wasm/package.json"] + ["node_modules/libraw-wasm/dist/" + n for n in ("index.js", "worker.js", "libraw.js", "libraw.wasm")]
    dump(MANIFEST, {"scope": "Approved F11 Auto-only source-coefficient substitution. Decoder controls before frozen-model inference; all in-sample, no fitting/selection/app/canonical changes or protected-tier media.",
        "bindings": {f: sha(ROOT / f) for f in files}, "scenes": scenes,
        "output_dir": "training/checkpoints/provia-neutral-wb-decode",
        "base_options": {"halfSize": False, "outputBps": 16, "outputColor": 1, "gamm": [1, 1], "noAutoBright": True,
                         "adjustMaximumThr": 0, "useCameraMatrix": 1, "useCameraWb": True, "useAutoWb": False,
                         "highlight": 0, "userQual": 3, "useFujiRotate": -1, "fbddNoiserd": 0},
        "arms": {"A": "Canonical camera-WB sRGB, must reproduce old FJLRGB16 byte-for-byte",
                 "B": "Explicit source as-shot userMul RGBG, camera/auto WB off; must equal A byte-for-byte",
                 "C": "Source f00d Auto userMul RGBG, otherwise B",
                 "D": "B but outputColor0 cameraRGB; not a model input, primaries flag unset",
                 "E": "C but outputColor0 cameraRGB; not a model input, primaries flag unset"},
        "decoder_screen": "Green must be smallest multiplier. D/E common mask: ALL3 channels in both arms >=0.05 and <=0.8; >=2% frame or unmeasurable. Per camera-channel E/D median within0.1%, p1/p99 within1% of source-gain relative ratio, including green1. Approximate numerical application screen after adaptive demosaic, not identity. Any A/B or camera-screen failure/unmeasurability blocks model inference. Never test raw channel ratios after non-diagonal camera-to-sRGB transform.",
        "A_B_failure_first_suspect": "useCameraMatrix1 may depend on cameraWB being enabled when an embedded matrix exists. Stop and inspect; never change canonicalA options after failure.",
        "B_model_evaluation": "B is byte-identical to A by gate, so reuse A metrics; no separate MPS forward or potentially spurious exact float-dictionary comparison.",
        "descriptive_model_summary": "Frozen R11 seed42 on A/B/C, same source as trained on A. Per-scene displayedDE mean/p95, signed Lab, model-free linear input means vs target, input per-channel clipped counts; mean/p95DE on common A/C-unclipped input mask. Median paired C-A, sign counts, session/shift means; no quality pass/fail or selection. Tone curve prevents interpreting input-mean distance as a WB correctness test.",
        "claims": ["Metadata match already measured on11 Auto sources, not universal semantics",
                   "A/B exact plus camera D/E application controls measure decoder behaviour; declared options and returned metadata are NOT internal parameter readback",
                   "Frozen-model effects are in-sample and may reflect learned compensation for old fine-tuned inputs; no unseen accuracy/parity claim"],
        "model_sha256": "1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c"})
    print("Manifest SHA", sha(MANIFEST))


def load(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    m = json.loads(MANIFEST.read_text())
    for path, h in m["bindings"].items():
        require(sha(ROOT / path) == h, "Bound file changed: " + path)
    for path in (MANIFEST, Path(__file__).resolve()):
        committed = subprocess.check_output(["git", "show", "HEAD:" + str(path.relative_to(ROOT))], cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == sha(path), "Uncommitted code/manifest")
    return m


def decoded(m):
    from fjlrg import load_fjlrg_linear
    p = OUTPUT / "decode-complete.json"
    d = json.loads(p.read_text())
    require(d["completed"] and d["manifest_sha256"] == sha(MANIFEST), "Decode incomplete or stale")
    require(len(d["scenes"]) == 11 and {r["name"] for r in d["scenes"]} == {r["name"] for r in m["scenes"]}, "Decode coverage mismatch")
    rows = {r["name"]: r for r in d["scenes"]}
    for s in m["scenes"]:
        arms = rows[s["name"]]["arms"]
        require(set(arms) == set("ABCDE"), "Arm coverage mismatch")
        arrays = {}
        for a in "ABCDE":
            expected = OUTPUT / f"{s['name']}-{a}.fjlrg"
            require(ROOT / arms[a]["file"] == expected and sha(expected) == arms[a]["sha256"], "Research decode changed")
            arrays[a], _, _ = load_fjlrg_linear(expected)
        require(arms["A"]["sha256"] == s["canonical_fjlrg_sha256"] == arms["B"]["sha256"], "Canonical/as-shot exact control failed")
        yield s, arrays, rows[s["name"]]


def controls(expected):
    m = load(expected)
    rows = []
    for s, a, meta in decoded(m):
        screen = ratio_screen(a["D"], a["E"], s["expected_camera_ratio_RGB"])
        rows.append({"name": s["name"], "session": s["session"], "shift": s["shift"], "A_equals_canonical": True,
                     "B_equals_A": True, "camera_ratio_screen": screen,
                     "input_clipped_fraction_by_channel": {k: np.mean(a[k] >= .999, axis=(0, 1)).tolist() for k in "ABCDE"}})
        print(s["name"], "camera screen", screen["passed"], "coverage", screen["coverage"], flush=True)
    out = {"manifest_sha256": expected, "decode_report_sha256": sha(OUTPUT / "decode-complete.json"),
           "rows": rows, "passed": all(r["camera_ratio_screen"]["passed"] for r in rows),
           "limits": "Approximate application screen; failure blocks model inference, not evidence against record meaning"}
    dump(CONTROLS, out)
    require(out["passed"], "Decoder application screen failed/unmeasurable; model inference blocked")


def score(expected):
    m = load(expected)
    gate = json.loads(CONTROLS.read_text())
    require(gate["passed"] and gate["manifest_sha256"] == expected and gate["decode_report_sha256"] == sha(OUTPUT / "decode-complete.json"), "No matching passed decoder gate")
    require(len(gate["rows"]) == 11 and {r["name"] for r in gate["rows"]} == {r["name"] for r in m["scenes"]}, "Decoder gate coverage mismatch")
    checked = {r["name"]: r for r in gate["rows"]}
    # Recompute every camera control, rather than trusting a saved passed=true.
    # This completes for all11 before even constructing the frozen model.
    for s, arrays, _ in decoded(m):
        fresh = ratio_screen(arrays["D"], arrays["E"], s["expected_camera_ratio_RGB"])
        require(fresh["passed"] and fresh == checked[s["name"]]["camera_ratio_screen"], "Camera screen stale, altered or failed")
    import torch
    import provia_color_pilot as P
    import run_provia_distill as D
    from color_metrics import ciede2000, linear_srgb_to_lab
    require(m["model_sha256"] == P.SOURCE_SHA, "Wrong model selection")
    data = P.load_manifest()
    pairs = P.Pairs(data)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, config = P.build(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for s, arrays, meta in decoded(m):
            x, y, cond = pairs.get(s["name"])
            target = y.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
            require(np.array_equal(x.squeeze(0).permute(1, 2, 0).numpy(), arrays["A"]), "Old/canonical loader mismatch")
            common = np.all((arrays["A"] > 0) & (arrays["A"] < .999) & (arrays["C"] > 0) & (arrays["C"] < .999), axis=-1)
            row = {"name": s["name"], "session": s["session"], "shift": s["shift"],
                   "common_input_unclipped_fraction": float(common.mean()), "arms": {}}
            for arm in "AC":
                t = torch.from_numpy(np.ascontiguousarray(arrays[arm].transpose(2, 0, 1))).unsqueeze(0)
                pred = P.fwd(model, config, t, cond, device)
                metrics = D.scene_metrics(pred, target)
                rendered = pred.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
                delta = ciede2000(linear_srgb_to_lab(rendered), linear_srgb_to_lab(target))
                metrics["common_input_unclipped_dE00"] = {"mean": float(delta[common].mean()), "p95": float(np.percentile(delta[common], 95))} if common.any() else None
                metrics["input_linear_channel_means"] = arrays[arm].mean(axis=(0, 1), dtype=np.float64).tolist()
                metrics["target_linear_channel_means"] = target.mean(axis=(0, 1)).tolist()
                row["arms"][arm] = metrics
            row["arms"]["B"] = row["arms"]["A"]
            row["B_metrics_reused_from_A_by_identical_input_hash"] = True
            row["C_minus_A_mean_dE00"] = row["arms"]["C"]["mean_dE00"] - row["arms"]["A"]["mean_dE00"]
            row["C_minus_A_p95_dE00"] = row["arms"]["C"]["p95_dE00"] - row["arms"]["A"]["p95_dE00"]
            rows.append(row)
            print(s["name"], "C-A meanDE", row["C_minus_A_mean_dE00"], flush=True)
    changes = [r["C_minus_A_mean_dE00"] for r in rows]
    groups = {}
    for axis in ("session", "shift"):
        groups[axis] = {str(k): {"n": len(v := [r for r in rows if str(r[axis]) == str(k)]),
                               "mean_C_minus_A": float(np.mean([r["C_minus_A_mean_dE00"] for r in v]))}
                        for k in sorted({str(r[axis]) for r in rows})}
    P.check_source()
    dump(RESULTS, {"scope": m["scope"], "manifest_sha256": expected, "decoder_gate_sha256": sha(CONTROLS), "model_sha256": m["model_sha256"],
                   "rows": rows, "median_paired_mean_dE00_change": float(np.median(changes)), "n_better": sum(v < 0 for v in changes),
                   "n_worse": sum(v > 0 for v in changes), "groups": groups, "limits": m["claims"], "quality_verdict": "DESCRIPTIVE ONLY; no selection or parity claim"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "controls", "score"))
    p.add_argument("--manifest-sha")
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        require(bool(args.manifest_sha), "Frozen manifest SHA required")
        (controls if args.command == "controls" else score)(args.manifest_sha)
