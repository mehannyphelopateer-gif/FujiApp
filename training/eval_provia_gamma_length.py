"""One-scene gamma-array ABI probe; no target/model/protected-tier access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from fjlrg import load_fjlrg_linear
from eval_provia_neutral_wb import ratio_screen
from diagnose_provia_wb_decode_control import response

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
MANIFEST = DIAG / "provia-gamma-length-probe-manifest.json"
REPORT = DIAG / "provia-gamma-length-probe.json"
OUT = ROOT / "training/checkpoints/provia-gamma-length-probe"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, data):
    require(not p.exists(), "Refusing overwrite")
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def libraw_forward_table(power=.45, toe=4.5):
    """Independent numerical evaluation of pinned 0.22.1 gamma_curve mode2.

    This follows its 48-step toe/power continuity solve, fixed full-scale
    white65536, then positive ushort truncation. No constants fit to pixels.
    Reference: LibRaw0.22.1 src/utils/curves.cpp gamma_curve (LGPL2.1/CDDL).
    """
    lo, hi = (0., 1.) if toe >= 1 else (1., 0.)
    pivot = 0.
    if toe and (toe - 1) * (power - 1) <= 0:
        for _ in range(48):
            pivot = (lo + hi) / 2
            test = ((pivot / toe) ** (-power) - 1) / power - 1 / pivot > -1
            if test:
                hi = pivot
            else:
                lo = pivot
    join = pivot / toe
    shoulder = pivot * (1 / power - 1)
    x = np.arange(65536, dtype=np.float64) / 65536.
    curve = np.where(x < join, x * toe, x ** power * (1 + shoulder) - shoulder)
    return np.clip(np.floor(65536 * curve), 0, 65535).astype(np.uint16), {"power": power, "toe": toe, "join": join, "shoulder": shoulder, "white": 65536}


def prepare():
    old_m_path = DIAG / "provia-neutral-wb-decode-manifest.json"
    old_m = json.loads(old_m_path.read_text())
    scene = dict(old_m["scenes"][0])
    require(scene["name"] == "DSCF0391.RAF", "Wrong sentinel")
    progress = ROOT / old_m["output_dir"] / "decode-progress.json"
    recorded = json.loads(progress.read_text())["scenes"][0]
    require(recorded["name"] == scene["name"], "Wrong old decode")
    scene["old_camera_D_sha256"] = recorded["arms"]["D"]["sha256"]
    scene["old_camera_D_declared_options_sha256"] = recorded["arms"]["D"]["declared_options_sha256"]
    base = recorded["arms"]["D"]["declared_options"]
    require(base["gamm"] == [1, 1] and base["outputColor"] == 0, "Wrong old gamma/camera space")
    tool_path = DIAG / "provia-gamma-wrapper-toolchain.json"
    tools = json.loads(tool_path.read_text())
    files = [old_m_path, progress, tool_path, DIAG / "provia-neutral-wb-decode-stop.json", ROOT / "scripts/probe-provia-gamma-length.mjs", Path(__file__).resolve(),
             ROOT / "training/eval_provia_neutral_wb.py", ROOT / "training/diagnose_provia_wb_decode_control.py", ROOT / "training/fjlrg.py",
             ROOT / "node_modules/libraw-wasm/package.json", ROOT / tools["local_source"]]
    files += [ROOT / x["path"] for x in tools["build_curve_sources"] + tools["build_sources"]]
    files += [ROOT / "node_modules/libraw-wasm/dist" / f for f in ("index.js", "worker.js", "libraw.js", "libraw.wasm")]
    files += [ROOT / old_m["output_dir"] / f"{scene['name']}-{a}.fjlrg" for a in "DE"]
    require(tools["npm_matches_installed_wasm"], "Installed binary is not verified npm release")
    dump(MANIFEST, {"scope": "Single approved train RAF DSCF0391, outputColor0, gamma option-length behavioural test. No targets, models, training, app/canonical edits, bulk conversion or protected-tier media.",
        "bindings": {str(p.relative_to(ROOT)): sha(p) for p in files}, "base_options": base, "scenes": [scene],
        "output_dir": "training/checkpoints/provia-gamma-length-probe",
        "arms": {"D2": "Fresh old as-shot gamm[1,1], exact old camera D options/hash required", "D6": "Same as-shot, only gamm=[1,1,0,0,0,0]", "E6": "Same six-gamma but source-f00d WB"},
        "samples": "Native returned16bit cameraRGB grid, x/y offset64 stride128, all3 channels, same fullresdimensions/coords acrossarms. BEFORE areaaverage. Ignored native numerics only; no native photos saved.",
        "encoding_gate": "Pinned LibRaw0.22.1 gamma_curve mode2(.45,4.5), fullscale65536 (noAutoBright, bright default1), ushort truncation; old2native ==table[sixnative] within mean2codes,p993codes, >=1000 channel samples with sixnative between.01 and.95. Exact matchfraction also reported. Never compare encoding-after-average to average-after-encoding.",
        "WB_gate": "Original working-resolution common-allchannels-botharms5–80% mask, >=2%frame, median relativeerr<=0.001 andp1/p99<=0.01, includinggreen1. Also per-channel descriptive E=aD+b slope within0.1%expected, absintercept<=.001. Not a model fit; no correction applied.",
        "difference_gate": "Sixgamma working output must differ from corresponding old2 output for both WB arms. Original WB-screen failure remains recorded, not relabelled.",
        "limits": "One-scene installed-binary behaviour only. Even confirmation does not justify changing R11 inputs: legacy app and training are consistent. No inference, model selection, migration or promise of Fuji parity."})
    print("Gamma probe manifest SHA", sha(MANIFEST))


def run(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    m = json.loads(MANIFEST.read_text())
    for p in (MANIFEST, Path(__file__).resolve()):
        b = subprocess.check_output(["git", "show", "HEAD:" + str(p.relative_to(ROOT))], cwd=ROOT)
        require(hashlib.sha256(b).hexdigest() == sha(p), "Not committed")
    for f, h in m["bindings"].items():
        require(sha(ROOT / f) == h, "Bound artifact drift: " + f)
    scene = m["scenes"][0]
    decode_path = OUT / "decode-complete.json"
    decoded = json.loads(decode_path.read_text())
    require(decoded["completed"] and decoded["manifest_sha256"] == expected and len(decoded["scenes"]) == 1, "Incomplete/wrong decode")
    arms = decoded["scenes"][0]["arms"]
    require(set(arms) == {"D2", "D6", "E6"}, "Arm coverage")
    arrays = {}
    for a in arms:
        p = OUT / f"{scene['name']}-{a}.fjlrg"
        require(ROOT / arms[a]["file"] == p and sha(p) == arms[a]["sha256"], "Changed decode")
        arrays[a] = load_fjlrg_linear(p)[0]
    require(arms["D2"]["sha256"] == scene["old_camera_D_sha256"], "Old decoder clone mismatch")
    require(arms["D2"]["declared_options_sha256"] == scene["old_camera_D_declared_options_sha256"], "Old options mismatch")
    d, e = arrays["D6"], arrays["E6"]
    screen = ratio_screen(d, e, scene["expected_camera_ratio_RGB"])
    mask = np.all((d >= .05) & (d <= .8) & (e >= .05) & (e <= .8), axis=-1)
    responses = []
    for c in range(3):
        r = response(d[mask, c], e[mask, c])
        r["expected_gain"] = scene["expected_camera_ratio_RGB"][c]
        r["gate_passed"] = bool(r["measurable"] and abs(r["slope"] / r["expected_gain"] - 1) <= .001 and abs(r["intercept"]) <= .001)
        responses.append(r)
    sam = {a: arms[a]["native_samples"] for a in arms}
    for a in sam:
        require((sam[a]["width"], sam[a]["height"], sam[a]["offset"], sam[a]["stride"]) == (sam["D2"]["width"], sam["D2"]["height"], 64, 128), "Native coordinates differ")
    old = np.asarray(sam["D2"]["data"], dtype=np.int64)
    linear = np.asarray(sam["D6"]["data"], dtype=np.int64)
    require(old.shape == linear.shape and old.size % 3 == 0, "Native sample coverage")
    lookup, constants = libraw_forward_table()
    keep = (linear >= .01 * 65535) & (linear <= .95 * 65535)
    err = np.abs(old[keep] - lookup[linear[keep]].astype(np.int64))
    encoding = {"n": int(keep.sum()), "mean_abs_codes": float(err.mean()) if err.size else None,
                "p99_abs_codes": float(np.percentile(err, 99)) if err.size else None,
                "max_abs_codes": int(err.max()) if err.size else None,
                "exact_fraction": float(np.mean(err == 0)) if err.size else None,
                "curve_constants": constants,
                "passed": bool(err.size >= 1000 and err.mean() <= 2 and np.percentile(err, 99) <= 3)}
    old_e = json.loads((DIAG / "provia-neutral-wb-decode-stop.json").read_text())["arm_hashes"]["E"]
    different = arms["D6"]["sha256"] != arms["D2"]["sha256"] and arms["E6"]["sha256"] != old_e
    out = {"scope": m["scope"], "manifest_sha256": expected, "decode_report_sha256": sha(decode_path),
           "old2_reproduced": True, "sixgamma_different": different, "WB_ratio_screen": screen,
           "WB_affine_response": responses, "native_default_curve_match": encoding,
           "behavioural_confirmation": bool(different and screen["passed"] and all(r["gate_passed"] for r in responses) and encoding["passed"]),
           "limits": m["limits"]}
    dump(REPORT, out)
    print(json.dumps({k: v for k, v in out.items() if k != "WB_affine_response"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--manifest-sha")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        require(bool(args.manifest_sha), "Manifest SHA required")
        run(args.manifest_sha)
