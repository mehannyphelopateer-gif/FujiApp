"""First-scene saved-array response diagnostic, not a calibration or model fit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from fjlrg import load_fjlrg_linear

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
BASE = ROOT / "training/checkpoints/provia-neutral-wb-decode"
NAME = "DSCF0391.RAF"
MANIFEST = DIAG / "provia-neutral-wb-control-diagnosis-manifest.json"
REPORT = DIAG / "provia-neutral-wb-control-diagnosis.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, obj):
    if p.exists():
        raise ValueError("Refusing overwrite")
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def response(x, y):
    """Descriptive slope/intercept; never applied to images or exported as a fix."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    xm, ym = x.mean(), y.mean()
    var = np.mean((x - xm) ** 2)
    if var <= 1e-15:
        return {"measurable": False}
    slope = np.mean((x - xm) * (y - ym)) / var
    intercept = ym - slope * xm
    residual = y - (slope * x + intercept)
    edges = np.quantile(x, np.linspace(0, 1, 11))
    bins = []
    for i in range(10):
        mask = (x >= edges[i]) & (x <= edges[i + 1] if i == 9 else x < edges[i + 1])
        bins.append({"n": int(mask.sum()), "x_low": float(edges[i]), "x_high": float(edges[i + 1]),
                     "median_ratio": float(np.median(y[mask] / x[mask])) if mask.any() else None})
    return {"measurable": True, "slope": float(slope), "intercept": float(intercept),
            "RMSE": float(np.sqrt(np.mean(residual ** 2))), "p99_abs_residual": float(np.percentile(np.abs(residual), 99)),
            "intensity_deciles": bins}


def prepare():
    files = [DIAG / "provia-neutral-wb-decode-manifest.json", DIAG / "provia-neutral-wb-decode-stop.json",
             BASE / "decode-progress.json", Path(__file__).resolve(), ROOT / "training/fjlrg.py"]
    files += [BASE / f"{NAME}-{a}.fjlrg" for a in "ABCDE"]
    dump(MANIFEST, {"scope": "Existing first-scene research arrays and returned metadata only. No RAF/TIFF/canonical-target reads, decoding, model weights, training, calibration or selected changes.",
        "bindings": {str(p.relative_to(ROOT)): sha(p) for p in files}, "name": NAME,
        "method": "Exact B/D and C/E payload checks. Same all3-channel both-arm5–80% mask as failed camera screen. Per-channel E=aD+b descriptive moments and decile medians E/D, not correction coefficients or acceptance. Inspect returned pre_mul ratios. Compare A vs M D and C vs M E with saved rgb_cam, mask both arrays plus projected channels5–80%, descriptive only. Only firstscene; partial scene2 unused.",
        "interpretation": "No old screen revised or passed. Payload identity would invalidate cameraRGB instrumentation; nonzero intercept indicates affine response, not its cause. Non-identity slope may be processing effect; inspect static source before causality. Metadata match remains separate. Model phase stays blocked."})
    print("Diagnosis manifest SHA", sha(MANIFEST))


def run(expected):
    if sha(MANIFEST) != expected:
        raise ValueError("Manifest drift")
    for p in (MANIFEST, Path(__file__).resolve()):
        b = subprocess.check_output(["git", "show", "HEAD:" + str(p.relative_to(ROOT))], cwd=ROOT)
        if hashlib.sha256(b).hexdigest() != sha(p):
            raise ValueError("Not committed")
    m = json.loads(MANIFEST.read_text())
    for f, h in m["bindings"].items():
        if sha(ROOT / f) != h:
            raise ValueError("Bound artifact changed")
    manifest = json.loads((DIAG / "provia-neutral-wb-decode-manifest.json").read_text())
    scene = next(s for s in manifest["scenes"] if s["name"] == NAME)
    decode = json.loads((BASE / "decode-progress.json").read_text())
    meta = next(s for s in decode["scenes"] if s["name"] == NAME)["arms"]
    a = {k: load_fjlrg_linear(BASE / f"{NAME}-{k}.fjlrg")[0] for k in "ABCDE"}
    d, e = a["D"], a["E"]
    mask = np.all((d >= .05) & (d <= .8) & (e >= .05) & (e <= .8), axis=-1)
    fits = []
    for c in range(3):
        r = response(d[mask, c], e[mask, c])
        r["expected_source_gain_ratio"] = scene["expected_camera_ratio_RGB"][c]
        r["returned_pre_mul_ratio"] = meta["E"]["returned_metadata"]["pre_mul"][c] / meta["D"]["returned_metadata"]["pre_mul"][c]
        fits.append(r)
    matrix_links = {}
    for arm, cam in (("A", "D"), ("C", "E")):
        matrix = np.asarray(meta[arm]["returned_metadata"]["rgb_cam"], dtype=np.float64)[:, :3]
        projected = a[cam].astype(np.float64) @ matrix.T
        good = np.all((a[arm] >= .05) & (a[arm] <= .8) & (a[cam] >= .05) & (a[cam] <= .8) & (projected >= .05) & (projected <= .8), axis=-1)
        err = a[arm][good].astype(np.float64) - projected[good]
        identity = a[arm][good].astype(np.float64) - a[cam][good]
        matrix_links[arm + "_vs_" + cam] = {"coverage": float(good.mean()), "mean_abs_matrix_error": float(np.abs(err).mean()) if good.any() else None,
                                          "p99_abs_matrix_error": float(np.percentile(np.abs(err), 99)) if good.any() else None,
                                          "mean_abs_identity_error": float(np.abs(identity).mean()) if good.any() else None}
    payload = {k: (BASE / f"{NAME}-{k}.fjlrg").read_bytes()[32:] for k in "ABCDE"}
    dump(REPORT, {"scope": m["scope"], "manifest_sha256": expected, "name": NAME,
                  "camera_mask_coverage": float(mask.mean()), "channel_responses_RGB": fits,
                  "payload_identity": {"D_equals_B": payload["D"] == payload["B"], "E_equals_C": payload["E"] == payload["C"], "D_equals_A": payload["D"] == payload["A"]},
                  "matrix_links": matrix_links, "old_camera_screen": "FAILED, unchanged; no model phase", "limits": m["interpretation"]})
    print(json.dumps({"channels": [{k: r.get(k) for k in ("slope", "intercept", "RMSE", "p99_abs_residual", "expected_source_gain_ratio", "returned_pre_mul_ratio")} for r in fits], "matrix_links": matrix_links}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--manifest-sha")
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        if not args.manifest_sha:
            raise ValueError("Frozen manifest SHA required")
        run(args.manifest_sha)
