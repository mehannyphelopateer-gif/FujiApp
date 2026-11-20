"""Native-clipping influence diagnostic. One allowed training RAF, no models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from fjlrg import load_fjlrg_linear
from diagnose_provia_wb_decode_control import response

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
MANIFEST = DIAG / "provia-native-clipping-manifest.json"
REPORT = DIAG / "provia-native-clipping.json"
OUT = ROOT / "training/checkpoints/provia-native-clipping"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, value):
    require(not p.exists(), "Refusing overwrite")
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def dilate(mask, radius=2):
    padded = np.pad(mask, radius, constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            out |= padded[y:y + h, x:x + w]
    return out


def influence(x, y, flagged, expected):
    """Exact additive OLS slope-bias identity; not a correction learner."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    require(x.shape == y.shape == flagged.shape and x.size >= 2, "Invalid influence arrays")
    dx = x - x.mean()
    denominator = np.sum(dx ** 2)
    require(denominator > 1e-15, "No intensity variance")
    residual = y - expected * x
    bias = np.sum(dx * residual) / denominator
    groups = {}
    for label, mask in (("flagged", flagged), ("clean", ~flagged)):
        contrib = float(np.sum(dx[mask] * residual[mask]) / denominator)
        groups[label] = {"n": int(mask.sum()), "fraction": float(mask.mean()),
                         "global_variance_fraction": float(np.sum(dx[mask] ** 2) / denominator),
                         "contribution_to_global_slope_bias": contrib,
                         "relative_contribution_to_bias": contrib / bias if abs(bias) > 1e-12 else None,
                         "own_response": response(x[mask], y[mask]) if mask.sum() >= 2 else {"measurable": False}}
    require(abs(sum(g["contribution_to_global_slope_bias"] for g in groups.values()) - bias) < 1e-12, "Influence decomposition failed")
    return {"expected": expected, "global_slope_bias": float(bias), "global_response": response(x, y), "groups": groups}


def prepare():
    gamma_path = DIAG / "provia-gamma-length-probe-manifest.json"
    g = json.loads(gamma_path.read_text())
    prev_path = ROOT / g["output_dir"] / "decode-complete.json"
    prev = json.loads(prev_path.read_text())
    scene = dict(g["scenes"][0])
    require(scene["name"] == "DSCF0391.RAF", "Wrong approved sentinel")
    recorded = prev["scenes"][0]["arms"]
    scene["reproduce_arm_hashes"] = {a: recorded[a]["sha256"] for a in ("D6", "E6")}
    scene["reproduce_option_hashes"] = {a: recorded[a]["declared_options_sha256"] for a in ("D6", "E6")}
    files = [gamma_path, prev_path, DIAG / "provia-gamma-length-probe.json", Path(__file__).resolve(),
             ROOT / "scripts/probe-provia-native-clipping.mjs", ROOT / "training/diagnose_provia_wb_decode_control.py", ROOT / "training/fjlrg.py"]
    files += [ROOT / "node_modules/libraw-wasm/dist" / f for f in ("index.js", "worker.js", "libraw.js", "libraw.wasm")]
    dump(MANIFEST, {"scope": "Single approved train RAF DSCF0391, source camera-RGB instrumentation only. No target, model, fitting, app/canonical edit or protected-tier access. Original failed joint WB qualification remains unchanged.",
        "bindings": {str(p.relative_to(ROOT)): sha(p) for p in files}, "base_options": g["base_options"], "scenes": [scene],
        "output_dir": "training/checkpoints/provia-native-clipping",
        "instrumentation": "Reproduce D6/E6 full working buffers and options byte-exact. During the same area average retain per-block native minRGB, maxRGB, whiteRGB counts (code==65535), samplecount, native-radius6 white-influence count, uint16-le11channels. No full-resolution image saved.",
        "masks": "Original ALLchannels BOTHarms working5–80% mask unchanged. Native-white flag if ANYchannel in eitherarm has any fullywhite native sample in that block. Primary flag: any native pixel within native Chebyshev-radius6 of a post-demosaic white pixel in either arm contributes to that working block. Also split by in-block white and radius2-working-pixel white sensitivity variants. These are diagnostic masks, not changes to original qualification.",
        "analysis": "Per-channel exact OLS slope-bias decomposition using global centredx and u=y-expected*x: bias=sum(dx*u)/sum(dx^2); flagged+clean contributions must add exactly. Report group counts, globalvariance fraction, residual contribution, ownresponse/ratio deciles and clean coverage. No image correction coefficients used.",
        "hypothesis_screen": "Support native clipping as dominant cause only if, for red ANDblue: native-radius6-flagged contribution has same sign as global bias and explains>=80%absbias, strictclean >=2%frame and>=1000pixels, own slope within0.1%expected, interceptabs<=.001. Otherwise incomplete/not supported. Gate unchanged; no model scoring even if this diagnostic supports mechanism.",
        "limits": "One-scene causal-mechanism probe, not unseen validation or general WB qualification. Post-demosaic white footprints may miss pre-demosaic clipping that left no white samples; fixed native-radius6 and working-radius2 are diagnostic neighborhoods, not measured reconstruction kernels. No tuned thresholds or per-scene exception."})
    print("Native clipping manifest SHA", sha(MANIFEST))


def run(expected):
    require(sha(MANIFEST) == expected, "Manifest drift")
    m = json.loads(MANIFEST.read_text())
    for p in (MANIFEST, Path(__file__).resolve()):
        b = subprocess.check_output(["git", "show", "HEAD:" + str(p.relative_to(ROOT))], cwd=ROOT)
        require(hashlib.sha256(b).hexdigest() == sha(p), "Not committed")
    for f, h in m["bindings"].items():
        require(sha(ROOT / f) == h, "Bound source changed: " + f)
    decoded_path = OUT / "decode-complete.json"
    decode = json.loads(decoded_path.read_text())
    require(decode["completed"] and decode["manifest_sha256"] == expected and len(decode["scenes"]) == 1, "Wrong/incomplete decode")
    scene = m["scenes"][0]
    recorded = decode["scenes"][0]["arms"]
    require(set(recorded) == {"D6", "E6"}, "Arm coverage")
    frames, stats = {}, {}
    for a in recorded:
        p = OUT / f"{scene['name']}-{a}.fjlrg"
        require(ROOT / recorded[a]["file"] == p and sha(p) == recorded[a]["sha256"] == scene["reproduce_arm_hashes"][a], "Working pixels changed")
        require(recorded[a]["declared_options_sha256"] == scene["reproduce_option_hashes"][a], "Options changed")
        frames[a], w, h = load_fjlrg_linear(p)
        q = OUT / f"{scene['name']}-{a}.native-stats.u16"
        require(ROOT / recorded[a]["native_stats_file"] == q and sha(q) == recorded[a]["native_stats_sha256"], "Native statistics changed")
        values = np.fromfile(q, dtype="<u2")
        require(values.size == h * w * 11, "Native stats shape")
        values = values.reshape(h, w, 11)
        require(np.all(values[..., :3] <= values[..., 3:6]) and np.all(values[..., 9] > 0), "Native extrema invalid")
        require(np.all(values[..., 6:9] <= values[..., 9:10]), "Native white count exceeds sample count")
        require(np.array_equal(values[..., 6:9] > 0, values[..., 3:6] == 65535), "Native white/extrema inconsistent")
        require(np.all(values[..., 10] <= values[..., 9]), "Native dilated count exceeds block size")
        stats[a] = values
    d, e = frames["D6"], frames["E6"]
    mask = np.all((d >= .05) & (d <= .8) & (e >= .05) & (e <= .8), axis=-1)
    white = np.any(stats["D6"][..., 6:9] > 0, axis=-1) | np.any(stats["E6"][..., 6:9] > 0, axis=-1)
    flags = {"native_white_in_block": white, "native_white_working_radius2": dilate(white),
             "native_white_native_radius6": (stats["D6"][..., 10] > 0) | (stats["E6"][..., 10] > 0)}
    partitions, supported = {}, []
    for label, flag in flags.items():
        clean = mask & ~flag
        channels = [influence(d[mask, c], e[mask, c], flag[mask], scene["expected_camera_ratio_RGB"][c]) for c in range(3)]
        for c, r in enumerate(channels):
            f, cl = r["groups"]["flagged"], r["groups"]["clean"]
            own = cl["own_response"]
            r["hypothesis_supported"] = bool(float(clean.mean()) >= .02 and clean.sum() >= 1000 and own.get("measurable")
                and abs(own["slope"] / r["expected"] - 1) <= .001 and abs(own["intercept"]) <= .001
                and f["relative_contribution_to_bias"] is not None and f["relative_contribution_to_bias"] >= .8)
        partitions[label] = {"flag_frame_fraction": float(flag.mean()), "clean_original_mask_frame_fraction": float(clean.mean()), "channels_RGB": channels}
        if label == "native_white_native_radius6":
            supported = [channels[c]["hypothesis_supported"] for c in (0, 2)]
    # Retain sparse leverage and signed covariance contribution for every measured block.
    coordinates = np.column_stack(np.nonzero(mask)).astype(np.int32)
    x = d[mask].astype(np.float64)
    y = e[mask].astype(np.float64)
    centred = x - x.mean(axis=0)
    denominator = np.sum(centred ** 2, axis=0)
    leverage = centred ** 2 / denominator
    contributions = centred * (y - x * np.array(scene["expected_camera_ratio_RGB"])) / denominator
    maps = OUT / "working-block-influence.npz"
    require(not maps.exists(), "Refusing existing influence map")
    np.savez_compressed(maps, coordinates_yx=coordinates, leverage_RGB=leverage, slope_bias_contributions_RGB=contributions,
                        native_radius6_flag=flags["native_white_native_radius6"][mask])
    top = {str(c): [{"yx": coordinates[i].tolist(), "leverage": float(leverage[i, c]),
                    "slope_bias_contribution": float(contributions[i, c]),
                    "native_radius6_flag": bool(flags["native_white_native_radius6"][mask][i])}
                   for i in np.argsort(leverage[:, c])[-20:][::-1]] for c in range(3)}
    out = {"scope": m["scope"], "manifest_sha256": expected, "decode_report_sha256": sha(decoded_path),
           "influence_map": {"file": str(maps.relative_to(ROOT)), "sha256": sha(maps), "n_blocks": len(coordinates)}, "top20_leverage_blocks_by_channel": top,
           "original_mask_frame_fraction": float(mask.mean()), "partitions": partitions,
           "hypothesis_supported_on_both_red_blue": all(supported), "original_joint_WB_qualification": "FAILED unchanged; no model scoring",
           "limits": m["limits"]}
    dump(REPORT, out)
    summary = {label: {"clean_frame_fraction": p["clean_original_mask_frame_fraction"], "channels": [
        {"bias": r["global_slope_bias"], "flagged_fraction": r["groups"]["flagged"]["fraction"],
         "flagged_variance_fraction": r["groups"]["flagged"]["global_variance_fraction"],
         "flagged_relative_bias_contribution": r["groups"]["flagged"]["relative_contribution_to_bias"],
         "clean_slope": r["groups"]["clean"]["own_response"].get("slope"),
         "clean_intercept": r["groups"]["clean"]["own_response"].get("intercept"), "hypothesis_supported": r["hypothesis_supported"]} for r in p["channels_RGB"]]}
        for label, p in partitions.items()}
    print(json.dumps(summary, indent=2))


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
