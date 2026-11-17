"""DR100 evaluation batch - predeclared scoring (docs/phase4-scene-adaptive-scope.md,
"DR100 evaluation batch", fixed 2026-11-14, Codex-reviewed; single-session
deviation user-approved 2026-11-17 and labelled in every report).

Committed BEFORE any Fuji TIFF target for this batch existed. Thresholds
below are copied from the doc and must not be edited after scoring.

Stages (run in order):
  python3 training/eval_dr100.py check-targets      # TIFF present + output DR100 per MakerNote
  python3 training/eval_dr100.py score              # criteria 1-4, writes diagnostics/dr100eval-results.json
  python3 training/eval_dr100.py parity-vectors     # native ONNX parity on these 20 inputs + vectors
  node scripts/verify-round11-onnx-web.mjs --vector-dir training/checkpoints/round11-onnx-dr100eval \
       --ort-dir training/checkpoints/onnx-web-deps/node_modules/onnxruntime-web/dist   # criterion 5
  python3 training/eval_dr100.py verdict            # combines 1-5

Loads ONLY the dr100Eval manifest; refuses any other tier. Never touches
finalHeldOut, never trains, never writes into the training manifest.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from check_dr100eval_metadata import exclusion_reasons
from diagnose_signed_errors import band_stats
from fjlrg import load_fjlrg_linear
from model import compute_luma_guide
from run_round11_drconditioning import frame_signed, load_model_from_ckpt
from target_tiff import load_xraw_target_linear_matching

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CAL = REPO / "calibration-input"
MANIFEST = CAL / "phase4-dr100eval-manifest.json"
EXIF = CAL / "phase4-dr100eval-exif.json"
TARGET_CACHE = CAL / "phase4-dr100eval-target-cache"
INPUT_NAME = "browser-phase4-dr100eval-linear.fjlrg"
TARGET_NAME = "xraw-phase4-dr100eval.tiff"
OUT = ROOT / "diagnostics" / "dr100eval-results.json"
TARGET_CHECK = ROOT / "diagnostics" / "dr100eval-target-check.json"
VEC_DIR = ROOT / "checkpoints" / "round11-onnx-dr100eval"
EXPORTS = ROOT / "exports" / "round11"

# ---- predeclared constants (doc, 2026-11-14) -------------------------------
CANDIDATE_SEEDS = [42, 43]
REPORT_ONLY_SEEDS = [44]
R11 = {s: f"r11-drcond-seed{s}-best.pt" for s in (42, 43, 44)}
R9 = {s: f"r9-expandeddata-seed{s}-best.pt" for s in (42, 43, 44)}
MID_SIGNED_ABS_LIMIT = 0.05        # |mean signed mid| <= 0.05
MID_MEAN_ABS_LIMIT = 0.05          # mean |per-scene mid signed| <= 0.05
FRAME_SIGNED_LIMIT = 0.10          # no scene |frame signed| > 0.10
MID_COVERAGE_MIN = 0.05            # mid band (target luma 0.2-0.5) >= 5% of pixels
MIN_MEASURABLE_ALL, MIN_MEASURABLE_PER_GROUP = 10, 3
LOW_ISO_MAX, HIGH_ISO_MIN = 200, 1600
MIN_FRAMES, MIN_PER_ISO_GROUP = 12, 4
DEVIATION = ("single-session batch (2026-11-17 15:57-18:22) used by user decision; the >=3-session "
             "rule is waived - results do not test session-level independence")


def load_scenes():
    manifest = json.loads(MANIFEST.read_text())
    if any(m.get("tier") != "dr100Eval" for m in manifest):
        sys.exit("refused: manifest contains a non-dr100Eval row")
    exif = {e["name"]: e for e in json.loads(EXIF.read_text())}
    scenes, excluded = [], []
    for m in manifest:
        e = exif.get(m["filename"])
        reasons = exclusion_reasons(e) if e else ["no EXIF entry"]
        (excluded if reasons else scenes).append({"name": m["filename"], "shoot": m["shoot_folder"],
                                                  "iso": (e or {}).get("iso"), "reasons": reasons})
    return scenes, excluded


# ---- TIFF output-DR check (Fuji MakerNote 0x1403, no exiftool needed) ------
def fuji_makernote_tag(tiff_path, tag):
    with Image.open(tiff_path) as im:
        exif_ifd = im.getexif().get_ifd(0x8769)
    mn = exif_ifd.get(0x927C)
    if not isinstance(mn, (bytes, bytearray)) or not mn.startswith(b"FUJIFILM"):
        return None
    ifd = struct.unpack_from("<I", mn, 8)[0]
    for i in range(struct.unpack_from("<H", mn, ifd)[0]):
        t, typ, cnt, val = struct.unpack_from("<HHII", mn, ifd + 2 + 12 * i)
        if t == tag:
            return val & 0xFFFF if typ == 3 else val
    return None


def check_targets():
    scenes, excluded = load_scenes()
    rows, ok = [], True
    for s in scenes:
        p = CAL / s["shoot"] / TARGET_NAME
        row = {"name": s["name"], "shoot": s["shoot"], "present": p.exists()}
        if p.exists():
            row["target_tiff_sha256"] = file_sha256(p)  # binds this check to these exact bytes
            row["tiff_DevelopmentDynamicRange"] = fuji_makernote_tag(p, 0x1403)
            row["output_dr100"] = row["tiff_DevelopmentDynamicRange"] == 100
        ok &= bool(row["present"] and row.get("output_dr100"))
        rows.append(row)
        print(row)
    TARGET_CHECK.write_text(json.dumps({"passed": ok, "rows": rows, "excluded_frames": excluded}, indent=1))
    print("targets OK" if ok else "TARGET CHECK FAILED - do not score")
    return 0 if ok else 1


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pair(s):
    """Returns (input, target, fingerprints). The target cache is keyed by the
    SHA-256 of the source TIFF (and the cached array's own SHA-256), so a
    re-exported TIFF or a tampered/partial cache is never silently reused."""
    in_path, tif_path = CAL / s["shoot"] / INPUT_NAME, CAL / s["shoot"] / TARGET_NAME
    fp = {"input_sha256": file_sha256(in_path), "target_tiff_sha256": file_sha256(tif_path)}
    rgb, w, h = load_fjlrg_linear(in_path)
    TARGET_CACHE.mkdir(exist_ok=True)
    cache, meta_path = TARGET_CACHE / f"{s['shoot']}.npy", TARGET_CACHE / f"{s['shoot']}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if (cache.exists() and meta.get("target_tiff_sha256") == fp["target_tiff_sha256"]
            and meta.get("shape") == list(rgb.shape) and meta.get("cache_sha256") == file_sha256(cache)):
        target = np.load(cache)
    else:
        target = load_xraw_target_linear_matching(tif_path, w, h)
        np.save(cache, target)
        meta_path.write_text(json.dumps({"target_tiff_sha256": fp["target_tiff_sha256"], "shape": list(target.shape),
                                         "cache_sha256": file_sha256(cache)}))
    if target.shape != rgb.shape:
        raise ValueError(f"{s['name']}: shape mismatch {rgb.shape} vs {target.shape}")
    return rgb, target, fp


def target_check_problems(scenes):
    """The passed target check must cover exactly these scenes and still
    describe the TIFF bytes on disk now - a TIFF replaced after its
    metadata check invalidates the check."""
    if not TARGET_CHECK.exists():
        return ["target check missing - run check-targets"]
    tc = json.loads(TARGET_CHECK.read_text())
    if tc.get("passed") is not True:
        return ["target check did not pass"]
    rows = {r["name"]: r for r in tc.get("rows", [])}
    problems = []
    if set(rows) != {s["name"] for s in scenes}:
        problems.append("target check scene set != accepted manifest scenes")
    for s in scenes:
        r = rows.get(s["name"], {})
        if not (r.get("output_dr100") is True and r.get("target_tiff_sha256")):
            problems.append(f"{s['name']}: no hash-bound passing target check")
        elif r["target_tiff_sha256"] != file_sha256(CAL / s["shoot"] / TARGET_NAME):
            problems.append(f"{s['name']}: TIFF changed since check-targets - rerun it")
    return problems


def score():
    scenes, excluded = load_scenes()
    tc_problems = target_check_problems(scenes)
    if tc_problems:
        for p in tc_problems:
            print("TARGET CHECK:", p)
        print("SCORE FAILED - target check invalid")
        return 1
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    models = {("r11", s): load_model_from_ckpt(c, device) for s, c in R11.items()}
    models.update({("r9", s): load_model_from_ckpt(c, device) for s, c in R9.items()})
    condition = torch.zeros(1, device=device)  # capture DR100 -> log2(100/100)/2 = 0
    per_scene = {}
    for s in scenes:
        rgb, tgt, fp = load_pair(s)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(device)
        t = torch.from_numpy(np.ascontiguousarray(tgt.transpose(2, 0, 1)))[None].to(device)
        tl = compute_luma_guide(t, "rec709").squeeze().cpu().numpy()
        il = compute_luma_guide(x, "rec709").squeeze().cpu().numpy()
        trgb = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        row = {"shoot": s["shoot"], "iso": s["iso"], **fp,
               "iso_group": "low" if s["iso"] <= LOW_ISO_MAX else "high" if s["iso"] >= HIGH_ISO_MIN else "other",
               "baseline_l1": float((x - t).abs().mean()),
               "exposure_ratio_stops": float(np.log2(tl.mean() / max(il.mean(), 1e-8))), "models": {}}
        for (rnd, seed), (model, cfg, conditioned) in models.items():
            with torch.no_grad():
                pred, _ = model.predict_and_apply(x, cfg["low_res"], cfg.get("luma_mode", "rec709"),
                                                  condition=condition if conditioned else None)
            st = band_stats(compute_luma_guide(pred, "rec709").squeeze().cpu().numpy(), tl, il,
                            pred.squeeze(0).permute(1, 2, 0).cpu().numpy(), trgb)
            l1 = float((pred - t).abs().mean())
            mid_frac = st["mid"]["pixel_fraction"]
            row["models"][f"{rnd}-s{seed}"] = {
                "model_l1": l1, "regressed": l1 > row["baseline_l1"],
                "mid_pixel_fraction": mid_frac, "mid_measurable": mid_frac >= MID_COVERAGE_MIN,
                "mid_signed": st["mid"].get("signed_linear_error") if mid_frac >= MID_COVERAGE_MIN else None,
                "shadow_signed": st["shadow"].get("signed_linear_error"),
                "highlight_signed": st["highlight"].get("signed_linear_error"),
                "frame_signed": frame_signed(st)}
        per_scene[s["name"]] = row
        r = row["models"]
        print(f"{s['name']} ISO {s['iso']}: base {row['baseline_l1']:.4f}  r9-s42 {r['r9-s42']['model_l1']:.4f}  "
              f"r11-s42 {r['r11-s42']['model_l1']:.4f}  r11-s43 {r['r11-s43']['model_l1']:.4f}  "
              f"mid r11-s42 {r['r11-s42']['mid_signed']}  exp {row['exposure_ratio_stops']:+.2f}", flush=True)

    def mid_stats(names, key):
        vals = [per_scene[n]["models"][key]["mid_signed"] for n in names
                if per_scene[n]["models"][key]["mid_signed"] is not None]
        return {"n_measurable": len(vals), "mean_signed": float(np.mean(vals)) if vals else None,
                "mean_abs": float(np.mean(np.abs(vals))) if vals else None}

    names = list(per_scene)
    groups = {"all": names, "low_iso": [n for n in names if per_scene[n]["iso_group"] == "low"],
              "high_iso": [n for n in names if per_scene[n]["iso_group"] == "high"]}
    short = []
    if len(names) < MIN_FRAMES:
        short.append(f"{len(names)} frames < {MIN_FRAMES}")
    for g in ("low_iso", "high_iso"):
        if len(groups[g]) < MIN_PER_ISO_GROUP:
            short.append(f"{g}: {len(groups[g])} frames < {MIN_PER_ISO_GROUP}")

    candidates = {}
    for seed in CANDIDATE_SEEDS + REPORT_ONLY_SEEDS:
        k11, k9 = f"r11-s{seed}", f"r9-s{seed}"
        ms = {g: mid_stats(v, k11) for g, v in groups.items()}
        evaluable = (ms["all"]["n_measurable"] >= MIN_MEASURABLE_ALL and
                     all(ms[g]["n_measurable"] >= MIN_MEASURABLE_PER_GROUP for g in ("low_iso", "high_iso")))
        def mid_ok(m):
            return (m["mean_signed"] is not None and abs(m["mean_signed"]) <= MID_SIGNED_ABS_LIMIT
                    and m["mean_abs"] <= MID_MEAN_ABS_LIMIT)
        l1_11 = float(np.mean([per_scene[n]["models"][k11]["model_l1"] for n in names]))
        l1_9 = float(np.mean([per_scene[n]["models"][k9]["model_l1"] for n in names]))
        frame_bad = [n for n in names if abs(per_scene[n]["models"][k11]["frame_signed"]) > FRAME_SIGNED_LIMIT]
        c = {"c1_mid_all": mid_ok(ms["all"]),
             "c2_zero_regressions": not any(per_scene[n]["models"][k11]["regressed"] for n in names),
             "c3_beats_matching_r9": l1_11 < l1_9,
             "c4_high_iso_mid_and_frame": mid_ok(ms["high_iso"]) and not frame_bad}
        candidates[seed] = {
            "role": "candidate" if seed in CANDIDATE_SEEDS else "report-only",
            "mid_stats": ms, "mid_criteria_evaluable": evaluable,
            "mean_l1_round11": l1_11, "mean_l1_round9_same_seed": l1_9,
            "regressed_vs_identity": [n for n in names if per_scene[n]["models"][k11]["regressed"]],
            "scenes_frame_signed_over_limit": frame_bad,
            "not_mid_measurable": [n for n in names if not per_scene[n]["models"][k11]["mid_measurable"]],
            "per_scene_l1_diff_r11_minus_r9": {n: per_scene[n]["models"][k11]["model_l1"]
                                               - per_scene[n]["models"][k9]["model_l1"] for n in names},
            "criteria_1_to_4": c,
            "criteria_1_to_4_pass": bool(evaluable and not short and all(c.values()))}
        print(f"seed {seed}: {c}  evaluable={evaluable}  L1 r11 {l1_11:.4f} vs r9 {l1_9:.4f}")

    OUT.write_text(json.dumps({
        "deviation": DEVIATION, "batch_short": short, "excluded_frames": excluded,
        "groups": groups, "per_scene": per_scene, "seeds": candidates,
        "thresholds": {"mid_signed_abs": MID_SIGNED_ABS_LIMIT, "mid_mean_abs": MID_MEAN_ABS_LIMIT,
                       "frame_signed": FRAME_SIGNED_LIMIT, "mid_coverage_min": MID_COVERAGE_MIN},
        "criterion_5_browser_parity": "pending - run parity-vectors, the web verifier, then verdict",
        "finalHeldOut_accessed": False}, indent=1, allow_nan=False))
    print(f"wrote {OUT}")
    failed = [s for s in CANDIDATE_SEEDS if not candidates[s]["criteria_1_to_4_pass"]]
    if short or failed:
        print(f"SCORE FAILED - batch short: {short}; candidate seeds failing criteria 1-4: {failed}")
        return 1
    return 0


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parity_vectors():
    """Native ONNX parity on these exact 20 inputs, plus vectors for the
    existing web verifier (same tolerances as the 2026-11-14 export gate)."""
    import onnxruntime as ort
    from export_round11_onnx import MAX_ERROR, MEAN_ERROR, InferenceWrapper
    from model import HybridPredictor
    scenes, _ = load_scenes()
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    # Clear everything from any earlier run so a stale report or vector can
    # never be picked up by the web verifier or by verdict.
    # (exFAT: skip macOS "._" AppleDouble sidecars - they vanish with their file.)
    for old in list(VEC_DIR.glob("*.f32")) + list(VEC_DIR.glob("*.json")) + list(VEC_DIR.glob("*.onnx")):
        if not old.name.startswith("._"):
            old.unlink(missing_ok=True)
    gate = json.loads((EXPORTS / "native-report.json").read_text())
    cases = []
    for s in scenes:
        src = CAL / s["shoot"] / INPUT_NAME
        rgb, _, _ = load_fjlrg_linear(src)
        img = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype("<f4")
        fn = f"{s['name'].removesuffix('.RAF')}-input.f32"
        img.tofile(VEC_DIR / fn)
        cases.append({"name": s["name"].removesuffix(".RAF"), "shape": list(img.shape), "capture_dr": 100,
                      "condition": 0.0, "input": fn, "source": "dr100Eval",
                      "source_input_sha256": file_sha256(src), "vector_sha256": file_sha256(VEC_DIR / fn)})
    report = {"format_version": 1, "opset": gate["opset"], "input_contract": gate["input_contract"],
              "native_tolerances": {"max_abs": MAX_ERROR, "mean_abs": MEAN_ERROR}, "cases": cases, "models": []}
    passed = True
    for seed in CANDIDATE_SEEDS:
        g = next(m for m in gate["models"] if m["seed"] == seed)
        onnx_path = EXPORTS / g["model"]
        if sha256(onnx_path) != g["model_sha256"]:
            sys.exit(f"{onnx_path.name}: SHA-256 differs from the export gate - refusing")
        (VEC_DIR / g["model"]).write_bytes(onnx_path.read_bytes())
        ckpt = torch.load(ROOT / "checkpoints" / R11[seed], map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        model = HybridPredictor(grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"], low_res=cfg["low_res"],
                                base_ch=cfg["base_ch"], refinement_base_ch=cfg["refinement_base_ch"],
                                refinement_max_delta=cfg["refinement_max_delta"], condition_channels=1)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        ref = InferenceWrapper(model, cfg).eval()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        entries = []
        for c in cases:
            img = np.fromfile(VEC_DIR / c["input"], dtype="<f4").reshape(c["shape"])
            cond = np.zeros((1, 1), dtype=np.float32)
            with torch.inference_mode():
                exp = ref(torch.from_numpy(img), torch.from_numpy(cond)).numpy()
            act = sess.run(None, {"linear_rgb": img, "capture_dr_condition": cond})[0]
            err = np.abs(exp - act)
            ok = bool(np.isfinite(act).all() and err.max() <= MAX_ERROR and err.mean() <= MEAN_ERROR)
            passed &= ok
            en = f"seed{seed}-{c['name']}-expected.f32"
            exp.astype("<f4").tofile(VEC_DIR / en)
            entries.append({"name": c["name"], "expected": en, "max_abs": float(err.max()),
                            "mean_abs": float(err.mean()), "passed": ok})
            print(f"native seed {seed} {c['name']}: max={err.max():.3g} mean={err.mean():.3g} pass={ok}", flush=True)
        report["models"].append({"seed": seed, "model": g["model"], "model_sha256": g["model_sha256"],
                                 "native_cases": entries})
    report["passed"] = passed
    (VEC_DIR / "native-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("native parity", "passed" if passed else "FAILED")
    return 0 if passed else 1


def within(case, tol):
    """Recorded errors must be real finite numbers within the frozen limits."""
    vals = [case.get("max_abs"), case.get("mean_abs")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in vals):
        return False
    return vals[0] <= tol["max_abs"] and vals[1] <= tol["mean_abs"]


def validate_reports(res):
    """Every way the three reports could disagree with each other, with the
    frozen export gate, or with the predeclared constants. Returns a list of
    problems; empty means consistent."""
    problems = []
    def need(cond, msg):
        if not cond:
            problems.append(msg)

    # results file vs manifest/constants
    scenes, _ = load_scenes()
    scene_names = {s["name"] for s in scenes}
    need(set(res.get("per_scene", {})) == scene_names, "results scene set != accepted manifest scenes")
    need(res.get("thresholds") == {"mid_signed_abs": MID_SIGNED_ABS_LIMIT, "mid_mean_abs": MID_MEAN_ABS_LIMIT,
                                   "frame_signed": FRAME_SIGNED_LIMIT, "mid_coverage_min": MID_COVERAGE_MIN},
         "results thresholds differ from the predeclared constants")
    need(set(res.get("seeds", {})) >= {str(s) for s in CANDIDATE_SEEDS}, "results missing a candidate seed")
    problems += target_check_problems(scenes)
    tc_rows = {r["name"]: r for r in json.loads(TARGET_CHECK.read_text()).get("rows", [])} if TARGET_CHECK.exists() else {}
    for n, row in res.get("per_scene", {}).items():
        if row.get("target_tiff_sha256") != tc_rows.get(n, {}).get("target_tiff_sha256"):
            problems.append(f"{n}: scored TIFF != TIFF that passed check-targets")
        s = next((x for x in scenes if x["name"] == n), None)
        if s and row.get("input_sha256") != file_sha256(CAL / s["shoot"] / INPUT_NAME):
            problems.append(f"{n}: input changed since scoring")
        if s and row.get("target_tiff_sha256") != file_sha256(CAL / s["shoot"] / TARGET_NAME):
            problems.append(f"{n}: target TIFF changed since scoring")

    # native report vs export gate and results
    native_path, browser_path = VEC_DIR / "native-report.json", VEC_DIR / "browser-report.json"
    if not native_path.exists():
        return problems + ["native report missing"]
    native = json.loads(native_path.read_text())
    gate = json.loads((EXPORTS / "native-report.json").read_text())
    gate_browser = json.loads((EXPORTS / "browser-report.json").read_text())
    stems = {n.removesuffix(".RAF") for n in scene_names}
    need(native.get("passed") is True, "native report not passed")
    need({c["name"] for c in native["cases"]} == stems, "native case set != scored scenes")
    need(native.get("native_tolerances") == gate.get("native_tolerances"), "native tolerances differ from export gate")
    need(sorted(m["seed"] for m in native["models"]) == CANDIDATE_SEEDS, "native models != candidate seeds")
    for m in native["models"]:
        g = next((x for x in gate["models"] if x["seed"] == m["seed"]), {})
        need(m.get("model_sha256") == g.get("model_sha256"), f"seed {m['seed']}: model SHA differs from export gate")
        need(file_sha256(VEC_DIR / m["model"]) == g.get("model_sha256"), f"seed {m['seed']}: served ONNX differs from export gate")
        for c in m["native_cases"]:  # recheck the numbers; never trust the recorded flag alone
            need(c.get("passed") is True and within(c, gate["native_tolerances"]),
                 f"native seed {m['seed']} {c['name']}: max/mean {c.get('max_abs')}/{c.get('mean_abs')} "
                 f"not finite or over frozen limits {gate['native_tolerances']}")
        need({c["name"] for c in m["native_cases"]} == stems, f"seed {m['seed']}: native cases incomplete")
    for c in native["cases"]:
        row = res["per_scene"].get(c["name"] + ".RAF", {})
        need(c.get("source_input_sha256") == row.get("input_sha256"), f"{c['name']}: parity input != scored input")
        need(file_sha256(VEC_DIR / c["input"]) == c.get("vector_sha256"), f"{c['name']}: parity vector changed")

    # browser report vs native report and the export gate's browser settings
    if not browser_path.exists():
        return problems + ["browser report missing"]
    browser = json.loads(browser_path.read_text())
    need(browser_path.stat().st_mtime >= native_path.stat().st_mtime, "browser report older than native report")
    need(browser.get("passed") is True, "browser report not passed")
    need(browser.get("tolerances") == gate_browser.get("tolerances"), "browser tolerances differ from export gate")
    need(browser.get("webgpuLayout") == "NCHW", "WebGPU layout is not NCHW")
    provs = browser.get("providers", [])
    need(sorted(p.get("provider") for p in provs) == ["wasm", "webgpu"], "providers are not exactly wasm + webgpu")
    for p in provs:
        need(not p.get("error"), f"{p.get('provider')}: error {p.get('error')}")
        if p.get("provider") == "webgpu":
            need((p.get("adapterInfo") or {}).get("isFallbackAdapter") is not True, "WebGPU ran on a fallback adapter")
        need(sorted(m["seed"] for m in p.get("models", [])) == CANDIDATE_SEEDS, f"{p.get('provider')}: models != candidates")
        for m in p.get("models", []):
            need({c["name"] for c in m["cases"]} == stems, f"{p.get('provider')} seed {m['seed']}: cases incomplete")
            for c in m["cases"]:  # recheck the numbers against the export gate's frozen browser limits
                need(c.get("passed") is True and within(c, gate_browser["tolerances"]),
                     f"{p.get('provider')} seed {m['seed']} {c['name']}: max/mean {c.get('max_abs')}/{c.get('mean_abs')} "
                     f"not finite or over frozen limits {gate_browser['tolerances']}")
    return problems


def verdict():
    try:
        res = json.loads(OUT.read_text())
        if not isinstance(res, dict) or not isinstance(res.get("seeds"), dict):
            raise ValueError("results file is not a scored results object")
    except (OSError, ValueError) as e:  # missing/malformed results = failure, same as any other report
        print(f"VALIDATION: results file unusable: {type(e).__name__}: {e}")
        print("overall: False")
        return 1
    try:
        problems = validate_reports(res)
    except (OSError, KeyError, ValueError, TypeError, StopIteration) as e:  # missing/malformed report or file = failure, not a crash
        problems = [f"validation could not complete: {type(e).__name__}: {e}"]
    c5 = not problems
    res["criterion_5_browser_parity"] = {"passed": c5, "validation_problems": problems,
                                         "native_report": str((VEC_DIR / "native-report.json").relative_to(REPO)),
                                         "browser_report": str((VEC_DIR / "browser-report.json").relative_to(REPO))}
    for seed in CANDIDATE_SEEDS:
        s = res["seeds"][str(seed)]
        s["pass_all_5"] = bool(s["criteria_1_to_4_pass"] and c5 and not res.get("batch_short"))
    res["overall_pass"] = all(res["seeds"][str(s)]["pass_all_5"] for s in CANDIDATE_SEEDS)
    OUT.write_text(json.dumps(res, indent=1, allow_nan=False))
    for p in problems:
        print("VALIDATION:", p)
    print(json.dumps({k: res["seeds"][str(k)]["pass_all_5"] for k in CANDIDATE_SEEDS}), "overall:", res["overall_pass"])
    return 0 if res["overall_pass"] else 1


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"check-targets": check_targets, "score": score, "parity-vectors": parity_vectors, "verdict": verdict}.get(stage)
    if not fn:
        sys.exit(__doc__)
    sys.exit(fn())
