"""Frozen-base Provia calibration assay (Codex scope 74da423). F24 only, four
frozen whole-session folds. Maps encode(clip(R11_linear,0,1)) -> encoded Fuji
PROVIA with EXACTLY the reference-transfer learner (run_cn_ref_transfer:
20 centred monomials, 60 coefficients, lambda 0.001 on all, equal scene
weights, 16,384 px/scene, same filename seeds, float64 closed-form ridge).
No R11 weight changes, no E4 supervision in B, no search, no final fit, no
K / monitor / finalHeldOut / T8, no app/LUT change.

Primary:   B_f(R11) vs Fuji Provia, against untouched R11 (validation pixels).
Secondary: frozen same-fold film transform T_f (cn-ref-transfer-coeffs,
           hash-verified, unchanged) composed: T_f(B_f(R11)) vs E4, with
           T_f(R11) alongside; rho with app_b = B_f(R11), ref_b = Provia.

  python3 training/run_cn_base_calib.py prepare  # cache + hash R11 predictions, freeze manifest
  python3 training/run_cn_base_calib.py smoke
  python3 training/run_cn_base_calib.py run
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cn_ref_transfer as T  # noqa: E402  (learner + metrics reused verbatim)
import run_cuban_step1 as S  # noqa: E402
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402

MANIFEST = S.DIAG / "cn-base-calib-manifest.json"
MANIFEST_SHA = "f366983b251e64417400eb0daf35d5b735de1dd4bf54a205f7413a5905bf0437"
COEFS = S.DIAG / "cn-base-calib-coeffs"
REPORT = S.DIAG / "cn-base-calib-report.json"
PRED = S.CAL / "cn-base-calib-r11-pred"
REF_REPORT = S.DIAG / "cn-ref-transfer-report.json"


def cmd_prepare():
    if MANIFEST.exists():
        S.fail("manifest already frozen")
    tm = T.load_manifest()
    ref = json.loads(REF_REPORT.read_text())
    tf = {}
    for f, d in ref["folds"].items():
        p = S.TRAIN.parent / d["file"]
        if S.sha256(p) != d["file_sha256"]:
            S.fail(f"T_f fold {f} file hash mismatch")
        tf[f] = {"file": d["file"], "file_sha256": d["file_sha256"], "W_sha256": d["W_sha256"]}
    from eval_cuban_step1_k import infer, load_model
    dev = S.device()
    r11 = load_model(S.SOURCE, dev)
    PRED.mkdir(exist_ok=True)
    scenes = []
    for s in tm["scenes"]:
        from fjlrg import load_fjlrg_linear
        fj = S.CAL / s["fjlrg"]
        if S.sha256(fj) != s["fjlrg_sha256"]:
            S.fail(f"{s['name']}: .fjlrg hash mismatch")
        x, _, _ = load_fjlrg_linear(fj)
        y = infer(r11, x, dev)  # float64 linear, condition [1.0] (capture DR400), Round 11 preprocessing
        p = PRED / f"{s['shoot']}-r11.npy"
        np.save(p, y)
        scenes.append({**s, "r11_pred": str(p.relative_to(S.CAL)), "r11_pred_sha256": S.sha256(p),
                       "r11_pred_array_sha256": T.arr_sha(y)})
        print("cached", s["name"], flush=True)
    S.check_source()
    out = {"status": "FROZEN before any fitting (Codex scope 74da423)",
           "bindings": {**tm["bindings"], "ref_transfer_manifest": T.MANIFEST_SHA, "ref_transfer_report": S.sha256(REF_REPORT),
                        "r11_seed42": S.SOURCE_SHA},
           "learner": {"identical_to": "run_cn_ref_transfer (imported functions phi/solve/normal_stats/sample_idx/apply)",
                       "monomials_ijk_order": T.MONOMIALS, "lambda": T.LAM, "samples_per_scene": T.N_SAMPLE,
                       "sampling_seed": "42 XOR uint64_le(sha256(name)[:8])", "search": None},
           "B": {"input": "encode(clip(R11_linear, 0, 1))", "target": "encode(Fuji Provia)", "supervision": "Provia only - no E4"},
           "T_f": tf,
           "r11_prediction": "eval_cuban_step1_k.infer on frozen seed 42, condition [1.0], float64; cached as .npy, hash-bound",
           "scoring": {"primary": "B_f(R11) vs Provia vs untouched R11, every validation pixel",
                       "secondary": "T_f(B_f(R11)) vs E4 and T_f(R11) vs E4; rho app_b=B_f(R11) ref_b=Provia target=E4",
                       "limits": "existing E4 per-scene limits counted descriptively; no new threshold"},
           "scenes": scenes}
    MANIFEST.write_text(json.dumps(out, indent=1))
    print("manifest sha256", S.sha256(MANIFEST))


def load_manifest():
    if MANIFEST_SHA == "__UNPINNED__" or S.sha256(MANIFEST) != MANIFEST_SHA:
        S.fail("base-calibration manifest missing, unpinned or changed")
    T.load_manifest()
    return json.loads(MANIFEST.read_text())


def load_scene(m, name):
    by = {s["name"]: s for s in m["scenes"]}
    if name not in by:
        S.fail(f"{name} not in the F24 base-calibration manifest - refused")
    s = by[name]
    p = S.CAL / s["r11_pred"]
    if S.sha256(p) != s["r11_pred_sha256"]:
        S.fail(f"{name}: cached R11 prediction hash mismatch")
    _, pv, e4 = T.load_refs(T.load_manifest(), name)
    return np.load(p), pv, e4


def load_tf(m, fold):
    d = m["T_f"][str(fold)]
    p = S.TRAIN.parent / d["file"]
    if S.sha256(p) != d["file_sha256"]:
        S.fail(f"T_f fold {fold} changed")
    W = np.array(json.loads(p.read_text())["W_20x3"])
    if T.arr_sha(W) != d["W_sha256"]:
        S.fail(f"T_f fold {fold} coefficients changed")
    return W


def cmd_smoke():
    m = load_manifest()
    T.cmd_smoke()  # prior synthetic / roundtrip / identity / refusal checks, unchanged
    n0 = m["scenes"][0]["name"]
    r11, pv, e4 = load_scene(m, n0)
    P = linear_srgb_to_lab(pv)
    base = T.metrics(linear_srgb_to_lab(np.clip(r11, 0, 1)), P)
    out, oor = T.apply(np.zeros((20, 3)), T.encode(r11))
    zero = T.metrics(linear_srgb_to_lab(T.decode(out)), P)
    dmax = float(np.abs(T.decode(out) - np.clip(r11, 0, 1)).max())
    assert dmax < 1e-12 and abs(zero["mean_dE00"] - base["mean_dE00"]) < 1e-9 and oor == 0.0, (dmax, zero, base)
    for f in range(4):
        load_tf(m, f)
    import subprocess
    r = subprocess.run([sys.executable, "-c", "import run_cn_base_calib as B; B.load_scene(B.load_manifest(), 'DSCF0515.RAF')"],
                       cwd=S.TRAIN, capture_output=True, text=True)
    assert r.returncode == 2 and "refused" in r.stdout, (r.stdout, r.stderr)
    bad = json.loads(json.dumps(m))
    bad["scenes"][0]["r11_pred_sha256"] = "0" * 64
    try:
        load_scene(bad, n0)
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("changed prediction not refused")
    print(f"base-calib smoke ok: zero-B reproduces R11 display path (max {dmax:.1e}, mean dE identical); "
          f"T_f hashes verified; K and changed prediction refused")


def cmd_run():
    if REPORT.exists():
        S.fail("report exists - not overwriting")
    self_test()
    m = load_manifest()
    folds = json.loads(S.FOLDS.read_text())["folds"]
    ref = json.loads(REF_REPORT.read_text())["per_scene"]
    cache, stats, samp = {}, {}, {}
    for s in m["scenes"]:
        n = s["name"]
        r11, pv, e4 = load_scene(m, n)
        clip_frac = float(((r11 < 0) | (r11 > 1)).any(-1).mean())
        x = T.encode(r11)
        cache[n] = (r11, pv, e4, clip_frac)
        xe, ye = x.reshape(-1, 3), T.encode(pv).reshape(-1, 3)
        idx = T.sample_idx(n, len(xe))
        samp[n] = T.arr_sha(idx)
        stats[n] = T.normal_stats(xe[idx], ye[idx])
        print("loaded", n, flush=True)
    COEFS.mkdir(exist_ok=True)
    per, fold_out = {}, {}
    for fo in folds:
        va = fo["validation_scenes"]
        tr = [s["name"] for s in m["scenes"] if s["name"] not in va]
        W, cond = T.solve([stats[n] for n in tr], T.LAM)
        Wt = load_tf(m, fo["fold"])
        cf = {"fold": fo["fold"], "train": tr, "validation": va, "lambda": T.LAM, "monomials_ijk_order": T.MONOMIALS,
              "W_20x3": W.tolist(), "W_sha256": T.arr_sha(W), "condition_number": cond,
              "train_sample_index_sha256": {n: samp[n] for n in tr}, "manifest_sha256": MANIFEST_SHA,
              "composed_with_T_f_W_sha256": m["T_f"][str(fo["fold"])]["W_sha256"]}
        p = COEFS / f"fold{fo['fold']}.json"
        p.write_text(json.dumps(cf, indent=1))
        fold_out[fo["fold"]] = {"file": str(p.relative_to(S.TRAIN.parent)), "file_sha256": S.sha256(p), "W_sha256": cf["W_sha256"],
                                "condition_number": cond, "max_abs_coefficient": float(np.abs(W).max())}
        for n in va:
            r11, pv, e4, clip_frac = cache[n]
            P, E = linear_srgb_to_lab(pv), linear_srgb_to_lab(e4)
            R = linear_srgb_to_lab(np.clip(r11, 0, 1))
            b_enc, oor_b = T.apply(W, T.encode(r11))
            Bl = linear_srgb_to_lab(T.decode(b_enc))
            tb_enc, oor_tb = T.apply(Wt, b_enc)
            TB = linear_srgb_to_lab(T.decode(tb_enc))
            tr_enc, oor_tr = T.apply(Wt, T.encode(r11))
            TR = linear_srgb_to_lab(T.decode(tr_enc))
            den = float(ciede2000(P, E).mean())
            row = {"fold": fo["fold"], "session": ref[n]["session"], "asShotWhiteBalance": ref[n]["asShotWhiteBalance"],
                   "r11_input_clip_fraction": clip_frac,
                   "provia": {"r11_untouched": T.metrics(R, P), "B_r11": {**T.metrics(Bl, P), "pre_clip_out_of_range": oor_b}},
                   "e4": {"T_r11": {**T.metrics(TR, E), "pre_clip_out_of_range": oor_tr,
                                    "rho_provia_to_E4": float(ciede2000(P + TR - R, E).mean()) / den},
                          "T_B_r11": {**T.metrics(TB, E), "pre_clip_out_of_range": oor_tb,
                                      "rho_provia_to_E4": float(ciede2000(P + TB - Bl, E).mean()) / den}}}
            for k in ("T_r11", "T_B_r11"):
                c = T.e4_checks(row["e4"][k], row["e4"][k]["rho_provia_to_E4"])
                row["e4"][k]["checks"], row["e4"][k]["meets_all_E4_limits"] = c, all(c.values())
            row["diff_B_minus_r11_vs_provia_mean_dE00"] = row["provia"]["B_r11"]["mean_dE00"] - row["provia"]["r11_untouched"]["mean_dE00"]
            row["diff_TB_minus_T_r11_vs_e4_mean_dE00"] = row["e4"]["T_B_r11"]["mean_dE00"] - row["e4"]["T_r11"]["mean_dE00"]
            row["context_oracle_T_provia_vs_e4_mean_dE00"] = ref[n]["T_provia"]["mean_dE00"]
            row["context_step1_seed42_vs_E4_IN_SAMPLE"] = ref[n]["context_step1_seed42_vs_E4_IN_SAMPLE"]
            per[n] = row
            print(n, f"vsProvia R11 {row['provia']['r11_untouched']['mean_dE00']:.2f} B {row['provia']['B_r11']['mean_dE00']:.2f} | "
                     f"vsE4 T(R11) {row['e4']['T_r11']['mean_dE00']:.2f} T(B) {row['e4']['T_B_r11']['mean_dE00']:.2f} "
                     f"rho {row['e4']['T_B_r11']['rho_provia_to_E4']:.2f}", flush=True)
    names = sorted(per)
    def summ(sec, k):
        rows = [per[n][sec][k] for n in names]
        out = {"mean_of_means": float(np.mean([r["mean_dE00"] for r in rows])), "median_mean": float(np.median([r["mean_dE00"] for r in rows])),
               "mean_p95": float(np.mean([r["p95_dE00"] for r in rows])),
               "cross_scene_mean_signed_dL_by_tone_band": {b: float(np.mean([r["signed_dL_by_tone_band"][b] for r in rows
                                                                             if r["signed_dL_by_tone_band"][b] is not None]))
                                                           for b in ("shadow", "mid", "high")},
               "cross_scene_mean_dC": float(np.mean([r["signed_dC_ref_C_gt_10"] for r in rows if r["signed_dC_ref_C_gt_10"] is not None])),
               "cross_scene_mean_hue_deg": float(np.mean([r["signed_hue_deg_ref_C_gt_20"] for r in rows
                                                          if r["signed_hue_deg_ref_C_gt_20"] is not None]))}
        if "checks" in rows[0]:
            out["scenes_meeting_all_E4_limits"] = int(sum(r["meets_all_E4_limits"] for r in rows))
            out["per_check_pass_counts"] = {c: int(sum(r["checks"][c] for r in rows)) for c in rows[0]["checks"]}
            out["median_rho"] = float(np.median([r["rho_provia_to_E4"] for r in rows]))
        return out
    sess = {}
    for n in names:
        sess.setdefault(per[n]["session"], []).append(n)
    groups = {str(s): {"n": len(v), "scenes": v,
                       "R11_vs_provia": float(np.mean([per[x]["provia"]["r11_untouched"]["mean_dE00"] for x in v])),
                       "B_vs_provia": float(np.mean([per[x]["provia"]["B_r11"]["mean_dE00"] for x in v])),
                       "T_r11_vs_e4": float(np.mean([per[x]["e4"]["T_r11"]["mean_dE00"] for x in v])),
                       "T_B_vs_e4": float(np.mean([per[x]["e4"]["T_B_r11"]["mean_dE00"] for x in v]))} for s, v in sorted(sess.items())}
    out = {"scope": "F24 frozen-base Provia calibration; adaptation-OOF by whole session; R11 warm start saw these Provia scenes - "
                    "not unseen-RAW acceptance; diagnostic only, nothing selected",
           "manifest_sha256": MANIFEST_SHA, "folds": fold_out,
           "summary": {"provia": {k: summ("provia", k) for k in ("r11_untouched", "B_r11")},
                       "e4": {k: summ("e4", k) for k in ("T_r11", "T_B_r11")}},
           "B_improves_vs_provia": [n for n in names if per[n]["diff_B_minus_r11_vs_provia_mean_dE00"] < 0],
           "B_regresses_vs_provia": [n for n in names if per[n]["diff_B_minus_r11_vs_provia_mean_dE00"] >= 0],
           "TB_improves_vs_T_r11": [n for n in names if per[n]["diff_TB_minus_T_r11_vs_e4_mean_dE00"] < 0],
           "TB_regresses_vs_T_r11": [n for n in names if per[n]["diff_TB_minus_T_r11_vs_e4_mean_dE00"] >= 0],
           "session_groups": groups, "per_scene": per}
    S.check_source()
    REPORT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("folds", "summary", "B_regresses_vs_provia", "TB_regresses_vs_T_r11")}, indent=1))


if __name__ == "__main__":
    {"prepare": cmd_prepare, "smoke": cmd_smoke, "run": cmd_run}[sys.argv[1]]()
