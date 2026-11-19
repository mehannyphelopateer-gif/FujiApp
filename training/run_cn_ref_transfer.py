"""Bounded reference-transfer assay (Codex scope 0dc109c): a fixed 60-coefficient
global degree-3 RGB residual polynomial, Fuji Provia -> Fuji E4, closed-form
ridge, fit on the four frozen F24 whole-session folds. Diagnostic learner only:
no R11 weight update, no scene/WB features, no search, no all-F final fit,
no K / monitor / finalHeldOut / T8 access, no app change.

Model (frozen): x = sRGB-encoded [0,1] RGB (from the linear TIFF loader);
c = 2x - 1; Phi(c) = 20 monomials c_r^i c_g^j c_b^k, i+j+k <= 3, in MONOMIALS
order; T(x) = x + Phi(c) @ W, W (20, 3). W = 0 is the identity.
Fit: per output channel, minimise  mean_s mean_px (x + Phi W - y)^2 + lam ||W||^2
with lam = 0.001 on every coefficient (intercept included), equal scene
weights: W = solve(mean_s Phi_s^T Phi_s / n + lam I, mean_s Phi_s^T (y_s - x_s) / n).
Per training scene a fixed 16,384-pixel sample without replacement,
rng seed = 42 XOR uint64_le(SHA256(filename)[:8]), identical in every fold.
float64 throughout.

Scoring (every validation pixel): output clipped to [0,1] (pre-clip
out-of-range fraction recorded), decoded to linear, frozen Lab / CIEDE2000.
Primary: T(Fuji Provia) vs E4, against identity (Fuji Provia vs E4).
Secondary: the SAME fold transform applied, no refit, to frozen R11
predictions (linear clipped to [0,1] first, fraction recorded).

  python3 training/run_cn_ref_transfer.py prepare  # freeze manifest
  python3 training/run_cn_ref_transfer.py smoke    # unit + refusal tests
  python3 training/run_cn_ref_transfer.py run      # 4 fold fits + report
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuban_step1 as S  # noqa: E402
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402
from cuban_contract_v2 import CONTRACT_V2  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from target_tiff import load_xraw_target_linear_matching  # noqa: E402

MANIFEST = S.DIAG / "cn-ref-transfer-manifest.json"
MANIFEST_SHA = "a0ba1ce2b2e32e211332cce2775e585352f3cb2ee2f50d65666eda9e16b25866"
COEFS = S.DIAG / "cn-ref-transfer-coeffs"
REPORT = S.DIAG / "cn-ref-transfer-report.json"
WB_MAN = S.DIAG / "cn-wb-assay-manifest.json"
DECOMP = S.DIAG / "cn-step1-f24-decomposition.json"
LAM, N_SAMPLE, BASE_SEED = 1e-3, 16384, 42
MONOMIALS = [(i, j, d - i - j) for d in range(4) for i in range(d, -1, -1) for j in range(d - i, -1, -1)]
assert len(MONOMIALS) == 20
LIM = CONTRACT_V2["stage_limits"]["E4"]


# ------------------------------------------------------------ transfer funcs
def encode(lin):
    x = np.clip(lin, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def decode(enc):
    x = np.clip(enc, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def phi(x):
    """x (..., 3) encoded [0,1] -> (..., 20)."""
    c = 2.0 * x - 1.0
    return np.stack([c[..., 0] ** i * c[..., 1] ** j * c[..., 2] ** k for i, j, k in MONOMIALS], axis=-1)


def apply(W, x):
    """Returns (clipped output, pre-clip out-of-range fraction of pixels)."""
    y = x + phi(x) @ W
    oor = float(((y < 0) | (y > 1)).any(-1).mean())
    return np.clip(y, 0.0, 1.0), oor


def scene_seed(name):
    return BASE_SEED ^ int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def sample_idx(name, n_px):
    return np.sort(np.random.default_rng(scene_seed(name)).choice(n_px, N_SAMPLE, replace=False))


def solve(stats, lam):
    A = np.mean([s[0] for s in stats], axis=0)
    b = np.mean([s[1] for s in stats], axis=0)
    M = A + lam * np.eye(A.shape[0])
    W = np.linalg.solve(M, b)
    if not np.isfinite(W).all():
        raise ValueError("non-finite ridge solution")
    return W, float(np.linalg.cond(M))


def normal_stats(x, y):
    P = phi(x)
    n = len(x)
    return P.T @ P / n, P.T @ (y - x) / n


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ------------------------------------------------------------ manifest
def cmd_prepare():
    if MANIFEST.exists():
        S.fail("manifest already frozen")
    man, folds, allow, k = S.preflight()
    wbm = {s["name"]: s for s in json.loads(WB_MAN.read_text())["scenes"]}
    scenes = []
    for n in sorted(allow):
        r = allow[n]
        pv = S.CAL / r["shoot"] / "xraw-phase3.tiff"
        if S.sha256(pv) != wbm[n]["provia_sha256"] or S.sha256(S.CAL / r["e4"]) != r["e4_sha256"]:
            S.fail(f"{n}: reference hash mismatch")
        if S.sha256(S.CAL / r["fjlrg"]) != r["fjlrg_sha256"]:
            S.fail(f"{n}: .fjlrg hash mismatch")
        scenes.append({"name": n, "shoot": r["shoot"], "session": r["session"], "asShotWhiteBalance": r["asShotWhiteBalance"],
                       "fjlrg": r["fjlrg"], "fjlrg_sha256": r["fjlrg_sha256"], "provia": f"{r['shoot']}/xraw-phase3.tiff",
                       "provia_sha256": wbm[n]["provia_sha256"], "e4": r["e4"], "e4_sha256": r["e4_sha256"],
                       "sample_seed": scene_seed(n)})
    out = {"status": "FROZEN before any fitting (Codex scope 0dc109c)",
           "bindings": {"contract_v1": S.V1_SHA, "contract_v2": S.V2_SHA, "folds": S.FOLDS_SHA, "step1_run_manifest": S.RUNMAN_SHA,
                        "cohort_proposal": S.sha256(S.DIAG / "cuban-cohort-proposal.json"), "r11_seed42": S.SOURCE_SHA,
                        "wb_assay_manifest": S.sha256(WB_MAN), "decomposition": S.sha256(DECOMP)},
           "model": {"space": "sRGB-encoded [0,1] from target_tiff linear loader (piecewise sRGB OETF)",
                     "centering": "c = 2x - 1", "monomials_ijk_order": MONOMIALS, "coefficients": 60,
                     "form": "T(x) = x + Phi(c) @ W, W (20,3); W=0 identity"},
           "fit": {"objective": "per output channel: mean over scenes of mean-over-sampled-pixels squared encoded residual "
                                "+ lambda * ||W_channel||^2 (sum over channels)",
                   "solver": "numpy.linalg.solve(mean_s Phi^T Phi/n + lambda I, mean_s Phi^T (y-x)/n), float64",
                   "lambda": LAM, "penalised": "all 60 including intercepts", "scene_weights": "equal",
                   "samples_per_scene": N_SAMPLE, "sampling": "without replacement, numpy default_rng(42 XOR uint64_le(sha256(name)[:8])), sorted",
                   "search": None},
           "scoring": {"output": "clip [0,1], record pre-clip out-of-range pixel fraction, decode to linear, frozen Lab/CIEDE2000",
                       "primary": "adaptation-OOF T(Fuji Provia) vs E4, against identity Fuji Provia vs E4",
                       "secondary": "same fold T on frozen R11 seed 42 predictions (linear clipped [0,1] first, fraction recorded); "
                                    "untouched R11 and step-1 (in-sample, from decomposition) as labelled context",
                       "per_scene_E4_limits": LIM, "rho": "app_b = transform input, app_s = transform output, ref_b Provia, ref_s E4",
                       "set_rule": None},
           "scenes": scenes}
    MANIFEST.write_text(json.dumps(out, indent=1))
    print("manifest sha256", S.sha256(MANIFEST))


def load_manifest():
    if MANIFEST_SHA == "__UNPINNED__" or S.sha256(MANIFEST) != MANIFEST_SHA:
        S.fail("transfer manifest missing, unpinned or changed")
    S.preflight()
    return json.loads(MANIFEST.read_text())


def load_refs(m, name):
    by = {s["name"]: s for s in m["scenes"]}
    if name not in by:
        S.fail(f"{name} not in the F24 transfer manifest - refused")
    s = by[name]
    for key in ("fjlrg", "provia", "e4"):
        if S.sha256(S.CAL / s[key]) != s[key + "_sha256"]:
            S.fail(f"{name}: {key} hash mismatch")
    x, w, h = load_fjlrg_linear(S.CAL / s["fjlrg"])
    pv = load_xraw_target_linear_matching(S.CAL / s["provia"], w, h).astype(np.float64)
    e4 = load_xraw_target_linear_matching(S.CAL / s["e4"], w, h).astype(np.float64)
    return x, pv, e4


# ------------------------------------------------------------ smoke
def cmd_smoke():
    m = load_manifest()
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (5000, 3))
    out, oor = apply(np.zeros((20, 3)), x)
    assert np.array_equal(out, x) and oor == 0.0, "identity failed"
    lin = rng.uniform(0, 1, (5000, 3))
    rt = float(np.abs(decode(encode(lin)) - lin).max())
    assert rt < 1e-12, rt
    # synthetic affine recovery (degree <= 1 residual), unregularised fixture
    Wt = np.zeros((20, 3))
    Wt[:4] = rng.normal(0, 0.05, (4, 3))
    xs = [rng.uniform(0, 1, (4000, 3)) for _ in range(3)]
    stats = [normal_stats(a, a + phi(a) @ Wt) for a in xs]
    W0, cond0 = solve(stats, 0.0)
    assert np.abs(W0 - Wt).max() < 1e-9, np.abs(W0 - Wt).max()
    Wl, cond = solve(stats, LAM)
    assert np.isfinite(Wl).all()
    # sample determinism
    assert np.array_equal(sample_idx("DSCF0391.RAF", 1536 * 1026), sample_idx("DSCF0391.RAF", 1536 * 1026))
    # refusals: K scene, mismatched hash
    import subprocess
    r = subprocess.run([sys.executable, "-c", "import run_cn_ref_transfer as T; T.load_refs(T.load_manifest(), 'DSCF0515.RAF')"],
                       cwd=S.TRAIN, capture_output=True, text=True)
    assert r.returncode == 2 and "refused" in r.stdout, (r.stdout, r.stderr)
    bad = json.loads(json.dumps(m))
    bad["scenes"][0]["e4_sha256"] = "0" * 64
    try:
        load_refs(bad, bad["scenes"][0]["name"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("hash mismatch not refused")
    print(f"smoke ok: identity exact; roundtrip max {rt:.1e}; affine recovery max err {np.abs(W0 - Wt).max():.1e}; "
          f"ridge finite (cond {cond:.1f}); K and hash-mismatch refused")


# ------------------------------------------------------------ run
def metrics(app_lab, ref_lab):
    e = ciede2000(app_lab, ref_lab)
    L = ref_lab[..., 0]
    dL = app_lab[..., 0] - L
    Cr, Ca = np.hypot(ref_lab[..., 1], ref_lab[..., 2]), np.hypot(app_lab[..., 1], app_lab[..., 2])
    dh = (np.degrees(np.arctan2(app_lab[..., 2], app_lab[..., 1]) - np.arctan2(ref_lab[..., 2], ref_lab[..., 1])) + 180) % 360 - 180
    bands = {"shadow": L < 25, "mid": (L >= 25) & (L < 75), "high": L >= 75}
    return {"mean_dE00": float(e.mean()), "p95_dE00": float(np.percentile(e, 95)),
            "signed_dL_by_tone_band": {b: (float(dL[r].mean()) if r.any() else None) for b, r in bands.items()},
            "signed_dC_ref_C_gt_10": float((Ca - Cr)[Cr > 10].mean()) if (Cr > 10).any() else None,
            "signed_hue_deg_ref_C_gt_20": float(dh[Cr > 20].mean()) if (Cr > 20).any() else None}


def e4_checks(mt, rho):
    mid = mt["signed_dL_by_tone_band"]["mid"]
    c = {"mean_dE00": mt["mean_dE00"] <= LIM["mean_dE00"], "p95_dE00": mt["p95_dE00"] <= LIM["p95_dE00"],
         "abs_mid_dL": mid is not None and abs(mid) <= LIM["abs_mid_dL"]}
    if rho is not None:
        c["rho_provia_to_E4"] = rho <= LIM["rho_provia_to_E4"]
    return c


def cmd_run():
    if REPORT.exists():
        S.fail("report exists - not overwriting")
    self_test()
    m = load_manifest()
    folds = json.loads(S.FOLDS.read_text())["folds"]
    from eval_cuban_step1_k import infer, load_model
    dev = S.device()
    r11 = load_model(S.SOURCE, dev)
    dec = json.loads(DECOMP.read_text())["per_scene"]
    cache, stats, samp_hash = {}, {}, {}
    for s in m["scenes"]:
        n = s["name"]
        x, pv, e4 = load_refs(m, n)
        r11_lin = infer(r11, x, dev)
        cache[n] = (pv, e4, r11_lin)
        xe, ye = encode(pv).reshape(-1, 3), encode(e4).reshape(-1, 3)
        idx = sample_idx(n, len(xe))
        samp_hash[n] = arr_sha(idx)
        stats[n] = normal_stats(xe[idx], ye[idx])
        print("loaded", n, flush=True)
    COEFS.mkdir(exist_ok=True)
    per, fold_out = {}, {}
    for fo in folds:
        va = fo["validation_scenes"]
        tr = [s["name"] for s in m["scenes"] if s["name"] not in va]
        W, cond = solve([stats[n] for n in tr], LAM)
        cf = {"fold": fo["fold"], "train": tr, "validation": va, "lambda": LAM, "monomials_ijk_order": MONOMIALS,
              "W_20x3": W.tolist(), "W_sha256": arr_sha(W), "condition_number": cond,
              "train_sample_index_sha256": {n: samp_hash[n] for n in tr}, "manifest_sha256": MANIFEST_SHA}
        p = COEFS / f"fold{fo['fold']}.json"
        p.write_text(json.dumps(cf, indent=1))
        fold_out[fo["fold"]] = {"file": str(p.relative_to(S.TRAIN.parent)), "file_sha256": S.sha256(p), "W_sha256": cf["W_sha256"],
                                "condition_number": cond, "max_abs_coefficient": float(np.abs(W).max())}
        for n in va:
            pv, e4, r11_lin = cache[n]
            P, E = linear_srgb_to_lab(pv), linear_srgb_to_lab(e4)
            den = float(ciede2000(P, E).mean())
            tp, oor_p = apply(W, encode(pv))
            TP = linear_srgb_to_lab(decode(tp))
            r11_clip = float(((r11_lin < 0) | (r11_lin > 1)).any(-1).mean())
            R = linear_srgb_to_lab(np.clip(r11_lin, 0, 1))
            tr11, oor_r = apply(W, encode(r11_lin))
            TR = linear_srgb_to_lab(decode(tr11))
            rows = {"identity_provia": metrics(P, E), "T_provia": metrics(TP, E),
                    "r11_untouched": metrics(R, E), "T_r11": metrics(TR, E)}
            rho_tp = float(ciede2000(TP, E).mean()) / den
            rho_tr = float(ciede2000(P + TR - R, E).mean()) / den
            rows["T_provia"].update(rho_provia_to_E4=rho_tp, pre_clip_out_of_range=oor_p)
            rows["T_provia"]["checks"] = e4_checks(rows["T_provia"], rho_tp)
            rows["T_r11"].update(rho_provia_to_E4=rho_tr, pre_clip_out_of_range=oor_r, input_linear_clip_fraction=r11_clip)
            rows["T_r11"]["checks"] = e4_checks(rows["T_r11"], rho_tr)
            rows["identity_provia"]["checks"] = e4_checks(rows["identity_provia"], None)
            rows["r11_untouched"]["checks"] = e4_checks(rows["r11_untouched"], None)
            for kk in ("T_provia", "T_r11", "identity_provia", "r11_untouched"):
                rows[kk]["meets_all_E4_limits"] = all(rows[kk]["checks"].values())
            ctx = dec[n]
            per[n] = {"fold": fo["fold"], "session": ctx["session"], "asShotWhiteBalance": ctx["asShotWhiteBalance"], **rows,
                      "diff_T_provia_minus_identity_mean_dE00": rows["T_provia"]["mean_dE00"] - rows["identity_provia"]["mean_dE00"],
                      "diff_T_r11_minus_r11_mean_dE00": rows["T_r11"]["mean_dE00"] - rows["r11_untouched"]["mean_dE00"],
                      "context_step1_seed42_vs_E4_IN_SAMPLE": ctx["2_cn_vs_e4"]["mean_dE00"],
                      "context_r11_vs_provia_IN_SAMPLE": ctx["1_r11_vs_provia"]["mean_dE00"]}
            print(n, f"identity {rows['identity_provia']['mean_dE00']:.2f} T(P) {rows['T_provia']['mean_dE00']:.2f} "
                     f"rho {rho_tp:.2f} | R11 {rows['r11_untouched']['mean_dE00']:.2f} T(R11) {rows['T_r11']['mean_dE00']:.2f}", flush=True)
    names = sorted(per)
    def col(a, k):
        return [per[n][a][k] for n in names]
    def summ(a):
        out = {"mean_of_means": float(np.mean(col(a, "mean_dE00"))), "median_mean": float(np.median(col(a, "mean_dE00"))),
               "mean_p95": float(np.mean(col(a, "p95_dE00"))),
               "cross_scene_mean_mid_dL": float(np.mean([per[n][a]["signed_dL_by_tone_band"]["mid"] for n in names
                                                         if per[n][a]["signed_dL_by_tone_band"]["mid"] is not None])),
               "cross_scene_mean_dC": float(np.mean([v for v in col(a, "signed_dC_ref_C_gt_10") if v is not None])),
               "cross_scene_mean_hue_deg": float(np.mean([v for v in col(a, "signed_hue_deg_ref_C_gt_20") if v is not None])),
               "scenes_meeting_all_E4_limits": int(sum(per[n][a]["meets_all_E4_limits"] for n in names)),
               "per_check_pass_counts": {c: int(sum(per[n][a]["checks"][c] for n in names)) for c in per[names[0]][a]["checks"]}}
        if "rho_provia_to_E4" in per[names[0]][a]:
            out["median_rho"] = float(np.median(col(a, "rho_provia_to_E4")))
        return out
    sessions = {}
    for n in names:
        sessions.setdefault(per[n]["session"], []).append(n)
    groups = {str(s): {"n": len(v), "scenes": v, **{a: float(np.mean([per[x][a]["mean_dE00"] for x in v]))
                                                     for a in ("identity_provia", "T_provia", "r11_untouched", "T_r11")}}
              for s, v in sorted(sessions.items())}
    out = {"scope": "F24 reference-transfer assay; adaptation-OOF by whole session for the transform; T(Provia) uses an ORACLE "
                    "Fuji reference input, not a runtime RAW decode; R11 trained on these Provia scenes (internal diagnostic only)",
           "manifest_sha256": MANIFEST_SHA, "folds": fold_out,
           "summary": {a: summ(a) for a in ("identity_provia", "T_provia", "r11_untouched", "T_r11")},
           "improved_T_provia_vs_identity": int(sum(per[n]["diff_T_provia_minus_identity_mean_dE00"] < 0 for n in names)),
           "improved_T_r11_vs_r11": int(sum(per[n]["diff_T_r11_minus_r11_mean_dE00"] < 0 for n in names)),
           "session_groups": groups, "per_scene": per,
           "note": "no F24 set acceptance rule; per-scene E4 limit counts are descriptive"}
    S.check_source()
    REPORT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("folds", "summary", "improved_T_provia_vs_identity", "improved_T_r11_vs_r11")}, indent=1))


if __name__ == "__main__":
    {"prepare": cmd_prepare, "smoke": cmd_smoke, "run": cmd_run}[sys.argv[1]]()
