"""CN step 1 - F24-only error decomposition (Codex scope, df2b052). No fitting,
no K / monitor / finalHeldOut / T8 access, no app change.

Per F scene, all at the contract working scale (target_tiff loader, frozen
Lab / CIEDE2000):
  1. Round 11 seed 42 (frozen) vs its actual Fuji PROVIA reference.
  2. Fuji Provia -> E4 and step-1 seed 42 (frozen) vs E4.
  3. Lab vectors: e_base = R11 - Provia, D = CN - R11, D_ref = E4 - Provia,
     e_delta = D - D_ref, e_total = CN - E4 = e_base + e_delta (checked).
     Oracle-anchored render Provia + D vs E4 (Delta E00). Energy split
     |e_total|^2 = |e_base|^2 + |e_delta|^2 + 2 e_base.e_delta, so
     cancellation is reported rather than inferred from magnitudes.
  4. Frame and E4-tone-band signed residuals, grouped / correlated with the
     already-recorded source WB shift and source-vs-reference gain ratios
     (descriptive, confounded; never runtime inputs).

IN-SAMPLE: both models trained on these 24 scenes' references (R11 on
Provia, step 1 on E4). CV fold checkpoints were not stored, so no OOF
pixel decomposition exists; stored OOF per-scene linear L1 from the CV
logs is attached as labelled context only.

  python3 training/diagnose_cn_step1_f24.py
    -> training/diagnostics/cn-step1-f24-decomposition.json
    -> training/diagnostics/cn-step1-f24-visual/<scene>.jpg (fixed 4 examples)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuban_step1 as S  # noqa: E402
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402
from eval_cuban_step1_k import infer, load_model  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from target_tiff import load_xraw_target_linear_matching  # noqa: E402
from rank_metrics import spearman  # noqa: E402

OUT = S.DIAG / "cn-step1-f24-decomposition.json"
VIS = S.DIAG / "cn-step1-f24-visual"
WB_AUDIT = S.DIAG / "cn-step1-wb-reference-audit.json"
VISUAL_SCENES = ["DSCF0523.RAF", "DSCF0964.RAF", "DSCF0514.RAF", "DSCF0511.RAF"]
CN_CKPT = "cn-step1-seed42.pt"
BANDS = {"shadow": (None, 25), "mid": (25, 75), "high": (75, None)}


def band_masks(L):
    return {b: ((L >= lo) if lo is not None else True) & ((L < hi) if hi is not None else True)
            for b, (lo, hi) in BANDS.items()}


def vec_stats(v, mask):
    if not mask.any():
        return None
    x = v[mask]
    return {"mean_dL": float(x[:, 0].mean()), "mean_da": float(x[:, 1].mean()), "mean_db": float(x[:, 2].mean()),
            "mean_abs_dE76": float(np.linalg.norm(x, axis=1).mean())}


def de_stats(a, b):
    e = ciede2000(a, b)
    return {"mean_dE00": float(e.mean()), "p95_dE00": float(np.percentile(e, 95))}, e


def chroma_hue(app, ref, mask_c10, mask_c20):
    Ca, Cr = np.hypot(app[..., 1], app[..., 2]), np.hypot(ref[..., 1], ref[..., 2])
    dh = (np.degrees(np.arctan2(app[..., 2], app[..., 1]) - np.arctan2(ref[..., 2], ref[..., 1])) + 180) % 360 - 180
    return {"signed_dC_ref_C_gt_10": float((Ca - Cr)[mask_c10].mean()) if mask_c10.any() else None,
            "signed_hue_deg_ref_C_gt_20": float(dh[mask_c20].mean()) if mask_c20.any() else None}


def srgb8(lin):
    x = np.clip(lin, 0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (x * 255 + 0.5).astype(np.uint8)


def heat(e, vmax=10.0):
    g = (np.clip(e / vmax, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def panel(name, imgs, heats):
    h = 420
    def rs(a):
        return cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h), interpolation=cv2.INTER_AREA)
    row1 = np.concatenate([label(rs(srgb8(v)), k) for k, v in imgs.items()], axis=1)
    row2 = np.concatenate([label(rs(heat(v)), k) for k, v in heats.items()], axis=1)
    pad = np.zeros((h, row1.shape[1] - row2.shape[1], 3), np.uint8)
    row2 = np.concatenate([row2, label(pad, "heat: 0-10 dE00 (dE76 for |e_delta|)") if pad.shape[1] > 300 else pad], axis=1)
    VIS.mkdir(exist_ok=True)
    out = VIS / f"{name.split('.')[0]}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate([row1, row2], axis=0), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def main():
    if OUT.exists():
        S.fail(f"{OUT.name} exists - not overwriting diagnostic evidence")
    self_test()
    man, folds, allow, k = S.preflight()
    fr = json.loads(S.FREEZE.read_text())
    if S.sha256(S.CKPT / CN_CKPT) != fr["checkpoints"][CN_CKPT]:
        S.fail("step-1 seed 42 checkpoint != frozen hash")
    wb = {r["name"]: r for r in json.loads(WB_AUDIT.read_text())["scenes"]}
    if set(wb) != set(allow):
        S.fail("WB audit scenes != F allowlist")
    oof = {}
    sel = json.loads(S.SELECTION.read_text())["chosen"]
    for fo in folds["folds"]:
        h = json.loads(S.cv_path(sel["lr"], fo["fold"]).read_text())["history"]
        for n in fo["validation_scenes"]:
            oof[n] = {"epoch0_warmstart_l1_vs_E4": h[0]["val"][n], "chosen_epoch_l1_vs_E4": h[sel["epoch"]]["val"][n]}
    dev = S.device()
    r11, cn = load_model(S.SOURCE, dev), load_model(S.CKPT / CN_CKPT, dev)
    rows, agg_vecs = {}, {"e_base": [], "e_delta": [], "e_total": []}
    for n in sorted(allow):
        r = allow[n]
        d = S.CAL / r["shoot"]
        fj, e4p, pvp = d / "browser-phase3-linear.fjlrg", S.CAL / r["e4"], d / "xraw-phase3.tiff"
        if S.sha256(fj) != r["fjlrg_sha256"] or S.sha256(e4p) != r["e4_sha256"]:
            S.fail(f"{n}: input/E4 hash != run manifest")
        x, w, h = load_fjlrg_linear(fj)
        lin = {"input": x.astype(np.float64),
               "provia_ref": load_xraw_target_linear_matching(pvp, w, h).astype(np.float64),
               "e4_ref": load_xraw_target_linear_matching(e4p, w, h).astype(np.float64),
               "r11_pred": infer(r11, x, dev), "cn_pred": infer(cn, x, dev)}
        lab = {kk: linear_srgb_to_lab(v) for kk, v in lin.items()}
        P, E, R, C = lab["provia_ref"], lab["e4_ref"], lab["r11_pred"], lab["cn_pred"]
        e_base, D, D_ref = R - P, C - R, E - P
        e_delta, e_total = D - D_ref, C - E
        ident = float(np.abs(e_total - (e_base + e_delta)).max())
        s_base, de_base = de_stats(R, P)
        s_fuji, _ = de_stats(P, E)
        s_cn, de_cn = de_stats(C, E)
        s_oracle, _ = de_stats(P + D, E)
        s_r11e4, _ = de_stats(R, E)
        frame = np.ones(P.shape[:2], bool)
        bE, bP = band_masks(E[..., 0]), band_masks(P[..., 0])
        CrE, CrP = np.hypot(E[..., 1], E[..., 2]), np.hypot(P[..., 1], P[..., 2])
        sq = lambda v: (v ** 2).sum(-1)
        energy = {"mean_sq_total": float(sq(e_total).mean()), "mean_sq_base": float(sq(e_base).mean()),
                  "mean_sq_delta": float(sq(e_delta).mean()),
                  "mean_cross_2_base_dot_delta": float((2 * (e_base * e_delta).sum(-1)).mean())}
        energy["cross_fraction_of_parts"] = energy["mean_cross_2_base_dot_delta"] / (energy["mean_sq_base"] + energy["mean_sq_delta"])
        dmag, drefmag = np.linalg.norm(D, axis=-1), np.linalg.norm(D_ref, axis=-1)
        dot = (D * D_ref).sum(-1)
        row = {
            "session": r["session"], "iso": r["iso"], "asShotWhiteBalance": r["asShotWhiteBalance"],
            "source_wb_shift_steps": wb[n]["source_wb_shift_steps"],
            "source_gain_RB_over_G": wb[n]["libraw_gain_RB_over_G"],
            "reference_level_RB_over_G": wb[n]["provia_recorded_level_RB_over_G"],
            "log_gain_ratio_source_over_ref": [float(np.log(wb[n]["libraw_gain_RB_over_G"][i] / wb[n]["provia_recorded_level_RB_over_G"][i])) for i in (0, 1)],
            "1_r11_vs_provia": {**s_base, **chroma_hue(R, P, CrP > 10, CrP > 20),
                                "signed_by_provia_tone_band": {b: vec_stats(e_base, m) for b, m in bP.items()},
                                "frame": vec_stats(e_base, frame)},
            "2_fuji_provia_vs_e4": s_fuji,
            "2_cn_vs_e4": {**s_cn, **chroma_hue(C, E, CrE > 10, CrE > 20)},
            "context_r11_vs_e4": s_r11e4,
            "3_oracle_anchored_provia_plus_D_vs_e4": s_oracle,
            "3_rho_in_sample": s_oracle["mean_dE00"] / s_fuji["mean_dE00"],
            "3_delta": {"mean_abs_D_ref_dE76": float(drefmag.mean()), "mean_abs_D_dE76": float(dmag.mean()),
                        "gain_ratio_D_on_D_ref": float(dot.sum() / (drefmag ** 2).sum()),
                        "cosine_D_D_ref_weighted": float(dot.sum() / np.sqrt((dmag ** 2).sum() * (drefmag ** 2).sum())),
                        "D_ref_signed_frame": vec_stats(D_ref, frame), "D_signed_frame": vec_stats(D, frame)},
            "3_decomposition_e4_bands": {b: {"e_total": vec_stats(e_total, m), "e_base": vec_stats(e_base, m),
                                             "e_delta": vec_stats(e_delta, m), "occupancy": float(m.mean())}
                                         for b, m in {"frame": frame, **bE}.items()},
            "3_energy": energy, "3_identity_max_abs": ident,
            "oof_context_linear_l1_vs_E4 (session-held-out CV logs, not a pixel decomposition)": oof[n],
            "files": {"fjlrg_sha256": r["fjlrg_sha256"], "e4_sha256": r["e4_sha256"], "provia_sha256": S.sha256(pvp)},
        }
        rows[n] = row
        for kk, v in (("e_base", e_base), ("e_delta", e_delta), ("e_total", e_total)):
            agg_vecs[kk].append(v.reshape(-1, 3).mean(0))
        if n in VISUAL_SCENES:
            p = panel(n, {"input .fjlrg": lin["input"], "R11 pred": lin["r11_pred"], "Fuji Provia": lin["provia_ref"],
                          "Step1 CN pred": lin["cn_pred"], "Fuji E4 (CN)": lin["e4_ref"]},
                      {"dE00 R11 vs Provia": de_base, "dE00 CN vs E4": de_cn,
                       "|e_delta| dE76": np.linalg.norm(e_delta, axis=-1)})
            row["visual"] = str(p.relative_to(S.TRAIN.parent))
        print(n, f"R11vsProv {s_base['mean_dE00']:.2f}  FujiP->E4 {s_fuji['mean_dE00']:.2f}  CNvsE4 {s_cn['mean_dE00']:.2f}  "
                 f"oracle(P+D) {s_oracle['mean_dE00']:.2f}  sq base/delta/cross {energy['mean_sq_base']:.1f}/"
                 f"{energy['mean_sq_delta']:.1f}/{energy['mean_cross_2_base_dot_delta']:.1f}  shift {row['source_wb_shift_steps']}", flush=True)
    names = sorted(rows)
    def col(f):
        return [f(rows[n]) for n in names]
    metrics = {
        "r11_vs_provia_mean_dE00": col(lambda r: r["1_r11_vs_provia"]["mean_dE00"]),
        "cn_vs_e4_mean_dE00": col(lambda r: r["2_cn_vs_e4"]["mean_dE00"]),
        "oracle_mean_dE00": col(lambda r: r["3_oracle_anchored_provia_plus_D_vs_e4"]["mean_dE00"]),
        "fuji_provia_e4_mean_dE00": col(lambda r: r["2_fuji_provia_vs_e4"]["mean_dE00"]),
        "e_base_mean_da": col(lambda r: r["1_r11_vs_provia"]["frame"]["mean_da"]),
        "e_base_mean_db": col(lambda r: r["1_r11_vs_provia"]["frame"]["mean_db"]),
        "e_base_mean_dL": col(lambda r: r["1_r11_vs_provia"]["frame"]["mean_dL"]),
        "e_delta_mean_da": col(lambda r: r["3_decomposition_e4_bands"]["frame"]["e_delta"]["mean_da"]),
        "e_delta_mean_db": col(lambda r: r["3_decomposition_e4_bands"]["frame"]["e_delta"]["mean_db"]),
    }
    wbvars = {"shift_R_steps": col(lambda r: r["source_wb_shift_steps"][0]),
              "shift_B_steps": col(lambda r: r["source_wb_shift_steps"][1]),
              "log_gain_ratio_R": col(lambda r: r["log_gain_ratio_source_over_ref"][0]),
              "log_gain_ratio_B": col(lambda r: r["log_gain_ratio_source_over_ref"][1])}
    corr = {m: {w: round(spearman(mv, wv), 3) for w, wv in wbvars.items()} for m, mv in metrics.items()}
    groups = {}
    for keyname, keyf in (("asShotWhiteBalance", lambda r: r["asShotWhiteBalance"]),
                          ("source_wb_shift_steps", lambda r: str(r["source_wb_shift_steps"]))):
        g = {}
        for n in names:
            g.setdefault(keyf(rows[n]), []).append(n)
        groups[keyname] = {kk: {"n": len(v), "scenes": v,
                                **{m: float(np.mean([metrics[m][names.index(s)] for s in v])) for m in metrics}}
                           for kk, v in g.items()}
    E = {kk: np.array(v) for kk, v in agg_vecs.items()}
    summary = {
        "median_r11_vs_provia_mean_dE00": float(np.median(metrics["r11_vs_provia_mean_dE00"])),
        "median_fuji_provia_vs_e4_mean_dE00": float(np.median(metrics["fuji_provia_e4_mean_dE00"])),
        "median_cn_vs_e4_mean_dE00": float(np.median(metrics["cn_vs_e4_mean_dE00"])),
        "median_oracle_provia_plus_D_vs_e4_mean_dE00": float(np.median(metrics["oracle_mean_dE00"])),
        "median_in_sample_rho": float(np.median(col(lambda r: r["3_rho_in_sample"]))),
        "median_gain_ratio_D_on_D_ref": float(np.median(col(lambda r: r["3_delta"]["gain_ratio_D_on_D_ref"]))),
        "median_cosine_D_D_ref": float(np.median(col(lambda r: r["3_delta"]["cosine_D_D_ref_weighted"]))),
        "mean_energy": {kk: float(np.mean(col(lambda r, kk=kk: r["3_energy"][kk]))) for kk in rows[names[0]]["3_energy"]},
        "cross_scene_mean_signed_Lab_frame": {kk: [round(float(x), 3) for x in v.mean(0)] for kk, v in E.items()},
        "max_identity_error": max(col(lambda r: r["3_identity_max_abs"])),
    }
    out = {"scope": "F24 only, in-sample diagnostic; no fitting, no K/monitor/finalHeldOut/T8 reads, no app change",
           "in_sample_note": "R11 trained on these Provia references; step 1 trained on these E4 references. Not validation.",
           "oof_note": "CV fold checkpoints were not stored; only per-scene OOF linear L1 vs E4 from CV logs is attached",
           "groups_note": "WB groups/correlations are descriptive, n=24, confounded with session/scene content; not causal",
           "fingerprints": {"contract_v1": S.V1_SHA, "contract_v2": S.V2_SHA, "run_manifest": S.RUNMAN_SHA,
                            "folds": S.FOLDS_SHA, "freeze": S.sha256(S.FREEZE),
                            "r11_seed42": S.sha256(S.SOURCE), "cn_step1_seed42": fr["checkpoints"][CN_CKPT],
                            "wb_audit": S.sha256(WB_AUDIT), "selection": S.sha256(S.SELECTION)},
           "summary": summary, "spearman_vs_source_wb": corr, "groups": groups, "per_scene": rows}
    S.check_source()
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"summary": summary, "spearman": corr}, indent=1))


if __name__ == "__main__":
    main()
