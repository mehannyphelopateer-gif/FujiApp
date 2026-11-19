"""Classic Cuban Neg - reference-only effect / measurability audit under
frozen contract v1. Fuji vs Fuji only: no app output, no model, no fitting,
no monitor or finalHeldOut data.

For the 8 C scenes (train tier): each control C1-C8 against its own E5 base,
and the recipe stages Provia->E4, E4->E5 (DR400 region), E5->E3, E3->E2 (WB),
plus Fuji grain statistics (E1-E2). Applies the contract's region-occupancy,
minimum-effect and detail-floor rules and reports, per control, how many of
the 8 scenes are measurable (contract needs >= 6, else UNVALIDATED).

  python3 training/cuban_reference_audit.py -> training/diagnostics/cuban-reference-audit.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402
from cuban_contract import CONTRACT, contract_sha256  # noqa: E402
from cuban_exports import BATCHES, CAL, CONTROLS, INSTALL_NAME, REPORT, VARIANTS, sha256  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from target_tiff import load_xraw_target_linear_matching  # noqa: E402

FROZEN_SHA = "c3f840ef9fad9d48de83b4edf183f2434523f352f358305b91eb7764d22d0067"
OUT = Path(__file__).resolve().parent / "diagnostics" / "cuban-reference-audit.json"
FJLRG = "browser-phase3-linear.fjlrg"
M, FLOOR = CONTRACT["measurability"], CONTRACT["repeat_floor"]
MIN_EFFECT = max(M["min_fuji_effect_mean_dE00"], M["min_fuji_effect_vs_floor"] * FLOOR["mean_dE00"])
BANDS = {"fine_s1_s3": (1.0, 3.0), "mid_s4_s16": (4.0, 16.0)}  # as cuban_repeat_floor.py


def region(name, lab, lin):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C, h = np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360
    return {"dr400": (L >= 70) | (lin >= 0.999).any(-1), "C1": L >= 60, "C2": L <= 35, "C3": C >= 20,
            "C4": C >= 30, "C5": (h >= 200) & (h <= 290) & (C >= 15)}.get(name, np.ones(L.shape, bool))


def detail_rms(L, band):
    lo, hi = BANDS[band]
    d = cv2.GaussianBlur(L, (0, 0), lo) - cv2.GaussianBlur(L, (0, 0), hi)
    return float(np.sqrt(np.mean(d ** 2)))


def effect(base_lab, base_lin, tgt_lab, reg):
    R = region(reg, base_lab, base_lin)
    occ = float(R.mean())
    out = {"region": reg, "occupancy": round(occ, 4)}
    if occ < M["min_region_fraction"] or not R.any():
        return {**out, "measurable": False, "why": "region below minimum occupancy",
                "effect_mean_dE00": None if not R.any() else round(float(ciede2000(base_lab[R], tgt_lab[R]).mean()), 4)}
    e = ciede2000(base_lab[R], tgt_lab[R])
    d = tgt_lab[R] - base_lab[R]
    Cb, Ct = np.hypot(*np.moveaxis(base_lab[R][:, 1:], -1, 0)), np.hypot(*np.moveaxis(tgt_lab[R][:, 1:], -1, 0))
    out.update({"effect_mean_dE00": round(float(e.mean()), 4), "effect_p95_dE00": round(float(np.percentile(e, 95)), 4),
                "mean_dL": round(float(d[:, 0].mean()), 4), "mean_dC": round(float((Ct - Cb).mean()), 4),
                "mean_da_db": [round(float(d[:, 1].mean()), 4), round(float(d[:, 2].mean()), 4)]})
    out["measurable"] = bool(e.mean() >= MIN_EFFECT)
    if not out["measurable"]:
        out["why"] = f"Fuji effect {e.mean():.3f} < {MIN_EFFECT:.3f}"
    return out


def verified_hashes():
    h = {}
    for kind in ("pilot", "controls", "cohort1"):
        rep = json.loads(REPORT[kind].read_text())
        if rep.get("passed") is not True:
            sys.exit(f"refused: {kind} verification has not passed")
        for e in rep["entries"]:
            h[(e["name"], e["variant"])] = e["source_sha256"]
    return h


def main():
    self_test()
    if contract_sha256() != FROZEN_SHA:
        sys.exit("refused: contract hash != frozen v1")
    hashes = verified_hashes()
    rows = json.loads(BATCHES["controls"][1].read_text())
    res = {"contract_sha256": FROZEN_SHA, "min_effect_dE00": MIN_EFFECT, "scenes": {}, "controls_summary": {},
           "stages_summary": {}}
    for r in rows:
        _, w, h = load_fjlrg_linear(CAL / r["shoot"] / FJLRG)
        lab, lin, files = {}, {}, {}
        for v in [*VARIANTS, *CONTROLS]:
            p = CAL / r["shoot"] / INSTALL_NAME[v]
            if sha256(p) != hashes[(r["name"], v)]:
                sys.exit(f"refused: {p.relative_to(CAL)} != verified hash")
            files[v] = p
        files["provia"] = CAL / r["shoot"] / "xraw-phase3.tiff"
        for v, p in files.items():
            li = load_xraw_target_linear_matching(p, w, h).astype(np.float64)
            lin[v], lab[v] = li, linear_srgb_to_lab(li)
        base = "E5-classic-neg-dr400"
        sc = {"session": r["session"], "iso": r["iso"], "asShotWB": r["asShotWhiteBalance"],
              "highlightClass": r["highlightClass"], "provia_sha256": sha256(files["provia"]), "controls": {}}
        for c, spec in CONTRACT["controls"].items():
            if not isinstance(spec, dict):
                continue
            if "detail_band" in spec:
                band = spec["detail_band"]
                b0, b1 = detail_rms(lab[base][..., 0], band), detail_rms(lab[c][..., 0], band)
                rel = (b1 - b0) / b0
                thr = M["detail_min_vs_floor"] * FLOOR["detail_rel"][band]
                ent = {"region": "frame", "occupancy": 1.0, "detail_band": band, "detail_rms": [round(b0, 4), round(b1, 4)],
                       "signed_rel_change": round(rel, 5), "threshold": thr, "measurable": bool(abs(rel) >= thr),
                       "frame_effect_mean_dE00": round(float(ciede2000(lab[base], lab[c]).mean()), 4)}
            else:
                ent = effect(lab[base], lin[base], lab[c], spec["region"])
            sc["controls"][c] = ent
        st = {"provia_to_E4": effect(lab["provia"], lin["provia"], lab["E4-classic-neg-only"], "frame"),
              "E4_to_E5_dr400": effect(lab["E4-classic-neg-only"], lin["E4-classic-neg-only"], lab[base], "dr400"),
              "E5_to_E3": effect(lab[base], lin[base], lab["E3-as-shot-wb"], "frame"),
              "E3_to_E2_wb": effect(lab["E3-as-shot-wb"], lin["E3-as-shot-wb"], lab["E2-no-grain"], "frame")}
        st["E4_to_E5_dr400"]["clipped_fraction"] = [round(float((lin[v] >= 0.999).any(-1).mean()), 5)
                                                     for v in ("E4-classic-neg-only", base)]
        g = lab["E1-full-recipe"] - lab["E2-no-grain"]
        st["grain_fuji"] = {"residual_std_Lab": [round(float(g[..., i].std()), 4) for i in range(3)],
                            "residual_mean_L": round(float(g[..., 0].mean()), 4)}
        sc["stages"] = st
        res["scenes"][r["name"]] = sc
        print(r["name"], {c: (e["measurable"], e.get("effect_mean_dE00", e.get("signed_rel_change")), e["occupancy"])
                          for c, e in sc["controls"].items()}, flush=True)
    need = CONTRACT["controls"]["min_measurable_scenes"]
    for c in sc["controls"]:
        ents = {n: s["controls"][c] for n, s in res["scenes"].items()}
        nm = sum(e["measurable"] for e in ents.values())
        res["controls_summary"][c] = {"measurable": nm, "of": len(ents),
                                      "status": "sufficient" if nm >= need else "UNVALIDATED - needs more train-only examples",
                                      "unmeasurable": {n: e.get("why", "detail change below floor threshold")
                                                       for n, e in ents.items() if not e["measurable"]}}
    for s in sc["stages"]:
        if s == "grain_fuji":
            continue
        ents = {n: x["stages"][s] for n, x in res["scenes"].items()}
        res["stages_summary"][s] = {"measurable": sum(e["measurable"] for e in ents.values()), "of": len(ents),
                                    "unmeasurable": {n: e.get("why") for n, e in ents.items() if not e["measurable"]}}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({"controls": res["controls_summary"], "stages": res["stages_summary"]}, indent=1))


if __name__ == "__main__":
    main()
