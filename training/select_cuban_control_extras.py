"""Classic Cuban Neg - PROPOSED extra train-tier control scenes for the
controls the reference audit left UNVALIDATED (C3 Color, C4 CCE, C5 FX Blue).

Contract v1 rule: extra examples are chosen by metadata or region occupancy
BEFORE their Classic Neg pixels exist or are scored. Occupancy here is
measured on each candidate's existing Provia reference (xraw-phase3.tiff) at
the contract's working scale - no Cuban export, no app output, no model.
Classic Neg is less saturated than Provia, so margins are used.

Pool: the 456-scene train pool (capture DR400, ISO 500-6400, pairReady,
minus DSCF0752 / Shoot 10), minus every shoot in F or K. One scene per shoot.
Monitor and finalHeldOut are never read.

Rule (fixed before running): eligible if Provia blue-region occupancy
(hue 200-290 deg, C* >= 15) >= 0.06 AND C* >= 30 occupancy >= 0.06; rank by
blue occupancy (desc), filename tiebreak, at most 2 per session; take 8.

  python3 training/select_cuban_control_extras.py
    -> calibration-input/cuban-control-extras-proposal.json (+ committed copy)
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_metrics import linear_srgb_to_lab  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from target_tiff import load_xraw_target_linear_matching  # noqa: E402

CAL = Path(__file__).resolve().parent.parent / "calibration-input"
OUT = CAL / "cuban-control-extras-proposal.json"
COMMITTED = Path(__file__).resolve().parent / "diagnostics" / "cuban-control-extras-proposal.json"
MIN_BLUE, MIN_SAT30, PER_SESSION, TAKE = 0.06, 0.06, 2, 8


def occupancy(shoot):
    _, w, h = load_fjlrg_linear(CAL / shoot / "browser-phase3-linear.fjlrg")
    lab = linear_srgb_to_lab(load_xraw_target_linear_matching(CAL / shoot / "xraw-phase3.tiff", w, h).astype(np.float64))
    C, hue = np.hypot(lab[..., 1], lab[..., 2]), np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    return {"blue_C5": float(((hue >= 200) & (hue <= 290) & (C >= 15)).mean()),
            "sat30_C4": float((C >= 30).mean()), "sat20_C3": float((C >= 20).mean())}


def main():
    m = json.loads((CAL / "phase4-corpus-manifest.json").read_text())
    ex = {e["name"]: e for e in json.loads((CAL / "corpus-exif.json").read_text())}
    cohort = json.loads((CAL / "cuban-cohort-proposal.json").read_text())["scenes"]
    used = {r["shoot"] for r in cohort}
    pool = sorted((s for s in m["tiers"]["train"]["scenes"]
                   if s.get("captureDynamicRange") == 400 and 500 <= s["iso"] <= 6400 and s.get("pairReady")
                   and s["name"] in ex and s["name"] != "DSCF0752.RAF" and s["shoot"] != "Shoot 10"
                   and s["shoot"] not in used), key=lambda s: s["name"])
    by_shoot = {}
    for s in pool:
        by_shoot.setdefault(s["shoot"], s)  # one scene per shoot: first by filename
    cands = []
    for i, s in enumerate(by_shoot.values()):
        o = occupancy(s["shoot"])
        cands.append({"name": s["name"], "shoot": s["shoot"], "session": s["sessionIndex"], "iso": s["iso"],
                      "asShotWhiteBalance": hex(int(ex[s["name"]]["fuji"]["WhiteBalance"].split()[0])),
                      "highlightClass": s.get("highlightClass"), **{k: round(v, 4) for k, v in o.items()}})
        if i % 50 == 0:
            print(f"{i}/{len(by_shoot)}", flush=True)
    elig = sorted((c for c in cands if c["blue_C5"] >= MIN_BLUE and c["sat30_C4"] >= MIN_SAT30),
                  key=lambda c: (-c["blue_C5"], c["name"]))
    pick, per = [], {}
    for c in elig:
        if per.get(c["session"], 0) < PER_SESSION and len(pick) < TAKE:
            pick.append(c)
            per[c["session"]] = per.get(c["session"], 0) + 1
    out = {"status": "PROPOSED - not approved; no exports requested",
           "rule": {"min_blue": MIN_BLUE, "min_sat30": MIN_SAT30, "per_session": PER_SESSION, "take": TAKE},
           "candidates_measured": len(cands), "eligible": len(elig), "picked": pick,
           "exports_if_approved": ["E5-classic-neg-dr400", "C3-color-plus4", "C4-cce-strong", "C5-fxblue-strong"],
           "all_candidates": cands}
    OUT.write_text(json.dumps(out, indent=1))
    shutil.copyfile(OUT, COMMITTED)
    for c in pick:
        print(c)
    print(f"measured {len(cands)}, eligible {len(elig)}, picked {len(pick)}")


if __name__ == "__main__":
    main()
