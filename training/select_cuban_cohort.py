"""Classic Cuban Neg - PROPOSED train-tier reference cohort (metadata only;
no pixels read, no export folders made, finalHeldOut never read).

Pool: train tier at the recipe's capture conditions (capture DR400,
ISO 500-6400, pairReady), minus DSCF0752 (diagnostic-only) and its Shoot 10.
Train and monitor tiers share no session or shoot, so the 12 monitor
validation scenes stay outside this cohort by construction.

  F  fit set, 24 scenes (includes the 2 verified pilot scenes)
  K  train-internal check set, 8 scenes, shoot-disjoint from F; used to
     check the frozen criteria on unseen train shoots before any monitor run
  C  isolated-control subset of F, 8 scenes, for one-setting-at-a-time exports

Rules: at most one scene per shoot across F and K; filename order inside
every filter; rare WB presets (White Fluorescent 0x302, Cloudy 0x200) forced
in; per-session quotas filled slot by slot with whichever of Auto/Kelvin
is behind in the group (Auto on ties), conditions rotating bright-highlight
/ mixed-lighting / ISO>=1600 / any, falling back to the other WB. Coverage
constraints are checked and reported, never tuned to pass.

  python3 training/select_cuban_cohort.py  ->  calibration-input/cuban-cohort-proposal.json
"""
import json
from pathlib import Path

CAL = Path(__file__).resolve().parent.parent / "calibration-input"
OUT = CAL / "cuban-cohort-proposal.json"
PILOTS = ["DSCF0523.RAF", "DSCF0964.RAF"]
F_QUOTA = {1: 1, 3: 4, 7: 4, 8: 4, 11: 4, 12: 3, 13: 4}   # 24
K_QUOTA = {2: 1, 3: 1, 7: 1, 8: 2, 11: 1, 12: 1, 13: 1}   # 8
KELVIN, AUTO, FLUOR, CLOUDY = 0xFF0, 0x0, 0x302, 0x200
CONDS = ["bright", "mixed", "hiso", None]
F_MIN = {"bright-highlight": 4, "mixed-lighting": 4, "iso>=1600": 6, "iso>=5000": 2}
K_MIN = {"bright-highlight": 1, "mixed-lighting": 1, "iso>=1600": 2}


def main():
    m = json.loads((CAL / "phase4-corpus-manifest.json").read_text())
    ex = {e["name"]: e for e in json.loads((CAL / "corpus-exif.json").read_text())}
    wb = lambda s: int(ex[s["name"]]["fuji"]["WhiteBalance"].split()[0])
    pool = sorted((s for s in m["tiers"]["train"]["scenes"]
                   if s.get("captureDynamicRange") == 400 and 500 <= s["iso"] <= 6400 and s.get("pairReady")
                   and s["name"] in ex and s["name"] != "DSCF0752.RAF" and s["shoot"] != "Shoot 10"),
                  key=lambda s: s["name"])
    used_shoots, picked = set(), {"F": [], "K": []}

    def cond(s, c):
        return (c is None or (c == "bright" and s.get("highlightClass") == "bright-highlight")
                or (c == "mixed" and s.get("highlightClass") == "mixed-lighting")
                or (c == "hiso" and s["iso"] >= 1600))

    def take(group, pred):
        s = next((s for s in pool if s["shoot"] not in used_shoots and pred(s)), None)
        if s:
            picked[group].append(s)
            used_shoots.add(s["shoot"])
        return s

    for n in PILOTS:
        take("F", lambda s, n=n: s["name"] == n)
    take("F", lambda s: wb(s) == FLUOR)
    take("F", lambda s: wb(s) == FLUOR and s["sessionIndex"] != picked["F"][-1]["sessionIndex"])
    take("F", lambda s: wb(s) == CLOUDY)
    take("K", lambda s: wb(s) == FLUOR)
    # bright-highlight is the rarest condition: force the minimum in, one per
    # session not yet holding one, alternating WB (Auto first), within quota
    for group, quota, need in (("F", F_QUOTA, F_MIN["bright-highlight"]), ("K", K_QUOTA, K_MIN["bright-highlight"])):
        bright = lambda: [s for s in picked[group] if s.get("highlightClass") == "bright-highlight"]
        for w in [AUTO, KELVIN] * need:
            if len(bright()) >= need:
                break
            have = {s["sessionIndex"] for s in bright()}
            room = lambda s: sum(p["sessionIndex"] == s["sessionIndex"] for p in picked[group]) < quota.get(s["sessionIndex"], 0)
            take(group, lambda s, w=w: s.get("highlightClass") == "bright-highlight" and wb(s) == w
                 and s["sessionIndex"] not in have and room(s))
    for group, quota in (("F", F_QUOTA), ("K", K_QUOTA)):
        slot = 0
        for sess, k in quota.items():
            while sum(s["sessionIndex"] == sess for s in picked[group]) < k:
                # WB that is behind in this group so far goes first (Auto on ties:
                # it is the recipe's WB); condition rotates bright/mixed/high-ISO/any.
                n_auto = sum(wb(s) == AUTO for s in picked[group])
                n_kel = sum(wb(s) == KELVIN for s in picked[group])
                order = [AUTO, KELVIN] if n_auto <= n_kel else [KELVIN, AUTO]
                conds = CONDS[slot % len(CONDS):] + CONDS[:slot % len(CONDS)]
                slot += 1
                got = next((s for w in order for c in conds
                            if (s := take(group, lambda s, w=w, c=c: s["sessionIndex"] == sess
                                          and wb(s) == w and cond(s, c)))), None)
                if not got and not take(group, lambda s: s["sessionIndex"] == sess):
                    break  # session exhausted; reported as unmet n

    F = picked["F"]
    first = lambda pred: next((s for s in F if pred(s) and s not in ctrl), None)
    ctrl = [s for s in F if s["name"] in PILOTS]
    for sess in (7, 8, 12, 13):
        ctrl.append(first(lambda s, sess=sess: s["sessionIndex"] == sess))
    ctrl.append(first(lambda s: wb(s) == FLUOR))
    ctrl.append(first(lambda s: s["iso"] >= 5000))
    ctrl = [s for s in ctrl if s]

    def cover(rows):
        return {"n": len(rows), "sessions": sorted({s["sessionIndex"] for s in rows}),
                "wb": {hex(c): sum(wb(s) == c for s in rows) for c in sorted({wb(s) for s in rows})},
                "bright-highlight": sum(s.get("highlightClass") == "bright-highlight" for s in rows),
                "mixed-lighting": sum(s.get("highlightClass") == "mixed-lighting" for s in rows),
                "iso>=1600": sum(s["iso"] >= 1600 for s in rows), "iso>=5000": sum(s["iso"] >= 5000 for s in rows)}

    cov = {"F": cover(F), "K": cover(picked["K"]), "C": cover(ctrl)}
    unmet = [f"{g} {k}: {cov[g][k]} < {v}" for g, mins in (("F", F_MIN), ("K", K_MIN))
             for k, v in mins.items() if cov[g][k] < v]
    unmet += [f"{g} n={cov[g]['n']} != {sum(q.values())}" for g, q in (("F", F_QUOTA), ("K", K_QUOTA))
              if cov[g]["n"] != sum(q.values())]
    row = lambda s, role: {"name": s["name"], "shoot": s["shoot"], "tier": "train", "role": role,
                           "session": s["sessionIndex"], "iso": s["iso"], "highlightClass": s.get("highlightClass"),
                           "asShotWhiteBalance": hex(wb(s)), "pilotExportsVerified": s["name"] in PILOTS,
                           "isolatedControls": s in ctrl}
    out = {"status": "PROPOSED - not approved; no exports requested; no fitting",
           "pool_size": len(pool), "coverage": cov, "unmet_constraints": unmet,
           "scenes": [row(s, "fit") for s in F] + [row(s, "train-check") for s in picked["K"]]}
    OUT.write_text(json.dumps(out, indent=1))
    for r in out["scenes"]:
        print(r["role"], r["name"], r["shoot"], "s", r["session"], "iso", r["iso"], r["asShotWhiteBalance"],
              r["highlightClass"], "CTRL" if r["isolatedControls"] else "")
    print(json.dumps(cov, indent=1))
    print("UNMET:", unmet or "none")


if __name__ == "__main__":
    main()
