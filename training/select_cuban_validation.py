"""Classic Cuban Neg validation - deterministic, metadata-only scene selection
and X RAW Studio export-folder preparation. No pixels are read.

Pool: monitor tier only (not seen by round 11 training), restricted to the
recipe's own capture conditions - captureDynamicRange 400, ISO 500-6400
(recipe: DR400, "Auto, ISO 500 up to ISO 6400"). Stratified by session,
highlight class, manual/auto WB and ISO; ties broken by filename. Plus
DSCF0752 (train tier) as a diagnostic link to the one existing full-recipe
X RAW Studio JPEG - never counted as validation. finalHeldOut is never read.
"""
import json
import os
from pathlib import Path

CAL = Path(__file__).resolve().parent.parent / "calibration-input"
OUT_MANIFEST = CAL / "cuban-validation-manifest.json"
BATCH = CAL / "cuban-validation-batch"
from cuban_exports import VARIANTS  # E1-E5, single source of truth
QUOTA = {4: 4, 5: 3, 9: 3, 10: 2}  # 12 scenes, roughly proportional to the 35/18/14/3 pool


def priority(s):
    # rarer conditions first so each is represented; filename breaks ties
    cls = {"bright-highlight": 0, "mixed-lighting": 1}.get(s.get("highlightClass"), 2)
    return (cls, 0 if s["iso"] > 500 else 1, s["name"])


def main():
    m = json.loads((CAL / "phase4-corpus-manifest.json").read_text())
    pool = [s for s in m["tiers"]["monitor"]["scenes"]
            if s.get("captureDynamicRange") == 400 and 500 <= s["iso"] <= 6400 and s.get("pairReady")]
    chosen = []
    for session, k in QUOTA.items():
        cands = sorted((s for s in pool if s["sessionIndex"] == session), key=priority)
        picked = []
        for want_manual in [1, 0] * k:  # alternate manual/auto WB where possible
            nxt = next((s for s in cands if s not in picked and s.get("wbManualFlag") == want_manual), None)
            nxt = nxt or next((s for s in cands if s not in picked), None)
            if nxt and len(picked) < k:
                picked.append(nxt)
        chosen += picked
    diag = next(s for s in m["tiers"]["train"]["scenes"] if s["name"] == "DSCF0752.RAF")

    rows = [{"name": s["name"], "shoot": s["shoot"], "tier": "monitor", "role": "validation",
             "session": s["sessionIndex"], "iso": s["iso"], "highlightClass": s.get("highlightClass"),
             "wbManualFlag": s.get("wbManualFlag"), "captureDynamicRange": 400} for s in chosen]
    rows.append({"name": diag["name"], "shoot": diag["shoot"], "tier": "train", "role": "diagnostic-only",
                 "session": diag["sessionIndex"], "iso": diag["iso"], "highlightClass": diag.get("highlightClass"),
                 "wbManualFlag": diag.get("wbManualFlag"), "captureDynamicRange": 400})
    OUT_MANIFEST.write_text(json.dumps(rows, indent=1))

    BATCH.mkdir(exist_ok=True)
    for v in VARIANTS:
        (BATCH / v).mkdir(exist_ok=True)
    for r in rows:
        src = CAL / r["shoot"] / r["name"]
        link = BATCH / r["name"]
        if not link.exists():
            os.symlink(os.path.relpath(src.resolve(), BATCH), link)
        assert link.resolve().exists(), link
    for r in rows:
        print(r)
    print(f"{len(rows)} RAFs linked in {BATCH.name}/, output folders: {list(VARIANTS)}")


if __name__ == "__main__":
    main()
