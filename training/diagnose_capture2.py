"""Capture batch 2 structural diagnostic - diagnostic-only, per instruction:
no training, no model selection, no finalHeldOut access. Characterizes the
38 new scenes (Shoot 706-743) on the same structural axes already used to
describe the 5 persistent-regression scenes in
docs/phase4-scene-adaptive-scope.md (highlight shape/compactness, p99
brightness, near-black/bright fraction), plus WB mode and capture timestamp
pulled from each RAF's embedded-preview EXIF (see
scripts/extract-capture2-exif.mjs, run first). Reuses
highlight_shape_stats/luma_distribution_stats from diagnose_structural.py
and compute_luma_guide from model.py - same feature definitions as the
existing 5-scene characterization, so results are directly comparable.

Sessions are derived with the corpus's own rule (phase4-corpus-manifest.json
sessionRule: >3h capture-timestamp gap = new session boundary), not assumed.

train + monitor + capture-batch-2 scenes only - finalHeldOut has no code
path here and is never read.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from diagnose_structural import highlight_shape_stats, luma_distribution_stats
from fjlrg import load_fjlrg_linear
from model import compute_luma_guide

REPO_ROOT = Path(__file__).resolve().parent.parent
CAL_DIR = REPO_ROOT / "calibration-input"
OUT_DIR = Path(__file__).resolve().parent / "diagnostics"

MANIFEST_PATH = CAL_DIR / "phase4-capture2-manifest.json"
EXIF_PATH = CAL_DIR / "phase4-capture2-exif.json"
SESSION_GAP_HOURS = 3.0


def parse_exif_dt(s):
    # "YYYY:MM:DD HH:MM:SS" (standard EXIF DateTimeOriginal format)
    return datetime.strptime(s, "%Y:%m:%d %H:%M:%S")


def cluster_sessions(rows):
    """rows sorted by dt ascending. Returns list of session ids (0-indexed),
    same order as rows, new session whenever the gap to the previous shot
    exceeds SESSION_GAP_HOURS."""
    session_ids = []
    current = 0
    prev_dt = None
    for r in rows:
        if prev_dt is not None:
            gap_hours = (r["dt"] - prev_dt).total_seconds() / 3600.0
            if gap_hours > SESSION_GAP_HOURS:
                current += 1
        session_ids.append(current)
        prev_dt = r["dt"]
    return session_ids


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    exif_rows = json.loads(EXIF_PATH.read_text())
    exif_by_name = {r["filename"]: r for r in exif_rows}

    rows = []
    for entry in manifest:
        name = entry["filename"]
        shoot = entry["shoot_folder"]
        exif = exif_by_name.get(name)
        if exif is None or exif.get("error"):
            raise ValueError(f"{name}: no usable EXIF entry ({exif})")

        fjlrg_path = CAL_DIR / shoot / "browser-phase4-capture2-linear.fjlrg"
        data, w, h = load_fjlrg_linear(fjlrg_path)
        full_res = torch.from_numpy(data.transpose(2, 0, 1)).unsqueeze(0)  # 1,3,H,W
        luma = compute_luma_guide(full_res, "rec709").squeeze(0).numpy()

        shape_stats = highlight_shape_stats(data, luma)
        dist_stats = luma_distribution_stats(luma)
        p99 = float(np.percentile(luma, 99))

        rows.append({
            "name": name,
            "shoot": shoot,
            "dt": parse_exif_dt(exif["dateTimeOriginal"]),
            "iso": exif["iso"],
            "wbMode": exif["wbMode"],
            "wbManualFlag": exif["wbManualFlag"],
            "p99": p99,
            **shape_stats,
            **dist_stats,
        })

    rows.sort(key=lambda r: r["dt"])
    session_ids = cluster_sessions(rows)
    for r, sid in zip(rows, session_ids):
        r["captureSessionId"] = sid

    n_sessions = session_ids[-1] + 1 if session_ids else 0
    session_sizes = {sid: session_ids.count(sid) for sid in range(n_sessions)}

    # Flag likely accidental duplicates: same WB mode, <20s apart, nearly
    # identical structure (NOT a deliberate WB-contrast pair, which differs
    # in wbMode by design and is expected/good, not a problem).
    duplicates = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        gap_s = (b["dt"] - a["dt"]).total_seconds()
        if gap_s > 20 or a["wbMode"] != b["wbMode"]:
            continue
        dp99 = abs(a["p99"] - b["p99"])
        dcompact = abs((a["largest_compactness"] or 0) - (b["largest_compactness"] or 0))
        darea = abs(a["largest_area_fraction"] - b["largest_area_fraction"])
        if dp99 < 0.02 and dcompact < 0.05 and darea < 0.001:
            duplicates.append({"a": a["name"], "b": b["name"], "gap_seconds": gap_s,
                                "d_p99": dp99, "d_compactness": dcompact, "d_area_fraction": darea})

    # Flag deliberate WB-contrast pairs: opposite WB mode, close in time,
    # near-identical structure (same scene, WB changed between shots - the
    # intended capture-plan behavior, not a defect).
    wb_pairs = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        gap_s = (b["dt"] - a["dt"]).total_seconds()
        if gap_s > 90 or a["wbMode"] == b["wbMode"]:
            continue
        dp99 = abs(a["p99"] - b["p99"])
        dcompact = abs((a["largest_compactness"] or 0) - (b["largest_compactness"] or 0))
        if dp99 < 0.05 and dcompact < 0.1:
            wb_pairs.append({"a": a["name"], "b": b["name"], "gap_seconds": gap_s,
                              "wb_a": a["wbMode"], "wb_b": b["wbMode"]})

    for r in rows:
        r["dt"] = r["dt"].isoformat()

    report = {
        "n_scenes": len(rows),
        "n_sessions": n_sessions,
        "session_sizes": session_sizes,
        "session_gap_rule_hours": SESSION_GAP_HOURS,
        "scenes": rows,
        "likely_accidental_duplicates": duplicates,
        "deliberate_wb_pairs_detected": wb_pairs,
    }
    (OUT_DIR / "capture2-structural-diagnosis.json").write_text(json.dumps(report, indent=2))

    print(f"n_scenes={len(rows)}  n_sessions={n_sessions}  session_sizes={session_sizes}")
    print(f"deliberate WB pairs detected: {len(wb_pairs)}")
    print(f"likely accidental duplicates: {len(duplicates)}")
    for d in duplicates:
        print(f"  {d['a']} / {d['b']}  gap={d['gap_seconds']:.0f}s  dp99={d['d_p99']:.4f}")
    print("\nfinalHeldOut was not accessed by this diagnostic pass.")


if __name__ == "__main__":
    main()
