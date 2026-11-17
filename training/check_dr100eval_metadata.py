"""DR100 evaluation batch - metadata gate (tag-only, no pixel decode).

Applies the exclusion rules fixed in docs/phase4-scene-adaptive-scope.md
("DR100 evaluation batch", 2026-11-14, Codex-reviewed) to every RAF in
calibration-input/phase4-dr100eval-raw/ and reports, per frame, whether it
is accepted or excluded and why. Also summarises the capture-spec coverage
(sessions under the >3h rule, ISO groups, WB modes). Never touches
finalHeldOut, never decodes pixels, never exports.

Usage:  python3 training/check_dr100eval_metadata.py            # whole folder
        python3 training/check_dr100eval_metadata.py --pair     # same, phrased for a 2-frame test
Runs scripts/dump-raf-exif.mjs itself.
"""
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "calibration-input" / "phase4-dr100eval-raw"
EXIF_OUT = REPO / "calibration-input" / "phase4-dr100eval-exif.json"
PATHS_OUT = REPO / "calibration-input" / "phase4-dr100eval-paths.json"
SESSION_GAP_H = 3.0
LOW_ISO_MAX, HIGH_ISO_MIN = 200, 1600


def tag_int(fuji, key):
    v = fuji.get(key)
    if v is None:
        return None
    try:
        return int(str(v).split()[0])
    except ValueError:
        return None


def wb_label(fuji):
    """Fuji MakerNote 0x1002: 0x0 Auto, 0x1 Auto (white priority), 0x2 Auto
    (ambience priority); 0x100+ are presets/custom, 0xff0 is Kelvin."""
    v = tag_int(fuji, "WhiteBalance")
    if v is None:
        return "unknown"
    if v < 0x100:
        return "Auto"
    return "Kelvin" if v == 0xff0 else f"manual(0x{v:x})"


def exclusion_reasons(e):
    """Rules exactly as written in the plan; returns [] if accepted."""
    r = []
    if e.get("error"):
        return [f"unreadable: {e['error']}"]
    f = e.get("fuji", {})
    if e.get("model") != "X100VI":
        r.append(f"model {e.get('model')!r} != X100VI")
    dev, auto = tag_int(f, "DevelopmentDynamicRange"), tag_int(f, "AutoDynamicRange")
    if auto is not None:
        r.append(f"AutoDynamicRange present ({auto}) - camera on DR AUTO")
    if dev is None:
        r.append("DevelopmentDynamicRange absent - DR not set to a fixed value")
    elif dev != 100:
        r.append(f"DevelopmentDynamicRange={dev}, not 100")
    for k in ("DRangePriority", "DRangePriorityAuto", "DRangePriorityFixed"):
        v = tag_int(f, k)
        if v not in (None, 0):
            r.append(f"{k}={v} (DRP must be off: absent or 0)")
    ec = tag_int(f, "ExposureCount")
    if ec not in (None, 1):
        r.append(f"ExposureCount={ec} (multiple exposure)")
    return r


def main():
    raw_dir = RAW_DIR
    if "--dir" in sys.argv:  # e.g. re-checking the excluded attempt-1 folder
        raw_dir = Path(sys.argv[sys.argv.index("--dir") + 1]).resolve()
        if "finalHeldOut" in str(raw_dir):
            sys.exit("refused: finalHeldOut is locked")
    rafs = sorted(raw_dir.glob("*.RAF"))
    if not rafs:
        sys.exit(f"no RAFs in {raw_dir}")
    paths_out = PATHS_OUT if raw_dir == RAW_DIR else raw_dir / "paths.json"
    exif_out = EXIF_OUT if raw_dir == RAW_DIR else raw_dir / "exif.json"
    paths_out.write_text(json.dumps([str(p.relative_to(REPO)) for p in rafs], indent=1))
    subprocess.run(["node", str(REPO / "scripts" / "dump-raf-exif.mjs"), "--list", str(paths_out),
                    "--out", str(exif_out)], check=True, cwd=REPO, capture_output=True)
    entries = json.loads(exif_out.read_text())

    accepted, excluded = [], []
    print(f"{len(entries)} frames in {raw_dir.name}\n")
    for e in entries:
        f = e.get("fuji", {})
        reasons = exclusion_reasons(e)
        line = (f"{e['name']}  {e.get('dateTimeOriginal')}  ISO {e.get('iso')}  {e.get('exposureTime')} {e.get('fNumber')}  "
                f"bias {e.get('exposureBias')}  WB {wb_label(f)}  DevDR {tag_int(f, 'DevelopmentDynamicRange')}  "
                f"AutoDR {tag_int(f, 'AutoDynamicRange')}")
        if reasons:
            excluded.append((e, reasons))
            print("EXCLUDED ", line)
            for r in reasons:
                print("           -", r)
        else:
            accepted.append(e)
            print("accepted ", line)

    print(f"\naccepted {len(accepted)} / excluded {len(excluded)}")
    if not accepted:
        return 1

    # capture-spec coverage over accepted frames only
    times = sorted((datetime.strptime(e["dateTimeOriginal"], "%Y:%m:%d %H:%M:%S"), e["name"]) for e in accepted)
    sessions, cur = [], [times[0]]
    for t in times[1:]:
        if (t[0] - cur[-1][0]).total_seconds() > SESSION_GAP_H * 3600:
            sessions.append(cur)
            cur = []
        cur.append(t)
    sessions.append(cur)
    by_session = {n: i for i, s in enumerate(sessions) for _, n in s}
    iso = {e["name"]: e.get("iso") or 0 for e in accepted}
    low = [n for n in iso if iso[n] <= LOW_ISO_MAX]
    high = [n for n in iso if iso[n] >= HIGH_ISO_MIN]
    mid = [n for n in iso if LOW_ISO_MAX < iso[n] < HIGH_ISO_MIN]
    wbs = Counter(wb_label(e.get("fuji", {})) for e in accepted)

    print(f"\nsessions (>{SESSION_GAP_H:g}h gap rule): {len(sessions)}  (need >= 3)")
    for i, s in enumerate(sessions):
        print(f"  session {i}: {s[0][0]:%Y-%m-%d %H:%M} -> {s[-1][0]:%H:%M}, {len(s)} frames")
    print(f"ISO <= {LOW_ISO_MAX}: {len(low)} (need >= 4), sessions {sorted({by_session[n] for n in low})} (need >= 2)")
    print(f"ISO >= {HIGH_ISO_MIN}: {len(high)} (need >= 4), sessions {sorted({by_session[n] for n in high})} (need >= 2)")
    print(f"ISO in between: {len(mid)}")
    print(f"WB modes: {dict(wbs)}")
    print("lighting-class spread (daylight / low light / bright window) is NOT checked here - "
          "assigned from the user's shot notes, each class in >= 2 sessions, each ISO group in >= 2 classes.")
    short = len(accepted) < 12 or len(low) < 4 or len(high) < 4 or len(sessions) < 3
    print("\nBATCH STATUS:", "SHORT - keep shooting" if short else "meets the metadata/ISO/session spec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
