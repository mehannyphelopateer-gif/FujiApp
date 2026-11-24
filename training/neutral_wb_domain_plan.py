"""Pure, source-only planning helpers for the isolated neutral-WB research domain. No file I/O.

Inputs: source metadata rows (WB keys, cam/auto levels and provenance only), the FROZEN fit-derived
profiles from wb-alignment-report.json, and optional sets of already-qualified FIT names for W and P reuse.
Policy is the existing wb_alignment_metrics.propose_policy, unchanged. Nothing here reads targets,
caches or pixels, fits anything, recalibrates a profile or opens a file. Any extra keys on a source row
(for example target hints) are dropped before planning and can never influence it.

basis                                -> kind, contract,                         declared coefficient source
source-auto-estimate                 -> W,    v2-linear-autowb,                 fuji_f00d  (actual source auto_rgbg)
calibrated-kelvin-profile-candidate  -> P,    research-linear-fitprofile-k6600, fit_profile (fixed 620,302,455,302)
zero-shift-as-shot                   -> G,    v2-linear-asshot,                 cam_mul    (actual source cam_mul, [3] kept)
unsupported                          -> excluded, no fallback

Development rows receive a metadata plan only: never export-eligible, never a reuse source, never a decode job.
"""
import re
from fractions import Fraction
from math import lcm

import fjnrgb_profile as P
import wb_alignment_metrics as W

PROVENANCE_FIELDS = ("name", "shoot", "role", "session", "captureDynamicRange", "raf_sha256")
SOURCE_FIELDS = PROVENANCE_FIELDS + tuple(sorted(W.SOURCE_KEYS))
ROLES = ("fit", "development")
CAPTURE_DRS = (100, 200, 400)
KINDS = {
    "source-auto-estimate": {"kind": "W", "contract": "v2-linear-autowb", "coeff_source": "fuji_f00d"},
    "calibrated-kelvin-profile-candidate": {"kind": "P", "contract": P.CONTRACT, "coeff_source": P.WB_SOURCE_KIND},
    "zero-shift-as-shot": {"kind": "G", "contract": "v2-linear-asshot", "coeff_source": "cam_mul"},
}
REUSE_TAGS = {"W": "qualified-fit-W", "P": "qualified-fit-P", "G": "existing-G-identity"}
_NAME = re.compile(r"DSCF\d{4}\.RAF")
_SHOOT = re.compile(r"Shoot \d+")
_HEX = re.compile(r"[0-9a-f]{64}")


class DomainPlanError(ValueError):
    pass


def _need(cond, msg):
    if not cond:
        raise DomainPlanError(msg)


def extract_source(row):
    """Strict copy of exactly the source/provenance fields; everything else is dropped."""
    _need(isinstance(row, dict), "source row must be a dict")
    missing = [k for k in SOURCE_FIELDS if k not in row]
    _need(not missing, f"source row missing {missing}")
    s = {k: row[k] for k in SOURCE_FIELDS}
    _need(isinstance(s["name"], str) and _NAME.fullmatch(s["name"]) is not None, f"unsafe source name {s['name']!r}")
    _need(isinstance(s["shoot"], str) and _SHOOT.fullmatch(s["shoot"]) is not None, f"unsafe shoot {s['shoot']!r}")
    _need(s["role"] in ROLES, f"role {s['role']!r} refused (fit/development only)")
    _need(type(s["session"]) is int and s["session"] >= 0, "session must be a non-negative int")
    _need(type(s["captureDynamicRange"]) is int and s["captureDynamicRange"] in CAPTURE_DRS, "captureDynamicRange must be 100/200/400")
    _need(isinstance(s["raf_sha256"], str) and _HEX.fullmatch(s["raf_sha256"]) is not None, "raf_sha256 must be 64 lowercase hex")
    for key in ("cam_mul", "auto_rgbg", "source_wb_fine_tune_raw"):
        _need(isinstance(s[key], (list, tuple)), f"{key} must be a list")
        s[key] = list(s[key])
    return s


def validate_frozen_profiles(report_profiles):
    """Accept only the frozen wb-alignment profile block whose single candidate equals the fixed codec profile."""
    _need(isinstance(report_profiles, dict) and isinstance(report_profiles.get("profiles"), dict), "profiles must be calibrated_profiles output")
    candidates = {k: v for k, v in report_profiles["profiles"].items() if isinstance(v, dict) and v.get("status") == "candidate"}
    _need(set(candidates) == {P.PROFILE_KEY}, f"exactly one frozen candidate profile {P.PROFILE_KEY} required, got {sorted(candidates)}")
    _need(report_profiles.get("allowlist") == [P.PROFILE_KEY], "profile allowlist differs from the frozen single family")
    prof = candidates[P.PROFILE_KEY]
    _need(prof.get("mode") == P.SOURCE_WB_MODE and prof.get("kelvin") == P.SOURCE_KELVIN, "profile family differs")
    _need(prof.get("source_shifts_observed") == [list(P.SOURCE_FINE_TUNE_RAW)], "profile calibration shift differs")
    t = prof.get("target_ratios")
    _need(isinstance(t, dict) and set(t) >= {"R/G", "B/G"}, "profile ratios missing")
    rg, bg = (Fraction(t[k]["num"], t[k]["den"]) for k in ("R/G", "B/G"))
    g = lcm(rg.denominator, bg.denominator)
    _need([int(rg * g), g, int(bg * g), g] == list(P.PROFILE_LEVELS_RGBG), "frozen profile differs from the fixed codec levels")
    return report_profiles


def _ratio(levels):
    return W.positive_rgb_ratios(levels)


def plan_row(row, profiles, *, qualified_w=frozenset(), qualified_p=frozenset()):
    """One source-only plan record. No file paths; reuse is a tag only."""
    s = extract_source(row)
    validate_frozen_profiles(profiles)
    policy = W.propose_policy({k: s[k] for k in W.SOURCE_KEYS}, profiles)
    basis = policy["basis"]
    record = {**{k: s[k] for k in PROVENANCE_FIELDS}, "policy_basis": basis, "policy_reason": policy["reason"],
              "source": s, "kind": None, "contract": None, "declared_coeff_source": None, "declared_levels_rgbg": None,
              "excluded": basis == "unsupported", "export_eligible": False, "requires_new_decode": None, "reuse_source": None}
    if basis == "unsupported":
        record["note"] = "unsupported family: excluded, no fallback"
        return record
    spec = KINDS[basis]
    if spec["kind"] == "W":
        _need(s["source_wb_mode"] == W.AUTO_MODE and s["source_kelvin"] is None, "W requires source Auto (mode 0, no Kelvin)")
        levels = list(s["auto_rgbg"])
    elif spec["kind"] == "P":
        _need(s["source_wb_mode"] == P.SOURCE_WB_MODE and s["source_kelvin"] == P.SOURCE_KELVIN
              and s["source_wb_fine_tune_raw"] == list(P.SOURCE_FINE_TUNE_RAW),
              "profile proposed outside the qualified 4080/6600/[20,-60] pattern: refused, no fallback")
        levels = list(P.PROFILE_LEVELS_RGBG)
    else:
        _need(s["source_wb_fine_tune_raw"] == [0, 0] and s["source_wb_mode"] != W.AUTO_MODE, "G identity requires a zero-shift non-Auto source")
        levels = list(s["cam_mul"])  # actual as-shot record, fourth slot kept (0); green not assumed minimum
    chosen = {k: Fraction(v["num"], v["den"]) for k, v in policy["ratios"].items()}
    _need(chosen == _ratio(levels), "declared levels disagree with the frozen policy ratios")
    record.update(kind=spec["kind"], contract=spec["contract"], declared_coeff_source=spec["coeff_source"], declared_levels_rgbg=levels)
    if s["role"] != "fit":
        record["note"] = "development: metadata plan only; no export job or reusable artifact"
        return record
    record["export_eligible"] = True
    if spec["kind"] == "G":
        record.update(requires_new_decode=False, reuse_source=REUSE_TAGS["G"])
    else:
        reusable = s["name"] in (qualified_w if spec["kind"] == "W" else qualified_p)
        record.update(requires_new_decode=not reusable, reuse_source=REUSE_TAGS[spec["kind"]] if reusable else None)
    return record


def _names(values, label):
    _need(isinstance(values, (set, frozenset)) and all(isinstance(v, str) and _NAME.fullmatch(v) for v in values),
          f"{label} must be a set of DSCF####.RAF names")
    return frozenset(values)


def plan_rows(rows, profiles, *, qualified_w=frozenset(), qualified_p=frozenset()):
    qw, qp = _names(qualified_w, "qualified_w"), _names(qualified_p, "qualified_p")
    validate_frozen_profiles(profiles)
    plan = [plan_row(r, profiles, qualified_w=qw, qualified_p=qp) for r in rows]
    names = [r["name"] for r in plan]
    _need(len(set(names)) == len(names), "duplicate source names")
    return plan


def fit_rows(plan):
    """Explicit filter: only current FIT, non-excluded rows may enter a fit export job."""
    return [r for r in plan if r["role"] == "fit" and r["export_eligible"] and not r["excluded"]]


def summarize(plan):
    counts = {}
    for r in plan:
        key = f"{r['role']}/{r['kind'] or 'excluded'}"
        counts[key] = counts.get(key, 0) + 1
    fit = fit_rows(plan)
    return {"by_role_kind": dict(sorted(counts.items())), "fit_export_rows": len(fit),
            "fit_new_decodes": sum(r["requires_new_decode"] is True for r in fit),
            "fit_reused": sum(r["requires_new_decode"] is False for r in fit)}
