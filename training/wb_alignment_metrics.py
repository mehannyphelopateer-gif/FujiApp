"""Pure exact-arithmetic helpers for a metadata-only WB alignment / profile-identifiability assay.

Standard library only (fractions). No I/O, no loaders, no images, no TIFF/RAF, no factors,
no inverses, no interpolation, no averaging. Nothing here qualifies a runtime WB policy:
outputs are CANDIDATE metadata policies for a separately frozen primitive test.

Conventions
  * Every RGB record is normalised by ITS OWN green: ratios are exact Fractions R/G and B/G.
    Green is never assumed to be 302 (e.g. a zero-shift Kelvin 2500 cam_mul [302,325,1137,0]
    is valid with G = 325). Minimum-channel observations are reported, never used as a rule.
  * cam_mul (as-shot) layout [R,G,B,0]; Auto estimate (auto_rgbg) layout [R,G,B,G2] with the
    two green slots positive and equal. Coefficients must be positive ints (bool refused).
  * Policies are distinct and none uses a learned factor:
      source-auto-estimate     source mode 0 (Auto), any fine-tune shift: the source's own
                               Auto estimate ratios.
      zero-shift-as-shot       any known non-Auto mode with zero shift: the source's own
                               as-shot cam_mul ratios (identity; includes G != 302).
      calibrated-kelvin-profile-candidate
                               nonzero-shift Kelvin source whose (mode, kelvin) family is in
                               NONZERO_SHIFT_PROFILE_ALLOWLIST and has a fit-only exact
                               constant neutral-target profile with coverage.
      unsupported              everything else (no fallback, no nearest family).
    The Auto estimate is never used as a manual/preset neutral reference.
  * Profiles: derived from fit rows only, grouped by (mode, kelvin), target shift must be
    [0,0], target ratios must be exactly constant, >= min_scenes distinct fit names in
    >= min_sessions distinct sessions. Development rows are counted and ignored.

Training-target metadata limits: neutral-target ratios come from archived export metadata of
ONE X100VI body in this corpus. They are not decoded applied gains, not pixel evidence, not
other cameras, and a candidate profile is not runtime qualification.
"""
from fractions import Fraction

FINE_TUNE_STEP = 20
FINE_TUNE_LIMIT = 180
KNOWN_MODES = (0, 1, 256, 512, 770, 4080)   # codes present in the frozen source inventory only
KELVIN_MODE = 4080
AUTO_MODE = 0
NONZERO_SHIFT_PROFILE_ALLOWLIST = frozenset({(4080, 6600)})
SOURCE_KEYS = frozenset({"source_wb_mode", "source_kelvin", "source_wb_fine_tune_raw", "cam_mul", "auto_rgbg"})
CHANNELS = ("R", "G", "B")
LIMITS = ("Archived metadata of one X100VI body only; not decoded applied gains, pixel evidence or other "
          "cameras. Candidates are not qualification; no factors, inverses, interpolation or averaging. "
          "Development rows never derive profiles.")


def _need(cond, msg):
    if not cond:
        raise ValueError(msg)


def _is_int(v):
    return type(v) is int


def _pos_int(v, name):
    _need(_is_int(v) and v > 0, f"{name} must be a positive int (bool refused), got {v!r}")
    return v


def _frac_json(f):
    return {"num": f.numerator, "den": f.denominator, "float": float(f)}


def _ratios_json(ratios):
    return {k: _frac_json(v) for k, v in ratios.items()}


# ------------------------------------------------------------------ exact ratios
def positive_rgb_ratios(coeffs):
    """[R,G,B] or [R,G,B,G2|0] positive ints -> {"R/G": Fraction, "B/G": Fraction}.

    A fourth value must be 0 (as-shot layout) or a positive int equal to G (Auto layout).
    """
    _need(isinstance(coeffs, (list, tuple)) and len(coeffs) in (3, 4), "coefficients must be a 3- or 4-item list")
    r, g, b = (_pos_int(v, f"coefficient {c}") for v, c in zip(coeffs[:3], CHANNELS))
    if len(coeffs) == 4:
        g2 = coeffs[3]
        _need(_is_int(g2) and (g2 == 0 or g2 == g), f"fourth coefficient must be 0 or equal green {g}, got {g2!r}")
    return {"R/G": Fraction(r, g), "B/G": Fraction(b, g)}


def _record(coeffs, kind):
    rgb = list(coeffs[:3])
    lo = min(rgb)
    return {"kind": kind, "rgb": rgb, "green": rgb[1], "min_value": lo,
            "min_channels": [c for c, v in zip(CHANNELS, rgb) if v == lo],
            "ratios": _ratios_json(positive_rgb_ratios(coeffs))}


def _check_cam(cam_mul):
    _need(isinstance(cam_mul, (list, tuple)) and len(cam_mul) == 4 and _is_int(cam_mul[3]) and cam_mul[3] == 0,
          "cam_mul must be [R,G,B,0]")
    positive_rgb_ratios(cam_mul)


def _check_auto(auto_rgbg):
    _need(isinstance(auto_rgbg, (list, tuple)) and len(auto_rgbg) == 4, "auto_rgbg must be [R,G,B,G2]")
    _pos_int(auto_rgbg[3], "Auto estimate second green")
    _need(auto_rgbg[1] == auto_rgbg[3], "Auto estimate green slots must be equal")
    positive_rgb_ratios(auto_rgbg)


def anchor_record(cam_mul, auto_rgbg):
    """Observed anchors of both source records. Reports minima/anchor channels; asserts no rule."""
    _check_cam(cam_mul)
    _check_auto(auto_rgbg)
    cam, auto = _record(cam_mul, "as-shot cam_mul"), _record(auto_rgbg, "auto estimate")
    auto["green_slots"] = [auto_rgbg[1], auto_rgbg[3]]
    return {"cam_mul": cam, "auto_estimate": auto,
            "green_equal_across_records": cam_mul[1] == auto_rgbg[1],
            "note": "Observations only; minimum or green anchoring is not assumed as a camera rule."}


# ------------------------------------------------------------------ fit-only profiles
def _shift(v, name):
    _need(isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_int(x) for x in v), f"{name} must be two ints")
    return tuple(v)


def _source_shift(v):
    s = _shift(v, "source fine-tune")
    _need(all(x % FINE_TUNE_STEP == 0 and -FINE_TUNE_LIMIT <= x <= FINE_TUNE_LIMIT for x in s),
          f"source fine-tune must be multiples of {FINE_TUNE_STEP} within +/-{FINE_TUNE_LIMIT}")
    return s


def _mode_kelvin(mode, kelvin):
    _need(_is_int(mode) and mode in KNOWN_MODES, f"unknown WB mode {mode!r} refused")
    if mode == KELVIN_MODE:
        _pos_int(kelvin, "kelvin")
    else:
        _need(kelvin is None, "kelvin must be None outside Kelvin mode")
    return mode, kelvin


def _key(mode, kelvin):
    return f"{mode}/{kelvin}"


def calibrated_profiles(rows, min_scenes=3, min_sessions=2):
    """Group fit rows by (mode, kelvin); emit an exact candidate only under full coverage.

    rows: dicts with name, role ('fit'|'development'), session, mode, kelvin, target_shift,
    source_shift, target_rgb, source_rgb. Source/target mode-Kelvin agreement is the caller's
    loader contract. Development rows are validated and counted but never used.
    """
    _need(_is_int(min_scenes) and min_scenes >= 1 and _is_int(min_sessions) and min_sessions >= 1,
          "min_scenes/min_sessions must be positive ints")
    groups, dev_counts, seen = {}, {}, set()
    for row in rows:
        _need(isinstance(row, dict), "row must be a dict")
        name, role = row.get("name"), row.get("role")
        _need(isinstance(name, str) and name, "row name required")
        _need(role in ("fit", "development"), f"role must be fit/development, got {role!r}")
        _need(name not in seen, f"duplicate row {name}")
        seen.add(name)
        _need(_is_int(row.get("session")), "session must be an int")
        mode, kelvin = _mode_kelvin(row.get("mode"), row.get("kelvin"))
        tshift = _shift(row.get("target_shift"), "target_shift")
        _source_shift(row.get("source_shift"))
        target = positive_rgb_ratios(row.get("target_rgb"))
        positive_rgb_ratios(row.get("source_rgb"))
        key = (mode, kelvin)
        if role == "development":
            dev_counts[key] = dev_counts.get(key, 0) + 1
            continue
        groups.setdefault(key, []).append({"name": name, "session": row["session"], "target_shift": tshift,
                                            "source_shift": tuple(row["source_shift"]),
                                            "target": (target["R/G"], target["B/G"])})
    out = {}
    for key in sorted(set(groups) | set(dev_counts), key=lambda k: (k[0], -1 if k[1] is None else k[1])):
        fit = groups.get(key, [])
        reasons = []
        sessions = sorted({r["session"] for r in fit})
        targets = sorted({r["target"] for r in fit})
        if not fit:
            reasons.append("no fit rows")
        if any(r["target_shift"] != (0, 0) for r in fit):
            reasons.append("target shift is not [0,0]")
        if len(targets) > 1:
            reasons.append("fit neutral-target ratios are not exactly constant")
        if len(fit) < min_scenes:
            reasons.append(f"fewer than {min_scenes} distinct fit scenes")
        if len(sessions) < min_sessions:
            reasons.append(f"fewer than {min_sessions} fit sessions")
        if key not in NONZERO_SHIFT_PROFILE_ALLOWLIST:
            reasons.append("family not in nonzero-shift profile allowlist")
        item = {"mode": key[0], "kelvin": key[1], "fit_scenes": len(fit), "fit_sessions": sessions,
                "development_rows_ignored": dev_counts.get(key, 0),
                "source_shifts_observed": [list(s) for s in sorted({r["source_shift"] for r in fit})],
                "distinct_target_ratio_count": len(targets),
                "status": "unvalidated" if reasons else "candidate", "reasons": reasons,
                "qualification": "NONE"}
        if not reasons:
            rg, bg = targets[0]
            item["target_ratios"] = _ratios_json({"R/G": rg, "B/G": bg})
            item["basis"] = "exact constant fit-only neutral-target metadata, one X100VI body"
        out[_key(*key)] = item
    return {"profiles": out, "min_scenes": min_scenes, "min_sessions": min_sessions,
            "allowlist": sorted(_key(*k) for k in NONZERO_SHIFT_PROFILE_ALLOWLIST), "limits": LIMITS}


# ------------------------------------------------------------------ policy proposal
def propose_policy(source_row, profiles):
    """One metadata-only candidate policy for a source row. Never reads targets.

    source_row keys must be exactly SOURCE_KEYS. profiles: calibrated_profiles(...) output.
    Returns basis label, chosen ratio record (or None) and reason. Never a qualification.
    """
    _need(isinstance(source_row, dict) and set(source_row) == SOURCE_KEYS,
          f"source_row keys must be exactly {sorted(SOURCE_KEYS)}")
    mode, kelvin = _mode_kelvin(source_row["source_wb_mode"], source_row["source_kelvin"])
    shift = _source_shift(source_row["source_wb_fine_tune_raw"])
    cam, auto = source_row["cam_mul"], source_row["auto_rgbg"]
    _check_cam(cam)
    _check_auto(auto)
    _need(isinstance(profiles, dict) and isinstance(profiles.get("profiles"), dict), "profiles must be calibrated_profiles output")

    def result(basis, ratios, reason):
        return {"basis": basis, "ratios": _ratios_json(ratios) if ratios is not None else None,
                "reason": reason, "source_mode": mode, "source_kelvin": kelvin, "source_shift": list(shift),
                "qualification": "NONE"}

    if mode == AUTO_MODE:
        return result("source-auto-estimate", positive_rgb_ratios(auto), "Auto source: own Auto estimate, any shift")
    if shift == (0, 0):
        return result("zero-shift-as-shot", positive_rgb_ratios(cam), "zero-shift known mode: as-shot identity")
    if mode == KELVIN_MODE and (mode, kelvin) in NONZERO_SHIFT_PROFILE_ALLOWLIST:
        prof = profiles["profiles"].get(_key(mode, kelvin))
        if prof is not None and prof.get("status") == "candidate":
            t = prof["target_ratios"]
            ratios = {k: Fraction(t[k]["num"], t[k]["den"]) for k in ("R/G", "B/G")}
            return result("calibrated-kelvin-profile-candidate", ratios,
                          "nonzero-shift Kelvin family with exact fit-only profile")
        return result("unsupported", None, "allowlisted Kelvin family lacks a candidate profile")
    return result("unsupported", None, "nonzero shift outside allowlisted profile families; no fallback")
