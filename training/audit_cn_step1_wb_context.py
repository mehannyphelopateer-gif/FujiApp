"""F-only reference/WB metadata audit after CN Step 1 failed.

No model inference, fitting, target pixel scoring or K/monitor/finalHeldOut
access. Source WB shifts are captured as metadata, not treated as a proven
explanation of the failed model. Run scripts/inspect-cn-wb-context.mjs first
with --all-fit to add actual LibRaw camera multipliers for the fixed F24.
"""
import hashlib
import io
import json
import struct
from pathlib import Path

from cuban_exports import CAL, VARIANTS, read_settings

DIAG = Path(__file__).resolve().parent / "diagnostics"
RUN = DIAG / "cuban-step1-run-manifest.json"
PROBE = DIAG / "cn-step1-wb-input-probe-all-F.json"
OUT = DIAG / "cn-step1-wb-reference-audit.json"
KEYS = ["WhiteBalance", "ColorTemperature", "WBFineTune", "WBLevelR", "WBLevelG", "WBLevelB"]


def main():
    rows = json.loads(RUN.read_text())["F_allowlist"]
    assert len(rows) == 24 and len({r["name"] for r in rows}) == 24
    cohort_bytes = (DIAG / "cuban-cohort-proposal.json").read_bytes()
    cohort = json.loads(cohort_bytes)["scenes"]
    allowed = {r["name"] for r in cohort if r["role"] == "fit" and r["tier"] == "train"}
    assert {r["name"] for r in rows} == allowed
    probe = json.loads(PROBE.read_text())
    probes = {r["name"]: r for r in probe["scenes"]}
    assert len(probe["scenes"]) == len(probes) == 24 and set(probes) == allowed
    assert probe["cohort_sha256"] == hashlib.sha256(cohort_bytes).hexdigest()
    records = []
    for row in rows:
        raf = CAL / row["shoot"] / row["name"]
        with raf.open("rb") as f:
            f.seek(84)
            offset, length = struct.unpack(">II", f.read(8))
            if not offset or not length:
                raise ValueError("Missing RAF preview location")
            f.seek(offset)
            preview = f.read(length)
        source = read_settings(io.BytesIO(preview))
        provia = read_settings(CAL / row["shoot"] / "xraw-phase3.tiff")
        negative = read_settings(CAL / row["e4"])
        expected = {**VARIANTS["E4-classic-neg-only"], "FilmMode": 0,
                    "WhiteBalance": source["WhiteBalance"]}
        mismatches = {k: {"actual": provia.get(k), "expected": v}
                      for k, v in expected.items() if provia.get(k) != v}
        if source["WhiteBalance"] == 0xFF0 and provia.get("ColorTemperature") != source.get("ColorTemperature"):
            mismatches["ColorTemperature"] = {"actual": provia.get("ColorTemperature"), "expected": source.get("ColorTemperature")}
        wb_matches = all(provia.get(k) == negative.get(k)
                         for k in ("WhiteBalance", "ColorTemperature", "WBFineTune"))
        r = {"name": row["name"], "shoot": row["shoot"],
             "source_raf_sha256_at_probe": probes[row["name"]]["source_raf_sha256"],
             "frozen_input_sha256": row["fjlrg_sha256"],
             "source_preview_sha256": hashlib.sha256(preview).hexdigest(),
             "source": {k: source.get(k) for k in KEYS},
             "provia": {k: provia.get(k) for k in KEYS},
             "classic_negative": {k: negative.get(k) for k in KEYS},
             "target_wb_matches": wb_matches, "provia_neutral_settings_mismatches": mismatches}
        level_keys = ("WBLevelR", "WBLevelG", "WBLevelB")
        r["target_normalized_wb_levels_match"] = (
            all(isinstance(v.get(k), int) for v in (provia, negative) for k in level_keys)
            and provia["WBLevelG"] > 0 and negative["WBLevelG"] > 0
            and provia["WBLevelR"] * negative["WBLevelG"] == negative["WBLevelR"] * provia["WBLevelG"]
            and provia["WBLevelB"] * negative["WBLevelG"] == negative["WBLevelB"] * provia["WBLevelG"])
        shift = source.get("WBFineTune")
        r["source_wb_shift_steps"] = [v / 20 for v in shift] if isinstance(shift, list) and len(shift) == 2 else None
        if row["name"] in probes:
            cm = probes[row["name"]]["cam_mul"]
            assert cm[:3] == [source[k] for k in ("WBLevelR", "WBLevelG", "WBLevelB")]
            r["libraw_cam_mul"] = cm
            r["libraw_gain_RB_over_G"] = [cm[0] / cm[1], cm[2] / cm[1]]
            for kind in ("provia", "classic_negative"):
                levels = r[kind]
                r[kind + "_recorded_level_RB_over_G"] = [levels["WBLevelR"] / levels["WBLevelG"],
                                                         levels["WBLevelB"] / levels["WBLevelG"]]
        records.append(r)
    result = {
        "scope": "F-only metadata; no model/target pixel scoring, fitting or K/monitor/finalHeldOut reads",
        "run_manifest_sha256": hashlib.sha256(RUN.read_bytes()).hexdigest(),
        "libraw_probe_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
        "target_WB_mode_temperature_and_shift_match": sum(r["target_wb_matches"] for r in records),
        "target_zero_shift_count": sum(r["provia"]["WBFineTune"] == r["classic_negative"]["WBFineTune"] == [0, 0] for r in records),
        "target_normalized_wb_levels_match_count": sum(r["target_normalized_wb_levels_match"] for r in records),
        "source_nonzero_shift_count": sum(r["source"]["WBFineTune"] not in (None, [0, 0]) for r in records),
        "provia_neutral_metadata_pass_count": sum(not r["provia_neutral_settings_mismatches"] for r in records),
        "libraw_probes": len(probes),
        "source_vs_target_normalized_level_difference_count": sum(
            r["libraw_gain_RB_over_G"] != r["provia_recorded_level_RB_over_G"] for r in records),
        "interpretation": "Canonical useCameraWb=true uses source cam_mul. All F24 probes exactly match source MakerNote WB levels; zero-shift target ratios differ. Level interpretation is inferred, and this does not quantify the contribution to the failed K error. Target ratios are diagnostic only, never proposed runtime inputs. No normalization or model change applied.",
        "scenes": records,
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "scenes"}, indent=2))
    for row in records:
        if row["provia_neutral_settings_mismatches"]:
            print(row["name"], row["provia_neutral_settings_mismatches"])


if __name__ == "__main__":
    main()
