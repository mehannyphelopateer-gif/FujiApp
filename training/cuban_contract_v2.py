"""Classic Cuban Neg evaluation contract v2 - FROZEN.

v2 = v1 (training/cuban_contract.py, sha c3f840ef..., left byte-identical)
plus ONLY the Codex-approved coverage clarification (docs/phase4-scene-
adaptive-scope.md, "Codex review: targeted support and E4-first fitting
sequence (2026-11-18)"):

  * C1/C2/C6/C7/C8 keep the original >= 6 of 8 measurable C-scene rule.
  * C3/C4/C5: the preselected T8 support set
    (training/diagnostics/cuban-control-support-selection.json, pinned by
    hash below) must INDEPENDENTLY provide >= 6 of its 8 scenes measurable.
  * Every measurable original-C-scene C3-C5 case stays scored and reported;
    an original failure is never discarded because support scenes pass.
  * No pooling as "6 of 16".
  * Effect/region/detail thresholds, accuracy limits, rho, grain and the
    whole-recipe conjunction are unchanged from v1.

Scoring code from 2026-11-19 on imports CONTRACT_V2 and refuses to run
unless contract_v2_sha256() equals the value pinned in the doc. v1 evidence
(reference audit, repeat floor) stays valid under v1's hash.

`python3 training/cuban_contract_v2.py` prints both hashes and runs checks.
"""
import copy
import hashlib
import json
from pathlib import Path

from cuban_contract import CONTRACT as CONTRACT_V1, contract_sha256 as contract_v1_sha256

V1_SHA256 = "c3f840ef9fad9d48de83b4edf183f2434523f352f358305b91eb7764d22d0067"
SUPPORT_SELECTION = Path(__file__).resolve().parent / "diagnostics" / "cuban-control-support-selection.json"
SUPPORT_SELECTION_SHA256 = "fdd3e3e890e2ed032ebb7b3baf240b18213434bcbd8810ef54266cb02b559fd3"

CONTRACT_V2 = copy.deepcopy(CONTRACT_V1)
CONTRACT_V2["version"] = 2
CONTRACT_V2["derived_from"] = {"version": 1, "sha256": V1_SHA256,
                               "change": "control coverage clarification only (Codex-approved 2026-11-18)"}
CONTRACT_V2["controls"]["min_measurable_scenes"] = 6  # unchanged value, now with explicit per-control sets
CONTRACT_V2["controls"]["coverage"] = {
    "original_C8_rule": {"controls": ["C1-highlight-minus2", "C2-shadow-plus1", "C6-nr-minus4",
                                      "C7-clarity-minus4", "C8-wbshift-r4-b5"],
                         "set": "original 8 C scenes", "min_measurable": 6, "of": 8},
    "support_T8_rule": {"controls": ["C3-color-plus4", "C4-cce-strong", "C5-fxblue-strong"],
                        "set": "T8 support set (cuban-control-support-selection.json)",
                        "support_scenes": ["DSCF0392.RAF", "DSCF0479.RAF", "DSCF0470.RAF", "DSCF0478.RAF",
                                           "DSCF0473.RAF", "DSCF0914.RAF", "DSCF0929.RAF", "DSCF1007.RAF"],
                        "support_selection_sha256": SUPPORT_SELECTION_SHA256,
                        "min_measurable": 6, "of": 8, "evaluated_independently": True},
    "original_C3_C5_cases": "all measurable original-C-scene C3-C5 cases remain scored and reported; "
                            "an original failure cannot be discarded by adding passing support scenes",
    "pooling": "forbidden - never 6 of 16",
    "support_set_role": "train-tier control support only; not F, not K, not independent test data",
}
CONTRACT_V2["sets"]["T8_support"] = "train-tier C3-C5 control support (preselected by Provia occupancy); not F or K"


def contract_v2_sha256():
    return hashlib.sha256(json.dumps(CONTRACT_V2, sort_keys=True).encode()).hexdigest()


def _check():
    assert contract_v1_sha256() == V1_SHA256, "contract v1 changed - refuse"
    sel = json.loads(SUPPORT_SELECTION.read_text())
    assert hashlib.sha256(SUPPORT_SELECTION.read_bytes()).hexdigest() == SUPPORT_SELECTION_SHA256
    names = sorted(s["name"] for s in sel["scenes"])
    assert names == sorted(CONTRACT_V2["controls"]["coverage"]["support_T8_rule"]["support_scenes"]), names
    cov = CONTRACT_V2["controls"]["coverage"]
    ctrls = {k for k, v in CONTRACT_V2["controls"].items() if isinstance(v, dict) and "region" in v}
    assert set(cov["original_C8_rule"]["controls"]) | set(cov["support_T8_rule"]["controls"]) == ctrls
    assert not set(cov["original_C8_rule"]["controls"]) & set(cov["support_T8_rule"]["controls"])
    # everything except version/derived_from/coverage/sets.T8_support is identical to v1
    v2 = copy.deepcopy(CONTRACT_V2)
    for k in ("version", "derived_from"):
        v2.pop(k)
    v2["controls"].pop("coverage")
    v2["sets"].pop("T8_support")
    v1 = copy.deepcopy(CONTRACT_V1)
    v1.pop("version")
    assert v1 == v2, "v2 changed something besides the coverage clarification"


if __name__ == "__main__":
    _check()
    print("contract v1 sha256", contract_v1_sha256())
    print("contract v2 consistent; sha256", contract_v2_sha256())
    print("support selection sha256", hashlib.sha256(SUPPORT_SELECTION.read_bytes()).hexdigest())
