"""Synthetic cross-language codec fixtures; never opens a calibration folder."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import fjlrg_v2 as V
from test_fjlrg_v2 import make_case


def main(folder):
    root = Path(folder).resolve()
    require = lambda ok, msg: None if ok else (_ for _ in ()).throw(ValueError(msg))
    require((root / "synthetic-marker.txt").read_text() == "FujiApp v2 synthetic conformance only\n", "Missing synthetic fixture marker")
    cases = json.loads((root / "cases.json").read_text())
    results = []
    for row in cases:
        name = row["name"]
        require(Path(name).name == name and name.startswith("js-") and name.endswith(".fjlrg"), "Non-synthetic filename")
        data = (root / name).read_bytes()
        provenance = V.strict_json_loads((root / (name + ".json")).read_text())
        try:
            f = V.decode_fjlrg_v2(data, provenance, expected_contract=row["contract"])
            accepted = True
        except (ValueError, TypeError):
            accepted = False
        require(accepted == row["expected_accepted"], "Python conformance mismatch for " + name)
        if accepted:
            rebuilt = V.encode_fjlrg_v2(f.width, f.height, f.rgb_uint16, input_contract=f.input_contract,
                                        wb_levels_rgbg=f.provenance["wb_source"]["levels_rgbg"], options_sha256=f.provenance["options_sha256"])
            require(rebuilt == data, "Python cannot reproduce JS bytes")
            (root / (name + ".python-reencoded")).write_bytes(rebuilt)
            if row.get("full_range"):
                floats = V.to_unit_float32(f).astype("<f4").tobytes()
                require(floats == (root / (name + ".js-float32")).read_bytes(), "Float32 normalisation differs")
        results.append({"name": name, "accepted": accepted, "reencoded_exact": accepted})
    py = []
    for contract, name in ((V.CONTRACT_ASSHOT, "py-asshot.fjlrg"), (V.CONTRACT_AUTOWB, "py-autowb.fjlrg")):
        data, prov, _ = make_case(contract)
        (root / name).write_bytes(data)
        (root / (name + ".json")).write_text(json.dumps(prov))
        py.append({"name": name, "contract": contract})
    # Include semantic JSON numbers that were a real implementation difference.
    data, prov, _ = make_case(V.CONTRACT_ASSHOT)
    prov["source_wb_mode"] = float(prov["source_wb_mode"])
    prov["source_wb_fine_tune_raw"] = [float(v) for v in prov["source_wb_fine_tune_raw"]]
    prov["capture_dynamic_range"] = 400.0
    prov["decoded_native_size"] = [float(v) for v in prov["decoded_native_size"]]
    V.decode_fjlrg_v2(data, prov, expected_contract=V.CONTRACT_ASSHOT)
    name = "py-number-semantics.fjlrg"
    (root / name).write_bytes(data)
    (root / (name + ".json")).write_text(json.dumps(prov))
    py.append({"name": name, "contract": V.CONTRACT_ASSHOT})
    report = {"js_cases": results, "python_cases": py}
    (root / "python-result.json").write_text(json.dumps(report))
    print(json.dumps({"js_cases_checked": len(results), "python_cases_written": len(py)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fixture_directory")
    main(p.parse_args().fixture_directory)
