"""Recheck native encoding with the unedited pinned C++ gamma function.

Compiles only a standalone numeric lookup-table shim into ignored tools.
No decoder, RAF, TIFF, target, model or protected-tier access.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from eval_provia_gamma_length import libraw_forward_table

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report_path = DIAG / "provia-gamma-length-probe.json"
    r = json.loads(report_path.read_text())
    manifest_path = DIAG / "provia-gamma-length-probe-manifest.json"
    m = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == r["manifest_sha256"]
    for p, h in m["bindings"].items():
        assert sha(ROOT / p) == h
    t = json.loads((DIAG / "provia-gamma-wrapper-toolchain.json").read_text())
    source = next(s for s in t["build_curve_sources"] if s["source_path"].endswith("/curves.cpp"))
    path = ROOT / source["path"]
    assert sha(path) == source["sha256"]
    text = path.read_text()
    start = text.index("void LibRaw::gamma_curve(")
    end = text.index("\nvoid LibRaw::", start + 1)
    function = text[start:end]
    out = ROOT / "training/checkpoints/tools/gamma-curve-numeric-reference"
    out.mkdir(exist_ok=True)
    cpp = out / "reference.cpp"
    program = """#include <cmath>
#include <cstring>
#include <cstdio>
typedef unsigned short ushort;
#define SQR(x) ((x)*(x))
class LibRaw { public: ushort curve[65536]; double gamm[6]; void gamma_curve(double,double,int,int); };
""" + function + "\nint main(){ LibRaw x{}; x.gamma_curve(.45,4.5,2,65536); return fwrite(x.curve,2,65536,stdout)==65536 ? 0 : 1; }\n"
    cpp.write_text(program)
    binary = out / "reference"
    command = ["/usr/bin/clang++", "-O2", "-std=c++17", str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    data = subprocess.check_output([str(binary)], timeout=10)
    table = np.frombuffer(data, dtype="<u2")
    assert table.shape == (65536,)
    independent, _ = libraw_forward_table()
    diff = np.abs(table.astype(int) - independent.astype(int))
    decoded_path = ROOT / m["output_dir"] / "decode-complete.json"
    assert sha(decoded_path) == r["decode_report_sha256"]
    arms = json.loads(decoded_path.read_text())["scenes"][0]["arms"]
    old = np.asarray(arms["D2"]["native_samples"]["data"], dtype=int)
    linear = np.asarray(arms["D6"]["native_samples"]["data"], dtype=int)
    use = (linear >= .01 * 65535) & (linear <= .95 * 65535)
    err = np.abs(old[use] - table[linear[use]].astype(int))
    assert len(err) == r["native_default_curve_match"]["n"] == 5793
    assert float(err.mean()) == r["native_default_curve_match"]["mean_abs_codes"] == 0.
    assert int(err.max()) == 0 and r["native_default_curve_match"]["passed"]
    assert r["WB_ratio_screen"]["passed"] and not all(v["gate_passed"] for v in r["WB_affine_response"])
    assert r["behavioural_confirmation"] is False  # Joint qualification stays failed.
    result = {"verified": True, "code_sha256": sha(Path(__file__)), "report_sha256": sha(report_path),
              "pinned_cpp_source_sha256": source["sha256"], "reference_shim_sha256": sha(cpp),
              "compiler": subprocess.check_output(["/usr/bin/clang++", "--version"], text=True).splitlines()[0],
              "all65536_CPP_vs_numpy_table_exact": bool(np.all(diff == 0)), "all_table_max_code_difference": int(diff.max()),
              "native_samples": len(err), "CPP_native_max_code_error": int(err.max()),
              "encoding_fact": "Default0.45/4.5 lookup exactly maps six-element linear native output to legacy two-element native output on5793 samples",
              "joint_WB_qualification": "FAILED unchanged; no model/scoring/migration/app change"}
    dest = DIAG / "provia-gamma-length-verification.json"
    assert not dest.exists()
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
