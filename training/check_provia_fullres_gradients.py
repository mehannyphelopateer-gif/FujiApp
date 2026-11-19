"""Full working-resolution CPU/MPS gradient comparison on one fixed F case.

No optimizer steps, fitting, saved weights or protected evaluation data.
Uses the already-frozen probe's gradient and original-data helpers.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
from pathlib import Path

import torch

import probe_provia_updates as Q

OUT = Q.P.DIAG / "provia-fullres-gradient-check.json"
NAME = "DSCF0514.RAF"


def main():
    if OUT.exists():
        raise ValueError("Report exists; refusing overwrite")
    probe = json.loads(Q.MANIFEST.read_text())
    if Q.P.sha256(Path(Q.__file__)) != probe["code_sha256"]:
        raise ValueError("Frozen diagnostic helper changed")
    if NAME not in [r["name"] for r in probe["scenes"]] or NAME not in Q.FIXED_F:
        raise ValueError("Case not in frozen F probe set")
    manifest = Q.P.load_manifest()
    x, y, cond = Q.P.Pairs(manifest).get(NAME)
    if not torch.backends.mps.is_available():
        raise ValueError("MPS not available")
    cpu, cfg = Q.P.build(torch.device("cpu")); cpu.eval()
    mps, mc = Q.P.build(torch.device("mps")); mps.eval()
    results = {}
    for arm in Q.P.ARMS:
        cpu_loss, a = Q.gradient(cpu, cfg, x, y, cond, arm, torch.device("cpu"))
        mps_loss, b = Q.gradient(mps, mc, x, y, cond, arm, torch.device("mps"))
        a, b = a.double(), b.double()
        rel = float((a - b).norm() / a.norm())
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        results[arm] = {"cpu_loss": cpu_loss, "mps_loss": mps_loss,
                        "gradient_relative_L2": rel, "gradient_cosine": cos,
                        "gradient_max_abs_difference": float((a - b).abs().max()),
                        "passed": rel <= 0.01 and cos >= 0.999}
        print(arm, json.dumps(results[arm]), flush=True)
    Q.P.check_source()
    row = next(r for r in manifest["scenes"] if r["name"] == NAME)
    result = {"scope": "One fixed F working-resolution numerical check only; no optimizer step or weight file",
              "name": NAME, "shape": list(x.shape), "source_checkpoint_sha256": Q.P.SOURCE_SHA,
              "data_manifest_sha256": Q.P.MANIFEST_SHA, "input_sha256": row["fjlrg_sha256"],
              "target_cache_sha256": row["npy_sha256"], "probe_manifest_sha256": Q.P.sha256(Q.MANIFEST),
              "checker_sha256": Q.P.sha256(Path(__file__)), "results": results,
              "passed": all(r["passed"] for r in results.values()),
              "limitation": "Does not certify every image/backend operation or establish generalisation/accuracy"}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
