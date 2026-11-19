"""F24-only audit of frozen WB-assay objective drift, without fitting.

Reuses saved adaptation-OOF L1/DeltaE and computes the missing untouched
R11 linear-L1 baseline on these same F scenes. No E4, K, monitor,
finalHeldOut or T8 targets. Old reports/checkpoints are never overwritten.
"""
import json

import numpy as np
import torch

import run_cn_wb_assay as A

OUT = A.S.DIAG / "cn-wb-assay-objective-audit.json"


def main():
    if OUT.exists():
        raise ValueError("Audit already exists; refusing to overwrite evidence")
    manifest = A.load_manifest()
    report = json.loads(A.REPORT.read_text())
    allowed = {s["name"] for s in manifest["scenes"]}
    if len(allowed) != 24 or set(report["per_scene"]) != allowed:
        raise ValueError("Assay report does not match F24")
    if report["manifest_sha256"] != A.MANIFEST_SHA:
        raise ValueError("Assay manifest hash mismatch")
    for name, digest in report["checkpoint_sha256"].items():
        if A.S.sha256(A.S.CKPT / name) != digest:
            raise ValueError(f"Frozen assay checkpoint changed: {name}")
    logs, fingerprints = {}, {}
    for fold in range(4):
        for arm in A.ARMS:
            path = A.log_path(arm, fold)
            log = json.loads(path.read_text())
            if log["manifest_sha256"] != A.MANIFEST_SHA or log["history"][-1]["epoch"] != 16:
                raise ValueError("Unexpected assay log provenance")
            ckname = A.ckpt_path(arm, fold).name
            if log["checkpoint_sha256"] != report["checkpoint_sha256"][ckname]:
                raise ValueError("Assay log/checkpoint mismatch")
            logs[arm, fold] = log
            fingerprints[path.name] = A.S.sha256(path)
    dev = A.S.device()
    model, config = A.r11_model(dev)
    rows = {}
    for name in sorted(allowed):
        x, target = A.load_pairs(manifest, [name])[name]
        with torch.no_grad():
            pred = A.fwd(model, config, x, torch.tensor([1.0]), dev)
            baseline = float((pred - target.to(dev)).abs().mean())
        r = report["per_scene"][name]
        fold = r["fold"]
        row = {"session": r["session"], "fold": fold, "r11_linear_l1": baseline,
               "r11_dE00": r["r11_untouched"]["mean_dE00"]}
        for arm in A.ARMS:
            log = logs[arm, fold]
            if name not in log["validation"] or name in log["train"]:
                raise ValueError("Scene is not adaptation-held-out in its fold")
            l1 = log["history"][-1]["val_l1"][name]
            de = r[arm]["mean_dE00"]
            row[arm] = {"linear_l1": l1, "dE00": de,
                        "l1_minus_r11": l1 - baseline,
                        "dE00_minus_r11": de - row["r11_dE00"]}
        rows[name] = row
        print(name, json.dumps(row), flush=True)
    summary = {}
    for arm in A.ARMS:
        vals = [r[arm] for r in rows.values()]
        summary[arm] = {
            "mean_linear_l1": float(np.mean([r["linear_l1"] for r in vals])),
            "mean_linear_l1_minus_r11": float(np.mean([r["l1_minus_r11"] for r in vals])),
            "l1_improved_scenes": sum(r["l1_minus_r11"] < 0 for r in vals),
            "dE00_improved_scenes": sum(r["dE00_minus_r11"] < 0 for r in vals),
            "both_worse_scenes": sum(r["l1_minus_r11"] > 0 and r["dE00_minus_r11"] > 0 for r in vals),
            "l1_improved_dE00_worse_scenes": sum(r["l1_minus_r11"] < 0 and r["dE00_minus_r11"] > 0 for r in vals),
        }
    sessions = sorted({r["session"] for r in rows.values()})
    effects = np.array([sum(r["source"]["dE00"] - r["meanctl"]["dE00"] for r in rows.values() if r["session"] == s) for s in sessions])
    counts = np.array([sum(r["session"] == s for r in rows.values()) for s in sessions])
    picks = np.random.default_rng(42).integers(len(sessions), size=(20000, len(sessions)))
    sampled = effects[picks].sum(axis=1) / counts[picks].sum(axis=1)
    A.S.check_source()
    result = {
        "scope": "F24 diagnostic, no fitting; R11 in-sample vs adaptation-OOF, not unseen-RAW acceptance",
        "fingerprints": {"manifest": A.MANIFEST_SHA, "assay_report": A.S.sha256(A.REPORT),
                         "source_checkpoint": A.S.SOURCE_SHA, "run_logs": fingerprints},
        "mean_r11_linear_l1": float(np.mean([r["r11_linear_l1"] for r in rows.values()])),
        "summary": summary,
        "paired_A_minus_B_cluster_bootstrap": {
            "unit": "whole capture session", "sessions": sessions, "resamples": 20000,
            "seed": 42, "equal_scene_weighted_ratio_of_cluster_sums": True,
            "percentile_95": np.percentile(sampled, [2.5, 97.5]).tolist(),
            "note": "Descriptive with only seven clusters; does not establish equivalence or causation",
        },
        "per_scene": rows,
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_scene", "fingerprints")}, indent=2))


if __name__ == "__main__":
    main()
