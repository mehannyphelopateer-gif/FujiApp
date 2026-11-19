"""One-shot, bounded export stage waiting for Claude's frozen CN checkpoint.

No training, K evaluation, monitor/finalHeldOut processing or app changes.
Waits at most 12 hours by default, then exports seed 42 and checks numerical
parity using F fixtures. Writes only to its own gitignored control directory.
This is a finite dependency stage, not a recurring task or deployment.
"""
import argparse
import datetime
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "training"
DIAG = TRAIN / "diagnostics"
CKPT = TRAIN / "checkpoints"
SOURCE_SHA = "1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c"
APP_MODEL_SHA = "5f155bbf83a1e112756a3c4a96554a9f87053de58dd68bb54d32e6d6f0fd264d"
V2_SHA = "9824b0799737960363c95bb028332fdd0e5118e748fc0139375f1d0e230099a0"
PINNED_RUN = "27dc5918ce962e3cac516f8768bcb5c340bcd761c11a6dacd3303ed6573e4416"
PINNED_FOLDS = "76b55d9126caf74d13f99aa9735235d51e013892dfdd99a95d8e36f9f7d1add0"
SOURCE = CKPT / "r11-drcond-seed42-best.pt"
APP_MODEL = TRAIN / "exports/round11/round11-seed42.onnx"
RUN = DIAG / "cuban-step1-run-manifest.json"
FOLDS = DIAG / "cuban-step1-folds.json"
FREEZE = DIAG / "cuban-step1-freeze.json"
SELECTION = DIAG / "cuban-step1-selection.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def protected_files():
    require(sha256(SOURCE) == SOURCE_SHA, "Original Round 11 checkpoint changed")
    require(sha256(APP_MODEL) == APP_MODEL_SHA, "Original app ONNX model changed")
    require(sha256(RUN) == PINNED_RUN, "Frozen CN run manifest changed")
    require(sha256(FOLDS) == PINNED_FOLDS, "Frozen CN fold manifest changed")


def frozen_candidate():
    """Metadata/weight checks only; never loads an image or target."""
    import torch
    from cuban_contract_v2 import _check, contract_v2_sha256
    _check()
    require(contract_v2_sha256() == V2_SHA, "Contract v2 changed")
    protected_files()
    freeze = json.loads(FREEZE.read_text())
    run = json.loads(RUN.read_text())
    require(freeze["primary"] == "cn-step1-seed42.pt", "Unexpected primary checkpoint")
    require(freeze["contract_v2_sha256"] == V2_SHA, "Frozen candidate contract mismatch")
    require(freeze["run_manifest_sha256"] == PINNED_RUN, "Candidate run fingerprint mismatch")
    require(freeze["folds_sha256"] == PINNED_FOLDS, "Candidate fold fingerprint mismatch")
    fit_names = sorted(r["name"] for r in run["F_allowlist"])
    require(len(fit_names) == len(set(fit_names)) == 24, "F allowlist is not exactly 24")
    require(not set(fit_names) & {r["name"] for r in run["K_eval_only"]}, "F/K overlap")
    require(freeze["selection_sha256"] == sha256(SELECTION), "Selection changed after freezing")
    selection = json.loads(SELECTION.read_text())["chosen"]
    require(selection["lr"] in (1e-4, 3e-4), "Unapproved selected LR")
    require(isinstance(selection["epoch"], int) and 1 <= selection["epoch"] <= 20, "Unapproved epoch")
    require(math.isfinite(selection["mean_oof_l1"]) and selection["mean_oof_l1"] >= 0, "Nonfinite OOF result")
    expected = {f"cn-step1-seed{s}.pt" for s in (42, 43, 44)}
    require(set(freeze["checkpoints"]) == expected, "Incomplete frozen final seed set")
    for name in expected:
        require(sha256(CKPT / name) == freeze["checkpoints"][name], f"Frozen weights changed: {name}")
    for seed in (42, 43, 44):
        name = f"cn-step1-final-seed{seed}.json"
        log = TRAIN / "runs" / name
        require(freeze["final_logs_sha256"][name] == sha256(log), "Final training log changed")
        for row in json.loads(log.read_text())["history"]:
            if "train_l1" in row:
                require(math.isfinite(row["train_l1"]) and row["train_l1"] >= 0, "Nonfinite final training loss")
    checkpoint = CKPT / freeze["primary"]
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    meta = ck["cuban_step1"]
    require(meta["seed"] == 42 and meta["train"] == fit_names, "Candidate was not adapted on F24 only")
    require(meta["lr"] == selection["lr"] and meta["stop_epoch"] == selection["epoch"], "Candidate selection mismatch")
    require(meta["T_max"] == 20 and meta["selection_sha256"] == freeze["selection_sha256"], "Schedule/selection mismatch")
    require(meta["run_manifest_sha256"] == PINNED_RUN and meta["folds_sha256"] == PINNED_FOLDS, "Candidate data provenance mismatch")
    require(meta["contract_v2_sha256"] == V2_SHA, "Candidate contract mismatch")
    require(meta["source_sha256_before_after"] == [SOURCE_SHA, SOURCE_SHA], "Initialization provenance mismatch")
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)["state_dict"]
    for name, value in source.items():
        if name.startswith("grid.encoder."):
            require(torch.equal(value, ck["state_dict"][name]), f"Frozen encoder changed: {name}")
    provenance = {"target_film_simulation": "Classic Negative", "target_output_dr": 100,
                  "seed": 42, "checkpoint_sha256": freeze["checkpoints"][checkpoint.name],
                  "contract_sha256": V2_SHA, "adaptation_fit_names": fit_names,
                  "run_manifest": {"path": str(RUN.relative_to(REPO)), "sha256": PINNED_RUN},
                  "fold_manifest": {"path": str(FOLDS.relative_to(REPO)), "sha256": PINNED_FOLDS}}
    return checkpoint, provenance, sha256(FREEZE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control-dir", type=Path, required=True)
    ap.add_argument("--wait-hours", type=float, default=12)
    args = ap.parse_args()
    require(0 < args.wait_hours <= 16, "Wait must be bounded to at most 16 hours")
    root = args.control_dir.resolve()
    root.relative_to(CKPT.resolve())
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stage.lock").open("x") as lock:
        import os
        lock.write(str(os.getpid()))
    def status(state, **extra):
        data = {"state": state, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scope": "one F-only export stage; no training, accuracy acceptance or app integration", **extra}
        tmp = root / "status.json.partial"
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        tmp.replace(root / "status.json")
        print(json.dumps(data), flush=True)
    try:
        protected_files()
        files = [TRAIN / "export_cn_step1_onnx.py", REPO / "scripts/verify-cn-step1-onnx-web.mjs",
                 REPO / "scripts/verify-round11-onnx-web.mjs", TRAIN / "export_round11_onnx.py", TRAIN / "model.py"]
        code_hashes = {str(p): sha256(p) for p in files}
        status("waiting-for-frozen-checkpoint", wait_hours=args.wait_hours)
        deadline = time.monotonic() + args.wait_hours * 3600
        while not FREEZE.exists():
            protected_files()
            if time.monotonic() >= deadline:
                status("timed-out-waiting", reason="No frozen candidate arrived; no export was attempted")
                return 2
            time.sleep(30)
        # Wait for a possibly just-created file to finish its small JSON write.
        for attempt in range(30):
            try:
                json.loads(FREEZE.read_text())
                break
            except (OSError, json.JSONDecodeError):
                if attempt == 29:
                    raise
                time.sleep(1)
        for p, h in code_hashes.items():
            require(sha256(p) == h, f"Export code changed while waiting: {p}")
        checkpoint, provenance, freeze_hash = frozen_candidate()
        provenance_path = root / "export-provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        output = root / "parity"
        status("exporting-native", checkpoint=checkpoint.name, freeze_sha256=freeze_hash)
        subprocess.run([sys.executable, str(TRAIN / "export_cn_step1_onnx.py"), "--checkpoint", str(checkpoint),
                        "--provenance", str(provenance_path), "--output-dir", str(output)], cwd=REPO, check=True, timeout=600)
        status("checking-wasm-and-webgpu", vector_dir=str(output.relative_to(REPO)))
        subprocess.run(["node", str(REPO / "scripts/verify-cn-step1-onnx-web.mjs"), "--vector-dir", str(output),
                        "--ort-dir", str(CKPT / "onnx-web-deps/node_modules/onnxruntime-web/dist")], cwd=REPO, check=True, timeout=600)
        protected_files()
        require(sha256(FREEZE) == freeze_hash, "Checkpoint freeze changed during parity checks")
        status("numerical-parity-passed", report=str((output / "cn-export-gate-report.json").relative_to(REPO)),
               accuracy_acceptance=False, app_integration=False)
        return 0
    except Exception as e:
        status("stopped", reason=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
