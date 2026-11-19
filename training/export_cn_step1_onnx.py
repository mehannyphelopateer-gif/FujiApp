"""F-only export/parity tooling for the Classic Negative E4 candidate.

No training, target loading, K/monitor/finalHeldOut processing or app changes.
--smoke-round11 tests the unchanged architecture with Provia weights and
explicitly does NOT produce a Classic Negative acceptance result.
See docs/classic-negative-export-gate.md for the candidate provenance schema.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from cuban_contract import CONTRACT as CONTRACT_V1, contract_sha256 as contract_v1_sha256
from export_round11_onnx import InferenceWrapper, replace_export_group_norms
from fjlrg import load_fjlrg_linear
from model import HybridPredictor

REPO = Path(__file__).resolve().parent.parent
CAL = REPO / "calibration-input"
COHORT = REPO / "training/diagnostics/cuban-cohort-proposal.json"
ROUND11 = REPO / "training/checkpoints/r11-drcond-seed42-best.pt"
FROZEN_ROUND11_SHA256 = "1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c"
FROZEN_V2_SHA256 = "9824b0799737960363c95bb028332fdd0e5118e748fc0139375f1d0e230099a0"
FROZEN_RUN_SHA256 = "27dc5918ce962e3cac516f8768bcb5c340bcd761c11a6dacd3303ed6573e4416"
FROZEN_FOLDS_SHA256 = "76b55d9126caf74d13f99aa9735235d51e013892dfdd99a95d8e36f9f7d1add0"
FIXTURES = ("DSCF0523.RAF", "DSCF0964.RAF", "DSCF0514.RAF")
MAX_ERROR, MEAN_ERROR = 1e-4, 1e-5
CONFIG = {"grid_spatial": 8, "grid_luma": 9, "low_res": 256, "base_ch": 16,
          "refinement_base_ch": 8, "refinement_max_delta": 0.08,
          "luma_mode": "rec709", "capture_dr_conditioning": True, "use_hybrid": True}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_checkpoint(path):
    path = path.resolve(strict=True)
    path.relative_to((REPO / "training/checkpoints").resolve())
    if path.suffix != ".pt":
        raise ValueError("Expected a local .pt checkpoint")
    return path


def validate_provenance(provenance, checkpoint, seed, fit_names, contract, contract_hash):
    if contract["version"] != 2:
        raise ValueError("Candidate export requires the approved contract v2 implementation")
    if provenance.get("target_film_simulation") != "Classic Negative" or provenance.get("target_output_dr") != 100:
        raise ValueError("Expected Classic Negative E4/output DR100 provenance")
    if provenance.get("seed") != seed or provenance.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("Checkpoint/seed does not match frozen provenance")
    if provenance.get("contract_sha256") != contract_hash:
        raise ValueError("Candidate contract fingerprint mismatch")
    names = provenance.get("adaptation_fit_names")
    if not isinstance(names, list) or len(names) != 24 or set(names) != fit_names:
        raise ValueError("Adaptation inputs must be exactly the approved F24, with no K/monitor scenes")
    for field in ("run_manifest", "fold_manifest"):
        entry = provenance[field]
        path = (REPO / entry["path"]).resolve(strict=True)
        expected_name, expected_hash = (
            ("cuban-step1-run-manifest.json", FROZEN_RUN_SHA256) if field == "run_manifest"
            else ("cuban-step1-folds.json", FROZEN_FOLDS_SHA256))
        if path != REPO / "training/diagnostics" / expected_name or entry["sha256"] != expected_hash:
            raise ValueError(f"Unexpected {field} provenance")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen {field} fingerprint mismatch")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-round11", action="store_true")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--provenance", type=Path)
    ap.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    ap.add_argument("--output-dir", type=Path, required=True, help="A new, empty output directory")
    args = ap.parse_args()
    if args.smoke_round11 and (args.checkpoint or args.provenance or args.seed != 42):
        ap.error("Smoke mode uses only frozen Round 11 seed 42; no candidate/provenance arguments")
    if not args.smoke_round11 and (not args.checkpoint or not args.provenance):
        ap.error("Candidate mode requires --checkpoint and --provenance")
    output = args.output_dir.resolve()
    output.relative_to((REPO / "training/checkpoints").resolve())
    if output.exists():
        ap.error("Use a new output directory to prevent stale models/vectors/reports")

    rows = json.loads(COHORT.read_text())["scenes"]
    fit = {s["name"]: s for s in rows if s["role"] == "fit" and s["tier"] == "train"}
    if len(fit) != 24 or not set(FIXTURES) <= set(fit):
        raise ValueError("Frozen F24 fixture membership mismatch")
    checkpoint = local_checkpoint(ROUND11 if args.smoke_round11 else args.checkpoint)
    original_hash = sha256(ROUND11)
    if original_hash != FROZEN_ROUND11_SHA256:
        raise ValueError("Frozen Round 11 source checkpoint changed")
    checkpoint_hash = sha256(checkpoint)
    model_source_hash = sha256(REPO / "training/model.py")
    provenance = None
    contract, contract_hash = CONTRACT_V1, contract_v1_sha256()
    contract_source = REPO / "training/cuban_contract.py"
    if not args.smoke_round11:
        from cuban_contract_v2 import CONTRACT_V2, contract_v2_sha256, _check
        _check()
        contract, contract_hash = CONTRACT_V2, contract_v2_sha256()
        contract_source = REPO / "training/cuban_contract_v2.py"
        if contract_hash != FROZEN_V2_SHA256:
            raise ValueError("Contract v2 differs from the reviewed frozen fingerprint")
        if checkpoint == ROUND11:
            raise ValueError("Original Provia weights cannot be labeled a Classic Negative candidate")
        provenance = json.loads(args.provenance.read_text())
        validate_provenance(provenance, checkpoint, args.seed, set(fit), contract, contract_hash)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if provenance is not None:
        meta = ckpt["cuban_step1"]
        if (meta["seed"] != args.seed or meta["train"] != sorted(fit)
                or meta["contract_v2_sha256"] != contract_hash
                or meta["run_manifest_sha256"] != FROZEN_RUN_SHA256
                or meta["folds_sha256"] != FROZEN_FOLDS_SHA256
                or meta["source_sha256_before_after"] != [FROZEN_ROUND11_SHA256] * 2):
            raise ValueError("Checkpoint metadata does not match the approved adaptation")
    cfg = ckpt["config"]
    if any(cfg.get(k) != v for k, v in CONFIG.items()):
        raise ValueError("Checkpoint architecture/preprocessing differs from approved Step 1")
    torch.set_num_threads(2)
    model = HybridPredictor(grid_spatial=8, grid_luma=9, low_res=256, base_ch=16,
                            refinement_base_ch=8, refinement_max_delta=0.08, condition_channels=1)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    reference = InferenceWrapper(model, cfg).eval()
    export_model = copy.deepcopy(model)
    replacements = replace_export_group_norms(export_model)
    wrapper = InferenceWrapper(export_model, cfg).eval()
    output.mkdir(parents=True)
    cases, images = [], []
    for name in FIXTURES:
        row = fit[name]
        source = CAL / row["shoot"] / "browser-phase3-linear.fjlrg"
        input_hash = sha256(source)
        rgb, _, _ = load_fjlrg_linear(source)
        if sha256(source) != input_hash or not np.isfinite(rgb).all():
            raise ValueError("Fixture changed or contains nonfinite data")
        image = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        cases.append({"name": name[:-4], "source": "F-fit", "source_name": name,
                      "source_path": str(source.relative_to(REPO)), "source_sha256": input_hash,
                      "shoot": row["shoot"]})
        images.append(image)
    rng = np.random.default_rng(11042)
    for h, w in ((73, 111), (111, 73)):
        images.append(rng.random((1, 3, h, w), dtype=np.float32))
        cases.append({"name": f"synthetic-{h}x{w}", "source": "synthetic"})
    for case, image in zip(cases, images):
        filename = f"{case['name']}-input.f32"
        image.astype("<f4").tofile(output / filename)
        case.update({"shape": list(image.shape), "capture_dr": 400, "condition": 1.0,
                     "input": filename, "input_sha256": sha256(output / filename)})
    purpose = "provia-plumbing-smoke" if args.smoke_round11 else "classic-negative-step1-parity"
    name = f"{'provia-plumbing-smoke' if args.smoke_round11 else 'cn-step1'}-seed{args.seed}.onnx"
    path = output / name
    condition = np.ones((1, 1), dtype=np.float32)
    with torch.inference_mode():
        torch.onnx.export(wrapper, (torch.from_numpy(images[-1]), torch.from_numpy(condition)), path,
                          opset_version=20, dynamo=False, external_data=False,
                          input_names=["linear_rgb", "capture_dr_condition"],
                          output_names=["corrected_linear_rgb"],
                          dynamic_axes={"linear_rgb": {2: "height", 3: "width"},
                                        "corrected_linear_rgb": {2: "height", 3: "width"}})
    onnx.checker.check_model(str(path), full_check=True)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    native_cases = []
    for case, image in zip(cases, images):
        with torch.inference_mode():
            expected = reference(torch.from_numpy(image), torch.from_numpy(condition)).numpy()
        actual = session.run(None, {"linear_rgb": image, "capture_dr_condition": condition})[0]
        if actual.shape != expected.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError("Invalid native inference shape or values")
        error = np.abs(actual - expected)
        expected_name = f"seed{args.seed}-{case['name']}-expected.f32"
        expected.astype("<f4").tofile(output / expected_name)
        entry = {"name": case["name"], "expected": expected_name,
                 "expected_sha256": sha256(output / expected_name),
                 "max_abs": float(error.max()), "mean_abs": float(error.mean())}
        entry["passed"] = entry["max_abs"] <= MAX_ERROR and entry["mean_abs"] <= MEAN_ERROR
        native_cases.append(entry)
        print(f"{purpose}: {case['name']} max={entry['max_abs']:.3g} mean={entry['mean_abs']:.3g} passed={entry['passed']}", flush=True)
    if sha256(checkpoint) != checkpoint_hash or sha256(ROUND11) != original_hash:
        raise ValueError("Checkpoint changed during export")
    if sha256(REPO / "training/model.py") != model_source_hash:
        raise ValueError("Model source changed during export")
    if provenance is not None:
        validate_provenance(provenance, checkpoint, args.seed, set(fit), contract, contract_hash)
    report = {"format_version": 1, "purpose": purpose, "opset": 20,
              "target_film_simulation": "Provia" if args.smoke_round11 else "Classic Negative",
              "target_output_dr": 100, "accuracy_acceptance": False,
              "scope": "F-only inference parity at working resolution, capture DR400; no targets or K/monitor/finalHeldOut",
              "contract_version": contract["version"], "contract_sha256": contract_hash,
              "contract_source": str(contract_source.relative_to(REPO)),
              "contract_source_sha256": sha256(contract_source),
              "export_code_sha256": sha256(Path(__file__)),
              "normalization_helper_sha256": sha256(REPO / "training/export_round11_onnx.py"),
              "cohort_sha256": sha256(COHORT), "model_source_sha256": model_source_hash,
              "original_round11_sha256": original_hash,
              "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
              "native_tolerances": {"max_abs": MAX_ERROR, "mean_abs": MEAN_ERROR},
              "cases": cases, "models": [{"seed": args.seed, "checkpoint": str(checkpoint.relative_to(REPO)),
                                          "checkpoint_sha256": checkpoint_hash, "model": name,
                                          "model_sha256": sha256(path), "native_cases": native_cases,
                                          "export_group_norm_replacements": replacements}],
              "passed": all(c["passed"] for c in native_cases)}
    if provenance is not None:
        report["candidate_provenance"] = provenance
        report["candidate_provenance_sha256"] = sha256(args.provenance)
    (output / "native-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise SystemExit("Native parity failed")
    print(f"Native parity passed: {output}; accuracy_acceptance=false", flush=True)


if __name__ == "__main__":
    main()
