"""Reproducible Round 11 export/parity gate; inference only, train/monitor only.

Exports seeds 42/43 as dynamic H/W ONNX graphs, including the real trained
DR conditioning. Writes FP32 test vectors for scripts/verify-round11-onnx-web.mjs.
All generated weights/vectors stay in the gitignored checkpoints directory.
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

from dataset import CAL_DIR, list_scenes
from fjlrg import load_fjlrg_linear
from model import HybridPredictor, apply_bilateral_grid, resize_for_network

ROOT = Path(__file__).resolve().parent
MAX_ERROR = 1e-4
MEAN_ERROR = 1e-5


class ExportGroupNorm(torch.nn.Module):
    """Same GroupNorm math/weights, with bounded-length reduction axes.

    ORT Web WASM's InstanceNormalization loses precision when reducing a
    full 1536px frame in one flat pass. Reducing width, height, then channels
    separately avoids long accumulations; variance remains centered. This
    conversion applies only to the export copy, never the trained model.
    """
    def __init__(self, source):
        super().__init__()
        self.groups = source.num_groups
        self.channels_per_group = source.num_channels // source.num_groups
        self.eps = source.eps
        self.weight = source.weight
        self.bias = source.bias

    def forward(self, x):
        b, _, h, w = x.shape
        grouped = x.reshape(b, self.groups, self.channels_per_group, h, w)
        mean = grouped.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).mean(dim=2, keepdim=True)
        centered = grouped - mean
        variance = centered.square().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).mean(dim=2, keepdim=True)
        normalized = (centered * torch.rsqrt(variance + self.eps)).reshape_as(x)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def replace_export_group_norms(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.GroupNorm):
            module._modules[name] = ExportGroupNorm(child)
            count += 1
        else:
            count += replace_export_group_norms(child)
    return count


class InferenceWrapper(torch.nn.Module):
    """One full-resolution RGB input plus one normalized capture-DR scalar.

    The fourth encoder plane is assembled inside the graph, after the exact
    training resize. RGB/refinement/grid-slicing inputs remain three-channel.
    """
    def __init__(self, model, config, stages=False):
        super().__init__()
        self.model = model
        self.low_res = config["low_res"]
        self.luma_mode = config["luma_mode"]
        self.stages = stages

    def forward(self, linear_rgb, capture_dr_condition):
        if self.stages:
            low = resize_for_network(linear_rgb, self.low_res)
            grid = self.model.grid(low, capture_dr_condition)
            grid_output = apply_bilateral_grid(grid, linear_rgb, self.luma_mode)
            final, delta = self.model.refinement(linear_rgb, grid_output)
            return final, low, grid, grid_output, delta
        return self.model(linear_rgb, self.low_res, self.luma_mode,
                          condition=capture_dr_condition)[0]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, choices=[42, 43], default=[42, 43])
    ap.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "round11-onnx")
    ap.add_argument("--diagnostic-stages", action="store_true", help="Include intermediate outputs to isolate backend drift")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    # Explicit scene names are resolved ONLY against the permitted tiers.
    approved = {s["name"]: s for tier in ("train", "monitor") for s in list_scenes(tier)}
    cases = []
    for name, expected_dr in [("DSCF1419.RAF", 100), ("DSCF0878.RAF", 200), ("DSCF0538.RAF", 400)]:
        scene = approved[name]
        if scene.get("captureDynamicRange") != expected_dr:
            raise ValueError(f"{name}: unexpected or missing capture DR")
        rgb, _, _ = load_fjlrg_linear(CAL_DIR / scene["shoot"] / "browser-phase3-linear.fjlrg")
        cases.append((name.removesuffix(".RAF"), np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), expected_dr,
                      {"shoot": scene["shoot"], "source": "train-or-monitor"}))
    rng = np.random.default_rng(11042)
    small = rng.random((1, 3, 73, 111), dtype=np.float32)
    for dr in (100, 200, 400):
        cases.append((f"synthetic-dr{dr}", small, dr, {"source": "synthetic"}))

    vectors = []
    for name, image, dr, source in cases:
        filename = f"{name}-input.f32"
        image.astype("<f4").tofile(args.output_dir / filename)
        vectors.append({"name": name, "shape": list(image.shape), "capture_dr": dr,
                        "condition": float(np.log2(dr / 100) / 2), "input": filename, **source})

    report = {"format_version": 1, "opset": 20,
              "input_contract": {"linear_rgb": "float32 NCHW, batch=1, linear sRGB primaries/as-shot WB from .fjlrg",
                                 "capture_dr_condition": "float32 [1,1], log2(capture DR/100)/2: 100=0, 200=0.5, 400=1; reject unknown",
                                 "target_output_dr": 100,
                                 "output": "unclamped linear RGB, same shape as linear_rgb"},
              "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
              "native_tolerances": {"max_abs": MAX_ERROR, "mean_abs": MEAN_ERROR},
              "cases": vectors, "models": []}
    passed = True
    for seed in args.seeds:
        checkpoint = ROOT / "checkpoints" / f"r11-drcond-seed{seed}-best.pt"
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        if not cfg.get("capture_dr_conditioning") or not cfg.get("use_hybrid"):
            raise ValueError("Expected a trained, DR-conditioned Round 11 hybrid checkpoint")
        model = HybridPredictor(grid_spatial=cfg["grid_spatial"], grid_luma=cfg["grid_luma"],
                                low_res=cfg["low_res"], base_ch=cfg["base_ch"],
                                refinement_base_ch=cfg["refinement_base_ch"],
                                refinement_max_delta=cfg["refinement_max_delta"], condition_channels=1)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        reference = InferenceWrapper(model, cfg, args.diagnostic_stages).eval()
        export_model = copy.deepcopy(model)
        replaced_norms = replace_export_group_norms(export_model)
        wrapper = InferenceWrapper(export_model, cfg, args.diagnostic_stages).eval()
        output_names = ["corrected_linear_rgb"] + (["low_res", "bilateral_grid", "grid_output", "refinement_delta"] if args.diagnostic_stages else [])
        model_name = f"round11-seed{seed}.onnx"
        model_path = args.output_dir / model_name
        with torch.inference_mode():
            torch.onnx.export(wrapper, (torch.from_numpy(small), torch.tensor([[0.5]])), model_path,
                              opset_version=20, dynamo=False, external_data=False,
                              input_names=["linear_rgb", "capture_dr_condition"], output_names=output_names,
                              dynamic_axes={"linear_rgb": {2: "height", 3: "width"},
                                            "corrected_linear_rgb": {2: "height", 3: "width"}})
        onnx.checker.check_model(str(model_path), full_check=True)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        entries = []
        synthetic_outputs = []
        for vector, (_, image, _, _) in zip(vectors, cases):
            condition = np.array([[vector["condition"]]], dtype=np.float32)
            with torch.inference_mode():
                raw_expected = reference(torch.from_numpy(image), torch.from_numpy(condition))
                expected_outputs = [t.numpy() for t in raw_expected] if args.diagnostic_stages else [raw_expected.numpy()]
                expected = expected_outputs[0]
            actual = session.run(None, {"linear_rgb": image, "capture_dr_condition": condition})[0]
            error = np.abs(expected - actual)
            finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            ok = finite and float(error.max()) <= MAX_ERROR and float(error.mean()) <= MEAN_ERROR
            passed &= ok
            expected_name = f"seed{seed}-{vector['name']}-expected.f32"
            expected.astype("<f4").tofile(args.output_dir / expected_name)
            entry = {"name": vector["name"], "expected": expected_name,
                     "max_abs": float(error.max()), "mean_abs": float(error.mean()), "passed": ok}
            if args.diagnostic_stages:
                entry["stages"] = {}
                for stage_name, stage in zip(output_names[1:], expected_outputs[1:]):
                    stage_file = f"seed{seed}-{vector['name']}-{stage_name}.f32"
                    stage.astype("<f4").tofile(args.output_dir / stage_file)
                    entry["stages"][stage_name] = stage_file
            entries.append(entry)
            print(f"seed {seed} {entry['name']}: max={entry['max_abs']:.3g} mean={entry['mean_abs']:.3g} pass={ok}", flush=True)
            if vector["source"] == "synthetic":
                synthetic_outputs.append(actual)
        sensitivity = float(np.abs(synthetic_outputs[0] - synthetic_outputs[2]).mean())
        if sensitivity <= 1e-5:
            raise ValueError("Export lost meaningful capture-DR conditioning")
        report["models"].append({"seed": seed, "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
                                 "best_epoch": ckpt["epoch"], "model": model_name, "model_sha256": sha256(model_path),
                                 "bytes": model_path.stat().st_size, "native_cases": entries,
                                 "export_group_norm_replacements": replaced_norms,
                                 "same_image_dr100_vs_dr400_mean_difference": sensitivity})
    report["passed"] = passed
    (args.output_dir / "native-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise SystemExit("Native ONNX parity failed")
    print(f"Native parity passed. Browser test vectors: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
