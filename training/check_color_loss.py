"""Synthetic CPU/MPS value and gradient checks; no model/data access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from color_loss import linear_rgb_to_lab, per_scene_terms
from color_metrics import linear_srgb_to_lab, self_test

OUT = Path(__file__).resolve().parent / "diagnostics" / "provia-color-loss-checks.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    output = parser.parse_args().output
    if output.exists():
        raise ValueError("Check report exists; refusing to overwrite")
    rng = np.random.default_rng(42)
    sample = rng.uniform(-0.2, 1.2, (2, 3, 17, 19))
    sample[:, :, 0, :4] = np.array([0, 1, -0.1, 1.1])
    ref = linear_srgb_to_lab(sample.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    double = torch.from_numpy(sample)
    error64 = float(np.abs(linear_rgb_to_lab(double).numpy() - ref).max())
    assert error64 < 1e-10, error64
    probe = torch.tensor(rng.uniform(0.02, 0.98, (1, 3, 3, 4)), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(linear_rgb_to_lab, (probe,), eps=1e-6, atol=1e-5, rtol=1e-4)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    results = {}
    target_values = rng.uniform(0, 1, sample.shape)
    cpu_grad, cpu_terms = None, None
    for device in devices:
        x = torch.tensor(sample, dtype=torch.float32, device=device, requires_grad=True)
        got = linear_rgb_to_lab(x)
        err = float(np.abs(got.detach().cpu().numpy() - ref).max())
        assert err < 1e-3, (device, err)
        target = torch.tensor(target_values, dtype=torch.float32, device=device)
        terms = per_scene_terms(x, target)
        assert all(bool(torch.isfinite(v).all()) for v in terms.values())
        terms["combined"].mean().backward()
        assert bool(torch.isfinite(x.grad).all()), device
        grad = x.grad.detach().cpu().numpy()
        term_values = {k: v.detach().cpu().numpy() for k, v in terms.items()}
        if device == "cpu":
            cpu_grad, cpu_terms = grad, term_values
        grad_error = float(np.abs(grad - cpu_grad).max())
        term_error = max(float(np.abs(v - cpu_terms[k]).max()) for k, v in term_values.items())
        assert grad_error < 1e-5 and term_error < 1e-5, (device, grad_error, term_error)
        # Identity, black and white must have finite zero-loss gradients.
        for value in (0.0, 0.5, 1.0):
            p = torch.full((1, 3, 3, 4), value, device=device, requires_grad=True)
            loss = per_scene_terms(p, p.detach().clone())["combined"].mean()
            loss.backward()
            assert float(loss.detach()) == 0 and bool(torch.isfinite(p.grad).all())
        # Display Lab alone cannot repair predictions outside [0,1]; L1 can.
        for value, expected in ((-0.1, -1), (1.1, 1)):
            p = torch.full((1, 3, 3, 4), value, device=device, requires_grad=True)
            y = torch.full_like(p, 0 if value < 0 else 1)
            per_scene_terms(p, y)["combined"].mean().backward()
            assert bool(torch.isfinite(p.grad).all()) and bool((p.grad * expected > 0).all())
        results[device] = {"max_abs_Lab_error_vs_numpy_float64": err,
                           "max_abs_gradient_difference_vs_cpu": grad_error,
                           "max_abs_loss_term_difference_vs_cpu": term_error,
                           "random_identity_endpoint_and_out_of_range_gradients": "finite; restoring outside gamut"}
    bad = torch.zeros(1, 4, 2, 2)
    try:
        linear_rgb_to_lab(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-RGB input not refused")
    result = {"scope": "synthetic loss checks only; no images, checkpoints or training",
              "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ("color_loss.py", "color_metrics.py", "check_color_loss.py")},
              "cpu_float64_max_abs_Lab_error": error64, "cpu_float64_gradcheck": True,
              "frozen_CIEDE2000_Sharma_max_error": self_test(), "float32_devices": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
