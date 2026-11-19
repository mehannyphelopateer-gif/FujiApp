"""Bounded optimizer diagnostics on frozen R11 copies; no candidate training.

Prepare/freeze the manifest and commit this implementation before `run`.
CPU/MPS synthetic gradient checks precede real-data reads. Real probes use
four already-fixed F examples plus metadata-first DR100/200 examples from
the existing 583-name train allowlist. Every trial is ONE fresh Adam step
from the same original weights, then discarded; nothing is selected/saved.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import provia_color_pilot as P

HERE = Path(__file__).resolve()
MANIFEST = P.DIAG / "provia-update-probe-manifest.json"
OUT = P.DIAG / "provia-update-probe-report.json"
RATES = (1e-4, 1e-5, 1e-6)
EPSILONS = (1e-3, 1e-4, 1e-5)
FIXED_F = ("DSCF0523.RAF", "DSCF0964.RAF", "DSCF0514.RAF", "DSCF0511.RAF")


def prepare():
    if MANIFEST.exists():
        raise ValueError("Manifest exists; refusing overwrite")
    source = P.load_manifest()
    by = {r["name"]: r for r in source["scenes"]}
    cohort = json.loads((P.DIAG / "cuban-cohort-proposal.json").read_text())["scenes"]
    f = {r["name"] for r in cohort if r["role"] == "fit" and r["tier"] == "train"}
    if not set(FIXED_F) <= f & set(by):
        raise ValueError("Fixed probes must remain approved F fitting examples")
    extra = [min(r["name"] for r in source["scenes"] if r["captureDynamicRange"] == dr) for dr in (100, 200)]
    names = list(FIXED_F) + extra
    if len(set(names)) != 6:
        raise ValueError("Expected six distinct train probes")
    result = {
        "status": "FROZEN before diagnostic updates; commit manifest/code before run",
        "scope": "Six-case local optimizer diagnostic, no validation/candidate selection or persistent model updates",
        "source_manifest_sha256": P.MANIFEST_SHA, "source_checkpoint_sha256": P.SOURCE_SHA,
        "code_sha256": P.sha256(HERE),
        "selection": "Original four fixed F visual examples, plus lowest filename per capture DR100/200 from the 583-name allowlist; metadata only",
        "scenes": [{"name": n, "captureDynamicRange": by[n]["captureDynamicRange"]} for n in names],
        "objectives": list(P.ARMS), "learning_rates": list(RATES),
        "synthetic": {"seed": 42, "shape": [1, 3, 73, 111], "input": "uniform 0.01..0.2", "target": "1.8*x+0.02",
                      "capture_dr_condition": 1.0, "CPU_double_directional_epsilons": list(EPSILONS),
                      "checks": "At least two directional relative errors <=0.05; CPU/MPS gradient cosine >=0.999 and relative L2 <=0.01"},
        "real_steps": "For each objective, compute six anchored per-image gradients and their equal-case mean; for each of seven directions and three rates apply exactly one fresh Adam step, restore original parameters before each trial, evaluate all six losses, discard updated weights",
        "adam": "betas 0.9/0.999, eps 1e-8, wd 0; same as earlier pilot",
        "limitations": "Six stratified cases are not a representative estimate of whole-pool risk or generalisation; rates are diagnostic perturbations, not HPO or selected settings; no endpoint checkpoint is written",
        "excluded": "K/T8/monitor/finalHeldOut/dr100Eval and all CN targets",
    }
    MANIFEST.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"manifest_sha256": P.sha256(MANIFEST), "scenes": result["scenes"]}, indent=2))


def parameters(model):
    return list(model.parameters())


def flat_weights(model):
    return torch.cat([p.detach().cpu().reshape(-1) for p in parameters(model)])


def assign_weights(model, vector):
    cursor = 0
    with torch.no_grad():
        for p in parameters(model):
            p.copy_(vector[cursor:cursor + p.numel()].view_as(p).to(p))
            cursor += p.numel()
    if cursor != vector.numel():
        raise ValueError("Parameter vector shape mismatch")


def assign_gradients(model, vector):
    cursor = 0
    for p in parameters(model):
        p.grad = vector[cursor:cursor + p.numel()].view_as(p).to(p).clone()
        cursor += p.numel()
    if cursor != vector.numel():
        raise ValueError("Gradient vector shape mismatch")


def gradient(model, config, x, y, cond, arm, device):
    model.zero_grad(set_to_none=True)
    terms = P.loss_terms(P.fwd(model, config, x, cond, device), y.to(device), arm)
    loss = P.arm_loss(arm, terms)
    loss.backward()
    g = torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).cpu().reshape(-1)
                   for p in parameters(model)])
    if not torch.isfinite(g).all() or not torch.isfinite(loss):
        raise ValueError("Non-finite diagnostic loss/gradient")
    return float(loss.detach()), g


def synthetic_checks():
    if not torch.backends.mps.is_available():
        raise ValueError("This backend diagnostic requires the approved local MPS device")
    rng = np.random.default_rng(42)
    values = rng.uniform(0.01, 0.2, (1, 3, 73, 111))
    results = {}
    for arm in P.ARMS:
        cpu, cfg = P.build(torch.device("cpu")); cpu.eval()
        x = torch.tensor(values, dtype=torch.float32)
        y = x * 1.8 + 0.02
        cond = torch.tensor([1.0])
        cpu_loss, gcpu = gradient(cpu, cfg, x, y, cond, arm, torch.device("cpu"))
        mps, mc = P.build(torch.device("mps")); mps.eval()
        mps_loss, gmps = gradient(mps, mc, x, y, cond, arm, torch.device("mps"))
        a, b = gcpu.double(), gmps.double()
        cosine = float(torch.dot(a, b) / (a.norm() * b.norm()))
        relative = float((a - b).norm() / a.norm())
        # CPU float64 directional derivatives through the entire trained model.
        cpu.double()
        xd, yd, cd = x.double(), y.double(), cond.double()
        value, gd = gradient(cpu, cfg, xd, yd, cd, arm, torch.device("cpu"))
        anchor = flat_weights(cpu)
        norm = gd.norm()
        direction = gd / norm
        checks = []
        for eps in EPSILONS:
            losses = []
            for sign in (-1, 1):
                assign_weights(cpu, anchor + sign * eps * direction)
                with torch.no_grad():
                    t = P.loss_terms(P.fwd(cpu, cfg, xd, cd, torch.device("cpu")), yd, arm)
                    losses.append(float(P.arm_loss(arm, t)))
            derivative = (losses[1] - losses[0]) / (2 * eps)
            checks.append({"epsilon": eps, "finite_difference": derivative, "autograd": float(norm),
                           "relative_error": abs(derivative - float(norm)) / float(norm)})
        assign_weights(cpu, anchor)
        passed = cosine >= 0.999 and relative <= 0.01 and sum(r["relative_error"] <= 0.05 for r in checks) >= 2
        results[arm] = {"cpu_loss_float32": cpu_loss, "mps_loss_float32": mps_loss,
                        "cpu_loss_float64": value, "gradient_cosine": cosine, "gradient_relative_L2": relative,
                        "directional_derivatives": checks, "passed": passed}
        print("synthetic", arm, json.dumps(results[arm]), flush=True)
        del cpu, mps
    return results


def evaluate(model, config, pairs, names, device):
    records = {}
    with torch.no_grad():
        for name in names:
            x, y, cond = pairs[name]
            terms = P.loss_terms(P.fwd(model, config, x, cond, device), y.to(device), "candidate")
            records[name] = {k: float(v) for k, v in terms.items()}
    return records


def run(expected_manifest_sha):
    if OUT.exists():
        raise ValueError("Report exists; refusing overwrite")
    if P.sha256(MANIFEST) != expected_manifest_sha:
        raise ValueError("Expected exact frozen probe manifest hash")
    manifest = json.loads(MANIFEST.read_text())
    if P.sha256(HERE) != manifest["code_sha256"]:
        raise ValueError("Probe implementation changed since freeze")
    data_manifest = P.load_manifest()
    synthetic = synthetic_checks()
    result = {"scope": manifest["scope"], "manifest_sha256": expected_manifest_sha,
              "synthetic": synthetic, "local_probes": {}, "selected_weights": False}
    if not all(r["passed"] for r in synthetic.values()):
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        raise ValueError("Gradient checks failed; no real-data trials run")
    names = [r["name"] for r in manifest["scenes"]]
    loader = P.Pairs(data_manifest)
    pairs = {name: loader.get(name) for name in names}
    device = torch.device("mps")
    model, config = P.build(device); model.eval()
    anchor = flat_weights(model)
    baseline = evaluate(model, config, pairs, names, device)
    result["baseline"] = baseline
    for arm in P.ARMS:
        gradients = {}
        for name in names:
            assign_weights(model, anchor)
            _, gradients[name] = gradient(model, config, *pairs[name], arm, device)
        mean_gradient = torch.stack(list(gradients.values())).mean(0)
        mat = torch.stack([gradients[n].double() for n in names])
        unit = mat / mat.norm(dim=1, keepdim=True)
        records = {"gradient_cosines": {n: dict(zip(names, (unit[i] @ unit.T).tolist())) for i, n in enumerate(names)},
                   "gradient_norms": {n: float(g.double().norm()) for n, g in gradients.items()}, "trials": []}
        objective = "linear_l1" if arm == "control" else "combined"
        baseline_mean = sum(r[objective] for r in baseline.values()) / len(names)
        for label, g in list(gradients.items()) + [("equal_six_case_mean", mean_gradient)]:
            direction = g.double() / (g.double().abs() + 1e-8)
            predicted_slope = -float(torch.dot(mean_gradient.double(), direction))
            for lr in RATES:
                assign_weights(model, anchor)
                model.zero_grad(set_to_none=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
                assign_gradients(model, g)
                optimizer.step()
                observed = evaluate(model, config, pairs, names, device)
                change = sum(r[objective] for r in observed.values()) / len(names) - baseline_mean
                records["trials"].append({"gradient_from": label, "lr": lr,
                                          "predicted_first_order_mean_loss_change": lr * predicted_slope,
                                          "observed_mean_loss_change": change,
                                          "own_loss_change": observed[label][objective] - baseline[label][objective] if label in baseline else None,
                                          "per_case_loss_changes": {n: {k: v - baseline[n][k] for k, v in row.items()} for n, row in observed.items()}})
            print("real", arm, label, "three single-step probes complete", flush=True)
        result["local_probes"][arm] = records
    assign_weights(model, anchor)
    if not torch.equal(flat_weights(model), anchor):
        raise ValueError("Failed to restore ephemeral model copy")
    P.check_source()
    result["source_checkpoint_unchanged"] = True
    result["limitations"] = manifest["limitations"]
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("report", OUT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--manifest-sha")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif not args.manifest_sha:
        parser.error("run requires --manifest-sha from the committed manifest")
    else:
        run(args.manifest_sha)
