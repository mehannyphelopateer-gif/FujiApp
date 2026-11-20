"""TRAIN-ONLY target-assisted per-image free-grid diagnostic (Codex scope e838a47;
plan provia-grid-oracle-plan.json 77ce67bc...). Not validation, not a learned
predictor, not an inference fallback.

For each of the fixed eight review scenes: initialise an independent
(1,12,9,8,8) grid leaf from original R11's encoder prediction; freeze every
model parameter; forward = check_provia_grid_oracle.oracle_forward (unchanged
bilateral slicing + frozen refinement); fresh Adam (lr 1e-3, betas 0.9/0.999,
eps 1e-8, wd 0), exactly 200 steps, no schedule; loss = color_loss.per_scene_terms
combined (original display-clamped). Observe steps 0/25/50/100/200 (200 = fixed
endpoint) with the unchanged display CIEDE2000 scorer. Endpoint grids are saved
only as hash-recorded diagnostic arrays (gitignored), never as model weights.

  python3 training/run_provia_grid_oracle.py manifest | run
"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_loss as CL  # noqa: E402
import provia_extended as E  # noqa: E402
from audit_r11_display_domain import domain_stats  # noqa: E402
from check_provia_grid_oracle import oracle_forward  # noqa: E402
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402
from model import resize_for_network  # noqa: E402
from review_provia_low_rate import encode  # noqa: E402

P = E.P
PLAN = P.DIAG / "provia-grid-oracle-plan.json"
PLAN_SHA = "77ce67bcc673f8ef0d1f1e623b5e714b1c46f3f2c297cbed494a33bf6b70dd08"
PREFLIGHT = P.DIAG / "provia-grid-oracle-preflight.json"
MANIFEST = P.DIAG / "provia-grid-oracle-run-manifest.json"
MANIFEST_SHA = "fed65c01e4d084139810dc6550b9f29b4539e47fb9e6356435a0da74d5d45f6e"
REPORT = P.DIAG / "provia-grid-oracle-report.json"
PANELS = P.DIAG / "provia-grid-oracle-panels"
GRIDS = P.CKPT / "provia-grid-oracle-grids"
LR, STEPS, OBS = 1e-3, 200, (0, 25, 50, 100, 200)
CODE = ["run_provia_grid_oracle.py", "check_provia_grid_oracle.py", "model.py", "color_loss.py", "color_metrics.py",
        "provia_extended.py", "provia_color_pilot.py", "audit_r11_display_domain.py", "review_provia_low_rate.py"]


def cmd_manifest():
    if MANIFEST.exists():
        P.fail("oracle run manifest already frozen")
    if P.sha256(PLAN) != PLAN_SHA:
        P.fail("oracle plan changed")
    plan = json.loads(PLAN.read_text())
    for f, h in plan["bindings"]["code"].items():
        if P.sha256(P.TRAIN / f) != h:
            P.fail(f"plan-bound code {f} changed")
    pre = json.loads(PREFLIGHT.read_text())
    if pre["plan_sha256"] != PLAN_SHA or sorted(pre["per_scene"]) != sorted(E.REVIEW_CASES) or \
            any(v["max_abs_vs_original_R11"] != 0.0 for v in pre["per_scene"].values()):
        P.fail("preflight evidence mismatch")
    man, data, _, _ = E.load()
    by = {s["name"]: s for s in data["scenes"]}
    if plan["scenes"] != [by[n] for n in E.REVIEW_CASES]:
        P.fail("scene binding mismatch")
    pr = plan["proposed_runner"]
    if (pr["lr"], pr["steps"], tuple(pr["observations"]), pr["endpoint"]) != (LR, STEPS, OBS, STEPS):
        P.fail("runner constants do not match plan")
    out = {"status": "FROZEN before oracle fitting (Codex scope e838a47)",
           "label": "TARGET-ASSISTED per-image diagnostic on train scenes - not validation, not a model",
           "bindings": {"plan": PLAN_SHA, "preflight": P.sha256(PREFLIGHT), "source_R11": P.SOURCE_SHA,
                        "extended_manifest": E.MANIFEST_SHA, "extended_eval": P.sha256(E.EVAL),
                        "code": {f: P.sha256(P.TRAIN / f) for f in CODE if f != "run_provia_grid_oracle.py"}},
           "runner": pr, "scenes": E.REVIEW_CASES,
           "outputs": {"report": str(REPORT.relative_to(P.TRAIN.parent)), "panels": "gitignored local JPEGs",
                       "endpoint_grids": "gitignored .npy per scene, sha256 recorded; diagnostic only, not weights"},
           "runner_note": "run_provia_grid_oracle.py binds this manifest by pinned hash; committed version is the one in the commit adding this file"}
    MANIFEST.write_text(json.dumps(out, indent=1))
    print("oracle run manifest sha256", P.sha256(MANIFEST))


def load():
    if MANIFEST_SHA == "__UNPINNED__" or P.sha256(MANIFEST) != MANIFEST_SHA:
        P.fail("oracle run manifest missing/unpinned/changed")
    m = json.loads(MANIFEST.read_text())
    for f, h in m["bindings"]["code"].items():
        if P.sha256(P.TRAIN / f) != h:
            P.fail(f"bound code {f} changed")
    if P.sha256(PLAN) != PLAN_SHA or P.sha256(PREFLIGHT) != m["bindings"]["preflight"] or P.sha256(E.EVAL) != m["bindings"]["extended_eval"]:
        P.fail("bound evidence changed")
    _, data, _, _ = E.load()
    return m, data


def observe(out, target, ref_np):
    terms = {k: float(v.item()) for k, v in CL.per_scene_terms(out, target).items()}
    p = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float64)
    e = ciede2000(linear_srgb_to_lab(np.clip(p, 0, 1)), linear_srgb_to_lab(ref_np))
    m = P.dev_metrics(out.detach(), target.cpu())
    d = domain_stats(p, ref_np, e)
    row = {"display_combined": terms["combined"], "linear_l1": terms["linear_l1"], "display_lab_l1_scaled": terms["lab_l1_scaled"],
           "mean_dE00": float(e.mean()), "p95_dE00": float(np.percentile(e, 95)),
           "tone_band_signed_Lab": m["tone_band_signed_Lab"], "signed_dC_ref_C_gt_10": m["signed_dC_ref_C_gt_10"],
           "signed_hue_deg_ref_C_gt_20": m["signed_hue_deg_ref_C_gt_20"],
           "tails": d["tail_thresholds_linear"], "negative_target_positive": d["negative_channels_whose_target_exceeds_0.001_pixel_fraction"],
           "over_one_target_below_0999": d["over_one_channels_whose_target_is_below_0.999_pixel_fraction"],
           "min": d["min"], "max": d["max"]}
    for k in ("display_combined", "linear_l1", "mean_dE00", "p95_dE00"):
        if not math.isfinite(row[k]):
            P.fail(f"non-finite {k}")
    return row, p, e


def panel(name, imgs, heats):
    h = 420
    rs = lambda a: cv2.resize(a, (round(a.shape[1] * h / a.shape[0]), h), interpolation=cv2.INTER_AREA)
    def lab(a, t):
        a = rs(a).copy(); cv2.rectangle(a, (0, 0), (a.shape[1], 27), (0, 0, 0), -1)
        cv2.putText(a, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA); return a
    heat = lambda e: cv2.cvtColor(cv2.applyColorMap((255 * np.clip(e / 10, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    top = np.concatenate([lab(encode(v), t) for t, v in imgs], 1)
    bot = np.concatenate([lab(heat(v), t) for t, v in heats], 1)
    if bot.shape[1] < top.shape[1]:
        bot = np.concatenate([bot, np.zeros((h, top.shape[1] - bot.shape[1], 3), np.uint8)], 1)
    path = PANELS / f"{Path(name).stem}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate([top, bot], 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def cmd_run():
    if REPORT.exists() or PANELS.exists() or GRIDS.exists():
        P.fail("oracle outputs exist - not overwriting")
    self_test()
    m, data = load()
    dev = P.device()
    model, config = P.build(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loader = P.Pairs(data)
    ctx = json.loads(E.EVAL.read_text())["per_model"]
    PANELS.mkdir(); GRIDS.mkdir()
    res = {}
    for name in E.REVIEW_CASES:
        t0 = time.time()
        x, target, cond = loader.get(name)
        x, target, cond = x.to(dev), target.to(dev), cond.to(dev)
        ref_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
        with torch.no_grad():
            original = P.fwd(model, config, x, cond, dev)
            g0 = model.grid(resize_for_network(x, config["low_res"]), cond).detach().clone()
        grid = torch.nn.Parameter(g0.clone())
        opt = torch.optim.Adam([grid], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        trace, obs, imgs = [], {}, {}
        for step in range(STEPS + 1):
            out = oracle_forward(model, config, x, grid)
            if step == 0 and (out.detach() - original).abs().max().item() != 0.0:
                P.fail(f"{name}: warm start not bit-exact to R11")
            terms = CL.per_scene_terms(out, target)
            loss = terms["combined"].mean()
            if not torch.isfinite(loss):
                P.fail(f"{name}: non-finite loss at step {step}", code=4)
            opt.zero_grad()
            loss.backward()
            if grid.grad is None or not torch.isfinite(grid.grad).all():
                P.fail(f"{name}: non-finite grid gradient at step {step}", code=4)
            if any(p.grad is not None for p in model.parameters()):
                P.fail("frozen model received a gradient")
            gnorm = float(grid.grad.norm().item())
            trace.append({"step": step, "display_combined": float(loss.item()),
                          "linear_l1": float(terms["linear_l1"].mean().item()),
                          "display_lab_l1_scaled": float(terms["lab_l1_scaled"].mean().item()), "grid_grad_norm": gnorm})
            if step in OBS:
                with torch.no_grad():
                    row, p_np, e = observe(out.detach(), target, ref_np)
                diff = (grid.detach() - g0)
                row.update(grid_grad_norm=gnorm, coef_l2_from_R11=float(diff.norm().item()),
                           coef_max_abs_from_R11=float(diff.abs().max().item()),
                           coef_rel_l2_from_R11=float(diff.norm().item() / g0.norm().item()))
                obs[step] = row
                if step in (0, STEPS):
                    imgs[step] = (p_np, e)
            if step < STEPS:
                opt.step()
            if not torch.isfinite(grid).all():
                P.fail(f"{name}: non-finite grid after step {step}", code=4)
        for k, v in model.state_dict().items():
            if not torch.equal(v, params0[k]):
                P.fail(f"frozen model tensor {k} changed")
        gpath = GRIDS / f"{Path(name).stem}-step{STEPS}.npy"
        np.save(gpath, grid.detach().cpu().numpy())
        if abs(obs[0]["mean_dE00"] - ctx["r11"][name]["mean_dE00"]) > 1e-8:
            P.fail(f"{name}: step-0 dE does not reproduce frozen R11 evaluation")
        pth = panel(name, [("R11 (step 0)", imgs[0][0]), (f"Free grid step {STEPS}", imgs[STEPS][0]), ("Fuji Provia", ref_np)],
                    [("R11 dE 0-10", imgs[0][1]), (f"Step {STEPS} dE 0-10", imgs[STEPS][1])])
        res[name] = {"observations": {str(k): v for k, v in obs.items()}, "trace": trace,
                     "endpoint_grid_sha256": P.sha256(gpath), "panel": str(pth.relative_to(P.TRAIN.parent)),
                     "context_frozen_in_sample": {k: ctx[k][name]["mean_dE00"] for k in ("r11", "lowrate_display_candidate", "extended_candidate")},
                     "delta_mean_dE00_step200_minus_R11": obs[STEPS]["mean_dE00"] - obs[0]["mean_dE00"],
                     "delta_p95_dE00_step200_minus_R11": obs[STEPS]["p95_dE00"] - obs[0]["p95_dE00"],
                     "seconds": round(time.time() - t0, 1)}
        print(name, " ".join(f"s{k}:{v['mean_dE00']:.3f}/{v['display_combined']:.5f}" for k, v in obs.items()),
              f"{res[name]['seconds']}s", flush=True)
        del x, target, cond, grid, opt, out, loss
    P.check_source()
    d = [r["delta_mean_dE00_step200_minus_R11"] for r in res.values()]
    out = {"label": "TARGET-ASSISTED per-image free-grid diagnostic on 8 train scenes; not validation, not a predictor",
           "manifest_sha256": MANIFEST_SHA, "runner": m["runner"],
           "summary": {"mean_dE00_step0": float(np.mean([r["observations"]["0"]["mean_dE00"] for r in res.values()])),
                       "mean_dE00_step200": float(np.mean([r["observations"][str(STEPS)]["mean_dE00"] for r in res.values()])),
                       "mean_p95_step0": float(np.mean([r["observations"]["0"]["p95_dE00"] for r in res.values()])),
                       "mean_p95_step200": float(np.mean([r["observations"][str(STEPS)]["p95_dE00"] for r in res.values()])),
                       "scenes_improved_mean_dE00": int(sum(v < 0 for v in d)), "per_scene_delta_mean_dE00": dict(zip(res, d))},
           "interpretation": json.loads(PLAN.read_text())["interpretation"],
           "per_scene": res}
    REPORT.write_text(json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    {"manifest": cmd_manifest, "run": cmd_run}[sys.argv[1]]()
