"""Provia colour-loss pilot (Codex scope 5053165): two matched, fixed 12-epoch
runs on separate read-only copies of frozen R11 seed 42.

  control   : mean |pred - target| (unclamped linear RGB)       - R11's loss
  candidate : 0.5 * (linear L1 + display-clamped D65 Lab L1 / 100)   (color_loss.py)

Everything else identical: unchanged R11 architecture/input contract (RGB +
capture-DR plane), ALL original parameters trainable, seed 42, batch 1, no
augmentation, fresh Adam (0.9, 0.999, 1e-8, wd 0), LR 1e-4, cosine T_max 12 /
eta_min 1e-6, 12 epochs, no early stopping / AMP / sweep. Per-epoch scene order
from one seeded generator (identical across arms, verified). Full
model/optimizer/scheduler/RNG/order state saved every epoch for exact resume.

Data: explicit allowlist from the frozen plan (467 fit / 116 development,
whole sessions 7/12/14 held out); every load re-hashes the .fjlrg and the
hash-bound Provia target cache produced by provia_color_prep.py. K, T8,
monitor, finalHeldOut, dr100Eval: no code path (refused by name).
Development evaluated at epochs 0, 4, 8, 12; epoch 12 is the predeclared
endpoint. Original R11 hash checked every epoch.

  python3 training/provia_color_pilot.py manifest
  python3 training/provia_color_pilot.py smoke
  python3 training/provia_color_pilot.py eval0
  python3 training/provia_color_pilot.py train control|candidate
  python3 training/provia_color_pilot.py report
"""
import hashlib
import json
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_loss import per_scene_terms  # noqa: E402
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402
from cuban_contract import contract_sha256  # noqa: E402
from cuban_contract_v2 import contract_v2_sha256  # noqa: E402
from dataset import _read_with_retry  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from model import HybridPredictor  # noqa: E402

TRAIN = Path(__file__).resolve().parent
CAL = TRAIN.parent / "calibration-input"
DIAG, CKPT, RUNS = TRAIN / "diagnostics", TRAIN / "checkpoints", TRAIN / "runs"
PLAN = DIAG / "provia-color-pilot-plan.json"
PLAN_SHA = "8dac1a02baa884bbc0aa1040e05f87c03fd746bd3fe3f581c8b9a6afc0355df1"
AUDIT = DIAG / "provia-color-data-audit.json"
MANIFEST = DIAG / "provia-color-run-manifest.json"
MANIFEST_SHA = "d6c04efd70df63aa38a9f2f08e0b291f858d5085a9670ffb7d3f1b3f4c6cf254"
REPORT = DIAG / "provia-color-pilot-report.json"
CACHE = CAL / "provia-color-target-cache"
SOURCE = CKPT / "r11-drcond-seed42-best.pt"
SOURCE_SHA = "1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c"
V1, V2 = "c3f840ef9fad9d48de83b4edf183f2434523f352f358305b91eb7764d22d0067", "9824b0799737960363c95bb028332fdd0e5118e748fc0139375f1d0e230099a0"
ARMS = ("control", "candidate")
SEED, LR, EPOCHS, T_MAX, ETA_MIN = 42, 1e-4, 12, 12, 1e-6
EVAL_EPOCHS = (0, 4, 8, 12)
BANDS = {"shadow": (None, 25), "mid": (25, 75), "high": (75, None)}
BOUND = ["color_loss.py", "color_metrics.py", "check_color_loss.py", "provia_color_prep.py", "provia_color_pilot.py",
         "model.py", "fjlrg.py", "target_tiff.py", "dataset.py", "cuban_contract.py", "cuban_contract_v2.py"]
EVIDENCE = ["provia-color-loss-checks.json", "provia-color-loss-parity-checks.json", "provia-color-pilot-plan.json",
            "provia-color-data-audit.json"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def fail(msg, code=2):
    print(f"INTEGRITY FAILURE: {msg}", flush=True)
    sys.exit(code)


def check_source():
    if sha256(SOURCE) != SOURCE_SHA:
        fail("original R11 seed 42 checkpoint hash changed")


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def mem():
    out = {"peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 1)}
    if torch.backends.mps.is_available():
        out["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)
        out["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
    return out


# ------------------------------------------------------------------ manifest
def cmd_manifest():
    if MANIFEST.exists():
        fail("run manifest already frozen")
    if sha256(PLAN) != PLAN_SHA:
        fail("plan changed")
    audit = json.loads(AUDIT.read_text())
    if audit["status"] != "PASS" or audit["scenes"] != 583 or audit["plan_sha256"] != PLAN_SHA:
        fail("data audit did not pass - not training")
    if contract_sha256() != V1 or contract_v2_sha256() != V2:
        fail("contracts changed")
    check_source()
    plan = json.loads(PLAN.read_text())
    scenes = []
    for m in audit["per_scene"]:
        npy = CACHE / f"{m['shoot']}.npy"
        if sha256(npy) != m["npy_sha256"]:
            fail(f"{m['name']}: cache hash != audit")
        scenes.append({k: m[k] for k in ("name", "shoot", "role", "session", "captureDynamicRange", "fjlrg_sha256",
                                         "tiff_sha256", "npy_sha256", "fjlrg_wh")})
    import cv2, PIL, tifffile
    out = {"status": "FROZEN before any pilot fitting (Codex scope 5053165)",
           "bindings": {"plan": PLAN_SHA, "contract_v1": V1, "contract_v2": V2, "source_checkpoint": SOURCE_SHA,
                        "code": {f: sha256(TRAIN / f) for f in BOUND if f != "provia_color_pilot.py"},
                        "evidence": {f: sha256(DIAG / f) for f in EVIDENCE}},
           "runner_note": "provia_color_pilot.py binds this manifest by pinned hash (self-hash would be circular); "
                          "its committed version is the one in the commit that adds this manifest",
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                           "cv2": cv2.__version__, "tifffile": tifffile.__version__, "PIL": PIL.__version__,
                           "platform": platform.platform(), "mps": torch.backends.mps.is_available()},
           "arms": {"control": "mean |pred - target| unclamped linear RGB",
                    "candidate": "color_loss.per_scene_terms combined = 0.5*(linear L1 + Lab L1/100)"},
           "training": {"seed": SEED, "batch": 1, "epochs": EPOCHS, "lr": LR, "optimizer": "Adam (0.9,0.999) eps 1e-8 wd 0, fresh",
                        "schedule": f"CosineAnnealingLR T_max {T_MAX} eta_min {ETA_MIN}, stepped per epoch",
                        "trainable": "all original R11 parameters", "order": "torch.randperm with one Generator(seed 42), per epoch",
                        "amp": None, "augmentation": None, "early_stopping": None},
           "evaluation": {"epochs": list(EVAL_EPOCHS), "endpoint": 12, "set": "development (sessions 7, 12, 14)",
                          "screen": plan["screening"]},
           "limitations": plan["DR100_limitation"],
           "counts": audit["role_counts"], "scenes": scenes}
    MANIFEST.write_text(json.dumps(out, indent=1))
    print("run manifest sha256", sha256(MANIFEST))


def load_manifest():
    if MANIFEST_SHA == "__UNPINNED__" or sha256(MANIFEST) != MANIFEST_SHA:
        fail("run manifest missing/unpinned/changed")
    m = json.loads(MANIFEST.read_text())
    for f, h in m["bindings"]["code"].items():
        if sha256(TRAIN / f) != h:
            fail(f"bound code {f} changed")
    for f, h in m["bindings"]["evidence"].items():
        if sha256(DIAG / f) != h:
            fail(f"bound evidence {f} changed")
    if contract_sha256() != V1 or contract_v2_sha256() != V2 or sha256(PLAN) != PLAN_SHA:
        fail("contracts/plan changed")
    check_source()
    return m


class Pairs:
    """Allowlist-only streaming loader: one pair per call, re-hashed every time."""

    def __init__(self, m):
        self.by = {s["name"]: s for s in m["scenes"]}

    def get(self, name):
        s = self.by.get(name)
        if s is None:
            fail(f"{name} is not in the pilot allowlist - refused")
        fj, npy = CAL / s["shoot"] / "browser-phase3-linear.fjlrg", CACHE / f"{s['shoot']}.npy"
        if _read_with_retry(sha256, fj) != s["fjlrg_sha256"]:
            fail(f"{name}: .fjlrg hash mismatch")
        if _read_with_retry(sha256, npy) != s["npy_sha256"]:
            fail(f"{name}: target cache hash mismatch")
        x, w, h = _read_with_retry(load_fjlrg_linear, fj)
        y = _read_with_retry(np.load, npy)
        if y.shape != x.shape:
            fail(f"{name}: shape mismatch")
        cond = float(np.log2(s["captureDynamicRange"] / 100.0) / 2.0)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).unsqueeze(0)
        return t(x), t(y), torch.tensor([cond])


def build(dev):
    ck = torch.load(SOURCE, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = HybridPredictor(grid_spatial=c["grid_spatial"], grid_luma=c["grid_luma"], low_res=c["low_res"], base_ch=c["base_ch"],
                        refinement_base_ch=c["refinement_base_ch"], refinement_max_delta=c["refinement_max_delta"],
                        condition_channels=1)
    m.load_state_dict(ck["state_dict"])
    for p in m.parameters():
        p.requires_grad_(True)
    return m.to(dev), c


def fwd(m, c, x, cond, dev):
    out, _ = m.predict_and_apply(x.to(dev), c["low_res"], c.get("luma_mode", "rec709"), condition=cond.to(dev))
    return out


def loss_terms(pred, y, arm="candidate"):
    """candidate: full graph. control: linear L1 with graph; Lab term logged without a graph."""
    if arm == "control":
        lin = (pred - y).abs().mean()
        with torch.no_grad():
            t = per_scene_terms(pred.detach(), y)
        return {"linear_l1": lin, "lab_l1_scaled": t["lab_l1_scaled"].mean(), "combined": t["combined"].mean()}
    t = per_scene_terms(pred, y)
    return {k: v.mean() for k, v in t.items()}


def arm_loss(arm, terms):
    return terms["linear_l1"] if arm == "control" else terms["combined"]


# ------------------------------------------------------------------ smoke
def cmd_smoke():
    m = load_manifest()
    pairs = Pairs(m)
    import subprocess
    plan = json.loads(PLAN.read_text())
    for bad in (plan["excluded_names"][0], plan["excluded_names"][-1], "DSCF9999.RAF"):
        r = subprocess.run([sys.executable, "-c", f"import provia_color_pilot as P; P.Pairs(P.load_manifest()).get({bad!r})"],
                           cwd=TRAIN, capture_output=True, text=True)
        if r.returncode != 2 or "refused" not in r.stdout:
            fail(f"allowlist did not refuse {bad}: {r.stdout} {r.stderr}")
    bad = json.loads(json.dumps(m))
    bad["scenes"][0]["npy_sha256"] = "0" * 64
    pb = Pairs(bad)
    refused = False
    try:
        pb.get(bad["scenes"][0]["name"])
    except SystemExit as e:
        refused = e.code == 2
    if not refused:
        fail("changed cache not refused")
    print("refusals ok (K, T8, unknown, changed cache)")
    dev = device()
    from eval_cuban_step1_k import load_model
    ref, rc = load_model(SOURCE, dev)
    fit = [s["name"] for s in m["scenes"] if s["role"] == "fit"]
    report = {}
    for arm in ARMS:
        mdl, c = build(dev)
        x, y, cond = pairs.get(fit[0])
        with torch.no_grad():
            d = (fwd(mdl.eval(), c, x, cond, dev) - fwd(ref, rc, x, cond, dev)).abs().max().item()
        if d != 0.0:
            fail(f"{arm}: warm start differs from frozen R11 ({d})")
        opt = torch.optim.Adam(mdl.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        mdl.train()
        ts = []
        for n in fit[:4]:
            x, y, cond = pairs.get(n)
            t0 = time.time()
            terms = loss_terms(fwd(mdl, c, x, cond, dev), y.to(dev), arm)
            loss = arm_loss(arm, terms)
            opt.zero_grad()
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in mdl.parameters() if p.grad is not None):
                fail(f"{arm}: non-finite loss/gradient in smoke")
            opt.step()
            if dev.type == "mps":
                torch.mps.synchronize()
            ts.append(time.time() - t0)
        report[arm] = {"warm_start_max_abs_diff": d, "step_seconds": [round(v, 2) for v in ts], **mem()}
    check_source()
    print(json.dumps(report, indent=1))
    print("smoke ok: no-op warm start exact for both arms; finite full-resolution forward/backward; source hash unchanged")


# ------------------------------------------------------------------ evaluation
def dev_metrics(pred, y):
    p = pred.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
    r = y.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
    lin_l1 = float(np.abs(p - r).mean())
    A, R = linear_srgb_to_lab(np.clip(p, 0, 1)), linear_srgb_to_lab(r)
    e = ciede2000(A, R)
    d = A - R
    L = R[..., 0]
    bands = {b: ((L >= lo) if lo is not None else True) & ((L < hi) if hi is not None else True) for b, (lo, hi) in BANDS.items()}
    Cr, Ca = np.hypot(R[..., 1], R[..., 2]), np.hypot(A[..., 1], A[..., 2])
    dh = (np.degrees(np.arctan2(A[..., 2], A[..., 1]) - np.arctan2(R[..., 2], R[..., 1])) + 180) % 360 - 180
    return {"linear_l1_unclamped": lin_l1, "mean_dE00": float(e.mean()), "p95_dE00": float(np.percentile(e, 95)),
            "tone_band_signed_Lab": {b: ([float(v) for v in d[mk].mean(0)] if mk.any() else None) for b, mk in bands.items()},
            "signed_dC_ref_C_gt_10": float((Ca - Cr)[Cr > 10].mean()) if (Cr > 10).any() else None,
            "signed_hue_deg_ref_C_gt_20": float(dh[Cr > 20].mean()) if (Cr > 20).any() else None,
            "frac_any_channel_below_0": float((p < 0).any(-1).mean()), "frac_any_channel_above_1": float((p > 1).any(-1).mean())}


def evaluate(mdl, c, m, pairs, dev, tag):
    mdl.eval()
    out = {}
    names = [s["name"] for s in m["scenes"] if s["role"] == "development"]
    with torch.no_grad():
        for n in names:
            x, y, cond = pairs.get(n)
            pred = fwd(mdl, c, x, cond, dev)
            row = dev_metrics(pred, y)
            t = per_scene_terms(pred, y.to(dev))
            row["loss_terms"] = {k: float(v.item()) for k, v in t.items()}
            s = pairs.by[n]
            row.update(session=s["session"], captureDynamicRange=s["captureDynamicRange"])
            out[n] = row
    print(f"[{tag}] dev mean dE00 {np.mean([r['mean_dE00'] for r in out.values()]):.4f}", flush=True)
    return out


def eval_path(arm, ep):
    return RUNS / f"provia-color-{arm}-dev-ep{ep}.json"


def cmd_eval0():
    m = load_manifest()
    p = RUNS / "provia-color-r11-dev-ep0.json"
    if p.exists():
        print("epoch-0 evaluation exists")
        return
    dev = device()
    mdl, c = build(dev)
    res = evaluate(mdl, c, m, Pairs(m), dev, "r11 ep0")
    p.write_text(json.dumps({"model": "frozen R11 seed 42 (= epoch 0 of both arms)", "manifest_sha256": MANIFEST_SHA,
                             "source_sha256": SOURCE_SHA, "per_scene": res}, indent=1))
    check_source()


# ------------------------------------------------------------------ training
def state_path(arm, ep):
    return CKPT / f"provia-color-{arm}-ep{ep}.pt"


def cmd_train(arm):
    if arm not in ARMS:
        fail(f"unknown arm {arm}")
    m = load_manifest()
    pairs = Pairs(m)
    dev = device()
    fit = [s["name"] for s in m["scenes"] if s["role"] == "fit"]
    mdl, c = build(dev)
    opt = torch.optim.Adam(mdl.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=ETA_MIN)
    gen = torch.Generator().manual_seed(SEED)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    start, history, orders = 1, [], []
    done = [e for e in range(1, EPOCHS + 1) if state_path(arm, e).exists()]
    if done:
        st = torch.load(state_path(arm, max(done)), map_location="cpu", weights_only=False)
        if st["manifest_sha256"] != MANIFEST_SHA:
            fail("resume state bound to another manifest")
        mdl.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); sch.load_state_dict(st["scheduler"])
        gen.set_state(st["generator"]); torch.set_rng_state(st["torch_rng"]); np.random.set_state(st["numpy_rng"])
        random.setstate(st["python_rng"])
        history, orders, start = st["history"], st["orders"], st["epoch"] + 1
        print(f"[{arm}] resumed after epoch {st['epoch']}", flush=True)
    for ep in range(start, EPOCHS + 1):
        check_source()
        t0 = time.time()
        mdl.train()
        order = [fit[i] for i in torch.randperm(len(fit), generator=gen).tolist()]
        orders.append(order)
        acc = {"linear_l1": [], "lab_l1_scaled": [], "combined": []}
        lr_used = opt.param_groups[0]["lr"]
        for n in order:
            x, y, cond = pairs.get(n)
            terms = loss_terms(fwd(mdl, c, x, cond, dev), y.to(dev), arm)
            loss = arm_loss(arm, terms)
            if not torch.isfinite(loss):
                fail(f"[{arm}] non-finite loss at epoch {ep} scene {n}", code=4)
            opt.zero_grad()
            loss.backward()
            for name, p in mdl.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    fail(f"[{arm}] non-finite gradient {name} at epoch {ep} scene {n}", code=4)
            opt.step()
            for k, v in terms.items():
                acc[k].append(float(v.item()))
        sch.step()
        for name, p in mdl.named_parameters():
            if not torch.isfinite(p).all():
                fail(f"[{arm}] non-finite weight {name} after epoch {ep}", code=4)
        secs = time.time() - t0
        row = {"epoch": ep, "lr": lr_used, **{f"train_{k}": float(np.mean(v)) for k, v in acc.items()},
               "seconds": round(secs, 1), "seconds_per_scene": round(secs / len(fit), 3), **mem()}
        history.append(row)
        print(f"[{arm}] epoch {ep}/{EPOCHS} lin {row['train_linear_l1']:.5f} lab/100 {row['train_lab_l1_scaled']:.5f} "
              f"lr {lr_used:.2e} {row['seconds']}s {mem()}", flush=True)
        st = {"arm": arm, "epoch": ep, "model": mdl.state_dict(), "optimizer": opt.state_dict(), "scheduler": sch.state_dict(),
              "generator": gen.get_state(), "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
              "python_rng": random.getstate(), "history": history, "orders": orders, "config": c,
              "manifest_sha256": MANIFEST_SHA, "source_sha256": SOURCE_SHA}
        tmp = state_path(arm, ep).with_suffix(".pt.partial")
        torch.save(st, tmp)
        tmp.rename(state_path(arm, ep))
    for ep in EVAL_EPOCHS[1:]:
        p = eval_path(arm, ep)
        if p.exists():
            continue
        st = torch.load(state_path(arm, ep), map_location="cpu", weights_only=False)
        mdl.load_state_dict(st["model"])
        res = evaluate(mdl, c, m, pairs, dev, f"{arm} ep{ep}")
        p.write_text(json.dumps({"arm": arm, "epoch": ep, "state_sha256": sha256(state_path(arm, ep)),
                                 "manifest_sha256": MANIFEST_SHA, "per_scene": res}, indent=1))
    final = torch.load(state_path(arm, EPOCHS), map_location="cpu", weights_only=False)
    (RUNS / f"provia-color-{arm}.json").write_text(json.dumps(
        {"arm": arm, "manifest_sha256": MANIFEST_SHA, "history": final["history"], "orders": final["orders"],
         "state_sha256": {f"ep{e}": sha256(state_path(arm, e)) for e in range(1, EPOCHS + 1)}}, indent=1))
    check_source()
    print(f"[{arm}] done")


# ------------------------------------------------------------------ report
def cmd_report():
    if REPORT.exists():
        fail("report exists - not overwriting")
    self_test()
    m = load_manifest()
    logs = {a: json.loads((RUNS / f"provia-color-{a}.json").read_text()) for a in ARMS}
    if logs["control"]["orders"] != logs["candidate"]["orders"]:
        fail("epoch scene orders differ between arms")
    for a in ARMS:
        for e in range(1, EPOCHS + 1):
            if sha256(state_path(a, e)) != logs[a]["state_sha256"][f"ep{e}"]:
                fail(f"{a} ep{e} state changed")
    ev = {"r11": json.loads((RUNS / "provia-color-r11-dev-ep0.json").read_text())["per_scene"]}
    curves = {}
    for a in ARMS:
        curves[a] = {}
        for e in EVAL_EPOCHS:
            d = ev["r11"] if e == 0 else json.loads(eval_path(a, e).read_text())["per_scene"]
            curves[a][e] = {"mean_dE00": float(np.mean([r["mean_dE00"] for r in d.values()])),
                            "mean_p95_dE00": float(np.mean([r["p95_dE00"] for r in d.values()])),
                            "mean_linear_l1": float(np.mean([r["linear_l1_unclamped"] for r in d.values()]))}
            if e == EPOCHS:
                ev[a] = d
    names = sorted(ev["r11"])
    def agg(d):
        rows = [d[n] for n in names]
        return {"mean_dE00": float(np.mean([r["mean_dE00"] for r in rows])), "median_dE00": float(np.median([r["mean_dE00"] for r in rows])),
                "mean_p95_dE00": float(np.mean([r["p95_dE00"] for r in rows])),
                "mean_linear_l1_unclamped": float(np.mean([r["linear_l1_unclamped"] for r in rows])),
                "mean_lab_l1_scaled": float(np.mean([r["loss_terms"]["lab_l1_scaled"] for r in rows])),
                "cross_scene_tone_band_signed_Lab": {b: [float(np.mean([r["tone_band_signed_Lab"][b][i] for r in rows
                                                                         if r["tone_band_signed_Lab"][b]])) for i in range(3)] for b in BANDS},
                "cross_scene_dC": float(np.mean([r["signed_dC_ref_C_gt_10"] for r in rows if r["signed_dC_ref_C_gt_10"] is not None])),
                "cross_scene_hue_deg": float(np.mean([r["signed_hue_deg_ref_C_gt_20"] for r in rows if r["signed_hue_deg_ref_C_gt_20"] is not None])),
                "mean_frac_below_0": float(np.mean([r["frac_any_channel_below_0"] for r in rows])),
                "mean_frac_above_1": float(np.mean([r["frac_any_channel_above_1"] for r in rows]))}
    A = {k: agg(ev[k]) for k in ("r11", *ARMS)}
    per = {}
    for n in names:
        r = {k: {"mean_dE00": ev[k][n]["mean_dE00"], "p95_dE00": ev[k][n]["p95_dE00"]} for k in ("r11", *ARMS)}
        r.update(session=ev["r11"][n]["session"], captureDynamicRange=ev["r11"][n]["captureDynamicRange"],
                 candidate_minus_control=ev["candidate"][n]["mean_dE00"] - ev["control"][n]["mean_dE00"],
                 candidate_minus_r11=ev["candidate"][n]["mean_dE00"] - ev["r11"][n]["mean_dE00"],
                 control_minus_r11=ev["control"][n]["mean_dE00"] - ev["r11"][n]["mean_dE00"])
        per[n] = r
    def groups(key):
        g = {}
        for n in names:
            g.setdefault(str(per[n][key]), []).append(n)
        return {k: {"n": len(v), **{a: float(np.mean([ev[a][s]["mean_dE00"] for s in v])) for a in ("r11", *ARMS)},
                    "candidate_minus_control": float(np.mean([per[s]["candidate_minus_control"] for s in v]))} for k, v in g.items()}
    screen = {"candidate_mean_below_control": A["candidate"]["mean_dE00"] < A["control"]["mean_dE00"],
              "candidate_mean_below_r11": A["candidate"]["mean_dE00"] < A["r11"]["mean_dE00"],
              "candidate_p95_not_worse_than_control": A["candidate"]["mean_p95_dE00"] <= A["control"]["mean_p95_dE00"],
              "candidate_p95_not_worse_than_r11": A["candidate"]["mean_p95_dE00"] <= A["r11"]["mean_p95_dE00"]}
    screen["passes_screen_for_further_consideration"] = all(screen.values())
    def regress(key):
        v = sorted(((per[n][key], n) for n in names if per[n][key] > 0), reverse=True)
        return {"count": len(v), "worst": v[:5], "all": [n for _, n in v]}
    out = {"scope": "Provia-only development pilot; adaptation (R11 previously trained on all these Provia scenes); no CN; "
                    "not accuracy acceptance or selection",
           "limitations": m["limitations"], "manifest_sha256": MANIFEST_SHA,
           "primary_endpoint_epoch12": {"candidate_minus_control_mean_dE00": A["candidate"]["mean_dE00"] - A["control"]["mean_dE00"],
                                        "candidate_minus_r11_mean_dE00": A["candidate"]["mean_dE00"] - A["r11"]["mean_dE00"],
                                        "control_minus_r11_mean_dE00": A["control"]["mean_dE00"] - A["r11"]["mean_dE00"]},
           "screen": screen, "aggregates_epoch12": A, "curves_dev": curves,
           "regressions": {"candidate_vs_r11": regress("candidate_minus_r11"), "control_vs_r11": regress("control_minus_r11"),
                           "candidate_vs_control": regress("candidate_minus_control")},
           "groups": {"session": groups("session"), "captureDynamicRange": groups("captureDynamicRange")},
           "training_history": {a: logs[a]["history"] for a in ARMS}, "per_scene": per}
    check_source()
    REPORT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("primary_endpoint_epoch12", "screen", "curves_dev", "groups")}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "all"} for k, v in out["regressions"].items()}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"manifest": cmd_manifest, "smoke": cmd_smoke, "eval0": cmd_eval0, "report": cmd_report,
     "train": lambda: cmd_train(sys.argv[2])}[cmd]()
