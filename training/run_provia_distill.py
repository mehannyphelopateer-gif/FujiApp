"""Paired rendered-teacher pilot on F24 (Codex scope 056e38a; plan
provia-oracle-distillation-plan.json 70d02cff...). Session-grouped
ADAPTATION-OOF only: original R11 already trained on F24 (all capture DR400).

Teachers: exact per-image oracle protocol (original R11 grid leaf, frozen
model, 200 fresh-Adam steps at 1e-3, original display-combined loss), the
four existing F24 endpoint grids reused only by recorded hash; each grid is
rendered through check_provia_grid_oracle.oracle_forward (frozen refinement)
and cached as unclipped float32 linear RGB with fingerprints.

Students: per fold x arm (direct = Fuji Provia target, distill = rendered
teacher), independently from original R11; train grid.encoder + grid.head
(105,084), refinement frozen but differentiated through; fresh Adam 1e-5
(0.9, 0.999, 1e-8, wd 0), batch 1, 16 epochs, no schedule; per-epoch order
torch.randperm(len(sorted fit), Generator(42)) identical across arms; loss
color_loss.per_scene_terms combined. Fit-fold loaders refuse held-fold and
non-F24 names. Endpoints frozen (hashed + committed) before any held-fold
scoring against actual Fuji Provia.

  python3 training/run_provia_distill.py manifest | smoke | teachers | train | freeze | score
"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_loss as CL  # noqa: E402
import run_provia_grid_oracle as O  # noqa: E402
from audit_r11_display_domain import domain_stats  # noqa: E402
from check_provia_grid_oracle import oracle_forward  # noqa: E402
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402
from model import resize_for_network  # noqa: E402

P = O.P
PLAN = P.DIAG / "provia-oracle-distillation-plan.json"
PLAN_SHA = "70d02cfffec6c30550e6a47f543a0ada951cd7ecffa0ced9addbce802080b6ec"
MANIFEST = P.DIAG / "provia-distill-run-manifest.json"
MANIFEST_SHA = "41918a2f7342ac691adff9e60e013c32e6000b4c1d0233a8157b84ea6e5bda05"
TEACHERS = P.CAL / "provia-distill-teachers"
TEACHER_LOG = P.DIAG / "provia-distill-teachers.json"
FREEZE = P.DIAG / "provia-distill-freeze.json"
REPORT = P.DIAG / "provia-distill-report.json"
ARMS = ("direct", "distill")
LR, EPOCHS, SEED = 1e-5, 16, 42
T_LR, T_STEPS = 1e-3, 200
TRAINABLE_N, FROZEN_N = 105084, 10635
CODE = ["run_provia_distill.py", "prepare_provia_oracle_distillation_plan.py", "run_provia_grid_oracle.py",
        "check_provia_grid_oracle.py", "model.py", "color_loss.py", "color_metrics.py", "provia_color_pilot.py",
        "provia_extended.py", "audit_r11_display_domain.py"]


def plan():
    if P.sha256(PLAN) != PLAN_SHA:
        P.fail("distillation plan changed")
    return json.loads(PLAN.read_text())


def cmd_manifest():
    if MANIFEST.exists():
        P.fail("distill manifest already frozen")
    pl = plan()
    b = pl["bindings"]
    if b["source_R11"] != P.SOURCE_SHA or b["pilot_manifest"] != P.MANIFEST_SHA or b["oracle_run_manifest"] != O.MANIFEST_SHA \
            or b["oracle_report"] != P.sha256(O.REPORT):
        P.fail("plan bindings mismatch")
    for f, h in b["code"].items():
        if P.sha256(P.TRAIN / f) != h:
            P.fail(f"plan code {f} changed")
    for n, h in pl["teacher_generation"]["reuse_endpoint_grids_sha256"].items():
        if P.sha256(O.GRIDS / f"{Path(n).stem}-step200.npy") != h:
            P.fail(f"reusable grid {n} changed")
    O.load()
    ps = pl["paired_students"]
    if (ps["lr"], ps["epochs"], ps["batch_size"], ps["seed"]) != (LR, EPOCHS, 1, SEED):
        P.fail("runner constants differ from plan")
    out = {"status": "FROZEN before teacher generation or student fitting (Codex scope 056e38a)",
           "label": "Session-grouped ADAPTATION-OOF pilot on F24 (R11 trained on these); not validation, not selection",
           "bindings": {"plan": PLAN_SHA, **{k: v for k, v in b.items() if k != "code"},
                        "code": {f: P.sha256(P.TRAIN / f) for f in CODE if f != "run_provia_distill.py"},
                        "reused_grids": pl["teacher_generation"]["reuse_endpoint_grids_sha256"]},
           "teachers": {"method": pl["teacher_generation"]["method"], "lr": T_LR, "steps": T_STEPS,
                        "cache": "calibration-input/provia-distill-teachers/<shoot>.npy (float32 unclipped linear) + .json sidecar",
                        "new": pl["teacher_generation"]["new_grids"]},
           "students": ps, "folds": pl["folds"], "evaluation": pl["evaluation"],
           "runner_note": "run_provia_distill.py binds this manifest by pinned hash; committed version is the one in the commit adding this file"}
    MANIFEST.write_text(json.dumps(out, indent=1))
    print("distill manifest sha256", P.sha256(MANIFEST))


def load():
    if MANIFEST_SHA == "__UNPINNED__" or P.sha256(MANIFEST) != MANIFEST_SHA:
        P.fail("distill manifest missing/unpinned/changed")
    m = json.loads(MANIFEST.read_text())
    for f, h in m["bindings"]["code"].items():
        if P.sha256(P.TRAIN / f) != h:
            P.fail(f"bound code {f} changed")
    pl = plan()
    _, data = O.load()
    return m, pl, data


class FoldData:
    """Fit-fold-only access: Provia targets and teachers for this fold's fit names; refuses everything else."""

    def __init__(self, m, pairs, fold, allow_teachers=True):
        f = next(x for x in m["folds"] if x["fold"] == fold)
        self.fit, self.held = list(f["fit"]), set(f["adaptation_OOF"])
        self.pairs, self.allow_teachers = pairs, allow_teachers

    def get(self, name, arm):
        if name not in self.fit:
            P.fail(f"{name} is not a fit scene of this fold - refused")
        x, y, cond = self.pairs.get(name)
        if arm == "distill":
            y = load_teacher(self.pairs.by[name])
        return x, y, cond


def teacher_path(shoot):
    return TEACHERS / f"{shoot}.npy"


def load_teacher(scene):
    p, side = teacher_path(scene["shoot"]), teacher_path(scene["shoot"]).with_suffix(".json")
    if not side.exists():
        P.fail(f"teacher for {scene['name']} missing")
    meta = json.loads(side.read_text())
    log = json.loads(TEACHER_LOG.read_text())
    if meta != log["teachers"][scene["name"]] or P.sha256(p) != meta["render_npy_sha256"]:
        P.fail(f"teacher fingerprint mismatch for {scene['name']}")
    if meta["fjlrg_sha256"] != scene["fjlrg_sha256"] or meta["target_npy_sha256"] != scene["npy_sha256"]:
        P.fail(f"teacher source binding mismatch for {scene['name']}")
    t = np.load(p)
    if t.dtype != np.float32 or not np.isfinite(t).all():
        P.fail("teacher not finite float32")
    return torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1))).unsqueeze(0)


def frozen_r11(dev):
    m, c = P.build(dev)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, c


def oracle_grid(model, config, x, target, cond, dev):
    """Exact oracle protocol (run_provia_grid_oracle.cmd_run inner loop)."""
    with torch.no_grad():
        g0 = model.grid(resize_for_network(x, config["low_res"]), cond).detach().clone()
    grid = torch.nn.Parameter(g0.clone())
    opt = torch.optim.Adam([grid], lr=T_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    first = last = None
    for step in range(T_STEPS + 1):
        out = oracle_forward(model, config, x, grid)
        loss = CL.per_scene_terms(out, target)["combined"].mean()
        if not torch.isfinite(loss):
            P.fail("non-finite teacher loss", code=4)
        opt.zero_grad()
        loss.backward()
        if not torch.isfinite(grid.grad).all():
            P.fail("non-finite teacher gradient", code=4)
        first = float(loss.item()) if step == 0 else first
        last = float(loss.item())
        if step < T_STEPS:
            opt.step()
    return grid.detach(), first, last


def render(model, config, x, grid):
    with torch.no_grad():
        return oracle_forward(model, config, x, grid).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def cmd_teachers():
    m, pl, data = load()
    pairs, dev = P.Pairs(data), P.device()
    model, config = frozen_r11(dev)
    TEACHERS.mkdir(exist_ok=True)
    log = json.loads(TEACHER_LOG.read_text()) if TEACHER_LOG.exists() else {"manifest_sha256": MANIFEST_SHA, "teachers": {}}
    reuse = m["bindings"]["reused_grids"]
    for s in pl["scenes"]:
        n = s["name"]
        if n in log["teachers"] and teacher_path(s["shoot"]).exists():
            continue
        t0 = time.time()
        x, y, cond = pairs.get(n)
        x, y, cond = x.to(dev), y.to(dev), cond.to(dev)
        if n in reuse:
            gp = O.GRIDS / f"{Path(n).stem}-step200.npy"
            if P.sha256(gp) != reuse[n]:
                P.fail(f"reused grid {n} changed")
            grid = torch.from_numpy(np.load(gp)).to(dev)
            src, first, last = f"reused oracle endpoint {gp.name}", None, None
        else:
            grid, first, last = oracle_grid(model, config, x, y, cond, dev)
            src = "generated: exact oracle protocol"
        img = render(model, config, x, grid)
        if not np.isfinite(img).all():
            P.fail(f"{n}: non-finite teacher render")
        gpath = TEACHERS / f"{s['shoot']}-grid.npy"
        np.save(gpath, grid.cpu().numpy())
        tmp = TEACHERS / f"{s['shoot']}.npy.partial"
        with open(tmp, "wb") as f:
            np.save(f, img)
        tmp.rename(teacher_path(s["shoot"]))
        ref = y.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
        e = ciede2000(linear_srgb_to_lab(np.clip(img.astype(np.float64), 0, 1)), linear_srgb_to_lab(ref))
        meta = {"name": n, "source": src, "grid_sha256": P.sha256(gpath), "render_npy_sha256": P.sha256(teacher_path(s["shoot"])),
                "fjlrg_sha256": s["fjlrg_sha256"], "target_npy_sha256": s["npy_sha256"], "dtype": "float32", "clipped": False,
                "teacher_loss_first_last": [first, last], "teacher_vs_provia_mean_dE00": float(e.mean()),
                "teacher_vs_provia_p95_dE00": float(np.percentile(e, 95)), "seconds": round(time.time() - t0, 1)}
        teacher_path(s["shoot"]).with_suffix(".json").write_text(json.dumps(meta, indent=1))
        log["teachers"][n] = meta
        TEACHER_LOG.write_text(json.dumps(log, indent=1))
        print(n, src, f"teacher dE {meta['teacher_vs_provia_mean_dE00']:.3f}", f"{meta['seconds']}s", flush=True)
    P.check_source()
    print("teachers done", len(log["teachers"]))


def student(dev):
    m, c = P.build(dev)
    tr = 0
    for n, p in m.named_parameters():
        on = n.startswith("grid.")
        p.requires_grad_(on)
        tr += p.numel() if on else 0
    fr = sum(p.numel() for n, p in m.named_parameters() if n.startswith("refinement."))
    if tr != TRAINABLE_N or fr != FROZEN_N:
        P.fail(f"trainable/frozen counts {tr}/{fr}")
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    return m, c, opt


def fold_order(fit, epochs):
    gen = torch.Generator().manual_seed(SEED)
    names = sorted(fit)
    return [[names[i] for i in torch.randperm(len(names), generator=gen).tolist()] for _ in range(epochs)]


def run_epochs(mdl, c, opt, fd, arm, orders, start, stop, dev):
    hist = []
    for ep in range(start, stop):
        mdl.train()
        acc = []
        for n in orders[ep]:
            x, y, cond = fd.get(n, arm)
            loss = CL.per_scene_terms(P.fwd(mdl, c, x, cond, dev), y.to(dev))["combined"].mean()
            if not torch.isfinite(loss):
                P.fail(f"non-finite loss {arm} {n}", code=4)
            opt.zero_grad()
            loss.backward()
            for name, p in mdl.named_parameters():
                if p.grad is not None and (name.startswith("refinement.") or not torch.isfinite(p.grad).all()):
                    P.fail(f"bad gradient on {name}", code=4)
            opt.step()
            acc.append(float(loss.item()))
        hist.append({"epoch": ep + 1, "train_combined_vs_own_target": float(np.mean(acc))})
    return hist


def state_path(fold, arm, ep):
    return P.CKPT / f"provia-distill-f{fold}-{arm}-ep{ep}.pt"


def cmd_smoke():
    m, pl, data = load()
    pairs = P.Pairs(data)
    fd = FoldData(m, pairs, 0)
    held = sorted(fd.held)[0]
    import subprocess
    for bad in (held, "DSCF0515.RAF", "DSCF9999.RAF"):
        r = subprocess.run([sys.executable, "-c", f"import run_provia_distill as D; m,p,d=D.load(); "
                            f"D.FoldData(m, D.P.Pairs(d), 0).get({bad!r}, 'direct')"], cwd=P.TRAIN, capture_output=True, text=True)
        if r.returncode != 2 or "refused" not in r.stdout:
            P.fail(f"fold loader did not refuse {bad}")
    print("fit-fold loader refuses held-fold, K and unknown names")
    # teacher-cache fingerprint: render one reused grid into a scratch file, verify hash binding and corruption refusal
    dev = P.device()
    model, config = frozen_r11(dev)
    n = sorted(m["bindings"]["reused_grids"])[0]
    s = pairs.by[n]
    x, y, cond = pairs.get(n)
    grid = torch.from_numpy(np.load(O.GRIDS / f"{Path(n).stem}-step200.npy")).to(dev)
    a, b = render(model, config, x.to(dev), grid), render(model, config, x.to(dev), grid)
    if not np.array_equal(a, b) or a.dtype != np.float32:
        P.fail("teacher render not deterministic float32")
    print(f"teacher render deterministic float32; unclipped range [{a.min():.3f}, {a.max():.3f}]")
    for arm in ARMS:
        o1, o2 = fold_order(fd.fit, EPOCHS), fold_order(fd.fit, EPOCHS)
        if o1 != o2:
            P.fail("order not reproducible")
    print("paired orders identical (same generator, same sorted fit list)")
    from eval_cuban_step1_k import load_model
    ref, rc = load_model(P.SOURCE, dev)
    mdl, c, opt = student(dev)
    with torch.no_grad():
        d = (P.fwd(mdl.eval(), c, x, cond, dev) - P.fwd(ref, rc, x, cond, dev)).abs().max().item()
    if d != 0.0:
        P.fail("student warm start differs")
    ref0 = {k: v.detach().clone() for k, v in mdl.state_dict().items() if k.startswith("refinement.")}
    enc0 = {k: v.detach().clone() for k, v in mdl.state_dict().items() if k.startswith("grid.")}
    mdl.train()
    yy = y.to(dev)
    loss = CL.per_scene_terms(P.fwd(mdl, c, x, cond, dev), yy)["combined"].mean()
    opt.zero_grad(); loss.backward()
    if any(p.grad is not None for nm, p in mdl.named_parameters() if nm.startswith("refinement.")) or \
            not all(torch.isfinite(p.grad).all() for nm, p in mdl.named_parameters() if p.grad is not None):
        P.fail("masked/finite backward failed")
    opt.step()
    sd = mdl.state_dict()
    if any(not torch.equal(sd[k], v) for k, v in ref0.items()):
        P.fail("refinement changed")
    if all(torch.equal(sd[k], v) for k, v in enc0.items()):
        P.fail("grid predictor did not update")
    print(f"warm start exact; masked update: refinement unchanged, grid predictor updated; trainable {TRAINABLE_N}")
    # exact resume on CPU: 2 updates vs 1 + save/reload + 1
    cpu = torch.device("cpu")
    fit2 = sorted(fd.fit)[:2]

    def two(resume):
        torch.manual_seed(SEED)
        mm, cc, oo = student(cpu)
        mm.train()
        for i, nm in enumerate(fit2):
            if resume and i == 1:
                tmp = P.CKPT / "provia-distill-smoke.pt.partial"
                torch.save({"model": mm.state_dict(), "opt": oo.state_dict()}, tmp)
                st = torch.load(tmp, map_location="cpu", weights_only=False); tmp.unlink()
                mm, cc, oo = student(cpu); mm.load_state_dict(st["model"]); oo.load_state_dict(st["opt"]); mm.train()
            xx, ty, cd = fd.get(nm, "direct")
            l = CL.per_scene_terms(P.fwd(mm, cc, xx, cd, cpu), ty)["combined"].mean()
            oo.zero_grad(); l.backward(); oo.step()
        return mm.state_dict()
    A, B = two(False), two(True)
    diff = max((A[k] - B[k]).abs().max().item() for k in A)
    if diff != 0.0:
        P.fail(f"resume not exact ({diff})")
    print("exact resume (CPU) 0.0")
    P.check_source()
    print("smoke ok")


def cmd_train():
    m, pl, data = load()
    if not TEACHER_LOG.exists() or len(json.loads(TEACHER_LOG.read_text())["teachers"]) != 24:
        P.fail("teachers incomplete")
    if FREEZE.exists():
        P.fail("endpoints already frozen")
    pairs, dev = P.Pairs(data), P.device()
    for f in m["folds"]:
        fd = FoldData(m, pairs, f["fold"])
        orders = fold_order(fd.fit, EPOCHS)
        for arm in ARMS:
            if state_path(f["fold"], arm, EPOCHS).exists():
                continue
            torch.manual_seed(SEED)
            mdl, c, opt = student(dev)
            ref0 = {k: v.detach().clone() for k, v in mdl.state_dict().items() if k.startswith("refinement.")}
            hist, start = [], 0
            done = [e for e in range(1, EPOCHS + 1) if state_path(f["fold"], arm, e).exists()]
            if done:
                st = torch.load(state_path(f["fold"], arm, max(done)), map_location="cpu", weights_only=False)
                mdl.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); hist, start = st["history"], st["epoch"]
            t0 = time.time()
            for ep in range(start, EPOCHS):
                P.check_source()
                hist += run_epochs(mdl, c, opt, fd, arm, orders, ep, ep + 1, dev)
                st = {"fold": f["fold"], "arm": arm, "epoch": ep + 1, "model": mdl.state_dict(), "optimizer": opt.state_dict(),
                      "history": hist, "orders": orders, "manifest_sha256": MANIFEST_SHA}
                tmp = state_path(f["fold"], arm, ep + 1).with_suffix(".pt.partial")
                torch.save(st, tmp); tmp.rename(state_path(f["fold"], arm, ep + 1))
            sd = mdl.state_dict()
            if any(not torch.equal(sd[k], v) for k, v in ref0.items()):
                P.fail("refinement changed during training")
            print(f"fold {f['fold']} {arm}: epoch16 own-target loss {hist[-1]['train_combined_vs_own_target']:.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    P.check_source()
    print("students done")


def cmd_freeze():
    m, pl, data = load()
    if FREEZE.exists():
        P.fail("already frozen")
    out = {"status": "8 epoch-16 student endpoints FROZEN before any held-fold scoring", "manifest_sha256": MANIFEST_SHA,
           "teachers_sha256": P.sha256(TEACHER_LOG), "endpoints": {}}
    for f in m["folds"]:
        orders = None
        for arm in ARMS:
            p = state_path(f["fold"], arm, EPOCHS)
            st = torch.load(p, map_location="cpu", weights_only=False)
            if st["epoch"] != EPOCHS:
                P.fail("endpoint epoch mismatch")
            if orders is not None and st["orders"] != orders:
                P.fail("paired orders differ")
            orders = st["orders"]
            out["endpoints"][p.name] = P.sha256(p)
    FREEZE.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def scene_metrics(pred, ref_np):
    p = pred.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
    e = ciede2000(linear_srgb_to_lab(np.clip(p, 0, 1)), linear_srgb_to_lab(ref_np))
    mt = P.dev_metrics(pred, torch.from_numpy(np.ascontiguousarray(ref_np.transpose(2, 0, 1))).unsqueeze(0).float())
    d = domain_stats(p, ref_np, e)
    return {"mean_dE00": float(e.mean()), "p95_dE00": float(np.percentile(e, 95)),
            "tone_band_signed_Lab": mt["tone_band_signed_Lab"], "signed_dC_ref_C_gt_10": mt["signed_dC_ref_C_gt_10"],
            "signed_hue_deg_ref_C_gt_20": mt["signed_hue_deg_ref_C_gt_20"],
            "frac_below_0": d["tail_thresholds_linear"]["0.0"]["below_minus_threshold_pixel_fraction"],
            "frac_above_1": d["tail_thresholds_linear"]["0.0"]["above_one_plus_threshold_pixel_fraction"],
            "frac_below_minus_0.01": d["tail_thresholds_linear"]["0.01"]["below_minus_threshold_pixel_fraction"],
            "negative_target_positive": d["negative_channels_whose_target_exceeds_0.001_pixel_fraction"]}


def cmd_score():
    if REPORT.exists():
        P.fail("report exists")
    self_test()
    m, pl, data = load()
    fr = json.loads(FREEZE.read_text())
    for name, h in fr["endpoints"].items():
        if P.sha256(P.CKPT / name) != h:
            P.fail(f"{name} changed after freeze")
    pairs, dev = P.Pairs(data), P.device()
    r11, rc = frozen_r11(dev)
    teachers = json.loads(TEACHER_LOG.read_text())["teachers"]
    per = {}
    for f in m["folds"]:
        stu = {}
        for arm in ARMS:
            mm, cc = P.build(dev)
            mm.load_state_dict(torch.load(state_path(f["fold"], arm, EPOCHS), map_location="cpu", weights_only=False)["model"])
            stu[arm] = (mm.eval(), cc)
        for n in f["adaptation_OOF"]:
            x, y, cond = pairs.get(n)  # actual Fuji Provia target; never the teacher
            ref_np = y.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
            row = {"fold": f["fold"], "session": pairs.by[n]["session"]}
            with torch.no_grad():
                row["r11"] = scene_metrics(P.fwd(r11, rc, x, cond, dev), ref_np)
                for arm, (mm, cc) in stu.items():
                    row[arm] = scene_metrics(P.fwd(mm, cc, x, cond, dev), ref_np)
            for k in ("r11", *ARMS):
                if not math.isfinite(row[k]["mean_dE00"]):
                    P.fail("non-finite score")
            row["distill_minus_direct"] = row["distill"]["mean_dE00"] - row["direct"]["mean_dE00"]
            row["context_teacher_vs_provia_mean_dE00 (in-sample oracle; not an evaluation target)"] = teachers[n]["teacher_vs_provia_mean_dE00"]
            per[n] = row
            print(n, f"f{f['fold']} R11 {row['r11']['mean_dE00']:.3f} direct {row['direct']['mean_dE00']:.3f} "
                     f"distill {row['distill']['mean_dE00']:.3f}", flush=True)
    names = sorted(per)
    col = lambda k, f: [per[n][k][f] for n in names]
    agg = {k: {"mean_dE00": float(np.mean(col(k, "mean_dE00"))), "median_dE00": float(np.median(col(k, "mean_dE00"))),
               "mean_p95_dE00": float(np.mean(col(k, "p95_dE00"))),
               "frac_below_0": float(np.mean(col(k, "frac_below_0"))), "frac_above_1": float(np.mean(col(k, "frac_above_1"))),
               "negative_target_positive": float(np.mean(col(k, "negative_target_positive"))),
               "dC": float(np.mean([v for v in col(k, "signed_dC_ref_C_gt_10") if v is not None])),
               "hue_deg": float(np.mean([v for v in col(k, "signed_hue_deg_ref_C_gt_20") if v is not None])),
               "hue_scenes": sum(v is not None for v in col(k, "signed_hue_deg_ref_C_gt_20"))} for k in ("r11", *ARMS)}
    reg = {arm: sorted([n for n in names if per[n][arm]["mean_dE00"] > per[n]["r11"]["mean_dE00"]]) for arm in ARMS}
    sess = {}
    for n in names:
        sess.setdefault(per[n]["session"], []).append(n)
    sessions = {str(s): {"n": len(v), "fold": per[v[0]]["fold"],
                         **{k: float(np.mean([per[x][k]["mean_dE00"] for x in v])) for k in ("r11", *ARMS)}} for s, v in sorted(sess.items())}
    for s in sessions.values():
        s["distill_minus_direct"] = s["distill"] - s["direct"]
    folds = {str(f["fold"]): {k: float(np.mean([per[x][k]["mean_dE00"] for x in f["adaptation_OOF"]])) for k in ("r11", *ARMS)}
             for f in m["folds"]}
    scr = m["evaluation"]["pilot_screen_only"]
    screen = {"mean_diff": agg["distill"]["mean_dE00"] - agg["direct"]["mean_dE00"],
              "median_diff": agg["distill"]["median_dE00"] - agg["direct"]["median_dE00"],
              "p95_diff": agg["distill"]["mean_p95_dE00"] - agg["direct"]["mean_p95_dE00"],
              "regressions_vs_r11": {a: len(v) for a, v in reg.items()},
              "max_session_diff": max(s["distill_minus_direct"] for s in sessions.values())}
    screen["checks"] = {"mean": screen["mean_diff"] <= scr["mean_dE00_distill_minus_direct_max"],
                        "median": screen["median_diff"] < scr["median_dE00_distill_minus_direct_strictly_less_than"],
                        "p95": screen["p95_diff"] <= scr["mean_p95_distill_minus_direct_max"],
                        "regressions": len(reg["distill"]) <= len(reg["direct"]),
                        "sessions": screen["max_session_diff"] <= scr["max_session_mean_dE00_distill_minus_direct"]}
    screen["pilot_screen_pass"] = all(screen["checks"].values())
    hist = {}
    for f in m["folds"]:
        for arm in ARMS:
            st = torch.load(state_path(f["fold"], arm, EPOCHS), map_location="cpu", weights_only=False)
            hist[f"f{f['fold']}-{arm}"] = st["history"]
    out = {"label": "Session-grouped ADAPTATION-OOF on F24 (R11 previously trained on all F24; all DR400). Scored against actual Fuji Provia. Not validation; no selection.",
           "manifest_sha256": MANIFEST_SHA, "freeze_sha256": P.sha256(FREEZE), "aggregate": agg, "screen": screen,
           "regressions_vs_r11": reg, "sessions": sessions, "folds": folds, "training_history_own_target": hist, "per_scene": per}
    P.check_source()
    REPORT.write_text(json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps({"aggregate": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in agg.items()},
                      "screen": screen, "sessions": sessions}, indent=1))


if __name__ == "__main__":
    {"manifest": cmd_manifest, "smoke": cmd_smoke, "teachers": cmd_teachers, "train": cmd_train, "freeze": cmd_freeze,
     "score": cmd_score}[sys.argv[1]]()
