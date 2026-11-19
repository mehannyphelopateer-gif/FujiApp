"""Matched lower-LR full-pool Provia replay (Codex scope 2112e33; plan
provia-low-rate-plan.json 1bf989c5...). ONLY change vs the high-rate replay
(provia_replay.py, manifest bc06d9ad...): the block LR curve divided by 10
(1e-5 initial, cosine T_max 12 blocks, eta_min 1e-7, stop after block four).
Same 583 hash-bound pairs, recorded 1,868-update order, both losses, original
R11 init, fresh Adam (0.9, 0.999, 1e-8, wd 0), batch 1, no accumulation.

Old modules are imported and called with explicit arguments (LR list passed
as a parameter); no globals are patched and no old file is edited. All
scores are IN-SAMPLE fitting/retention. K/T8/monitor/finalHeldOut/dr100Eval:
no code path.

  python3 training/provia_low_rate.py manifest | smoke | eval0 | train control|candidate | report
"""
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provia_color_pilot as P  # noqa: E402
import provia_replay as R  # noqa: E402
from color_metrics import self_test  # noqa: E402

PLAN = P.DIAG / "provia-low-rate-plan.json"
PLAN_SHA = "1bf989c5959a97501d7a3eb97674ce3257961c2127b5380e5514d5fc75d7912b"
MANIFEST = P.DIAG / "provia-low-rate-run-manifest.json"
MANIFEST_SHA = "defb2f0614e83ada8c645ca8619c0037ebf2913c4667d9fa12133af43cfd907a"
REPORT = P.DIAG / "provia-low-rate-report.json"
ARMS, BLOCK, BLOCKS, TOTAL = P.ARMS, R.BLOCK, R.BLOCKS, R.TOTAL
LR0, ETA = 1e-5, 1e-7
EVIDENCE = ["provia-update-probe-report.json", "provia-fullres-gradient-check.json", "provia-replay-report.json",
            "provia-replay-run-manifest.json", "provia-low-rate-plan.json"]
CODE = ["provia_low_rate.py", "provia_replay.py", "provia_color_pilot.py", "prepare_provia_low_rate_plan.py",
        "color_loss.py", "color_metrics.py", "model.py", "fjlrg.py", "dataset.py", "target_tiff.py"]
HIGH = {a: P.RUNS / f"provia-replay-{a}-update{TOTAL}.json" for a in ARMS}
HIGH_R11 = P.RUNS / "provia-replay-r11-update0.json"


def plan():
    if P.sha256(PLAN) != PLAN_SHA:
        P.fail("low-rate plan changed")
    return json.loads(PLAN.read_text())


def sched_lrs():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=LR0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12, eta_min=ETA)
    out = []
    for _ in range(BLOCKS):
        out.append(opt.param_groups[0]["lr"])
        opt.step(); sch.step()
    return out


def cmd_manifest():
    if MANIFEST.exists():
        P.fail("low-rate manifest already frozen")
    pl = plan()
    rman, old, rpl = R.load()  # verifies replay manifest, pilot manifest, code/evidence/data bindings, source
    b = pl["bindings"]
    if b["original_replay_run_manifest"] != R.MANIFEST_SHA or b["original_replay_report"] != P.sha256(R.REPORT) \
            or b["source_checkpoint"] != P.SOURCE_SHA:
        P.fail("plan bindings mismatch")
    if pl["unchanged"]["order_sha256"] != rman["order_sha256"] or pl["unchanged"]["source_replay_plan"] != R.PLAN_SHA:
        P.fail("order/plan mismatch")
    lrs, s = pl["block_learning_rates"], sched_lrs()
    if max(abs(x - y) for x, y in zip(lrs, s)) > 1e-15:
        P.fail(f"plan LRs {lrs} != scheduler {s}")
    old_lrs = rpl["training"]["learning_rates_by_467_update_block"]
    if max(abs(x / 10 - y) for x, y in zip(old_lrs, lrs)) > 1e-15:
        P.fail("plan LRs are not the old curve / 10")
    out = {"status": "FROZEN before low-rate fitting (Codex scope 2112e33)",
           "label": "IN-SAMPLE fitting/retention diagnostic - not validation, not selection",
           "bindings": {"low_rate_plan": PLAN_SHA, "replay_manifest": R.MANIFEST_SHA, "replay_plan": R.PLAN_SHA,
                        "pilot_manifest": P.MANIFEST_SHA, "source_checkpoint": P.SOURCE_SHA,
                        "order_sha256": rman["order_sha256"],
                        "evidence": {f: P.sha256(P.DIAG / f) for f in EVIDENCE},
                        "code": {f: P.sha256(P.TRAIN / f) for f in CODE if f != "provia_low_rate.py"},
                        "high_rate_endpoints": {a: P.sha256(p) for a, p in HIGH.items()},
                        "high_rate_r11_update0": P.sha256(HIGH_R11)},
           "change": "LR curve / 10 only", "block_learning_rates": lrs, "scheduler_crosscheck": s,
           "unchanged": pl["unchanged"],
           "reporting": {"hue": "null with valid-scene count when no scene in a group has ref C*>20 pixels; any other non-finite fails"},
           "runner_note": "provia_low_rate.py binds this manifest by pinned hash; committed version is the one in the commit adding this file"}
    MANIFEST.write_text(json.dumps(out, indent=1))
    print("low-rate manifest sha256", P.sha256(MANIFEST))


def load():
    if MANIFEST_SHA == "__UNPINNED__" or P.sha256(MANIFEST) != MANIFEST_SHA:
        P.fail("low-rate manifest missing/unpinned/changed")
    man = json.loads(MANIFEST.read_text())
    bd = man["bindings"]
    for f, h in bd["code"].items():
        if P.sha256(P.TRAIN / f) != h:
            P.fail(f"bound code {f} changed")
    for f, h in bd["evidence"].items():
        if P.sha256(P.DIAG / f) != h:
            P.fail(f"bound evidence {f} changed")
    for a, p in HIGH.items():
        if P.sha256(p) != bd["high_rate_endpoints"][a]:
            P.fail("high-rate endpoint evidence changed")
    rman, old, rpl = R.load()
    plan()
    return man, old, rpl


def state_path(arm, b):
    return P.CKPT / f"provia-low-rate-{arm}-block{b}.pt"


def eval_path(name):
    return P.RUNS / f"provia-low-rate-{name}.json"


def cmd_smoke():
    man, old, rpl = load()
    pairs = P.Pairs(old)
    import subprocess
    for bad in (json.loads(P.PLAN.read_text())["excluded_names"][0], "DSCF9999.RAF"):
        r = subprocess.run([sys.executable, "-c", f"import provia_low_rate as L; m,o,p=L.load(); L.P.Pairs(o).get({bad!r})"],
                           cwd=P.TRAIN, capture_output=True, text=True)
        if r.returncode != 2 or "refused" not in r.stdout:
            P.fail(f"refusal failed for {bad}")
    order, lrs = rpl["ordering"]["updates"], man["block_learning_rates"]
    vc = Counter(order)
    assert len(order) == TOTAL and len(vc) == 583 and Counter(vc.values()) == {3: 464, 4: 119}
    print("refusals ok; order/coverage ok")
    cpu = torch.device("cpu")
    mA, c, oA = R.new_arm(cpu)
    R.run_updates("candidate", mA, c, oA, pairs, order, 0, 3, lrs, cpu)
    mB, c, oB = R.new_arm(cpu)
    R.run_updates("candidate", mB, c, oB, pairs, order, 0, 2, lrs, cpu)
    st = {"model": {k: v.clone() for k, v in mB.state_dict().items()}, "optimizer": oB.state_dict()}
    tmp = P.CKPT / "provia-low-rate-smoke-resume.pt.partial"
    torch.save(st, tmp)
    st = torch.load(tmp, map_location="cpu", weights_only=False)
    tmp.unlink()
    mC, c, oC = R.new_arm(cpu)
    mC.load_state_dict(st["model"]); oC.load_state_dict(st["optimizer"])
    R.run_updates("candidate", mC, c, oC, pairs, order, 2, 3, lrs, cpu)
    diff = max((a - b).abs().max().item() for a, b in zip(mA.state_dict().values(), mC.state_dict().values()))
    if diff != 0.0:
        P.fail(f"resume not exact ({diff})")
    if oA.param_groups[0]["lr"] != lrs[0]:
        P.fail("applied LR != block-1 LR")
    print("exact resume (CPU) 0.0; applied LR", oA.param_groups[0]["lr"])
    dev = P.device()
    from eval_cuban_step1_k import load_model
    ref, rc = load_model(P.SOURCE, dev)
    for arm in ARMS:
        mdl, c, opt = R.new_arm(dev)
        x, y, cond = pairs.get(order[0])
        with torch.no_grad():
            d = (P.fwd(mdl.eval(), c, x, cond, dev) - P.fwd(ref, rc, x, cond, dev)).abs().max().item()
        if d != 0.0:
            P.fail(f"{arm} warm start differs")
        mdl.train()
        t0 = time.time()
        R.run_updates(arm, mdl, c, opt, pairs, order, 0, 2, lrs, dev)
        print(f"{arm}: no-op warm start exact; 2 finite full-res updates at lr {opt.param_groups[0]['lr']:.1e} "
              f"in {time.time() - t0:.1f}s")
    P.check_source()
    print("smoke ok")


def cmd_eval0():
    man, old, rpl = load()
    p = eval_path("r11-update0")
    if p.exists():
        return print("exists")
    dev = P.device()
    mdl, c = P.build(dev)
    res = R.evaluate_all(mdl, c, old, P.Pairs(old), dev, "r11 update0 (low-rate run)")
    prev = json.loads(HIGH_R11.read_text())["per_scene"]
    dmax = max(abs(res[n][k] - prev[n][k]) for n in res for k in ("mean_dE00", "p95_dE00", "linear_l1_unclamped"))
    p.write_text(json.dumps({"label": "IN-SAMPLE (R11 trained on all 583)", "manifest_sha256": MANIFEST_SHA,
                             "max_abs_diff_vs_high_rate_update0": dmax, "per_scene": res}, indent=1))
    print("update-0 consistency vs high-rate replay: max abs diff", dmax)
    P.check_source()


def cmd_train(arm):
    if arm not in ARMS:
        P.fail("unknown arm")
    man, old, rpl = load()
    pairs, dev = P.Pairs(old), P.device()
    order, lrs = rpl["ordering"]["updates"], man["block_learning_rates"]
    torch.manual_seed(42)
    mdl, c, opt = R.new_arm(dev)
    hist, start = [], 0
    done = [b for b in range(1, BLOCKS + 1) if state_path(arm, b).exists()]
    if done:
        st = torch.load(state_path(arm, max(done)), map_location="cpu", weights_only=False)
        if st["manifest_sha256"] != MANIFEST_SHA or st["cursor"] != max(done) * BLOCK:
            P.fail("resume state mismatch")
        mdl.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        hist, start = st["history"], st["cursor"]
        print(f"[{arm}] resumed at update {start}", flush=True)
    mdl.train()
    for b in range(start // BLOCK, BLOCKS):
        P.check_source()
        t0 = time.time()
        acc = R.run_updates(arm, mdl, c, opt, pairs, order, b * BLOCK, (b + 1) * BLOCK, lrs, dev)
        for n, p in mdl.named_parameters():
            if not torch.isfinite(p).all():
                P.fail(f"[{arm}] non-finite weight {n}", code=4)
        secs = time.time() - t0
        row = {"block": b + 1, "lr": lrs[b], "applied_lr": opt.param_groups[0]["lr"],
               **{f"train_{k}": float(np.mean(v)) for k, v in acc.items()}, "seconds": round(secs, 1), **P.mem()}
        hist.append(row)
        print(f"[{arm}] block {b + 1}/{BLOCKS} lin {row['train_linear_l1']:.5f} lab/100 {row['train_lab_l1_scaled']:.5f} "
              f"comb {row['train_combined']:.5f} lr {lrs[b]:.3e} {row['seconds']}s", flush=True)
        st = {"arm": arm, "cursor": (b + 1) * BLOCK, "model": mdl.state_dict(), "optimizer": opt.state_dict(),
              "history": hist, "config": c, "manifest_sha256": MANIFEST_SHA, "order_sha256": man["bindings"]["order_sha256"]}
        tmp = state_path(arm, b + 1).with_suffix(".pt.partial")
        torch.save(st, tmp)
        tmp.rename(state_path(arm, b + 1))
    p = eval_path(f"{arm}-update{TOTAL}")
    if not p.exists():
        st = torch.load(state_path(arm, BLOCKS), map_location="cpu", weights_only=False)
        mdl.load_state_dict(st["model"])
        res = R.evaluate_all(mdl, c, old, pairs, dev, f"{arm} low-rate update{TOTAL}")
        p.write_text(json.dumps({"label": "IN-SAMPLE fitting/retention", "arm": arm, "update": TOTAL,
                                 "state_sha256": P.sha256(state_path(arm, BLOCKS)), "manifest_sha256": MANIFEST_SHA,
                                 "per_scene": res}, indent=1))
    (P.RUNS / f"provia-low-rate-{arm}.json").write_text(json.dumps(
        {"arm": arm, "manifest_sha256": MANIFEST_SHA, "history": hist,
         "state_sha256": {f"block{b}": P.sha256(state_path(arm, b)) for b in range(1, BLOCKS + 1)}}, indent=1))
    P.check_source()
    print(f"[{arm}] done")


def finite_mean(vals, what):
    for v in vals:
        if v is not None and not math.isfinite(v):
            P.fail(f"unexpected non-finite {what}")
    v = [x for x in vals if x is not None]
    return (float(np.mean(v)) if v else None), len(v)


def agg(d, sel, arm):
    rows = [d[n] for n in sel]
    m = lambda k: finite_mean([r[k] for r in rows], k)[0]
    hue, hue_n = finite_mean([r["signed_hue_deg_ref_C_gt_20"] for r in rows], "hue")
    dC, dC_n = finite_mean([r["signed_dC_ref_C_gt_10"] for r in rows], "dC")
    obj_key = "linear_l1" if arm == "control" else "combined"
    out = {"n": len(rows), "mean_dE00": m("mean_dE00"), "median_dE00": float(np.median([r["mean_dE00"] for r in rows])),
           "mean_p95_dE00": m("p95_dE00"), "mean_linear_l1_unclamped": m("linear_l1_unclamped"),
           "mean_lab_l1_scaled": finite_mean([r["loss_terms"]["lab_l1_scaled"] for r in rows], "lab")[0],
           "mean_combined": finite_mean([r["loss_terms"]["combined"] for r in rows], "combined")[0],
           "dC": dC, "dC_scenes": dC_n, "hue_deg": hue, "hue_scenes": hue_n,
           "frac_below_0": m("frac_any_channel_below_0"), "frac_above_1": m("frac_any_channel_above_1"),
           "tone_band_signed_Lab": {}}
    if arm in ARMS:
        out["own_objective_" + obj_key] = finite_mean([r["loss_terms"][obj_key] for r in rows], obj_key)[0]
    for b in P.BANDS:
        vals = [r["tone_band_signed_Lab"][b] for r in rows if r["tone_band_signed_Lab"][b] is not None]
        out["tone_band_signed_Lab"][b] = {"Lab": [finite_mean([v[i] for v in vals], "tone")[0] for i in range(3)] if vals else None,
                                          "scenes": len(vals)}
    return out


def cmd_report():
    if REPORT.exists():
        P.fail("report exists")
    self_test()
    man, old, rpl = load()
    for a in ARMS:
        lg = json.loads((P.RUNS / f"provia-low-rate-{a}.json").read_text())
        for b in range(1, BLOCKS + 1):
            if P.sha256(state_path(a, b)) != lg["state_sha256"][f"block{b}"]:
                P.fail(f"{a} block {b} changed")
    ev = {"r11": json.loads(eval_path("r11-update0").read_text())["per_scene"]}
    for a in ARMS:
        ev[f"low_{a}"] = json.loads(eval_path(f"{a}-update{TOTAL}").read_text())["per_scene"]
        ev[f"high_{a}"] = json.loads(HIGH[a].read_text())["per_scene"]
    names = sorted(ev["r11"])
    groups = {"all": names, "former_fit": [n for n in names if ev["r11"][n]["former_role"] == "fit"],
              "former_development": [n for n in names if ev["r11"][n]["former_role"] == "development"]}
    for key in ("session", "captureDynamicRange"):
        for n in names:
            groups.setdefault(f"{key}={ev['r11'][n][key]}", []).append(n)
    cols = ("r11", "low_control", "low_candidate", "high_control", "high_candidate")
    table = {g: {k: agg(ev[k], sel, k.split("_")[-1]) for k in cols} for g, sel in groups.items()}
    for g, t in table.items():
        t["diffs_mean_dE00"] = {
            "low_control_minus_r11": t["low_control"]["mean_dE00"] - t["r11"]["mean_dE00"],
            "low_candidate_minus_r11": t["low_candidate"]["mean_dE00"] - t["r11"]["mean_dE00"],
            "low_control_minus_high_control": t["low_control"]["mean_dE00"] - t["high_control"]["mean_dE00"],
            "low_candidate_minus_high_candidate": t["low_candidate"]["mean_dE00"] - t["high_candidate"]["mean_dE00"],
            "low_candidate_minus_low_control": t["low_candidate"]["mean_dE00"] - t["low_control"]["mean_dE00"]}
    per = {n: {"session": ev["r11"][n]["session"], "captureDynamicRange": ev["r11"][n]["captureDynamicRange"],
               "former_role": ev["r11"][n]["former_role"],
               **{k: {"mean_dE00": ev[k][n]["mean_dE00"], "p95_dE00": ev[k][n]["p95_dE00"]} for k in cols}} for n in names}

    def regress(k, base):
        v = sorted(((ev[k][n]["mean_dE00"] - ev[base][n]["mean_dE00"], n) for n in names
                    if ev[k][n]["mean_dE00"] > ev[base][n]["mean_dE00"]), reverse=True)
        return {"count": len(v), "worst": v[:8], "all": [n for _, n in v]}
    hist = {a: json.loads((P.RUNS / f"provia-low-rate-{a}.json").read_text())["history"] for a in ARMS}
    out = {"label": "IN-SAMPLE fitting/retention: R11 and all arms trained on all 583 scenes. Not validation, not selection.",
           "manifest_sha256": MANIFEST_SHA, "groups": table,
           "regressions": {"low_control_vs_r11": regress("low_control", "r11"), "low_candidate_vs_r11": regress("low_candidate", "r11"),
                           "low_control_vs_high_control": regress("low_control", "high_control"),
                           "low_candidate_vs_high_candidate": regress("low_candidate", "high_candidate"),
                           "low_candidate_vs_low_control": regress("low_candidate", "low_control")},
           "training_history_low": hist, "per_scene": per}
    P.check_source()
    REPORT.write_text(json.dumps(out, indent=1, allow_nan=False))
    brief = {g: {k: round(t[k]["mean_dE00"], 3) for k in cols} | {"n": t["r11"]["n"]} for g, t in table.items()}
    print(json.dumps({"mean_dE00": brief, "diffs_all": table["all"]["diffs_mean_dE00"],
                      "regression_counts": {k: v["count"] for k, v in out["regressions"].items()},
                      "worst": {k: v["worst"][:3] for k, v in out["regressions"].items()}}, indent=1))


if __name__ == "__main__":
    {"manifest": cmd_manifest, "smoke": cmd_smoke, "eval0": cmd_eval0, "report": cmd_report,
     "train": lambda: cmd_train(sys.argv[2])}[sys.argv[1]]()
