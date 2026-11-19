"""Bounded full-pool Provia replay diagnostic (Codex scope eba1a94; plan
provia-replay-plan.json c2353843...). ALL 583 eligible canonical train scenes
are training data here; every score is IN-SAMPLE training/retention, never
validation.

Reuses, unchanged and hash-checked, the pilot's verified 583-pair dataset,
allowlist loader, model build, losses and metrics (provia_color_pilot.py under
run manifest d6c04efd...). Two arms (control linear L1 / candidate
pixel-colour) from separate copies of original R11 seed 42, all parameters
trainable, fresh Adam (0.9, 0.999, 1e-8, wd 0), batch 1, no AMP/augmentation.
Exactly 1,868 updates per arm in the plan's recorded order (identical for both
arms), four 467-update blocks at the plan's LRs (first four blocks of cosine
T_max 12 / eta_min 1e-6), no rescaling, no extension. State saved at each
block boundary (exact resume cursor). Evaluation at update 0 and 1,868 only,
streaming all 583 pairs. K/T8/monitor/finalHeldOut/dr100Eval: no code path.

  python3 training/provia_replay.py manifest | smoke | eval0 | train control|candidate | report
"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch import (grid_sampler_3d_backward)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provia_color_pilot as P  # noqa: E402
from color_metrics import self_test  # noqa: E402
from prepare_provia_replay_plan import uniform_order  # noqa: E402

PLAN = P.DIAG / "provia-replay-plan.json"
PLAN_SHA = "c23538434022320fff73d74388e960410ff668d15eee1f79631783d2bbad111a"
MANIFEST = P.DIAG / "provia-replay-run-manifest.json"
MANIFEST_SHA = "bc06d9adefa610d0d98d658609a84df72df0f4311dfd5164f13582c1ef082910"
REPORT = P.DIAG / "provia-replay-report.json"
ARMS = P.ARMS
BLOCK, BLOCKS, TOTAL = 467, 4, 1868
CODE = ["provia_color_pilot.py", "provia_replay.py", "prepare_provia_replay_plan.py", "color_loss.py", "color_metrics.py",
        "model.py", "fjlrg.py", "dataset.py", "target_tiff.py"]
PRIOR_EP4 = {a: P.RUNS / f"provia-color-{a}-dev-ep4.json" for a in ARMS}


def plan():
    if P.sha256(PLAN) != PLAN_SHA:
        P.fail("replay plan changed")
    return json.loads(PLAN.read_text())


def cmd_manifest():
    if MANIFEST.exists():
        P.fail("replay run manifest already frozen")
    pl = plan()
    old = P.load_manifest()  # verifies old manifest hash, its code/evidence bindings, contracts, source
    if pl["source_run_manifest_sha256"] != P.MANIFEST_SHA or pl["previous_pilot_report_sha256"] != P.sha256(P.REPORT):
        P.fail("plan provenance mismatch")
    names = [s["name"] for s in old["scenes"]]
    order = pl["ordering"]["updates"]
    if order != uniform_order(pl["ordering"]["source_name_order"], TOTAL, 42) or pl["ordering"]["source_name_order"] != names:
        P.fail("recorded order does not reproduce")
    if hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest() != pl["ordering"]["sha256_compact_JSON_order"]:
        P.fail("order hash mismatch")
    excl = set(json.loads(P.PLAN.read_text())["excluded_names"])
    if set(order) - set(names) or excl & set(order) or len(order) != TOTAL:
        P.fail("order contains non-allowlisted names")
    lrs = pl["training"]["learning_rates_by_467_update_block"]
    # cross-check: the plan's block LRs equal CosineAnnealingLR(T_max 12, eta_min 1e-6) stepped per block
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=P.LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12, eta_min=1e-6)
    sched = []
    for _ in range(BLOCKS):
        sched.append(opt.param_groups[0]["lr"])
        opt.step(); sch.step()
    if max(abs(a - b) for a, b in zip(lrs, sched)) > 1e-12:
        P.fail(f"plan LRs {lrs} != scheduler {sched}")
    out = {"status": "FROZEN before replay fitting (Codex scope eba1a94)",
           "label": "IN-SAMPLE training/retention diagnostic - not validation, not acceptance",
           "bindings": {"replay_plan": PLAN_SHA, "pilot_run_manifest": P.MANIFEST_SHA, "pilot_report": P.sha256(P.REPORT),
                        "source_checkpoint": P.SOURCE_SHA, "contract_v1": P.V1, "contract_v2": P.V2,
                        "code": {f: P.sha256(P.TRAIN / f) for f in CODE if f != "provia_replay.py"},
                        "prior_epoch4_dev_evals": {a: P.sha256(p) for a, p in PRIOR_EP4.items()},
                        "target_cache": "provia-color-target-cache, per-file hashes from the pilot run manifest (re-checked every load)"},
           "data": {"scenes": 583, "capture_dr": pl["coverage_by_capture_dr"], "former_roles": {"fit": 467, "development": 116}},
           "training": {**pl["training"], "block_lrs_verified_against_CosineAnnealingLR": sched},
           "order_sha256": pl["ordering"]["sha256_compact_JSON_order"], "visits": dict(sorted(
               {k: v for k, v in __import__("collections").Counter(__import__("collections").Counter(order).values()).items()}.items())),
           "evaluation": {"updates": [0, TOTAL], "set": "all 583, streamed", "groups": ["all", "former fit 467", "former development 116",
                                                                                  "session", "captureDynamicRange"],
                          "prior_comparison": "former-116 endpoint vs frozen pilot epoch-4 dev results per arm (coverage differs; labelled)"},
           "runner_note": "provia_replay.py binds this manifest by pinned hash; its committed version is the one in the commit adding this file"}
    MANIFEST.write_text(json.dumps(out, indent=1))
    print("replay manifest sha256", P.sha256(MANIFEST))


def load():
    if MANIFEST_SHA == "__UNPINNED__" or P.sha256(MANIFEST) != MANIFEST_SHA:
        P.fail("replay manifest missing/unpinned/changed")
    man = json.loads(MANIFEST.read_text())
    for f, h in man["bindings"]["code"].items():
        if P.sha256(P.TRAIN / f) != h:
            P.fail(f"bound code {f} changed")
    for a, p in PRIOR_EP4.items():
        if P.sha256(p) != man["bindings"]["prior_epoch4_dev_evals"][a]:
            P.fail("prior epoch-4 evidence changed")
    old = P.load_manifest()
    return man, old, plan()


def state_path(arm, block):
    return P.CKPT / f"provia-replay-{arm}-block{block}.pt"


def new_arm(dev):
    mdl, c = P.build(dev)
    opt = torch.optim.Adam(mdl.parameters(), lr=P.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    return mdl, c, opt


def run_updates(arm, mdl, c, opt, pairs, order, start, stop, lrs, dev, log=True):
    acc = {"linear_l1": [], "lab_l1_scaled": [], "combined": []}
    for u in range(start, stop):
        lr = lrs[u // BLOCK]
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, cond = pairs.get(order[u])
        terms = P.loss_terms(P.fwd(mdl, c, x, cond, dev), y.to(dev), arm)
        loss = P.arm_loss(arm, terms)
        if not torch.isfinite(loss):
            P.fail(f"[{arm}] non-finite loss at update {u} ({order[u]})", code=4)
        opt.zero_grad()
        loss.backward()
        for n, p in mdl.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                P.fail(f"[{arm}] non-finite gradient {n} at update {u}", code=4)
        opt.step()
        for k, v in terms.items():
            acc[k].append(float(v.item()))
    return acc


def cmd_smoke():
    man, old, pl = load()
    pairs = P.Pairs(old)
    import subprocess
    for bad in (json.loads(P.PLAN.read_text())["excluded_names"][0], "DSCF9999.RAF"):
        r = subprocess.run([sys.executable, "-c", f"import provia_replay as R; m,o,p=R.load(); R.P.Pairs(o).get({bad!r})"],
                           cwd=P.TRAIN, capture_output=True, text=True)
        if r.returncode != 2 or "refused" not in r.stdout:
            P.fail(f"refusal failed for {bad}")
    order = pl["ordering"]["updates"]
    from collections import Counter
    vc = Counter(order)
    assert len(vc) == 583 and Counter(vc.values()) == {3: 464, 4: 119}, Counter(vc.values())
    print("refusals ok; coverage 583 (464x3, 119x4)")
    lrs = pl["training"]["learning_rates_by_467_update_block"]
    # exact resume on CPU (deterministic): 3 straight updates vs 2 + save/reload + 1
    cpu = torch.device("cpu")
    torch.manual_seed(42)
    mA, c, oA = new_arm(cpu)
    run_updates("candidate", mA, c, oA, pairs, order, 0, 3, lrs, cpu)
    mB, c, oB = new_arm(cpu)
    run_updates("candidate", mB, c, oB, pairs, order, 0, 2, lrs, cpu)
    tmp = P.CKPT / "provia-replay-smoke-resume.pt.partial"
    torch.save({"model": mB.state_dict(), "optimizer": oB.state_dict(), "cursor": 2}, tmp)
    st = torch.load(tmp, map_location="cpu", weights_only=False)
    tmp.unlink()
    mC, c, oC = new_arm(cpu)
    mC.load_state_dict(st["model"]); oC.load_state_dict(st["optimizer"])
    run_updates("candidate", mC, c, oC, pairs, order, st["cursor"], 3, lrs, cpu)
    diff = max((a - b).abs().max().item() for a, b in zip(mA.state_dict().values(), mC.state_dict().values()))
    if diff != 0.0:
        P.fail(f"resume not exact on CPU (max diff {diff})")
    print("exact resume (CPU): max param diff 0.0")
    dev = P.device()
    from eval_cuban_step1_k import load_model
    ref, rc = load_model(P.SOURCE, dev)
    for arm in ARMS:
        mdl, c, opt = new_arm(dev)
        x, y, cond = pairs.get(order[0])
        with torch.no_grad():
            d = (P.fwd(mdl.eval(), c, x, cond, dev) - P.fwd(ref, rc, x, cond, dev)).abs().max().item()
        if d != 0.0:
            P.fail(f"{arm} warm start differs")
        mdl.train()
        t0 = time.time()
        run_updates(arm, mdl, c, opt, pairs, order, 0, 2, lrs, dev)
        print(f"{arm}: no-op warm start exact; 2 finite full-res updates in {time.time() - t0:.1f}s; {P.mem()}")
    P.check_source()
    print("smoke ok")


def evaluate_all(mdl, c, old, pairs, dev, tag):
    mdl.eval()
    out = {}
    role = {s["name"]: s["role"] for s in old["scenes"]}
    with torch.no_grad():
        for s in old["scenes"]:
            x, y, cond = pairs.get(s["name"])
            pred = P.fwd(mdl, c, x, cond, dev)
            row = P.dev_metrics(pred, y)
            row["loss_terms"] = {k: float(v.item()) for k, v in P.per_scene_terms(pred, y.to(dev)).items()}
            row.update(session=s["session"], captureDynamicRange=s["captureDynamicRange"], former_role=role[s["name"]])
            out[s["name"]] = row
    print(f"[{tag}] all-pool mean dE00 {np.mean([r['mean_dE00'] for r in out.values()]):.4f}", flush=True)
    return out


def eval_path(name):
    return P.RUNS / f"provia-replay-{name}.json"


def cmd_eval0():
    man, old, pl = load()
    p = eval_path("r11-update0")
    if p.exists():
        return print("exists")
    dev = P.device()
    mdl, c = P.build(dev)
    res = evaluate_all(mdl, c, old, P.Pairs(old), dev, "r11 update0")
    p.write_text(json.dumps({"label": "IN-SAMPLE (R11 trained on all 583)", "model": "original R11 seed 42 = update 0 of both arms",
                             "manifest_sha256": MANIFEST_SHA, "per_scene": res}, indent=1))
    P.check_source()


def cmd_train(arm):
    if arm not in ARMS:
        P.fail("unknown arm")
    man, old, pl = load()
    pairs, dev = P.Pairs(old), P.device()
    order, lrs = pl["ordering"]["updates"], pl["training"]["learning_rates_by_467_update_block"]
    torch.manual_seed(42)
    mdl, c, opt = new_arm(dev)
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
        acc = run_updates(arm, mdl, c, opt, pairs, order, b * BLOCK, (b + 1) * BLOCK, lrs, dev)
        for n, p in mdl.named_parameters():
            if not torch.isfinite(p).all():
                P.fail(f"[{arm}] non-finite weight {n}", code=4)
        secs = time.time() - t0
        row = {"block": b + 1, "updates": [b * BLOCK, (b + 1) * BLOCK], "lr": lrs[b],
               **{f"train_{k}": float(np.mean(v)) for k, v in acc.items()}, "seconds": round(secs, 1),
               "seconds_per_update": round(secs / BLOCK, 3), **P.mem()}
        hist.append(row)
        print(f"[{arm}] block {b + 1}/{BLOCKS} lin {row['train_linear_l1']:.5f} lab/100 {row['train_lab_l1_scaled']:.5f} "
              f"lr {lrs[b]:.3e} {row['seconds']}s", flush=True)
        st = {"arm": arm, "cursor": (b + 1) * BLOCK, "model": mdl.state_dict(), "optimizer": opt.state_dict(),
              "torch_rng": torch.get_rng_state(), "history": hist, "config": c, "manifest_sha256": MANIFEST_SHA,
              "order_sha256": man["order_sha256"]}
        tmp = state_path(arm, b + 1).with_suffix(".pt.partial")
        torch.save(st, tmp)
        tmp.rename(state_path(arm, b + 1))
    p = eval_path(f"{arm}-update{TOTAL}")
    if not p.exists():
        st = torch.load(state_path(arm, BLOCKS), map_location="cpu", weights_only=False)
        mdl.load_state_dict(st["model"])
        res = evaluate_all(mdl, c, old, pairs, dev, f"{arm} update{TOTAL}")
        p.write_text(json.dumps({"label": "IN-SAMPLE training/retention", "arm": arm, "update": TOTAL,
                                 "state_sha256": P.sha256(state_path(arm, BLOCKS)), "manifest_sha256": MANIFEST_SHA,
                                 "per_scene": res}, indent=1))
    (P.RUNS / f"provia-replay-{arm}.json").write_text(json.dumps(
        {"arm": arm, "manifest_sha256": MANIFEST_SHA, "order_sha256": man["order_sha256"], "history": hist,
         "state_sha256": {f"block{b}": P.sha256(state_path(arm, b)) for b in range(1, BLOCKS + 1)}}, indent=1))
    P.check_source()
    print(f"[{arm}] done")


def cmd_report():
    if REPORT.exists():
        P.fail("report exists")
    self_test()
    man, old, pl = load()
    logs = {a: json.loads((P.RUNS / f"provia-replay-{a}.json").read_text()) for a in ARMS}
    for a in ARMS:
        if logs[a]["order_sha256"] != man["order_sha256"]:
            P.fail("order mismatch")
        for b in range(1, BLOCKS + 1):
            if P.sha256(state_path(a, b)) != logs[a]["state_sha256"][f"block{b}"]:
                P.fail(f"{a} block {b} state changed")
    ev = {"r11": json.loads(eval_path("r11-update0").read_text())["per_scene"]}
    for a in ARMS:
        ev[a] = json.loads(eval_path(f"{a}-update{TOTAL}").read_text())["per_scene"]
    prior = {a: json.loads(PRIOR_EP4[a].read_text())["per_scene"] for a in ARMS}
    names = sorted(ev["r11"])

    def agg(d, sel):
        rows = [d[n] for n in sel]
        f = lambda k: [r[k] for r in rows if r[k] is not None]
        return {"n": len(rows), "mean_dE00": float(np.mean(f("mean_dE00"))), "median_dE00": float(np.median(f("mean_dE00"))),
                "mean_p95_dE00": float(np.mean(f("p95_dE00"))), "mean_linear_l1_unclamped": float(np.mean(f("linear_l1_unclamped"))),
                "mean_lab_l1_scaled": float(np.mean([r["loss_terms"]["lab_l1_scaled"] for r in rows])),
                "tone_band_signed_Lab": {b: [float(np.mean([r["tone_band_signed_Lab"][b][i] for r in rows if r["tone_band_signed_Lab"][b]]))
                                             for i in range(3)] for b in P.BANDS},
                "dC": float(np.mean(f("signed_dC_ref_C_gt_10"))), "hue_deg": float(np.mean(f("signed_hue_deg_ref_C_gt_20"))),
                "frac_below_0": float(np.mean(f("frac_any_channel_below_0"))), "frac_above_1": float(np.mean(f("frac_any_channel_above_1")))}

    groups = {"all": names, "former_fit": [n for n in names if ev["r11"][n]["former_role"] == "fit"],
              "former_development": [n for n in names if ev["r11"][n]["former_role"] == "development"]}
    for key in ("session", "captureDynamicRange"):
        for n in names:
            groups.setdefault(f"{key}={ev['r11'][n][key]}", []).append(n)
    table = {g: {k: agg(ev[k], sel) for k in ("r11", *ARMS)} for g, sel in groups.items()}
    for g in table:
        t = table[g]
        t["candidate_minus_control"] = t["candidate"]["mean_dE00"] - t["control"]["mean_dE00"]
        t["candidate_minus_r11"] = t["candidate"]["mean_dE00"] - t["r11"]["mean_dE00"]
        t["control_minus_r11"] = t["control"]["mean_dE00"] - t["r11"]["mean_dE00"]
    dev116 = groups["former_development"]
    prior_cmp = {a: {"replay_update1868_former116_mean_dE00": table["former_development"][a]["mean_dE00"],
                     "pilot_epoch4_former116_mean_dE00 (116 withheld from training)": float(np.mean([prior[a][n]["mean_dE00"] for n in dev116])),
                     "r11": table["former_development"]["r11"]["mean_dE00"]} for a in ARMS}
    per = {n: {"session": ev["r11"][n]["session"], "captureDynamicRange": ev["r11"][n]["captureDynamicRange"],
               "former_role": ev["r11"][n]["former_role"],
               **{k: {"mean_dE00": ev[k][n]["mean_dE00"], "p95_dE00": ev[k][n]["p95_dE00"],
                      "linear_l1": ev[k][n]["linear_l1_unclamped"]} for k in ("r11", *ARMS)},
               "candidate_minus_r11": ev["candidate"][n]["mean_dE00"] - ev["r11"][n]["mean_dE00"],
               "control_minus_r11": ev["control"][n]["mean_dE00"] - ev["r11"][n]["mean_dE00"],
               "candidate_minus_control": ev["candidate"][n]["mean_dE00"] - ev["control"][n]["mean_dE00"],
               **({f"pilot_epoch4_{a}": prior[a][n]["mean_dE00"] for a in ARMS} if n in dev116 else {})} for n in names}

    def regress(key):
        v = sorted(((per[n][key], n) for n in names if per[n][key] > 0), reverse=True)
        return {"count": len(v), "worst": v[:8]}
    out = {"label": "IN-SAMPLE training/retention diagnostic: all 583 scenes are training data for R11 and both replay arms. "
                    "Not validation, not acceptance; single seed; coverage and order changed together vs the pilot.",
           "manifest_sha256": MANIFEST_SHA, "groups": table, "former116_vs_pilot_epoch4": prior_cmp,
           "regressions_vs_r11": {"candidate": regress("candidate_minus_r11"), "control": regress("control_minus_r11")},
           "candidate_worse_than_control": regress("candidate_minus_control"),
           "training_history": {a: logs[a]["history"] for a in ARMS}, "per_scene": per}
    P.check_source()
    REPORT.write_text(json.dumps(out, indent=1))
    brief = {g: {k: round(table[g][k]["mean_dE00"], 3) for k in ("r11", *ARMS)} | {"n": table[g]["r11"]["n"]} for g in table}
    print(json.dumps({"mean_dE00_by_group": brief, "former116_vs_pilot_epoch4": prior_cmp,
                      "regressions": out["regressions_vs_r11"], "cand_worse_than_control": out["candidate_worse_than_control"]["count"]}, indent=1))


if __name__ == "__main__":
    {"manifest": cmd_manifest, "smoke": cmd_smoke, "eval0": cmd_eval0, "report": cmd_report,
     "train": lambda: cmd_train(sys.argv[2])}[sys.argv[1]]()
