"""Single matched extended-colour-loss trial (Codex scope 2a5cafc; plan
provia-extended-loss-plan.json a0c96e93...). In-sample fitting only.

Only change vs the frozen low-rate candidate (provia_low_rate.py, manifest
defb2f06...): the TRAINING colour term uses color_loss_extended.per_scene_terms
(no RGB hard clamp before Lab). Everything else identical: original R11 seed 42
init, fresh Adam (0.9, 0.999, 1e-8, wd 0), all parameters trainable, the 583
hash-bound pairs, recorded 1,868-update order, batch 1, block LRs 1e-5 ...
8.5502e-6 (cosine horizon 12, eta_min 1e-7), stop after block four.

Evaluation uses the UNCHANGED display-clamped numpy Lab/CIEDE2000 scorer. Both
objective definitions (display-combined from color_loss.py, extended-combined
from color_loss_extended.py) are recomputed for R11, the frozen low-rate
candidate and the new run, under explicit names - never compared as one loss.
Old modules are imported, never edited or patched. K/T8/monitor/finalHeldOut/
dr100Eval: no code path.

  python3 training/provia_extended.py manifest | smoke | train | evaluate | review | report
"""
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_loss as CL  # noqa: E402
import color_loss_extended as CX  # noqa: E402
import provia_color_pilot as P  # noqa: E402
import provia_low_rate as L  # noqa: E402
import provia_replay as R  # noqa: E402
from audit_r11_display_domain import domain_stats  # noqa: E402
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402

PLAN = P.DIAG / "provia-extended-loss-plan.json"
PLAN_SHA = "a0c96e93cd515937dcef7e033e0b2467ce908c497d883afb7bc08c486d4dcda1"
MANIFEST = P.DIAG / "provia-extended-run-manifest.json"
MANIFEST_SHA = "298af0bc5d8be43e3f8b7b8225597a03575376ab24a749a3984342f63ed9ab30"
EVAL = P.RUNS / "provia-extended-eval.json"
REVIEW = P.DIAG / "provia-extended-review.json"
PANELS = P.DIAG / "provia-extended-review"
REPORT = P.DIAG / "provia-extended-report.json"
BASE_STATE = L.state_path("candidate", 4)
BASE_EVAL = L.eval_path(f"candidate-update{L.TOTAL}")
BLOCK, BLOCKS, TOTAL = L.BLOCK, L.BLOCKS, L.TOTAL
CODE = ["provia_extended.py", "color_loss.py", "color_loss_extended.py", "color_metrics.py", "check_extended_color_loss.py",
        "provia_low_rate.py", "provia_replay.py", "provia_color_pilot.py", "audit_r11_display_domain.py", "model.py",
        "fjlrg.py", "dataset.py", "target_tiff.py", "prepare_extended_loss_plan.py", "review_provia_low_rate.py"]
EVIDENCE = ["provia-extended-loss-plan.json", "provia-extended-loss-probe-manifest.json", "provia-extended-loss-probes.json",
            "provia-extended-loss-synthetic.json", "provia-low-rate-report.json", "provia-low-rate-run-manifest.json",
            "provia-low-rate-tail-review.json"]
REVIEW_CASES = ["DSCF0523.RAF", "DSCF0964.RAF", "DSCF0514.RAF", "DSCF0511.RAF", "DSCF1391.RAF", "DSCF0346.RAF",
                "DSCF0826.RAF", "DSCF0819.RAF"]
MODELS = ("r11", "lowrate_display_candidate", "extended_candidate")


def plan():
    if P.sha256(PLAN) != PLAN_SHA:
        P.fail("extended plan changed")
    return json.loads(PLAN.read_text())


def cmd_manifest():
    if MANIFEST.exists():
        P.fail("extended manifest already frozen")
    pl = plan()
    lman, old, rpl = L.load()
    b = pl["bindings"]
    checks = {"baseline_low_rate_manifest": L.MANIFEST_SHA, "baseline_candidate_state": P.sha256(BASE_STATE),
              "baseline_candidate_evaluation": P.sha256(BASE_EVAL), "baseline_report": P.sha256(L.REPORT),
              "probe_manifest": P.sha256(P.DIAG / "provia-extended-loss-probe-manifest.json"),
              "probe_report": P.sha256(P.DIAG / "provia-extended-loss-probes.json"),
              "synthetic_checks": P.sha256(P.DIAG / "provia-extended-loss-synthetic.json"), "source_R11": P.SOURCE_SHA}
    for k, v in checks.items():
        if b[k] != v:
            P.fail(f"plan binding {k} mismatch")
    for f, h in b["code"].items():
        if P.sha256(P.TRAIN / f) != h:
            P.fail(f"plan code {f} changed")
    u = pl["unchanged"]
    if u["order_sha256"] != lman["bindings"]["order_sha256"] or u["block_learning_rates"] != lman["block_learning_rates"]:
        P.fail("order/LR mismatch with low-rate run")
    probes = json.loads((P.DIAG / "provia-extended-loss-probes.json").read_text())
    if sorted(probes["per_case"]) != sorted(REVIEW_CASES) or not all(v["change"] < 0 for v in probes["per_case"].values()):
        P.fail("probe evidence/case list mismatch")
    out = {"status": "FROZEN before extended-loss fitting (Codex scope 2a5cafc)",
           "label": "IN-SAMPLE fitting only - not validation, not selection",
           "change": pl["change"], "loss": pl["loss"], "unchanged": u,
           "bindings": {"plan": PLAN_SHA, **checks, "lowrate_order": lman["bindings"]["order_sha256"],
                        "pilot_manifest": P.MANIFEST_SHA, "replay_manifest": R.MANIFEST_SHA,
                        "code": {f: P.sha256(P.TRAIN / f) for f in CODE if f != "provia_extended.py"},
                        "evidence": {f: P.sha256(P.DIAG / f) for f in EVIDENCE}},
           "evaluation": {"scorer": "unchanged display-clamped color_metrics Lab/CIEDE2000",
                          "models": list(MODELS), "objectives": ["display_combined (color_loss.py)", "extended_combined (color_loss_extended.py)"],
                          "tails": "audit_r11_display_domain.domain_stats thresholds 0/1e-4/1e-3/0.01/0.05 + target-positive negative occupancy",
                          "review_cases": REVIEW_CASES},
           "runner_note": "provia_extended.py binds this manifest by pinned hash; committed version is the one in the commit adding this file"}
    MANIFEST.write_text(json.dumps(out, indent=1))
    print("extended manifest sha256", P.sha256(MANIFEST))


def load():
    if MANIFEST_SHA == "__UNPINNED__" or P.sha256(MANIFEST) != MANIFEST_SHA:
        P.fail("extended manifest missing/unpinned/changed")
    man = json.loads(MANIFEST.read_text())
    b = man["bindings"]
    for f, h in b["code"].items():
        if P.sha256(P.TRAIN / f) != h:
            P.fail(f"bound code {f} changed")
    for f, h in b["evidence"].items():
        if P.sha256(P.DIAG / f) != h:
            P.fail(f"bound evidence {f} changed")
    if P.sha256(BASE_STATE) != b["baseline_candidate_state"] or P.sha256(BASE_EVAL) != b["baseline_candidate_evaluation"]:
        P.fail("baseline candidate state/evaluation changed")
    lman, old, rpl = L.load()
    plan()
    return man, old, rpl, lman


def state_path(b):
    return P.CKPT / f"provia-extended-block{b}.pt"


def ext_terms(pred, y):
    return {k: v.mean() for k, v in CX.per_scene_terms(pred, y).items()}


def run_updates(mdl, c, opt, pairs, order, start, stop, lrs, dev):
    acc = {"linear_l1": [], "lab_l1_scaled": [], "combined": []}
    for u in range(start, stop):
        for g in opt.param_groups:
            g["lr"] = lrs[u // BLOCK]
        x, y, cond = pairs.get(order[u])
        terms = ext_terms(P.fwd(mdl, c, x, cond, dev), y.to(dev))
        loss = terms["combined"]
        if not torch.isfinite(loss):
            P.fail(f"non-finite loss at update {u} ({order[u]})", code=4)
        opt.zero_grad()
        loss.backward()
        for n, p in mdl.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                P.fail(f"non-finite gradient {n} at update {u}", code=4)
        opt.step()
        for k, v in terms.items():
            acc[k].append(float(v.item()))
    return acc


def cmd_smoke():
    man, old, rpl, lman = load()
    pairs = P.Pairs(old)
    import subprocess
    for bad in (json.loads(P.PLAN.read_text())["excluded_names"][0], "DSCF9999.RAF"):
        r = subprocess.run([sys.executable, "-c", f"import provia_extended as E; m,o,p,l=E.load(); E.P.Pairs(o).get({bad!r})"],
                           cwd=P.TRAIN, capture_output=True, text=True)
        if r.returncode != 2 or "refused" not in r.stdout:
            P.fail(f"refusal failed for {bad}")
    order, lrs = rpl["ordering"]["updates"], lman["block_learning_rates"]
    vc = Counter(order)
    assert len(order) == TOTAL and Counter(vc.values()) == {3: 464, 4: 119}
    print("refusals ok; order/coverage ok")
    cpu = torch.device("cpu")
    mA, c, oA = R.new_arm(cpu)
    run_updates(mA, c, oA, pairs, order, 0, 3, lrs, cpu)
    mB, c, oB = R.new_arm(cpu)
    run_updates(mB, c, oB, pairs, order, 0, 2, lrs, cpu)
    tmp = P.CKPT / "provia-extended-smoke-resume.pt.partial"
    torch.save({"model": mB.state_dict(), "optimizer": oB.state_dict()}, tmp)
    st = torch.load(tmp, map_location="cpu", weights_only=False)
    tmp.unlink()
    mC, c, oC = R.new_arm(cpu)
    mC.load_state_dict(st["model"]); oC.load_state_dict(st["optimizer"])
    run_updates(mC, c, oC, pairs, order, 2, 3, lrs, cpu)
    diff = max((a - b).abs().max().item() for a, b in zip(mA.state_dict().values(), mC.state_dict().values()))
    if diff != 0.0:
        P.fail(f"resume not exact ({diff})")
    print("exact resume (CPU) 0.0")
    dev = P.device()
    from eval_cuban_step1_k import load_model
    ref, rc = load_model(P.SOURCE, dev)
    mdl, c, opt = R.new_arm(dev)
    x, y, cond = pairs.get(order[0])
    with torch.no_grad():
        d = (P.fwd(mdl.eval(), c, x, cond, dev) - P.fwd(ref, rc, x, cond, dev)).abs().max().item()
    if d != 0.0:
        P.fail("warm start differs")
    mdl.train()
    t0 = time.time()
    run_updates(mdl, c, opt, pairs, order, 0, 2, lrs, dev)
    print(f"no-op warm start exact; 2 finite full-res extended-loss updates at lr {opt.param_groups[0]['lr']:.1e} in {time.time()-t0:.1f}s")
    P.check_source()
    print("smoke ok")


def cmd_train():
    man, old, rpl, lman = load()
    pairs, dev = P.Pairs(old), P.device()
    order, lrs = rpl["ordering"]["updates"], lman["block_learning_rates"]
    torch.manual_seed(42)
    mdl, c, opt = R.new_arm(dev)
    hist, start = [], 0
    done = [b for b in range(1, BLOCKS + 1) if state_path(b).exists()]
    if done:
        st = torch.load(state_path(max(done)), map_location="cpu", weights_only=False)
        if st["manifest_sha256"] != MANIFEST_SHA or st["cursor"] != max(done) * BLOCK:
            P.fail("resume state mismatch")
        mdl.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        hist, start = st["history"], st["cursor"]
        print(f"resumed at update {start}", flush=True)
    mdl.train()
    for b in range(start // BLOCK, BLOCKS):
        P.check_source()
        t0 = time.time()
        acc = run_updates(mdl, c, opt, pairs, order, b * BLOCK, (b + 1) * BLOCK, lrs, dev)
        for n, p in mdl.named_parameters():
            if not torch.isfinite(p).all():
                P.fail(f"non-finite weight {n}", code=4)
        row = {"block": b + 1, "lr": lrs[b], **{f"train_extended_{k}": float(np.mean(v)) for k, v in acc.items()},
               "seconds": round(time.time() - t0, 1), **P.mem()}
        hist.append(row)
        print(f"[extended] block {b+1}/{BLOCKS} lin {row['train_extended_linear_l1']:.5f} extlab/100 "
              f"{row['train_extended_lab_l1_scaled']:.5f} comb {row['train_extended_combined']:.5f} {row['seconds']}s", flush=True)
        st = {"cursor": (b + 1) * BLOCK, "model": mdl.state_dict(), "optimizer": opt.state_dict(), "history": hist,
              "config": c, "manifest_sha256": MANIFEST_SHA, "order_sha256": lman["bindings"]["order_sha256"]}
        tmp = state_path(b + 1).with_suffix(".pt.partial")
        torch.save(st, tmp)
        tmp.rename(state_path(b + 1))
    (P.RUNS / "provia-extended-train.json").write_text(json.dumps(
        {"manifest_sha256": MANIFEST_SHA, "history": hist,
         "state_sha256": {f"block{b}": P.sha256(state_path(b)) for b in range(1, BLOCKS + 1)}}, indent=1))
    P.check_source()
    print("[extended] done")


def models(dev):
    log = json.loads((P.RUNS / "provia-extended-train.json").read_text())
    if P.sha256(state_path(BLOCKS)) != log["state_sha256"][f"block{BLOCKS}"]:
        P.fail("extended endpoint state changed")
    out = {}
    m, c = P.build(dev); out["r11"] = (m.eval(), c)
    for key, path in (("lowrate_display_candidate", BASE_STATE), ("extended_candidate", state_path(BLOCKS))):
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st["cursor"] != TOTAL:
            P.fail(f"{key} state not at update {TOTAL}")
        m, c = P.build(dev); m.load_state_dict(st["model"]); out[key] = (m.eval(), c)
    return out


def objectives(pred, y):
    d, x = CL.per_scene_terms(pred, y), CX.per_scene_terms(pred, y)
    return {"display_combined": float(d["combined"].item()), "display_lab_l1_scaled": float(d["lab_l1_scaled"].item()),
            "extended_combined": float(x["combined"].item()), "extended_lab_l1_scaled": float(x["lab_l1_scaled"].item()),
            "linear_l1": float(d["linear_l1"].item())}


def cmd_evaluate():
    if EVAL.exists():
        P.fail("evaluation exists")
    self_test()
    man, old, rpl, lman = load()
    dev = P.device()
    ms, pairs = models(dev), P.Pairs(old)
    frozen = {"r11": json.loads(L.eval_path("r11-update0").read_text())["per_scene"],
              "lowrate_display_candidate": json.loads(BASE_EVAL.read_text())["per_scene"]}
    role = {s["name"]: s["role"] for s in old["scenes"]}
    res = {k: {} for k in MODELS}
    t0 = time.time()
    for i, s in enumerate(old["scenes"]):
        n = s["name"]
        x, y, cond = pairs.get(n)
        yd = y.to(dev)
        ref = y.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
        for k, (m, c) in ms.items():
            with torch.no_grad():
                pred = P.fwd(m, c, x, cond, dev)
                row = P.dev_metrics(pred, y)
                row["objectives"] = objectives(pred, yd)
            p = pred.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
            e = ciede2000(linear_srgb_to_lab(np.clip(p, 0, 1)), linear_srgb_to_lab(ref))
            row["domain"] = domain_stats(p, ref, e)
            if k in frozen and abs(row["mean_dE00"] - frozen[k][n]["mean_dE00"]) > 1e-8:
                P.fail(f"{k} {n} does not reproduce frozen mean dE00")
            for v in (row["mean_dE00"], row["p95_dE00"], *row["objectives"].values()):
                if not math.isfinite(v):
                    P.fail(f"non-finite metric {k} {n}")
            row.update(session=s["session"], captureDynamicRange=s["captureDynamicRange"], former_role=role[n])
            res[k][n] = row
        if i % 50 == 0:
            print(f"{i+1}/583 {time.time()-t0:.0f}s", flush=True)
    EVAL.write_text(json.dumps({"label": "IN-SAMPLE fitting/retention; unchanged display scorer", "manifest_sha256": MANIFEST_SHA,
                                "per_model": res}, allow_nan=False))
    P.check_source()
    print("evaluation done", {k: round(float(np.mean([r["mean_dE00"] for r in v.values()])), 4) for k, v in res.items()})


def cmd_review():
    if REVIEW.exists() or PANELS.exists():
        P.fail("review exists")
    man, old, rpl, lman = load()
    dev = P.device()
    ms, pairs = models(dev), P.Pairs(old)
    ev = json.loads(EVAL.read_text())["per_model"]
    PANELS.mkdir()
    from review_provia_low_rate import encode
    rec = {}
    for n in REVIEW_CASES:
        x, y, cond = pairs.get(n)
        ref = y.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
        preds = {}
        for k, (m, c) in ms.items():
            with torch.no_grad():
                preds[k] = P.fwd(m, c, x, cond, dev).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
        L_ref = linear_srgb_to_lab(ref)
        es = {k: ciede2000(linear_srgb_to_lab(np.clip(v, 0, 1)), L_ref) for k, v in preds.items()}
        for k in MODELS:
            if abs(es[k].mean() - ev[k][n]["mean_dE00"]) > 1e-8:
                P.fail(f"review {k} {n} does not reproduce evaluation")
        neg = {k: ((v < 0) & (ref > 0.001)).any(-1) for k, v in preds.items()}
        new_vs_r11 = {k: (neg[k] & ~neg["r11"]) for k in ("lowrate_display_candidate", "extended_candidate")}
        h = 420
        rs = lambda a: cv2.resize(a, (round(a.shape[1] * h / a.shape[0]), h), interpolation=cv2.INTER_AREA)
        def lab(a, t):
            a = rs(a).copy(); cv2.rectangle(a, (0, 0), (a.shape[1], 27), (0, 0, 0), -1)
            cv2.putText(a, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA); return a
        top = np.concatenate([lab(encode(preds["r11"]), "R11"), lab(encode(preds["lowrate_display_candidate"]), "Low-rate display cand"),
                              lab(encode(preds["extended_candidate"]), "Extended cand"), lab(encode(ref), "Fuji Provia")], axis=1)
        heat = lambda e: cv2.cvtColor(cv2.applyColorMap((255 * np.clip(e / 10, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        mask = np.zeros((*ref.shape[:2], 3), np.uint8)
        mask[neg["extended_candidate"]] = [255, 40, 40]
        bot = np.concatenate([lab(heat(es["r11"]), "R11 dE 0-10"), lab(heat(es["lowrate_display_candidate"]), "Low-rate dE 0-10"),
                              lab(heat(es["extended_candidate"]), "Extended dE 0-10"), lab(mask, "Ext negative; target>.001")], axis=1)
        path = PANELS / f"{Path(n).stem}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate([top, bot], 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rec[n] = {"mean_dE00": {k: float(es[k].mean()) for k in MODELS},
                  "negative_target_positive_fraction": {k: float(neg[k].mean()) for k in MODELS},
                  "new_negative_vs_r11_fraction": {k: float(v.mean()) for k, v in new_vs_r11.items()},
                  "new_negative_vs_r11_region_dE_change": {k: (float((es[k] - es["r11"])[v].mean()) if v.any() else None)
                                                          for k, v in new_vs_r11.items()},
                  "tails": {k: ev[k][n]["domain"]["tail_thresholds_linear"] for k in MODELS},
                  "panel": str(path.relative_to(P.TRAIN.parent))}
        print(n, {k: round(v, 3) for k, v in rec[n]["mean_dE00"].items()}, flush=True)
    REVIEW.write_text(json.dumps({"scope": "eight fixed train-only cases; diagnostic, not validation", "cases": REVIEW_CASES,
                                  "note": "negative occupancy is not proof of damage or repair; display scorer unchanged",
                                  "per_case": rec}, indent=1, allow_nan=False))
    P.check_source()


def cmd_report():
    if REPORT.exists():
        P.fail("report exists")
    man, old, rpl, lman = load()
    ev = json.loads(EVAL.read_text())["per_model"]
    names = sorted(ev["r11"])

    def mean_nullable(vals, what):
        for v in vals:
            if v is not None and not math.isfinite(v):
                P.fail(f"unexpected non-finite {what}")
        v = [x for x in vals if x is not None]
        return (float(np.mean(v)) if v else None), len(v)

    def agg(d, sel):
        rows = [d[n] for n in sel]
        hue, hue_n = mean_nullable([r["signed_hue_deg_ref_C_gt_20"] for r in rows], "hue")
        dC, dC_n = mean_nullable([r["signed_dC_ref_C_gt_10"] for r in rows], "dC")
        tails = {t: {k: float(np.mean([r["domain"]["tail_thresholds_linear"][t][k] for r in rows]))
                     for k in ("below_minus_threshold_pixel_fraction", "above_one_plus_threshold_pixel_fraction")}
                 for t in rows[0]["domain"]["tail_thresholds_linear"]}
        return {"n": len(rows), "mean_dE00": float(np.mean([r["mean_dE00"] for r in rows])),
                "median_dE00": float(np.median([r["mean_dE00"] for r in rows])),
                "mean_p95_dE00": float(np.mean([r["p95_dE00"] for r in rows])),
                "objective_display_combined": float(np.mean([r["objectives"]["display_combined"] for r in rows])),
                "objective_extended_combined": float(np.mean([r["objectives"]["extended_combined"] for r in rows])),
                "linear_l1": float(np.mean([r["objectives"]["linear_l1"] for r in rows])),
                "tone_band_signed_Lab": {b: [mean_nullable([r["tone_band_signed_Lab"][b][i] if r["tone_band_signed_Lab"][b] else None
                                                                 for r in rows], "tone")[0] for i in range(3)] for b in P.BANDS},
                "dC": dC, "dC_scenes": dC_n, "hue_deg": hue, "hue_scenes": hue_n,
                "frac_any_below_0": float(np.mean([r["frac_any_channel_below_0"] for r in rows])),
                "frac_any_above_1": float(np.mean([r["frac_any_channel_above_1"] for r in rows])),
                "negative_target_positive_fraction": float(np.mean([r["domain"]["negative_channels_whose_target_exceeds_0.001_pixel_fraction"]
                                                                    for r in rows])),
                "over_one_target_below_0999_fraction": float(np.mean([r["domain"]["over_one_channels_whose_target_is_below_0.999_pixel_fraction"]
                                                                      for r in rows])),
                "min_output": float(min(r["domain"]["min"] for r in rows)), "max_output": float(max(r["domain"]["max"] for r in rows)),
                "tails": tails}

    groups = {"all": names, "former_fit": [n for n in names if ev["r11"][n]["former_role"] == "fit"],
              "former_development": [n for n in names if ev["r11"][n]["former_role"] == "development"]}
    for key in ("session", "captureDynamicRange"):
        for n in names:
            groups.setdefault(f"{key}={ev['r11'][n][key]}", []).append(n)
    table = {g: {k: agg(ev[k], sel) for k in MODELS} for g, sel in groups.items()}
    for g, t in table.items():
        t["diff_mean_dE00"] = {"extended_minus_r11": t["extended_candidate"]["mean_dE00"] - t["r11"]["mean_dE00"],
                               "extended_minus_lowrate": t["extended_candidate"]["mean_dE00"] - t["lowrate_display_candidate"]["mean_dE00"],
                               "lowrate_minus_r11": t["lowrate_display_candidate"]["mean_dE00"] - t["r11"]["mean_dE00"]}

    def regress(a, b):
        v = sorted(((ev[a][n]["mean_dE00"] - ev[b][n]["mean_dE00"], n) for n in names if ev[a][n]["mean_dE00"] > ev[b][n]["mean_dE00"]),
                   reverse=True)
        return {"count": len(v), "worst": v[:10], "all": [[round(d, 5), n] for d, n in v]}
    hist = json.loads((P.RUNS / "provia-extended-train.json").read_text())["history"]
    out = {"label": "IN-SAMPLE fitting only: R11 and both candidates trained on all 583 scenes. Not validation; nothing selected.",
           "manifest_sha256": MANIFEST_SHA, "scorer": "unchanged display-clamped CIEDE2000",
           "objective_note": "display_combined and extended_combined are different definitions; compare each across models, never with each other",
           "groups": table,
           "regressions": {"extended_vs_r11": regress("extended_candidate", "r11"),
                           "extended_vs_lowrate": regress("extended_candidate", "lowrate_display_candidate"),
                           "lowrate_vs_r11": regress("lowrate_display_candidate", "r11")},
           "training_history": hist,
           "per_scene": {n: {"session": ev["r11"][n]["session"], "captureDynamicRange": ev["r11"][n]["captureDynamicRange"],
                             "former_role": ev["r11"][n]["former_role"],
                             **{k: {"mean_dE00": ev[k][n]["mean_dE00"], "p95_dE00": ev[k][n]["p95_dE00"],
                                    "objectives": ev[k][n]["objectives"]} for k in MODELS}} for n in names},
           "review": json.loads(REVIEW.read_text())["per_case"] if REVIEW.exists() else None}
    REPORT.write_text(json.dumps(out, indent=1, allow_nan=False))
    P.check_source()
    brief = {g: {k: round(t[k]["mean_dE00"], 3) for k in MODELS} | {"n": t["r11"]["n"]} for g, t in table.items()}
    print(json.dumps({"mean_dE00": brief, "regression_counts": {k: v["count"] for k, v in out["regressions"].items()}}, indent=1))


if __name__ == "__main__":
    {"manifest": cmd_manifest, "smoke": cmd_smoke, "train": cmd_train, "evaluate": cmd_evaluate, "review": cmd_review,
     "report": cmd_report}[sys.argv[1]]()
