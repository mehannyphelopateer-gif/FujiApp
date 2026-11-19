"""Classic Cuban Neg step 1 - .fjlrg -> E4 (Classic Negative, DR100, As-Shot,
neutral) adaptation. Codex-approved design (docs/phase4-scene-adaptive-
scope.md 2026-11-18); frozen inputs: contract v2, cuban-step1-folds.json,
cuban-step1-run-manifest.json (hashes pinned below - any mismatch refuses).

Separate HybridPredictor warm-started from a COPY of Round 11 seed 42
(source file hash checked before and after every run, never written).
grid.encoder frozen; grid.head + refinement trainable. Per-scene linear L1
on E4 targets, Adam, cosine T_max=20 (eta_min 0.01*lr), batch 1.

Training inputs come ONLY from the run manifest's F allowlist (24 scenes),
re-hashed at load. K, monitor, finalHeldOut and T8 have no code path here.

  python3 training/run_cuban_step1.py smoke   # F-only, tiny, writes to scratch only
  python3 training/run_cuban_step1.py cv      # 2 lr x 4 folds x 20 epochs (resumable)
  python3 training/run_cuban_step1.py select  # chosen (lr, epoch) from OOF L1 -> diagnostics
  python3 training/run_cuban_step1.py final   # seeds 42/43/44 on all F at the chosen setting
  python3 training/run_cuban_step1.py freeze  # checkpoint hash manifest (before any K evaluation)
"""
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuban_contract import contract_sha256  # noqa: E402
from cuban_contract_v2 import contract_v2_sha256, _check as v2_check  # noqa: E402
from dataset import _read_with_retry  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from model import HybridPredictor  # noqa: E402
from target_tiff import load_xraw_target_linear_matching  # noqa: E402

TRAIN = Path(__file__).resolve().parent
CAL = TRAIN.parent / "calibration-input"
DIAG = TRAIN / "diagnostics"
CKPT = TRAIN / "checkpoints"
RUNS = TRAIN / "runs"
CACHE = CAL / "cuban-step1-target-cache"
FOLDS = DIAG / "cuban-step1-folds.json"
RUNMAN = DIAG / "cuban-step1-run-manifest.json"
SELECTION = DIAG / "cuban-step1-selection.json"
FREEZE = DIAG / "cuban-step1-freeze.json"

V1_SHA = "c3f840ef9fad9d48de83b4edf183f2434523f352f358305b91eb7764d22d0067"
V2_SHA = "9824b0799737960363c95bb028332fdd0e5118e748fc0139375f1d0e230099a0"
FOLDS_SHA = "76b55d9126caf74d13f99aa9735235d51e013892dfdd99a95d8e36f9f7d1add0"
RUNMAN_SHA = "27dc5918ce962e3cac516f8768bcb5c340bcd761c11a6dacd3303ed6573e4416"
SOURCE = CKPT / "r11-drcond-seed42-best.pt"
SOURCE_SHA = "1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c"
LRS, MAX_EPOCHS, T_MAX, CV_SEED, FINAL_SEEDS = (1e-4, 3e-4), 20, 20, 42, (42, 43, 44)
EXPECTED_TRAINABLE, EXPECTED_TOTAL = 17655, 115719


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    print(f"INTEGRITY FAILURE: {msg}", flush=True)
    sys.exit(2)


def check_source():
    if sha256(SOURCE) != SOURCE_SHA:
        fail("Round 11 seed 42 source checkpoint hash changed")


def preflight():
    v2_check()
    if contract_sha256() != V1_SHA or contract_v2_sha256() != V2_SHA:
        fail("contract v1/v2 fingerprint mismatch")
    if sha256(FOLDS) != FOLDS_SHA or sha256(RUNMAN) != RUNMAN_SHA:
        fail("frozen fold/run manifest changed")
    check_source()
    man = json.loads(RUNMAN.read_text())
    folds = json.loads(FOLDS.read_text())
    if man["folds_sha256"] != FOLDS_SHA:
        fail("run manifest does not bind the frozen folds")
    allow = {r["name"]: r for r in man["F_allowlist"]}
    k = {r["name"] for r in man["K_eval_only"]}
    if len(allow) != 24 or allow.keys() & k:
        fail("F allowlist malformed or overlaps K")
    in_folds = [n for f in folds["folds"] for n in f["validation_scenes"]]
    if sorted(in_folds) != sorted(allow) or len(set(in_folds)) != 24:
        fail("folds do not partition the F allowlist exactly")
    return man, folds, allow, k


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_pairs(allow, names, k_names):
    """F-allowlist-only loader: every file re-hashed against the frozen run manifest."""
    out = {}
    CACHE.mkdir(exist_ok=True)
    for n in names:
        if n not in allow or n in k_names:
            fail(f"{n} is not in the F allowlist - training refused")
        r = allow[n]
        fj, e4 = CAL / r["fjlrg"], CAL / r["e4"]
        if _read_with_retry(sha256, fj) != r["fjlrg_sha256"]:
            fail(f"{r['fjlrg']} hash != run manifest")
        x, w, h = _read_with_retry(load_fjlrg_linear, fj)
        cp, side = CACHE / f"{r['shoot']}-E4.npy", CACHE / f"{r['shoot']}-E4.json"
        meta = json.loads(side.read_text()) if side.exists() else None
        if meta and meta.get("e4_sha256") == r["e4_sha256"] and meta.get("shape") == list(x.shape) and cp.exists():
            y = _read_with_retry(np.load, cp)
            if sha256(cp) != meta["npy_sha256"]:
                fail(f"target cache {cp.name} corrupted")
        else:
            if _read_with_retry(sha256, e4) != r["e4_sha256"]:
                fail(f"{r['e4']} hash != run manifest (verified)")
            y = _read_with_retry(load_xraw_target_linear_matching, e4, w, h).astype(np.float32)
            np.save(cp, y)
            side.write_text(json.dumps({"e4_sha256": r["e4_sha256"], "shape": list(y.shape), "npy_sha256": sha256(cp),
                                        "loader": "target_tiff.load_xraw_target_linear_matching"}))
        if y.shape != x.shape:
            fail(f"{n}: input {x.shape} vs target {y.shape}")
        out[n] = (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0),
                  torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1))).unsqueeze(0))
    return out


def build_from_source(dev):
    ck = torch.load(SOURCE, map_location="cpu", weights_only=False)
    c = ck["config"]
    assert c["use_hybrid"] and c["capture_dr_conditioning"]
    m = HybridPredictor(grid_spatial=c["grid_spatial"], grid_luma=c["grid_luma"], low_res=c["low_res"],
                        base_ch=c["base_ch"], refinement_base_ch=c["refinement_base_ch"],
                        refinement_max_delta=c["refinement_max_delta"], condition_channels=1)
    m.load_state_dict(ck["state_dict"])
    trainable, frozen = [], []
    for name, p in m.named_parameters():
        if name.startswith(("grid.head.", "refinement.")):
            p.requires_grad_(True)
            trainable.append(name)
        elif name.startswith("grid.encoder."):
            p.requires_grad_(False)
            frozen.append(name)
        else:
            fail(f"parameter {name} matches neither trainable nor frozen prefixes")
    nt = sum(p.numel() for p in m.parameters() if p.requires_grad)
    ntot = sum(p.numel() for p in m.parameters())
    if nt != EXPECTED_TRAINABLE or ntot != EXPECTED_TOTAL:
        fail(f"trainable {nt}/{ntot} != {EXPECTED_TRAINABLE}/{EXPECTED_TOTAL}")
    return m.to(dev), c, trainable, frozen


COND = torch.tensor([1.0])  # capture DR400 -> stops 2 -> /2 = 1.0 (all F/K verified DR400 in the cohort pool)


def forward(m, c, x, dev):
    out, _ = m.predict_and_apply(x.to(dev), c["low_res"], c.get("luma_mode", "rec709"), condition=COND.to(dev))
    return out


def eval_l1(m, c, pairs, names, dev):
    m.eval()
    res = {}
    with torch.no_grad():
        for n in names:
            x, y = pairs[n]
            res[n] = float((forward(m, c, x, dev) - y.to(dev)).abs().mean().item())
    return res


def train(pairs, train_names, val_names, lr, seed, stop_epoch, dev, tag, log=print):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m, c, trainable, frozen = build_from_source(dev)
    enc0 = {k: v.detach().clone() for k, v in m.state_dict().items() if k.startswith("grid.encoder.")}
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=lr * 0.01)
    gen = torch.Generator().manual_seed(seed)
    hist = [{"epoch": 0, "val": eval_l1(m, c, pairs, val_names, dev) if val_names else {}}]
    order_log = []
    for ep in range(1, stop_epoch + 1):
        t0 = time.time()
        m.train()
        order = [train_names[i] for i in torch.randperm(len(train_names), generator=gen).tolist()]
        order_log.append(order)
        tl = []
        for n in order:
            x, y = pairs[n]
            loss = (forward(m, c, x, dev) - y.to(dev)).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(loss.item())
        lr_used = opt.param_groups[0]["lr"]
        sch.step()
        val = eval_l1(m, c, pairs, val_names, dev) if val_names else {}
        row = {"epoch": ep, "train_l1": float(np.mean(tl)), "lr": lr_used, "val": val,
               "val_mean": float(np.mean(list(val.values()))) if val else None, "seconds": round(time.time() - t0, 1)}
        hist.append(row)
        log(f"[{tag}] epoch {ep}/{stop_epoch} train_l1={row['train_l1']:.5f} "
            f"val_mean={row['val_mean'] if row['val_mean'] is None else round(row['val_mean'], 5)} lr={lr_used:.2e} {row['seconds']}s")
    for k, v in m.state_dict().items():
        if k.startswith("grid.encoder.") and not torch.equal(v.cpu(), enc0[k].cpu()):
            fail(f"frozen encoder tensor {k} changed during training")
    return m, c, hist, order_log, trainable, frozen


def cmd_smoke():
    man, folds, allow, k = preflight()
    dev = device()
    scratch = Path(os.environ.get("CUBAN_SMOKE_DIR", "/tmp")) / "cuban-step1-smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    # 1) allowlist refuses a K scene and an unknown name before touching any file
    for bad in (sorted(k)[0], "DSCF9999.RAF"):
        try:
            import subprocess  # run the refusal in a child so fail()'s exit doesn't stop the smoke test
            r = subprocess.run([sys.executable, "-c", f"import run_cuban_step1 as s; m,f,a,k=s.preflight(); "
                                f"s.load_pairs(a, [{bad!r}], k)"], cwd=TRAIN, capture_output=True, text=True)
            assert r.returncode == 2 and "not in the F allowlist" in r.stdout, (bad, r.stdout, r.stderr)
        except AssertionError as e:
            fail(f"allowlist did not refuse {e}")
    print("allowlist refuses K/unknown names: ok")
    # 2) tiny F-only fit: 3 train + 1 val from fold 2/3 (the smallest folds), 2 epochs
    tr = folds["folds"][2]["validation_scenes"][:3]
    va = folds["folds"][3]["validation_scenes"][:1]
    pairs = load_pairs(allow, tr + va, k)
    m, c, hist, _, trainable, frozen = train(pairs, tr, va, 3e-4, 42, 2, dev, "smoke")
    torch.save({"state_dict": m.state_dict()}, scratch / "smoke.pt")
    check_source()
    out = {"train": tr, "val": va, "trainable_tensors": len(trainable), "frozen_tensors": len(frozen),
           "history": [{kk: vv for kk, vv in h.items()} for h in hist], "source_sha_after": SOURCE_SHA}
    (scratch / "smoke.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["history"], indent=1))
    print("smoke ok; outputs only in", scratch)


def cv_path(lr, f):
    return RUNS / f"cn-step1-cv-lr{lr:.0e}-fold{f}.json"


def cmd_cv():
    man, folds, allow, k = preflight()
    dev = device()
    pairs = load_pairs(allow, list(allow), k)
    for lr in LRS:
        for fo in folds["folds"]:
            p = cv_path(lr, fo["fold"])
            if p.exists():
                print(f"{p.name} exists - reusing")
                continue
            va = fo["validation_scenes"]
            tr = [n for n in sorted(allow) if n not in va]
            tag = f"cv lr={lr:.0e} fold={fo['fold']}"
            check_source()
            _, _, hist, order, trainable, _ = train(pairs, tr, va, lr, CV_SEED, MAX_EPOCHS, dev, tag)
            check_source()
            p.write_text(json.dumps({"lr": lr, "fold": fo["fold"], "seed": CV_SEED, "train": tr, "val": va,
                                     "T_max": T_MAX, "history": hist, "order": order,
                                     "source_sha256_before_after": [SOURCE_SHA, SOURCE_SHA],
                                     "folds_sha256": FOLDS_SHA, "run_manifest_sha256": RUNMAN_SHA}, indent=1))


def cmd_select():
    man, folds, allow, k = preflight()
    if SELECTION.exists():
        print(SELECTION.read_text())
        return
    table = {}
    for lr in LRS:
        rows = [json.loads(cv_path(lr, fo["fold"]).read_text()) for fo in folds["folds"]]
        for r in rows:
            if r["folds_sha256"] != FOLDS_SHA or r["run_manifest_sha256"] != RUNMAN_SHA:
                fail(f"CV log {r['lr']}/{r['fold']} not bound to the frozen manifests")
        for ep in range(0, MAX_EPOCHS + 1):
            per = {}
            for r in rows:
                per.update(r["history"][ep]["val"])
            if sorted(per) != sorted(allow):
                fail("OOF set is not exactly the 24 F scenes")
            table[(lr, ep)] = float(np.mean(list(per.values())))
    cands = [(v, ep, lr) for (lr, ep), v in table.items() if ep >= 1]
    best = min(cands)  # mean OOF L1, then fewer epochs, then smaller lr
    sel = {"chosen": {"lr": best[2], "epoch": best[1], "mean_oof_l1": best[0]},
           "rule": man["search"]["selection"] + "; tie-break " + man["search"]["tie_break"],
           "epoch0_warmstart_oof_l1": table[(LRS[0], 0)],
           "oof_mean_l1": {f"{lr:.0e}": [round(table[(lr, ep)], 6) for ep in range(0, MAX_EPOCHS + 1)] for lr in LRS},
           "folds_sha256": FOLDS_SHA, "run_manifest_sha256": RUNMAN_SHA,
           "cv_logs_sha256": {cv_path(lr, fo["fold"]).name: sha256(cv_path(lr, fo["fold"]))
                              for lr in LRS for fo in folds["folds"]}}
    SELECTION.write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


def cmd_final():
    man, folds, allow, k = preflight()
    sel = json.loads(SELECTION.read_text())
    lr, ep = sel["chosen"]["lr"], sel["chosen"]["epoch"]
    dev = device()
    pairs = load_pairs(allow, list(allow), k)
    for seed in FINAL_SEEDS:
        ck = CKPT / f"cn-step1-seed{seed}.pt"
        if ck.exists():
            print(f"{ck.name} exists - reusing")
            continue
        check_source()
        m, c, hist, order, trainable, frozen = train(pairs, sorted(allow), [], lr, seed, ep, dev, f"final seed={seed}")
        check_source()
        meta = {"seed": seed, "lr": lr, "stop_epoch": ep, "T_max": T_MAX, "train": sorted(allow),
                "source_checkpoint": str(SOURCE.name), "source_sha256_before_after": [SOURCE_SHA, SOURCE_SHA],
                "trainable_parameter_names": trainable, "frozen_parameter_names": frozen,
                "optimizer": man["optimizer"], "schedule": man["schedule"],
                "contract_v2_sha256": V2_SHA, "folds_sha256": FOLDS_SHA, "run_manifest_sha256": RUNMAN_SHA,
                "selection_sha256": sha256(SELECTION)}
        tmp = ck.with_suffix(".pt.partial")
        torch.save({"state_dict": m.state_dict(), "config": c, "cuban_step1": meta}, tmp)
        tmp.rename(ck)
        (RUNS / f"cn-step1-final-seed{seed}.json").write_text(json.dumps({**meta, "history": hist, "order": order}, indent=1))
        # also record train-tier (in-sample) L1 for context
        print(f"[final seed={seed}] saved {ck.name}")


def cmd_freeze():
    man, folds, allow, k = preflight()
    if FREEZE.exists():
        fail("freeze manifest already exists - checkpoints are already frozen")
    ck = {s: CKPT / f"cn-step1-seed{s}.pt" for s in FINAL_SEEDS}
    for p in ck.values():
        if not p.exists():
            fail(f"missing {p.name}")
    out = {"status": "FROZEN before the single K evaluation",
           "checkpoints": {p.name: sha256(p) for p in ck.values()}, "primary": "cn-step1-seed42.pt",
           "source_checkpoint": {SOURCE.name: sha256(SOURCE)}, "selection_sha256": sha256(SELECTION),
           "contract_v2_sha256": V2_SHA, "folds_sha256": FOLDS_SHA, "run_manifest_sha256": RUNMAN_SHA,
           "final_logs_sha256": {f"cn-step1-final-seed{s}.json": sha256(RUNS / f"cn-step1-final-seed{s}.json")
                                 for s in FINAL_SEEDS}}
    if out["source_checkpoint"][SOURCE.name] != SOURCE_SHA:
        fail("source checkpoint changed")
    FREEZE.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"smoke": cmd_smoke, "cv": cmd_cv, "select": cmd_select, "final": cmd_final, "freeze": cmd_freeze}[sys.argv[1]]()
