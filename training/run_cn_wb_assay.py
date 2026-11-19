"""Paired Provia-only F24 source-WB-shift conditioning assay (Codex scope,
f707a72). Diagnostic model only: does not alter model.py defaults, the
shipped decode, Round 11, contracts or earlier evidence. No E4/CN loss.
No K / monitor / finalHeldOut / T8 access, no app integration.

Arms (identical architecture, parameter count, init, data order):
  A "source": per-scene source WB fine-tune R/9, B/9 constant planes
  B "meanctl": the same two planes filled with the TRAINING-fold mean of
               those source features, for train and validation images
Features: RAF preview MakerNote WBFineTune (tag / 20 = camera steps), /9.
Zero-shift TIFF WB gains are never read into a feature.

Model: HybridPredictor(condition_channels=3: [captureDR/2, R/9, B/9]) from a
read-only copy of r11-drcond-seed42-best.pt. grid.encoder.0.0.weight is
widened from 4 to 6 input channels with new columns zeroed (exact no-op at
init); a gradient mask keeps its original 4 columns fixed. Every other
encoder tensor frozen. Trainable: new columns + grid.head + refinement.

Fixed settings (no search): the four frozen whole-session folds, seed 42,
per-scene linear RGB L1 vs genuine Provia, batch 1, fresh Adam (0.9,
0.999, 1e-8, wd 0), LR 3e-4, cosine T_max 20 / eta_min 0.01*LR, EXACTLY 16
epochs, no early stopping. 8 runs. Each epoch-16 fold checkpoint saved.

  python3 training/run_cn_wb_assay.py prepare   # freeze manifest (refuses overwrite)
  python3 training/run_cn_wb_assay.py smoke     # refusal paths, no-op warm start, masked slices
  python3 training/run_cn_wb_assay.py run       # 2 arms x 4 folds (resumable)
  python3 training/run_cn_wb_assay.py report    # OOF vs Provia, paired A-B, every scene
"""
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
import run_cuban_step1 as S  # noqa: E402
from color_metrics import ciede2000, linear_srgb_to_lab, self_test  # noqa: E402
from cuban_exports import read_settings  # noqa: E402
from dataset import _read_with_retry  # noqa: E402
from fjlrg import load_fjlrg_linear  # noqa: E402
from model import HybridPredictor  # noqa: E402
from rank_metrics import spearman  # noqa: E402
from target_tiff import load_xraw_target_linear_matching  # noqa: E402

MANIFEST = S.DIAG / "cn-wb-assay-manifest.json"
MANIFEST_SHA = "9f82541cf618439e0cb3631576a0268b2a23602887d36695d996f72de85209d0"
REPORT = S.DIAG / "cn-wb-assay-report.json"
WB_AUDIT = S.DIAG / "cn-step1-wb-reference-audit.json"
DECOMP = S.DIAG / "cn-step1-f24-decomposition.json"
CACHE = S.CAL / "cn-wb-assay-provia-cache"
ARMS = ("source", "meanctl")
LR, EPOCHS, T_MAX, SEED = 3e-4, 16, 20, 42
FIRST = "grid.encoder.0.0.weight"
N_ORIG_IN, N_IN = 4, 6
NOOP_MAX, NOOP_MEAN = 1e-4, 1e-5  # existing native export-gate tolerances
BANDS = {"shadow": (None, 25), "mid": (25, 75), "high": (75, None)}


def ckpt_path(arm, fold):
    return S.CKPT / f"cn-wbassay-{arm}-fold{fold}.pt"


def log_path(arm, fold):
    return S.RUNS / f"cn-wbassay-{arm}-fold{fold}.json"


# ---------------------------------------------------------------- metadata
def shift_steps(tag):
    """WBFineTune tag -> (R, B) camera steps. Refuses malformed/out-of-range."""
    if not isinstance(tag, (list, tuple)) or len(tag) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in tag):
        raise ValueError(f"malformed WBFineTune {tag!r}")
    if any(v % 20 for v in tag):
        raise ValueError(f"WBFineTune {tag!r} not a multiple of 20")
    steps = [v // 20 for v in tag]
    if any(abs(s) > 9 for s in steps):
        raise ValueError(f"WBFineTune {tag!r} outside +-9 steps")
    return steps


def source_features(r, audit_row):
    raf = S.CAL / r["shoot"] / r["name"]
    if not raf.exists():
        raise ValueError(f"missing RAF {raf}")
    if S.sha256(raf) != audit_row["source_raf_sha256_at_probe"]:
        raise ValueError(f"{r['name']}: RAF hash != WB audit")
    tag = read_settings(raf).get("WBFineTune")
    if tag != audit_row["source"]["WBFineTune"]:
        raise ValueError(f"{r['name']}: WBFineTune {tag} != audit {audit_row['source']['WBFineTune']}")
    steps = shift_steps(tag)
    return tag, steps, [steps[0] / 9.0, steps[1] / 9.0]


# ---------------------------------------------------------------- prepare
def cmd_prepare():
    if MANIFEST.exists():
        S.fail("assay manifest already frozen - not overwriting")
    man, folds, allow, k = S.preflight()
    audit = {r["name"]: r for r in json.loads(WB_AUDIT.read_text())["scenes"]}
    dec = json.loads(DECOMP.read_text())["per_scene"]
    if set(audit) != set(allow) or set(dec) != set(allow):
        S.fail("WB audit / decomposition scenes != F allowlist")
    scenes = []
    for n in sorted(allow):
        r = allow[n]
        tag, steps, feat = source_features(r, audit[n])
        pv = S.CAL / r["shoot"] / "xraw-phase3.tiff"
        ph = S.sha256(pv)
        if ph != dec[n]["files"]["provia_sha256"]:
            S.fail(f"{n}: Provia hash != decomposition record")
        scenes.append({"name": n, "shoot": r["shoot"], "session": r["session"], "asShotWhiteBalance": r["asShotWhiteBalance"],
                       "fjlrg": r["fjlrg"], "fjlrg_sha256": r["fjlrg_sha256"],
                       "provia": f"{r['shoot']}/xraw-phase3.tiff", "provia_sha256": ph,
                       "raf_sha256": audit[n]["source_raf_sha256_at_probe"], "wbfinetune_tag": tag,
                       "shift_steps": steps, "feature_R_over_9_B_over_9": feat})
    by = {s["name"]: s for s in scenes}
    fold_means = {}
    for fo in folds["folds"]:
        tr = [n for n in sorted(allow) if n not in fo["validation_scenes"]]
        fold_means[str(fo["fold"])] = [float(np.mean([by[n]["feature_R_over_9_B_over_9"][i] for n in tr])) for i in (0, 1)]
    out = {"status": "FROZEN before any assay fitting (Codex scope f707a72)",
           "question": "does per-scene source WB fine-tune context improve the Provia baseline vs a matched training-fold-mean constant?",
           "not_established": "Classic Negative accuracy; causality of shift vs its correlated WB mode/session; unseen-RAW acceptance",
           "bindings": {"contract_v1": S.V1_SHA, "contract_v2": S.V2_SHA, "step1_run_manifest": S.RUNMAN_SHA,
                        "folds": S.FOLDS_SHA, "source_checkpoint": S.SOURCE_SHA, "wb_audit": S.sha256(WB_AUDIT),
                        "decomposition": S.sha256(DECOMP)},
           "arms": {"source": "per-scene [R/9, B/9]", "meanctl": "training-fold mean of [R/9, B/9] for every image in that fold"},
           "condition_vector": ["captureDR_stops/2 (=1.0, all DR400)", "R/9", "B/9"],
           "model": {"base": "HybridPredictor Round 11 config, condition_channels 3",
                     "widened": f"{FIRST} 4->6 input channels, new columns zero-initialised",
                     "masked": f"{FIRST}[:, :4] gradient-masked (fixed); bias and all other grid.encoder.* frozen",
                     "trainable": "first-conv new columns (288 coeffs) + grid.head.* + refinement.*"},
           "training": {"loss": "per-scene linear RGB L1 vs Provia", "batch": 1, "optimizer": "Adam (0.9,0.999) eps 1e-8 wd 0 fresh",
                        "lr": LR, "schedule": "cosine T_max 20 eta_min 0.01*lr", "epochs": EPOCHS, "seed": SEED,
                        "early_stopping": None, "selection": None, "data_order": "identical per fold across arms (seeded randperm)"},
           "fold_feature_means": fold_means, "scenes": scenes,
           "report": "equal-scene OOF mean dE00 vs Provia, paired A-B; both arms and untouched R11; median, p95, tone-band "
                     "signed Lab, frame-mean chroma-offset energy, every per-scene difference, session/shift groups",
           "in_sample_caveat": "R11 warm start trained on all 24 Provia references, so OOF here is adaptation-OOF only"}
    MANIFEST.write_text(json.dumps(out, indent=1))
    print(json.dumps(fold_means, indent=1))
    print("manifest sha256", S.sha256(MANIFEST))


def load_manifest():
    if MANIFEST_SHA == "__UNPINNED__" or S.sha256(MANIFEST) != MANIFEST_SHA:
        S.fail("assay manifest missing, unpinned or changed")
    S.preflight()
    return json.loads(MANIFEST.read_text())


# ---------------------------------------------------------------- data
def load_pairs(m, names):
    by = {s["name"]: s for s in m["scenes"]}
    CACHE.mkdir(exist_ok=True)
    out = {}
    for n in names:
        if n not in by:
            S.fail(f"{n} not in assay F24 manifest - refused")
        s = by[n]
        fj = S.CAL / s["fjlrg"]
        if _read_with_retry(S.sha256, fj) != s["fjlrg_sha256"]:
            S.fail(f"{n}: .fjlrg hash mismatch")
        x, w, h = _read_with_retry(load_fjlrg_linear, fj)
        cp, side = CACHE / f"{s['shoot']}-provia.npy", CACHE / f"{s['shoot']}-provia.json"
        meta = json.loads(side.read_text()) if side.exists() else None
        if meta and meta.get("provia_sha256") == s["provia_sha256"] and cp.exists() and S.sha256(cp) == meta["npy_sha256"]:
            y = _read_with_retry(np.load, cp)
        else:
            pv = S.CAL / s["provia"]
            if _read_with_retry(S.sha256, pv) != s["provia_sha256"]:
                S.fail(f"{n}: Provia hash mismatch")
            y = _read_with_retry(load_xraw_target_linear_matching, pv, w, h).astype(np.float32)
            np.save(cp, y)
            side.write_text(json.dumps({"provia_sha256": s["provia_sha256"], "npy_sha256": S.sha256(cp)}))
        if y.shape != x.shape:
            S.fail(f"{n}: shape mismatch")
        out[n] = (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0),
                  torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1))).unsqueeze(0))
    return out


def condition(m, arm, fold, name):
    by = {s["name"]: s for s in m["scenes"]}
    wb = by[name]["feature_R_over_9_B_over_9"] if arm == "source" else m["fold_feature_means"][str(fold)]
    return torch.tensor([1.0, float(wb[0]), float(wb[1])])


# ---------------------------------------------------------------- model
def build(dev):
    ck = torch.load(S.SOURCE, map_location="cpu", weights_only=False)
    c = ck["config"]
    mdl = HybridPredictor(grid_spatial=c["grid_spatial"], grid_luma=c["grid_luma"], low_res=c["low_res"],
                          base_ch=c["base_ch"], refinement_base_ch=c["refinement_base_ch"],
                          refinement_max_delta=c["refinement_max_delta"], condition_channels=3)
    sd = dict(ck["state_dict"])
    w0 = sd[FIRST]
    if w0.shape[1] != N_ORIG_IN:
        S.fail(f"unexpected source first-conv shape {tuple(w0.shape)}")
    sd[FIRST] = torch.cat([w0, torch.zeros(w0.shape[0], N_IN - N_ORIG_IN, *w0.shape[2:])], dim=1)
    mdl.load_state_dict(sd)
    req = []
    for name, p in mdl.named_parameters():
        on = name == FIRST or name.startswith(("grid.head.", "refinement."))
        if not on and not name.startswith("grid.encoder."):
            S.fail(f"unclassified parameter {name}")
        p.requires_grad_(on)
        if on:
            req.append(name)
    mask = torch.zeros_like(mdl.grid.encoder[0][0].weight)
    mask[:, N_ORIG_IN:] = 1.0
    mdl.grid.encoder[0][0].weight.register_hook(lambda g, mask=mask: g * mask.to(g.device))
    counts = {"whole_tensor_requires_grad_params": int(sum(p.numel() for p in mdl.parameters() if p.requires_grad)),
              "effective_trainable_coefficients": int(sum(p.numel() for n, p in mdl.named_parameters() if p.requires_grad and n != FIRST)
                                                      + mask.sum().item()),
              "total_params": int(sum(p.numel() for p in mdl.parameters())),
              "requires_grad_tensors": req}
    return mdl.to(dev), c, counts


def fixed_snapshot(mdl):
    sd = mdl.state_dict()
    snap = {k: v.detach().cpu().clone() for k, v in sd.items() if k.startswith("grid.encoder.") and k != FIRST}
    snap[FIRST + "[:, :4]"] = sd[FIRST][:, :N_ORIG_IN].detach().cpu().clone()
    return snap


def check_fixed(mdl, snap):
    sd = mdl.state_dict()
    for k, v in snap.items():
        cur = sd[FIRST][:, :N_ORIG_IN] if k == FIRST + "[:, :4]" else sd[k]
        if not torch.equal(cur.detach().cpu(), v):
            S.fail(f"fixed tensor {k} changed")


def fwd(mdl, c, x, cond, dev):
    out, _ = mdl.predict_and_apply(x.to(dev), c["low_res"], c.get("luma_mode", "rec709"), condition=cond.view(1, -1).to(dev))
    return out


def r11_model(dev):
    from eval_cuban_step1_k import load_model
    return load_model(S.SOURCE, dev)


# ---------------------------------------------------------------- smoke
def cmd_smoke():
    m = load_manifest()
    for bad in ([100], [100, "x"], [110, -120], [200, 0], None, [True, 0]):
        try:
            shift_steps(bad)
            S.fail(f"shift_steps accepted {bad!r}")
        except ValueError:
            pass
    assert shift_steps([180, -180]) == [9, -9]
    audit = {r["name"]: r for r in json.loads(WB_AUDIT.read_text())["scenes"]}
    s0 = m["scenes"][0]
    bad_row = {**audit[s0["name"]], "source_raf_sha256_at_probe": "0" * 64}
    try:
        source_features({"shoot": s0["shoot"], "name": s0["name"]}, bad_row)
        S.fail("RAF hash mismatch not refused")
    except ValueError:
        pass
    try:
        load_pairs(m, ["DSCF0515.RAF"])  # a K scene - must be refused before any read
    except SystemExit as e:
        if e.code != 2:
            raise
    else:
        S.fail("K scene not refused")
    print("metadata/allowlist refusal paths: ok")
    dev = S.device()
    names = [s["name"] for s in m["scenes"][:2]]
    pairs = load_pairs(m, names)
    r11, rc = r11_model(dev)
    noop = {}
    for arm in ARMS:
        mdl, c, counts = build(dev)
        mdl.eval()
        for n in names:
            with torch.no_grad():
                a = fwd(mdl, c, pairs[n][0], condition(m, arm, 0, n), dev)
                b = fwd(r11, rc, pairs[n][0], torch.tensor([1.0]), dev)
            d = (a - b).abs()
            noop[f"{arm}/{n}"] = {"max": float(d.max()), "mean": float(d.mean())}
            if d.max() > NOOP_MAX or d.mean() > NOOP_MEAN:
                S.fail(f"warm start not a no-op for {arm}/{n}: {noop[f'{arm}/{n}']}")
    print("no-op warm start:", json.dumps(noop))
    print("counts:", {k: v for k, v in counts.items() if k != "requires_grad_tensors"})
    # 2 optimizer steps; only approved slices may change
    mdl, c, counts = build(dev)
    snap = fixed_snapshot(mdl)
    before = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    opt = torch.optim.Adam([p for p in mdl.parameters() if p.requires_grad], lr=LR, betas=(0.9, 0.999), eps=1e-8)
    mdl.train()
    for n in names:
        loss = (fwd(mdl, c, pairs[n][0], condition(m, "source", 0, n), dev) - pairs[n][1].to(dev)).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    check_fixed(mdl, snap)
    after = mdl.state_dict()
    changed = sorted(k for k in before if not torch.equal(before[k], after[k].detach().cpu()))
    newcols = not torch.equal(before[FIRST][:, N_ORIG_IN:], after[FIRST][:, N_ORIG_IN:].detach().cpu())
    bad = [k for k in changed if not (k == FIRST or k.startswith(("grid.head.", "refinement.")))]
    if bad or not newcols:
        S.fail(f"unexpected changes {bad} / new columns changed={newcols}")
    S.check_source()
    print(f"masked update ok: {len(changed)} tensors changed, new first-conv columns changed, fixed slices identical; source hash unchanged")


# ---------------------------------------------------------------- run
def train_fold(m, pairs, arm, fo, dev):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    mdl, c, counts = build(dev)
    snap = fixed_snapshot(mdl)
    opt = torch.optim.Adam([p for p in mdl.parameters() if p.requires_grad], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=LR * 0.01)
    gen = torch.Generator().manual_seed(SEED)
    va = fo["validation_scenes"]
    tr = [s["name"] for s in m["scenes"] if s["name"] not in va]
    hist, orders = [], []
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        mdl.train()
        order = [tr[i] for i in torch.randperm(len(tr), generator=gen).tolist()]
        orders.append(order)
        tl = []
        for n in order:
            loss = (fwd(mdl, c, pairs[n][0], condition(m, arm, fo["fold"], n), dev) - pairs[n][1].to(dev)).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl.append(loss.item())
        lr_used = opt.param_groups[0]["lr"]
        sch.step()
        mdl.eval()
        with torch.no_grad():
            val = {n: float((fwd(mdl, c, pairs[n][0], condition(m, arm, fo["fold"], n), dev) - pairs[n][1].to(dev)).abs().mean())
                   for n in va}
        hist.append({"epoch": ep, "train_l1": float(np.mean(tl)), "lr": lr_used, "val_l1": val, "seconds": round(time.time() - t0, 1)})
        print(f"[{arm} fold {fo['fold']}] epoch {ep}/{EPOCHS} train_l1={hist[-1]['train_l1']:.5f} "
              f"val_l1={np.mean(list(val.values())):.5f} {hist[-1]['seconds']}s", flush=True)
    check_fixed(mdl, snap)
    return mdl, c, counts, hist, orders, tr


def cmd_run():
    m = load_manifest()
    folds = json.loads(S.FOLDS.read_text())["folds"]
    dev = S.device()
    pairs = load_pairs(m, [s["name"] for s in m["scenes"]])
    for fo in folds:
        for arm in ARMS:
            ck = ckpt_path(arm, fo["fold"])
            if ck.exists() and log_path(arm, fo["fold"]).exists():
                print(f"{ck.name} exists - reusing")
                continue
            S.check_source()
            mdl, c, counts, hist, orders, tr = train_fold(m, pairs, arm, fo, dev)
            S.check_source()
            meta = {"arm": arm, "fold": fo["fold"], "train": tr, "validation": fo["validation_scenes"],
                    "fold_feature_mean": m["fold_feature_means"][str(fo["fold"])], "counts": counts,
                    "settings": m["training"], "manifest_sha256": MANIFEST_SHA,
                    "source_sha256_before_after": [S.SOURCE_SHA, S.SOURCE_SHA]}
            tmp = ck.with_suffix(".pt.partial")
            torch.save({"state_dict": mdl.state_dict(), "config": {**c, "condition_channels": 3}, "cn_wb_assay": meta}, tmp)
            tmp.rename(ck)
            log_path(arm, fo["fold"]).write_text(json.dumps({**meta, "checkpoint_sha256": S.sha256(ck),
                                                             "history": hist, "orders": orders}, indent=1))
    # paired data order check
    for fo in folds:
        a, b = (json.loads(log_path(arm, fo["fold"]).read_text())["orders"] for arm in ARMS)
        if a != b:
            S.fail(f"fold {fo['fold']}: data order differs between arms")
    print("all 8 runs done; paired data orders identical")


# ---------------------------------------------------------------- report
def lab_metrics(app, ref):
    e = ciede2000(app, ref)
    d = app - ref
    L = ref[..., 0]
    bands = {b: ((L >= lo) if lo is not None else True) & ((L < hi) if hi is not None else True) for b, (lo, hi) in BANDS.items()}
    fm = d.reshape(-1, 3).mean(0)
    return {"mean_dE00": float(e.mean()), "p95_dE00": float(np.percentile(e, 95)),
            "frame_mean_Lab": [float(v) for v in fm], "frame_mean_chroma_offset_energy": float(fm[1] ** 2 + fm[2] ** 2),
            "tone_band_signed_Lab": {b: ([float(v) for v in d[mk].mean(0)] if mk.any() else None) for b, mk in bands.items()},
            "tone_band_occupancy": {b: float(mk.mean()) for b, mk in bands.items()}}


def cmd_report():
    if REPORT.exists():
        S.fail("report exists - not overwriting")
    self_test()
    m = load_manifest()
    folds = json.loads(S.FOLDS.read_text())["folds"]
    dev = S.device()
    r11, rc = r11_model(dev)
    by = {s["name"]: s for s in m["scenes"]}
    per, ck_hashes = {}, {}
    for fo in folds:
        models = {}
        for arm in ARMS:
            lg = json.loads(log_path(arm, fo["fold"]).read_text())
            h = S.sha256(ckpt_path(arm, fo["fold"]))
            if h != lg["checkpoint_sha256"]:
                S.fail(f"{ckpt_path(arm, fo['fold']).name} != logged hash")
            ck_hashes[ckpt_path(arm, fo["fold"]).name] = h
            mdl, c, _ = build(dev)
            ckd = torch.load(ckpt_path(arm, fo["fold"]), map_location="cpu", weights_only=False)
            mdl.load_state_dict(ckd["state_dict"])
            models[arm] = (mdl.eval(), c)
        pairs = load_pairs(m, fo["validation_scenes"])
        for n in fo["validation_scenes"]:
            x, y = pairs[n]
            ref = linear_srgb_to_lab(y.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64))
            row = {"fold": fo["fold"], "session": by[n]["session"], "asShotWhiteBalance": by[n]["asShotWhiteBalance"],
                   "shift_steps": by[n]["shift_steps"]}
            with torch.no_grad():
                outs = {"r11_untouched": fwd(r11, rc, x, torch.tensor([1.0]), dev)}
                for arm in ARMS:
                    outs[arm] = fwd(*models[arm], x, condition(m, arm, fo["fold"], n), dev)
            for k, o in outs.items():
                row[k] = lab_metrics(linear_srgb_to_lab(o.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)), ref)
            row["A_minus_B_mean_dE00"] = row["source"]["mean_dE00"] - row["meanctl"]["mean_dE00"]
            row["A_minus_B_p95_dE00"] = row["source"]["p95_dE00"] - row["meanctl"]["p95_dE00"]
            row["A_minus_R11_mean_dE00"] = row["source"]["mean_dE00"] - row["r11_untouched"]["mean_dE00"]
            row["B_minus_R11_mean_dE00"] = row["meanctl"]["mean_dE00"] - row["r11_untouched"]["mean_dE00"]
            per[n] = row
            print(n, f"R11 {row['r11_untouched']['mean_dE00']:.3f}  A {row['source']['mean_dE00']:.3f}  "
                     f"B {row['meanctl']['mean_dE00']:.3f}  A-B {row['A_minus_B_mean_dE00']:+.3f}", flush=True)
    names = sorted(per)
    def agg(key):
        v = [per[n][key]["mean_dE00"] for n in names]
        return {"mean_of_scene_means": float(np.mean(v)), "median": float(np.median(v)),
                "mean_p95": float(np.mean([per[n][key]["p95_dE00"] for n in names])),
                "mean_chroma_offset_energy": float(np.mean([per[n][key]["frame_mean_chroma_offset_energy"] for n in names])),
                "cross_scene_mean_tone_band_signed_Lab": {b: [float(np.mean([per[n][key]["tone_band_signed_Lab"][b][i] for n in names
                                                                              if per[n][key]["tone_band_signed_Lab"][b]]))
                                                              for i in range(3)] for b in BANDS}}
    diffs = np.array([per[n]["A_minus_B_mean_dE00"] for n in names])
    rng = np.random.default_rng(0)
    boot = [rng.choice(diffs, len(diffs)).mean() for _ in range(10000)]
    groups = {}
    for gk in ("session", "shift_steps"):
        g = {}
        for n in names:
            g.setdefault(str(per[n][gk]), []).append(n)
        groups[gk] = {kk: {"n": len(v), "scenes": v, "mean_A_minus_B": float(np.mean([per[s]["A_minus_B_mean_dE00"] for s in v])),
                           "mean_R11": float(np.mean([per[s]["r11_untouched"]["mean_dE00"] for s in v])),
                           "mean_A": float(np.mean([per[s]["source"]["mean_dE00"] for s in v])),
                           "mean_B": float(np.mean([per[s]["meanctl"]["mean_dE00"] for s in v]))} for kk, v in g.items()}
    out = {"scope": "Provia-only F24 paired WB-context assay; adaptation-OOF (R11 warm start saw these Provia scenes); "
                    "diagnostic only - no selection, ship or pass declaration; says nothing about Classic Negative accuracy",
           "manifest_sha256": MANIFEST_SHA, "checkpoint_sha256": ck_hashes, "source_checkpoint_sha256": S.sha256(S.SOURCE),
           "primary_paired_A_minus_B_mean_dE00": {"mean": float(diffs.mean()), "median": float(np.median(diffs)),
                                                   "bootstrap95_scene_resample": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                                   "A_better_scenes": int((diffs < 0).sum()), "B_better_scenes": int((diffs > 0).sum()),
                                                   "note": "bootstrap ignores session clustering; descriptive"},
           "arms": {"r11_untouched": agg("r11_untouched"), "source": agg("source"), "meanctl": agg("meanctl")},
           "regressions_vs_r11": {"source": [n for n in names if per[n]["A_minus_R11_mean_dE00"] > 0],
                                  "meanctl": [n for n in names if per[n]["B_minus_R11_mean_dE00"] > 0]},
           "groups": groups, "per_scene": per}
    S.check_source()
    REPORT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("primary_paired_A_minus_B_mean_dE00", "regressions_vs_r11")}, indent=1))
    print(json.dumps({a: {k: v for k, v in d.items() if k != "cross_scene_mean_tone_band_signed_Lab"} for a, d in out["arms"].items()}, indent=1))


if __name__ == "__main__":
    {"prepare": cmd_prepare, "smoke": cmd_smoke, "run": cmd_run, "report": cmd_report}[sys.argv[1]]()
