"""Classic Cuban Neg step 1 (E4 simulation base) - FREEZE the session-grouped
CV folds and the run manifest BEFORE any training (Codex-approved design,
docs/phase4-scene-adaptive-scope.md 2026-11-18 / 2026-11-19).

Metadata + file hashes only: reads no pixels, trains nothing, scores nothing.
Monitor and finalHeldOut have no code path here; K scenes are listed only
as the separate frozen evaluation set (never a training input).

Fold rule (fixed before running): the 24 F scenes are grouped by whole
capture session (sessions never split). Enumerate every assignment of the
sessions to 4 non-empty folds; choose the one minimising (1) the fold-size
range (max - min), then (2) the summed squared deviation of each fold's
Auto-WB fraction from the F-wide Auto fraction, then (3) the lexically
smallest sorted tuple of session lists. Rare WB modes confined to one
session (White Fluorescent / Cloudy: session 3 only) cannot be stratified.

  python3 training/cuban_step1_prepare.py
    -> training/diagnostics/cuban-step1-folds.json
    -> training/diagnostics/cuban-step1-run-manifest.json
Refuses to overwrite either file (they are frozen once written).
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuban_contract import contract_sha256  # noqa: E402
from cuban_contract_v2 import V1_SHA256, contract_v2_sha256, _check as v2_check  # noqa: E402
from cuban_exports import CAL, INSTALL_NAME, REPORT, sha256  # noqa: E402

TRAIN = Path(__file__).resolve().parent
DIAG = TRAIN / "diagnostics"
FOLDS_OUT = DIAG / "cuban-step1-folds.json"
RUN_OUT = DIAG / "cuban-step1-run-manifest.json"
V2_SHA256 = "9824b0799737960363c95bb028332fdd0e5118e748fc0139375f1d0e230099a0"
SOURCE_CKPT = TRAIN / "checkpoints" / "r11-drcond-seed42-best.pt"
SOURCE_CKPT_SHA256 = "1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c"
E4 = "E4-classic-neg-only"
FJLRG = "browser-phase3-linear.fjlrg"
N_FOLDS = 4
EXPECTED_F, EXPECTED_K = 24, 8

TRAINABLE_PREFIXES = ["grid.head.", "refinement."]
FROZEN_PREFIXES = ["grid.encoder."]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def verified_e4_hashes():
    """(name -> E4 sha256) from every passed export-check report that holds E4."""
    out = {}
    for kind in ("pilot", "cohort1", "base-step1"):
        rep = json.loads(REPORT[kind].read_text())
        if rep.get("passed") is not True:
            sys.exit(f"refused: {kind} verification report has not passed")
        for e in rep["entries"]:
            if e["variant"] != E4:
                continue
            if e["name"] in out and out[e["name"]] != e["source_sha256"]:
                sys.exit(f"refused: conflicting verified E4 hashes for {e['name']}")
            out[e["name"]] = e["source_sha256"]
    return out


def choose_folds(fit):
    sessions = sorted({r["session"] for r in fit})
    size = {s: sum(r["session"] == s for r in fit) for s in sessions}
    auto = {s: sum(r["session"] == s and r["asShotWhiteBalance"] == "0x0" for r in fit) for s in sessions}
    p_auto = sum(auto.values()) / len(fit)
    best = None
    for assign in itertools.product(range(N_FOLDS), repeat=len(sessions)):
        if len(set(assign)) != N_FOLDS:
            continue
        folds = [sorted(s for s, a in zip(sessions, assign) if a == f) for f in range(N_FOLDS)]
        folds = sorted(folds)  # canonical (label-free) form
        sizes = [sum(size[s] for s in fo) for fo in folds]
        wb = sum((sum(auto[s] for s in fo) / n - p_auto) ** 2 for fo, n in zip(folds, sizes))
        key = (max(sizes) - min(sizes), round(wb, 12), tuple(tuple(fo) for fo in folds))
        if best is None or key < best[0]:
            best = (key, folds)
    return best[1], size, auto, p_auto


def main():
    for p in (FOLDS_OUT, RUN_OUT):
        if p.exists():
            sys.exit(f"refused: {p.name} already frozen - not overwriting")
    v2_check()
    if contract_sha256() != V1_SHA256 or contract_v2_sha256() != V2_SHA256:
        sys.exit("refused: contract v1/v2 fingerprint mismatch")
    if sha256(SOURCE_CKPT) != SOURCE_CKPT_SHA256:
        sys.exit("refused: Round 11 seed 42 source checkpoint hash changed")
    prop = CAL / "cuban-cohort-proposal.json"
    committed = DIAG / "cuban-cohort-proposal.json"
    if prop.read_bytes() != committed.read_bytes():
        sys.exit("refused: calibration-input cohort proposal != committed copy")
    cohort = json.loads(prop.read_text())["scenes"]
    fit = sorted((r for r in cohort if r["role"] == "fit"), key=lambda r: r["name"])
    chk = sorted((r for r in cohort if r["role"] == "train-check"), key=lambda r: r["name"])
    if len(fit) != EXPECTED_F or len(chk) != EXPECTED_K:
        sys.exit(f"refused: F={len(fit)} K={len(chk)}")
    if {r["shoot"] for r in fit} & {r["shoot"] for r in chk}:
        sys.exit("refused: F/K share a shoot")
    # base-step1 manifest roles must agree with the cohort roles
    bm = json.loads((CAL / "cuban-base-step1-manifest.json").read_text())
    role = {r["name"]: r["role"] for r in cohort}
    for r in bm:
        if role.get(r["name"]) != r["role"]:
            sys.exit(f"refused: base-step1 role mismatch for {r['name']}")
    # monitor / finalHeldOut names must not appear in F or K (names only, from the manifest's tier keys)
    man = json.loads((CAL / "phase4-corpus-manifest.json").read_text())
    train_names = {s["name"] for s in man["tiers"]["train"]["scenes"]}
    for r in fit + chk:
        if r["name"] not in train_names:
            sys.exit(f"refused: {r['name']} is not a train-tier scene")

    e4 = verified_e4_hashes()
    def files(r):
        d = CAL / r["shoot"]
        t = d / INSTALL_NAME[E4]
        if r["name"] not in e4:
            sys.exit(f"refused: no verified E4 report entry for {r['name']}")
        th = sha256(t)
        if th != e4[r["name"]]:
            sys.exit(f"refused: {t.relative_to(CAL)} != verified hash")
        out = {"fjlrg": f"{r['shoot']}/{FJLRG}", "fjlrg_sha256": sha256(d / FJLRG),
               "e4": f"{r['shoot']}/{INSTALL_NAME[E4]}", "e4_sha256": th}
        return out

    fit_rows = [{k: r[k] for k in ("name", "shoot", "session", "iso", "asShotWhiteBalance", "highlightClass")} | files(r)
                for r in fit]
    folds, size, auto, p_auto = choose_folds(fit)
    fold_rows = []
    for i, fo in enumerate(folds):
        names = [r["name"] for r in fit_rows if r["session"] in fo]
        fold_rows.append({"fold": i, "sessions": fo, "n": len(names),
                          "auto_wb": sum(auto[s] for s in fo), "validation_scenes": names})
    folds_doc = {
        "status": "FROZEN before any step-1 training",
        "rule": __doc__.split("Fold rule (fixed before running): ")[1].split("\n\n")[0].replace("\n", " "),
        "session_sizes": {str(s): n for s, n in size.items()},
        "session_auto_wb": {str(s): n for s, n in auto.items()},
        "f_auto_wb_fraction": round(p_auto, 4),
        "folds": fold_rows,
        "unstratifiable": "White Fluorescent (DSCF0514) and Cloudy (DSCF0511) occur only in session 3",
    }
    FOLDS_OUT.write_text(json.dumps(folds_doc, indent=1))
    folds_sha = sha256(FOLDS_OUT)

    k_rows = [{k: r[k] for k in ("name", "shoot", "session", "iso", "asShotWhiteBalance", "highlightClass")} | files(r)
              for r in chk]
    run = {
        "status": "FROZEN before any step-1 training (Codex-approved design 2026-11-18)",
        "contract": {"v1_sha256": V1_SHA256, "v2_sha256": V2_SHA256},
        "cohort_proposal_sha256": sha256(prop),
        "base_step1_manifest_sha256": sha256(CAL / "cuban-base-step1-manifest.json"),
        "folds_sha256": folds_sha,
        "source_checkpoint": {"path": "training/checkpoints/" + SOURCE_CKPT.name, "sha256": SOURCE_CKPT_SHA256,
                              "rule": "loaded read-only into a copy; hash re-checked before and after every run"},
        "architecture": "HybridPredictor, Round 11 config read from the source checkpoint (grid 8x9, base_ch 16, "
                        "refinement base_ch 8, max_delta 0.08, capture-DR condition plane; all F/K are capture DR400 -> 1.0)",
        "trainable_prefixes": TRAINABLE_PREFIXES, "frozen_prefixes": FROZEN_PREFIXES,
        "loss": "per-scene linear-RGB L1 (mean over pixels), batch 1 - Round 11 loss",
        "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
                      "params": "trainable only", "state": "fresh (no Round 11 optimizer state)"},
        "schedule": {"name": "CosineAnnealingLR", "T_max": 20, "eta_min": "0.01*lr", "step": "per epoch",
                     "horizon_fixed": "T_max=20 for CV and final fits; final fits stop at the chosen epoch, no rescaling"},
        "search": {"lr": [1e-4, 3e-4], "max_epochs": 20, "cv_seed": 42,
                   "selection": "minimise mean out-of-fold per-scene L1 over all 24 F scenes, epochs 1..20, each "
                                "scene scored by the fold model that did not train on it",
                   "tie_break": "fewer epochs, then smaller lr",
                   "epoch0": "untrained warm-start OOF L1 reported for context only; not selectable"},
        "final_fits": {"seeds": [42, 43, 44], "primary": 42, "data": "all 24 F",
                       "note": "seeds change only data order (and MPS nondeterminism): the weights start from the "
                               "same copied checkpoint; 43/44 report-only, never selected"},
        "no_capacity_loss_reg_search": True,
        "F_allowlist": fit_rows,
        "K_eval_only": k_rows,
        "excluded": "monitor (incl. 12 Cuban monitor scenes), finalHeldOut, T8 support, DSCF0752 - no code path",
        "k_evaluation": {
            "when": "once, after the final checkpoints are frozen (hash manifest committed) - E4-only",
            "model_used_for_decision": "seed 42 only",
            "per_scene_E4": "contract v1/v2 stage_limits.E4: mean_dE00<=2.0, p95_dE00<=5.0, |mid dL*|<=2.0, "
                            "rho_provia_to_E4<=0.5 (app_b = untouched Round 11 seed 42 output, ref_b = Provia "
                            "xraw-phase3, ref_s = E4), all must hold",
            "set": ">=7 of 8 K scenes pass the per-scene E4 conjunction",
            "aggregates": "median mean_dE00 <= 1.5; |cross-scene mean mid dL*| <= 1.0; |mean dC*| <= 1.5; "
                          "|mean hue| <= 3.0 deg",
            "label": "internal simulation-adaptation check (K shares sessions with F; Round 11 trained on their "
                     "Provia targets) - not unseen-RAW acceptance, not a whole-recipe pass",
            "baselines_context_only": "untouched Round 11 seed 42 vs E4; seed42+app LUT baseline not run (would need "
                                      "app-pipeline reimplementation - out of scope tonight)",
            "visual_review": "pending user review; not part of the automated result",
        },
    }
    RUN_OUT.write_text(json.dumps(run, indent=1))
    print(json.dumps(folds_doc["folds"], indent=1))
    print("folds sha256", folds_sha, " run manifest sha256", sha256(RUN_OUT))


if __name__ == "__main__":
    main()
