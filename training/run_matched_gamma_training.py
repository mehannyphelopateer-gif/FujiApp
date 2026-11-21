"""Matched gamma-only Provia training: legacy v1 input vs linear G input, 3 paired seeds, fixed endpoint.

Same Round 11 Hybrid configuration (built by matched_gamma_data.initialise), trained from FRESH seeded
weights - Round 11 weights are never loaded, warm-started or written - on the 467 fit scenes only. The
116 development scenes (session-disjoint; they hold all 12 DR100 scenes, which are outside fit support
and therefore NOT validated) are never trained on. Adam lr 1e-3, betas (0.9, 0.999), eps 1e-8, wd 0;
cosine T_max = 20 epochs to eta_min 1e-5, stepped once per epoch as in train.py; linear L1, batch 1; no
augmentation, AMP, TV, gradient clipping, early stopping or selection. The epoch-20 state is the only
endpoint. Per-seed scene orders are the frozen lists in the manifest (20 x 467 names).

  smoke --manifest-sha SHA                    allowlist refusals, CPU exact-resume fixture, finite
                                              full-resolution updates on 2 fit pairs per arm; diagnostic proof only
  train --manifest-sha SHA --seed S --arm A   one arm; an exact, validated same-manifest resume is allowed
  all   --manifest-sha SHA                    all six arms sequentially, evaluating after each endpoint

"TRAINING_STARTED" is printed only after the first real optimizer step of an arm (never at preflight,
never by smoke or the fixture). Progress every 25 updates and at each epoch end; checkpoint every 50
updates and at each epoch end; status.json per arm. Any non-finite loss, gradient or parameter is a hard
failure with no automatic continue.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

import matched_gamma_data as D

ARMS = ("legacy", "linear")
FROZEN_TRAINING = {"seeds": [42, 43, 44], "epochs": 20, "lr": 0.001, "eta_min": 1e-5, "batch_size": 1}
BETAS, EPS, WEIGHT_DECAY = (0.9, 0.999), 1e-8, 0.0
LOG_EVERY, CKPT_EVERY = 25, 50
FIT_N, DEV_N, DR100_DEV_N = 467, 116, 12
STATE_FORMAT, ENDPOINT_FORMAT = "matched-gamma-state-v1", "matched-gamma-endpoint-v1"
CODE = Path(__file__).resolve()


def log(msg):
    print(msg, flush=True)


def _quiet(_msg):
    pass


# ------------------------------------------------------------------ paths and atomic writes
def run_dir(seed, arm):
    return D.CKPT / f"matched-gamma-seed{seed}-{arm}"


def state_path(seed, arm):
    return run_dir(seed, arm) / "state.pt"


def endpoint_path(seed, arm):
    return run_dir(seed, arm) / "endpoint.pt"


def endpoint_record_path(seed, arm):
    return run_dir(seed, arm) / "endpoint.json"


def status_path(seed, arm):
    return run_dir(seed, arm) / "status.json"


def lock_path():
    return D.CKPT / "matched-gamma.lock"


def _rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(Path(D.ROOT).resolve()))
    except ValueError:
        return str(p)


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _refused(fn):
    """True iff fn() is refused (SystemExit from D.need or any exception). KeyboardInterrupt propagates."""
    try:
        fn()
    except (SystemExit, Exception):
        return True
    return False


_LOCK_HELD = False


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def global_lock():
    """One matched-gamma training process at a time (no concurrent arms). Re-entrant within a process."""
    global _LOCK_HELD
    if _LOCK_HELD:
        yield
        return
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text().strip())
            except (OSError, ValueError):
                pid = None
            D.need(pid is not None and not _alive(pid), f"Another matched-gamma run holds {path} (pid {pid}); refusing")
            path.unlink()  # own lock file left by a dead process
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        break
    else:
        D.need(False, f"Could not acquire {path}")
    _LOCK_HELD = True
    try:
        yield
    finally:
        _LOCK_HELD = False
        path.unlink(missing_ok=True)


# ------------------------------------------------------------------ RNG
def _mps():
    return hasattr(torch, "mps") and torch.backends.mps.is_available()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _mps():
        torch.mps.manual_seed(seed)


def rng_state():
    """Plain lists/tensors only, so checkpoints load with weights_only=True."""
    py, npst = random.getstate(), np.random.get_state()
    s = {"python": [py[0], list(py[1]), py[2]],
         "numpy": [npst[0], npst[1].tolist(), int(npst[2]), int(npst[3]), float(npst[4])],
         "torch_cpu": torch.get_rng_state()}
    if _mps():
        s["torch_mps"] = torch.mps.get_rng_state()
    return s


def set_rng_state(s):
    random.setstate((s["python"][0], tuple(s["python"][1]), s["python"][2]))
    n = s["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(s["torch_cpu"])
    if "torch_mps" in s:
        D.need(_mps(), "Checkpoint carries MPS RNG state but MPS is unavailable; refusing an inexact resume")
        torch.mps.set_rng_state(s["torch_mps"])


# ------------------------------------------------------------------ manifest checks
def check_r11():
    D.need(D.sha(D.R11_PATH) == D.R11_SHA, "Round 11 selected checkpoint hash changed")


def check_training_config(m):
    t = m["training"]
    for k, v in FROZEN_TRAINING.items():
        D.need(t.get(k) == v, f"Manifest training.{k} = {t.get(k)!r}, frozen design is {v!r}")


def validate_split(m):
    """Returns fit names in manifest order. Split/session/DR facts are frozen; nothing is changed here."""
    fit = [s for s in m["scenes"] if s["role"] == "fit"]
    dev = [s for s in m["scenes"] if s["role"] == "development"]
    D.need(len(fit) == FIT_N and len(dev) == DEV_N and len(m["scenes"]) == FIT_N + DEV_N,
           f"Split is {len(fit)} fit / {len(dev)} development / {len(m['scenes'])} total, expected {FIT_N}/{DEV_N}")
    names = [s["name"] for s in m["scenes"]]
    D.need(len(set(names)) == len(names), "Duplicate scene names")
    D.need(not ({s["session"] for s in fit} & {s["session"] for s in dev}), "Fit and development sessions overlap")
    D.need(not any(s["captureDynamicRange"] == 100 for s in fit), "A DR100 scene is in fit")
    D.need(sum(s["captureDynamicRange"] == 100 for s in dev) == DR100_DEV_N, "Development DR100 count differs from 12")
    return [s["name"] for s in fit]


def order_digest(orders):
    return hashlib.sha256(json.dumps(orders, separators=(",", ":")).encode()).hexdigest()


def validate_orders(orders, names, epochs):
    D.need(isinstance(orders, list) and len(orders) == epochs, f"Need {epochs} epoch orders")
    want = sorted(names)
    for e, o in enumerate(orders):
        D.need(isinstance(o, list) and len(o) == len(names) and sorted(o) == want,
               f"Epoch {e + 1} order is not a permutation of the fit allowlist")


def identity_for(m, seed, arm, orders):
    """Everything a resume or endpoint must match exactly. Never inferred from a filename."""
    t = m["training"]
    return {"manifest_sha256": m["_manifest_sha256"], "seed": seed, "arm": arm,
            "initial_state_sha256": m["initial_state_sha256"][str(seed)], "order_sha256": order_digest(orders),
            "epochs": t["epochs"], "lr": t["lr"], "eta_min": t["eta_min"], "batch_size": t["batch_size"],
            "betas": list(BETAS), "eps": EPS, "weight_decay": WEIGHT_DECAY, "loss": "linear_l1_mean",
            "fit_count": len(orders[0]), "trainer_sha256": D.sha(CODE)}


# ------------------------------------------------------------------ training primitives
def make_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)


def make_scheduler(opt, epochs, eta_min):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)


def update(model, opt, x, y, cond, forward):
    """One batch-1 linear-L1 Adam step. Non-finite loss/gradient/parameter is a hard failure."""
    pred = forward(model, x, cond)
    loss = (pred - y.to(pred.device)).abs().mean()
    D.need(bool(torch.isfinite(loss)), f"Non-finite loss {float(loss.detach())}")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    D.need(len(grads) > 0, "No gradients")
    D.need(bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()), "Non-finite gradient")
    opt.step()
    D.need(bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()), "Non-finite parameter after step")
    return float(loss.detach())


def _checkpoint(model, opt, sched, progress, identity, path):
    atomic_torch_save({"format": STATE_FORMAT, "identity": identity, "progress": json.loads(json.dumps(progress)),
                       "model_state_sha256": D.state_digest(model.state_dict()),
                       "model": _to_cpu(model.state_dict()), "optimizer": _to_cpu(opt.state_dict()),
                       "scheduler": sched.state_dict(), "rng": rng_state()}, path)


def run(model, opt, sched, orders, get_pair, forward, path, identity, *, ckpt_every=CKPT_EVERY,
        log_every=LOG_EVERY, max_updates=None, status=None, announce=True, tag="", say=log):
    """Train through `orders` (list of per-epoch name lists) with exact checkpoint/resume at `path`.

    A state file at `path` is resumed only if its format and full identity match; anything else is refused.
    `max_updates` stops after that many updates in this call WITHOUT saving (a simulated crash, for tests).
    Returns {"complete": bool, "progress": dict}.
    """
    path = Path(path)
    epochs, n = len(orders), len(orders[0])
    total = epochs * n
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        D.need(ck.get("format") == STATE_FORMAT and ck.get("identity") == identity,
               f"Refusing unknown or stale state {path}: identity differs from this run")
        D.need(D.state_digest(ck["model"]) == ck.get("model_state_sha256"),
               "Resume model weights differ from saved digest")
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        set_rng_state(ck["rng"])
        pr = ck["progress"]
        D.need(0 <= pr["epoch"] <= epochs and 0 <= pr["next_index"] < n
               and pr["global_updates"] == pr["epoch"] * n + pr["next_index"]
               and pr["epoch_count"] == pr["next_index"] and len(pr["history"]) == pr["epoch"],
               f"Inconsistent progress in {path}")
        say(f"[{tag}] TRAINING_RESUMED at epoch {pr['epoch'] + 1}/{epochs} index {pr['next_index']} "
            f"(global {pr['global_updates']}/{total})")
        if status:
            status("training", pr, total, resumed=True)
    else:
        pr = {"epoch": 0, "next_index": 0, "global_updates": 0, "epoch_loss_sum": 0.0, "epoch_count": 0, "history": []}
    D.need(sched.last_epoch == pr["epoch"], "Scheduler position does not match progress")
    model.train()
    done_here, t0 = 0, time.time()
    while pr["epoch"] < epochs:
        e = pr["epoch"]
        lr = opt.param_groups[0]["lr"]
        while pr["next_index"] < n:
            if max_updates is not None and done_here >= max_updates:
                return {"complete": False, "progress": pr}
            x, y, cond = get_pair(orders[e][pr["next_index"]])
            loss = update(model, opt, x, y, cond, forward)
            pr["next_index"] += 1
            pr["global_updates"] += 1
            pr["epoch_loss_sum"] += loss
            pr["epoch_count"] += 1
            done_here += 1
            g = pr["global_updates"]
            if g == 1:
                if announce:
                    say(f"[{tag}] TRAINING_STARTED first optimizer step done: loss {loss:.6f} lr {lr:.3e}")
                if status:
                    status("training", pr, total)
            if g % log_every == 0:
                rate = done_here / max(time.time() - t0, 1e-9)
                say(f"[{tag}] epoch {e + 1}/{epochs} update {pr['next_index']}/{n} global {g}/{total} "
                    f"loss {loss:.6f} epoch-mean {pr['epoch_loss_sum'] / pr['epoch_count']:.6f} lr {lr:.3e} "
                    f"{rate:.2f} upd/s eta {(total - g) / rate / 3600:.2f} h")
            if g % ckpt_every == 0 and pr["next_index"] < n:
                _checkpoint(model, opt, sched, pr, identity, path)
                if status:
                    status("training", pr, total)
        sched.step()
        pr["history"].append({"epoch": e + 1, "mean_linear_l1": pr["epoch_loss_sum"] / pr["epoch_count"],
                              "updates": pr["epoch_count"], "lr": lr, "next_lr": opt.param_groups[0]["lr"]})
        pr["epoch"] += 1
        pr["next_index"], pr["epoch_loss_sum"], pr["epoch_count"] = 0, 0.0, 0
        _checkpoint(model, opt, sched, pr, identity, path)
        say(f"[{tag}] EPOCH {e + 1}/{epochs} done: mean linear L1 {pr['history'][-1]['mean_linear_l1']:.6f} "
            f"lr {lr:.3e} -> {opt.param_groups[0]['lr']:.3e} ({time.time() - t0:.0f} s this process)")
        if status:
            status("training", pr, total)
    return {"complete": True, "progress": pr}


# ------------------------------------------------------------------ status and endpoint
def status_writer(seed, arm, identity):
    path = status_path(seed, arm)

    def write(state, pr=None, total=None, **extra):
        obj = {"seed": seed, "arm": arm, "state": state, "manifest_sha256": identity["manifest_sha256"],
               "pid": os.getpid(), "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
        if pr is not None:
            obj.update(training_started=pr["global_updates"] > 0, epochs_completed=pr["epoch"],
                       next_index=pr["next_index"], global_updates=pr["global_updates"], total_updates=total,
                       last_epoch_mean_linear_l1=pr["history"][-1]["mean_linear_l1"] if pr["history"] else None)
        atomic_json(path, obj)
    return write


def write_endpoint(model, identity, pr, seed, arm):
    ep, rec = endpoint_path(seed, arm), endpoint_record_path(seed, arm)
    D.need(not ep.exists() and not rec.exists(), f"Endpoint already exists in {run_dir(seed, arm)}; refusing to overwrite")
    D.need(pr["epoch"] == identity["epochs"] and pr["global_updates"] == identity["epochs"] * identity["fit_count"],
           "Run is not at the fixed epoch-20 endpoint")
    sd = _to_cpu(model.state_dict())
    meta = dict(identity, epoch=identity["epochs"], complete=True, global_updates=pr["global_updates"],
                history=pr["history"], state_sha256=D.state_digest(sd))
    atomic_torch_save({"format": ENDPOINT_FORMAT, "model": sd, "meta": meta}, ep)
    D.write_once_json(rec, {"endpoint": _rel(ep), "endpoint_sha256": D.sha(ep),
                            "meta": {k: v for k, v in meta.items() if k != "history"},
                            "note": "Fixed epoch-20 endpoint; no selection. Not a best checkpoint."})
    return ep


def validate_endpoint(m, seed, arm, checkpoint_path=None):
    """Validate a complete endpoint for (manifest, seed, arm). Returns (path, checkpoint dict, sha256)."""
    ep, rec = endpoint_path(seed, arm), endpoint_record_path(seed, arm)
    if checkpoint_path is not None:
        D.need(Path(checkpoint_path).resolve() == ep.resolve(), f"{checkpoint_path} is not the seed{seed}-{arm} endpoint")
    D.need(ep.exists() and rec.exists(), f"seed{seed}-{arm}: no complete endpoint; refusing to use an incomplete run")
    r = json.loads(rec.read_text())
    h = D.sha(ep)
    D.need(r.get("endpoint_sha256") == h, f"{ep} hash differs from its endpoint record")
    ck = torch.load(ep, map_location="cpu", weights_only=True)
    meta = ck.get("meta", {})
    want = identity_for(m, seed, arm, m["orders"][str(seed)])
    D.need(ck.get("format") == ENDPOINT_FORMAT and {k: meta.get(k) for k in want} == want,
           f"{ep}: identity differs (manifest/seed/arm/initial hash/order/config/trainer)")
    D.need(meta.get("complete") is True and meta.get("epoch") == want["epochs"]
           and meta.get("global_updates") == want["epochs"] * want["fit_count"], f"{ep} is not a complete epoch-20 run")
    D.need(D.state_digest(ck["model"]) == meta["state_sha256"] == r["meta"]["state_sha256"], f"{ep} weights differ from recorded digest")
    return ep, ck, h


def train_arm(m, seed, arm):
    D.need(arm in ARMS and seed in m["training"]["seeds"], f"Unknown arm/seed {arm}/{seed}")
    check_training_config(m)
    for filename in ("matched-gamma-training-preflight.json", "matched-gamma-training-smoke.json"):
        proof = json.loads((D.DIAG / filename).read_text())
        D.need(proof.get("passed") is True and proof.get("manifest_sha256") == m["_manifest_sha256"],
               "Missing or unrelated preflight proof: " + filename)
        if "preflight" in filename:
            D.need(proof.get("pairs") == 583, "Full-pair preflight is short")
        else:
            D.need(proof.get("trainer_sha256") == D.sha(CODE)
                   and proof["resume"]["equal"] is True and proof["resume"]["stale_refused"] is True,
                   "Resume/smoke proof differs from the frozen trainer")
    names = validate_split(m)
    t = m["training"]
    orders = m["orders"][str(seed)]
    validate_orders(orders, names, t["epochs"])
    identity = identity_for(m, seed, arm, orders)
    tag = f"seed{seed} {arm}"
    if endpoint_path(seed, arm).exists() or endpoint_record_path(seed, arm).exists():
        ep, _, h = validate_endpoint(m, seed, arm)
        log(f"[{tag}] endpoint already complete and valid ({h[:12]}); not retraining")
        return ep
    with global_lock():
        check_r11()
        st = status_writer(seed, arm, identity)
        try:
            dev = D.device()
            model = D.initialise(m, seed, dev)  # fresh seeded weights; validates the frozen initial-state hash
            seed_everything(seed)
            opt = make_optimizer(model, t["lr"])
            sched = make_scheduler(opt, t["epochs"], t["eta_min"])
            pairs = D.Pairs(m, arm, role="fit")
            log(f"[{tag}] preflight ok on {dev}; {t['epochs']} epochs x {len(names)} fit scenes; not yet training")
            res = run(model, opt, sched, orders, pairs.get, lambda mdl, x, c: D.forward(mdl, m, x, c, dev),
                      state_path(seed, arm), identity, status=st, tag=tag)
            D.need(res["complete"], "Run did not complete")
            ep = write_endpoint(model, identity, res["progress"], seed, arm)
            check_r11()
            st("complete", res["progress"], t["epochs"] * len(names), endpoint=_rel(ep), endpoint_sha256=D.sha(ep))
            log(f"[{tag}] ENDPOINT_WRITTEN {_rel(ep)}")
            return ep
        except BaseException as exc:
            st("failed", error=f"{type(exc).__name__}: {exc}")
            raise


# ------------------------------------------------------------------ CPU fixture (no real data)
class _Tiny(torch.nn.Module):
    """Conditioned 1x1 conv with dropout, so RNG restoration is exercised."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 1)
        self.drop = torch.nn.Dropout(0.25)

    def forward(self, x, cond):
        c = cond.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return x + self.drop(self.conv(torch.cat([x, c], 1)))


def tiny_fixture(n=6, epochs=3, seed=7):
    g = torch.Generator().manual_seed(1234)
    names = [f"FIX{i:02d}" for i in range(n)]
    data = {k: (torch.rand(1, 3, 8, 8, generator=g), torch.rand(1, 3, 8, 8, generator=g), torch.rand(1, generator=g)) for k in names}
    og = torch.Generator().manual_seed(seed)
    orders = [[names[i] for i in torch.randperm(n, generator=og).tolist()] for _ in range(epochs)]
    return names, data, orders


def _tiny_objects(init_seed, epochs, lr=1e-2, eta_min=1e-4):
    torch.manual_seed(init_seed)
    model = _Tiny()
    opt = make_optimizer(model, lr)
    return model, opt, make_scheduler(opt, epochs, eta_min)


def fixture_exact_resume(tmpdir=None, epochs=3, interrupt_after=10, ckpt_every=3, seed=7):
    """Uninterrupted run vs (interrupted after `interrupt_after` updates, then resumed from the last
    checkpoint into freshly constructed objects with different init and RNG). CPU, synthetic data only."""
    names, data, orders = tiny_fixture(epochs=epochs, seed=seed)
    get = data.__getitem__
    fwd = lambda mdl, x, c: mdl(x, c)  # noqa: E731
    ident = {"fixture": True, "seed": seed, "order_sha256": order_digest(orders), "epochs": epochs}
    kw = dict(ckpt_every=ckpt_every, log_every=10 ** 9, announce=False, say=_quiet)
    with tempfile.TemporaryDirectory(dir=tmpdir) as td:
        td = Path(td)
        ma, oa, sa = _tiny_objects(seed, epochs)
        seed_everything(seed)
        ra = run(ma, oa, sa, orders, get, fwd, td / "a" / "state.pt", ident, **kw)
        mb, ob, sb = _tiny_objects(seed, epochs)
        seed_everything(seed)
        rb = run(mb, ob, sb, orders, get, fwd, td / "b" / "state.pt", ident, max_updates=interrupt_after, **kw)
        mc, oc, sc = _tiny_objects(seed + 1000, epochs)
        seed_everything(seed + 1000)
        rc = run(mc, oc, sc, orders, get, fwd, td / "b" / "state.pt", ident, **kw)
        sda, sdc = ma.state_dict(), mc.state_dict()
        diff = max(float((sda[k] - sdc[k]).abs().max()) for k in sda)
        md, od, sd = _tiny_objects(seed, epochs)
        stale_refused = _refused(lambda: run(md, od, sd, orders, get, fwd, td / "b" / "state.pt", dict(ident, seed=seed + 1), **kw))
    equal = (ra["complete"] and rc["complete"] and not rb["complete"]
             and all(torch.equal(sda[k], sdc[k]) for k in sda) and ra["progress"]["history"] == rc["progress"]["history"]
             and json.dumps(_to_cpu(oa.state_dict()["state"]), default=lambda t: t.tolist())
             == json.dumps(_to_cpu(oc.state_dict()["state"]), default=lambda t: t.tolist()))
    return {"equal": bool(equal), "max_abs_diff": diff, "interrupted_complete": rb["complete"],
            "stale_refused": stale_refused, "updates": ra["progress"]["global_updates"]}


# ------------------------------------------------------------------ commands
def cmd_smoke(m):
    check_training_config(m)
    names = validate_split(m)
    for s in m["training"]["seeds"]:
        validate_orders(m["orders"][str(s)], names, m["training"]["epochs"])
    check_r11()
    dev_names = [s["name"] for s in m["scenes"] if s["role"] == "development"]
    for arm in ARMS:
        p = D.Pairs(m, arm, role="fit")
        for bad in (dev_names[0], dev_names[-1], "DSCF9999.RAF"):
            D.need(_refused(lambda: p.get(bad)), f"{arm}: fit allowlist did not refuse {bad}")
    log("refusals ok (development scenes and unknown name refused by the fit allowlist, both arms)")
    fx = fixture_exact_resume()
    D.need(fx["equal"] and fx["stale_refused"], f"CPU exact-resume fixture failed: {fx}")
    log(f"CPU exact-resume fixture ok: {fx}")
    dev = D.device()
    t = m["training"]
    report = {}
    for arm in ARMS:
        model = D.initialise(m, 42, dev)
        seed_everything(42)
        opt = make_optimizer(model, t["lr"])
        model.train()
        pairs = D.Pairs(m, arm, role="fit")
        losses, secs = [], []
        for name in names[:2]:
            x, y, cond = pairs.get(name)
            t0 = time.time()
            losses.append(update(model, opt, x, y, cond, lambda mdl, xx, c: D.forward(mdl, m, xx, c, dev)))
            if dev.type == "mps":
                torch.mps.synchronize()
            secs.append(round(time.time() - t0, 2))
        report[arm] = {"device": str(dev), "scenes": names[:2], "losses": losses, "step_seconds": secs,
                       "input_shape": list(x.shape)}
    check_r11()
    log(json.dumps(report, indent=1))
    D.write_once_json(D.DIAG / "matched-gamma-training-smoke.json",
                      {"passed": True, "manifest_sha256": m["_manifest_sha256"],
                       "trainer_sha256": D.sha(CODE), "resume": fx, "arms": report,
                       "scope": "Smoke only; no production training run or checkpoint."})
    log("SMOKE_OK: finite full-resolution updates for both arms; diagnostic proof only; no training run started")


def cmd_all(m):
    import eval_matched_gamma_training as E
    with global_lock():
        for seed in m["training"]["seeds"]:
            for arm in ARMS:
                ep = train_arm(m, seed, arm)
                log(f"[seed{seed} {arm}] EVAL_WRITTEN {_rel(E.evaluate(m, seed, arm, ep))}")
                log(f"report: {_rel(E.report(m))}")
        final = E.report(m)
    log(f"ALL_COMPLETE final report {_rel(final)}")


def main(argv=None):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("smoke", "train", "all"))
    ap.add_argument("--manifest-sha", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--arm", choices=ARMS)
    a = ap.parse_args(argv)
    m = D.load_manifest(a.manifest_sha)
    if a.command == "smoke":
        cmd_smoke(m)
    elif a.command == "train":
        D.need(a.seed is not None and a.arm is not None, "train needs --seed and --arm")
        log(f"ENDPOINT {_rel(train_arm(m, a.seed, a.arm))}")
    else:
        cmd_all(m)


if __name__ == "__main__":
    main()
