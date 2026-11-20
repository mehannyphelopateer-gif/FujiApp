"""Synthetic and frozen-train-case preflight for a training-only loss variant.

Commit the prepared probe manifest/code BEFORE `probe`; no full training,
selected models, or protected targets. Each real test is one discarded step.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import color_loss as OLD
import color_loss_extended as NEW
import provia_low_rate as L
import probe_provia_updates as Q
from color_metrics import _M, _WHITE

SYN = L.P.DIAG / "provia-extended-loss-synthetic.json"
MAN = L.P.DIAG / "provia-extended-loss-probe-manifest.json"
OUT = L.P.DIAG / "provia-extended-loss-probes.json"


def synthetic():
    if SYN.exists():
        raise ValueError("Synthetic report exists; refusing overwrite")
    rng = np.random.default_rng(42)
    inside = rng.uniform(0.01, 0.99, (2, 3, 13, 17))
    outside = rng.uniform(-1, 3, (2, 3, 13, 17))
    targets = rng.uniform(0, 1, inside.shape)
    xyz = outside.transpose(0, 2, 3, 1) @ _M.T / _WHITE
    d = 6 / 29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3*d*d) + 4/29)
    ref = np.stack((116*f[..., 1]-16, 500*(f[..., 0]-f[..., 1]), 200*(f[..., 1]-f[..., 2])), axis=1)
    double = torch.tensor(rng.uniform(-0.2, 1.2, (1, 3, 3, 4)), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(NEW.linear_rgb_to_lab, (double,), eps=1e-6, atol=1e-5, rtol=1e-4)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    records, cpu_grad = {}, None
    for dev in devices:
        x = torch.tensor(inside, dtype=torch.float32, device=dev, requires_grad=True)
        y = torch.tensor(targets, dtype=torch.float32, device=dev)
        old, new = OLD.per_scene_terms(x, y), NEW.per_scene_terms(x, y)
        value_diff = max(float((old[k]-new[k]).abs().max().detach()) for k in old)
        ga = torch.autograd.grad(old["combined"].mean(), x)[0]
        gb = torch.autograd.grad(new["combined"].mean(), x)[0]
        grad_diff = float((ga-gb).abs().max())
        assert value_diff < 1e-6 and grad_diff < 1e-6
        z = torch.tensor(outside, dtype=torch.float32, device=dev, requires_grad=True)
        got = NEW.linear_rgb_to_lab(z)
        assert np.allclose(got.detach().cpu().numpy(), ref, rtol=1e-5, atol=1e-3)
        NEW.per_scene_terms(z, y)["combined"].mean().backward()
        g = z.grad.detach().cpu().numpy()
        assert np.isfinite(g).all()
        if dev == "cpu":
            cpu_grad = g
        backend_error = float(np.abs(g-cpu_grad).max())
        assert backend_error < 1e-5
        toy = torch.tensor([-.01, .05, .02], device=dev).view(1,3,1,1).requires_grad_()
        target = torch.tensor([.01,.05,.02], device=dev).view_as(toy)
        old_grad = torch.autograd.grad(OLD.per_scene_terms(toy,target)["lab_l1_scaled"].mean(),toy)[0]
        new_grad = torch.autograd.grad(NEW.per_scene_terms(toy,target)["lab_l1_scaled"].mean(),toy)[0]
        assert float(old_grad[0,0,0,0]) == 0 and float(new_grad[0,0,0,0]) < 0
        directional = []
        for value, sign in ((-.1,-1),(1.1,1)):
            p = torch.full((1,3,2,2),value,device=dev,requires_grad=True)
            t = torch.full_like(p,0 if value<0 else 1)
            loss = NEW.per_scene_terms(p,t)["combined"].mean()
            loss.backward()
            toward = t-p.detach()
            derivative = float((p.grad*toward).sum())
            # Lab is a coupled colour space: componentwise RGB signs need
            # not all agree. Check the target-directed ray, not that false
            # property (especially near neutral L1 chroma kinks).
            assert bool(torch.isfinite(p.grad).all()) and derivative < 0
            with torch.no_grad():
                closer = NEW.per_scene_terms(p.detach()+0.001*toward,t)["combined"].mean()
            assert float(closer) < float(loss.detach())
            directional.append(derivative)
        records[dev] = {"in_gamut_max_loss_difference": value_diff, "in_gamut_max_gradient_difference": grad_diff,
                        "extended_max_Lab_error_vs_numpy": float(np.abs(got.detach().cpu().numpy()-ref).max()),
                        "extended_max_gradient_difference_vs_cpu": backend_error,
                        "negative_red_old_Lab_gradient": float(old_grad[0,0,0,0]),
                        "negative_red_extended_Lab_gradient": float(new_grad[0,0,0,0]),
                        "out_of_range_target_ray_derivatives": directional}
    result = {"scope": "Synthetic values/gradient checks only; no image/model access", "passed": True,
              "preflight_note": "An initial componentwise RGB-sign assertion was replaced before any real optimizer probe: coupled Lab gradients need not restore every RGB component independently. Target-directed derivative and finite difference are checked instead; accuracy contracts are unchanged.",
              "cpu_float64_gradcheck": True, "devices": records,
              "code_sha256": {f: L.P.sha256(L.P.TRAIN/f) for f in ("color_loss.py","color_loss_extended.py","color_metrics.py","check_extended_color_loss.py")}}
    SYN.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


def prepare():
    if MAN.exists():
        raise ValueError("Manifest exists; refusing overwrite")
    L.load()
    syn = json.loads(SYN.read_text())
    if not syn["passed"] or any(L.P.sha256(L.P.TRAIN/f)!=h for f,h in syn["code_sha256"].items()):
        raise ValueError("Synthetic checks/code mismatch")
    review_path = L.P.DIAG / "provia-low-rate-tail-review.json"
    review = json.loads(review_path.read_text())
    names = list(review["per_scene"])
    if len(names)!=8:
        raise ValueError("Expected exactly the frozen eight review cases")
    data = L.P.load_manifest()
    if not set(names)<={r["name"] for r in data["scenes"]}:
        raise ValueError("Non-allowlisted probe name")
    result = {"status":"FROZEN before one-step probes; commit code/manifest before probe",
              "synthetic_sha256": L.P.sha256(SYN), "code_sha256": syn["code_sha256"],
              "data_manifest_sha256": L.P.MANIFEST_SHA,"source_checkpoint_sha256": L.P.SOURCE_SHA,
              "selection": "Same eight fixed train-only visual review cases, including two largest training regressions",
              "review_sha256":L.P.sha256(review_path),"names":names,"lr":1e-5,
              "scope":"One discarded fresh-Adam update per case from original R11; no saved weights/selection/full training",
              "gate":"Finite gradients and strictly decreasing own extended-combined objective on all eight; otherwise stop, do not retune"}
    MAN.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print('manifest sha256',L.P.sha256(MAN))


def probe(expected):
    if OUT.exists() or L.P.sha256(MAN)!=expected:
        raise ValueError("Existing report or manifest mismatch")
    m=json.loads(MAN.read_text())
    if any(L.P.sha256(L.P.TRAIN/f)!=h for f,h in m["code_sha256"].items()) or L.P.sha256(SYN)!=m["synthetic_sha256"]:
        raise ValueError("Bound code/synthetic checks changed")
    _,data,_=L.load();pairs=L.P.Pairs(data);dev=L.P.device()
    model,config=L.P.build(dev);model.eval();anchor=Q.flat_weights(model)
    rows={}
    for name in m["names"]:
        Q.assign_weights(model,anchor);model.zero_grad(set_to_none=True)
        opt=torch.optim.Adam(model.parameters(),lr=m["lr"],betas=(.9,.999),eps=1e-8,weight_decay=0)
        x,y,cond=pairs.get(name);y=y.to(dev)
        before=NEW.per_scene_terms(L.P.fwd(model,config,x,cond,dev),y)["combined"].mean()
        before.backward()
        finite=bool(torch.isfinite(before)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not finite:
            raise ValueError("Non-finite probe gradient/loss")
        opt.step()
        with torch.no_grad():
            after=NEW.per_scene_terms(L.P.fwd(model,config,x,cond,dev),y)["combined"].mean()
        a,b=float(before.detach()),float(after)
        rows[name]={"before":a,"after":b,"change":b-a,"passed":b<a}
        print(name,rows[name],flush=True)
    Q.assign_weights(model,anchor);L.P.check_source()
    result={"scope":m["scope"],"manifest_sha256":expected,"per_case":rows,
            "passed":all(r["passed"] for r in rows.values()),"source_checkpoint_unchanged":True,"weights_saved":False}
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('synthetic','prepare','probe'));p.add_argument('--manifest-sha')
    a=p.parse_args()
    if a.command=='synthetic':synthetic()
    elif a.command=='prepare':prepare()
    elif not a.manifest_sha:p.error('probe requires committed --manifest-sha')
    else:probe(a.manifest_sha)
