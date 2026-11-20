"""Train-only visual/tail review of the frozen low-rate candidate.

No fitting or optimizer updates. Fixed six numerical probes plus the two
largest reported training regressions, selected by a recorded rule. Old
evidence and K/monitor/finalHeldOut/T8/dr100Eval/CN targets stay untouched.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
from pathlib import Path

import cv2
import numpy as np
import torch

import provia_low_rate as L
from audit_r11_display_domain import domain_stats
from color_metrics import ciede2000, linear_srgb_to_lab

OUT = L.P.DIAG / "provia-low-rate-tail-review.json"
PANELS = L.P.DIAG / "provia-low-rate-review"


def encode(rgb):
    x = np.clip(rgb, 0, 1)
    return (255 * np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) + 0.5).astype(np.uint8)


def panel(name, images, heatmaps, new_negative):
    def resize(a):
        return cv2.resize(a, (round(a.shape[1] * 420 / a.shape[0]), 420), interpolation=cv2.INTER_AREA)
    def label(a, text):
        a = resize(a).copy()
        cv2.rectangle(a, (0, 0), (a.shape[1], 27), (0, 0, 0), -1)
        cv2.putText(a, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return a
    top = np.concatenate([label(encode(x), title) for title, x in images], axis=1)
    bottom = []
    for title, de in heatmaps:
        h = cv2.applyColorMap((255 * np.clip(de / 10, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO)
        bottom.append(label(cv2.cvtColor(h, cv2.COLOR_BGR2RGB), title))
    mask = np.zeros((*new_negative.shape, 3), dtype=np.uint8)
    mask[new_negative] = [255, 40, 40]
    bottom.append(label(mask, "New negative; target >.001"))
    image = np.concatenate([top, np.concatenate(bottom, axis=1)], axis=0)
    path = PANELS / (Path(name).stem + ".jpg")
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def main():
    if OUT.exists() or PANELS.exists():
        raise ValueError("Review exists; refusing overwrite")
    man, data, _ = L.load()
    report = json.loads(L.REPORT.read_text())
    probe = json.loads((L.P.DIAG / "provia-update-probe-manifest.json").read_text())
    fixed = [r["name"] for r in probe["scenes"]]
    worst = sorted(report["per_scene"], key=lambda n: (-(report["per_scene"][n]["low_candidate"]["mean_dE00"]
                    - report["per_scene"][n]["r11"]["mean_dE00"]), n))[:2]
    names = list(dict.fromkeys(fixed + worst))
    loader = L.P.Pairs(data)
    if set(names) - set(loader.by):
        raise ValueError("Review contains a non-allowlisted name")
    source = L.state_path("candidate", 4)
    log = json.loads((L.P.RUNS / "provia-low-rate-candidate.json").read_text())
    if L.P.sha256(source) != log["state_sha256"]["block4"]:
        raise ValueError("Candidate checkpoint changed")
    st = torch.load(source, map_location="cpu", weights_only=False)
    if st["cursor"] != 1868 or st["manifest_sha256"] != L.MANIFEST_SHA:
        raise ValueError("Unexpected candidate state")
    dev = L.P.device()
    original, config = L.P.build(dev); original.eval()
    candidate, cc = L.P.build(dev); candidate.load_state_dict(st["model"]); candidate.eval()
    PANELS.mkdir()
    records = {}
    for name in names:
        x, target, cond = loader.get(name)
        image = lambda t: t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
        ref = image(target)
        with torch.no_grad():
            base = image(L.P.fwd(original, config, x, cond, dev))
            low = image(L.P.fwd(candidate, cc, x, cond, dev))
        lab = linear_srgb_to_lab(ref)
        e0, e1 = [ciede2000(linear_srgb_to_lab(p), lab) for p in (base, low)]
        # Check fresh reads against frozen recorded means, not a different path.
        if max(abs(e0.mean() - report["per_scene"][name]["r11"]["mean_dE00"]),
               abs(e1.mean() - report["per_scene"][name]["low_candidate"]["mean_dE00"])) > 1e-8:
            raise ValueError("Review scores do not reproduce frozen report")
        new = ((low < 0) & (base >= 0) & (ref > 0.001)).any(-1)
        new_deep = ((low < -0.01) & (base >= -0.01) & (ref > 0.001)).any(-1)
        path = panel(name, [("R11", base), ("Low-rate candidate", low), ("Fuji Provia", ref)],
                     [("R11 error, 0-10 dE", e0), ("Candidate error, 0-10 dE", e1)], new)
        records[name] = {
            "r11": domain_stats(base, ref, e0), "candidate": domain_stats(low, ref, e1),
            "new_negative_target_positive_pixel_fraction": float(new.mean()),
            "new_negative_region_mean_dE_change": float((e1 - e0)[new].mean()) if new.any() else None,
            "new_deep_negative_target_positive_pixel_fraction": float(new_deep.mean()),
            "panel": str(path.relative_to(L.P.TRAIN.parent)),
        }
        print(name, "DE", float(e0.mean()), float(e1.mean()), "new negative", float(new.mean()),
              "new deep negative", float(new_deep.mean()), flush=True)
    L.P.check_source()
    result = {"scope": "Eight train-only diagnostic cases, not validation; no fitting",
              "selection": "Frozen six numerical probes plus two largest reported low-candidate-minus-R11 training regressions, ties filename",
              "source_report_sha256": L.P.sha256(L.REPORT), "candidate_checkpoint_sha256": L.P.sha256(source),
              "data_manifest_sha256": L.P.MANIFEST_SHA, "script_sha256": L.P.sha256(Path(__file__)),
              "note": "Negative occupancy is not proof of damage. All errors use the existing display clamp. The target-positive mask and -0.01 threshold are diagnostic only, not acceptance limits.",
              "per_scene": records}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
