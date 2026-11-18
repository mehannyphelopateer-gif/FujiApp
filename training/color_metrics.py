"""Colour metrics for Cuban Neg evaluation: sRGB-encoded -> CIELAB (D65) and
CIEDE2000. Pure numpy. `python3 training/color_metrics.py` runs the full
34-pair Sharma, Wu & Dalal (2005) reference self-test."""
import hashlib
from pathlib import Path

import numpy as np

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    """rgb: float array [...,3] of sRGB-encoded values in [0,1]."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M.T) / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def linear_srgb_to_lab(lin):
    """lin: linear-light sRGB-primaries [...,3]; clipped to [0,1] (display gamut)."""
    xyz = (np.clip(lin, 0.0, 1.0) @ _M.T) / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(lab1, lab2):
    L1, a1, b1 = np.moveaxis(lab1, -1, 0)
    L2, a2, b2 = np.moveaxis(lab2, -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cm7 / (Cm7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hpm = np.where(C1p * C2p == 0, hs,
                   np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Cpm7 = Cpm ** 7
    RC = 2 * np.sqrt(Cpm7 / (Cpm7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    SC, SH = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    return np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))


SHARMA = Path(__file__).resolve().parent / "data" / "ciede2000-sharma-testdata.txt"
# Complete published set (34 pairs, incl. hue-wrap and achromatic cases):
# Sharma, Wu & Dalal (2005), ciede2000testdata.txt from the author's site
# (hajim.rochester.edu/ece/sites/gsharma/ciede2000/), byte-checked equal in
# values to scikit-image v0.24.0's ciede2000_test_data.txt.
SHARMA_SHA256 = "44aebb39107128328add54fbef5ac8ee89909e50508f448a1580adea2058a4b8"


def self_test():
    raw = SHARMA.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHARMA_SHA256, "Sharma data file changed"
    r = np.array([[float(x) for x in line.split()] for line in raw.decode().splitlines() if line.strip()])
    assert r.shape == (34, 7), r.shape
    fwd, rev = ciede2000(r[:, 0:3], r[:, 3:6]), ciede2000(r[:, 3:6], r[:, 0:3])
    err = max(float(np.abs(fwd - r[:, 6]).max()), float(np.abs(rev - r[:, 6]).max()))
    assert np.array_equal(np.round(fwd, 4), r[:, 6]) and np.array_equal(np.round(rev, 4), r[:, 6]), err
    assert np.allclose(srgb_to_lab(np.array([1.0, 1.0, 1.0])), [100, 0, 0], atol=1e-3)
    assert np.allclose(srgb_to_lab(np.array([0.0, 0.0, 0.0])), [0, 0, 0], atol=1e-6)
    return err


if __name__ == "__main__":
    print(f"CIEDE2000 self-test OK: 34/34 Sharma pairs exact to 4 dp, both orders (max |err| {self_test():.2e})")
