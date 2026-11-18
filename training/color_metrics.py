"""Colour metrics for Cuban Neg evaluation: sRGB-encoded -> CIELAB (D65) and
CIEDE2000. Pure numpy. `python3 training/color_metrics.py` runs the Sharma,
Wu & Dalal (2005) reference-pair self-test."""
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


if __name__ == "__main__":
    # (L1,a1,b1, L2,a2,b2, dE00) from Sharma et al. 2005, Table 1 (selected rows)
    ref = [(50, 2.6772, -79.7751, 50, 0, -82.7485, 2.0425),
           (50, 3.1571, -77.2803, 50, 0, -82.7485, 2.8615),
           (50, -1.3802, -84.2814, 50, 0, -82.7485, 1.0000),
           (50, 2.5, 0, 50, 0, -2.5, 4.3065),
           (50, 2.5, 0, 73, 25, -18, 27.1492),
           (50, 2.5, 0, 50, 3.1736, 0.5854, 1.0000),
           (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
           (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
           (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082)]
    r = np.array(ref)
    got = ciede2000(r[:, 0:3], r[:, 3:6])
    print(np.round(got, 4), "max err", float(np.abs(got - r[:, 6]).max()))
    assert np.allclose(got, r[:, 6], atol=1e-4)
    assert np.allclose(srgb_to_lab(np.array([1.0, 1.0, 1.0])), [100, 0, 0], atol=1e-3)
    print("CIEDE2000 self-test OK")
