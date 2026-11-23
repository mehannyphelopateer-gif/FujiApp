"""Synthetic tests for gamma_guide_metrics. Hand-computed fixtures only; no real data or model."""
import json
import unittest

import numpy as np

import gamma_guide_metrics as G


def grey_linear(L):
    """Linear grey whose CIE L* is L (L > 8): Y = ((L + 16) / 116) ** 3."""
    return ((L + 16.0) / 116.0) ** 3


def image(values):
    """1 x N x 3 grey image from a list of linear values."""
    return np.repeat(np.asarray(values, np.float64).reshape(1, -1, 1), 3, axis=2)


def synthetic_lab(L_values):
    lab = np.zeros((1, len(L_values), 3), np.float64)
    lab[0, :, 0] = L_values
    return lab


def json_roundtrip(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


class ReferenceLab(unittest.TestCase):
    def test_black_and_white(self):
        lab = G.reference_lab(image([0.0, 1.0]))
        self.assertAlmostEqual(lab[0, 0, 0], 0.0, places=9)      # f = 4/29 -> L = 116*4/29 - 16 = 0
        self.assertAlmostEqual(lab[0, 1, 0], 100.0, places=3)    # sRGB white -> L ~ 100

    def test_grey_L(self):
        lab = G.reference_lab(image([grey_linear(50.0)]))
        self.assertAlmostEqual(lab[0, 0, 0], 50.0, places=4)
        self.assertLess(abs(lab[0, 0, 1]) + abs(lab[0, 0, 2]), 0.05)

    def test_roundoff_above_one_accepted(self):
        two_ulp = np.nextafter(np.nextafter(np.float32(1), np.float32(2)), np.float32(2))
        t = np.ones((1, 2, 3), np.float32)
        t[0, 0, 0] = two_ulp
        lab = G.reference_lab(t)                                 # no strict unit-range refusal
        self.assertTrue(np.isfinite(lab).all())

    def test_refusals(self):
        for bad in (np.zeros((2, 2, 3), np.int32), np.full((1, 1, 3), np.nan), np.zeros((2, 2)),
                    np.zeros((2, 2, 4)), np.zeros((0, 2, 3))):
            with self.assertRaises(ValueError):
                G.reference_lab(bad)


class ErrorStatistics(unittest.TestCase):
    def test_zero_error_black_white(self):
        t = image([0.0, 1.0])
        r = G.error_statistics(t, G.reference_lab(t))
        self.assertEqual(r["pixels"], 2)
        self.assertEqual(r["whole"]["mean_dE00"], 0.0)
        self.assertIsNone(r["whole"]["dE00_share"])               # zero total error -> null share, not NaN
        self.assertEqual((r["bands"]["shadow"]["count"], r["bands"]["mid"]["count"], r["bands"]["high"]["count"]), (1, 0, 1))
        self.assertEqual(r["bands"]["shadow"]["dE00_contribution"], 0.0)
        self.assertIsNone(r["bands"]["shadow"]["dE00_share"])
        json_roundtrip(r)

    def test_signed_bias_cancels_but_unsigned_error_does_not(self):
        target = G.reference_lab(image([grey_linear(50.0), grey_linear(50.0)]))
        pred = image([grey_linear(60.0), grey_linear(40.0)])
        r = G.error_statistics(pred, target)["whole"]
        self.assertAlmostEqual(r["mean_signed_Lab"][0], 0.0, places=4)   # +10 and -10 cancel
        self.assertAlmostEqual(r["mean_abs_dL"], 10.0, places=4)        # but every pixel is 10 off
        self.assertGreater(r["mean_dE00"], 5.0)
        self.assertGreater(r["mean_abs_dL"], abs(r["mean_signed_Lab"][0]) + 9.0)

    def test_exact_band_boundaries(self):
        target = synthetic_lab([24.999999, 25.0, 74.999999, 75.0])
        r = G.error_statistics(np.zeros((1, 4, 3)), target)
        self.assertEqual([r["bands"][b]["count"] for b in ("shadow", "mid", "high")], [1, 2, 1])
        self.assertEqual([r["bands"][b]["fraction"] for b in ("shadow", "mid", "high")], [0.25, 0.5, 0.25])

    def test_empty_bands_null(self):
        target = G.reference_lab(image([grey_linear(50.0)] * 3))
        r = G.error_statistics(image([grey_linear(55.0)] * 3), target)
        for b in ("shadow", "high"):
            blk = r["bands"][b]
            self.assertEqual((blk["count"], blk["fraction"], blk["dE00_contribution"]), (0, 0.0, 0.0))
            for k in ("mean_dE00", "p95_dE00", "mean_abs_dL", "mean_abs_dC", "mean_signed_Lab", "dE00_share"):
                self.assertIsNone(blk[k])
        self.assertEqual(r["bands"]["mid"]["count"], 3)
        self.assertAlmostEqual(r["bands"]["mid"]["dE00_share"], 1.0, places=12)
        json_roundtrip(r)

    def test_pixel_weighted_reconstruction(self):
        target = synthetic_lab([5.0, 10.0, 30.0, 50.0, 60.0, 80.0, 95.0])
        pred = image([0.0, 0.02, 0.1, 0.3, 0.2, 0.9, 0.5])
        r = G.error_statistics(pred, target)
        whole = r["whole"]["mean_dE00"]
        bands = r["bands"].values()
        self.assertEqual(sum(b["count"] for b in bands), 7)
        self.assertAlmostEqual(sum(b["dE00_contribution"] for b in bands), whole, places=12)
        self.assertAlmostEqual(sum(b["fraction"] * b["mean_dE00"] for b in bands if b["count"]), whole, places=12)
        self.assertAlmostEqual(sum(b["dE00_share"] for b in bands), 1.0, places=12)

    def test_prediction_clamped_before_lab(self):
        target = G.reference_lab(image([1.0, 0.0]))
        r = G.error_statistics(image([1.7, -0.4]), target)
        self.assertEqual(r["whole"]["mean_dE00"], 0.0)

    def test_refusals(self):
        ok = np.zeros((2, 2, 3))
        for pred, tgt in ((ok, np.zeros((2, 3, 3))), (np.full((2, 2, 3), np.inf), ok),
                          (np.zeros((2, 2, 3), np.uint16), ok), (ok, np.full((2, 2, 3), np.nan))):
            with self.assertRaises(ValueError):
                G.error_statistics(pred, tgt)


class GuideStatistics(unittest.TestCase):
    def test_hand_computed_planes(self):
        # u = g*8: 0 -> plane0; .0625 -> u=.5 -> 0.5/0.5 on planes 0,1; .125 -> plane1; .5 -> plane4; 1 -> plane8.
        g = np.array([[0.0, 0.0625, 0.125, 0.5, 1.0]])
        r = G.guide_statistics(g, np.full(g.shape, 50.0))["whole"]
        mass = [0.3, 0.3, 0, 0, 0.2, 0, 0, 0, 0.2]
        hist = [0.4, 0.2, 0, 0, 0.2, 0, 0, 0, 0.2]
        for got, want in zip(r["plane_mass"], mass):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(r["lower_index_hist"], hist):
            self.assertAlmostEqual(got, want, places=12)
        self.assertAlmostEqual(r["effective_planes"], 1.0 / 0.26, places=12)
        self.assertAlmostEqual(r["first_interval_fraction"], 0.4, places=12)   # 0 and .0625 are < 1/8
        self.assertEqual(r["out_of_unit_fraction_before_clamp"], 0.0)
        self.assertEqual(r["count"], 5)

    def test_clamp_ends(self):
        g = np.array([[-0.5, 1.5]])
        r = G.guide_statistics(g, np.full(g.shape, 50.0))["whole"]
        self.assertEqual(r["out_of_unit_fraction_before_clamp"], 1.0)
        self.assertAlmostEqual(r["plane_mass"][0], 0.5, places=12)
        self.assertAlmostEqual(r["plane_mass"][8], 0.5, places=12)
        self.assertAlmostEqual(r["effective_planes"], 2.0, places=12)
        self.assertAlmostEqual(sum(r["plane_mass"]), 1.0, places=12)

    def test_per_band_and_empty_band(self):
        g = np.array([[0.0625, 0.5, 0.75]])
        L = np.array([[10.0, 50.0, 60.0]])
        r = G.guide_statistics(g, L)
        self.assertEqual(r["bands"]["shadow"]["count"], 1)
        self.assertAlmostEqual(r["bands"]["shadow"]["plane_mass"][0], 0.5, places=12)
        self.assertAlmostEqual(r["bands"]["shadow"]["plane_mass"][1], 0.5, places=12)
        self.assertEqual(r["bands"]["mid"]["count"], 2)
        self.assertAlmostEqual(r["bands"]["mid"]["plane_mass"][4], 0.5, places=12)
        self.assertAlmostEqual(r["bands"]["mid"]["plane_mass"][6], 0.5, places=12)
        high = r["bands"]["high"]
        self.assertEqual(high["count"], 0)
        self.assertIsNone(high["plane_mass"])
        self.assertIsNone(high["effective_planes"])
        json_roundtrip(r)

    def test_other_depth(self):
        r = G.guide_statistics(np.array([[0.25]]), np.array([[50.0]]), depth=3)["whole"]   # u = .5
        for got, want in zip(r["plane_mass"], [0.5, 0.5, 0.0]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(r["first_interval_fraction"], 1.0)                                # .25 < 1/2

    def test_refusals(self):
        g, L = np.zeros((2, 2)), np.zeros((2, 2))
        for depth in (1, 0, 9.0, True, "9"):
            with self.assertRaises(ValueError):
                G.guide_statistics(g, L, depth=depth)
        for gg, LL in ((np.zeros((2, 3)), L), (np.full((2, 2), np.nan), L), (g, np.full((2, 2), np.inf)),
                       (np.zeros((2, 2, 1)), L), (np.zeros((2, 2), np.int64), L)):
            with self.assertRaises(ValueError):
                G.guide_statistics(gg, LL)


try:
    import torch
    import provia_color_pilot as P
    _HAVE_PILOT = True
except (Exception, SystemExit):
    _HAVE_PILOT = False


@unittest.skipUnless(_HAVE_PILOT, "provia_color_pilot / torch not importable")
class MatchesFrozenDevMetric(unittest.TestCase):
    def test_whole_mean_p95_and_bands_match_dev_metrics(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 1, (5, 7, 3)).astype(np.float32)
        y[0, 0, 0] = np.nextafter(np.nextafter(np.float32(1), np.float32(2)), np.float32(2))
        p = rng.uniform(-0.1, 1.1, (5, 7, 3)).astype(np.float32)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).unsqueeze(0)  # noqa: E731
        ref = P.dev_metrics(to_t(p), to_t(y))
        r = G.error_statistics(p, G.reference_lab(y))
        self.assertAlmostEqual(r["whole"]["mean_dE00"], ref["mean_dE00"], delta=1e-9)
        self.assertAlmostEqual(r["whole"]["p95_dE00"], ref["p95_dE00"], delta=1e-9)
        for b in ("shadow", "mid", "high"):
            want = ref["tone_band_signed_Lab"][b]
            got = r["bands"][b]["mean_signed_Lab"]
            if want is None:
                self.assertIsNone(got)
            else:
                for gv, wv in zip(got, want):
                    self.assertAlmostEqual(gv, wv, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
