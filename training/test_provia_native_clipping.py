"""Synthetic OLS influence identities; no media or model loading."""
import unittest
import numpy as np
from diagnose_provia_native_clipping import influence, dilate


class InfluenceTests(unittest.TestCase):
    def test_rare_high_leverage_points_can_explain_bias(self):
        # One percent of points can dominate an unconstrained OLS fit.
        x = np.r_[np.linspace(.05, .08, 9900), np.linspace(.5, .8, 100)]
        flag = np.arange(len(x)) >= 9900
        y = .8 * x + .02 * flag
        r = influence(x, y, flag, .8)
        self.assertLess(r["groups"]["flagged"]["fraction"], .05)
        self.assertGreater(r["groups"]["flagged"]["relative_contribution_to_bias"], .99)
        self.assertGreater(r["groups"]["flagged"]["global_variance_fraction"], .9)
        self.assertAlmostEqual(r["groups"]["clean"]["own_response"]["slope"], .8, places=12)
        self.assertAlmostEqual(sum(g["contribution_to_global_slope_bias"] for g in r["groups"].values()), r["global_slope_bias"], places=12)

    def test_white_neighborhood_edges(self):
        x = np.zeros((9, 9), bool)
        x[4, 4] = True
        self.assertEqual(int(dilate(x).sum()), 25)
        x[:] = False
        x[0, 0] = True
        self.assertEqual(int(dilate(x).sum()), 9)

    def test_constant_input_refused(self):
        with self.assertRaises(ValueError):
            influence(np.ones(20), np.ones(20), np.zeros(20, bool), 1.)


if __name__ == "__main__":
    unittest.main()
