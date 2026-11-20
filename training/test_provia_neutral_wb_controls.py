"""Synthetic camera-RGB controls; never loads media or model weights."""
import unittest
import numpy as np
from eval_provia_neutral_wb import ratio_screen


class CameraRatioTests(unittest.TestCase):
    def test_known_gain_and_green(self):
        d = np.full((20, 20, 3), .2)
        expected = [.8, 1., 1.2]
        self.assertTrue(ratio_screen(d, d * expected, expected)["passed"])
        self.assertFalse(ratio_screen(d, d * expected * [1, 1.02, 1], expected)["passed"])

    def test_mask_covers_both_arms_all_channels(self):
        d = np.full((20, 20, 3), .2)
        e = d.copy()
        e[:10, :, 0] = .9
        r = ratio_screen(d, e, [1, 1, 1])
        self.assertEqual(r["coverage"], .5)
        self.assertTrue(r["passed"])
        e[:] = 1
        self.assertFalse(ratio_screen(d, e, [1, 1, 1])["measurable"])

    def test_shape_and_tails(self):
        d = np.full((20, 20, 3), .2)
        e = d.copy()
        e[:2] *= 1.02
        self.assertFalse(ratio_screen(d, e, [1, 1, 1])["passed"])
        with self.assertRaises(ValueError):
            ratio_screen(d, e[:1], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
