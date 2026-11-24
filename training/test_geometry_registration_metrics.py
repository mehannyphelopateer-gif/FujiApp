"""Synthetic-only tests for the registration helpers. No RAW, G, target, model or file I/O."""
import math
import unittest

import cv2
import numpy as np

import geometry_registration_metrics as G


def scene(h=240, w=360, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.random((h, w, 3)).astype(np.float32)
    return np.stack([cv2.GaussianBlur(base[..., c], (0, 0), 2.5) for c in range(3)], axis=-1) * 0.8 + 0.05


def box_of(mask):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]


class ZoomWindow(unittest.TestCase):
    def test_identity_and_canary_like_values(self):
        self.assertEqual(G.normalized_zoom_window((10, 20), (0, 0), (10, 20), 1), [0.0, 0.0, 1.0, 1.0])
        w = G.normalized_zoom_window((5152, 7728), (752, 1128), (3648, 5472), 1)
        self.assertAlmostEqual(w[0], 1128 / 7728); self.assertAlmostEqual(w[3], (752 + 3648) / 5152)

    def test_orientations_match_array_rotation(self):
        H, W, top, left, h, w = 40, 60, 6, 9, 20, 30
        mask = np.zeros((H, W), bool); mask[top:top + h, left:left + w] = True
        expected = {1: box_of(mask), 8: box_of(np.rot90(mask, 1)), 6: box_of(np.rot90(mask, -1))}
        for ori, box in expected.items():
            got = G.normalized_zoom_window((H, W), (top, left), (h, w), ori)
            for g, e in zip(got, box):
                self.assertAlmostEqual(g, e, places=12, msg=f"orientation {ori}")
        self.assertEqual(G.normalized_zoom_window((H, W), (top, left), (h, w), 8),
                         [top / H, 1 - (left + w) / W, (top + h) / H, 1 - left / W])

    def test_refusals(self):
        bad = [((0, 10), (0, 0), (1, 1), 1), ((10, 10), (-1, 0), (1, 1), 1), ((10, 10), (0, 0), (0, 5), 1),
               ((10, 10), (5, 5), (6, 5), 1), ((10, 10), (0, 0), (5, 5), 3), ((10, 10), (0, 0), (5, 5), 6.0),
               ((10, float("nan")), (0, 0), (5, 5), 1), ((True, 10), (0, 0), (5, 5), 1), ((10,), (0, 0), (5, 5), 1)]
        for args in bad:
            with self.assertRaises(G.RegistrationError):
                G.normalized_zoom_window(*args)


class Resample(unittest.TestCase):
    def test_identity_window_is_exact_copy_without_mutation(self):
        s = scene(); before = s.copy()
        out = G.zoom_resample(s, [0, 0, 1, 1])
        self.assertTrue(np.array_equal(out, s))
        self.assertTrue(np.array_equal(s, before))
        self.assertEqual(out.dtype, np.float32)

    def test_refusals(self):
        s = scene(20, 20)
        for win in ([0, 0, 1], [0.5, 0, 0.5, 1], [0, 0, 1.1, 1], [-0.1, 0, 1, 1], [0, 0, float("inf"), 1]):
            with self.assertRaises(G.RegistrationError):
                G.zoom_resample(s, win)
        for arr in (s[..., 0], s.astype(np.int32), np.full((4, 4, 3), np.nan, np.float32)):
            with self.assertRaises(G.RegistrationError):
                G.zoom_resample(arr, [0, 0, 1, 1])


class Features(unittest.TestCase):
    def test_feature_and_ncc_do_not_mutate_inputs(self):
        rgb=scene(90,120);before=rgb.copy()
        feature=G.registration_feature(rgb);self.assertTrue(np.array_equal(rgb,before))
        saved=feature.copy();self.assertAlmostEqual(G.feature_ncc(feature,feature),1.0,places=12)
        self.assertTrue(np.array_equal(feature,saved))

    def test_same_element_count_but_transposed_shape_refused(self):
        feature=G.registration_feature(scene(90,120))
        with self.assertRaises(ValueError):G.feature_ncc(feature,feature.T)

    def test_uniform_features_null_and_finite(self):
        f = G.registration_feature(np.full((60, 60, 3), 0.3, np.float32))
        self.assertTrue(np.isfinite(f).all())
        self.assertIsNone(G.feature_ncc(f, f))
        recs = G.tile_phase_shifts(f, f)
        self.assertEqual(len(recs), 9)
        self.assertTrue(all(r["usable"] is False and r["dx"] is None for r in recs))
        self.assertEqual(G.summarize_shifts(recs)["median_dx"], None)

    def test_known_subpixel_shift_recovered_without_mutation(self):
        s = scene()
        M = np.float32([[1, 0, 2.0], [0, 1, -1.5]])  # content moves +2 px right, 1.5 px up
        moved = np.stack([cv2.warpAffine(s[..., c], M, (s.shape[1], s.shape[0]), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT) for c in range(3)], axis=-1)
        a, b = G.registration_feature(s), G.registration_feature(moved)
        a0, b0 = a.copy(), b.copy()
        recs = G.tile_phase_shifts(a, b)
        self.assertTrue(np.array_equal(a, a0) and np.array_equal(b, b0))
        summary = G.summarize_shifts(recs)
        self.assertGreaterEqual(summary["usable_tiles"], 5)
        self.assertLess(abs(summary["median_dx"] - 2.0), 0.3)
        self.assertLess(abs(summary["median_dy"] + 1.5), 0.3)
        for r in recs:
            self.assertEqual(set(r), {"row", "col", "bounds_yx", "std_reference", "std_moving", "dx", "dy", "response", "usable", "reasons"})

    def test_source_zoom_beats_identity_under_colour_and_gamma_change(self):
        s = scene(300, 450, seed=11)
        k = 1 / math.sqrt(2)
        window = [(1 - k) / 2, (1 - k) / 2, 1 - (1 - k) / 2, 1 - (1 - k) / 2]
        target = G.zoom_resample(s, window)
        target = np.clip(np.power(np.clip(target * np.float32([1.2, 0.9, 1.1]), 0, 1), 0.8), 0, 1).astype(np.float32)
        ft = G.registration_feature(target)
        zoom = G.feature_ncc(G.registration_feature(G.zoom_resample(s, window)), ft)
        identity = G.feature_ncc(G.registration_feature(s), ft)
        self.assertIsNotNone(zoom); self.assertIsNotNone(identity)
        self.assertGreaterEqual(zoom - identity, 0.2)

    def test_invalid_feature_inputs(self):
        f = np.zeros((60, 60), np.float32)
        for a, b in ((f, f[:, :59]), (f[..., None], f), (f.astype(np.int16), f), (np.full((60, 60), np.inf, np.float32), f)):
            with self.assertRaises(G.RegistrationError):
                G.feature_ncc(a, b)
        with self.assertRaises(G.RegistrationError):
            G.tile_phase_shifts(np.zeros((40, 40), np.float32), np.zeros((40, 40), np.float32))  # too small
        with self.assertRaises(G.RegistrationError):
            G.registration_feature(np.zeros((10, 10, 4), np.float32))


if __name__ == "__main__":
    unittest.main()
