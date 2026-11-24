"""Synthetic tests for encoder_information_metrics. Hand-built fixtures only; no data, model files or I/O."""
import json
import math
import unittest

import numpy as np
import torch
import torch.nn as nn

import encoder_information_metrics as M


def roundtrip(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


def grey(values, h=1):
    """(1,3,h,len(values)) float64 grey image whose luma equals each value."""
    v = torch.tensor(values, dtype=torch.float64).view(1, 1, 1, -1).expand(1, 3, h, len(values))
    return v.contiguous()


def ramp(h=5, w=6):
    y = torch.arange(h, dtype=torch.float64).view(h, 1)
    x = torch.arange(w, dtype=torch.float64).view(1, w)
    plane = 10 * y + x
    return torch.stack([plane, plane + 100, plane + 200]).unsqueeze(0)


class VectorResponse(unittest.TestCase):
    def test_equal(self):
        b = np.array([1.0, -2.0, 2.0])
        r = M.vector_response(b, b.copy(), scale=4.0)
        self.assertEqual((r["l2"], r["rms"], r["relative_l2"], r["normalized_l2"]), (0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(r["cosine"], 1.0, places=12)
        roundtrip(r)

    def test_negation(self):
        b = np.array([3.0, 4.0])                          # |b| = 5
        r = M.vector_response(b, -b)
        self.assertAlmostEqual(r["l2"], 10.0, places=12)
        self.assertAlmostEqual(r["rms"], 10.0 / math.sqrt(2), places=12)
        self.assertAlmostEqual(r["cosine"], -1.0, places=12)
        self.assertAlmostEqual(r["relative_l2"], 2.0, places=12)
        self.assertIsNone(r["normalized_l2"])             # no scale given

    def test_scaled(self):
        b = torch.tensor([[1.0, 2.0], [2.0, 4.0]])        # torch input, |b| = 5
        r = M.vector_response(b, 2 * b, scale=2.5)
        self.assertAlmostEqual(r["cosine"], 1.0, places=6)
        self.assertAlmostEqual(r["relative_l2"], 1.0, places=6)
        self.assertAlmostEqual(r["normalized_l2"], 2.0, places=6)
        self.assertEqual(r["elements"], 4)

    def test_zero_norm_nulls(self):
        z = np.zeros(3)
        r = M.vector_response(z, np.array([0.0, 1.0, 0.0]))
        self.assertIsNone(r["relative_l2"])
        self.assertIsNone(r["cosine"])
        self.assertEqual(r["l2"], 1.0)
        r = M.vector_response(z, z)
        self.assertIsNone(r["cosine"])
        self.assertIsNone(r["relative_l2"])
        roundtrip(r)

    def test_null_normaliser(self):
        b, c = np.ones(2), np.zeros(2)
        for s in (None, 0.0, 1e-12, 5e-13):
            self.assertIsNone(M.vector_response(b, c, scale=s)["normalized_l2"])
        self.assertIsNotNone(M.vector_response(b, c, scale=2e-12)["normalized_l2"])

    def test_refusals(self):
        ok = np.ones(3)
        bad = [(ok, np.ones(4), None), (ok, np.ones((3, 1)), None), (np.array([1.0, np.nan, 0.0]), ok, None),
               (ok, ok, -1.0), (ok, ok, float("inf")), (ok, ok, True), (np.ones(3, np.int64), ok, None),
               (np.ones(0), np.ones(0), None)]
        for b, c, s in bad:
            with self.assertRaises(ValueError):
                M.vector_response(b, c, scale=s)


class SceneScale(unittest.TestCase):
    def test_median_unique_pairs(self):
        g = [np.array([0.0]), np.array([3.0]), np.array([5.0])]   # pairs 3, 5, 2
        self.assertEqual(M.scene_distance_scale(g), 3.0)
        g.append(np.array([9.0]))                                   # pairs 3,5,9,2,6,4 -> median 4.5
        self.assertEqual(M.scene_distance_scale(g), 4.5)

    def test_fewer_than_two_and_identical(self):
        self.assertIsNone(M.scene_distance_scale([]))
        self.assertIsNone(M.scene_distance_scale([np.ones(4)]))
        same = [np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))]
        s = M.scene_distance_scale(same)
        self.assertEqual(s, 0.0)
        self.assertIsNone(M.vector_response(same[0], same[0] + 1, scale=s)["normalized_l2"])

    def test_refusals(self):
        with self.assertRaises(ValueError):
            M.scene_distance_scale([np.ones(2), np.ones(3)])
        with self.assertRaises(ValueError):
            M.scene_distance_scale([np.ones(2), np.array([np.inf, 0.0])])


class ImageSummary(unittest.TestCase):
    def test_log_floor_and_mean(self):
        floor = 1.0 / 65535.0
        r = M.image_summary(grey([0.0, floor / 2, 0.25, 1.0]))
        want = (2 * math.log2(floor) - 2.0 + 0.0) / 4
        self.assertAlmostEqual(r["mean_log2_luma_floored"], want, places=9)
        self.assertAlmostEqual(r["mean_luma"], (floor / 2 + 0.25 + 1.0) / 4, places=9)
        self.assertEqual(r["log2_floor"], floor)
        self.assertAlmostEqual(r["fraction_luma_at_or_below_floor"], 0.5, places=12)
        roundtrip(r)

    def test_p99_linear_interpolation(self):
        r = M.image_summary(grey([k / 100 for k in range(101)]))
        self.assertAlmostEqual(r["p99_luma"], 0.99, places=9)
        self.assertEqual(r["pixels"], 101)

    def test_channel_at_or_above_one_and_negative(self):
        x = torch.zeros(1, 3, 1, 4, dtype=torch.float64)
        x[0, 0, 0, 0] = 1.0           # counted (>= 1)
        x[0, 2, 0, 1] = 0.9999        # not counted
        x[0, 1, 0, 2] = 1.5           # counted
        x[0, 1, 0, 3] = -0.1          # negative
        r = M.image_summary(x)
        self.assertAlmostEqual(r["fraction_any_channel_at_or_above_1"], 0.5, places=12)
        self.assertAlmostEqual(r["fraction_any_channel_below_0"], 0.25, places=12)

    def test_trim_excludes_border(self):
        x = torch.full((1, 3, 6, 6), 0.5, dtype=torch.float64)
        x[..., 0, :] = 100.0
        x[..., :, -1] = 100.0
        self.assertGreater(M.image_summary(x)["mean_luma"], 1.0)
        r = M.image_summary(x, trim=1)
        self.assertAlmostEqual(r["mean_luma"], 0.5, places=9)
        self.assertEqual((r["pixels"], r["trim"]), (16, 1))

    def test_numpy_input_and_refusals(self):
        self.assertEqual(M.image_summary(np.full((1, 3, 2, 2), 0.25))["pixels"], 4)
        for bad, trim in ((torch.zeros(1, 4, 2, 2), 0), (torch.zeros(2, 3, 2, 2), 0), (torch.zeros(3, 2, 2), 0),
                          (torch.zeros(1, 3, 4, 4), 2), (torch.zeros(1, 3, 4, 4), -1), (torch.zeros(1, 3, 4, 4), 1.0),
                          (torch.zeros(1, 3, 4, 4), True), (torch.full((1, 3, 2, 2), float("nan")), 0)):
            with self.assertRaises(ValueError):
                M.image_summary(bad, trim=trim)

    def test_weights_match_model_guide(self):
        try:
            from model import LUMA_WEIGHTS
        except Exception:            # pragma: no cover - model import optional here
            self.skipTest("model not importable")
        self.assertEqual(tuple(LUMA_WEIGHTS), M.REC709_LUMA)


class BoxPrefilter(unittest.TestCase):
    def test_constant_preserved_and_shape(self):
        x = torch.full((1, 3, 9, 11), 0.37, dtype=torch.float64)
        y = M.fixed_box_prefilter(x, kernel=7)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x, atol=1e-13, rtol=0))

    def test_checkerboard_interior(self):
        h = w = 15
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        s = ((-1.0) ** (yy + xx)).to(torch.float64)
        x = s.expand(1, 3, h, w).contiguous()
        y3 = M.fixed_box_prefilter(x, kernel=3)
        y7 = M.fixed_box_prefilter(x, kernel=7)
        # 3x3: 5 same-sign - 4 opposite = s/9; 7x7: 25 - 24 = s/49 (interior, away from reflection)
        self.assertTrue(torch.allclose(y3[..., 1:-1, 1:-1], x[..., 1:-1, 1:-1] / 9, atol=1e-12))
        self.assertTrue(torch.allclose(y7[..., 3:-3, 3:-3], x[..., 3:-3, 3:-3] / 49, atol=1e-12))

    def test_kernel_one_identity_and_refusals(self):
        x = ramp()
        self.assertTrue(torch.equal(M.fixed_box_prefilter(x, kernel=1), x))
        for k in (0, 2, 4, -3, 3.0, True):
            with self.assertRaises(ValueError):
                M.fixed_box_prefilter(x, kernel=k)
        with self.assertRaises(ValueError):
            M.fixed_box_prefilter(torch.zeros(1, 3, 3, 3), kernel=7)     # pad 3 >= size 3
        with self.assertRaises(ValueError):
            M.fixed_box_prefilter(np.zeros((1, 3, 9, 9)), kernel=3)      # tensor required


class Translation(unittest.TestCase):
    def test_positive_dy_moves_content_down(self):
        x = ramp()
        y = M.translated_same_shape(x, 1, 0)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.equal(y[..., 1:, :], x[..., :-1, :]))      # out[y] = in[y-1]
        self.assertTrue(torch.equal(y[..., 0, :], x[..., 1, :]))         # reflect: row -1 -> row 1

    def test_negative_dx_moves_content_left(self):
        x = ramp()
        w = x.shape[3]
        y = M.translated_same_shape(x, 0, -2)
        self.assertTrue(torch.equal(y[..., :, :w - 2], x[..., :, 2:]))   # out[x] = in[x+2]
        self.assertTrue(torch.equal(y[..., :, w - 1], x[..., :, w - 3])) # reflect: col w+1 -> w-3
        self.assertTrue(torch.equal(y[..., :, w - 2], x[..., :, w - 2])) # reflect: col w   -> w-2

    def test_diagonal_and_zero(self):
        x = ramp()
        self.assertTrue(torch.equal(M.translated_same_shape(x, 0, 0), x))
        y = M.translated_same_shape(x, -1, 2)
        self.assertTrue(torch.equal(y[..., :-1, 2:], x[..., 1:, :-2]))

    def test_refusals(self):
        x = ramp()
        for dy, dx in ((3, 0), (0, -3), (1.0, 0), (True, 0), (0, None)):
            with self.assertRaises(ValueError):
                M.translated_same_shape(x, dy, dx)
        with self.assertRaises(ValueError):
            M.translated_same_shape(torch.zeros(1, 3, 2, 2), 2, 0)       # pad 2 >= height 2
        with self.assertRaises(ValueError):
            M.translated_same_shape(x, 0, 0, max_abs=-1)
        self.assertEqual(M.translated_same_shape(ramp(8, 8), 3, 0, max_abs=3).shape, (1, 3, 8, 8))


class GroupNormIsNotGenerallyInvariant(unittest.TestCase):
    """Toy evidence for the assay's framing: gain/condition responses are empirical, not guaranteed zero."""

    def block(self, in_ch, weight, bias):
        conv = nn.Conv2d(in_ch, 4, 1, bias=bias is not None).double()
        gn = nn.GroupNorm(2, 4, eps=1e-12).double()
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(weight, dtype=torch.float64).view(4, in_ch, 1, 1))
            if bias is not None:
                conv.bias.copy_(torch.tensor(bias, dtype=torch.float64))
        return nn.Sequential(conv, gn).eval()

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(1, 3, 8, 8, generator=g, dtype=torch.float64)
        self.w3 = [[0.5, -0.3, 0.8], [0.2, 0.9, -0.4], [-0.6, 0.1, 0.7], [0.3, 0.3, -0.9]]

    def test_gain_invariant_only_without_bias(self):
        with torch.no_grad():
            f = self.block(3, self.w3, None)
            self.assertLess(M.vector_response(f(self.x), f(2 * self.x))["relative_l2"], 1e-6)
            f = self.block(3, self.w3, [1.0, -0.5, 0.3, 2.0])
            self.assertGreater(M.vector_response(f(self.x), f(2 * self.x))["relative_l2"], 0.05)

    def condition_input(self, c):
        return torch.cat([self.x, torch.full((1, 1, 8, 8), c, dtype=torch.float64)], dim=1)

    def test_condition_removed_only_when_equal_within_group(self):
        with torch.no_grad():
            equal = [w + [k] for w, k in zip(self.w3, [0.7, 0.7, -0.2, -0.2])]
            f = self.block(4, equal, None)
            self.assertLess(M.vector_response(f(self.condition_input(0.5)), f(self.condition_input(1.0)))["l2"], 1e-9)
            differ = [w + [k] for w, k in zip(self.w3, [0.7, -0.7, 0.4, 0.1])]
            f = self.block(4, differ, None)
            r = M.vector_response(f(self.condition_input(0.5)), f(self.condition_input(1.0)))
            self.assertGreater(r["relative_l2"], 0.05)
            roundtrip(r)


if __name__ == "__main__":
    unittest.main()
