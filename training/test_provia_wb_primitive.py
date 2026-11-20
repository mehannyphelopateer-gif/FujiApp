"""Synthetic tests for the WB primitive gate v2. No media, no models."""
import unittest

import numpy as np

import eval_provia_wb_primitive as W

# A synthetic 6x6 pattern with all three channels (not claimed to be the X100VI layout).
PATTERN = [1, 0, 1, 1, 2, 1,
           2, 1, 2, 0, 1, 0,
           1, 0, 1, 1, 2, 1,
           1, 2, 1, 1, 0, 1,
           0, 1, 0, 2, 1, 2,
           1, 2, 1, 1, 0, 1]


PAT = np.array(PATTERN).reshape(6, 6)


def channel_map(height, width):
    return PAT[np.arange(height)[:, None] % 6, np.arange(width)[None, :] % 6]


def mosaic(width, height, scale, seed=3, raw_max=25000):
    """Native mosaic from shared raw values: floor(raw * scale[c]), clipped like scale_colors."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, raw_max, size=(height, width))
    ch = channel_map(height, width)
    vals = np.minimum(65535, np.floor(raw * np.asarray(scale)[ch])).astype(np.int32)
    img = np.zeros((height, width, 3), dtype=np.int32)
    np.put_along_axis(img, ch[..., None], vals[..., None], axis=2)
    return img


def arm_record(img):
    """Same statistics the JS probe computes in the browser."""
    h, w, _ = img.shape
    nz = (img != 0).sum(axis=2)
    k = (np.arange(h)[:, None] % 6) * 6 + np.arange(w)[None, :] % 6
    active = np.argmax(img != 0, axis=2)
    phase = np.zeros((36, 4))
    np.add.at(phase, (k[nz == 1], active[nz == 1]), 1)
    np.add.at(phase, (k[nz == 0], 3), 1)
    return {"shape": {"width": w, "height": h, "colors": 3, "bits": 16},
            "mixed_pixels": int((nz > 1).sum()), "phase_counts": phase.reshape(-1).tolist()}


def grid_samples(img):
    """Row-major grid (y outer, x inner), matching the JS loops."""
    return img[15::31, 15::31].reshape(-1).astype(np.uint16)


AS_SHOT = [628, 302, 445, 302]
AUTO = [518, 302, 570, 302]


def scales(rgbg):
    return [rgbg[c] / rgbg[1] for c in range(3)]


SCENE = {"name": "synthetic", "session": 0, "shift": [5, -6], "expected_gain_RGB": W.expected_gains(AS_SHOT, AUTO)}


class Gains(unittest.TestCase):
    def test_expected_gain_from_source_multipliers(self):
        g = W.expected_gains(AS_SHOT, AUTO)
        self.assertAlmostEqual(g[0], (518 / 302) / (628 / 302))
        self.assertEqual(g[1], 1.0)
        self.assertAlmostEqual(g[2], (570 / 302) / (445 / 302))

    def test_green_must_be_minimum(self):
        with self.assertRaises(ValueError):
            W.expected_gains([200, 302, 445, 302], AUTO)


class Structure(unittest.TestCase):
    def test_pattern_inferred_and_consistent(self):
        img = mosaic(66, 66, scales(AS_SHOT))
        pattern, problems = W.infer_pattern(arm_record(img)["phase_counts"])
        self.assertEqual(problems, [])
        self.assertEqual(pattern, PATTERN)

    def test_mixed_pixels_stop(self):
        img = mosaic(66, 66, scales(AS_SHOT))
        img[10, 10] = [100, 100, 0]
        pattern, problems = W.check_structure(arm_record(img), arm_record(mosaic(66, 66, scales(AUTO))))
        self.assertIsNone(pattern)
        self.assertTrue(problems)

    def test_inconsistent_phase_channel_stops(self):
        img = mosaic(66, 66, scales(AS_SHOT))
        k = PATTERN[0]
        img[6, 6] = 0
        img[6, 6, (k + 1) % 3] = 500  # same phase as (0,0), different channel
        pattern, problems = W.infer_pattern(arm_record(img)["phase_counts"])
        self.assertIsNone(pattern)
        self.assertEqual(problems[0]["phase"], [0, 0])

    def test_wrong_colors_or_size_stop(self):
        a = arm_record(mosaic(66, 66, scales(AS_SHOT)))
        b = arm_record(mosaic(72, 66, scales(AUTO)))
        self.assertTrue(W.check_structure(a, b)[1])
        c = dict(a, shape=dict(a["shape"], colors=4))
        self.assertTrue(W.check_structure(c, a)[1])


class Primitive(unittest.TestCase):
    def run_scene(self, img_d, img_e):
        return W.evaluate_scene(SCENE, arm_record(img_d), arm_record(img_e), grid_samples(img_d), grid_samples(img_e))

    # 80 x 80 grid sites = 6400; R and B each hold 8/36 of phases, so ~1400 per channel before the mask.
    WIDTH = HEIGHT = 15 + 31 * 80

    def test_truncated_scaling_passes(self):
        d = mosaic(self.WIDTH, self.HEIGHT, scales(AS_SHOT), 3)
        e = mosaic(self.WIDTH, self.HEIGHT, scales(AUTO), 3)
        row = self.run_scene(d, e)
        self.assertTrue(row["structure_ok"])
        for ch in row["channels_RGB"]:
            self.assertTrue(ch["measurable"], ch)
            self.assertLessEqual(ch["max_abs_codes"], 3)
        self.assertTrue(row["primitive_passed"])
        self.assertEqual(row["channels_RGB"][1]["exact_fraction"], 1.0)

    def test_wrong_gain_fails(self):
        d = mosaic(self.WIDTH, self.HEIGHT, scales(AS_SHOT), 3)
        e = mosaic(self.WIDTH, self.HEIGHT, [s * 1.01 for s in scales(AUTO)], 3)  # 1% gain error
        row = self.run_scene(d, e)
        self.assertTrue(row["structure_ok"])
        self.assertFalse(row["primitive_passed"])

    def test_short_sample_is_not_a_pass(self):
        d = mosaic(66, 66, scales(AS_SHOT), 3)
        e = mosaic(66, 66, scales(AUTO), 3)
        row = self.run_scene(d, e)
        self.assertFalse(row["primitive_passed"])
        self.assertTrue(all(not c["measurable"] for c in row["channels_RGB"]))

    def test_inactive_channel_value_refused(self):
        d = mosaic(66, 66, scales(AS_SHOT), 3)
        samples = grid_samples(d).reshape(-1, 3).copy()
        pattern = PATTERN
        y, x = 15, 15
        c = pattern[(y % 6) * 6 + (x % 6)]
        samples[0, (c + 1) % 3] = 7  # value in an inactive channel at the first grid site
        with self.assertRaises(ValueError):
            W.active_values(samples.reshape(-1), 66, 66, pattern)

    def test_grid_shape_matches_js_loops(self):
        self.assertEqual(W.grid_shape(7728, 5152), (len(range(15, 5152, 31)), len(range(15, 7728, 31))))


if __name__ == "__main__":
    unittest.main()
