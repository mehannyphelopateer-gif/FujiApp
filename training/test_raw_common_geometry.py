"""Synthetic metadata-only tests for raw_common_geometry (stdlib only; no files, media or pixels)."""
import copy
import unittest

import raw_common_geometry as G

CROP_TL, CROP_SIZE = (21, 12), (5152, 7728)


def geo(zoomed=False, flip=0, **extra):
    iw, ih, left, top = (7728, 5149, 12, 24) if zoomed else (7752, 5178, 0, 6)
    w, h = (iw, ih) if flip == 0 else (ih, iw)
    g = {"raw_width": 7872, "raw_height": 5196, "iwidth": iw, "iheight": ih, "left_margin": left, "top_margin": top,
         "width": w, "height": h, "flip": flip, "raw_pitch": 0,
         "raw_inset_crops": [{"cleft": 12, "ctop": 24, "cwidth": 7728, "cheight": 5149}, {"cleft": 65535, "ctop": 65535, "cwidth": 0, "cheight": 0}]}
    g.update(extra)
    return g


def run(g, ori, tl=CROP_TL, size=CROP_SIZE):
    return G.common_sensor_windows(g, tl, size, ori)


class Expected(unittest.TestCase):
    def test_unzoomed_offsets_before_orientation(self):
        r = run(geo(), 1)
        self.assertEqual(r["common_rect_xywh"], [12, 21, 7728, 5152])
        self.assertEqual(r["source_rect_xywh"], [0, 6, 7752, 5178])
        self.assertEqual(r["target_rect_xywh"], [12, 21, 7728, 5152])
        self.assertEqual(r["source_window_exact"], [[1, 646], [5, 1726], [645, 646], [5167, 5178]])  # 12/7752, 15/5178, 7740/7752 reduced
        self.assertEqual(r["source_window_unrotated"], [12 / 7752, 15 / 5178, 7740 / 7752, 5167 / 5178])
        self.assertEqual(r["target_window_unrotated"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(r["source_cropped_px"], {"left": 12, "top": 15, "right": 12, "bottom": 11})
        self.assertEqual(r["target_cropped_px"], {"left": 0, "top": 0, "right": 0, "bottom": 0})

    def test_zoomed_source_identity_target_trimmed(self):
        r = run(geo(zoomed=True), 1)
        self.assertEqual(r["common_rect_xywh"], [12, 24, 7728, 5149])
        self.assertEqual(r["source_window_unrotated"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(r["target_window_unrotated"], [0.0, 3 / 5152, 1.0, 1.0])
        self.assertEqual(r["target_cropped_px"], {"left": 0, "top": 3, "right": 0, "bottom": 0})

    def test_all_six_class_orientation_cases(self):
        for zoomed in (False, True):
            base = run(geo(zoomed), 1)
            x0, y0, x1, y1 = base["source_window_unrotated"]
            tx0, ty0, tx1, ty1 = base["target_window_unrotated"]
            for flip, ori, src, tgt in ((0, 1, [x0, y0, x1, y1], [tx0, ty0, tx1, ty1]),
                                        (5, 8, [y0, 1 - x1, y1, 1 - x0], [ty0, 1 - tx1, ty1, 1 - tx0]),
                                        (6, 6, [1 - y1, x0, 1 - y0, x1], [1 - ty1, tx0, 1 - ty0, tx1])):
                with self.subTest(zoomed=zoomed, flip=flip):
                    r = run(geo(zoomed, flip), ori)
                    for a, b in zip(r["source_window"], src):
                        self.assertAlmostEqual(a, b, places=15)
                    for a, b in zip(r["target_window"], tgt):
                        self.assertAlmostEqual(a, b, places=15)
                    self.assertEqual(r["common_rect_xywh"], base["common_rect_xywh"])   # sensor rect is orientation-free
                    self.assertEqual((r["flip"], r["orientation"]), (flip, ori))
                    for w in (r["source_window"], r["target_window"]):
                        self.assertTrue(0 <= w[0] < w[2] <= 1 and 0 <= w[1] < w[3] <= 1)

    def test_extra_fields_ignored_and_inputs_not_mutated(self):
        g = geo(gps=[1, 2], target_hint=[0, 0, 1, 1], raw_inset_crops=[{"cleft": 0, "ctop": 0, "cwidth": 1, "cheight": 1}])
        before, tl, size = copy.deepcopy(g), [21, 12], [5152, 7728]
        r = G.common_sensor_windows(g, tl, size, 1)
        self.assertEqual(r["common_rect_xywh"], run(geo(), 1)["common_rect_xywh"])
        self.assertEqual((g, tl, size), (before, [21, 12], [5152, 7728]))
        self.assertIn("includes borders", r["labels"]["target"])
        self.assertIn("not an applied geometry fix", r["labels"]["status"])


class Refusals(unittest.TestCase):
    def bad(self, g, ori=1, tl=CROP_TL, size=CROP_SIZE):
        with self.assertRaises(G.CommonGeometryError):
            G.common_sensor_windows(g, tl, size, ori)

    def test_geometry_refusals(self):
        self.bad(geo(flip=3), 3)
        self.bad(geo(flip=5), 6)                          # orientation mismatch
        self.bad(geo(flip=0), 8)
        self.bad(geo(flip=True))
        self.bad(geo(width=5178, height=7752))            # unrotated flip with rotated dims
        self.bad(geo(flip=5, width=7752, height=5178), 8)
        self.bad(geo(left_margin=-1))
        self.bad(geo(left_margin=12.0))
        self.bad(geo(iwidth=True))
        self.bad(geo(iwidth=0, width=0))
        self.bad(geo(raw_width=2 ** 40))
        self.bad(geo(left_margin=200))                    # visible exceeds raw frame
        g = geo(); del g["top_margin"]
        self.bad(g)
        self.bad("not a dict")
        self.bad(geo(), "1")

    def test_crop_refusals(self):
        self.bad(geo(), tl=(21, 200))                     # header crop exceeds raw frame
        self.bad(geo(), tl=(-1, 12))
        self.bad(geo(), tl=(21.0, 12))
        self.bad(geo(), size=(0, 7728))
        self.bad(geo(), size=(5152,))
        self.bad(geo(), size=(True, 7728))

    def test_disjoint_refused(self):
        g = geo(iwidth=100, iheight=100, width=100, height=100, left_margin=0, top_margin=0)
        self.bad(g, tl=(5000, 7000), size=(100, 100))


if __name__ == "__main__":
    unittest.main()
