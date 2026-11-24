"""Synthetic metadata fixtures for wb_alignment_metrics. No TIFF/RAF/pixels/targets opened."""
import json
import unittest
from fractions import Fraction

import wb_alignment_metrics as W


def roundtrip(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


def row(name, role, session, mode=4080, kelvin=6600, target=(620, 302, 455), tshift=(0, 0),
        sshift=(20, -60), source=(636, 302, 414)):
    return {"name": name, "role": role, "session": session, "mode": mode, "kelvin": kelvin,
            "target_shift": list(tshift), "source_shift": list(sshift),
            "target_rgb": list(target), "source_rgb": list(source)}


def k6600_fit():
    return [row("a", "fit", 3), row("b", "fit", 8), row("c", "fit", 11)]


def source(mode=4080, kelvin=6600, shift=(20, -60), cam=(636, 302, 414, 0), auto=(461, 302, 995, 302)):
    return {"source_wb_mode": mode, "source_kelvin": kelvin, "source_wb_fine_tune_raw": list(shift),
            "cam_mul": list(cam), "auto_rgbg": list(auto)}


class Ratios(unittest.TestCase):
    def test_scale_invariant_and_exact(self):
        a = W.positive_rgb_ratios([620, 302, 455])
        b = W.positive_rgb_ratios([1240, 604, 910, 604])
        self.assertEqual(a, b)
        self.assertEqual(a["R/G"], Fraction(310, 151))

    def test_g325_supported(self):
        r = W.positive_rgb_ratios([302, 325, 1137, 0])
        self.assertEqual((r["R/G"], r["B/G"]), (Fraction(302, 325), Fraction(1137, 325)))

    def test_refusals(self):
        for bad in ([True, 302, 400], [0, 302, 400], [-1, 302, 400], [300.0, 302, 400], [1, 2],
                    [1, 2, 3, 4], [1, 2, 3, 0, 0], "302", [1, 2, 3, True]):
            with self.assertRaises(ValueError):
                W.positive_rgb_ratios(bad)


class Anchors(unittest.TestCase):
    def test_min_anchor_vs_green_anchor_distinguished(self):
        r = W.anchor_record([302, 325, 1137, 0], [289, 302, 1044, 302])
        self.assertEqual((r["cam_mul"]["min_value"], r["cam_mul"]["min_channels"], r["cam_mul"]["green"]), (302, ["R"], 325))
        self.assertEqual((r["auto_estimate"]["min_value"], r["auto_estimate"]["min_channels"]), (289, ["R"]))
        self.assertEqual((r["auto_estimate"]["green"], r["auto_estimate"]["green_slots"]), (302, [302, 302]))
        self.assertFalse(r["green_equal_across_records"])
        roundtrip(r)

    def test_typical_green_anchor(self):
        r = W.anchor_record([440, 302, 828, 0], [461, 302, 995, 302])
        self.assertEqual(r["cam_mul"]["min_channels"], ["G"])
        self.assertTrue(r["green_equal_across_records"])

    def test_refusals(self):
        bad = [([440, 302, 828, 5], [461, 302, 995, 302]), ([440, 302, 828], [461, 302, 995, 302]),
               ([440, 302, 828, 0], [461, 302, 995, 301]), ([440, 302, 828, 0], [461, 302, 995, 0]),
               ([440, 302, 828, 0], [461, 302, 995, True]), ([440, True, 828, 0], [461, 302, 995, 302])]
        for cam, auto in bad:
            with self.assertRaises(ValueError):
                W.anchor_record(cam, auto)


class Profiles(unittest.TestCase):
    def test_kelvin6600_from_three_fit_sessions_is_candidate(self):
        p = W.calibrated_profiles(k6600_fit())["profiles"]["4080/6600"]
        self.assertEqual(p["status"], "candidate")
        self.assertEqual(p["reasons"], [])
        self.assertEqual((p["target_ratios"]["R/G"]["num"], p["target_ratios"]["R/G"]["den"]), (310, 151))
        self.assertEqual(p["fit_sessions"], [3, 8, 11])
        self.assertEqual(p["qualification"], "NONE")
        roundtrip(p)

    def test_development_contradiction_never_changes_fit_candidate(self):
        base = W.calibrated_profiles(k6600_fit())["profiles"]["4080/6600"]
        rows = k6600_fit() + [row("d1", "development", 7, target=(999, 302, 111)),
                              row("d2", "development", 12, tshift=(3, 3))]
        with_dev = W.calibrated_profiles(rows)["profiles"]["4080/6600"]
        self.assertEqual(with_dev["development_rows_ignored"], 2)
        with_dev = dict(with_dev, development_rows_ignored=0)
        self.assertEqual(with_dev, base)

    def test_development_only_family_unvalidated(self):
        p = W.calibrated_profiles([row("d", "development", 7, kelvin=5600)])["profiles"]["4080/5600"]
        self.assertEqual(p["status"], "unvalidated")
        self.assertIn("no fit rows", p["reasons"])

    def test_mixed_targets_refused(self):
        rows = k6600_fit() + [row("e", "fit", 13, target=(621, 302, 455))]
        p = W.calibrated_profiles(rows)["profiles"]["4080/6600"]
        self.assertEqual(p["status"], "unvalidated")
        self.assertIn("fit neutral-target ratios are not exactly constant", p["reasons"])
        self.assertNotIn("target_ratios", p)

    def test_scaled_identical_targets_are_constant(self):
        rows = k6600_fit()[:2] + [row("c", "fit", 11, target=(1240, 604, 910))]
        self.assertEqual(W.calibrated_profiles(rows)["profiles"]["4080/6600"]["status"], "candidate")

    def test_insufficient_scenes_sessions_and_nonzero_target_shift(self):
        one = W.calibrated_profiles([row("a", "fit", 3)])["profiles"]["4080/6600"]
        self.assertEqual(one["status"], "unvalidated")
        same = W.calibrated_profiles([row("a", "fit", 3), row("b", "fit", 3), row("c", "fit", 3)])["profiles"]["4080/6600"]
        self.assertIn("fewer than 2 fit sessions", same["reasons"])
        shifted = W.calibrated_profiles(k6600_fit()[:2] + [row("c", "fit", 11, tshift=(1, -3))])["profiles"]["4080/6600"]
        self.assertIn("target shift is not [0,0]", shifted["reasons"])

    def test_other_families_unvalidated_even_if_constant(self):
        rows = [row(n, "fit", s, kelvin=5600) for n, s in (("a", 3), ("b", 8), ("c", 11))]
        rows += [row(n, "fit", s, mode=770, kelvin=None) for n, s in (("p", 3), ("q", 8), ("r", 11))]
        out = W.calibrated_profiles(rows)["profiles"]
        for key in ("4080/5600", "770/None"):
            self.assertEqual(out[key]["status"], "unvalidated")
            self.assertEqual(out[key]["reasons"], ["family not in nonzero-shift profile allowlist"])

    def test_row_refusals(self):
        for bad in ([row("a", "monitor", 3)], [row("a", "fit", 3), row("a", "fit", 8)],
                    [row("a", "fit", 3, mode=1024)], [row("a", "fit", 3, kelvin=None)],
                    [row("a", "fit", 3, mode=770, kelvin=6600)], [row("a", "fit", 3, sshift=(10, 0))],
                    [row("a", "fit", 3, target=(620, 0, 455))]):
            with self.assertRaises(ValueError):
                W.calibrated_profiles(bad)
        for kw in ({"min_scenes": 0}, {"min_sessions": True}):
            with self.assertRaises(ValueError):
                W.calibrated_profiles(k6600_fit(), **kw)


class Policy(unittest.TestCase):
    def setUp(self):
        self.profiles = W.calibrated_profiles(k6600_fit())
        self.empty = W.calibrated_profiles([])

    def test_source_auto_chooses_auto_estimate_not_cam(self):
        for shift in ((80, -100), (0, 0)):
            p = W.propose_policy(source(mode=0, kelvin=None, shift=shift, cam=(440, 302, 828, 0)), self.profiles)
            self.assertEqual(p["basis"], "source-auto-estimate")
            self.assertEqual(p["ratios"]["R/G"]["num"] / p["ratios"]["R/G"]["den"], 461 / 302)
            self.assertEqual(p["qualification"], "NONE")
            roundtrip(p)

    def test_manual_zero_shift_chooses_cam_not_auto_including_g325(self):
        p = W.propose_policy(source(kelvin=2500, shift=(0, 0), cam=(302, 325, 1137, 0), auto=(289, 302, 1044, 302)), self.empty)
        self.assertEqual(p["basis"], "zero-shift-as-shot")
        self.assertEqual((p["ratios"]["R/G"]["num"], p["ratios"]["R/G"]["den"]), (302, 325))
        q = W.propose_policy(source(mode=1, kelvin=None, shift=(0, 0), cam=(302, 360, 1360, 0), auto=(427, 302, 751, 302)), self.empty)
        self.assertEqual(q["basis"], "zero-shift-as-shot")
        self.assertEqual((q["ratios"]["B/G"]["num"], q["ratios"]["B/G"]["den"]), (34, 9))

    def test_kelvin6600_nonzero_uses_profile_not_per_photo_values(self):
        p = W.propose_policy(source(cam=(700, 302, 400, 0)), self.profiles)
        self.assertEqual(p["basis"], "calibrated-kelvin-profile-candidate")
        self.assertEqual((p["ratios"]["R/G"]["num"], p["ratios"]["R/G"]["den"]), (310, 151))
        self.assertEqual(W.propose_policy(source(), self.empty)["basis"], "unsupported")

    def test_other_nonzero_families_unsupported(self):
        for s in (source(kelvin=5600), source(mode=770, kelvin=None, shift=(-120, -80)),
                  source(mode=512, kelvin=None, shift=(0, -20)), source(mode=1, kelvin=None, shift=(20, 0))):
            p = W.propose_policy(s, self.profiles)
            self.assertEqual(p["basis"], "unsupported")
            self.assertIsNone(p["ratios"])

    def test_refusals(self):
        bad = [source(mode=2), source(mode=1024, kelvin=None), source(shift=(10, 0)), source(shift=(200, 0)),
               source(kelvin=None), source(mode=0, kelvin=6600), source(cam=(636, 302, 414, 1)),
               source(auto=(461, 302, 995, 300))]
        for s in bad:
            with self.assertRaises(ValueError):
                W.propose_policy(s, self.profiles)
        leaked = dict(source(), target_rgb=[620, 302, 455])
        with self.assertRaises(ValueError):
            W.propose_policy(leaked, self.profiles)
        with self.assertRaises(ValueError):
            W.propose_policy(source(), {"not": "profiles"})

    def test_limits_documented(self):
        self.assertIn("one X100VI body", W.LIMITS)
        self.assertIn("not decoded applied gains", W.__doc__)
        self.assertIn("never derive profiles", W.LIMITS)


if __name__ == "__main__":
    unittest.main()
