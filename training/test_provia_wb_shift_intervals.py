"""Synthetic exact-rational tests for the WB shift interval probe. No media, no inventory reads."""
import unittest
from fractions import Fraction as F

import probe_provia_wb_shift_intervals as P


def rhu(v):
    """Round half up for a non-negative Fraction."""
    return int((F(v) + F(1, 2)).__floor__())


class Intervals(unittest.TestCase):
    def test_row_interval_endpoints_and_closures(self):
        x, y = 4, 7
        expect = {"source_half_up": (F(13, 8), True, F(15, 8), False),
                  "source_floor": (F(7, 4), True, F(2), False),
                  "source_ceil": (F(3, 2), False, F(7, 4), True),
                  "both_half_up": (F(13, 9), False, F(15, 7), False),
                  "both_floor": (F(7, 5), False, F(2), False),
                  "both_ceil": (F(3, 2), False, F(7, 3), False)}
        for variant, tup in expect.items():
            self.assertEqual(P.row_interval(variant, x, y).as_tuple(), tup, variant)

    def test_half_open_endpoint_touch_is_empty(self):
        iv, _, _ = P.intersect([P.Interval(1, True, 2, False), P.Interval(2, True, 3, False)])
        self.assertTrue(iv.is_empty())
        self.assertIsNone(P.representative(iv))

    def test_closed_unique_endpoint_is_accepted(self):
        iv, lo_idx, hi_idx = P.intersect([P.Interval(0, False, 2, True), P.Interval(2, True, 3, False)])
        self.assertFalse(iv.is_empty())
        self.assertEqual(P.representative(iv), F(2))
        self.assertEqual((lo_idx, hi_idx), ([1], [0]))

    def test_tied_bound_closed_only_if_all_tied_closed(self):
        iv, lo_idx, _ = P.intersect([P.Interval(1, True, 3, True), P.Interval(1, False, 4, True)])
        self.assertEqual(lo_idx, [0, 1])
        self.assertFalse(iv.lo_closed)


class Groups(unittest.TestCase):
    def test_known_constant_gain_is_compatible(self):
        g = F(7, 5)
        rows = [(f"r{x}", x, rhu(g * x)) for x in (300, 451, 512, 377)]
        out = P.analyse_channel("source_half_up", rows)
        self.assertTrue(out["compatible"] and out["evaluable"])
        lo, hi = F(out["lo"]["num"], out["lo"]["den"]), F(out["hi"]["num"], out["hi"]["den"])
        self.assertTrue(lo <= g < hi)
        self.assertTrue(all(s["contains"] for s in out["row_gain_interval_slack"]))

    def test_conflicting_rows_are_incompatible(self):
        rows = [("a", 100, 150), ("b", 100, 160)]
        for variant in P.VARIANTS:
            out = P.analyse_channel(variant, rows)
            self.assertFalse(out["compatible"], variant)
            self.assertIsNone(out["representative_gain"])
            self.assertLess(out["width_or_negative_gap"]["float"], 0)

    def test_row_order_invariance(self):
        rows = [("a", 300, 420), ("b", 451, 631), ("c", 512, 717)]
        for variant in P.VARIANTS:
            a = P.analyse_channel(variant, rows)
            b = P.analyse_channel(variant, list(reversed(rows)))
            self.assertEqual((a["lo"], a["hi"], a["lo_closed"], a["hi_closed"], a["compatible"]),
                             (b["lo"], b["hi"], b["lo_closed"], b["hi_closed"], b["compatible"]), variant)

    def test_both_quantised_feasible_where_exact_source_is_not(self):
        # g = 1.2 with unrounded sources 2.4 -> 2 and 2.6 -> 3 gives y = 3 for both rows.
        rows = [("a", 2, 3), ("b", 3, 3)]
        self.assertEqual(rhu(F(12, 10) * F(24, 10)), 3)
        self.assertEqual(rhu(F(12, 10) * F(26, 10)), 3)
        self.assertFalse(P.analyse_channel("source_half_up", rows)["compatible"])
        both = P.analyse_channel("both_half_up", rows)
        self.assertTrue(both["compatible"])
        self.assertTrue(F(1) < F(12, 10) < F(7, 5))
        self.assertEqual((both["lo"]["num"], both["lo"]["den"], both["hi"]["num"], both["hi"]["den"]), (1, 1, 7, 5))

    def test_singleton_is_non_evaluable(self):
        out = P.analyse_channel("source_floor", [("a", 300, 420)])
        self.assertFalse(out["evaluable"])
        self.assertTrue(out["compatible"])


class InputValidation(unittest.TestCase):
    def test_invalid_levels_refused(self):
        for x, y in ((0, 3), (-2, 3), (2.0, 3), (True, 3), (2, 0), (2, "3")):
            with self.assertRaises(ValueError):
                P.row_interval("source_floor", x, y)
        with self.assertRaises(ValueError):
            P.row_interval("both_ceil", 1, 3)
        with self.assertRaises(ValueError):
            P.row_interval("nearest", 2, 3)

    def row(self, **over):
        r = {"name": "DSCF0001.RAF", "shoot": "Shoot 1", "source_wb_mode": 0, "source_wb_fine_tune_raw": [100, -120],
             "auto_rgbg": [518, 302, 570, 302], "cam_mul": [628, 302, 445, 0]}
        r.update(over)
        return r

    def test_grouping_and_source_checks(self):
        groups = P.source_pairs([self.row(), self.row(name="b", source_wb_mode=4080), self.row(name="c", source_wb_fine_tune_raw=[80, -100])])
        self.assertEqual(sorted(groups), [(80, -100), (100, -120)])
        for bad in (self.row(auto_rgbg=[518, 300, 570, 300]), self.row(cam_mul=[628, 302, 445, 302]),
                    self.row(source_wb_fine_tune_raw=[90, -120]), self.row(cam_mul=[0, 302, 445, 0])):
            with self.assertRaises(ValueError):
                P.source_pairs([bad])

    def test_group_records_report_steps_and_descriptives(self):
        rows = [self.row(name=f"s{i}", auto_rgbg=[500 + 7 * i, 302, 560 + 5 * i, 302], cam_mul=[610 + 8 * i, 302, 440 + 4 * i, 0]) for i in range(4)]
        rec = P.analyse_groups(P.source_pairs(rows))[0]
        self.assertEqual(rec["steps"], [5, -6])
        self.assertEqual(set(rec["channels"]), {"R", "B"})
        self.assertEqual(set(rec["channels"]["R"]["variants"]), set(P.VARIANTS))
        self.assertIsNotNone(rec["channels"]["R"]["descriptive"]["pearson_ratio_vs_zero_white_G_over_R"])
        self.assertIsNone(P.pearson([1.0, 2.0], [1.0, 2.0]))
        self.assertIsNone(P.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
