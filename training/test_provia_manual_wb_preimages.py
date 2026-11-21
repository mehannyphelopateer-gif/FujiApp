"""Synthetic tests for the manual-capture Q10 preimage probe. Own fixtures only; no real data."""
import unittest

import probe_provia_manual_wb_preimages as P

FACTORS = {"R4": 1159, "B-5": 852, "B-4": 904}   # synthetic qualified map for these tests


def row(mode, fine, cam, name="X.RAF"):
    return {"name": name, "shoot": "Shoot 1", "source_wb_mode": mode, "source_wb_fine_tune_raw": fine, "cam_mul": cam}


class Preimages(unittest.TestCase):
    def test_expanding_factor_is_unique_or_empty(self):
        for x in (300, 451, 512, 777):
            self.assertEqual(P.q10_preimages(P.q10_forward(x, 1159), 1159), [x])
        ys = {P.q10_forward(x, 1159) for x in range(300, 400)}
        gap = next(y for y in range(min(ys), max(ys)) if y not in ys)
        self.assertEqual(P.q10_preimages(gap, 1159), [])

    def test_contracting_factor_can_be_ambiguous(self):
        hits = [y for y in range(200, 400) if len(P.q10_preimages(y, 852)) == 2]
        self.assertTrue(hits)
        for y in hits[:5]:
            self.assertTrue(all(P.q10_forward(x, 852) == y for x in P.q10_preimages(y, 852)))

    def test_identity_factor(self):
        for y in (1, 302, 65535):
            self.assertEqual(P.q10_preimages(y, 1024), [y])

    def test_candidate_domain_is_uint32(self):
        self.assertEqual(P.q10_preimages(P.U32_MAX, 1), [])
        self.assertEqual(P.q10_preimages(P.U32_MAX, 1024), [P.U32_MAX])
        self.assertTrue(all(x <= P.U32_MAX for x in P.q10_preimages(1, 1)))

    def test_quantisation_boundaries(self):
        # x*n + 512 landing exactly on a multiple of 1024 belongs to the upper y (half rounds up).
        self.assertEqual(P.q10_forward(1, 512), 1)      # (512 + 512) // 1024 = 1
        self.assertEqual(P.q10_forward(1, 511), 0)      # (511 + 512) // 1024 = 0
        self.assertEqual(P.q10_preimages(1, 512), [1, 2])   # 2*512+512 = 1536 -> 1; 3*512+512 = 2048 -> 2
        self.assertEqual(P.q10_preimages(2, 2048), [1])

    def test_invalid_inputs_refused(self):
        for y, n in ((0, 1024), (-1, 1024), (2.0, 1024), (True, 1024), (2, 0), (2, 1.5), (2 ** 32, 1024)):
            with self.assertRaises(ValueError):
                P.q10_preimages(y, n)


class Rows(unittest.TestCase):
    def test_auto_modes_are_not_manual(self):
        for mode, label in ((0, "auto_normal"), (1, "auto_white_priority")):
            r = P.analyse_row(row(mode, [80, -100], [628, 302, 445, 0]), FACTORS)
            self.assertEqual((r["kind"], r["mode_label"]), ("auto", label))
            self.assertNotIn("channels", r)

    def test_unknown_mode_refused(self):
        for mode in (2, 768, 3840, "0"):
            with self.assertRaises(ValueError):
                P.analyse_row(row(mode, [0, 0], [600, 302, 500, 0]), FACTORS)

    def test_qualified_zero_and_unqualified_axes(self):
        y_r = P.q10_forward(560, 1159)
        r = P.analyse_row(row(4080, [80, 0], [y_r, 302, 455, 0]), FACTORS)
        self.assertEqual(r["channels"]["R"]["candidates"], [560])
        self.assertEqual(r["channels"]["B"]["status"], "unique")
        self.assertIn("ASSUMPTION", r["channels"]["B"]["basis"])
        self.assertTrue(r["both_axes_zero_or_qualified"] and r["both_unique"])
        kelvin = P.analyse_row(row(4080, [20, -60], [636, 302, 414, 0]), FACTORS)   # R+1, B-3: not qualified here
        self.assertEqual({c: kelvin["channels"][c]["status"] for c in "RB"}, {"R": "unqualified", "B": "unqualified"})
        self.assertIsNone(kelvin["channels"]["R"]["candidates"])
        self.assertFalse(kelvin["both_axes_zero_or_qualified"])

    def test_source_level_and_fine_tune_validation(self):
        for cam in ([600, 300, 500, 0], [600, 302, 500, 302], [0, 302, 500, 0], [600.0, 302, 500, 0]):
            with self.assertRaises(ValueError):
                P.analyse_row(row(512, [0, -20], cam), FACTORS)
        with self.assertRaises(ValueError):
            P.analyse_row(row(512, [0, -30], [600, 302, 500, 0]), FACTORS)

    def test_qualified_factors_require_complete_exact_loso(self):
        q10 = {"future_transfer_proposal": {"axes": [{"axis": "R4", "n": 1159}, {"axis": "B-5", "n": 852}, {"axis": "B-6", "n": 800}]},
               "loso": {"summary": {"per_axis": {"R4": {"held_rows": 5, "measurable_rows": 5, "exact_rows": 5},
                                                 "B-5": {"held_rows": 5, "measurable_rows": 4, "exact_rows": 4},
                                                 "B-6": {"held_rows": 3, "measurable_rows": 3, "exact_rows": 2}}}}}
        self.assertEqual(P.qualified_factors(q10), {"R4": 1159})

    def test_summary_counts(self):
        rows = [P.analyse_row(row(0, [0, 0], [600, 302, 500, 0], "a"), FACTORS),
                P.analyse_row(row(4080, [20, -60], [636, 302, 414, 0], "b"), FACTORS),
                P.analyse_row(row(512, [0, -20], [600, 302, 500, 0], "c"), FACTORS)]
        s = P.summarise(rows)
        self.assertEqual((s["rows"], s["manual_rows"]), (3, 2))
        self.assertEqual(s["manual_both_axes_zero_or_qualified"], 0)   # c has B-1, unqualified
        self.assertEqual(s["unqualified_axis_counts"], {"B-1": 1, "B-3": 1, "R1": 1})


if __name__ == "__main__":
    unittest.main()
