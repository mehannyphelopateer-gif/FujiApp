"""Synthetic tests for the source-only Q10 diagnostic. Own fixtures only; no real data."""
import copy
import unittest
from fractions import Fraction as F

import probe_provia_wb_q10 as Q
import probe_provia_wb_shift_intervals as I

N_RED = {2: 1100, -1: 977}     # synthetic factors, not discovery values
N_BLUE = {-3: 900, -4: 870}


def q10(x, n):
    return (x * n + 512) // 1024


def row(name, session, red, blue, xr, xb):
    return {"name": name, "shoot": f"Shoot {name}", "session": session, "source_wb_mode": 0,
            "source_wb_fine_tune_raw": [20 * red, 20 * blue],
            "auto_rgbg": [xr, 302, xb, 302], "cam_mul": [q10(xr, N_RED[red]), 302, q10(xb, N_BLUE[blue]), 0]}


def fixture():
    xs = [(503, 571), (517, 563), (529, 556), (541, 548), (553, 540), (566, 533), (577, 525), (589, 518)]
    rows, k = [], 0
    for session in ("s1", "s2", "s3"):
        for red, blue in ((2, -3), (2, -4), (-1, -3)):
            for xr, xb in xs[k % len(xs):k % len(xs) + 2]:
                rows.append(row(f"{session}-{red}-{blue}-{xr}", session, red, blue, xr, xb))
            k += 1
    return rows


class Candidates(unittest.TestCase):
    def test_endpoint_closures(self):
        self.assertEqual(Q.integer_candidates(I.Interval(1, True, 2, False), 4), [4, 5, 6, 7])
        self.assertEqual(Q.integer_candidates(I.Interval(1, False, 2, True), 4), [5, 6, 7, 8])
        self.assertEqual(Q.integer_candidates(I.Interval(1, True, 2, True), 4), [4, 5, 6, 7, 8])
        self.assertEqual(Q.integer_candidates(I.Interval(1, False, 2, False), 4), [5, 6, 7])
        self.assertEqual(Q.integer_candidates(I.Interval(F(1, 3), True, F(2, 5), True), 4), [])  # 4/3..8/5: none

    def test_round_half_up_formula(self):
        self.assertEqual(Q.predict_level(3, 1024), 3)
        self.assertEqual(Q.predict_level(1, 512), 1)          # 0.5 rounds up
        self.assertEqual(Q.predict_level(1, 511), 0)
        self.assertEqual(Q.predict_level(518, 1159), (518 * 1159 + 512) // 1024)
        for bad in ((0, 1024), (3.0, 1024), (True, 1024)):
            with self.assertRaises(ValueError):
                Q.predict_level(*bad)
        with self.assertRaises(ValueError):
            Q.predict_level(3, 1024, den=1023)


class Lookup(unittest.TestCase):
    def obs(self, pairs, sessions):
        return [{"name": f"r{i}", "session": s, "x": x, "y": y, "other_step": 0} for i, ((x, y), s) in enumerate(zip(pairs, sessions))]

    def test_qualified_unique(self):
        pairs = [(x, q10(x, 1100)) for x in (503, 529, 553, 577, 589)]
        rec = Q.axis_lookup(self.obs(pairs, ["a", "b", "a", "b", "c"]))
        self.assertEqual(rec["status"], "qualified_unique")
        self.assertEqual(rec["n"], 1100)

    def test_singleton_and_one_session_are_sparse(self):
        pairs = [(503, q10(503, 1100)), (529, q10(529, 1100))]
        self.assertEqual(Q.axis_lookup(self.obs(pairs[:1], ["a"]))["status"], "sparse")
        rec = Q.axis_lookup(self.obs(pairs, ["a", "a"]))
        self.assertEqual(rec["status"], "sparse")
        self.assertIsNone(rec["n"])

    def test_ambiguous_and_empty(self):
        self.assertEqual(Q.axis_lookup(self.obs([(3, 3), (4, 4)], ["a", "b"]))["status"], "ambiguous")
        self.assertEqual(Q.axis_lookup(self.obs([(100, 150), (100, 160)], ["a", "b"]))["status"], "empty")

    def test_unseen_axis(self):
        rec = Q.axis_lookup([])
        self.assertEqual(rec["status"], "unseen")
        self.assertIsNone(rec["n"])


class Loso(unittest.TestCase):
    def test_full_data_resubstitution_and_loso_exact(self):
        result = Q.analyse(fixture())
        resub = result["primary"]["resubstitution"]
        self.assertEqual({k: v["n"] for k, v in resub.items()}, {"R2": 1100, "R-1": 977, "B-3": 900, "B-4": 870})
        self.assertTrue(all(v["all_rows_reproduced"] for v in resub.values()))
        summary = result["loso"]["summary"]
        self.assertEqual(summary["rows_both_measurable"], summary["rows_both_exact"])
        self.assertGreater(summary["rows_both_measurable"], 0)

    def test_held_rows_never_used_to_fit_their_fold(self):
        rows = fixture()
        changed = copy.deepcopy(rows)
        for r in changed:
            if r["session"] == "s3":
                r["cam_mul"][0] += 7   # corrupt held red levels only
        obs_a, obs_b = Q.axis_observations(rows), Q.axis_observations(changed)
        folds_a, rows_a = Q.loso(obs_a)
        folds_b, rows_b = Q.loso(obs_b)
        fa = next(f for f in folds_a if f["held_session"] == "s3")["channels"]["R"]
        fb = next(f for f in folds_b if f["held_session"] == "s3")["channels"]["R"]
        self.assertEqual({k: v["fold_n"] for k, v in fa.items()}, {k: v["fold_n"] for k, v in fb.items()})
        held_b = [r for r in rows_b if r["session"] == "s3"]
        self.assertTrue(all(r["R"]["measurable"] and r["R"]["exact"] is False for r in held_b))

    def test_axis_only_in_held_session_is_unmeasurable_not_identity(self):
        rows = fixture()
        rows.append(dict(row("s3-new", "s3", 2, -3, 600, 520), source_wb_fine_tune_raw=[20 * 5, 20 * -3],
                         cam_mul=[q10(600, 1242), 302, q10(520, 900), 0]))
        folds, loso_rows = Q.loso(Q.axis_observations(rows))
        new = next(r for r in loso_rows if r["name"] == "s3-new")
        self.assertFalse(new["R"]["measurable"])
        self.assertEqual(new["R"]["reason"], "unseen")
        self.assertIsNone(new["R"]["fold_n"])
        self.assertFalse(new["both_channels_measurable"])
        self.assertIsNone(new["both_channels_exact"])

    def test_missing_session_and_invalid_levels_refused(self):
        bad = fixture()
        bad[0] = dict(bad[0], session=None)
        with self.assertRaises(ValueError):
            Q.axis_observations(bad)
        for change in ({"auto_rgbg": [0, 302, 571, 302]}, {"cam_mul": [503.0, 302, 571, 0]}, {"auto_rgbg": [503, 300, 571, 300]}):
            rows = fixture()
            rows[0] = dict(rows[0], **change)
            with self.assertRaises(ValueError):
                Q.axis_observations(Q.validated_auto_rows(rows))

    def test_denominator_scan_and_separability_are_descriptive(self):
        result = Q.analyse(fixture())
        self.assertEqual(set(result["denominator_scan"]), {str(d) for d in Q.DENOMINATORS})
        self.assertIn(result["coarsest_tested_covering_denominator"], (None,) + Q.DENOMINATORS)
        self.assertIn("2", result["separability_descriptive"]["R"])      # R+2 appears with B-3 and B-4
        self.assertEqual(set(result["all_variants_descriptive"]), set(I.VARIANTS))


if __name__ == "__main__":
    unittest.main()
