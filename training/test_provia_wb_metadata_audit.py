"""Data-free tests of WB units, coverage gating and tied-rank null handling."""
import unittest

from audit_provia_wb_metadata import correlation, group_stats, steps


def row(session, mode, ratio=1.):
    return {"name": str((session, mode, ratio)), "session": session, "source_wb_mode": mode,
            "target_over_source_normalized_RB": [ratio, ratio]}


class MetadataAuditTests(unittest.TestCase):
    def test_raw_units_and_refusals(self):
        self.assertEqual(steps([80, -100]), (4, -5))
        self.assertEqual(steps([20, -60]), (1, -3))
        for raw in ([81, -100], [True, 0], [200, 0], [0], None):
            with self.assertRaises(ValueError):
                steps(raw)

    def test_singletons_and_mode_session_confounds(self):
        one = group_stats([row(1, 0)])
        self.assertIsNone(one["within_group_consistent_at_0_5pct"])
        same_mode = group_stats([row(1, 0), row(2, 0)])
        self.assertTrue(same_mode["within_group_consistent_at_0_5pct"])
        self.assertIsNone(same_mode["universal_fine_tune_only_mapping_validated"])
        same_session = group_stats([row(1, 0), row(1, 1)])
        self.assertIsNone(same_session["universal_fine_tune_only_mapping_validated"])
        covered = group_stats([row(1, 0), row(2, 1)])
        self.assertTrue(covered["universal_fine_tune_only_mapping_validated"])
        inconsistent = group_stats([row(1, 0), row(2, 1, 1.02)])
        self.assertFalse(inconsistent["universal_fine_tune_only_mapping_validated"])

    def test_correlation_nulls_and_ties(self):
        self.assertIsNone(correlation([], [])["rho"])
        self.assertIsNone(correlation([1, 1], [2, 3])["rho"])
        self.assertAlmostEqual(correlation([1, 1, 2, 2], [3, 3, 8, 8])["rho"], 1.)


if __name__ == "__main__":
    unittest.main()
