import copy
import unittest

from diagnose_gamma_guide import choose_cohort, bucket, aggregate


class CohortTests(unittest.TestCase):
    def fixture(self):
        rows, inv = [], []
        i = 0
        for dr in (200, 400):
            for wb in (0, 1, 4080):
                for session in (3, 8, 11):
                    for frame in range(5):
                        i += 1
                        row = {"name": f"fixture{i}", "shoot": f"Shoot {i}", "role": "fit", "session": session, "captureDynamicRange": dr}
                        rows.append(row)
                        inv.append(dict(row, source_wb_mode=wb))
        return rows, inv

    def test_balanced_metadata_only_reproducible(self):
        rows, inv = self.fixture()
        chosen, strata = choose_cohort(rows, inv)
        self.assertEqual(len(chosen), 24)
        self.assertTrue(all(s["selected"] == 4 for s in strata))
        self.assertEqual((chosen, strata), choose_cohort(list(reversed(rows)), list(reversed(inv))))
        for stratum in {r["stratum"] for r in chosen}:
            counts = {}
            for row in chosen:
                if row["stratum"] == stratum:
                    counts[row["session"]] = counts.get(row["session"], 0) + 1
            self.assertLessEqual(max(counts.values()), 2)

    def test_development_not_selected_even_with_better_rank(self):
        rows, inv = self.fixture()
        first = choose_cohort(rows, inv)
        rows.append(dict(rows[0], name="000", shoot="outside", role="development"))
        inv.append(dict(rows[-1], source_wb_mode=0))
        self.assertEqual(first, choose_cohort(rows, inv))

    def test_short_strata_not_backfilled(self):
        rows, inv = self.fixture()
        rows = [r for r in rows if r["captureDynamicRange"] == 200 and r["session"] == 3]
        chosen, strata = choose_cohort(rows, inv)
        self.assertEqual(len(chosen), 6)
        self.assertTrue(all(s["short"] for s in strata))

    def test_unknown_mode_refused(self):
        for mode in (True, "0", 2, 1024):
            with self.assertRaises(ValueError):
                bucket(mode)

    def test_duplicate_inventory_refused(self):
        rows, inv = self.fixture()
        with self.assertRaises(ValueError):
            choose_cohort(rows, inv + [inv[0]])


class ContributionTests(unittest.TestCase):
    def test_contributions_and_signed_deltas(self):
        left = {"whole": {"mean_dE00": 6}, "bands": {"shadow": {"dE00_contribution": 1}, "mid": {"dE00_contribution": 2}, "high": {"dE00_contribution": 3}}}
        right = {"whole": {"mean_dE00": 7}, "bands": {"shadow": {"dE00_contribution": 3}, "mid": {"dE00_contribution": 2}, "high": {"dE00_contribution": 2}}}
        case = {"name": "fixture", "models": {str(s): {"legacy": left, "linear": right} for s in (42, 43, 44)}}
        result = aggregate([case])["42"]
        self.assertEqual(result["whole_delta_mean_dE00"], 1)
        self.assertEqual(result["band_contribution_deltas"], {"shadow": 2., "mid": 0., "high": -1.})
        bad = copy.deepcopy(case)
        bad["models"]["42"]["linear"]["whole"]["mean_dE00"] = 9
        with self.assertRaises(ValueError):
            aggregate([bad])


if __name__ == "__main__":
    unittest.main()
