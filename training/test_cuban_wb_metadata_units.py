import copy
import unittest

from diagnose_cuban_wb_metadata_units import compare_vector, diagnose_rows


class UnitsDiagnosisTests(unittest.TestCase):
    def test_common_unit_factor(self):
        result = compare_vector([2048, 1208, 2112], [512, 302, 528])
        self.assertEqual(result["scale"], "1/4")
        self.assertTrue(result["rgb_proportional_exact"])

    def test_scaling_cannot_hide_chromatic_error(self):
        result = compare_vector([2048, 1208, 2112], [513, 302, 528])
        self.assertFalse(result["rgb_proportional_exact"])
        self.assertEqual(result["cross_product_errors_rgb"], [1208, 0, 0])

    def test_fraction_not_floating_point_tolerance(self):
        self.assertTrue(compare_vector([6, 3, 12], [10, 5, 20])["rgb_proportional_exact"])
        self.assertFalse(compare_vector([2**32-1, 2**32-2, 2**32-3],
                                        [2**32-1, 2**32-2, 2**32-4])["rgb_proportional_exact"])

    def test_invalid_levels(self):
        for bad in ([1, 0, 2], [1, -2, 3], [True, 2, 3], [1.0, 2, 3],
                    [1, 2], [1, 2, 2**32], "123"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare_vector(bad, [1, 2, 3])

    def fixture(self):
        return [{"name": str(i), "capture_mode": 0 if i < 3 else 4080,
                 "predictions": {v: [40, 20, 60] for v in ("H0", "H1", "H2", "H3", "H4")},
                 "observed": [10, 5, 15]} for i in range(8)]

    def test_auto_control_required_for_same_variant(self):
        rows = self.fixture()
        rows[0]["predictions"]["H0"][0] += 1
        results = diagnose_rows(rows)
        self.assertFalse(results["H0"]["non_auto_interpretable_under_same_variant"])
        self.assertTrue(results["H1"]["non_auto_interpretable_under_same_variant"])
        self.assertEqual(results["H0"]["source_non_auto_rgb_exact"], 5)

    def test_no_mixed_unit_convention_automatic_pass(self):
        rows = self.fixture()
        rows[-1]["observed"] = [20, 10, 30]
        result = diagnose_rows(rows)["H1"]
        self.assertFalse(result["one_scale_across_all_scenes"])
        self.assertFalse(result["non_auto_interpretable_under_same_variant"])

    def test_invalid_coverage(self):
        rows = self.fixture()
        for bad in [rows[:-1], rows + [rows[0]], [copy.deepcopy(rows[0]) for _ in range(8)]]:
            with self.assertRaises(ValueError):
                diagnose_rows(bad)
        rows[3]["capture_mode"] = 0
        with self.assertRaises(ValueError):
            diagnose_rows(rows)


if __name__ == "__main__":
    unittest.main()
