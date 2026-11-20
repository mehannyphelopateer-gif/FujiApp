"""Data-free checks of diagnostic labels and descriptive weight movement."""
import copy
import unittest

import torch

from audit_provia_distill_fit import classify, closure, module_movement


def fixture(fit, oof):
    def metrics(v):
        return {"objective_closure": v, "display_dE_closure": v}
    return {str(f): {a: {"own_target": "fuji" if a == "direct" else "teacher",
                         "fit": {"fuji": metrics(fit), "teacher": metrics(fit)},
                         "adaptation_OOF": {"fuji": metrics(oof), "teacher": metrics(oof)}}
                    for a in ("direct", "distill")} for f in range(4)}


class FitAuditTests(unittest.TestCase):
    def test_labels_and_boundary_mixed_cases(self):
        self.assertTrue(classify(fixture(.1, .01)).startswith("A_"))
        self.assertTrue(classify(fixture(.5, .1)).startswith("B_"))
        self.assertEqual(classify(fixture(.25, .01)), "MIXED_OR_INCONCLUSIVE")
        self.assertEqual(classify(fixture(.6, .11)), "MIXED_OR_INCONCLUSIVE")
        x = fixture(.1, .01)
        x["2"]["distill"]["fit"]["teacher"]["objective_closure"] = .3
        self.assertEqual(classify(x), "MIXED_OR_INCONCLUSIVE")

    def test_null_and_negative_closure(self):
        self.assertIsNone(closure(0, 0))
        self.assertAlmostEqual(closure(1, 1.5), -.5)
        self.assertTrue(classify(fixture(-.5, -.6)).startswith("A_"))
        x = fixture(.1, .01)
        x["0"]["direct"]["fit"]["fuji"]["objective_closure"] = None
        self.assertEqual(classify(x), "INCONCLUSIVE_zero_baseline")
        x = fixture(.1, .01)
        x["0"]["distill"]["adaptation_OOF"]["teacher"]["display_dE_closure"] = None
        self.assertEqual(classify(x), "INCONCLUSIVE_zero_baseline")
        for b, a in ((float("nan"), 1), (1, float("inf")), (-1, 1), (1, -1)):
            with self.assertRaises(ValueError):
                closure(b, a)

    def test_module_norms_are_descriptive_and_separate(self):
        src = {"grid.encoder.w": torch.ones(2), "grid.head.w": torch.ones(1),
               "refinement.w": torch.zeros(1)}
        endpoint = copy.deepcopy(src)
        endpoint["grid.head.w"][0] += .2
        r = module_movement(endpoint, src)
        self.assertEqual(r["grid.encoder."]["delta_l2"], 0)
        self.assertAlmostEqual(r["grid.head."]["max_abs_delta"], .2, places=6)
        self.assertEqual(r["refinement."]["delta_l2"], 0)
        self.assertIsNone(r["refinement."]["relative_l2"])


if __name__ == "__main__":
    unittest.main()
