"""Synthetic tamper checks; no training evidence or image pixels modified."""
import copy
import json
import unittest

import torch

import audit_provia_distill_artifacts as A


class ArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.source = {"grid.weight": torch.ones(2, 2), "refinement.weight": torch.zeros(2)}
        self.order = [["fit-only"] for _ in range(A.D.EPOCHS)]
        self.state = {"manifest_sha256": A.D.MANIFEST_SHA, "fold": 0, "arm": "direct",
                      "epoch": A.D.EPOCHS, "orders": self.order,
                      "history": [{"epoch": e, "train_combined_vs_own_target": 0.1} for e in range(1, A.D.EPOCHS + 1)],
                      "model": {k: v.clone() for k, v in self.source.items()}}

    def test_valid_endpoint_and_identity_order_tampering(self):
        A.check_endpoint(self.state, self.source, 0, "direct", self.order, "fixture")
        for key, value in (("manifest_sha256", "wrong"), ("fold", 1), ("arm", "distill"),
                           ("epoch", 15), ("orders", [["held-out"]]), ("history", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                st = copy.deepcopy(self.state)
                st[key] = value
                A.check_endpoint(st, self.source, 0, "direct", self.order, "fixture")

    def test_frozen_tensor_nonfinite_and_history_tampering(self):
        for label in ("frozen", "nan_weight", "shape", "nan_history"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                st = copy.deepcopy(self.state)
                if label == "frozen":
                    st["model"]["refinement.weight"][0] = 1
                elif label == "nan_weight":
                    st["model"]["grid.weight"][0, 0] = float("nan")
                elif label == "shape":
                    st["model"]["grid.weight"] = torch.ones(3)
                else:
                    st["history"][0]["train_combined_vs_own_target"] = float("inf")
                A.check_endpoint(st, self.source, 0, "direct", self.order, "fixture")

    @staticmethod
    def results_fixture():
        plan = json.loads(A.D.PLAN.read_text())
        folds = {n: f["fold"] for f in plan["folds"] for n in f["adaptation_OOF"]}
        per = {s["name"]: {"fold": folds[s["name"]], "session": s["session"],
                            "r11": {"mean_dE00": 5., "p95_dE00": 10.},
                            "direct": {"mean_dE00": 4., "p95_dE00": 8.},
                            "distill": {"mean_dE00": 3.5, "p95_dE00": 7.5},
                            "distill_minus_direct": -0.5} for s in plan["scenes"]}
        sessions = {}
        for s in plan["scenes"]:
            r = sessions.setdefault(str(s["session"]), {"n": 0, "r11": 5., "direct": 4.,
                                                        "distill": 3.5, "distill_minus_direct": -0.5})
            r["n"] += 1
        report = {"manifest_sha256": A.D.MANIFEST_SHA, "freeze_sha256": "frozen",
                  "per_scene": per, "sessions": sessions,
                  "folds": {str(f["fold"]): {"r11": 5., "direct": 4., "distill": 3.5} for f in plan["folds"]},
                  "aggregate": {"r11": {"mean_dE00": 5., "median_dE00": 5., "mean_p95_dE00": 10.},
                                "direct": {"mean_dE00": 4., "median_dE00": 4., "mean_p95_dE00": 8.},
                                "distill": {"mean_dE00": 3.5, "median_dE00": 3.5, "mean_p95_dE00": 7.5}},
                  "regressions_vs_r11": {"direct": [], "distill": []},
                  "screen": {"mean_diff": -0.5, "median_diff": -0.5, "p95_diff": -0.5,
                             "max_session_diff": -0.5, "regressions_vs_r11": {"direct": 0, "distill": 0},
                             "checks": {k: True for k in ("mean", "median", "p95", "regressions", "sessions")},
                             "pilot_screen_pass": True}}
        return report, plan

    def test_valid_result_and_tamper_refusals(self):
        report, plan = self.results_fixture()
        self.assertTrue(A.check_results(report, plan, "frozen")["pilot_screen_pass"])
        for label in ("case", "aggregate", "nan", "string", "fold", "verdict", "missing_check", "freeze"):
            with self.subTest(label=label), self.assertRaises((ValueError, KeyError)):
                r = copy.deepcopy(report)
                first = next(iter(r["per_scene"]))
                if label == "case":
                    del r["per_scene"][first]
                elif label == "aggregate":
                    r["aggregate"]["distill"]["mean_dE00"] = 1.
                elif label in ("nan", "string"):
                    r["per_scene"][first]["distill"]["mean_dE00"] = float("nan") if label == "nan" else "3.5"
                elif label == "fold":
                    r["per_scene"][first]["fold"] = 999
                elif label == "verdict":
                    r["screen"]["pilot_screen_pass"] = "true"
                elif label == "missing_check":
                    del r["screen"]["checks"]["sessions"]
                else:
                    r["freeze_sha256"] = "changed"
                A.check_results(r, plan, "frozen")


if __name__ == "__main__":
    unittest.main()
