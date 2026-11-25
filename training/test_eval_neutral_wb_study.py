"""Synthetic CPU-only tests for eval_neutral_wb_study report arithmetic, screen, labels and eval validation.

Uses metadata-only synthetic rows and synthetic metric dicts; no model, checkpoint, input or target is opened.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import neutral_wb_study_data as D
import run_neutral_wb_study as T
import eval_neutral_wb_study as E
from test_run_neutral_wb_study import synthetic_manifest

SEEDS = (42, 43, 44)


def evals_for(m, delta, p95_delta=None, version="e" * 64):
    """delta(scene, seed) -> neutral minus G mean dE00. Dyadic values keep the arithmetic exact."""
    p95_delta = p95_delta or delta
    out = {}
    for s in SEEDS:
        G = {"evaluator_sha256": version, "checkpoint_sha256": f"G{s}", "scenes": {}}
        N = {"evaluator_sha256": version, "checkpoint_sha256": f"N{s}", "scenes": {}}
        for sc in m["scenes"]:
            G["scenes"][sc["name"]] = {"metrics": {"mean_dE00": 5.0, "p95_dE00": 10.0}}
            N["scenes"][sc["name"]] = {"metrics": {"mean_dE00": 5.0 + delta(sc, s), "p95_dE00": 10.0 + p95_delta(sc, s)}}
        out[(s, "G")], out[(s, "neutral")] = G, N
    return out


def by_subgroup(treated, identity, fit=0.0, seed_override=None):
    def f(sc, s):
        if seed_override and s in seed_override:
            return seed_override[s](sc,s)
        if sc["role"] == "fit":
            return fit
        return treated if sc["policy_subgroup"] == "treated" else identity
    return f


class Report(unittest.TestCase):
    def setUp(self):
        self.m = synthetic_manifest(with_orders=False)

    def test_paired_is_neutral_minus_G(self):
        G = {"scenes": {"a": {"metrics": {"mean_dE00": 2., "p95_dE00": 5.}}, "b": {"metrics": {"mean_dE00": 4., "p95_dE00": 8.}}}}
        N = {"scenes": {"a": {"metrics": {"mean_dE00": 1., "p95_dE00": 3.}}, "b": {"metrics": {"mean_dE00": 5., "p95_dE00": 9.}}}}
        r = E._paired(G, N, ["a", "b"])
        self.assertEqual(r["delta_mean_dE00"], 0)
        self.assertEqual(r["delta_mean_p95_dE00"], -.5)
        self.assertEqual(r["scenes_neutral_worse"], 1)
        self.assertEqual(r["scenes_neutral_better"], 1)
        self.assertEqual(r["G_median_scene_dE00"], 3)
        self.assertEqual(E._paired(G, N, []), {"n": 0})

    def test_all_five_pass(self):
        r = E.build_report(self.m, evals_for(self.m, by_subgroup(-0.5, 0.0625)))
        self.assertEqual(r["status"], "complete")
        c = r["screening"]["criteria"]
        self.assertEqual(len(c), 5)
        self.assertTrue(all(v["pass"] for v in c.values()), c)
        self.assertTrue(r["screening"]["passed"])
        pd = r["pooled_across_seeds"]["development"]
        self.assertAlmostEqual(pd["delta_mean_dE00"], (79 * -0.5 + 37 * 0.0625) / 116)
        self.assertEqual(r["pooled_across_seeds"]["development_treated79"]["delta_mean_dE00"], -0.5)
        self.assertEqual(r["pooled_across_seeds"]["development_identity37"]["delta_mean_dE00"], 0.0625)
        self.assertEqual(r["pooled_across_seeds"]["development_DR100_out_of_support"]["n_per_seed"], 12)
        self.assertEqual(r["pooled_across_seeds"]["development_identity37_DR200"]["n_per_seed"], 25)
        self.assertEqual(len(r["per_scene_all"]), 571)
        self.assertEqual(len(r["regressions_development_across_seeds"]), 37)
        self.assertEqual(r["regressions_fit_across_seeds"], [])
        self.assertEqual(set(r["pooled_across_seeds"]["per_dr_development"]), {"100", "200", "400"})
        self.assertEqual(len(r["pooled_across_seeds"]["per_session_development"]), 4)
        self.assertEqual(len(r["pooled_across_seeds"]["per_session_fit"]), 10)
        json.dumps(r, allow_nan=False)

    def test_labels_are_explicit_metadata(self):
        r = E.build_report(self.m, evals_for(self.m, by_subgroup(-0.5, 0.0625)))
        self.assertEqual(len(r["dr100"]["names"]), 12)
        self.assertIn("NEVER QUALIFIED", r["dr100"]["status"])
        self.assertIn("CONFOUND", r["confound"])
        self.assertEqual(r["subgroups"]["treated"], {"n": 79, "dr_counts": {"400": 79}})
        self.assertEqual(r["subgroups"]["identity"], {"n": 37, "dr_counts": {"100": 12, "200": 25}})
        self.assertIn("metadata", r["subgroups"]["source"])
        dr100 = [a for a in r["per_scene_all"] if a["dr100_out_of_support"]]
        self.assertEqual(len(dr100), 12)
        self.assertTrue(all(a["policy_subgroup"] == "identity" and a["role"] == "development" for a in dr100))

    def test_identity_margin_failure_in_one_seed(self):
        f = by_subgroup(-0.5, 0.0625, seed_override={43: by_subgroup(-0.5, 0.125)})
        r = E.build_report(self.m, evals_for(self.m, f))
        c = r["screening"]["criteria"]
        self.assertFalse(c["every_seed_identity37_mean_delta_dE00_le_engineering_margin"]["pass"])
        self.assertEqual(c["every_seed_identity37_mean_delta_dE00_le_engineering_margin"]["values"]["43"], 0.125)
        self.assertTrue(c["pooled_development_mean_delta_dE00_le"]["pass"])
        self.assertFalse(r["screening"]["passed"])

    def test_strict_and_pooled_failures(self):
        r = E.build_report(self.m, evals_for(self.m, by_subgroup(0.0, 0.0)))
        c = r["screening"]["criteria"]
        self.assertFalse(c["treated79_pooled_mean_delta_dE00_lt"]["pass"])  # strict < 0
        self.assertFalse(c["every_seed_development_mean_delta_dE00_lt"]["pass"])
        self.assertFalse(c["pooled_development_mean_delta_dE00_le"]["pass"])
        self.assertTrue(c["pooled_development_mean_scene_p95_delta_dE00_le"]["pass"])  # <= 0 at exactly 0
        self.assertFalse(r["screening"]["passed"])
        r = E.build_report(self.m, evals_for(self.m, by_subgroup(-0.5, 0.0625), p95_delta=by_subgroup(0.25, 0.25)))
        self.assertFalse(r["screening"]["criteria"]["pooled_development_mean_scene_p95_delta_dE00_le"]["pass"])
        self.assertFalse(r["screening"]["passed"])

    def test_incomplete_is_progress_only(self):
        ev = evals_for(self.m, by_subgroup(-0.5, 0.0625))
        ev[(44, "neutral")] = None
        r = E.build_report(self.m, ev)
        self.assertEqual(r["status"], "incomplete")
        self.assertIsNone(r["screening"])
        self.assertEqual(r["seeds_complete"], [42, 43])

    def test_mixed_evaluator_versions_and_screen_drift_refused(self):
        ev = evals_for(self.m, by_subgroup(-0.5, 0.0625))
        ev[(43, "G")]["evaluator_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "evaluator versions"):
            E.build_report(self.m, ev)
        with self.assertRaises(ValueError):
            E.build_report(dict(self.m, screen=dict(self.m["screen"], pooled_dev_mean_delta_max=-0.1)),
                           evals_for(self.m, by_subgroup(-0.5, 0.0625)))


class EvalValidation(unittest.TestCase):
    def setUp(self):
        self.m = synthetic_manifest(with_orders=False)
        self.r = {"kind": "neutral-wb-study-endpoint-eval", "manifest_sha256": self.m["_manifest_sha256"],
                  "rows_sha256": self.m["_rows_sha256"], "seed": 42, "arm": "neutral", "checkpoint_sha256": "ck",
                  "device": "cpu", "evaluator_sha256": "e" * 64,
                  "scenes": {s["name"]: {**{k: s[k] for k in E.IDENTITY_FIELDS},
                                         "metrics": {"mean_dE00": 2., "p95_dE00": 5., "linear_l1_unclamped": .01}}
                             for s in self.m["scenes"]}}

    def test_valid_and_tampered(self):
        E.validate_eval(self.m, 42, "neutral", self.r, "ck")
        bad = []
        bad.append(dict(self.r, seed=43))
        bad.append(dict(self.r, device="mps"))
        bad.append(dict(self.r, rows_sha256="0" * 64))
        r = copy.deepcopy(self.r); r["scenes"].pop(next(iter(r["scenes"]))); bad.append(r)  # noqa: E702
        r = copy.deepcopy(self.r); next(iter(r["scenes"].values()))["policy_subgroup"] = "treated"; bad.append(r)  # noqa: E702
        r = copy.deepcopy(self.r); next(iter(r["scenes"].values()))["metrics"]["mean_dE00"] = float("nan"); bad.append(r)  # noqa: E702
        for b in bad:
            with self.assertRaises(ValueError):
                E.validate_eval(self.m, 42, "neutral", b, "ck")
        with self.assertRaises(ValueError):
            E.validate_eval(self.m, 42, "neutral", self.r, "other-checkpoint")


class ReportFiles(unittest.TestCase):
    def test_no_evals_writes_progress_only(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.object(T, "CKPT", Path(td) / "ckpt"), patch.object(T, "RUNS", Path(td) / "runs"):
            m = synthetic_manifest()
            p = E.report(m)
            self.assertEqual(p, E.progress_path())
            self.assertEqual(json.loads(p.read_text())["status"], "incomplete")
            self.assertFalse(E.final_path().exists())


if __name__ == "__main__":
    unittest.main()
