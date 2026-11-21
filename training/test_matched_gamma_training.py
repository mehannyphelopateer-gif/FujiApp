"""Synthetic contract tests; no real photo or selected checkpoint is opened."""
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

import matched_gamma_data as D
from test_fjlrg_v2 import make_case, H64, WASM


class PairContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data, provenance, pixels = make_case()
        legacy = self.root / "old.fjlrg"
        linear = self.root / "new.fjlrg"
        target = self.root / "target.npy"
        legacy.write_bytes(struct.pack("<8sHHIIHHII", b"FJLRGB16", 1, 32, 4, 3, 3, 16, 72, 3) + pixels.tobytes())
        linear.write_bytes(data)
        D.V.sidecar_path(linear).write_text(json.dumps(provenance))
        np.save(target, pixels.astype(np.float32) / np.float32(65535))
        self.row = {"name": "fixture", "role": "fit", "fjlrg_wh": [4, 3], "captureDynamicRange": 400,
                    "target_path": "target.npy", "npy_sha256": D.sha(target),
                    "inputs": {"legacy": {"path": "old.fjlrg", "sha256": D.sha(legacy)},
                               "linear": {"path": "new.fjlrg", "sha256": D.sha(linear),
                                          "sidecar_sha256": D.sha(D.V.sidecar_path(linear)),
                                          "source_raf_sha256": H64, "wasm_sha256": WASM}}}
        self.m = {"scenes": [self.row]}
        self.root_patch = patch.object(D, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_same_fixture_pixels_targets_conditions(self):
        a = D.Pairs(self.m, "legacy", "fit").get("fixture")
        b = D.Pairs(self.m, "linear", "fit").get("fixture")
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(tuple(a[0].shape), (1, 3, 3, 4))
        self.assertTrue(torch.equal(a[0], a[1]))
        self.assertEqual(a[2].item(), 1)

    def test_refuse_unknown_role_arm_and_name(self):
        for role in ("monitor", "finalHeldOut", "K", "T8", "dr100Eval"):
            with self.assertRaisesRegex(ValueError, "refused"):
                D.Pairs(self.m, "legacy", role)
        with self.assertRaisesRegex(ValueError, "Unknown input arm"):
            D.Pairs(self.m, "guess")
        with self.assertRaisesRegex(ValueError, "refused"):
            D.Pairs(self.m, "legacy", "development").get("fixture")
        with self.assertRaisesRegex(ValueError, "refused"):
            D.Pairs(self.m, "legacy").get("outside")

    def test_hash_and_arm_contract_refusals(self):
        bad = copy.deepcopy(self.m)
        bad["scenes"][0]["npy_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Target cache hash"):
            D.Pairs(bad, "linear").get("fixture")
        bad = copy.deepcopy(self.m)
        bad["scenes"][0]["inputs"]["legacy"] = self.row["inputs"]["linear"]
        with self.assertRaisesRegex(ValueError, "unsupported format version 2"):
            D.Pairs(bad, "legacy").get("fixture")
        bad = copy.deepcopy(self.m)
        bad["scenes"][0]["inputs"]["linear"]["source_raf_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Source RAF SHA256"):
            D.Pairs(bad, "linear").get("fixture")

    def test_nonfinite_target_refused_even_with_current_hash(self):
        target = self.root / "target.npy"
        values = np.zeros((3, 4, 3), np.float32)
        values[0, 0, 0] = np.nan
        np.save(target, values)
        self.row["npy_sha256"] = D.sha(target)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            D.Pairs(self.m, "legacy").get("fixture")


class Pairing(unittest.TestCase):
    def test_order_is_independent_of_global_rng(self):
        names = [str(i) for i in range(467)]
        a = D.make_orders(names, 42)
        torch.manual_seed(897)
        torch.rand(100)
        b = D.make_orders(names, 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)
        self.assertTrue(all(sorted(x) == sorted(names) for x in a))
        self.assertNotEqual(a, D.make_orders(names, 43))

    def test_initialisation_hash_pairing(self):
        config = {"grid_spatial": 8, "grid_luma": 9, "base_ch": 16, "low_res": 256,
                  "refinement_base_ch": 8, "refinement_max_delta": .08}
        a = D.fresh_model(config, 42)
        b = D.fresh_model(config, 42)
        self.assertEqual(D.state_digest(a.state_dict()), D.state_digest(b.state_dict()))
        self.assertNotEqual(D.state_digest(a.state_dict()), D.state_digest(D.fresh_model(config, 43).state_dict()))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 115719)

    def test_write_once(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "record.json"
            D.write_once_json(path, {"n": 1})
            with self.assertRaises(FileExistsError):
                D.write_once_json(path, {"n": 2})
            self.assertEqual(json.loads(path.read_text()), {"n": 1})


class TrainerAndMetrics(unittest.TestCase):
    def test_real_checkpoint_loop_resumes_exactly_across_boundaries(self):
        import run_matched_gamma_training as T
        for stop in (7, 10, 13):
            with self.subTest(stop=stop):
                result = T.fixture_exact_resume(interrupt_after=stop)
                self.assertTrue(result["equal"] and result["stale_refused"])
                self.assertEqual(result["max_abs_diff"], 0)

    def test_nonfinite_loss_and_gradient_stop_before_step(self):
        import run_matched_gamma_training as T
        model = torch.nn.Conv2d(3, 3, 1)
        opt = T.make_optimizer(model, .001)
        before = D.state_digest(model.state_dict())
        x, y, cond = torch.ones(1, 3, 4, 4), torch.zeros(1, 3, 4, 4), torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "Non-finite loss"):
            T.update(model, opt, x * float("nan"), y, cond, lambda m, a, c: m(a))
        self.assertEqual(before, D.state_digest(model.state_dict()))
        hook = model.weight.register_hook(lambda g: g * float("nan"))
        try:
            with self.assertRaisesRegex(ValueError, "Non-finite gradient"):
                T.update(model, opt, x, y, cond, lambda m, a, c: m(a))
            self.assertEqual(before, D.state_digest(model.state_dict()))
        finally:
            hook.remove()

    def test_paired_metric_arithmetic(self):
        import eval_matched_gamma_training as E
        legacy = {"scenes": {str(i): {"metrics": {"mean_dE00": a, "p95_dE00": b}}
                              for i, (a, b) in enumerate([(2, 5), (4, 8)])}}
        linear = {"scenes": {str(i): {"metrics": {"mean_dE00": a, "p95_dE00": b}}
                              for i, (a, b) in enumerate([(1, 3), (5, 9)])}}
        result = E._paired(legacy, linear, ["0", "1"])
        self.assertEqual(result["delta_mean_dE00"], 0)
        self.assertEqual(result["delta_mean_p95_dE00"], -.5)
        self.assertEqual(result["scenes_linear_worse"], 1)
        self.assertEqual(result["legacy_median_scene_dE00"], 3)
        self.assertIsNone(E._mean([None]))

    def test_cached_eval_refuses_tampering(self):
        import eval_matched_gamma_training as E
        source = {"name": "fixture", "role": "fit", "shoot": "fixture", "session": 99, "captureDynamicRange": 400}
        m = {"_manifest_sha256": "test", "scenes": [source]}
        row = {k: source[k] for k in ("role", "shoot", "session", "captureDynamicRange")}
        row["metrics"] = {"mean_dE00": 2., "p95_dE00": 5., "linear_l1_unclamped": .01}
        report = {"manifest_sha256": "test", "seed": 42, "arm": "linear", "checkpoint_sha256": "endpoint", "scenes": {"fixture": row}}
        E.validate_eval(m, 42, "linear", report, "endpoint")
        for bad in (dict(report, scenes={}), dict(report, seed=43)):
            with self.assertRaises(ValueError):
                E.validate_eval(m, 42, "linear", bad, "endpoint")
        bad = copy.deepcopy(report)
        bad["scenes"]["fixture"]["metrics"]["mean_dE00"] = float("nan")
        with self.assertRaises(ValueError):
            E.validate_eval(m, 42, "linear", bad, "endpoint")


if __name__ == "__main__":
    unittest.main()
