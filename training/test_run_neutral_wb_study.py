"""Synthetic CPU-only contract tests for run_neutral_wb_study.

No real manifest, RAF, target array, input artifact, checkpoint, Round 11 weight or study model is opened:
the only model trained is the tiny synthetic fixture (and a 1x1 conv used for endpoint bookkeeping).
All writes go to temporary directories via the trainer's own CKPT/DIAG/RUNS attributes.
"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

import neutral_wb_study_data as D
import run_neutral_wb_study as T


def synthetic_manifest(with_orders=True):
    """571 metadata-only rows shaped like the frozen study (455 fit / 116 dev; dev 79 treated DR400,
    37 identity = 25 DR200 + 12 DR100). No paths are ever opened by the tests that use it."""
    scenes = []
    for i in range(T.FIT_N):
        kind = "GWP"[i % 3]
        scenes.append({"name": f"DSCF{i:04d}.RAF", "shoot": f"Shoot {i}", "role": "fit", "session": i % 10,
                       "captureDynamicRange": (200, 400)[i % 2], "neutral_kind": kind,
                       "policy_subgroup": "identity" if kind == "G" else "treated"})
    for j in range(T.DEV_N):
        i = T.FIT_N + j
        if j < 79:
            kind, dr = "WP"[j % 2], 400
        elif j < 104:
            kind, dr = "G", 200
        else:
            kind, dr = "G", 100
        scenes.append({"name": f"DSCF{i:04d}.RAF", "shoot": f"Shoot {i}", "role": "development", "session": 10 + j % 4,
                       "captureDynamicRange": dr, "neutral_kind": kind,
                       "policy_subgroup": "identity" if kind == "G" else "treated"})
    fit = [s["name"] for s in scenes if s["role"] == "fit"]
    m = {"_manifest_sha256": "a" * 64, "scenes": scenes, "training": copy.deepcopy(T.FROZEN_TRAINING),
         "model_config": dict(T.FROZEN_MODEL), "screen": dict(D.SCREEN), "environment": {"mps_available": False},
         "initial_state_sha256": {str(s): format(s, "064x") for s in T.FROZEN_TRAINING["seeds"]}}
    m["orders"] = ({str(s): D.make_orders(fit, s) for s in T.FROZEN_TRAINING["seeds"]} if with_orders
                   else {str(s): [fit] for s in T.FROZEN_TRAINING["seeds"]})
    m["_rows_sha256"] = D.rows_digest(scenes)
    return m


class TempRoots(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.patches = [patch.object(T, "CKPT", root / "ckpt"), patch.object(T, "DIAG", root / "diag"),
                        patch.object(T, "RUNS", root / "runs")]
        for p in self.patches:
            p.start()
        T.DIAG.mkdir(parents=True)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.temp.cleanup()


class Constants(unittest.TestCase):
    def test_frozen_budget_and_namespace(self):
        self.assertEqual(T.ARMS, ("G", "neutral"))
        self.assertEqual(T.UPDATES_PER_RUN, 9100)
        self.assertEqual((T.FIT_N, T.DEV_N, T.ALL_PAIRS), (455, 116, 571))
        self.assertEqual(T.FROZEN_TRAINING, {"seeds": [42, 43, 44], "epochs": 20, "lr": 0.001, "eta_min": 1e-5, "batch_size": 1})
        self.assertEqual(T.FROZEN_MODEL["grid_spatial"], 8)
        self.assertEqual(T.FROZEN_MODEL["grid_luma"], 9)
        self.assertEqual(T.FROZEN_MODEL["refinement_max_delta"], 0.08)
        self.assertEqual(tuple(D.SEEDS), (42, 43, 44))
        self.assertEqual(D.ARMS, T.ARMS)
        self.assertNotIn("matched-gamma", T.STATE_FORMAT + T.ENDPOINT_FORMAT + str(T.lock_path()) + str(T.run_dir(42, "G")))
        self.assertTrue(str(T.run_dir(42, "neutral")).endswith("neutral-wb-study-seed42-neutral"))


class Split(unittest.TestCase):
    def setUp(self):
        self.m = synthetic_manifest(with_orders=False)

    def test_valid_split_and_metadata_facts(self):
        names = T.validate_split(self.m)
        self.assertEqual(len(names), 455)
        f = T.metadata_facts(self.m)
        self.assertEqual(f["treated"]["n"], 79)
        self.assertEqual(f["treated"]["dr_counts"], {"400": 79})
        self.assertEqual(f["identity"]["dr_counts"], {"100": 12, "200": 25})
        self.assertEqual(len(f["dr100_development"]), 12)

    def test_tampered_splits_refused(self):
        def mutate(fn):
            m = copy.deepcopy(self.m)
            fn(m["scenes"])
            return m
        dev = lambda sc: [s for s in sc if s["role"] == "development"]  # noqa: E731
        cases = {
            "dev DR100 count": lambda sc: dev(sc)[-1].update(captureDynamicRange=200),
            "treated not DR400": lambda sc: dev(sc)[0].update(captureDynamicRange=200),
            "fit DR100": lambda sc: sc[1].update(captureDynamicRange=100),
            "session leak": lambda sc: sc[0].update(session=10),
            "subgroup relabel": lambda sc: dev(sc)[0].update(policy_subgroup="identity"),
            "kind unknown": lambda sc: sc[0].update(neutral_kind="E", policy_subgroup="treated"),
            "fit count": lambda sc: sc.pop(0),
            "duplicate": lambda sc: sc[1].update(name=sc[0]["name"]),
        }
        for label, fn in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    T.validate_split(mutate(fn))

    def test_config_drift_refused(self):
        T.check_training_config(self.m)
        for key, val in (("training", dict(self.m["training"], lr=0.002)),
                         ("model_config", dict(self.m["model_config"], base_ch=32)),
                         ("screen", dict(self.m["screen"], identity37_each_seed_mean_delta_max=0.2))):
            with self.subTest(key):
                with self.assertRaises(ValueError):
                    T.check_training_config(dict(self.m, **{key: val}))


class Identity(unittest.TestCase):
    def test_every_identity_component_matters(self):
        m = synthetic_manifest()
        base = T.identity_for(m, 42, "G", m["orders"]["42"])
        self.assertEqual(base["fit_count"] * base["epochs"], 9100)
        self.assertEqual(base["trainer_sha256"], D.sha(T.CODE))
        variants = [T.identity_for(m, 43, "G", m["orders"]["43"]),
                    T.identity_for(m, 42, "neutral", m["orders"]["42"]),
                    T.identity_for(m, 42, "G", m["orders"]["43"]),
                    T.identity_for(dict(m, _manifest_sha256="b" * 64), 42, "G", m["orders"]["42"]),
                    T.identity_for(dict(m, _rows_sha256="c" * 64), 42, "G", m["orders"]["42"]),
                    T.identity_for(dict(m, initial_state_sha256={"42": "d" * 64}), 42, "G", m["orders"]["42"]),
                    T.identity_for(dict(m, model_config=dict(m["model_config"], low_res=128)), 42, "G", m["orders"]["42"])]
        for v in variants:
            self.assertNotEqual(base, v)


class Trainer(unittest.TestCase):
    def test_cpu_fixture_exact_resume_including_rng_optimizer_schedule(self):
        for stop in (5, 7, 10, 13):
            with self.subTest(stop=stop):
                r = T.fixture_exact_resume(interrupt_after=stop)
                self.assertTrue(r["equal"])
                self.assertEqual(r["max_abs_diff"], 0)
                self.assertTrue(r["stale_refused"])
                self.assertTrue(r["unknown_refused"])
                self.assertFalse(r["interrupted_complete"])

    def test_nonfinite_loss_and_gradient_stop_before_step(self):
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

    def test_orders_must_be_fit_permutations(self):
        names = ["a", "b", "c"]
        T.validate_orders([names, list(reversed(names))], names, 2)
        for bad in ([names], [names, ["a", "b", "d"]], [names, ["a", "b"]]):
            with self.assertRaises(ValueError):
                T.validate_orders(bad, names, 2)


class AllowlistRefusals(unittest.TestCase):
    def test_dev_unknown_protected_refused_before_pixels(self):
        m = synthetic_manifest(with_orders=False)
        # No path in this manifest exists: a refusal for any reason other than the allowlist would carry a
        # different message, so the text check proves refusal happens before any file/pixel access.
        out = T.check_refusals(m)
        self.assertEqual(set(out), {"G", "neutral"})
        self.assertIn("DSCF9999.RAF", out["G"]["refused_names"])
        self.assertEqual(out["neutral"]["refused_roles"], list(T.PROTECTED_ROLES))


class Endpoints(TempRoots):
    def setUp(self):
        super().setUp()
        self.m = synthetic_manifest()
        self.model = torch.nn.Conv2d(3, 3, 1)
        self.done = {"epoch": 20, "global_updates": 9100, "history": []}

    def _write(self, seed=42, arm="G", m=None, pr=None):
        m = m or self.m
        ident = T.identity_for(m, seed, arm, m["orders"][str(seed)])
        return T.write_endpoint(self.model, ident, pr or self.done, seed, arm)

    def test_round_trip_and_refusals(self):
        ep = self._write()
        path, ck, h = T.validate_endpoint(self.m, 42, "G", ep)
        self.assertEqual(h, D.sha(ep))
        self.assertEqual(ck["meta"]["global_updates"], 9100)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self._write()
        with self.assertRaisesRegex(ValueError, "not at the fixed epoch-20"):
            self._write(seed=43, pr={"epoch": 19, "global_updates": 8645, "history": []})
        with self.assertRaisesRegex(ValueError, "no complete endpoint"):
            T.validate_endpoint(self.m, 44, "G")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            T.validate_endpoint(dict(self.m, _manifest_sha256="f" * 64), 42, "G")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            T.validate_endpoint(dict(self.m, _rows_sha256="e" * 64), 42, "G")
        with self.assertRaisesRegex(ValueError, "is not the"):
            T.validate_endpoint(self.m, 42, "G", T.endpoint_path(42, "neutral"))
        with self.assertRaisesRegex(ValueError, "Unknown arm"):
            T.validate_endpoint(self.m, 42, "legacy")

    def test_tampered_record_or_weights_refused(self):
        ep = self._write()
        rec = T.endpoint_record_path(42, "G")
        r = json.loads(rec.read_text())
        rec.write_text(json.dumps(dict(r, endpoint_sha256="0" * 64)))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            T.validate_endpoint(self.m, 42, "G")
        rec.write_text(json.dumps(r))
        ck = torch.load(ep, map_location="cpu", weights_only=True)
        ck["model"]["bias"] = ck["model"]["bias"] + 1
        torch.save(ck, ep)
        rec.write_text(json.dumps(dict(r, endpoint_sha256=D.sha(ep))))
        with self.assertRaisesRegex(ValueError, "weights differ"):
            T.validate_endpoint(self.m, 42, "G")

    def test_stale_state_file_refused_by_run(self):
        names, data, orders = T.tiny_fixture()
        model, opt, sched = T._tiny_objects(7, len(orders))
        ident = {"fixture": True}
        state = T.state_path(42, "G")
        r = T.run(model, opt, sched, orders, data.__getitem__, lambda mdl, x, c: mdl(x, c), state, ident,
                  ckpt_every=3, log_every=10 ** 9, announce=False, say=T._quiet, max_updates=4)
        self.assertFalse(r["complete"])
        self.assertTrue(state.exists())
        m2, o2, s2 = T._tiny_objects(7, len(orders))
        with self.assertRaisesRegex(ValueError, "stale"):
            T.run(m2, o2, s2, orders, data.__getitem__, lambda mdl, x, c: mdl(x, c), state, dict(ident, arm="neutral"),
                  ckpt_every=3, log_every=10 ** 9, announce=False, say=T._quiet)


class ProductionGate(TempRoots):
    def setUp(self):
        super().setUp()
        self.m = synthetic_manifest(with_orders=False)
        self.pre = {"passed": True, "manifest_sha256": self.m["_manifest_sha256"], "pairs": 571}
        self.smoke = {"passed": True, "manifest_sha256": self.m["_manifest_sha256"], "trainer_sha256": D.sha(T.CODE),
                      "data_api_sha256": D.sha(T.DATA_API),
                      "resume": {"equal": True, "stale_refused": True, "unknown_refused": True}}

    def _put(self, pre=None, smoke=None):
        if pre is not None:
            (T.DIAG / T.PREFLIGHT_NAME).write_text(json.dumps(pre))
        if smoke is not None:
            (T.DIAG / T.SMOKE_NAME).write_text(json.dumps(smoke))

    def test_gate(self):
        with patch.object(T, "committed", lambda p: True):
            with self.assertRaisesRegex(ValueError, "Missing"):
                T.check_proofs(self.m)
            self._put(pre=dict(self.pre, pairs=570))
            with self.assertRaisesRegex(ValueError, "571"):
                T.check_proofs(self.m)
            self._put(pre=self.pre)
            with self.assertRaisesRegex(ValueError, "Missing"):
                T.check_proofs(self.m)
            for bad in (dict(self.smoke, trainer_sha256="0" * 64), dict(self.smoke, passed=False),
                        dict(self.smoke, data_api_sha256="0" * 64),
                        dict(self.smoke, manifest_sha256="b" * 64),
                        dict(self.smoke, resume={"equal": True, "stale_refused": False, "unknown_refused": True}),
                        dict(self.smoke, resume={"equal": False, "stale_refused": True, "unknown_refused": True})):
                self._put(smoke=bad)
                with self.assertRaises(ValueError):
                    T.check_proofs(self.m)
            self._put(smoke=self.smoke)
            T.check_proofs(self.m)
            with self.assertRaisesRegex(ValueError, "Unknown arm"):
                T.train_arm(self.m, 42, "legacy")
        with patch.object(T, "committed", lambda p: False):
            with self.assertRaisesRegex(ValueError, "committed"):
                T.check_proofs(self.m)
            T.check_preflight(self.m, require_committed=False)  # smoke may run on an uncommitted passed preflight

    def test_train_refuses_before_any_model_without_proofs(self):
        # The gate raises before the lock, the R11 check, any model or any pair access.
        with patch.object(T, "committed", lambda p: True):
            with self.assertRaisesRegex(ValueError, "Missing"):
                T.train_arm(self.m, 42, "G")
        self.assertFalse(T.lock_path().exists())
        self.assertFalse(T.run_dir(42, "G").exists())


class Lock(TempRoots):
    def test_lock_reentrant_stale_and_live(self):
        with T.global_lock():
            self.assertEqual(T.lock_path().read_text(), str(os.getpid()))
            with T.global_lock():
                self.assertTrue(T.lock_path().exists())
        self.assertFalse(T.lock_path().exists())
        T.lock_path().parent.mkdir(parents=True, exist_ok=True)
        T.lock_path().write_text("999999999")
        with T.global_lock():
            pass
        T.lock_path().write_text(str(os.getppid()))
        with self.assertRaisesRegex(ValueError, "holds"):
            with T.global_lock():
                pass
        T.lock_path().unlink()


if __name__ == "__main__":
    unittest.main()
