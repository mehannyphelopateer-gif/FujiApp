"""Synthetic/metadata-only runner tests; no source, target or checkpoint reads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import matched_gamma_data as D
import run_gamma_encoded_guide as G
import run_matched_gamma_training as T


class RunnerTests(unittest.TestCase):
    def test_unknown_seeds_and_unpinned_manifest_refused(self):
        for seed in (True,41,45,'42'):
            with self.assertRaises(ValueError):G.paths(seed)
        with self.assertRaises(ValueError):G.load(None)
        with self.assertRaises(ValueError):G.load('0'*64)

    def test_fresh_weights_match_parent_metadata_all_seeds(self):
        # JSON metadata only: does not call the scene loader or read images.
        parent=json.loads(D.MANIFEST.read_text())
        for seed in D.SEEDS:
            model=G.construct(parent,seed,'cpu')
            self.assertEqual(D.state_digest(model.state_dict()),parent['initial_state_sha256'][str(seed)])

    def test_guided_model_exact_cpu_resume_with_stale_identity_refusal(self):
        parent=json.loads(D.MANIFEST.read_text())
        names=['a','b'];orders=[names,names[::-1]]
        torch.manual_seed(7)
        data={n:(torch.rand(1,3,17,29),torch.rand(1,3,17,29),torch.tensor([.5])) for n in names}
        ident={'assay':'synthetic-guided-resume','seed':42}
        fwd=lambda model,x,c:D.forward(model,parent,x,c,'cpu')
        def objects():
            model=G.construct(parent,42,'cpu')
            opt=T.make_optimizer(model,1e-3)
            return model,opt,T.make_scheduler(opt,2,1e-5)
        kw={'ckpt_every':1,'log_every':10000,'announce':False,'say':lambda _x:None}
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            a,oa,sa=objects()
            T.run(a,oa,sa,orders,data.__getitem__,fwd,tmp/'full.pt',ident,**kw)
            b,ob,sb=objects()
            T.run(b,ob,sb,orders,data.__getitem__,fwd,tmp/'resume.pt',ident,max_updates=3,**kw)
            c,oc,sc=objects()
            T.run(c,oc,sc,orders,data.__getitem__,fwd,tmp/'resume.pt',ident,**kw)
            self.assertEqual(D.state_digest(a.state_dict()),D.state_digest(c.state_dict()))
            self.assertEqual(sa.state_dict(),sc.state_dict())
            d,od,sd=objects()
            with self.assertRaises(ValueError):
                T.run(d,od,sd,orders,data.__getitem__,fwd,tmp/'resume.pt',dict(ident,seed=43),**kw)

    def test_no_endpoint_without_both_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            with patch.object(G,'paths',return_value=(base,base/'state.pt',base/'endpoint.pt',base/'endpoint.json')):
                with self.assertRaises(FileNotFoundError):G.validate_endpoint({}, {},42)

    def test_pair_sign_and_all_regressions_reported(self):
        def result(x):
            return {'scenes':{'a':{'metrics':x}}}
        # Use a real-shaped synthetic metric record for the existing reducer.
        from provia_color_pilot import dev_metrics
        zero=torch.zeros(1,3,3,5)
        metrics=dev_metrics(zero,zero)
        candidate=dict(metrics,mean_dE00=2.,p95_dE00=4.)
        control=dict(metrics,mean_dE00=1.,p95_dE00=3.)
        pair=G.paired(result(candidate),result(control),['a'])
        self.assertEqual(pair['per_scene'][0]['delta_mean_dE00'],1.)
        self.assertEqual(len(pair['regressions']),1)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
