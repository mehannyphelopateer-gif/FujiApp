"""Synthetic loader/routing fixtures; never read real photos, targets or weights."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import diagnose_encoder_information as R
import encoder_information_metrics as M
from model import HybridPredictor,resize_for_network


def fixture():
    row={'name':'allowed.RAF','shoot':'Shoot 1','role':'fit','session':1,'captureDynamicRange':400,
         'fjlrg_wh':[11,9],'target_path':'FORBIDDEN-TARGET.npy',
         'inputs':{'legacy':{'path':'fake-legacy.fjlrg','sha256':'h'},
                   'linear':{'path':'fake-G.fjlrg','sha256':'h','sidecar_sha256':'h',
                             'source_raf_sha256':'source','wasm_sha256':'wasm'}}}
    dev=dict(row,name='development.RAF',role='development')
    return {'scenes':[row,dev]},[row]


class EncoderRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(53)

    def test_role_and_cohort_refusals_precede_file_access(self):
        parent,cohort=fixture()
        with patch.object(R.D,'sha',side_effect=AssertionError('file access')):
            with self.assertRaises(ValueError):R.Inputs(parent,[parent['scenes'][1]])
            with self.assertRaises(ValueError):R.Inputs(parent,cohort*2)
            loader=R.Inputs(parent,cohort)
            for name,arm in [('development.RAF','linear'),('finalHeldOut','legacy'),('allowed.RAF','W')]:
                with self.assertRaises(ValueError):loader.get(name,arm)

    def test_both_input_codecs_never_call_target_or_pair_loader(self):
        parent,cohort=fixture();loader=R.Inputs(parent,cohort)
        image=np.full((9,11,3),.25,dtype=np.float32)
        frame=SimpleNamespace(provenance={'capture_dynamic_range':400},width=11,height=9,
                              to_unit_float32=lambda:image)
        with patch.object(R.D,'sha',return_value='h'), \
             patch.object(R,'load_fjlrg_linear',return_value=(image,11,9)), \
             patch.object(R.V,'load_fjlrg_v2',return_value=frame) as load, \
             patch.object(R.D,'Pairs',side_effect=AssertionError('target-bearing loader')), \
             patch.object(R.np,'load',side_effect=AssertionError('target read')):
            for arm in ('legacy','linear'):
                x,c=loader.get('allowed.RAF',arm)
                self.assertEqual(tuple(x.shape),(1,3,9,11))
                self.assertTrue(torch.equal(x,torch.full_like(x,.25)))
                self.assertEqual(float(c[0]),1.)
            self.assertEqual(load.call_args.kwargs,{'expected_contract':R.V.CONTRACT_ASSHOT,
                         'expected_source_sha256':'source','expected_wasm_sha256':'wasm'})

    def test_input_hash_and_geometry_refused(self):
        parent,cohort=fixture();loader=R.Inputs(parent,cohort)
        with patch.object(R.D,'sha',return_value='wrong'):
            with self.assertRaises(ValueError):loader.get('allowed.RAF','legacy')
        with patch.object(R.D,'sha',return_value='h'),patch.object(R,'load_fjlrg_linear',return_value=(np.zeros((9,11,3),dtype=np.float32),12,9)):
            with self.assertRaises(ValueError):loader.get('allowed.RAF','legacy')

    def test_encode_is_exact_original_grid_and_only_encoder(self):
        model=HybridPredictor(condition_channels=1).eval()
        x=torch.rand(1,3,256,256);c=torch.tensor([.5])
        with torch.inference_mode(),patch.object(model.refinement,'forward',side_effect=AssertionError('refinement forbidden')):
            expected=model.grid(x,c)
            grid,first=R.encode(model,x,c)
            self.assertTrue(torch.equal(grid,expected))
            planes=c.reshape(1,1,1,1).expand(1,1,256,256)
            self.assertTrue(torch.equal(first,model.grid.encoder[0](torch.cat([x,planes],dim=1))))
            self.assertEqual(len(model.grid.encoder[0]._forward_hooks),0)
            with self.assertRaises(ValueError):R.encode(model,x,None)
            self.assertEqual(len(model.grid.encoder[0]._forward_hooks),0)

    def test_perturbations_keep_original_shape_and_do_not_clamp_gain(self):
        x=torch.rand(1,3,31,47)*.5+.4
        versions=R.perturbations(x)
        self.assertTrue(torch.equal(versions['unfiltered']['base'],resize_for_network(x,256)))
        for parts in versions.values():
            self.assertEqual(set(parts['shifts']),{f'{dy},{dx}' for dy,dx in R.SHIFTS})
            self.assertTrue(torch.equal(parts['gains']['0.5'],parts['base']*.5))
            self.assertTrue(torch.equal(parts['gains']['2.0'],parts['base']*2))
            self.assertGreater(float(parts['gains']['2.0'].max()),1.)
            self.assertTrue(all(v.shape==(1,3,256,256) for v in parts['shifts'].values()))

    def test_same_normalizer_used_for_filtered_and_unfiltered(self):
        def stats(n):
            base=torch.zeros(2);new=torch.tensor([float(n),0.])
            return {'grid':M.vector_response(base,new),'block1':M.vector_response(base,new)}
        def branch(n):return {'gains':{str(k):stats(n) for k in R.GAINS},
                              'shifts':{f'{dy},{dx}':stats(n) for dy,dx in R.SHIFTS},'DR05_to_10':stats(n)}
        cases=[{'models':{'a':{'unfiltered':branch(3),'box7':branch(1)}}}]
        result=R.summary(cases,{'a':{'unfiltered':2.,'box7':.25}},[{'id':'a'}])['a']
        self.assertEqual(result['unfiltered']['gains']['0.5']['mean_grid_normalized_l2'],1.5)
        self.assertEqual(result['box7']['gains']['0.5']['mean_grid_normalized_l2'],.5)
        self.assertAlmostEqual(result['box7_over_unfiltered_phase_l2_ratio']['median'],1/3)

    def test_manifest_missing_or_changed_is_refused(self):
        with self.assertRaises(ValueError):R.load(None)
        with self.assertRaises(ValueError):R.load('0'*64)

    def test_encoded_endpoint_uses_same_encoder_state_keys(self):
        from gamma_encoded_guide import GammaEncodedGuideHybrid
        old=HybridPredictor(condition_channels=1)
        new=GammaEncodedGuideHybrid(condition_channels=1)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        old.load_state_dict(new.state_dict(),strict=True)


if __name__=='__main__':unittest.main()
