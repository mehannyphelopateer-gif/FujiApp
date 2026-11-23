"""Synthetic-only invariants for the isolated guide intervention."""
import unittest

import torch

from gamma_encoded_guide import (GammaEncodedGuideHybrid, apply_with_guide,
                                encoded_guide, qualified_table)
from matched_gamma_data import fresh_model, state_digest
from model import apply_bilateral_grid, compute_luma_guide, resize_for_network

CONFIG = dict(grid_spatial=8, grid_luma=9, low_res=256, base_ch=16,
              refinement_base_ch=8, refinement_max_delta=.08)


class GuideTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)

    def test_every_uint16_code_is_restored_and_curve_bounded_monotone(self):
        codes = torch.arange(65536)
        rgb = (codes.float() / 65535.).view(1, 1, 256, 256).expand(1, 3, 256, 256)
        self.assertTrue(torch.equal(torch.round(rgb * 65535.).long()[0, 0].flatten(), codes))
        table = qualified_table()
        guide = encoded_guide(rgb, table)[0].flatten()
        torch.testing.assert_close(guide, table, atol=1.2e-7, rtol=0)
        self.assertTrue(bool((table[1:] >= table[:-1]).all()))
        self.assertEqual(float(table[0]), 0.)
        self.assertLessEqual(float(table[-1]), 1.)

    def test_boundary_clamp_is_finite(self):
        x = torch.tensor([-1., 0., .018, .5, 1., 2.]).view(1, 1, 2, 3).expand(1, 3, 2, 3)
        guide = encoded_guide(x, qualified_table())
        self.assertTrue(bool(torch.isfinite(guide).all()))
        self.assertEqual(float(guide[0, 0, 0]), 0.)
        self.assertEqual(float(guide[0, -1, -1]), float(guide[0, -1, -2]))

    def test_explicit_slicing_matches_original_bit_exact_and_gradients(self):
        for h, w in [(17, 29), (29, 17), (1, 5)]:
            rgb = torch.rand(1, 3, h, w)
            a = torch.randn(1, 12, 9, 8, 8, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            old = apply_bilateral_grid(a, rgb)
            new = apply_with_guide(b, rgb, compute_luma_guide(rgb))
            self.assertTrue(torch.equal(old, new))
            old.square().mean().backward(); new.square().mean().backward()
            self.assertTrue(torch.equal(a.grad, b.grad))

    def test_same_initial_weights_and_parameter_count_all_seeds(self):
        for seed in (42, 43, 44):
            old = fresh_model(CONFIG, seed)
            torch.manual_seed(seed)
            new = GammaEncodedGuideHybrid(**CONFIG, condition_channels=1)
            self.assertEqual(state_digest(old.state_dict()), state_digest(new.state_dict()))
            self.assertEqual(sum(p.numel() for p in old.parameters()), 115719)
            self.assertEqual(sum(p.numel() for p in new.parameters()), 115719)

    def test_encoder_affine_and_refiner_receive_linear_rgb(self):
        new = GammaEncodedGuideHybrid(**CONFIG, condition_channels=1)
        rgb = torch.rand(1, 3, 37, 53) * .1
        cond = torch.tensor([.5])
        observed = {}
        def record_grid(_module, inputs): observed['low'] = inputs[0].detach().clone()
        def record_refiner(_module, inputs): observed['full'] = inputs[0].detach().clone()
        h1 = new.grid.register_forward_pre_hook(record_grid)
        h2 = new.refinement.register_forward_pre_hook(record_refiner)
        final, gout, delta = new(rgb, condition=cond)
        h1.remove(); h2.remove()
        self.assertTrue(torch.equal(observed['low'], resize_for_network(rgb, 256)))
        self.assertTrue(torch.equal(observed['full'], rgb))
        self.assertTrue(torch.equal(final, gout))  # zero-initialized refiner
        self.assertEqual(float(delta.detach().abs().max()), 0.)
        expected = apply_with_guide(new.grid(resize_for_network(rgb,256),cond), rgb,
                                    encoded_guide(rgb, new.gamma_guide_table))
        self.assertTrue(torch.equal(gout, expected))
        final.square().mean().backward()
        grads = [p.grad for p in new.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
        self.assertIsNone(new.gamma_guide_table.grad)

    def test_zero_and_identity_grid_preserve_linear_values(self):
        rgb = torch.rand(1, 3, 17, 21)
        grid = torch.zeros(1, 12, 9, 8, 8)
        guide = encoded_guide(rgb, qualified_table())
        self.assertTrue(torch.equal(apply_with_guide(grid,rgb,guide), torch.zeros_like(rgb)))
        for channel, index in enumerate((0, 5, 10)):
            grid[:, index] = 1
        torch.testing.assert_close(apply_with_guide(grid,rgb,guide),rgb,atol=2e-7,rtol=0)


if __name__ == '__main__':
    unittest.main()
