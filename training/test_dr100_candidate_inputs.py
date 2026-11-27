"""Synthetic input-scope/canary checks, never loading any photo or model."""
import copy
import unittest
import dr100_candidate_inputs as G


class Scope(unittest.TestCase):
    def test_canaries_are_fixed_metadata_groups(self):
        rows=[{'name':f'DSCF{i}.RAF','source_wb_mode':1 if i%4 else 4080} for i in range(1424,1441,2)]
        chosen=G.canaries(rows)
        self.assertEqual(len(chosen),2)
        self.assertEqual(chosen,G.canaries(list(reversed(rows))))
        self.assertEqual({s['source_wb_mode'] for s in rows if s['name'] in chosen},{1,4080})

    def test_G_canonical_options_keep_true_linear_gamma(self):
        opts=G.V.canonical_options(G.V.CONTRACT_ASSHOT,[529,302,555,0])
        self.assertEqual(opts['gamm'],[1,1,0,0,0,0]);self.assertEqual(opts['outputBps'],16)
        self.assertTrue(opts['useCameraWb']);self.assertFalse(opts['halfSize']);self.assertTrue(opts['noAutoBright'])


if __name__=='__main__':unittest.main()
