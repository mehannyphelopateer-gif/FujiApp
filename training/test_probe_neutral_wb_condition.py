"""Synthetic-only guard, accounting and phase-order tests; no source media."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import probe_neutral_wb_condition as P
import verify_neutral_wb_condition_probe as V


def fixture():
    identity = [dict(name=f'id{i}', role='development', session=14,
                     neutral_kind='G', policy_subgroup='identity',
                     captureDynamicRange=100 if i < 12 else 200,
                     inputs={'G': {'path': str(i)}, 'neutral': {'path': str(i)}})
                for i in range(37)]
    other = [dict(name=f'fit{i}', role='fit', session=3, neutral_kind='W', policy_subgroup='treated')
             for i in range(455)]
    other += [dict(name=f'treated{i}', role='development', session=7,
                   neutral_kind='P', policy_subgroup='treated') for i in range(79)]
    parent = {'scenes': identity+other}
    evals = {(s, a): {'scenes': {}} for s in V.SEEDS for a in V.ARMS}
    actuals, counters = [], []
    for s in V.SEEDS:
        for a in V.ARMS:
            for row in identity:
                actual = 0.0 if row['captureDynamicRange'] == 100 else 0.5
                offset = 0.2 if a == 'neutral' else 0.0
                for c in V.CONDITIONS:
                    metrics = {'mean_dE00': 2+c+offset, 'p95_dE00': 4+c+offset,
                               'linear_l1_unclamped': 0.02+c/10+offset/100}
                    yp, yt, baseline = 0.2+c/10, 0.3, 0.2+actual/10
                    b = {'meanYpred': yp, 'meanYtarget': yt, 'B_err': yp-yt,
                         'B_stops': math.log2(yp/yt), 'B_shift': math.log2(yp/baseline),
                         'null_reasons': {'B_stops':None, 'B_shift':None}}
                    cell = {'seed': s, 'arm': a, 'name': row['name'], 'sourceDR': row['captureDynamicRange'],
                            'condition': c, 'is_actual': c == actual, 'metrics': metrics, 'brightness': b}
                    if c == actual:
                        actuals.append(cell);evals[(s,a)]['scenes'][row['name']] = {'metrics': metrics.copy()}
                    else:counters.append(cell)
    return parent, evals, actuals+counters


class Guards(unittest.TestCase):
    def test_unknown_fit_treated_protected_name_before_pair_io(self):
        m, _, _ = fixture();scope = {'source_names': [r['name'] for r in V.source_rows(m)]}
        with patch.object(P.D, 'Pairs') as pairs:
            for name in ('fit0', 'treated0', 'unknown', 'finalHeldOut', 'dr100Eval'):
                with self.assertRaisesRegex(ValueError, 'allowlist'):P.allowlisted_pair(m, scope, name)
            pairs.assert_not_called()

    def test_poisoned_name_list_still_refuses_before_io(self):
        m, _, _ = fixture()
        with patch.object(P.D, 'Pairs') as pairs:
            with self.assertRaisesRegex(ValueError, 'identity'):P.allowlisted_pair(m, {'source_names':['fit0']}, 'fit0')
            pairs.assert_not_called()

    def test_wrong_actual_condition_refused(self):
        m, _, _ = fixture();scope = {'source_names':['id0']};x = torch.zeros(1,3,3,5)
        with patch.object(P.D, 'Pairs') as pairs:
            pairs.return_value.get.return_value = (x, x, torch.tensor([0.5]))
            with self.assertRaisesRegex(ValueError, 'condition'):P.allowlisted_pair(m, scope, 'id0')

    def test_reproduction_failure_never_calls_counterfactual(self):
        called = []
        def fail():
            called.append('actual');raise ValueError('bad actual reproduction')
        with self.assertRaises(ValueError):P.phases(fail, lambda: called.append('counter'))
        self.assertEqual(called, ['actual'])

    def test_successful_phase_order(self):
        called = [];P.phases(lambda: called.append('actual222'), lambda: called.append('counter444'))
        self.assertEqual(called, ['actual222','counter444'])

    def test_metric_reproduction_tolerance(self):
        old = dict(mean_dE00=1.0, p95_dE00=2.0, linear_l1_unclamped=0.1)
        P.reproduce(old, old)
        with self.assertRaisesRegex(ValueError, 'reproduction'):P.reproduce({**old,'mean_dE00':1.01}, old)

    def test_scope_grid_drift_before_evidence_io(self):
        scope = {'format':'neutral-wb-condition-probe-v1', 'parent_manifest_sha256':P.PARENT_SHA,
                 'conditions':[0,1]}
        with patch.object(P, 'bound', return_value=scope),patch.object(P, 'completed_evidence') as evidence:
            with self.assertRaisesRegex(ValueError, 'constants'):P.load_scope('a'*64)
            evidence.assert_not_called()

    def test_identity_metadata_drift(self):
        m, _, _ = fixture();m['scenes'][0]['inputs']['neutral']['path'] = 'different'
        with self.assertRaisesRegex(ValueError, 'bytes'):V.source_rows(m)


class Accounting(unittest.TestCase):
    def test_full_synthetic666_and_known_gap(self):
        m, ev, cells = fixture();r = V.scalar_accounting(m, cells, ev)
        self.assertEqual((r['cell_count'],r['reproduction_cells'],r['counterfactual_cells']), (666,222,444))
        self.assertEqual(len(r['independent_groups']), 36)
        for row in r['independent_paired_gaps']:self.assertAlmostEqual(row['mean_gap'], 0.002 if row['metric']=='linear_l1_unclamped' else 0.2)

    def test_missing_duplicate_and_out_of_order_refused(self):
        m, ev, cells = fixture()
        for bad in (cells[:-1], cells+[cells[0]], cells[222:]+cells[:222]):
            with self.assertRaises(ValueError):V.scalar_accounting(m, bad, ev)

    def test_nonfinite_metric_refused(self):
        m, ev, cells = fixture();cells[400]['metrics']['mean_dE00'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):V.scalar_accounting(m,cells,ev)

    def test_actual_parity_failure_refused(self):
        m, ev, cells = fixture();cells[0]['metrics']['p95_dE00'] += 0.1
        with self.assertRaisesRegex(ValueError, 'reproduction'):V.scalar_accounting(m,cells,ev)

    def test_brightness_algebra_tamper_refused(self):
        m, ev, cells = fixture();cells[-1]['brightness']['B_shift'] += 0.1
        with self.assertRaisesRegex(ValueError, 'response'):V.scalar_accounting(m,cells,ev)

    def test_undefined_brightness_remains_null(self):
        m, ev, cells = fixture()
        for row in cells:
            row['brightness'].update(meanYpred=0.0, meanYtarget=0.3, B_err=-0.3, B_stops=None, B_shift=None,
                                     null_reasons={'B_stops':'nonpositive', 'B_shift':'nonpositive'})
        r = V.scalar_accounting(m,cells,ev)
        for group in r['independent_groups']:
            self.assertEqual(group['brightness']['B_stops'], {'n_defined':0,'partial_coverage':True,'mean':None})

    def test_numpy_and_torch_brightness_agree(self):
        x = torch.tensor([[[[0.0,0.2]],[[0.1,0.4]],[[0.2,0.7]]]], dtype=torch.float32)
        y = x+0.01
        a = P.brightness(x,y,0.3);b = V.brightness(x,y,0.3)
        for k in V.BRIGHTNESS_TOLERANCES:self.assertAlmostEqual(a[k],b[k],places=12)
        z = P.brightness(torch.zeros_like(x), y, 0.0)
        self.assertIsNone(z['B_stops']);self.assertIsNone(z['B_shift'])
        self.assertTrue(z['null_reasons']['B_stops'])

    def test_metadata_canaries30_cover_all_codes(self):
        m, _, _ = fixture();r = V.canary_cases(m)
        self.assertEqual(len(r),30)
        self.assertEqual({x['condition'] for x in r}, {0,0.5,1})
        self.assertEqual(sum(x['is_actual'] for x in r),12)

    def test_output_write_once(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'receipt.json';P.D.write_once_json(path, {'status':'first'})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):P.D.write_once_json(path, {'status':'changed'})
            self.assertEqual(path.read_bytes(),before)


if __name__ == '__main__':unittest.main()
