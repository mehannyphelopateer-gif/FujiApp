"""Independent check arithmetic and metadata selection; no media or real checkpoints."""
import copy
import math
import random
import unittest
import verify_neutral_wb_study as V
from test_run_neutral_wb_study import synthetic_manifest
from test_eval_neutral_wb_study import evals_for,by_subgroup


class Verification(unittest.TestCase):
    def test_selection_is_order_independent_and_covers_roles_kinds_and_dr100(self):
        m=synthetic_manifest(with_orders=False);rows=m['scenes'];before=copy.deepcopy(rows)
        names=V.canary_names(rows);shuffled=rows.copy();random.Random(31).shuffle(shuffled)
        self.assertEqual(names,V.canary_names(shuffled));self.assertEqual(rows,before)
        self.assertTrue(6<=len(names)<=7);by={r['name']:r for r in rows}
        self.assertEqual({(by[n]['role'],by[n]['neutral_kind']) for n in names},
                         {(r,k) for r in ('fit','development') for k in ('G','W','P')})
        self.assertTrue(any(by[n]['captureDynamicRange']==100 for n in names))

    def test_short_canary_coverage_refused(self):
        m=synthetic_manifest(with_orders=False)
        with self.assertRaises(ValueError):V.canary_names([r for r in m['scenes'] if not(r['role']=='fit' and r['neutral_kind']=='P')])

    def test_independent_differences_and_single_seed_retention(self):
        m=synthetic_manifest(with_orders=False)
        delta=by_subgroup(-.5,.0625,seed_override={43:by_subgroup(-.5,.125)})
        result=V.independent_values(m,evals_for(m,delta))
        self.assertEqual(result['42']['treated'],-.5);self.assertEqual(result['42']['identity'],.0625)
        self.assertEqual(result['43']['identity'],.125)
        self.assertAlmostEqual(result['43']['dev'],(-.5*79+.125*37)/116)

    def test_nonfinite_bool_and_strings_are_not_numbers(self):
        for v in (True,False,float('nan'),float('inf'),'-0.5',None):self.assertFalse(V.finite(v))
        self.assertTrue(V.finite(-.5))
        with self.assertRaises(ValueError):V.mean([.5,float('nan')])
        with self.assertRaises(ValueError):V.mean([])
        self.assertEqual(V.mean([1e20,1.,-1e20]),1/3)


if __name__=='__main__':unittest.main()
