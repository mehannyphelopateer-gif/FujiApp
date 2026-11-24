"""Synthetic-only diagnostic metric checks; no media or model reads."""
import unittest
import numpy as np
import diagnose_common_geometry as A


class DiagnosticTests(unittest.TestCase):
    def test_absolute_not_signed_summary(self):
        rows=[{'usable':True,'dx':-2.,'dy':1.},{'usable':True,'dx':2.,'dy':-1.},
              {'usable':False,'dx':99.,'dy':99.}]
        s=A.summarize(rows)
        self.assertEqual(s['usable_tiles'],2)
        self.assertEqual(s['median_abs_dx'],2.)
        self.assertEqual(s['median_abs_dy'],1.)
        self.assertAlmostEqual(s['p95_shift_norm'],np.sqrt(5))

    def test_empty_is_null(self):
        s=A.summarize([])
        self.assertEqual(s,{'usable_tiles':0,'median_abs_dx':None,'median_abs_dy':None,'p95_shift_norm':None})

    def test_identity_and_no_mutation(self):
        a=np.random.default_rng(42).random((144,144,3),dtype=np.float32)
        before=a.copy();m=A.measure(a,a)
        self.assertAlmostEqual(m['feature_ncc'],1)
        self.assertEqual(m['summary']['usable_tiles'],9)
        self.assertLess(m['summary']['p95_shift_norm'],1e-6)
        self.assertTrue(np.array_equal(a,before))

    def test_odd_optimal_dft_shape_refused(self):
        a=np.random.default_rng(42).random((120,150,3),dtype=np.float32)
        with self.assertRaisesRegex(AssertionError,'Odd optimal'):
            A.measure(a,a)


if __name__=='__main__':unittest.main()
