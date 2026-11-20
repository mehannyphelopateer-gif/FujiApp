import unittest
import numpy as np
from verify_provia_v2_fullnative import histogram_stats, sparse_histogram, transfer_errors, is_subset_histogram, validate_histogram

class HistogramTests(unittest.TestCase):
    def test_quantiles(self):
        for n in (1,2,100,1001,100000):
            e=np.random.default_rng(42).integers(0,65536,n)
            h=sparse_histogram(e);s=histogram_stats(h)
            self.assertAlmostEqual(s['p99_abs_codes'],np.percentile(e,99),places=8)
            self.assertEqual(s['mean_abs_codes'],e.mean())
            self.assertEqual(s['max_abs_codes'],e.max())
    def test_empty(self):
        self.assertIsNone(histogram_stats([]))
    def test_invalid(self):
        for h in ([[0,1],[0,2]],[[2,1],[1,2]],[[0,0]],[[65536,1]],[[-1,1]],[[True,1]],[[0,1.0]]):
            with self.assertRaises(ValueError):validate_histogram(h)
    def test_mask_boundaries(self):
        a=np.array([655,656,62258,62259],dtype=np.uint16)
        e,m=transfer_errors(a,a,np.arange(65536,dtype=np.uint16))
        np.testing.assert_array_equal(m,[False,True,True,False]);self.assertEqual(e.size,2);self.assertEqual(e.sum(),0)
    def test_subset(self):
        self.assertTrue(is_subset_histogram([[0,2],[2,1]],[[0,5],[2,1]]))
        self.assertFalse(is_subset_histogram([[0,6]],[[0,5]]))
        self.assertFalse(is_subset_histogram([[1,1]],[[0,5]]))
if __name__=='__main__':unittest.main()
