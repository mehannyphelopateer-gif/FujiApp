"""Synthetic all-site coverage fixtures; no RAF, target or model data."""
import unittest

import numpy as np

import qualify_manual_kelvin_wb_fullnative as F


def fixture(h=96,w=192):
    pattern=(np.arange(6)[:,None]+np.arange(6)[None,:])%3
    channels=pattern[np.arange(h)[:,None]%6,np.arange(w)[None,:]%6]
    d=np.zeros((h,w,3),dtype='<u2');e=np.zeros_like(d)
    gains=[620/636,1.,455/414]
    for c,g in enumerate(gains):
        active=channels==c
        vals=np.arange(np.count_nonzero(active))%20000+1000
        d[:,:,c][active]=vals
        e[:,:,c][active]=np.floor(g*vals)
    return d,e,pattern,gains,channels


class FullNativeTests(unittest.TestCase):
    def test_all_sites_match_existing_gate_and_chunk_boundaries(self):
        d,e,p,g,ch=fixture();result=F.full_statistics(d,e,p,g,block_rows=7)
        second=F.full_statistics(d,e,p,g,block_rows=128)
        self.assertEqual(sum(v['all_active_sites'] for v in result),d.shape[0]*d.shape[1])
        for c in range(3):
            old=F.S.primitive_channel(d[:,:,c][ch==c],e[:,:,c][ch==c],g[c])
            self.assertEqual(result[c]['n'],old['n'])
            self.assertEqual(result[c]['max_abs_codes'],old['max_abs_codes'])
            self.assertAlmostEqual(result[c]['mean_abs_codes'],old['mean_abs_codes'],places=12)
            self.assertAlmostEqual(result[c]['mean_abs_codes'],second[c]['mean_abs_codes'],places=12)
            self.assertTrue(result[c]['passed'])
        self.assertEqual(result[1]['exact_fraction'],1.)

    def test_single_bad_eligible_site_cannot_hide_in_mean(self):
        d,e,p,g,ch=fixture();ys,xs=np.nonzero(ch==2)
        e[ys[0],xs[0],2]+=10
        result=F.full_statistics(d,e,p,g)
        self.assertLess(result[2]['mean_abs_codes'],1.)
        self.assertGreater(result[2]['max_abs_codes'],3.)
        self.assertFalse(result[2]['passed'])

    def test_inactive_channel_corruption_refused(self):
        d,e,p,g,ch=fixture();e[0,0,1]=1
        with self.assertRaises(ValueError):F.full_statistics(d,e,p,g)

    def test_empty_and_short_remain_unqualified(self):
        d,e,p,g,_=fixture(6,6)
        result=F.full_statistics(d,e,p,g)
        self.assertTrue(all(not v['measurable'] and not v['passed'] for v in result))
        result=F.full_statistics(np.zeros_like(d),np.zeros_like(e),p,g)
        self.assertTrue(all(v['n']==0 and v['mean_abs_codes'] is None and not v['passed'] for v in result))

    def test_both_arm_mask_boundaries_match_old_gate(self):
        d,e,p,g,ch=fixture()
        for c,scale in enumerate(g):
            active=ch==c;values=np.resize(np.array([0,655,656,1000,40000,52428,52429,60000],dtype='<u2'),np.count_nonzero(active))
            d[:,:,c][active]=values;e[:,:,c][active]=np.minimum(65535,np.floor(scale*values))
        result=F.full_statistics(d,e,p,g)
        for c in range(3):
            expected=F.S.primitive_channel(d[:,:,c][ch==c],e[:,:,c][ch==c],g[c])
            self.assertEqual(result[c]['n'],expected['n'])
            self.assertEqual(result[c]['max_abs_codes'],expected['max_abs_codes'])

    def test_dtype_shape_pattern_and_manifest_refused(self):
        d,e,p,g,_=fixture()
        for args in ((d.astype(np.float32),e,p,g),(d,e[:-1],p,g),(d,e,np.zeros((6,6)),g)):
            with self.assertRaises(ValueError):F.full_statistics(*args)
        with self.assertRaises(ValueError):F.full_statistics(d,e,p,g,block_rows=True)
        with self.assertRaises(ValueError):F.load(None)
        with self.assertRaises(ValueError):F.load('0'*64)


if __name__=='__main__':unittest.main()
