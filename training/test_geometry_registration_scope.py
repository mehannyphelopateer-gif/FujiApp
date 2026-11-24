import unittest
import numpy as np
import diagnose_geometry_registration as R

class RegistrationScopeTests(unittest.TestCase):
    def test_synthetic_hypotheses_distinguish_zoom_and_rotation_without_mutation(self):
        import geometry_registration_metrics as G
        from test_geometry_registration_metrics import scene
        source=scene(120,180);saved=source.copy()
        tags={'raw_zoom_active':1,'raw_image_cropped_size':[1000,1500],'raw_zoom_top_left':[146,219],'raw_zoom_size':[708,1062]}
        win=G.normalized_zoom_window(tags['raw_image_cropped_size'],tags['raw_zoom_top_left'],tags['raw_zoom_size'],1)
        target=G.zoom_resample(source,win);before=target.copy()
        scores,_,_=R.measure_candidates(source,target,tags,1)
        self.assertGreater(scores['source_metadata_zoom']['feature_ncc'],scores['identity']['feature_ncc']+0.2)
        self.assertTrue(np.array_equal(source,saved) and np.array_equal(target,before))
        scores,win,_=R.measure_candidates(source,np.ascontiguousarray(source[::-1,::-1]),{'raw_zoom_active':0},6)
        self.assertIsNone(win);self.assertEqual(set(scores),{'identity','target_180','source_metadata_zoom','target_metadata_zoom'})
        self.assertFalse(scores['source_metadata_zoom']['applicable']);self.assertIsNone(scores['source_metadata_zoom']['feature_ncc'])
        self.assertGreater(scores['target_180']['feature_ncc'],0.99)
    def pool(self):
        def row(name,ori,active,size):
            return {'name':name,'role':'fit','target_orientation':ori,'geometry':{'directories':[{'index':0,'tags':{'raw_zoom_active':active,'raw_zoom_size':size}}]}}
        return [row('DSCF0464.RAF',8,0,[5152,7728]),row('DSCF1068.RAF',6,0,[5152,7728]),row('DSCF1054.RAF',1,0,[5152,7728]),
                row('DSCF1011.RAF',8,1,[3648,5472]),row('DSCF2000.RAF',1,0,[5152,7728]),row('DSCF2001.RAF',8,1,[2592,3888]),
                row('DSCF2002.RAF',1,1,[3648,5472]),row('DSCF2003.RAF',1,1,[2592,3888]),row('DSCF2004.RAF',1,0,[5152,7728])]
    def test_eight_source_only_anchors_and_classes(self):
        p=self.pool();a=R.select(p);b=R.select(list(reversed(p)))
        self.assertEqual([r['name'] for r in a],[r['name'] for r in b]);self.assertEqual(len(a),8)
        self.assertEqual([r['name'] for r in a[:4]],['DSCF0464.RAF','DSCF1068.RAF','DSCF1054.RAF','DSCF1011.RAF'])
    def test_short_not_replaced_or_relaxed(self):
        p=[r for r in self.pool() if r['name']!='DSCF2003.RAF']
        with self.assertRaises(ValueError):R.select(p)
        p=self.pool();p[0]['role']='development'
        with self.assertRaises(ValueError):R.select(p)
    def test_unknown_primary_and_unfrozen_refused(self):
        with self.assertRaises(ValueError):R.primary_tags({'geometry':{'directories':[]}})
        with self.assertRaises(ValueError):R.load(None)
        with self.assertRaises(ValueError):R.load('0'*64)

if __name__=='__main__':unittest.main()
