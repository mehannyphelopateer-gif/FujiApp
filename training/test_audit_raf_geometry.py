import unittest
import audit_raf_geometry as A

class GeometryScopeTests(unittest.TestCase):
    def test_source_refusals(self):
        row={'role':'fit','shoot':'Shoot 123','name':'DSCF1234.RAF'}
        self.assertTrue(str(A.source_path(row)).endswith('/Shoot 123/DSCF1234.RAF'))
        for key,value in [('role','development'),('role','monitor'),('role','finalHeldOut'),('shoot','Shoot ../3'),('name','DSCF../3.RAF')]:
            with self.assertRaises(ValueError):A.source_path(dict(row,**{key:value}))
    def test_unfrozen_scope_refused(self):
        with self.assertRaises(ValueError):A.load(None)
        with self.assertRaises(ValueError):A.load('0'*64)
    def test_cross_tabs_preserve_classes_no_transform(self):
        row={'G_native_size':[5178,7752],'session':3,'captureDynamicRange':400,'source_wb_mode':4080,'geometry':{'compression':2}}
        other={**row,'G_native_size':[5149,7728]}
        groups=A.summarize([row,row,other])
        self.assertEqual(len(groups),2);self.assertEqual(sorted(g['n'] for g in groups),[1,2])
    def test_geometry_missing_tags_remain_separate_without_merging(self):
        row={'G_native_size':[5178,7752],'target_orientation':8,'geometry':{'compression':2,'directories':[{'index':0,'tags':{'raw_image_cropped_size':[5152,7728]}}]}}
        missing={**row,'geometry':{'compression':None,'directories':[{'index':0,'tags':{'raw_image_cropped_size':None}}]}}
        groups=A.geometry_summary([row,row,missing])
        self.assertEqual(len(groups),2);self.assertEqual(sorted(g['n'] for g in groups),[1,2])
        self.assertTrue(any(g['compression_code'] is None and g['RAF_directories'][0]['raw_tags']['raw_image_cropped_size'] is None for g in groups))

if __name__=='__main__':unittest.main()
