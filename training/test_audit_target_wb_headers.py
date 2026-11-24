import unittest
from fractions import Fraction
import audit_target_wb_headers as A

class HeaderAuditTests(unittest.TestCase):
    def row(self):return {'name':'DSCF0001.RAF','role':'fit','shoot':'Shoot 1','source_wb_mode':4080,'source_kelvin':6600,'cam_mul':[636,302,414,0]}
    def tags(self):return {'WhiteBalance':4080,'ColorTemperature':6600,'WBFineTune':[0,0],'WBLevelR':620,'WBLevelG':302,'WBLevelB':455}
    def policy(self):return {'ratios':{'R/G':{'num':310,'den':151},'B/G':{'num':455,'den':302}}}
    def test_source_only_profile_exact_own_green(self):
        r=A.comparison(self.row(),self.policy(),self.tags())
        self.assertTrue(r['neutral_settings_match']);self.assertTrue(r['exact_policy_ratio_match']);self.assertFalse(r['source_as_shot_ratio_match'])
        t=self.tags();t.update(WBLevelR=1240,WBLevelG=604,WBLevelB=910)
        self.assertTrue(A.comparison(self.row(),self.policy(),t)['exact_policy_ratio_match'])
        t['WBLevelB']+=1;self.assertFalse(A.comparison(self.row(),self.policy(),t)['exact_policy_ratio_match'])
    def test_unsupported_stays_null(self):
        self.assertIsNone(A.comparison(self.row(),{'ratios':None},self.tags())['exact_policy_ratio_match'])
    def test_mismatches_and_bad_gain_not_hidden(self):
        for key,value in [('WhiteBalance',0),('ColorTemperature',5600),('WBFineTune',[20,-60])]:
            t=self.tags();t[key]=value;self.assertFalse(A.comparison(self.row(),self.policy(),t)['neutral_settings_match'])
        t=self.tags();t['WBLevelG']=0
        with self.assertRaises(ValueError):A.comparison(self.row(),self.policy(),t)
    def test_role_and_path_refusals(self):
        for role in ('monitor','K','finalHeldOut','T8','dr100Eval'):
            r=self.row();r['role']=role
            with self.assertRaises(ValueError):A.target_path(r)
        for shoot in ('Shoot ../7','../Shoot 7','Shoot 1/x'):
            r=self.row();r['shoot']=shoot
            with self.assertRaises(ValueError):A.target_path(r)
    def test_counts_include_invalid_and_unsupported(self):
        common={'role':'fit','source_wb_mode':0,'source_kelvin':None,'policy_basis':'test'}
        rows=[dict(common,status='checked',exact_policy_ratio_match=True),dict(common,status='checked',exact_policy_ratio_match=False),dict(common,status='checked',exact_policy_ratio_match=None),dict(common,status='invalid')]
        g=A.summarize(rows)[0]
        self.assertEqual((g['n'],g['exact'],g['mismatch'],g['unsupported'],g['invalid']),(4,1,1,1,1))
    def test_manifest_pin_refusal(self):
        with self.assertRaises(ValueError):A.load(None)
        with self.assertRaises(ValueError):A.load('0'*64)

if __name__=='__main__':unittest.main()
