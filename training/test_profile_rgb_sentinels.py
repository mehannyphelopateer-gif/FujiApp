import copy
import unittest
import qualify_profile_rgb_sentinels as Q

class SelectionTests(unittest.TestCase):
    def pool(self):
        rows=[]
        for name,session in zip(Q.FIRST_THREE,(3,8,11)):
            rows.append({'name':name,'role':'fit','session':session,'frozen_source_only_policy':{'basis':'calibrated-kelvin-profile-candidate'}})
        rows.extend({'name':f'DSCF{n}.RAF','role':'fit','session':13,'frozen_source_only_policy':{'basis':'calibrated-kelvin-profile-candidate'}} for n in (1200,1201))
        return rows
    def test_selection_reordering_invariant_no_pixels_or_errors(self):
        rows=self.pool();wanted=[r['name'] for r in Q.choose(rows)]
        self.assertEqual(wanted[:3],list(Q.FIRST_THREE))
        self.assertEqual([r['name'] for r in Q.choose(list(reversed(rows)))],wanted)
        rows.append({'name':'DSCF0000.RAF','role':'development','session':13,'frozen_source_only_policy':{'basis':'calibrated-kelvin-profile-candidate'}})
        self.assertEqual([r['name'] for r in Q.choose(rows)],wanted)
    def test_missing_session_or_primitive_refused(self):
        for rows in (self.pool()[:3],self.pool()[1:]):
            with self.assertRaises(ValueError):Q.choose(rows)
    def test_control_truthful_and_new_options_fresh(self):
        s={'cam_mul':[636,302,414,0]}
        self.assertEqual(Q.options(s,'C')['userMul'],[636,302,414,302])
        self.assertEqual(Q.options(s,'P')['userMul'],[620,302,455,302])
        self.assertEqual(Q.options(s,'P'),Q.options(s,'R'))
        o=Q.options(s,'P');o['userMul'][0]=0
        self.assertEqual(Q.options(s,'P')['userMul'][0],620)
        with self.assertRaises(ValueError):Q.options(s,'unknown')
    def test_unpinned_scope_refused(self):
        with self.assertRaises(ValueError):Q.load(None)
        with self.assertRaises(ValueError):Q.load('0'*64)

if __name__=='__main__':unittest.main()
