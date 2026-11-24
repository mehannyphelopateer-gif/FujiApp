"""Synthetic metadata fixtures only; no files, source photos or target images."""
import unittest
from unittest.mock import patch
from fractions import Fraction

import audit_wb_alignment as A


def fixture():
    rows=[];refs=[]
    for i in range(6):
        name=f'fit{i}.RAF';session=i//2
        rows.append({'name':name,'role':'fit','session':session,'source_wb_mode':4080,
                     'source_kelvin':6600,'source_wb_fine_tune_raw':[20,-60],
                     'cam_mul':[636,302,414,0],'auto_rgbg':[500,302,600,302]})
        refs.append({'name':name,'role':'fit','session':session,'mode':4080,'kelvin':6600,
                     'source_shift':[20,-60],'target_shift':[0,0],
                     'source_rgb':[636,302,414],'target_rgb':[2480,1208,1820]})
    rows.append({'name':'dev.RAF','role':'development','session':9,'source_wb_mode':4080,
                 'source_kelvin':6600,'source_wb_fine_tune_raw':[20,-60],
                 'cam_mul':[636,302,414,0],'auto_rgbg':[500,302,600,302]})
    refs.append({'name':'dev.RAF','role':'development','session':9,'mode':4080,'kelvin':6600,
                 'source_shift':[20,-60],'target_shift':[0,0],
                 'source_rgb':[636,302,414],'target_rgb':[2480,1208,1820]})
    return rows,refs


class AlignmentRunnerTests(unittest.TestCase):
    def test_source_record_has_no_target_fields(self):
        rows,_=fixture();r=dict(rows[0],target_rgb=[1,2,3])
        self.assertEqual(set(A.source_record(r)),A.W.SOURCE_KEYS)
        self.assertNotIn('target_rgb',A.source_record(r))

    def test_metadata_calibration_and_loso_use_only_fit(self):
        rows,refs=fixture();r=A.audit(rows,refs)
        self.assertEqual(r['profiles']['profiles']['4080/6600']['fit_scenes'],6)
        self.assertTrue(all(x['exact_target_metadata_match'] is True for x in r['kelvin6600_fit_LOSO']))
        self.assertTrue(all(x['fit_reference_count']==4 for x in r['kelvin6600_fit_LOSO']))
        self.assertTrue(all(x['exact_target_metadata_match'] is True for x in r['archived_reference_checks']))
        self.assertEqual(sum(x['calibration_member'] for x in r['archived_reference_checks']),6)
        self.assertFalse(r['archived_reference_checks'][-1]['calibration_member'])
        self.assertEqual(r['policy_reference_coverage']['zero-shift-as-shot'],0)
        # A contradictory development target changes its comparison, never the profile.
        refs[-1]['target_rgb']=[1800,1208,1200]
        after=A.audit(rows,refs)
        self.assertEqual(after['profiles'],r['profiles'])
        self.assertFalse(after['archived_reference_checks'][-1]['exact_target_metadata_match'])

    def test_short_loso_is_null_not_pass(self):
        rows,refs=fixture();rows,refs=rows[:3],refs[:3]
        report=A.audit(rows,refs)
        self.assertTrue(all(x['exact_target_metadata_match'] is None for x in report['kelvin6600_fit_LOSO']))

    def test_unsupported_reference_is_null_not_zero_error(self):
        rows,refs=fixture()
        rows[0].update(source_wb_mode=770,source_kelvin=None)
        refs[0].update(mode=770,kelvin=None)
        r=A.audit(rows,refs)
        self.assertIsNone(r['archived_reference_checks'][0]['exact_target_metadata_match'])
        self.assertEqual(r['policies'][rows[0]['name']]['basis'],'unsupported')

    def test_non302_green_is_observed_and_own_green_normalized(self):
        rows,refs=fixture()
        rows.append({'name':'warm.RAF','role':'development','session':10,'source_wb_mode':4080,
                     'source_kelvin':2500,'source_wb_fine_tune_raw':[0,0],
                     'cam_mul':[302,325,1137,0],'auto_rgbg':[289,302,1044,302]})
        r=A.audit(rows,refs);obs=r['observations']['cam_green_not_302']
        self.assertEqual(len(obs),1)
        self.assertEqual(obs[0]['anchors']['cam_mul']['min_channels'],['R'])
        self.assertEqual(obs[0]['candidate_policy']['basis'],'zero-shift-as-shot')
        self.assertEqual(A.exact_ratios(obs[0]['candidate_policy']['ratios'])['R/G'],Fraction(302,325))
        self.assertEqual(r['observations']['auto_minimum_not_302_names'],['warm.RAF'])

    def test_mixed_mode_sessions_are_reported_not_called_confounded(self):
        rows,refs=fixture();rows.append(dict(rows[0],name='auto.RAF',source_wb_mode=0,source_kelvin=None))
        r=A.audit(rows,refs)
        self.assertEqual(r['observations']['multiple_WB_mode_sessions'],[{'role':'fit','session':0,'modes':[0,4080]}])

    def test_refuse_unknown_mode_and_unpinned_manifest(self):
        rows,refs=fixture();rows[0]['source_wb_mode']=999
        with self.assertRaises(ValueError):A.audit(rows,refs)
        with self.assertRaises(ValueError):A.load(None)
        with self.assertRaises(ValueError):A.load('0'*64)


if __name__=='__main__':unittest.main()
