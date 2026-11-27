"""Synthetic policy/refusal fixtures only; no source bytes or target/reference values."""
import copy
import unittest
import check_dr100_candidate_wb as B


def fixture():
    mapping={'whiteBalanceMode':'WhiteBalance','autoDynamicRange':'AutoDynamicRange','exposureCount':'ExposureCount',
             'developmentDynamicRange':'DevelopmentDynamicRange','dynamicRange':'DynamicRangeSetting',
             'dynamicRangeSetting':'DynamicRangeSetting2','dRangePriority':'DRangePriority',
             'dRangePriorityAuto':'DRangePriorityAuto','dRangePriorityFixed':'DRangePriorityFixed'}
    fuji={k:None for k in mapping};fuji.update(whiteBalanceMode=1,autoDynamicRange=100,exposureCount=1,dynamicRange=1,dynamicRangeSetting=0)
    fuji['formats']={k:None if v is None else {} for k,v in fuji.items()}
    wb={'whiteBalanceModeRaw':1,'whiteBalanceFineTuneRaw':{'red':0,'blue':0},'colorTemperatureKelvinRaw':5600,
        'asShotCamMul':[350,302,450,0],'autoRgbg':[360,302,440,302]}
    fuji.update(colorTemperatureKelvin=5600,whiteBalanceFineTune={'red':0,'blue':0})
    ex={'model':'X100VI','dateTimeOriginal':'2026:11:16 20:00:00','iso':125}
    legacy={**ex,'fuji':{mapping[k]:f'{v} (0x{v:x})' for k,v in fuji.items() if k in mapping and v is not None}}
    return {'schema':'raf-training-candidate-metadata/v1','exif':ex,'fuji':fuji,'wb':wb},legacy,{'profiles':{}}


class Classification(unittest.TestCase):
    def test_known_zero_shift_is_identity_without_new_profile(self):
        m,l,p=fixture();r=B.classify(m,l,p)
        self.assertEqual(r['neutral_kind_candidate'],'G');self.assertTrue(r['policy_supported'])
        self.assertIsNone(r['source_policy_fields']['source_kelvin'])
        self.assertEqual(r['kelvin_tag_raw'],5600)

    def test_missing_shift_and_active_kelvin_are_held(self):
        m,l,p=fixture();m['wb']['whiteBalanceFineTuneRaw']=None
        m['fuji']['whiteBalanceFineTune']=None
        with self.assertRaisesRegex(ValueError,'fine tune'):B.classify(m,l,p)
        m,l,p=fixture();m['wb'].update(whiteBalanceModeRaw=4080,colorTemperatureKelvinRaw=None)
        m['fuji'].update(whiteBalanceMode=4080,colorTemperatureKelvin=None)
        l['fuji']['WhiteBalance']='4080 (0xff0)'
        with self.assertRaisesRegex(ValueError,'Kelvin'):B.classify(m,l,p)

    def test_nonzero_unsupported_family_has_no_fallback(self):
        m,l,p=fixture();m['wb']['whiteBalanceFineTuneRaw']={'red':20,'blue':0}
        m['fuji']['whiteBalanceFineTune']={'red':20,'blue':0}
        r=B.classify(m,l,p);self.assertFalse(r['policy_supported']);self.assertIsNone(r['neutral_kind_candidate'])

    def test_metadata_disagreement_and_capture_DR_refused(self):
        m,l,p=fixture();m['fuji']['autoDynamicRange']=200
        with self.assertRaisesRegex(ValueError,'disagreement'):B.classify(m,l,p)
        m,l,p=fixture();m['exif']['iso']=1600
        with self.assertRaisesRegex(ValueError,'metadata'):B.classify(m,l,p)

    def test_wb_alias_mismatch_refused(self):
        m,l,p=fixture();m['wb']['whiteBalanceModeRaw']=4080
        with self.assertRaisesRegex(ValueError,'aliases'):B.classify(m,l,p)

    def test_P_requires_exact_primitive_qualified_source_shift(self):
        m,l,p=fixture()
        m['fuji'].update(whiteBalanceMode=4080,colorTemperatureKelvin=6600,whiteBalanceFineTune={'red':0,'blue':40})
        m['wb'].update(whiteBalanceModeRaw=4080,colorTemperatureKelvinRaw=6600,whiteBalanceFineTuneRaw={'red':0,'blue':40})
        l['fuji']['WhiteBalance']='4080 (0xff0)'
        p={'profiles':{'4080/6600':{'status':'candidate','target_ratios':{'R/G':{'num':620,'den':302},'B/G':{'num':455,'den':302}}}}}
        r=B.classify(m,l,p)
        self.assertFalse(r['policy_supported']);self.assertIsNone(r['neutral_kind_candidate'])
        self.assertIn('[20,-60]',r['policy_gate_hold_reason'])
        m['fuji']['whiteBalanceFineTune']=m['wb']['whiteBalanceFineTuneRaw']={'red':20,'blue':-60}
        r=B.classify(m,l,p)
        self.assertTrue(r['policy_supported']);self.assertEqual(r['neutral_kind_candidate'],'P')
        self.assertEqual(r['declared_levels_rgbg_candidate'],list(B.N.P.PROFILE_LEVELS_RGBG))

    def test_W_permits_qualified_nonzero_shift(self):
        m,l,p=fixture();m['fuji'].update(whiteBalanceMode=0,whiteBalanceFineTune={'red':20,'blue':-60})
        m['wb'].update(whiteBalanceModeRaw=0,whiteBalanceFineTuneRaw={'red':20,'blue':-60})
        l['fuji']['WhiteBalance']='0 (0x0)'
        r=B.classify(m,l,p)
        self.assertTrue(r['policy_supported']);self.assertEqual(r['neutral_kind_candidate'],'W')
        self.assertEqual(r['declared_levels_rgbg_candidate'],m['wb']['autoRgbg'])


if __name__=='__main__':unittest.main()
