"""Synthetic native-array/metadata fixtures only, never a real RAF or target."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import qualify_manual_kelvin_wb as Q


class PrimitiveTests(unittest.TestCase):
    def test_common_integer_scale_is_derived_not_assumed(self):
        self.assertEqual(Q.integer_levels({'R/G':Fraction(620,302),'B/G':Fraction(455,302)}),[620,302,455,302])
        self.assertEqual(Q.integer_levels({'R/G':Fraction(3,2),'B/G':Fraction(7,5)}),[15,10,14,10])
        with self.assertRaises(ValueError):Q.integer_levels({'R/G':Fraction(-1,2),'B/G':Fraction(1)})

    def test_metadata_selection_excludes_dev_auto_and_calibration(self):
        rows=[]
        for s in (3,8,11):
            for i in range(3):rows.append({'name':f'{s}-{i}.RAF','role':'fit','session':s,'source_wb_mode':4080,
                                         'source_kelvin':6600,'source_wb_fine_tune_raw':[20,-60]})
        original=Q.select(rows,set());excluded={original[0]['name']}
        rows += [dict(rows[0],name='dev.RAF',role='development'),dict(rows[0],name='auto.RAF',source_wb_mode=0)]
        selected=Q.select(rows,excluded)
        self.assertEqual(len(selected),3)
        self.assertEqual({r['session'] for r in selected},{3,8,11})
        self.assertTrue(all(r['name'] not in excluded|{'dev.RAF','auto.RAF'} for r in selected))
        self.assertEqual(selected,Q.select(list(reversed(rows)),excluded))
        with self.assertRaises(ValueError):Q.select(rows[:3],set())

    def test_camera_and_explicit_arms_change_only_declared_wb(self):
        b={'gamm':[1,1,0,0,0,0],'useCameraWb':False,'useAutoWb':False}
        s={'as_shot_rgbg':[636,302,414,302],'neutral_rgbg':[620,302,455,302]}
        self.assertNotIn('userMul',Q.expected_options(b,s,'C'))
        self.assertTrue(Q.expected_options(b,s,'C')['useCameraWb'])
        self.assertFalse(Q.expected_options(b,s,'D')['useCameraWb'])
        self.assertEqual(Q.expected_options(b,s,'E'),Q.expected_options(b,s,'E-repeat'))
        for a in ('D','E','E-repeat'):
            self.assertEqual({k:v for k,v in Q.expected_options(b,s,a).items() if k!='userMul'},b)
        with self.assertRaises(ValueError):Q.expected_options(b,s,'unknown')

    def test_physical_quantisation_and_short_gate_unchanged(self):
        gains=Q.S.expected_gains([636,302,414,302],[620,302,455,302])
        self.assertEqual(gains[1],1.)
        d=np.tile(np.array([1000,5000,10000,20000],dtype=float),500)
        for g in gains:
            valid=Q.S.primitive_channel(d,np.floor(g*d),g)
            self.assertTrue(valid['passed'])
            self.assertFalse(Q.S.primitive_channel(d,np.floor(g*d)+5,g)['passed'])
        self.assertFalse(Q.S.primitive_channel(d[:10],d[:10],1.)['passed'])

    def test_native_sample_hash_geometry_phase_coverage_and_source_guards(self):
        scene={'name':'test.RAF','cam_mul':[636,302,414,0],'as_shot_rgbg':[636,302,414,302],
               'neutral_rgbg':[620,302,455,302]}
        m={'base_options':{'useCameraWb':False,'useAutoWb':False}}
        width=height=62;nr,nc=Q.S.grid_shape(width,height)
        counts=np.zeros(144,dtype=int)
        for y in range(height):
            for x in range(width):counts[((y%6)*6+x%6)*4+((y+x)%3)]+=1
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);p=out/'test.RAF-D.samples.u16le'
            np.zeros(nr*nc*3,dtype='<u2').tofile(p)
            opts=Q.expected_options(m['base_options'],scene,'D')
            arm={'declared_options':opts,'declared_options_sha256':hashlib.sha256(json.dumps(opts,separators=(',',':')).encode()).hexdigest(),
                 'returned_metadata':{'cam_mul':scene['cam_mul'],'camera_model':'X100VI','fujiWB':4080},
                 'shape':{'width':width,'height':height,'bits':16,'colors':3},'phase_counts':counts.tolist(),
                 'native_sha256':'a'*64,'samples_file':str(p),'samples_sha256':Q.D.sha(p)}
            with patch.object(Q,'OUT',out):
                self.assertEqual(Q.arm_samples(scene,'D',arm,m).size,nr*nc*3)
                arm['samples_sha256']='0'*64
                with self.assertRaises(ValueError):Q.arm_samples(scene,'D',arm,m)
                arm['samples_sha256']=Q.D.sha(p);arm['phase_counts'][0]+=1
                with self.assertRaises(ValueError):Q.arm_samples(scene,'D',arm,m)
                arm['phase_counts'][0]-=1;arm['native_sha256']='bad'
                with self.assertRaises(ValueError):Q.arm_samples(scene,'D',arm,m)

    def test_manifest_refused_before_any_native_read(self):
        with self.assertRaises(ValueError):Q.load(None)
        with self.assertRaises(ValueError):Q.load('0'*64)


if __name__=='__main__':unittest.main()
