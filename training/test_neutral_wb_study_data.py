"""Synthetic codecs/arrays and metadata only. Never reads project media or model weights."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import neutral_wb_study_data as D
import test_fjlrg_v2 as VF
import test_fjnrgb_profile as PF


def fixture(root,kind='P'):
    gbytes,gprov,gpx=VF.make_case(width=4,height=3)
    src=gprov['source_raf_sha256'];wasm=gprov['wasm_sha256']
    if kind=='P':
        npx,nbytes,nprov=PF.fixture(width=4,height=3,native=(4,3))
        nprov['source_raf_sha256']=src;nprov['wasm_sha256']=wasm
        gprov['source_wb_fine_tune_raw']=[20,-60]
        contract=D.P.CONTRACT;ext='fjnrgb'
    elif kind=='W':
        nbytes,nprov,npx=VF.make_case(D.V.CONTRACT_AUTOWB,width=4,height=3)
        gprov['source_wb_mode']=0;gprov['source_kelvin']=None
        contract=D.V.CONTRACT_AUTOWB;ext='fjlrg'
    else:nbytes,nprov,npx=gbytes,copy.deepcopy(gprov),gpx;contract=D.V.CONTRACT_ASSHOT;ext='fjlrg'
    gp='training/checkpoints/provia-input-v2-train-g/Shoot 1/input-v2-linear-asshot.fjlrg'
    npth=gp if kind=='G' else f'training/checkpoints/neutral-wb-fit-export/Shoot 1/N-neutral.{ext}'
    for p,b,prov in ((gp,gbytes,gprov),(npth,nbytes,nprov)):
        path=root/p;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b);Path(str(path)+'.json').write_text(json.dumps(prov))
    target='calibration-input/provia-color-target-cache/Shoot 1.npy'
    path=root/target;path.parent.mkdir(parents=True,exist_ok=True)
    y=np.full((3,4,3),.25,dtype=np.float32)
    y[0,0,0]=np.nextafter(np.nextafter(np.float32(1),np.float32(2)),np.float32(2))
    np.save(path,y)
    def spec(p,k,c):return {'path':p,'kind':k,'contract':c,'sha256':D.sha(root/p),
                          'sidecar_sha256':D.sha(Path(str(root/p)+'.json')),'source_raf_sha256':src,'wasm_sha256':wasm}
    row={'name':'DSCF0001.RAF','shoot':'Shoot 1','role':'fit','session':0,'captureDynamicRange':400,
         'fjlrg_wh':[4,3],'target_path':target,'npy_sha256':D.sha(path),'neutral_kind':kind,
         'inputs':{'G':spec(gp,'G',D.V.CONTRACT_ASSHOT),'neutral':spec(npth,kind,contract)}}
    m={'scenes':[row],'_manifest_sha256':'f'*64};m['_rows_sha256']=D.rows_digest(m['scenes'])
    return m,y,gpx,npx


class PairLoading(unittest.TestCase):
    def test_three_strict_contracts_and_unchanged_epsilon_target(self):
        for kind in ('P','W','G'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td,patch.object(D,'ROOT',Path(td)):
                m,y,gpx,npx=fixture(Path(td),kind)
                for arm,px in (('G',gpx),('neutral',npx)):
                    x,t,c=D.Pairs(m,arm).get('DSCF0001.RAF')
                    self.assertEqual(list(x.shape),[1,3,3,4]);self.assertEqual(c.item(),1)
                    self.assertTrue(torch.equal(t[0].permute(1,2,0),torch.from_numpy(y)))
                    self.assertGreater(t.max().item(),1)
                    self.assertTrue(np.array_equal(x[0].permute(1,2,0).numpy(),px.astype(np.float32)/np.float32(65535)))

    def test_default_fit_refuses_development_and_unknown_before_io(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'ROOT',Path(td)):
            m,_,_,_=fixture(Path(td));dev=copy.deepcopy(m['scenes'][0]);dev.update(name='DSCF0002.RAF',role='development',session=1)
            m['scenes'].append(dev);m['_rows_sha256']=D.rows_digest(m['scenes'])
            pairs=D.Pairs(m,'neutral')
            with patch.object(D,'sha',side_effect=AssertionError('unexpected file access')) as read:
                for name in ('DSCF0002.RAF','DSCF9999.RAF'):
                    with self.assertRaisesRegex(ValueError,'allowlist'):pairs.get(name)
                read.assert_not_called()
            with self.assertRaisesRegex(ValueError,'allowlist'):D.Pairs(m,'G',role='development').get('DSCF0001.RAF')

    def test_protected_roles_and_mutated_rows_refused_before_io(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'ROOT',Path(td)):
            m,_,_,_=fixture(Path(td))
            for role in ('monitor','finalHeldOut','K','T8','dr100Eval'):
                with self.assertRaises(ValueError):D.Pairs(m,'G',role=role)
            m['scenes'][0]['inputs']['neutral']['path']='training/checkpoints/protected/secret.fjnrgb'
            with self.assertRaisesRegex(ValueError,'mutated'):D.Pairs(m,'neutral')

    def test_input_and_target_tampering_fail_closed(self):
        for which in ('input','target','sidecar'):
            with self.subTest(which=which),tempfile.TemporaryDirectory() as td,patch.object(D,'ROOT',Path(td)):
                m,_,_,_=fixture(Path(td));row=m['scenes'][0];pairs=D.Pairs(m,'neutral')
                p=Path(td)/(row['target_path'] if which=='target' else row['inputs']['neutral']['path'])
                if which=='sidecar':p=Path(str(p)+'.json')
                p.write_bytes(p.read_bytes()+b'x')
                with self.assertRaisesRegex(ValueError,'hash'):pairs.get('DSCF0001.RAF')

    def test_constructor_snapshots_rows(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'ROOT',Path(td)):
            m,_,_,_=fixture(Path(td));pairs=D.Pairs(m,'neutral')
            m['scenes'][0]['inputs']['neutral']['path']='bad'
            self.assertEqual(list(pairs.get('DSCF0001.RAF')[0].shape),[1,3,3,4])

    def test_wrong_policy_or_paired_source_refused_before_io(self):
        for change in ('kind','source'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as td,patch.object(D,'ROOT',Path(td)):
                m,_,_,_=fixture(Path(td));row=m['scenes'][0]
                if change=='kind':row['neutral_kind']='W'
                else:row['inputs']['neutral']['source_raf_sha256']='c'*64
                m['_rows_sha256']=D.rows_digest(m['scenes']);pairs=D.Pairs(m,'neutral')
                with patch.object(D,'sha',side_effect=AssertionError('unexpected file access')) as read:
                    with self.assertRaises(ValueError):pairs.get('DSCF0001.RAF')
                    read.assert_not_called()


class Prerequisites(unittest.TestCase):
    def test_missing_input_receipt_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'pending'):
                D.receipt(Path(td)/'absent.json','a'*64,116,Path(td)/'never-open.py')

    def test_profile_and_identity_exact_paths(self):
        base={'shoot':'Shoot 7','reuse':None,'G':{'file':'fixed-G'}}
        self.assertEqual(D.expected_neutral_path({**base,'kind':'G'},'output'),'fixed-G')
        self.assertEqual(D.expected_neutral_path({**base,'kind':'P'},'output'),'output/Shoot 7/N-neutral.fjnrgb')
        self.assertEqual(D.expected_neutral_path({**base,'kind':'W'},'output'),'output/Shoot 7/N-neutral.fjlrg')


def metadata_fixture():
    parent={'scenes':[]};fm={'scenes':[],'output_dir':'training/checkpoints/neutral-wb-fit-export'}
    em={'scenes':[],'output_dir':'training/checkpoints/neutral-wb-development-export'}
    fp={'rows':[]};ep={'rows':[]}
    for i in range(583):
        role='fit' if i<467 else 'development';name=f'DSCF{i+1:04d}.RAF';shoot=f'Shoot {i+1}'
        g={'path':f'training/checkpoints/provia-input-v2-train-g/{shoot}/input-v2-linear-asshot.fjlrg',
           'sha256':'a'*64,'sidecar_sha256':'b'*64,'source_raf_sha256':f'{i:064x}','wasm_sha256':'c'*64}
        old={'name':name,'shoot':shoot,'role':role,'session':0 if role=='fit' else 1,
             'captureDynamicRange':400,'inputs':{'linear':g},'target_path':f'calibration-input/provia-color-target-cache/{shoot}.npy',
             'npy_sha256':'d'*64,'fjlrg_wh':[4,3]}
        parent['scenes'].append(old)
        if 455<=i<467:continue  # unsupported FIT presets, no fallback
        j=i if role=='fit' else i-467
        kind=('W' if j<288 else 'P' if j<454 else 'G') if role=='fit' else ('W' if j<5 else 'P' if j<79 else 'G')
        s={k:old[k] for k in ('name','shoot','role','session')}
        s.update(kind=kind,capture_dynamic_range=400,raf_sha256=g['source_raf_sha256'],reuse=None,
                 source_wb_mode=0 if kind=='W' else 4080,source_kelvin=None if kind=='W' else 6600,
                 G={'file':g['path'],'sha256':g['sha256'],'sidecar_sha256':g['sidecar_sha256']},
                 contract={'G':D.V.CONTRACT_ASSHOT,'W':D.V.CONTRACT_AUTOWB,'P':D.P.CONTRACT}[kind])
        cohort,proof=(fm,fp) if role=='fit' else (em,ep);cohort['scenes'].append(s)
        r={'name':name,'kind':kind,'verified':True}
        if kind=='G':r['identity_reference_only']=True
        else:r.update(control_matches_G=True,neutral_repeat_exact=True,neutral_file=D.expected_neutral_path(s,cohort['output_dir']),
                      neutral_sha256='e'*64,neutral_sidecar_sha256='f'*64)
        proof['rows'].append(r)
    return parent,fm,em,fp,ep


class MetadataAssembly(unittest.TestCase):
    def test_exact_eligible_pool_and_identity_mapping(self):
        context=metadata_fixture();rows=D.assemble_rows(*context)
        self.assertEqual(len(rows),571);self.assertEqual(sum(r['role']=='fit' for r in rows),455)
        self.assertFalse(set(r['name'] for r in rows)&{f'DSCF{i:04d}.RAF' for i in range(456,468)})
        self.assertEqual(sum(r['role']=='development' and r['policy_subgroup']=='identity' for r in rows),37)
        for row in rows:
            if row['neutral_kind']=='G':self.assertEqual(row['inputs']['G'],row['inputs']['neutral'])

    def test_missing_duplicate_and_wrong_role_receipts_fail(self):
        for mutation in ('missing','duplicate','swapped'):
            with self.subTest(mutation=mutation):
                p,f,e,fp,ep=metadata_fixture()
                if mutation=='missing':fp['rows'].pop()
                elif mutation=='duplicate':f['scenes'][0]=copy.deepcopy(f['scenes'][1])
                else:fp['rows'][0],ep['rows'][0]=ep['rows'][0],fp['rows'][0]
                with self.assertRaises(ValueError):D.assemble_rows(p,f,e,fp,ep)

    def test_true_flags_required_not_strings_and_path_cannot_escape(self):
        for mutation in ('flag','path','kind','control','source'):
            with self.subTest(mutation=mutation):
                p,f,e,fp,ep=metadata_fixture()
                if mutation=='flag':fp['rows'][0]['control_matches_G']='true'
                elif mutation=='path':fp['rows'][0]['neutral_file']='training/checkpoints/protected/secret.fjlrg'
                elif mutation=='kind':fp['rows'][0]['kind']='G'
                elif mutation=='control':f['scenes'][0]['G']['file']='foreign'
                else:f['scenes'][0]['raf_sha256']='a'*64
                with self.assertRaises(ValueError):D.assemble_rows(p,f,e,fp,ep)


if __name__=='__main__':unittest.main()
