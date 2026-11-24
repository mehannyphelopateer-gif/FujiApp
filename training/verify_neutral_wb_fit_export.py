"""Independent strict stored-byte/sidecar/identity/repeat checks. FIT artifacts only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matched_gamma_data as D
import prepare_neutral_wb_fit_export as A
import fjlrg_v2 as V
import fjnrgb_profile as P


def digest(b):return hashlib.sha256(b).hexdigest()


def inside(path):
    p=D.ROOT/path
    # Only paths frozen from the allowed source artifacts or this job's output.
    assert path.startswith(('training/checkpoints/provia-input-v2-train-g/','training/checkpoints/provia-input-v2-wb-pilot/',
                            'training/checkpoints/profile-rgb-sentinels/','training/checkpoints/neutral-wb-fit-export/'))
    assert '..' not in Path(path).parts
    return p


def artifact(s,r,arm,wasm):
    assert s['role']=='fit' and s['kind'] in ('W','P') and arm in ('C','N','R')
    if s['reuse']:
        old=s['reuse']['source_record']['arms']
        prior=old[{'C':'control','N':'W','R':'W-repeat'}[arm]] if s['kind']=='W' else old[{'C':'C','N':'P','R':'R'}[arm]]
        expected_file=prior['file'];expected_strips=prior['strips']['file'] if s['kind']=='W' else prior['strips_file']
    else:
        suffix='-control.u16le' if arm=='C' else '-neutral.fjlrg' if s['kind']=='W' else '-neutral.fjnrgb'
        expected_file=f"training/checkpoints/neutral-wb-fit-export/{s['shoot']}/{arm}{suffix}"
        expected_strips=f"training/checkpoints/neutral-wb-fit-export/{s['shoot']}/{arm}-native-strips.u16le"
    assert r['file']==expected_file and r['strips_file']==expected_strips, 'Outside exact scene/arm allowlist'
    file=inside(r['file']);b=file.read_bytes();assert digest(b)==r['sha256']
    strip=inside(r['strips_file']).read_bytes();assert digest(strip)==r['strips_sha256']
    rows=list(range(64,s['expected_native_size'][1],128))
    assert r['strip_rows']==rows and len(strip)==len(rows)*s['expected_native_size'][0]*6
    assert r['native_size']==s['expected_native_size'] and [r['width'],r['height']]==s['expected_working_size']
    opts=P.canonical_options();opts['userMul']=[*s['cam_mul'][:3],s['cam_mul'][1]] if arm=='C' else s['declared_levels_rgbg']
    canonical=json.dumps(dict(sorted(opts.items())),separators=(',',':'))
    assert r['options_json']==canonical and digest(canonical.encode())==r['options_sha256']
    assert r['source_coefficients']=={'autoRgbg':s['auto_rgbg'],'asShotCamMul':s['cam_mul']}
    assert isinstance(r['native_sha256'],str) and len(r['native_sha256'])==64 and all(c in '0123456789abcdef' for c in r['native_sha256'])
    if s['kind']=='W':
        assert r['returned_metadata'] is None and 'not returned' in r['metadata_note'].lower()
    else:
        assert r['returned_metadata']['camera_model']=='X100VI'
        assert r['returned_metadata']['fujiWB']==4080 and r['returned_metadata']['cam_mul']==s['cam_mul']
    if arm=='C':
        payload=b;assert r['contract'] is None and r.get('sidecar_sha256') is None
    else:
        assert r['contract']==s['contract'] and D.sha(Path(str(file)+'.json'))==r['sidecar_sha256']
        if s['kind']=='W':
            a=V.load_fjlrg_v2(file,expected_contract=V.CONTRACT_AUTOWB,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=wasm)
        else:
            a=P.load(file,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=wasm)
        assert [a.width,a.height]==s['expected_working_size']
        payload=a.rgb_uint16.astype('<u2',copy=False).tobytes()
        provenance=json.loads(Path(str(file)+'.json').read_text())
        assert provenance['decoded_native_size']==s['expected_native_size']
        assert provenance['options_json']==canonical and provenance['capture_dynamic_range']==s['capture_dynamic_range']
    assert len(payload)==s['expected_working_size'][0]*s['expected_working_size'][1]*6
    assert digest(payload)==r['working_payload_sha256']
    return b,payload,strip


def check_scene(s,seal,wasm):
    assert s['role']=='fit'
    assert seal['name']==s['name'] and seal['role']=='fit' and seal['kind']==s['kind']
    assert seal['source_raf_sha256']==s['raf_sha256']
    old=inside(s['G']['file']);assert D.sha(old)==s['G']['sha256'] and D.sha(Path(str(old)+'.json'))==s['G']['sidecar_sha256']
    g=V.load_fjlrg_v2(old,expected_contract=V.CONTRACT_ASSHOT,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=wasm)
    gpayload=g.rgb_uint16.astype('<u2',copy=False).tobytes()
    assert digest(gpayload)==s['G']['working_payload_sha256']
    if s['kind']=='G':
        assert seal['identity']==s['G'] and not seal.get('arms')
        return {'name':s['name'],'kind':'G','verified':True,'identity_reference_only':True}
    assert set(seal['arms'])=={'C','N','R'}
    checked={arm:artifact(s,seal['arms'][arm],arm,wasm) for arm in ('C','N','R')}
    c,n,r=(seal['arms'][k] for k in ('C','N','R'))
    assert checked['C'][1]==gpayload and c['native_sha256']==s['G']['native_sha256']
    oldstrips=inside(s['G']['strips']['file']);assert D.sha(oldstrips)==s['G']['strips']['sha256']
    assert checked['C'][2]==oldstrips.read_bytes()
    for k in ('native_sha256','working_payload_sha256','sha256','strips_sha256','options_json','options_sha256','sidecar_sha256'):
        assert n[k]==r[k], 'Repeat mismatch '+k
    assert checked['N']==checked['R']
    assert Path(str(inside(n['file']))+'.json').read_bytes()==Path(str(inside(r['file']))+'.json').read_bytes()
    return {'name':s['name'],'kind':s['kind'],'verified':True,'control_matches_G':True,'neutral_repeat_exact':True,
            'neutral_changed_payload':checked['N'][1]!=gpayload,'neutral_file':n['file'],'neutral_sha256':n['sha256'],
            'neutral_sidecar_sha256':n['sidecar_sha256'],'native_hash_independently_recomputed':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--canaries',action='store_true');a=ap.parse_args()
    m=A.load(a.manifest_sha);scenes=m['scenes']
    if a.canaries:scenes=[s for s in scenes if not s['requires_new_decode'] or s['name'] in m['canaries']]
    expected={s['name'] for s in scenes};rows=[];seals={}
    for s in scenes:
        seal_file=A.OUT/s['shoot']/'scene-seal.json';seal=json.loads(seal_file.read_text())
        assert seal['manifest_sha256']==a.manifest_sha
        rows.append(check_scene(s,seal,m['wasm_sha256']));seals[s['name']]=D.sha(seal_file)
        if s['name'] in m['canaries']:
            assert rows[-1]['neutral_changed_payload']
            assert seal['arms']['N']['native_sha256']!=s['G']['native_sha256']
        print('VERIFY_NEUTRAL_WB_FIT',len(rows),len(scenes),s['name'],s['kind'],flush=True)
    assert len(rows)==len(expected)==(16 if a.canaries else 455)
    if not a.canaries:
        complete=A.read(A.OUT/'complete.json')
        assert complete['completed'] and complete['manifest_sha256']==a.manifest_sha and complete['scene_seal_sha256']==seals
    assert D.sha(D.R11_PATH)==D.R11_SHA
    out={'verified':True,'completed_fit_rows':len(rows),'manifest_sha256':a.manifest_sha,'canary_only':a.canaries,
         'scene_seal_sha256':seals,'rows':rows,'verification_script_sha256':D.sha(Path(__file__)),
         'limits':'FIT-only input identity/repeat/provenance qualification. Saved bytes/sidecars independently checked; full native hashes remain browser-reported. No target or Fuji colour scores, development/protected pixels, training/model/app changes.',
         'R11_sha256':D.R11_SHA}
    path=D.DIAG/('neutral-wb-fit-canary-verification.json' if a.canaries else 'neutral-wb-fit-export-verification.json')
    D.write_once_json(path,out);print('NEUTRAL_WB_FIT_VERIFIED',len(rows),flush=True)


if __name__=='__main__':main()
