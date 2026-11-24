"""Four-fit-source demosaiced-RGB qualification for the isolated neutral profile.

No Fuji target pixels, development media, model inference or training. This
qualifies research input provenance/identity/determinism, not Fuji rendering.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matched_gamma_data as D
import audit_target_wb_headers as H
import fjnrgb_profile as P
import fjlrg_v2 as V2

HEADER_SHA='16fcb23a2d4fb48aad7f12251c00015f4231c1373214618b264a21b7eb889ddc'
MANIFEST=D.DIAG/'profile-rgb-sentinel-manifest.json'
REPORT=D.DIAG/'profile-rgb-sentinel-report.json'
OUT=D.CKPT/'profile-rgb-sentinels'
CODE=Path(__file__).resolve()
FIRST_THREE=('DSCF0464.RAF','DSCF0809.RAF','DSCF1011.RAF')
SALT='manual-profile-rgb-sentinel-v1\n'


def choose(rows):
    eligible=[r for r in rows if r['role']=='fit' and r['frozen_source_only_policy']['basis']=='calibrated-kelvin-profile-candidate']
    by={r['name']:r for r in eligible}
    candidates=[r for r in eligible if r['session']==13]
    D.need(candidates and all(n in by for n in FIRST_THREE),'Sentinel coverage missing')
    last=min(candidates,key=lambda r:hashlib.sha256((SALT+r['name']).encode()).hexdigest())
    result=[by[n] for n in FIRST_THREE]+[last]
    D.need(len({r['name'] for r in result})==4 and len({r['session'] for r in result})==4,'Four distinct fit sessions required')
    return result


def options(scene,arm):
    D.need(arm in ('C','P','R'),'Unknown arm')
    o=P.canonical_options()
    if arm=='C':o['userMul']=[*scene['cam_mul'][:3],scene['cam_mul'][1]]
    return o


def prepare():
    h=H.load(HEADER_SHA);audit=json.loads(H.REPORT.read_text());proof_path=D.DIAG/'target-wb-header-verification.json';proof=json.loads(proof_path.read_text())
    D.need(audit['completed'] and audit['manifest_sha256']==HEADER_SHA and audit['invalid']==audit['settings_mismatch']==0
           and proof['verified'] and proof['report_sha256']==D.sha(H.REPORT),'Target-header audit not verified')
    D.need(all(r['exact_policy_ratio_match'] is not False for r in audit['rows']),'Supported source policy mismatch')
    selected=choose(h['scenes'])
    archived={r['name'] for r in json.loads(H.A.ARCHIVE.read_text())['scenes']}
    D.need(not archived.intersection(r['name'] for r in selected),'Sentinel overlaps archived calibration references')
    baseline=D.CKPT/'provia-input-v2-train-g/complete.json'
    baseline_ver=D.DIAG/'provia-v2-train-export-verification.json';bv=json.loads(baseline_ver.read_text())
    D.need(bv['passed'] and bv['complete_sha256']==D.sha(baseline),'Original G proof changed')
    old=json.loads(baseline.read_text());by={r['name']:r for r in old['scenes']}
    D.need(old['completed'] and len(by)==583,'Original G coverage differs')
    scenes=[]
    for s in selected:
        r=by[s['name']]
        D.need(r['shoot']==s['shoot'] and r['file']==f"training/checkpoints/provia-input-v2-train-g/{s['shoot']}/input-v2-linear-asshot.fjlrg"
               and r['source_coefficients']['asShotCamMul']==s['cam_mul'] and r['source_coefficients']['autoRgbg']==s['auto_rgbg'],
               'G source/path mismatch')
        scenes.append({k:s[k] for k in ('name','shoot','role','session','raf_sha256','cam_mul','auto_rgbg','source_wb_mode','source_kelvin','source_wb_fine_tune_raw')})
        scenes[-1].update(capture_dynamic_range=s['captureDynamicRange'],expected_native_size=r['native_size'],
                         expected_working_size=[r['width'],r['height']],
                         G={'file':r['file'],'sha256':r['sha256'],
                            'native_sha256':r['native_sha256'],'working_payload_sha256':r['working_payload_sha256'],'strips':r['strips']})
        # Use the frozen G verification for the sidecar hash.
        v=next(x for x in bv['scenes'] if x['name']==s['name'])
        scenes[-1]['G']['sidecar_sha256']=v['sidecar_sha256']
        D.need(v['sha256']==r['sha256'] and v['source_raf_sha256']==s['raf_sha256'],'G verification source differs')
    codec_qual=D.DIAG/'fjnrgb-profile-codec-qualification.json';cq=json.loads(codec_qual.read_text())
    D.need(cq['passed'] and cq['real_photo_exports']==0,'Codec synthetic qualification missing')
    files=[CODE,D.TRAIN/'test_profile_rgb_sentinels.py',D.ROOT/'scripts/capture-profile-rgb-sentinels.mjs',
           D.ROOT/'scripts/lib/decode-fjnrgb-profile-research.mjs',D.ROOT/'scripts/test-decode-fjnrgb-profile-research.mjs',
           D.TRAIN/'fjnrgb_profile.py',D.TRAIN/'fjlrg_v2.py',D.ROOT/'scripts/lib/fjnrgb-profile.mjs',
           D.ROOT/'scripts/lib/fjlrg-v2.mjs',D.ROOT/'scripts/lib/raf-wb-coefficients.mjs',codec_qual,
           H.MANIFEST,H.REPORT,proof_path,baseline,baseline_ver]
    files += [D.ROOT/'node_modules/libraw-wasm/dist'/n for n in ('index.js','worker.js','libraw.js','libraw.wasm')]
    m={'scope':__doc__.strip(),'header_audit_manifest_sha256':HEADER_SHA,'scenes':scenes,'output_dir':str(OUT.relative_to(D.ROOT)),
       'wasm_sha256':D.sha(D.ROOT/'node_modules/libraw-wasm/dist/libraw.wasm'),
       'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files},
       'selection':'Exactly three original primitive fit sources plus salted-name-hash minimum among eligible session13 fit rows. Source metadata only, no pixel/error selection.',
       'arms':{'C':'Explicit as-shot userMul control; raw uint16 working payload, truthful options, no false v2 container label.',
               'P':'Fixed fit-derived neutral profile in new research container.','R':'Exact repetition of P.'},
       'gates':['All source/options/camera/WB/binary/provenance/geometry checks pass.',
                'C native RGB SHA and working payload bytes equal original G; native strips equal original G. Entire container equality NOT required because options differ.',
                'P/R native, working payload, strips, container and provenance identical.',
                'P differs from C in native and working hashes (observable treatment sanity only).',
                'Four new containers validate independently in Python; all originals and R11 unchanged.'],
       'limits':'Only input identity/determinism/provenance. No direct RGB gain formula, target matching, Fuji parity, domain bulk export or training follows automatically.'}
    D.write_once_json(MANIFEST,m);print('PROFILE_RGB_SENTINEL_MANIFEST',D.sha(MANIFEST),flush=True)


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected and D.committed(MANIFEST),'Unpinned/uncommitted RGB scope')
    m=json.loads(MANIFEST.read_text())
    for rel,h in m['bindings'].items():
        p=D.ROOT/rel;D.need(D.sha(p)==h,'Bound artifact drift: '+rel)
        if not rel.startswith('node_modules/') and 'checkpoints/' not in rel:D.need(D.committed(p),'Uncommitted binding')
    h=H.load(HEADER_SHA)
    D.need([r['name'] for r in m['scenes']]==[r['name'] for r in choose(h['scenes'])],'Source selection changed')
    source={r['name']:r for r in h['scenes']}
    for scene in m['scenes']:
        prior=source[scene['name']]
        D.need(all(scene[k]==prior[k] for k in ('name','shoot','role','session','raf_sha256','cam_mul','auto_rgbg','source_wb_mode','source_kelvin','source_wb_fine_tune_raw'))
               and scene['capture_dynamic_range']==prior['captureDynamicRange'],'Source metadata differs')
    D.need(m['output_dir']==str(OUT.relative_to(D.ROOT)),'Output namespace differs')
    return m


def run(expected):
    m=load(expected);D.need(not REPORT.exists(),'Prior RGB result exists')
    path=OUT/'complete.json';complete=json.loads(path.read_text())
    D.need(complete['completed'] and complete['manifest_sha256']==expected,'Incomplete/wrong RGB capture')
    rows=complete['scenes'];by={r['name']:r for r in rows}
    D.need(len(rows)==len(by)==4 and set(by)=={s['name'] for s in m['scenes']},'Wrong scene coverage')
    result=[]
    for s in m['scenes']:
        row=by[s['name']];D.need(set(row['arms'])=={'C','P','R'} and row['raf_sha256_after']==s['raf_sha256'],'Incomplete source/arms')
        old=s['G'];gfile=D.ROOT/old['file']
        D.need(D.sha(gfile)==old['sha256'] and D.sha(Path(str(gfile)+'.json'))==old['sidecar_sha256'],'Original G changed')
        g=V2.load_fjlrg_v2(gfile,expected_contract=V2.CONTRACT_ASSHOT,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=m['wasm_sha256'])
        payload=g.rgb_uint16.astype('<u2').tobytes();D.need(hashlib.sha256(payload).hexdigest()==old['working_payload_sha256'],'Original G payload proof differs')
        old_strip=D.ROOT/old['strips']['file'];D.need(D.sha(old_strip)==old['strips']['sha256'],'Original G strips changed')
        arrays={};provenances={}
        for arm in ('C','P','R'):
            a=row['arms'][arm];f=D.ROOT/a['file'];strip=D.ROOT/a['strips_file']
            D.need(f.parent==OUT/s['shoot'] and strip.parent==f.parent,'Output path escaped')
            D.need(D.sha(f)==a['sha256'] and D.sha(strip)==a['strips_sha256'],'New output hash mismatch')
            D.need(a['declared_options']==options(s,arm) and a['native_size']==s['expected_native_size'] and [a['width'],a['height']]==s['expected_working_size'],'Options/geometry differs')
            md=a['returned_metadata'];D.need(md['camera_model']=='X100VI' and md['fujiWB']==4080 and md['cam_mul']==s['cam_mul'],'Current decoder metadata differs')
            D.need(a['source_coefficients']['asShotCamMul']==s['cam_mul'] and a['source_coefficients']['autoRgbg']==s['auto_rgbg'],'RAF coefficient re-read differs')
            if arm=='C':
                D.need(f.read_bytes()==payload and a['native_sha256']==old['native_sha256'] and strip.read_bytes()==old_strip.read_bytes(),'Explicit as-shot control differs from G')
                arrays[arm]=g.rgb_uint16
            else:
                dec=P.load(f,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=m['wasm_sha256'])
                D.need(dec.provenance['decoded_native_size']==s['expected_native_size'] and dec.provenance['capture_dynamic_range']==s['capture_dynamic_range'],'Profile native/DR provenance differs')
                arrays[arm]=dec.rgb_uint16;provenances[arm]=dec.provenance
            D.need(a['working_payload_sha256']==hashlib.sha256(arrays[arm].astype('<u2').tobytes()).hexdigest(),'Working payload differs')
        p,r=row['arms']['P'],row['arms']['R']
        D.need(all(p[k]==r[k] for k in ('native_sha256','working_payload_sha256','sha256','strips_sha256')) and provenances['P']==provenances['R'],'Profile repeat differs')
        D.need(p['native_sha256']!=old['native_sha256'] and p['working_payload_sha256']!=old['working_payload_sha256'],'Treatment not observable; no qualification')
        delta=arrays['P'].astype(float)-arrays['C'].astype(float)
        result.append({'name':s['name'],'session':s['session'],'passed':True,'control_matches_G':True,'profile_repeat_exact':True,
                       'mean_abs_working_codes_RGB':np.abs(delta).mean(axis=(0,1)).tolist(),'working_mean_delta_RGB':delta.mean(axis=(0,1)).tolist(),
                       'interpretation':'Descriptive RGB change only; no target/expected RGB gain comparison.'})
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'Selected R11 changed')
    out={'manifest_sha256':expected,'complete_sha256':D.sha(path),'passed':True,'rows':result,
         'qualification':'RESEARCH_INPUT_IDENTITY_DETERMINISM_PROVENANCE_ONLY','R11_sha256':D.R11_SHA,'limits':m['limits']}
    D.write_once_json(REPORT,out);print('PROFILE_RGB_SENTINELS_PASS',len(result),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'));ap.add_argument('--manifest-sha');a=ap.parse_args()
    prepare() if a.command=='prepare' else run(a.manifest_sha)


if __name__=='__main__':main()
