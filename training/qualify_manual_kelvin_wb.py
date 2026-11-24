"""Narrow manual-Kelvin6600 WB-application primitive, separate from any input arm.

Three metadata-selected FIT RAFs outside the24 archived WB references.
No Fuji target pixels, model inference, canonical inputs or runtime changes.
Tests sparse native camera-RGB photosites only, before demosaic/matrix/resize.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from math import lcm
from pathlib import Path

import numpy as np

import matched_gamma_data as D
import audit_wb_alignment as A
import eval_provia_wb_primitive as S

ALIGN_SHA='6f6dac1c4bdd619ae1daacac36581866b1b71b059fc970f23145b89ffbed8ffc'
MANIFEST=D.DIAG/'manual-kelvin-wb-primitive-manifest.json'
REPORT=D.DIAG/'manual-kelvin-wb-primitive-report.json'
OUT=D.CKPT/'manual-kelvin-wb-primitive'
ARMS=('C','D','E','E-repeat')
OLD=S.MANIFEST
CODE=Path(__file__).resolve()


def integer_levels(ratios):
    """Canonical positive integer RGBG from EXACT ratio denominators, no assumed green."""
    r,b=ratios['R/G'],ratios['B/G']
    D.need(r>0 and b>0,'Invalid profile ratios')
    g=lcm(r.denominator,b.denominator)
    return [int(r*g),g,int(b*g),g]


def select(rows,excluded):
    groups=defaultdict(list)
    for r in rows:
        if (r['role']=='fit' and r['source_wb_mode']==4080 and r['source_kelvin']==6600
            and r['source_wb_fine_tune_raw']==[20,-60] and r['name'] not in excluded):
            groups[r['session']].append(r)
    D.need(len(groups)>=3,'Insufficient fit sessions for primitive')
    selected=[]
    for session in sorted(groups)[:3]:
        ordered=sorted(groups[session],key=lambda r:hashlib.sha256(('manual-kelvin-native-v1/'+r['name']).encode()).hexdigest())
        selected.append(ordered[0])
    return selected


def expected_options(base,scene,arm):
    D.need(arm in ARMS,'Unknown primitive arm')
    result=dict(base)
    if arm=='C':result['useCameraWb']=True
    else:result['userMul']=scene['as_shot_rgbg'] if arm=='D' else scene['neutral_rgbg']
    return result


def prepare():
    am,rows,refs=A.load(ALIGN_SHA)
    alignment=json.loads(A.REPORT.read_text());proof=D.DIAG/'wb-alignment-verification.json'
    checked=json.loads(proof.read_text())
    D.need(checked['verified'] is True and checked['report_sha256']==D.sha(A.REPORT)
           and alignment['manifest_sha256']==ALIGN_SHA,'Alignment metadata proof missing/stale')
    profile=alignment['profiles']['profiles']['4080/6600']
    D.need(profile['status']=='candidate' and profile['fit_scenes']==6 and profile['fit_sessions']==[3,8,11]
           and len(alignment['kelvin6600_fit_LOSO'])==6
           and all(r['exact_target_metadata_match'] is True for r in alignment['kelvin6600_fit_LOSO']),
           'Exact fit-only profile provenance/LOSO not met')
    levels=integer_levels(A.exact_ratios(profile['target_ratios']))
    D.need(levels[1]==min(levels[:3]),'This narrow primitive requires green minimum in candidate profile')
    scenes=[]
    for r in select(rows,{ref['name'] for ref in refs}):
        cam=r['cam_mul'];D.need(cam[1]==min(cam[:3]),'Green not minimum in chosen as-shot control')
        source=cam[:3]+[cam[1]]
        scenes.append({k:r[k] for k in ('name','shoot','role','session','source_wb_mode','source_kelvin','source_wb_fine_tune_raw','raf_sha256','cam_mul')})
        scenes[-1].update(as_shot_rgbg=source,neutral_rgbg=levels,
                         expected_gain_RGB=S.expected_gains(source,levels),shift=[1,-3])
    original=json.loads(OLD.read_text());base=original['base_options']
    D.need(base['gamm']==[1,1,0,0,0,0] and base['outputColor']==0 and base['noInterpolation'] is True
           and base['useCameraWb'] is False and base['useAutoWb'] is False
           and base['highlight']==0 and base['outputBps']==16 and base['halfSize'] is False
           and base['noAutoBright'] is True and 'userMul' not in base,'Wrong primitive options')
    files=[CODE,D.TRAIN/'test_qualify_manual_kelvin_wb.py',D.ROOT/'scripts/probe-manual-kelvin-wb.mjs',
           Path(S.__file__).resolve(),A.MANIFEST,A.REPORT,proof,OLD,D.TRAIN/'audit_wb_alignment.py']
    files += [D.ROOT/p for p in original['bindings'] if p.startswith('node_modules/libraw-wasm/')]
    # Bind the already qualified/pinned implementation of the primitive stage.
    files += [D.DIAG/'provia-wb-primitive-source-pin.json']
    files += [D.ROOT/r['path'] for r in json.loads((D.DIAG/'provia-wb-primitive-source-pin.json').read_text())['sources']]
    manifest={'scope':__doc__.strip(),'alignment_manifest_sha256':ALIGN_SHA,
              'alignment_report_sha256':D.sha(A.REPORT),'profile':profile,'neutral_rgbg':levels,
              'selection':'Current fit only, Kelvin6600/fineTune[20,-60], exclude ALL24 archived WB references; one per first3ascending eligible sessions, inside session lowest salted-name SHA. No pixels/errors select cases.',
              'base_options':base,'sample_grid':{'offset':15,'stride':31},'scenes':scenes,
              'arms':{'C':'Camera as-shot WB, no userMul','D':'Explicit source as-shot RGBG','E':'Fixed fit-only neutral Kelvin6600 profile','E-repeat':'Exact deterministic repeat of E'},
              'output_dir':str(OUT.relative_to(D.ROOT)),
              'controls':'C/D full native hash and samples must be identical; E/E-repeat full native hash and samples identical. Sparse 16bit3color6x6CFA structure and dimensions identical in all4 arms, else STOP.',
              'primitive_limits':{'minimum_sites_per_channel':1000,'mask_min':.01,'mask_max':.8,'mean_abs_codes_max':1.,'max_abs_codes_max':3.},
              'interpretation':'Physical application of declared sensor WB ratios only. Target-derived factory calibration is explicit; no target pixels/per-image target fields. Does not qualify full sRGB/demosaic/color rendering, a new input contract, broader manual WB, recipes, runtime policy or model selection. Old source-only manual inverse STOP and source-Auto gates unchanged.',
              'boundaries':'Only3 allowlisted fit RAFs, sparse ignored native samples, no TIFF/cache/model inference/development/protected K/monitor/finalHeldOut/T8/dr100Eval. No training, canonical writes or app change.',
              'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files}}
    D.write_once_json(MANIFEST,manifest)
    print('MANUAL_KELVIN_PRIMITIVE_MANIFEST',D.sha(MANIFEST),[s['name'] for s in scenes],flush=True)


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected
           and D.committed(MANIFEST),'Unpinned/uncommitted primitive')
    m=json.loads(MANIFEST.read_text())
    for rel,h in m['bindings'].items():
        p=D.ROOT/rel;D.need(D.sha(p)==h,'Bound primitive file changed: '+rel)
        if not rel.startswith('node_modules/') and 'checkpoints/' not in rel:D.need(D.committed(p),'Uncommitted binding: '+rel)
    _,rows,refs=A.load(ALIGN_SHA)
    expected_rows=select(rows,{r['name'] for r in refs})
    D.need([r['name'] for r in expected_rows]==[r['name'] for r in m['scenes']],'Source selection differs')
    D.need(len(m['scenes'])==3 and len({r['session'] for r in m['scenes']})==3,'Scene/session coverage')
    for s,r in zip(m['scenes'],expected_rows):
        D.need(all(s[k]==r[k] for k in ('name','shoot','role','session','source_wb_mode','source_kelvin','source_wb_fine_tune_raw','raf_sha256','cam_mul')),'Source metadata differs')
        D.need(s['expected_gain_RGB']==S.expected_gains(s['as_shot_rgbg'],s['neutral_rgbg']),'Gain expectation drift')
    D.need(m['primitive_limits']=={'minimum_sites_per_channel':1000,'mask_min':.01,'mask_max':.8,'mean_abs_codes_max':1.,'max_abs_codes_max':3.},'Primitive tolerance drift')
    return m


def arm_samples(scene,label,arm,m):
    D.need(arm['declared_options']==expected_options(m['base_options'],scene,label),'Declared options differ')
    compact=json.dumps(arm['declared_options'],separators=(',',':'))
    D.need(hashlib.sha256(compact.encode()).hexdigest()==arm['declared_options_sha256'],'Options fingerprint differs')
    meta=arm['returned_metadata']
    D.need(meta['cam_mul']==scene['cam_mul'] and meta['camera_model'].strip()=='X100VI'
           and meta['fujiWB']==4080,'Decoded source metadata differs')
    h=arm['native_sha256']
    D.need(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Invalid native digest')
    width,height=arm['shape']['width'],arm['shape']['height']
    D.need(type(width) is int and type(height) is int and width>15 and height>15,'Invalid native size')
    counts=np.asarray(arm['phase_counts'],dtype=np.float64)
    D.need(counts.shape==(144,) and np.isfinite(counts).all() and np.equal(counts,np.floor(counts)).all()
           and (counts>=0).all() and counts.sum()==width*height,'Native phase coverage differs')
    path=D.ROOT/arm['samples_file']
    D.need(path.resolve().parent==OUT.resolve() and path.name==f"{scene['name']}-{label}.samples.u16le"
           and D.sha(path)==arm['samples_sha256'],'Native sample file/path/hash differs')
    values=np.fromfile(path,dtype='<u2')
    nr,nc=S.grid_shape(width,height)
    D.need(values.size==nr*nc*3,'Native sample grid length differs')
    return values


def run(expected):
    m=load(expected);D.need(not REPORT.exists(),'Prior result exists')
    complete=OUT/'decode-complete.json';progress=OUT/'decode-progress.json'
    path=complete if complete.exists() else progress
    decoded=json.loads(path.read_text())
    D.need(decoded['manifest_sha256']==expected,'Decoder belongs to another primitive')
    allowed={s['name']:s for s in m['scenes']};found={r['name']:r for r in decoded['scenes']}
    D.need(len(found)==len(decoded['scenes']) and set(found)<=set(allowed),'Duplicate/unapproved decoded scene')
    results=[]
    for name,row in found.items():
        s=allowed[name];arms=row['arms']
        if row.get('stopped') or set(arms)!=set(ARMS) or any(a.get('structure_error') for a in arms.values()):
            results.append({'name':name,'passed':False,'reason':'Incomplete/failed native structural arm','decoder_stop':decoded.get('stopped')});continue
        samples={a:arm_samples(s,a,arms[a],m) for a in ARMS}
        patterns=[]
        for a in ('D','E','E-repeat'):
            pattern,problems=S.check_structure(arms['C'],arms[a])
            D.need(not problems,'CFA structure changed: '+str(problems));patterns.append(pattern)
        D.need(patterns[0]==patterns[1]==patterns[2],'Native patterns differ')
        controls=all(arms[a]['native_sha256']==arms[b]['native_sha256']
                     and arms[a]['samples_sha256']==arms[b]['samples_sha256']
                     and np.array_equal(samples[a],samples[b]) for a,b in (('C','D'),('E','E-repeat')))
        result=S.evaluate_scene(s,arms['D'],arms['E'],samples['D'],samples['E'])
        result.update(controls_and_repeat_exact=controls,passed=bool(controls and result['primitive_passed']))
        results.append(result)
    passed=(complete.exists() and decoded.get('completed') is True and set(found)==set(allowed)
            and len(results)==3 and all(r['passed'] for r in results))
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'Selected R11 changed')
    report={'manifest_sha256':expected,'decode_report_sha256':D.sha(path),'verdict':'PASSED' if passed else 'NOT_QUALIFIED',
            'rows':results,'limits':m['interpretation'],'boundaries':m['boundaries'],'R11_unchanged':D.R11_SHA}
    D.write_once_json(REPORT,report)
    print(json.dumps(report,indent=2),flush=True)
    if not passed:raise SystemExit(1)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'))
    ap.add_argument('--manifest-sha');a=ap.parse_args()
    if a.command=='prepare':prepare()
    else:run(a.manifest_sha)


if __name__=='__main__':main()
