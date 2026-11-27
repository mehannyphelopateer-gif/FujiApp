"""Phase B: bounded WB/capture-tag consistency and existing-policy metadata classification.

No source-image decoding, references, arrays, fitting, tier assignment or
admission. Phase A evidence stays immutable; unsupported metadata is held.
"""
import argparse
import json
import subprocess
from pathlib import Path
from fractions import Fraction
import check_dr100_training_candidates as A
import wb_alignment_metrics as W
import neutral_wb_domain_plan as N

ROOT=A.ROOT
PHASE_A_SHA='3446ea4b44128249b129387d61943742c9ca69f17ad38771e1bdfd41ef4fd54b'
SCOPE=ROOT/'training/diagnostics/dr100-training-candidates-phaseB-scope.json'
OUTPUT=ROOT/'training/diagnostics/dr100-training-candidates-phaseB.json'
PROFILE=ROOT/'training/diagnostics/wb-alignment-report.json'
PLAN=ROOT/'training/coordination/dr100-candidate-phaseB-plan.md'
HOST=ROOT/'scripts/read-dr100-candidate-metadata.mjs'
CODE=[Path(__file__).resolve(),ROOT/'training/test_dr100_candidate_wb.py',HOST,
      ROOT/'scripts/test-dr100-candidate-host.mjs',ROOT/'scripts/lib/raf-training-candidate-metadata.mjs',
      ROOT/'scripts/test-raf-training-candidate-metadata.mjs',ROOT/'scripts/lib/raf-wb-coefficients.mjs',
      ROOT/'training/wb_alignment_metrics.py',ROOT/'training/neutral_wb_domain_plan.py',ROOT/'training/fjnrgb_profile.py',
      PROFILE,PLAN,ROOT/'package-lock.json',A.SCOPE,A.OUTPUT,A.FRESH]
LEGACY_TAGS={'whiteBalanceMode':'WhiteBalance','exposureCount':'ExposureCount','dynamicRange':'DynamicRangeSetting',
             'dynamicRangeSetting':'DynamicRangeSetting2','developmentDynamicRange':'DevelopmentDynamicRange',
             'autoDynamicRange':'AutoDynamicRange','dRangePriority':'DRangePriority',
             'dRangePriorityAuto':'DRangePriorityAuto','dRangePriorityFixed':'DRangePriorityFixed'}


def evidence():
    A.load_scope(PHASE_A_SHA)
    for p in (A.OUTPUT,A.FRESH):A.committed(p)
    prior=A.read(A.OUTPUT)
    A.need(prior['completed_phaseA'] is True and prior['scope_sha256']==PHASE_A_SHA
           and prior['training_admission'] is False and A.sha(A.FRESH)==prior['fresh_metadata_sha256'], 'Phase A evidence changed')
    A.need([s['name'] for s in prior['rows']]==list(A.NAMES),'Phase A source coverage changed')
    profiles=A.read(PROFILE)['profiles'];N.validate_frozen_profiles(profiles)
    return prior,profiles


def prepare():
    prior,_=evidence()
    for p in CODE:A.committed(p)
    A.dump(SCOPE,{'format':'dr100-candidate-phaseB-v1','phaseA_scope_sha256':PHASE_A_SHA,
                  'training_admission':False,'protected_access':False,
                  'scenes':[{k:r[k] for k in ('name','path','raf_sha256')} for r in prior['rows']],
                  'bindings':{str(p.relative_to(ROOT)):A.sha(p) for p in CODE},
                  'limits':'Bounded field extraction and unchanged source-only WB policy only; no rendered qualification, training admission, references, export or protected data.'})
    print('DR100_CANDIDATE_PHASE_B_SCOPE',A.sha(SCOPE),flush=True)


def load_scope(expected):
    A.need(isinstance(expected,str) and len(expected)==64 and A.sha(SCOPE)==expected,'Pinned Phase B scope required')
    A.committed(SCOPE);m=A.read(SCOPE)
    A.need(m['format']=='dr100-candidate-phaseB-v1' and m['phaseA_scope_sha256']==PHASE_A_SHA
           and m['training_admission'] is False and m['protected_access'] is False,'Scope authority changed')
    A.need(set(m['bindings'])=={str(p.relative_to(ROOT)) for p in CODE},'Scope bindings expanded/incomplete')
    for rel,h in m['bindings'].items():
        p=ROOT/rel;A.need(A.sha(p)==h,'Changed Phase B binding: '+rel);A.committed(p)
    prior,profiles=evidence()
    A.need(m['scenes']==[{k:r[k] for k in ('name','path','raf_sha256')} for r in prior['rows']],'Source path/hash drift')
    return m,prior,profiles


def legacy_integer(value):
    if value is None:return None
    A.need(isinstance(value,str) and value.split(),'Malformed historical tag')
    return int(value.split()[0])


def classify(metadata,legacy,profiles):
    A.need(metadata['schema']=='raf-training-candidate-metadata/v1','Unknown reader schema')
    ex,fuji,wb=metadata['exif'],metadata['fuji'],metadata['wb']
    A.need(ex['model']==legacy['model']=='X100VI' and ex['dateTimeOriginal']==legacy['dateTimeOriginal']
           and ex['iso']==legacy['iso'],'Standard metadata disagrees with Phase A')
    # Compare every legacy-known critical tag, including absences, independently of the format table.
    for name,key in LEGACY_TAGS.items():
        A.need(fuji[name]==legacy_integer(legacy['fuji'].get(key)),'Raw/legacy tag disagreement: '+name)
    A.need(wb['whiteBalanceModeRaw']==fuji['whiteBalanceMode']
           and wb['colorTemperatureKelvinRaw']==fuji['colorTemperatureKelvin']
           and wb['whiteBalanceFineTuneRaw']==fuji['whiteBalanceFineTune'],'WB output aliases disagree')
    A.need(fuji['autoDynamicRange']==100 and fuji['developmentDynamicRange'] is None,'Capture DR is not Auto-resolved100')
    A.need(fuji['exposureCount'] in (None,1) and all(fuji[k] in (None,0) for k in ('dRangePriority','dRangePriorityAuto','dRangePriorityFixed')),
           'Exposure count/DR priority unsupported')
    mode=wb['whiteBalanceModeRaw'];shift=wb['whiteBalanceFineTuneRaw']
    A.need(type(mode) is int and isinstance(shift,dict) and set(shift)=={'red','blue'},'WB mode/fine tune missing; no guessing')
    kelvin=wb['colorTemperatureKelvinRaw'] if mode==4080 else None
    A.need(mode!=4080 or type(kelvin) is int and kelvin>0,'Active Kelvin missing/ambiguous')
    source={'source_wb_mode':mode,'source_kelvin':kelvin,
            'source_wb_fine_tune_raw':[shift['red'],shift['blue']],
            'cam_mul':wb['asShotCamMul'],'auto_rgbg':wb['autoRgbg']}
    policy=W.propose_policy(source,profiles)
    # Match plan_row's primitive-qualified family gates without fabricating a role/tier.
    kind=N.KINDS.get(policy['basis'],{}).get('kind');levels=None;hold=None
    if kind is not None:
        try:
            if kind=='W':
                A.need(mode==W.AUTO_MODE and kelvin is None,'W requires source mode0 with no active Kelvin')
                levels=list(source['auto_rgbg'])
            elif kind=='P':
                A.need(mode==N.P.SOURCE_WB_MODE and kelvin==N.P.SOURCE_KELVIN
                       and source['source_wb_fine_tune_raw']==list(N.P.SOURCE_FINE_TUNE_RAW),
                       'P requires the primitive-qualified4080/6600/[20,-60] family')
                levels=list(N.P.PROFILE_LEVELS_RGBG)
            else:
                A.need(source['source_wb_fine_tune_raw']==[0,0] and mode!=W.AUTO_MODE,
                       'G requires a zero-shift source outside mode0')
                levels=list(source['cam_mul'])
            chosen={k:Fraction(v['num'],v['den']) for k,v in policy['ratios'].items()}
            A.need(chosen==W.positive_rgb_ratios(levels),'Declared levels disagree with policy ratios')
        except ValueError as exc:
            hold=str(exc);kind=None;levels=None
    else:hold=policy['reason']
    return {'capture_dynamic_range':100,'capture_DR_origin':'Auto resolved, not fixed',
            'source_policy_fields':source,'policy':policy,
            'neutral_kind_candidate':kind,'declared_levels_rgbg_candidate':levels,
            'policy_supported':kind is not None,'policy_gate_hold_reason':hold,
            'kelvin_tag_raw':wb['colorTemperatureKelvinRaw'],
            'qualification':'Metadata consistency/policy classification only; no new rendered input qualified.'}


def run(expected):
    A.need(not OUTPUT.exists(),'Existing Phase B receipt; overwrite refused')
    scope,prior,profiles=load_scope(expected);by={s['name']:s for s in prior['rows']};rows=[]
    for scene in scope['scenes']:
        p=ROOT/scene['path'];A.need(A.sha(p)==scene['raf_sha256'],'Source changed before bounded reader')
        proc=subprocess.run(['node',str(HOST),'--scope-sha',expected,'--name',scene['name']],cwd=ROOT,capture_output=True,text=True)
        record={**scene,'training_admission':False}
        if proc.returncode:
            record.update(metadata_valid=False,hold_reason=proc.stderr.strip())
        else:
            extracted=json.loads(proc.stdout)
            A.need(all(extracted[k]==scene[k] for k in ('name','path','raf_sha256')),'Reader source identity mismatch')
            record['metadata']=extracted['metadata']
            try:
                decision=classify(extracted['metadata'],by[scene['name']]['fresh_legacy'],profiles)
                record.update(metadata_valid=True,decision=decision)
                if not decision['policy_supported']:record['hold_reason']=decision['policy_gate_hold_reason']
            except (ValueError,KeyError,TypeError) as exc:
                record.update(metadata_valid=False,hold_reason=type(exc).__name__+': '+str(exc))
        A.need(A.sha(p)==scene['raf_sha256'],'Source changed after bounded reader')
        rows.append(record);print('DR100_CANDIDATE_B',scene['name'],record.get('metadata_valid'),record.get('hold_reason','metadata candidate only'),flush=True)
    A.need(len(rows)==9 and A.sha(A.STORED)==A.STORED_SHA,'Coverage/old metadata changed')
    A.dump(OUTPUT,{'completed_phaseB':True,'scope_sha256':expected,'script_sha256':A.sha(Path(__file__)),
                   'rows':rows,'metadata_supported_count':sum(r.get('metadata_valid') is True and r['decision']['policy_supported'] for r in rows),
                   'training_admission':False,'protected_data_access':False,
                   'remaining_holds':['complete training-purpose and boundary clearance','genuine neutral Provia TIFF references',
                                      'current true-linear Gv2/neutral input and target qualification','new frozen training-study scope'],
                   'limits':'No source-image decode, rendered WB parity, source copy/export, new role/tier, fitting, acceptance, condition remap or app change. Original exclusions/failed study unchanged.'})
    print('DR100_CANDIDATE_PHASE_B_COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'));ap.add_argument('--scope-sha');a=ap.parse_args()
    prepare() if a.command=='prepare' else run(a.scope_sha)
