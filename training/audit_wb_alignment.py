"""Archived-metadata-only WB alignment/identifiability audit, never a decode policy rollout.

No photo/target/learned-weight loading, pixel inference, factors or inverses.
The calibration uses fit metadata only; all results are retrospective metadata
observations, not independent acceptance or rendered Fuji parity.
"""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
import json
from pathlib import Path

import matched_gamma_data as D
import wb_alignment_metrics as W

PARENT_SHA='ff0fad18b4fd6ca9e8ca36e728e526b95bf975b6dd07f88da844db50089e62b2'
INV=D.DIAG/'provia-v2-source-inventory.json'
QUAL=D.DIAG/'provia-v2-source-qualification.json'
ARCHIVE=D.DIAG/'cn-step1-wb-reference-audit.json'
OLD_MANIFEST=D.DIAG/'provia-wb-metadata-manifest.json'
OLD_AUDIT=D.DIAG/'provia-wb-metadata-audit.json'
MANIFEST=D.DIAG/'wb-alignment-manifest.json'
REPORT=D.DIAG/'wb-alignment-report.json'
POLICY={'nonzero_profile_families':[[4080,6600]],'min_fit_scenes':3,'min_fit_sessions':2,
        'exact_target_ratios_only':True,'development_profile_calibration':False,
        'source_auto_mode':0,'zero_shift_identity_known_modes':True}


def raw_rgb(record):
    values=[record[k] for k in ('WBLevelR','WBLevelG','WBLevelB')]
    W.positive_rgb_ratios(values)
    return values


def source_record(row):
    return {key:row[key] for key in W.SOURCE_KEYS}


def load_sources(parent):
    """Read only named, frozen JSON evidence. No target file loader exists here."""
    inventory=json.loads(INV.read_text())
    rows=inventory['rows'];by={r['name']:r for r in rows}
    allow={r['name']:r for r in parent['scenes']}
    D.need(inventory['completed'] is True and len(rows)==len(by)==583 and set(by)==set(allow),
           'Source inventory coverage differs from583 allowlist')
    qualification=json.loads(QUAL.read_text())
    D.need(qualification['passed'] is True and qualification['camera_model_counts']=={'X100VI':583}
           and {(r['name'],r['shoot']) for r in qualification['rows']}=={(r['name'],r['shoot']) for r in rows},
           'Source camera/metadata qualification changed')
    old=json.loads(OLD_MANIFEST.read_text())
    D.need(old['source_hashes']['cn-step1-wb-reference-audit.json']==D.sha(ARCHIVE),
           'Archive differs from prior hash-bound WB audit')
    audit=json.loads(OLD_AUDIT.read_text())
    D.need(audit['manifest_sha256']==D.sha(OLD_MANIFEST),'Prior metadata audit binding differs')
    references=json.loads(ARCHIVE.read_text())['scenes']
    D.need(len(references)==24 and len({r['name'] for r in references})==24
           and {r['name'] for r in references}==set(audit['rows']), 'Expected24 unique archived references')
    for name,row in by.items():
        original=allow[name]
        D.need(all(row[k]==original[k] for k in ('shoot','role','session','captureDynamicRange','tiff_sha256','fjlrg_sha256'))
               and row['raf_sha256']==original['inputs']['linear']['source_raf_sha256'],
               'Source/current role or fingerprints differ: '+name)
        W.anchor_record(row['cam_mul'],row['auto_rgbg'])
    profile_rows=[]
    for ref in references:
        name=ref['name'];row=by[name];src,tar=ref['source'],ref['provia']
        D.need(ref['shoot']==row['shoot'] and ref['source_raf_sha256_at_probe']==row['raf_sha256']
               and ref['frozen_input_sha256']==row['fjlrg_sha256'], 'Archived source identity differs')
        D.need(not ref['provia_neutral_settings_mismatches'] and tar['WBFineTune']==[0,0],
               'Archive not neutral Provia')
        D.need(src['WhiteBalance']==tar['WhiteBalance']==row['source_wb_mode']
               and src['ColorTemperature']==tar['ColorTemperature']==row['source_kelvin']
               and src['WBFineTune']==row['source_wb_fine_tune_raw']
               and raw_rgb(src)==row['cam_mul'][:3], 'Source/target WB settings or levels differ')
        profile_rows.append({'name':name,'role':row['role'],'session':row['session'],
                             'mode':src['WhiteBalance'],'kelvin':src['ColorTemperature'],
                             'target_shift':tar['WBFineTune'],'source_shift':src['WBFineTune'],
                             'target_rgb':raw_rgb(tar),'source_rgb':raw_rgb(src)})
    return rows,profile_rows


def prepare():
    # Metadata/checksums only; do not construct/select policy profiles here.
    parent=D.load_manifest(PARENT_SHA)
    rows,refs=load_sources(parent)
    code=[Path(__file__).resolve(),D.TRAIN/'test_audit_wb_alignment.py',
          D.TRAIN/'wb_alignment_metrics.py',D.TRAIN/'test_wb_alignment_metrics.py']
    files=code+[INV,QUAL,ARCHIVE,OLD_MANIFEST,OLD_AUDIT,D.MANIFEST]
    manifest={'scope':__doc__.strip(),'parent_manifest_sha256':PARENT_SHA,'policy_rules':POLICY,
              'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files},
              'source_names':sorted(r['name'] for r in rows),'reference_names':sorted(r['name'] for r in refs),
              'calibration_roles':'Only CURRENT467fit role may derive the exact Kelvin6600 profile. Old F24 fit label does not override current development role.',
              'coverage':'Proposed source-only runtime metadata policy scored against already archived24 neutral-target metadata records, all reported by current role/mode. No pixels or claim of decoder-applied or Fuji-rendered WB parity.',
              'LOSO':'For each eligible fit Kelvin6600 reference, exclude its ENTIRE current capture session from calibration, require>=3remaining fit scenes>=2sessions and exact constant neutral-target ratios; all failures/nulls reported. Development rows never calibrate.',
              'retrospective':'Target WB metadata and normalization observations were previously visible. This is a frozen reproducible audit, not a blind acceptance test. Profiles are calibration candidates, not an inverse learned from source alone.',
              'bounds':'No source decode, new input arm, canonical data/model/runtime replacement, fitting or protected K/monitor/finalHeldOut/T8/dr100Eval access. No target-dependent ratio selected per runtime image. Presets/nonzero unsupported families have no fallback; original stopped manual probe remains unchanged.'}
    D.write_once_json(MANIFEST,manifest)
    print('WB_ALIGNMENT_MANIFEST',D.sha(MANIFEST),flush=True)


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected
           and D.committed(MANIFEST),'Unpinned/uncommitted metadata assay')
    manifest=json.loads(MANIFEST.read_text())
    D.need(manifest['policy_rules']==POLICY and manifest['parent_manifest_sha256']==PARENT_SHA,'Rules differ')
    for rel,h in manifest['bindings'].items():
        p=D.ROOT/rel;D.need(D.sha(p)==h and D.committed(p),'Metadata/code binding changed: '+rel)
    parent=D.load_manifest(PARENT_SHA);rows,refs=load_sources(parent)
    D.need(manifest['source_names']==sorted(r['name'] for r in rows)
           and manifest['reference_names']==sorted(r['name'] for r in refs),'Audit coverage differs')
    manifest['_manifest_sha256']=expected
    return manifest,rows,refs


def exact_ratios(record):
    return {k:Fraction(v['num'],v['den']) for k,v in record.items()}


def matches(policy,target_rgb):
    if policy['ratios'] is None:return None
    return exact_ratios(policy['ratios'])==W.positive_rgb_ratios(target_rgb)


def audit(rows,refs):
    """Pure archived-metadata computation; no loaders, targets or checkpoint functions."""
    profiles=W.calibrated_profiles(refs,min_scenes=3,min_sessions=2)
    by={r['name']:r for r in rows};policies={};anchors={};counts=Counter();sessions=defaultdict(set)
    for row in rows:
        name=row['name'];anchors[name]=W.anchor_record(row['cam_mul'],row['auto_rgbg'])
        policies[name]=W.propose_policy(source_record(row),profiles)
        counts[(row['role'],policies[name]['basis'])]+=1
        sessions[(row['role'],row['session'])].add(row['source_wb_mode'])
    reference_checks=[]
    for ref in refs:
        policy=policies[ref['name']]
        source_ratios=W.positive_rgb_ratios(ref['source_rgb'])
        target_ratios=W.positive_rgb_ratios(ref['target_rgb'])
        prof=profiles['profiles'].get(f"{ref['mode']}/{ref['kelvin']}",{})
        calibration_member=(ref['role']=='fit' and prof.get('status')=='candidate')
        reference_checks.append({'name':ref['name'],'role':ref['role'],'session':ref['session'],
                                 'mode':ref['mode'],'kelvin':ref['kelvin'],'policy':policy,
                                 'calibration_member':calibration_member,
                                 'exact_target_metadata_match':matches(policy,ref['target_rgb']),
                                 'as_shot_metadata_already_matches':source_ratios==target_ratios})
    reference_counts=[]
    for role,mode,member in sorted({(r['role'],r['mode'],r['calibration_member']) for r in reference_checks}):
        group=[r for r in reference_checks if (r['role'],r['mode'],r['calibration_member'])==(role,mode,member)]
        reference_counts.append({'role':role,'mode':mode,'calibration_member':member,'references':len(group),
                                  'exact_matches':sum(r['exact_target_metadata_match'] is True for r in group),
                                  'mismatches':sum(r['exact_target_metadata_match'] is False for r in group),
                                  'unsupported':sum(r['exact_target_metadata_match'] is None for r in group)})
    policy_reference_coverage={basis:sum(r['policy']['basis']==basis for r in reference_checks)
                              for basis in ('source-auto-estimate','zero-shift-as-shot','calibrated-kelvin-profile-candidate','unsupported')}
    loso=[]
    for ref in refs:
        if ref['role']!='fit' or (ref['mode'],ref['kelvin'])!=(4080,6600):continue
        calibration=[r for r in refs if r['role']=='fit' and r['session']!=ref['session']]
        fold_profiles=W.calibrated_profiles(calibration,min_scenes=3,min_sessions=2)
        policy=W.propose_policy(source_record(by[ref['name']]),fold_profiles)
        loso.append({'name':ref['name'],'held_session':ref['session'],
                     'fit_reference_count':len(calibration),'profiles':fold_profiles,
                     'candidate_policy':policy,'exact_target_metadata_match':matches(policy,ref['target_rgb'])})
    non302=[{'name':row['name'],'role':row['role'],'session':row['session'],
             'mode':row['source_wb_mode'],'kelvin':row['source_kelvin'],
             'fine_tune':row['source_wb_fine_tune_raw'],'anchors':anchors[row['name']],
             'candidate_policy':policies[row['name']]} for row in rows if row['cam_mul'][1]!=302]
    return {'profiles':profiles,'source_count':len(rows),'source_anchors':anchors,'policies':policies,
            'per_role_policy_counts':{role:{basis:n for (r,basis),n in counts.items() if r==role} for role in ('fit','development')},
            'observations':{'cam_minimum_302_count':sum(min(r['cam_mul'][:3])==302 for r in rows),
                            'auto_green_302_count':sum(r['auto_rgbg'][1]==r['auto_rgbg'][3]==302 for r in rows),
                            'auto_minimum_not_302_names':[r['name'] for r in rows if min(r['auto_rgbg'][:3])!=302],
                            'cam_green_not_302':non302,
                            'multiple_WB_mode_sessions':[{'role':r,'session':s,'modes':sorted(modes)} for (r,s),modes in sorted(sessions.items()) if len(modes)>1],
                            'normalization_rule':'Observed convention only, not a universal camera rule; Auto estimate and cam_mul use their own green.'},
            'archived_reference_checks':reference_checks,'reference_counts_by_role_mode_membership':reference_counts,
            'policy_reference_coverage':policy_reference_coverage,'kelvin6600_fit_LOSO':loso,
            'reference_limits':'Fit profile calibration members are resubstitution, not validation. Auto metadata matches reproduce earlier evidence. LOSO and development metadata were previously visible, not independent acceptance. The real archive has no zero-fine-tune source references; zero-shift-as-shot has zero target-metadata coverage, including both G-not302 sources.',
            'qualification':'NONE: exact metadata alignment and fit-derived candidate profile only; no physical decode or rendered Fuji test.'}


def run(expected):
    manifest,rows,refs=load(expected)
    D.need(not REPORT.exists(),'Prior result exists')
    result=audit(rows,refs)
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'Selected model changed')
    report={'manifest_sha256':expected,'completed':True,**result,
            'retrospective':manifest['retrospective'],'boundaries':manifest['bounds'],
            'R11_sha256_unchanged':D.R11_SHA}
    D.write_once_json(REPORT,report)
    print(json.dumps({'source_count':result['source_count'],'per_role_policy_counts':result['per_role_policy_counts'],
                      'references':len(refs),'exact_matches':sum(r['exact_target_metadata_match'] is True for r in result['archived_reference_checks']),
                      'unsupported_references':sum(r['exact_target_metadata_match'] is None for r in result['archived_reference_checks']),
                      'fit_kelvin_LOSO_exact':sum(r['exact_target_metadata_match'] is True for r in result['kelvin6600_fit_LOSO']),
                      'qualification':'NONE'},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','run'));ap.add_argument('--manifest-sha')
    args=ap.parse_args()
    if args.command=='prepare':prepare()
    else:run(args.manifest_sha)


if __name__=='__main__':main()
