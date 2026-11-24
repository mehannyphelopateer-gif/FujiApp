"""Frozen 583-target WB metadata comparison; no rendered pixels or fitting.

Only IFD/MakerNote values are interpreted. Complete TIFF bytes are streamed
opaquely for SHA-256 identity, never decoded. The 467 fit / 116 development
headers are a retrospective metadata diagnostic, not acceptance. The source
policy/profile was frozen before these reads; no new profile is derived.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matched_gamma_data as D
import audit_wb_alignment as A
import wb_alignment_metrics as W

ALIGN_SHA='6f6dac1c4bdd619ae1daacac36581866b1b71b059fc970f23145b89ffbed8ffc'
PRIMITIVE_SHA='2896f5af1fab83b182f5d6243328930e3c047554acf8b144f2e107bed4197736'
MANIFEST=D.DIAG/'target-wb-header-manifest.json'
REPORT=D.DIAG/'target-wb-header-report.json'
PROGRESS=D.CKPT/'target-wb-header-progress.json'
CODE=Path(__file__).resolve()


def target_path(row):
    shoot=row['shoot']
    D.need(isinstance(shoot,str) and shoot.startswith('Shoot ') and shoot[6:].isdigit(),'Unsafe shoot path')
    D.need(row['role'] in ('fit','development'),'Protected/nonparent role refused')
    return D.ROOT/'calibration-input'/shoot/'xraw-phase3.tiff'


def comparison(row,policy,tags):
    levels=[tags[k] for k in ('WBLevelR','WBLevelG','WBLevelB')]
    W.positive_rgb_ratios(levels)
    settings=(tags['WhiteBalance']==row['source_wb_mode'] and tags['ColorTemperature']==row['source_kelvin']
              and tags['WBFineTune']==[0,0])
    return {'neutral_settings_match':settings,'exact_policy_ratio_match':A.matches(policy,levels),
            'source_as_shot_ratio_match':W.positive_rgb_ratios(row['cam_mul'][:3])==W.positive_rgb_ratios(levels)}


def prepare():
    import qualify_manual_kelvin_wb_fullnative_v2 as V
    V.load(PRIMITIVE_SHA)
    primitive=json.loads(V.REPORT.read_text());verification=json.loads((D.DIAG/'manual-kelvin-wb-fullnative-v2-verification.json').read_text())
    D.need(primitive['verdict']=='PASSED' and verification['verified'] and verification['report_sha256']==D.sha(V.REPORT),
           'Physical primitive not verified')
    _,rows,_=A.load(ALIGN_SHA)
    alignment=json.loads(A.REPORT.read_text());D.need(alignment['manifest_sha256']==ALIGN_SHA,'Alignment binding differs')
    scenes=[]
    for row in rows:
        p=target_path(row)
        scenes.append({**row,'target_tiff_path':str(p.relative_to(D.ROOT)),
                       'frozen_source_only_policy':alignment['policies'][row['name']]})
    files=[CODE,D.TRAIN/'test_audit_target_wb_headers.py',D.TRAIN/'tiff_wb_header.py',D.TRAIN/'test_tiff_wb_header.py',
           Path(A.__file__).resolve(),Path(W.__file__).resolve(),A.MANIFEST,A.REPORT,A.INV,A.ARCHIVE,
           V.MANIFEST,V.REPORT,D.DIAG/'manual-kelvin-wb-fullnative-v2-verification.json']
    m={'scope':__doc__.strip(),'alignment_manifest_sha256':ALIGN_SHA,'primitive_manifest_sha256':PRIMITIVE_SHA,
       'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files},'scenes':scenes,
       'counts':dict(Counter(s['role'] for s in scenes)),
       'policy':'Exact existing wb-alignment-report policies only; no recalibration/inverse/interpolation or target-selected gain. Unsupported stays unsupported.',
       'reads':'Only583 named TIFFs. Stream entire files opaquely for frozen SHA; interpret bounded IFD/Exif/MakerNote headers only. No strip/tile decoding, PIL, target cache, RAF, FJLRG, inference or training.',
       'development':'116 development target WB HEADERS explicitly included only for retrospective metadata comparison. No development image pixels interpreted, profiles remain fixed fit-derived.',
       'protected':'K, monitor, finalHeldOut, T8 and dr100Eval never accessed. R11/app/canonical files unchanged.',
       'reporting':'Every scene, invalid/missing/hash mismatch retained; exact/mismatch/unsupported by role/mode/Kelvin/session/policy. A metadata match is not physical qualification or Fuji RGB accuracy.'}
    D.need(m['counts']=={'fit':467,'development':116} and len({s['name'] for s in scenes})==583,'Parent coverage differs')
    D.write_once_json(MANIFEST,m);print('TARGET_WB_HEADER_MANIFEST',D.sha(MANIFEST),flush=True)


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected and D.committed(MANIFEST),'Unpinned/uncommitted header audit')
    m=json.loads(MANIFEST.read_text())
    for rel,sha in m['bindings'].items():
        p=D.ROOT/rel;D.need(D.sha(p)==sha and D.committed(p),'Bound artifact changed: '+rel)
    _,rows,_=A.load(ALIGN_SHA);source={r['name']:r for r in rows};alignment=json.loads(A.REPORT.read_text())
    D.need(len(m['scenes'])==583 and {s['name'] for s in m['scenes']}==set(source),'Coverage changed')
    for row in m['scenes']:
        D.need(all(row[k]==v for k,v in source[row['name']].items()),'Source metadata changed')
        D.need(D.ROOT/row['target_tiff_path']==target_path(row),'TIFF path changed')
        D.need(row['frozen_source_only_policy']==alignment['policies'][row['name']],'Policy changed')
    return m


def summarize(rows):
    groups=[]
    for role,mode,kelvin,basis in sorted({(r['role'],r['source_wb_mode'],str(r['source_kelvin']),r['policy_basis']) for r in rows}):
        g=[r for r in rows if (r['role'],r['source_wb_mode'],str(r['source_kelvin']),r['policy_basis'])==(role,mode,kelvin,basis)]
        groups.append({'role':role,'mode':mode,'kelvin':kelvin,'basis':basis,'n':len(g),
                       'exact':sum(r.get('exact_policy_ratio_match') is True for r in g),
                       'mismatch':sum(r.get('exact_policy_ratio_match') is False for r in g),
                       'unsupported':sum(r.get('status')=='checked' and r.get('exact_policy_ratio_match') is None for r in g),
                       'invalid':sum(r.get('status')!='checked' for r in g),
                       'settings_mismatch':sum(r.get('neutral_settings_match') is False for r in g),
                       'exact_with_correct_settings':sum(r.get('neutral_settings_match') is True and r.get('exact_policy_ratio_match') is True for r in g)})
    return groups


def run(expected):
    import tiff_wb_header as H
    m=load(expected);D.need(not REPORT.exists() and not PROGRESS.exists(),'Prior audit evidence exists')
    archived={r['name']:r['provia'] for r in json.loads(A.ARCHIVE.read_text())['scenes']}
    D.need(len(archived)==24 and set(archived)<={s['name'] for s in m['scenes']},'Archived reader cross-check coverage differs')
    rows=[]
    for index,s in enumerate(m['scenes']):
        row={k:s[k] for k in ('name','shoot','role','session','source_wb_mode','source_kelvin')}
        row['policy_basis']=s['frozen_source_only_policy']['basis']
        p=target_path(s)
        try:
            before=p.stat();h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
                D.need(h.hexdigest()==s['tiff_sha256'],'TIFF SHA256 differs from frozen parent')
                f.seek(0)
                header=H.read_wb_header(f)
            after=p.stat()
            D.need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'TIFF changed during audit')
            tags={k:header[k] for k in ('WhiteBalance','ColorTemperature','WBFineTune','WBLevelR','WBLevelG','WBLevelB')}
            row.update(status='checked',tiff_sha256=h.hexdigest(),target_settings=tags,header_bytes_read=header['bytes_read'],
                       header_reads=header['reads'],
                       **comparison(s,s['frozen_source_only_policy'],tags))
            row['archived_reader_crosscheck']=None
            if s['name'] in archived:
                row['archived_reader_crosscheck']=all(tags[k]==archived[s['name']].get(k) for k in tags)
                D.need(row['archived_reader_crosscheck'],'Reader/provenance differs from archived WB reference')
        except Exception as exc:row.update(status='invalid',error=str(exc))
        rows.append(row)
        PROGRESS.write_text(json.dumps({'manifest_sha256':expected,'completed':False,'rows':rows},indent=2)+'\n')
        print('TARGET_WB_HEADER',index+1,len(m['scenes']),s['name'],row['status'],row.get('exact_policy_ratio_match'),flush=True)
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'Selected R11 changed')
    result={'completed':True,'manifest_sha256':expected,'source_policy_unchanged':True,'qualification':'METADATA_ONLY; no decoder policy or rendered Fuji accuracy qualification',
            'invalid':sum(r['status']!='checked' for r in rows),'settings_mismatch':sum(r.get('neutral_settings_match') is False for r in rows),
            'archived_reader_matches':sum(r.get('archived_reader_crosscheck') is True for r in rows),
            'groups':summarize(rows),'rows':rows,'R11_sha256':D.R11_SHA,'scope':m['scope'],'development':m['development'],'protected':m['protected']}
    D.write_once_json(REPORT,result);print('HEADER_AUDIT_COMPLETE',result['invalid'],result['settings_mismatch'],flush=True)
    if result['invalid'] or result['settings_mismatch']:raise SystemExit(1)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'));ap.add_argument('--manifest-sha')
    a=ap.parse_args();prepare() if a.command=='prepare' else run(a.manifest_sha)


if __name__=='__main__':main()
