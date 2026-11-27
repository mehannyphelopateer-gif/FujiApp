"""Phase A: nine-source hashes and fresh legacy tags, never TRAIN admission.

No pixel decode, target/cache access, protected-data read, source copy or
fitting. The original acceptance exclusion and stored EXIF stay unchanged.
Prepare/commit an exact allowlist before reading any candidate RAF bytes.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from check_dr100eval_metadata import exclusion_reasons

ROOT = Path(__file__).resolve().parents[1]
BASE = 'calibration-input/phase4-dr100eval-attempt1-excluded'
NAMES = tuple(f'DSCF{i}.RAF' for i in range(1424,1441,2))
STORED = ROOT/BASE/'exif.json'
STORED_SHA = '0898a25bea42a32ed75a9d598d89a174a57557ca664d65326303ececc228437a'
PARENT = ROOT/'training/diagnostics/matched-gamma-training-manifest-v2.json'
PARENT_SHA = 'ff0fad18b4fd6ca9e8ca36e728e526b95bf975b6dd07f88da844db50089e62b2'
SCOPE = ROOT/'training/diagnostics/dr100-training-candidates-phaseA-scope.json'
OUTPUT = ROOT/'training/diagnostics/dr100-training-candidates-phaseA.json'
FRESH = ROOT/'training/diagnostics/dr100-training-candidates-phaseA-fresh-exif.json'
READER = ROOT/'scripts/dump-raf-exif.mjs'
PLAN = ROOT/'training/coordination/dr100-support-data-proposal.md'
CODE_PATHS = [Path(__file__).resolve(), READER, PLAN,
              ROOT/'training/check_dr100eval_metadata.py', ROOT/'training/test_dr100_training_candidates.py',
              ROOT/'package-lock.json']


def need(condition, message):
    if not condition:raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20), b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def dump(path,obj):
    with path.open('x') as f:f.write(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def committed(path):
    b = subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)
    need(hashlib.sha256(b).hexdigest() == sha(path),'Unpublished/changed dependency: '+str(path))


def candidate_paths(names):
    need(tuple(names) == NAMES,'Exact nine-source allowlist required; no other/protected source allowed')
    return [f'{BASE}/{name}' for name in names]


def stored_rows():
    need(sha(STORED) == STORED_SHA,'Stored historical EXIF changed')
    rows = read(STORED);by = {s['name']:s for s in rows}
    need(len(rows) == len(by) == 18 and set(by) == {f'DSCF{i}.RAF' for i in range(1424,1442)},'Unexpected stored metadata pool')
    for name,path in zip(NAMES,candidate_paths(NAMES)):need(by[name]['path'] == path,'Historical source path differs')
    return by


def prepare():
    stored_rows();need(sha(PARENT) == PARENT_SHA,'Parent pool changed')
    for p in CODE_PATHS+[PARENT]:committed(p)
    dump(SCOPE,{'format':'dr100-candidate-phaseA-v1','names':list(NAMES),'paths':candidate_paths(NAMES),
                'stored_metadata_sha256':STORED_SHA,'parent_sha256':PARENT_SHA,
                'bindings':{str(p.relative_to(ROOT)):sha(p) for p in CODE_PATHS+[PARENT]},
                'approval':'hashes_and_fresh_legacy_metadata_only',
                'training_admission':False,'exports':False,'protected_access':False,
                'holds':['sourceWBfineTune/Kelvin/policy qualification pending',
                         'complete source/session/protected-boundary clearance pending',
                         'no new Provia TIFF references or current-domain qualified inputs']})
    print('DR100_CANDIDATE_PHASE_A_SCOPE',sha(SCOPE),flush=True)


def load_scope(expected):
    need(isinstance(expected,str) and len(expected)==64 and sha(SCOPE)==expected,'Pinned phase-A scope required')
    committed(SCOPE);scope=read(SCOPE)
    need(scope['format']=='dr100-candidate-phaseA-v1' and scope['names']==list(NAMES)
         and scope['paths']==candidate_paths(NAMES),'Source scope expanded/changed')
    need(scope['stored_metadata_sha256']==STORED_SHA and scope['parent_sha256']==PARENT_SHA
         and scope['training_admission'] is False and scope['exports'] is False
         and scope['protected_access'] is False,'Qualification authority changed')
    need(set(scope['bindings'])=={str(p.relative_to(ROOT)) for p in CODE_PATHS+[PARENT]},'Binding coverage differs')
    for rel,h in scope['bindings'].items():
        p=ROOT/rel;need(sha(p)==h,'Changed phase-A binding: '+rel);committed(p)
    stored_rows()
    return scope


def check_fresh(expected, fresh):
    need(len(fresh)==len(NAMES) and len({r['name'] for r in fresh})==len(NAMES),'Fresh extraction coverage differs')
    by={s['name']:s for s in fresh}
    need(set(by)==set(NAMES),'Unexpected fresh source')
    for name in NAMES:
        need(by[name]==expected[name],'Fresh/stored metadata differs: '+name)
    return by


def session_groups(rows):
    times=sorted((datetime.strptime(s['dateTimeOriginal'],'%Y:%m:%d %H:%M:%S'),s['name']) for s in rows)
    groups=[]
    for time,name in times:
        if not groups or (time-groups[-1][-1][0]).total_seconds()>3*3600:groups.append([])
        groups[-1].append((time,name))
    return [{'names':[n for _,n in g],'first':g[0][0].isoformat(),'last':g[-1][0].isoformat()} for g in groups]


def run(expected):
    need(not OUTPUT.exists() and not FRESH.exists(),'Existing receipt/extract; overwrite refused')
    scope=load_scope(expected);old=stored_rows();parent=read(PARENT)
    need(len(parent['scenes'])==583 and all(s['role'] in ('fit','development') for s in parent['scenes']),'Unapproved parent population')
    old_hashes={s['inputs']['linear']['source_raf_sha256'] for s in parent['scenes']}
    source_hashes={};before={}
    for name,path in zip(NAMES,scope['paths']):
        p=ROOT/path;st=p.stat();before[name]=(st.st_size,st.st_mtime_ns)
        with p.open('rb') as f:need(f.read(16)==b'FUJIFILMCCD-RAW ','RAF header mismatch: '+name)
        source_hashes[name]=sha(p)
    need(len(set(source_hashes.values()))==9 and not set(source_hashes.values())&old_hashes,'Source duplicate/current-pool alias')
    local=ROOT/'training/coordination/.local';local.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dr100-candidate-phaseA-',dir=local) as td:
        paths=Path(td)/'paths.json';out=Path(td)/'fresh.json'
        dump(paths,scope['paths'])
        # Only this fresh private path is passed to the old reader. Never --dir on the old checker.
        subprocess.run(['node',str(READER),'--list',str(paths),'--out',str(out)],cwd=ROOT,check=True,capture_output=True,text=True)
        fresh_bytes=out.read_bytes();fresh=check_fresh(old,json.loads(fresh_bytes))
    records=[]
    for name,path in zip(NAMES,scope['paths']):
        p=ROOT/path;st=p.stat()
        need((st.st_size,st.st_mtime_ns)==before[name] and sha(p)==source_hashes[name],'Source changed during extraction')
        e=fresh[name]
        need(e['model']=='X100VI' and e['fuji'].get('AutoDynamicRange')=='100 (0x64)'
             and 'DevelopmentDynamicRange' not in e['fuji'],'Candidate capture metadata differs')
        reasons=exclusion_reasons(e)
        need(reasons,'Historical acceptance exclusion unexpectedly absent')
        records.append({'name':name,'path':path,'raf_sha256':source_hashes[name],
                        'fresh_legacy':e,'historical_acceptance_exclusion':reasons,
                        'captureDR_origin':'Auto resolved to100','training_admission':False})
    need(sha(STORED)==STORED_SHA,'Historical EXIF modified')
    with FRESH.open('xb') as f:f.write(fresh_bytes)
    need(sha(FRESH)==hashlib.sha256(fresh_bytes).hexdigest(),'Fresh extract write mismatch')
    dump(OUTPUT,{'completed_phaseA':True,'scope_sha256':expected,'script_sha256':sha(Path(__file__)),
                 'fresh_metadata_path':str(FRESH.relative_to(ROOT)),
                 'fresh_metadata_sha256':sha(FRESH),'rows':records,
                 'current583_raf_hash_disjoint':True,'sessions_within_nine':session_groups(list(fresh.values())),
                 'training_admission':False,'exports':False,'protected_data_access':False,
                 'holds':scope['holds'],
                 'limits':['Source hash establishes current identity; historical dump did not pin original RAF hashes.',
                           'Old acceptance exclusions unchanged. Stored metadata equality is not WB-policy or reference qualification.',
                           'Hashes read whole files; legacy tag reader reads a bounded header region without image decoding.',
                           'No targets, caches, recipes, trained models, protected sources or new tier assignment.']})
    print('DR100_CANDIDATE_PHASE_A_COMPLETE',len(records),'metadata consistent; admission still held',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'));ap.add_argument('--scope-sha');a=ap.parse_args()
    prepare() if a.command=='prepare' else run(a.scope_sha)
