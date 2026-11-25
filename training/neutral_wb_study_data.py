"""Strict data contract for a FUTURE fresh G/neutral WB comparison.

No training/evaluation entrypoint. Real manifests require BOTH completed
input qualifications. Target arrays are first opened by the separate
preflight after that study manifest has been committed.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import json
import hashlib
import re
from pathlib import Path
import numpy as np
import torch
import matched_gamma_data as B
import prepare_neutral_wb_fit_export_v2 as F
import prepare_neutral_wb_development_export as E
import fjlrg_v2 as V
import fjnrgb_profile as P

ROOT,TRAIN,DIAG,CKPT,RUNS=B.ROOT,B.TRAIN,B.DIAG,B.CKPT,B.RUNS
R11_PATH,R11_SHA=B.R11_PATH,B.R11_SHA
need,sha,committed,write_once_json=B.need,B.sha,B.committed,B.write_once_json
state_digest,fresh_model,initialise,forward=B.state_digest,B.fresh_model,B.initialise,B.forward
device,make_orders=B.device,B.make_orders
SEEDS=B.SEEDS
ARMS=('G','neutral')
FIT_SHA='bea38e7e6c0ce4b7768e1286d1aba744638bac660693d532075093691c268a60'
DEV_SHA='530911d5d4dbe08810e21588798c149fe1593bdf2fcbd46d2d5ea6e4da01600a'
PARENT_SHA='ff0fad18b4fd6ca9e8ca36e728e526b95bf975b6dd07f88da844db50089e62b2'
MANIFEST=DIAG/'neutral-wb-study-manifest.json'
PREFLIGHT=DIAG/'neutral-wb-study-preflight.json'
FIT_REPORT=DIAG/'neutral-wb-fit-export-verification-v2.json'
DEV_REPORT=DIAG/'neutral-wb-development-export-verification.json'
SCREEN={'pooled_dev_mean_delta_max':-0.25,'each_seed_dev_mean_delta_strict_max':0.0,
        'pooled_dev_mean_scene_p95_delta_max':0.0,'treated79_pooled_mean_delta_strict_max':0.0,
        'identity37_each_seed_mean_delta_max':0.10}
HEX=re.compile(r'[0-9a-f]{64}')


def read(path):return json.loads(Path(path).read_text())


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def receipt(path,manifest_sha,expected_count,verifier):
    need(path.exists(),'Input qualification still pending: '+str(path))
    r=read(path)
    need(r.get('verified') is True and r.get('canary_only') is False and r.get('manifest_sha256')==manifest_sha,
         'Incomplete/wrong input qualification')
    need(len(r['rows'])==len({s['name'] for s in r['rows']})==expected_count and all(s['verified'] is True for s in r['rows']),
         'Input receipt coverage failure')
    need(r.get('completed_fit_rows',r.get('completed_development_rows'))==expected_count and r.get('R11_sha256')==R11_SHA,
         'Input receipt count/fingerprint differs')
    need(r['verification_script_sha256']==sha(verifier) and committed(path) and committed(verifier),
         'Unpublished/changed input qualification code/evidence')
    return r


def qualified_metadata():
    # Check completed receipts BEFORE any source artifact or target access.
    fp=receipt(FIT_REPORT,FIT_SHA,455,TRAIN/'verify_neutral_wb_fit_export_v2.py')
    ep=receipt(DEV_REPORT,DEV_SHA,116,TRAIN/'verify_neutral_wb_development_export.py')
    parent=B.load_manifest(PARENT_SHA);fm=F.load(FIT_SHA);em=E.load(DEV_SHA)
    return parent,fm,em,fp,ep


def expected_neutral_path(source,output):
    if source['kind']=='G':return source['G']['file']
    if source['reuse']:
        arm='W' if source['kind']=='W' else 'P'
        return source['reuse']['source_record']['arms'][arm]['file']
    suffix='fjlrg' if source['kind']=='W' else 'fjnrgb'
    return f"{output}/{source['shoot']}/N-neutral.{suffix}"


def assemble_rows(parent,fm,em,fp,ep):
    eligible={s['name']:s for m in (fm,em) for s in m['scenes']}
    need(len(eligible)==571 and len(fm['scenes'])==455 and len(em['scenes'])==116,'Wrong eligible source pool')
    need({s['name'] for s in fp['rows']}=={s['name'] for s in fm['scenes']}
         and {s['name'] for s in ep['rows']}=={s['name'] for s in em['scenes']},'Per-role qualification coverage differs')
    receipts={s['name']:s for p in (fp,ep) for s in p['rows']}
    need(set(receipts)==set(eligible),'Receipt/source coverage differs')
    rows=[]
    for old in parent['scenes']:
        if old['name'] not in eligible:continue
        s=eligible[old['name']];proof=receipts[s['name']]
        need(proof.get('verified') is True and proof.get('kind')==s['kind'],'Qualification kind/status drift')
        need(s['role']==old['role'] and s['role'] in ('fit','development') and s['shoot']==old['shoot'] and s['session']==old['session'],
             'Source split/identity drift')
        need(s['capture_dynamic_range']==old['captureDynamicRange'] and s['raf_sha256']==old['inputs']['linear']['source_raf_sha256'],
             'Source DR/hash drift')
        need(s['G']['file']==old['inputs']['linear']['path'] and s['G']['sha256']==old['inputs']['linear']['sha256']
             and s['G']['sidecar_sha256']==old['inputs']['linear']['sidecar_sha256'],'Control source qualification differs')
        output=fm['output_dir'] if s['role']=='fit' else em['output_dir']
        expected=expected_neutral_path(s,output)
        g={**old['inputs']['linear'],'kind':'G','contract':V.CONTRACT_ASSHOT}
        if s['kind']=='G':
            need(proof['identity_reference_only'] is True,'Identity policy not qualified')
            n=dict(g)
        else:
            need(proof['control_matches_G'] is True and proof['neutral_repeat_exact'] is True and proof['kind']==s['kind'], 'Input gates not passed')
            need(proof['neutral_file']==expected,'Neutral input outside exact qualified scene allowlist')
            n={'path':expected,'sha256':proof['neutral_sha256'],'sidecar_sha256':proof['neutral_sidecar_sha256'],
               'source_raf_sha256':s['raf_sha256'],'wasm_sha256':old['inputs']['linear']['wasm_sha256'],
               'kind':s['kind'],'contract':s['contract']}
        rows.append({**old,'neutral_kind':s['kind'],'policy_subgroup':'identity' if s['kind']=='G' else 'treated',
                     'source_wb_mode':s['source_wb_mode'],'source_kelvin':s['source_kelvin'],
                     'inputs':{'G':g,'neutral':n}})
    need(len(rows)==571 and len({s['name'] for s in rows})==571,'Parent eligible coverage differs')
    return rows


def validate_manifest(m,expected_rows,parent):
    need(m['parent_manifest_sha256']==PARENT_SHA,'Wrong parent study')
    need(m['scenes']==expected_rows,'Source roles/targets/qualified paths changed')
    need(m['model_config']==parent['model_config'] and m['training']==parent['training'], 'Architecture/budget changed')
    need(m['screen']==SCREEN and m['initial_state_sha256']==parent['initial_state_sha256'],'Screen/initial state drift')
    fit=[s['name'] for s in m['scenes'] if s['role']=='fit'];dev=[s for s in m['scenes'] if s['role']=='development']
    need(len(fit)==455 and len(dev)==116,'Wrong study split')
    need(not ({s['session'] for s in m['scenes'] if s['role']=='fit'} & {s['session'] for s in dev}),'Session leakage')
    need(sum(s['neutral_kind']=='G' for s in dev)==37 and sum(s['neutral_kind']!='G' for s in dev)==79,'Subgroup drift')
    for seed in SEEDS:need(m['orders'][str(seed)]==make_orders(fit,seed),'Orders do not reproduce')


def load_manifest(expected_sha):
    need(isinstance(expected_sha,str) and HEX.fullmatch(expected_sha) and MANIFEST.exists() and sha(MANIFEST)==expected_sha,
         'Unpinned/missing study manifest')
    need(committed(MANIFEST),'Study manifest must be committed before any pair access')
    m=read(MANIFEST)
    core={'training/neutral_wb_study_data.py','training/prepare_neutral_wb_study.py',
          'training/run_neutral_wb_study.py','training/eval_neutral_wb_study.py','training/model.py',
          'training/provia_color_pilot.py','training/color_metrics.py','training/verify_neutral_wb_study.py'}
    need(core<=set(m['bindings']),'Missing core study bindings')
    for rel,h in m['bindings'].items():
        path=ROOT/rel;need(sha(path)==h,'Changed study binding: '+rel)
        if not rel.startswith('training/checkpoints/'):need(committed(path),'Unpublished study dependency: '+rel)
    parent,fm,em,fp,ep=qualified_metadata()
    need(m['input_qualifications']=={'fit':{'manifest_sha256':FIT_SHA,'report_sha256':sha(FIT_REPORT)},
                                    'development':{'manifest_sha256':DEV_SHA,'report_sha256':sha(DEV_REPORT)}},
         'Study qualification receipt binding differs')
    validate_manifest(m,assemble_rows(parent,fm,em,fp,ep),parent)
    verification=m.get('independent_verification',{})
    names=verification.get('canary_names',[])
    need(isinstance(names,list) and 6<=len(names)<=7 and len(set(names))==len(names)
         and set(names)<={r['name'] for r in m['scenes']},'Invalid independent canary allowlist')
    need(verification.get('canary_metric_tolerances')=={'mean_dE00':1e-5,'p95_dE00':1e-4,'linear_l1_unclamped':1e-7},
         'Independent numerical canary limits changed')
    need(sha(R11_PATH)==R11_SHA,'Selected R11 changed')
    need(m['environment']['torch']==torch.__version__ and m['environment']['numpy']==np.__version__
         and m['environment']['mps_available']==torch.backends.mps.is_available(),'Training environment changed')
    m['_manifest_sha256']=expected_sha
    m['_rows_sha256']=rows_digest(m['scenes'])
    return m


class Pairs:
    """Default is FIT-only. All-role access is explicit for frozen preflight/evaluation."""
    def __init__(self,m,arm,role='fit'):
        need(arm in ARMS and role in (None,'fit','development'),'Unknown arm/protected role')
        need(isinstance(m.get('_manifest_sha256'),str) and HEX.fullmatch(m['_manifest_sha256']),'Unvalidated study object')
        need(m.get('_rows_sha256')==rows_digest(m['scenes']),'Study rows mutated after validation')
        need(all(s['role'] in ('fit','development') for s in m['scenes']),'Protected row in study')
        self.arm=arm
        self.by={s['name']:json.loads(json.dumps(s)) for s in m['scenes'] if role is None or s['role']==role}
        need(len(self.by)==sum(role is None or s['role']==role for s in m['scenes']),'Duplicate source names')

    def get(self,name):
        need(name in self.by,'Scene outside explicit role/name allowlist: '+str(name))
        row=self.by[name];spec=row['inputs'][self.arm]
        need(spec['kind'] in ('G','W','P') and spec['contract']=={'G':V.CONTRACT_ASSHOT,'W':V.CONTRACT_AUTOWB,'P':P.CONTRACT}[spec['kind']],
             'Unknown kind/contract; fallback forbidden')
        need(self.arm!='G' or spec['kind']=='G','G control contract changed')
        need(spec['kind']==('G' if self.arm=='G' else row['neutral_kind']),'Policy discriminator changed')
        need(spec['source_raf_sha256']==row['inputs']['G']['source_raf_sha256']
             and spec['wasm_sha256']==row['inputs']['G']['wasm_sha256'],'Paired source/WASM provenance differs')
        need(spec['path'].startswith('training/checkpoints/') and '..' not in Path(spec['path']).parts,'Input path escape')
        need(row['target_path']==f"calibration-input/provia-color-target-cache/{row['shoot']}.npy",'Target path escape')
        path,target=ROOT/spec['path'],ROOT/row['target_path']
        need(sha(path)==spec['sha256'] and sha(Path(str(path)+'.json'))==spec['sidecar_sha256'],'Input/sidecar hash changed')
        need(sha(target)==row['npy_sha256'],'Target hash changed')
        kwargs={'expected_source_sha256':spec['source_raf_sha256'],'expected_wasm_sha256':spec['wasm_sha256']}
        frame=P.load(path,**kwargs) if spec['kind']=='P' else V.load_fjlrg_v2(path,expected_contract=spec['contract'],**kwargs)
        need(frame.provenance['capture_dynamic_range']==row['captureDynamicRange'],'DR provenance mismatch')
        x=frame.to_unit_float32();y=np.load(target,allow_pickle=False)
        need([frame.width,frame.height]==row['fjlrg_wh'] and x.shape==y.shape==(frame.height,frame.width,3),'Pair shape differs')
        need(x.dtype==y.dtype==np.float32 and np.isfinite(x).all() and np.isfinite(y).all(),'Bad pair dtype/finiteness')
        need(0<=x.min()<=x.max()<=1,'Input outside unit range')
        # Preserve audited target epsilon values and the original objective exactly.
        tensor=lambda a:torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).unsqueeze(0)
        cond=torch.tensor([np.log2(row['captureDynamicRange']/100)/2],dtype=torch.float32)
        return tensor(x),tensor(y),cond
