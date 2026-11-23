"""One-variable, fresh-seeded encoded-guide assay against frozen linear-G controls.

prepare -> commit -> preflight -> commit -> all --manifest-sha SHA.
No automatic model selection/app update. Only parent-allowlisted fit sources
train; development sources are scored once at the fixed epoch20 endpoint.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

import matched_gamma_data as D
import run_matched_gamma_training as T
import eval_matched_gamma_training as E
from gamma_encoded_guide import (GammaEncodedGuideHybrid, GUIDE_CONTRACT, TABLE_SHA256,
                                encoded_guide, qualified_table)

PARENT_SHA = "ff0fad18b4fd6ca9e8ca36e728e526b95bf975b6dd07f88da844db50089e62b2"
MANIFEST = D.DIAG / "gamma-encoded-guide-training-manifest.json"
PREFLIGHT = D.DIAG / "gamma-encoded-guide-preflight.json"
REPORT = D.RUNS / "gamma-encoded-guide-report.json"
CODE = Path(__file__).resolve()
SCREEN = {"mean_de00_delta_max": -.25, "every_seed_mean_improves": True,
          "mean_scene_p95_delta_max": 0.}
ARM = "encoded-guide-linear-rgb"


def paths(seed):
    D.need(type(seed) is int and seed in D.SEEDS, "Unknown seed refused")
    base = D.CKPT / f"gamma-encoded-guide-seed{seed}"
    return base, base/'state.pt', base/'endpoint.pt', base/'endpoint.json'


def eval_path(seed):
    paths(seed)
    return D.RUNS / f"gamma-encoded-guide-seed{seed}-eval.json"


def construct(parent, seed, device):
    T.seed_everything(seed)
    cfg = dict(parent['model_config'])
    cfg.pop('luma_mode')
    model = GammaEncodedGuideHybrid(**cfg)
    D.need(D.state_digest(model.state_dict()) == parent['initial_state_sha256'][str(seed)],
           "Fresh initial weights differ from paired control")
    return model.to(device)


def identity(manifest, parent, seed):
    ident = T.identity_for(parent, seed, ARM, parent['orders'][str(seed)])
    ident.update(manifest_sha256=manifest['_manifest_sha256'], parent_manifest_sha256=PARENT_SHA,
                 trainer_sha256=D.sha(CODE), guide_contract=GUIDE_CONTRACT, table_sha256=TABLE_SHA256)
    return ident


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected and D.committed(MANIFEST),
           "Unpinned, changed or uncommitted manifest")
    m = json.loads(MANIFEST.read_text())
    D.need(m['parent_manifest_sha256']==PARENT_SHA and m['guide_contract']==GUIDE_CONTRACT
           and m['table_sha256']==TABLE_SHA256 and m['screen']==SCREEN, "Study contract changed")
    for rel,h in m['bindings'].items():
        p = D.ROOT / rel
        D.need(D.sha(p)==h, "Bound artifact changed: "+rel)
        if rel.startswith('training/') and p.suffix in ('.py','.json') and 'checkpoints/' not in rel:
            D.need(D.committed(p), "Bound code/evidence must be committed: "+rel)
    parent = D.load_manifest(PARENT_SHA)
    D.need(m['training']==parent['training'] and m['model_config']==parent['model_config'],
           "Training recipe differs from paired control")
    for seed in D.SEEDS:
        _,_,h=T.validate_endpoint(parent,seed,'linear')
        D.need(h==m['controls'][str(seed)]['checkpoint_sha256'], "Control endpoint changed")
        ep=E.eval_path(seed,'linear')
        control=json.loads(ep.read_text())
        E.validate_eval(parent,seed,'linear',control,h)
        D.need(D.sha(ep)==m['controls'][str(seed)]['eval_sha256'],"Control evaluation changed")
    m['_manifest_sha256']=expected
    return m,parent


def prepare():
    parent=D.load_manifest(PARENT_SHA)
    native=D.CKPT/'gamma-guide-export-probe/native-report.json'
    browser=D.CKPT/'gamma-guide-export-probe/browser-report.json'
    n,b=json.loads(native.read_text()),json.loads(browser.read_text())
    D.need(n['passed'] is True and b['passed'] is True,"Export feasibility must pass before freeze")
    D.need(n['native_tolerances']=={'max_abs':1e-4,'mean_abs':1e-5}
           and b['tolerances']=={'max_abs':5e-4,'mean_abs':5e-5},"Export tolerance drift")
    D.need({p['provider'] for p in b['providers']}=={'wasm','webgpu'},"Missing browser backend")
    names={v['name'] for v in n['cases']}
    D.need(len(names)==len(n['cases'])==4 and len(n['models'])==1,'Export vector coverage')
    for rel,h in n['bindings'].items():
        D.need(D.sha(D.ROOT/rel)==h,'Changed export implementation: '+rel)
    D.need(n['guide_contract']==GUIDE_CONTRACT and n['table_sha256']==TABLE_SHA256
           and D.sha(native.parent/n['models'][0]['model'])==n['models'][0]['model_sha256'],
           'Export graph/guide drift')
    def check_cases(cases,max_abs,mean_abs):
        D.need(len(cases)==4 and {r['name'] for r in cases}==names,'Incomplete parity cases')
        for row in cases:
            D.need(row['passed'] is True,'Parity case failed')
            for key,limit in [('max_abs',max_abs),('mean_abs',mean_abs)]:
                value=row[key]
                D.need(type(value) in (int,float) and np.isfinite(value) and 0<=value<=limit,
                       'Invalid/out-of-limit parity value')
    check_cases(n['models'][0]['native_cases'],1e-4,1e-5)
    for provider in b['providers']:
        D.need(not provider.get('error') and not provider['consoleErrors'],"Browser error")
        D.need(len(provider['models'])==1,'Browser model coverage')
        check_cases(provider['models'][0]['cases'],5e-4,5e-5)
        if provider['provider']=='webgpu':
            D.need(provider['adapterInfo'].get('isFallbackAdapter') is False,"Fallback GPU refused")
    evidence=[]
    for source,name in ((native,'gamma-guide-export-native.json'),(browser,'gamma-guide-export-browser.json')):
        dest=D.DIAG/name
        D.write_once_json(dest,json.loads(source.read_text()))
        evidence.append(dest)
    files=[CODE,D.TRAIN/'gamma_encoded_guide.py',D.TRAIN/'test_gamma_encoded_guide.py',
           D.TRAIN/'test_run_gamma_encoded_guide.py',D.TRAIN/'export_gamma_guide_probe.py',
           D.TRAIN/'eval_provia_gamma_length.py',D.TRAIN/'export_round11_onnx.py',
           D.TRAIN/'eval_matched_gamma_training.py',D.TRAIN/'run_matched_gamma_training.py',
           D.DIAG/'gamma-guide-diagnostic-report.json',D.DIAG/'gamma-guide-diagnostic-verification.json',
           D.ROOT/'scripts/verify-round11-onnx-web.mjs',native,browser]+evidence
    files += [D.CKPT/'gamma-guide-export-probe'/n['models'][0]['model']]
    controls={}
    for seed in D.SEEDS:
        p,_,h=T.validate_endpoint(parent,seed,'linear')
        e=E.eval_path(seed,'linear')
        E.validate_eval(parent,seed,'linear',json.loads(e.read_text()),h)
        controls[str(seed)]={'checkpoint':str(p.relative_to(D.ROOT)),'checkpoint_sha256':h,
                             'eval':str(e.relative_to(D.ROOT)),'eval_sha256':D.sha(e)}
        files += [p,e,T.endpoint_record_path(seed,'linear')]
    m={'scope':'Single guide-coordinate intervention; local development study, not acceptance/selection/integration',
       'parent_manifest_sha256':PARENT_SHA,'guide_contract':GUIDE_CONTRACT,'table_sha256':TABLE_SHA256,
       'mapping':'round(clamp(linear RGB,0,1)*65535) indexes qualified default LibRaw uint16 table /65535, then rec709 luma; only grid depth uses this',
       'unchanged':'Linear G in encoder/affine RGB/refiner; same fresh initial tensors, conditions, 467 fit sources, 116 session-disjoint development, 20 epoch orders and optimisation as matched study',
       'training':parent['training'],'model_config':parent['model_config'],'controls':controls,
       'screen':SCREEN,
       'evaluation':'Every epoch20 endpoint scored on same583, unchanged CPU dev_metrics; report all per-scene differences, fit/development, DR and sessions. Primary screen all116 dev incl12 unsupported DR100; exclude-DR100 context reported separately. All3 seeds complete regardless of interim metrics.',
       'limitations':'Historical paired controls reused, not retrained; MPS deterministic reproducibility not guaranteed. No independent acceptance, no DR100 qualification, no automatic selection. Passing only supports this guide choice at tested budget. K/monitor/finalHeldOut/T8/dr100Eval locked; R11/app unchanged.',
       'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files}}
    D.write_once_json(MANIFEST,m)
    print('GUIDE_MANIFEST',D.sha(MANIFEST),flush=True)


def preflight(m,parent):
    torch.set_num_threads(2)
    proof={'manifest_sha256':m['_manifest_sha256'],'code_sha256':D.sha(CODE)}
    # Uses the exact reusable run/checkpoint/RNG primitives of the new run.
    proof['cpu_exact_resume']=T.fixture_exact_resume()
    D.need(proof['cpu_exact_resume']['equal'] and proof['cpu_exact_resume']['stale_refused'],
           "Resume fixture failed")
    for seed in D.SEEDS:
        construct(parent,seed,'cpu')
    pairs=D.Pairs(parent,'linear')
    for i,row in enumerate(parent['scenes']):
        pairs.get(row['name'])
        if (i+1)%50==0:print('GUIDE_PREFLIGHT',i+1,'/583',flush=True)
    # Full working-resolution finite smoke uses fit-only input and a throwaway
    # network/optimizer; no smoke weights enter the real fresh-seeded runs.
    fit=[r for r in parent['scenes'] if r['role']=='fit']
    dev=D.device()
    codes=torch.arange(65536).float()/65535.
    codes=codes.view(1,1,256,256).expand(1,3,256,256)
    cpu_guide=encoded_guide(codes,qualified_table())
    device_guide=encoded_guide(codes.to(dev),qualified_table().to(dev)).cpu()
    proof['all_codes_device_guide_max_abs']=float((cpu_guide-device_guide).abs().max())
    D.need(proof['all_codes_device_guide_max_abs']<=1e-7,'Device lookup/code restoration differs')
    model=construct(parent,42,dev)
    opt=T.make_optimizer(model,parent['training']['lr'])
    x,y,c=pairs.get(fit[0]['name'])
    proof['mps_smoke_loss']=T.update(model,opt,x,y,c,lambda mdl,a,b:D.forward(mdl,parent,a,b,dev))
    proof.update(passed=True,pairs=583,fit_count=467,development_count=116,
                 device=str(dev),learned_params=sum(p.numel() for p in model.parameters()),
                 R11_sha256=D.sha(D.R11_PATH))
    D.need(proof['learned_params']==115719 and proof['R11_sha256']==D.R11_SHA,'Parameter/R11 drift')
    D.write_once_json(PREFLIGHT,proof)
    print('GUIDE_PREFLIGHT_PASS',flush=True)


def validate_endpoint(m,parent,seed):
    _,_,p,rec=paths(seed)
    record=json.loads(rec.read_text())
    D.need(D.sha(p)==record['sha256'],'Guide endpoint hash mismatch')
    ck=torch.load(p,map_location='cpu',weights_only=True)
    D.need(ck.get('format')=='gamma-encoded-guide-endpoint-v1'
           and ck['identity']==identity(m,parent,seed),'Wrong endpoint identity')
    pr=ck['progress']
    D.need(pr['epoch']==20 and pr['global_updates']==9340 and pr['next_index']==0
           and len(pr['history'])==20,'Incomplete endpoint')
    D.need(D.state_digest(ck['model'])==record['state_sha256'],'Guide weight digest mismatch')
    return p,ck,record['sha256']


def train(m,parent,seed):
    proof=json.loads(PREFLIGHT.read_text())
    D.need(D.committed(PREFLIGHT) and proof['passed'] is True and proof['pairs']==583
           and proof['manifest_sha256']==m['_manifest_sha256'] and proof['code_sha256']==D.sha(CODE),
           'Missing/uncommitted/mismatched preflight')
    base,state,ep,record=paths(seed)
    if ep.exists() or record.exists():
        validate_endpoint(m,parent,seed)
        return
    dev=D.device()
    model=construct(parent,seed,dev)
    T.seed_everything(seed)
    opt=T.make_optimizer(model,parent['training']['lr'])
    sched=T.make_scheduler(opt,20,1e-5)
    ident=identity(m,parent,seed)
    pairs=D.Pairs(parent,'linear',role='fit')
    def status(phase,pr=None,total=None,**extra):
        T.atomic_json(base/'status.json',{'phase':phase,'progress':pr,'total':total,
                     'identity':ident,'updated_unix':time.time(),**extra})
    try:
        with T.global_lock():
            result=T.run(model,opt,sched,parent['orders'][str(seed)],pairs.get,
                         lambda mdl,x,c:D.forward(mdl,parent,x,c,dev),state,ident,
                         tag=f'encoded-guide seed{seed}',status=status)
            D.need(result['complete'],'Incomplete training')
            sd=T._to_cpu(model.state_dict())
            T.atomic_torch_save({'format':'gamma-encoded-guide-endpoint-v1','identity':ident,
                                'model':sd,'progress':result['progress']},ep)
            D.write_once_json(record,{'sha256':D.sha(ep),'state_sha256':D.state_digest(sd)})
            T.check_r11()
            status('complete',result['progress'],9340)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise


def evaluate(m,parent,seed):
    path,ck,h=validate_endpoint(m,parent,seed)
    out=eval_path(seed)
    if out.exists():
        result=json.loads(out.read_text())
        D.need(result['manifest_sha256']==m['_manifest_sha256'] and result['checkpoint_sha256']==h
               and result['seed']==seed and set(result['scenes'])=={r['name'] for r in parent['scenes']},
               'Stale/incomplete evaluation')
        for row in parent['scenes']:
            stored=result['scenes'][row['name']]
            D.need(all(stored[k]==row[k] for k in ('role','shoot','session','captureDynamicRange')),
                   'Stored evaluation identity mismatch')
            for k in ('mean_dE00','p95_dE00','linear_l1_unclamped'):
                D.need(type(stored['metrics'][k]) in (int,float) and np.isfinite(stored['metrics'][k]),
                       'Missing/non-numeric stored metric')
        E._check_finite(result)
        return result
    model=construct(parent,seed,'cpu')
    model.load_state_dict(ck['model'],strict=True);model.eval()
    pairs=D.Pairs(parent,'linear')
    rows={}
    print('GUIDE_FUJI_SCORING_STARTED',seed,'fixed endpoint',flush=True)
    with torch.inference_mode():
        for i,row in enumerate(parent['scenes']):
            x,y,c=pairs.get(row['name'])
            pred=D.forward(model,parent,x,c,'cpu')
            D.need(bool(torch.isfinite(pred).all()),'Non-finite prediction')
            rows[row['name']]={k:row[k] for k in ('role','shoot','session','captureDynamicRange')}
            rows[row['name']]['metrics']=E.dev_metrics(pred,y)
            if (i+1)%50==0:print('GUIDE_SCORING',seed,i+1,'/583',flush=True)
    result={'manifest_sha256':m['_manifest_sha256'],'checkpoint_sha256':h,'seed':seed,
            'metric_function':'unchanged provia_color_pilot.dev_metrics','device':'cpu',
            'scenes':rows,'DR100':'development-only, outside fit support, NOT VALIDATED'}
    E._check_finite(result)
    D.write_once_json(out,result)
    T.check_r11()
    return result


def paired(a,b,names):
    # Candidate minus control; every scene kept, no post-result exclusions.
    rows=[]
    for name in names:
        am,bm=a['scenes'][name]['metrics'],b['scenes'][name]['metrics']
        rows.append({'name':name,'delta_mean_dE00':am['mean_dE00']-bm['mean_dE00'],
                     'delta_p95_dE00':am['p95_dE00']-bm['p95_dE00']})
    return {'n':len(names),'candidate':E.summarise([a['scenes'][n] for n in names]),
            'control':E.summarise([b['scenes'][n] for n in names]),'per_scene':rows,
            'regressions':sorted([r for r in rows if r['delta_mean_dE00']>0],key=lambda r:-r['delta_mean_dE00'])}


def report(m,parent):
    all_seeds={}
    for seed in D.SEEDS:
        candidate=evaluate(m,parent,seed)
        control=json.loads(E.eval_path(seed,'linear').read_text())
        groups={'fit':[r['name'] for r in parent['scenes'] if r['role']=='fit'],
                'development':[r['name'] for r in parent['scenes'] if r['role']=='development']}
        for role in ('fit','development'):
            for key in ('session','captureDynamicRange'):
                for value in sorted({r[key] for r in parent['scenes'] if r['role']==role}):
                    groups[f'{role}_{key}_{value}']=[r['name'] for r in parent['scenes'] if r['role']==role and r[key]==value]
        groups['development_excluding_DR100']=[r['name'] for r in parent['scenes'] if r['role']=='development' and r['captureDynamicRange']!=100]
        all_seeds[str(seed)]={g:paired(candidate,control,names) for g,names in groups.items()}
    dev=[all_seeds[str(s)]['development'] for s in D.SEEDS]
    mean=float(np.mean([r['candidate']['mean_dE00']-r['control']['mean_dE00'] for r in dev]))
    p95=float(np.mean([r['candidate']['p95_dE00']-r['control']['p95_dE00'] for r in dev]))
    criteria={'mean_gain_at_least_0.25':mean<=-.25,
              'every_seed_improves':all(r['candidate']['mean_dE00']<r['control']['mean_dE00'] for r in dev),
              'mean_scene_p95_no_worse':p95<=0}
    result={'manifest_sha256':m['_manifest_sha256'],'status':'complete','comparison':'encoded-guide minus frozen linear-G guide control',
            'per_seed':all_seeds,'mean_development_delta_dE00':mean,'mean_development_delta_p95':p95,
            'screen':{'criteria':criteria,'passed':all(criteria.values())},
            'limitations':m['limitations'],'selection':'NONE; no app change or independent acceptance'}
    E._check_finite(result)
    if REPORT.exists():
        D.need(json.loads(REPORT.read_text())==result,'Existing final report differs')
    else:D.write_once_json(REPORT,result)
    print('GUIDE_REPORT',json.dumps(result['screen']),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','preflight','all','report'))
    ap.add_argument('--manifest-sha')
    args=ap.parse_args()
    torch.set_num_threads(2)
    if args.command=='prepare':return prepare()
    m,parent=load(args.manifest_sha)
    if args.command=='preflight':return preflight(m,parent)
    if args.command=='report':return report(m,parent)
    for seed in D.SEEDS:
        # Recheck bindings between runs; integrity problems stop all work.
        m,parent=load(args.manifest_sha)
        train(m,parent,seed)
        evaluate(m,parent,seed)
    report(m,parent)


if __name__=='__main__':main()
