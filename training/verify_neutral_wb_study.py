"""Independent endpoint/report arithmetic checks plus fixed metadata-selected CPU canaries.

Input/target media are restricted to the already frozen571study sources.
Numerical canary tolerances verify reproducibility, not Fuji accuracy.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
import neutral_wb_study_data as D
import run_neutral_wb_study as T
import eval_neutral_wb_study as E
from provia_color_pilot import dev_metrics

TOLERANCES={'mean_dE00':1e-5,'p95_dE00':1e-4,'linear_l1_unclamped':1e-7}


def canary_names(scenes):
    selected=[]
    key=lambda s:hashlib.sha256(('neutral-wb-study-verification-v1\n'+s['name']).encode()).hexdigest()
    for role in ('fit','development'):
        for kind in ('G','W','P'):
            group=[s for s in scenes if s['role']==role and s['neutral_kind']==kind]
            D.need(group,'Canary metadata coverage short')
            selected.append(min(group,key=key)['name'])
    dr100=min((s for s in scenes if s['role']=='development' and s['captureDynamicRange']==100),key=key)['name']
    if dr100 not in selected:selected.append(dr100)
    return selected


def finite(v):return type(v) in (int,float) and math.isfinite(v)


def mean(vals):
    D.need(vals and all(finite(v) for v in vals),'Missing/nonfinite numeric values')
    return math.fsum(vals)/len(vals)


def independent_values(m,evals):
    dev=[s for s in m['scenes'] if s['role']=='development'];out={}
    D.need(len(dev)==116,'Wrong development count')
    for seed in D.SEEDS:
        g,n=evals[(seed,'G')]['scenes'],evals[(seed,'neutral')]['scenes']
        def delta(rows,metric):return mean([n[s['name']]['metrics'][metric]-g[s['name']]['metrics'][metric] for s in rows])
        identity=[s for s in dev if s['policy_subgroup']=='identity'];treated=[s for s in dev if s['policy_subgroup']=='treated']
        D.need(len(identity)==37 and len(treated)==79,'Wrong metadata groups')
        out[str(seed)]={'dev':delta(dev,'mean_dE00'),'p95':delta(dev,'p95_dE00'),
                        'identity':delta(identity,'mean_dE00'),'treated':delta(treated,'mean_dE00')}
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest-sha',required=True);a=ap.parse_args()
    m=D.load_manifest(a.manifest_sha);T.validate_split(m)
    report=json.loads(E.final_path().read_text());D.need(report['status']=='complete','Study not complete')
    evals={};fingerprints={};checks=[];names=canary_names(m['scenes'])
    D.need(m['independent_verification']=={'canary_names':names,'canary_metric_tolerances':TOLERANCES},'Canary scope changed')
    for seed in D.SEEDS:
        for arm in D.ARMS:
            endpoint,ck,h=T.validate_endpoint(m,seed,arm)
            evpath=E.eval_path(seed,arm);ev=json.loads(evpath.read_text());E.validate_eval(m,seed,arm,ev,h)
            D.need(ev['evaluator_sha256']==D.sha(Path(E.__file__)),'Wrong evaluator version')
            history=ck['meta']['history'];D.need(len(history)==20,'Endpoint history short')
            for epoch,row in enumerate(history,1):
                expected=1e-5+(1e-3-1e-5)*(1+math.cos(math.pi*(epoch-1)/20))/2
                D.need(row['epoch']==epoch and row['updates']==455 and finite(row['mean_linear_l1'])
                       and abs(row['lr']-expected)<1e-12,'History/budget/schedule mismatch')
            model=D.initialise(m,seed,torch.device('cpu'));model.load_state_dict(ck['model'],strict=True);model.eval()
            pairs=D.Pairs(m,arm,role=None)
            with torch.no_grad():
                for name in names:
                    x,y,c=pairs.get(name);pred=D.forward(model,m,x,c,torch.device('cpu'))
                    D.need(bool(torch.isfinite(pred).all()),'Nonfinite canary prediction')
                    metrics=dev_metrics(pred,y);gaps={}
                    for metric,limit in TOLERANCES.items():
                        gap=abs(metrics[metric]-ev['scenes'][name]['metrics'][metric])
                        D.need(finite(gap) and gap<=limit,'Independent canary metric mismatch: '+name+'/'+metric)
                        gaps[metric]=gap
                    checks.append({'seed':seed,'arm':arm,'name':name,'metric_abs_gaps':gaps})
            evals[(seed,arm)]=ev;fingerprints[f'{seed}/{arm}']={'checkpoint_sha256':h,'eval_sha256':D.sha(evpath)}
            print('INDEPENDENT_NEUTRAL_WB_ENDPOINT',seed,arm,len(names),'canaries verified',flush=True)
    values=independent_values(m,evals);criteria=report['screening']['criteria']
    scalar={
      'pooled_development_mean_delta_dE00_le':mean([v['dev'] for v in values.values()]),
      'pooled_development_mean_scene_p95_delta_dE00_le':mean([v['p95'] for v in values.values()]),
      'treated79_pooled_mean_delta_dE00_lt':mean([v['treated'] for v in values.values()])}
    D.need(len(criteria)==5,'Unexpected criterion count')
    for key,v in scalar.items():D.need(finite(criteria[key]['value']) and abs(v-criteria[key]['value'])<=1e-10,'Independent aggregate mismatch')
    for key,field in [('every_seed_development_mean_delta_dE00_lt','dev'),('every_seed_identity37_mean_delta_dE00_le_engineering_margin','identity')]:
        D.need(set(criteria[key]['values'])==set(values),'Seed coverage mismatch')
        for seed,v in values.items():D.need(abs(v[field]-criteria[key]['values'][seed])<=1e-10,'Per-seed aggregate mismatch')
    expected_specs={
      'pooled_development_mean_delta_dE00_le':('<=',-0.25),
      'every_seed_development_mean_delta_dE00_lt':('<',0.0),
      'pooled_development_mean_scene_p95_delta_dE00_le':('<=',0.0),
      'treated79_pooled_mean_delta_dE00_lt':('<',0.0),
      'every_seed_identity37_mean_delta_dE00_le_engineering_margin':('<=',0.10)}
    for key,(op,limit) in expected_specs.items():
        c=criteria[key];D.need(c['op']==op and type(c['threshold']) in (int,float) and c['threshold']==limit,'Screen threshold changed')
        vals=list(c['values'].values()) if 'values' in c else [c['value']]
        passed=all(v<limit if op=='<' else v<=limit for v in vals)
        D.need(c['pass'] is passed,'Screen pass flag incorrect')
    D.need(report['screening']['passed'] is all(c['pass'] for c in criteria.values()),'Overall screen flag incorrect')
    # Full report reproduction also checks all per-scene/group/regression tables.
    D.need(json.loads(json.dumps(E.build_report(m,evals),allow_nan=False))==report,'Final report does not reproduce')
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'R11 changed')
    D.write_once_json(D.DIAG/'neutral-wb-study-verification.json',{
      'verified':True,'manifest_sha256':a.manifest_sha,'report_sha256':D.sha(E.final_path()),
      'verification_script_sha256':D.sha(Path(__file__)),'screen_passed':report['screening']['passed'],
      'independent_paired_values':values,'endpoint_fingerprints':fingerprints,'canary_names':names,
      'canary_tolerances':TOLERANCES,'canary_checks':checks,'R11_sha256':D.R11_SHA,
      'limits':'Internal study only; six full eval reports structurally verified and reproduced, independent aggregate arithmetic and6-7metadata-selected CPU canaries per endpoint. Not full independent re-inference of571images, Fuji parity, selection or protected acceptance.'})
    print('NEUTRAL_WB_STUDY_VERIFIED','screen_passed',report['screening']['passed'],flush=True)


if __name__=='__main__':main()
