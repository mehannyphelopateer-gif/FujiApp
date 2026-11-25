"""Prepare/preflight the bounded fresh G vs neutral-WB study; never trains.

Preparation refuses until both full input receipts are complete/published.
The subsequent preflight reads only the571frozen FIT/development pairs.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import argparse
import platform
from pathlib import Path
import numpy as np
import torch
import neutral_wb_study_data as D


def prepare():
    import verify_neutral_wb_study as verifier
    parent,fm,em,fp,ep=D.qualified_metadata()
    rows=D.assemble_rows(parent,fm,em,fp,ep)
    fit=[s['name'] for s in rows if s['role']=='fit']
    code=['neutral_wb_study_data.py','prepare_neutral_wb_study.py','test_neutral_wb_study_data.py',
          'run_neutral_wb_study.py','eval_neutral_wb_study.py','test_run_neutral_wb_study.py',
          'verify_neutral_wb_study.py','test_verify_neutral_wb_study.py',
          'continue_neutral_wb_study.py','test_continue_neutral_wb_study.py',
          'matched_gamma_data.py','run_matched_gamma_training.py','eval_matched_gamma_training.py',
          'fjlrg_v2.py','fjnrgb_profile.py','model.py','provia_color_pilot.py','color_metrics.py','color_loss.py']
    paths=[D.TRAIN/n for n in code]
    optional=D.TRAIN/'test_eval_neutral_wb_study.py'
    if optional.exists():paths.append(optional)
    paths += [D.B.MANIFEST,D.F.MANIFEST,D.E.MANIFEST,D.FIT_REPORT,D.DEV_REPORT]
    # Preserve all existing audited metric/model helper dependencies transitively.
    paths += [D.ROOT/rel for rel in parent['bindings']]
    manifest={'scope':__doc__.strip(),'parent_manifest_sha256':D.PARENT_SHA,
              'input_qualifications':{'fit':{'manifest_sha256':D.FIT_SHA,'report_sha256':D.sha(D.FIT_REPORT)},
                                      'development':{'manifest_sha256':D.DEV_SHA,'report_sha256':D.sha(D.DEV_REPORT)}},
              'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in paths},
              'model_config':parent['model_config'],'training':parent['training'],
              'orders':{str(seed):D.make_orders(fit,seed) for seed in D.SEEDS},
              'initial_state_sha256':parent['initial_state_sha256'],
              'screen':D.SCREEN,'scenes':rows,
              'independent_verification':{'canary_names':verifier.canary_names(rows),'canary_metric_tolerances':verifier.TOLERANCES},
              'optimisation':'Fresh paired initial states/orders, linearL1mean, Adam1e-3 cosine20 to1e-5, batch1. No earlystop, best-epoch selection, extra guide, TV, AMP, augmentation, geometry or model changes.',
              'evaluation':{'endpoint':20,'all_six':'All three paired seeds run regardless of early results unless an integrity/nonfinite failure stops the job.',
                            'independence':'Internal previously-used development split, not independent acceptance. FIT metrics in-sample. No automatic model/app selection.',
                            'subgroups':'Development treated79=W5/P74 allDR400; identity37=G25DR200/G12DR100. Groups defined by source metadata, never actual pixel changes.',
                            'DR100':'12development-only, no FIT support; reported as extrapolation, not qualified.',
                            'geometry':'Existing geometry held fixed. WB/geometry interactions and absolute impact of measured eight-fit crop discrepancy are unmeasured.',
                            'guide':'Rec709 guide changes as part of WB input treatment; encoded-guide intervention is not repeated.'},
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                             'mps_available':torch.backends.mps.is_available(),'platform':platform.platform()}}
    D.validate_manifest(manifest,rows,parent)
    D.write_once_json(D.MANIFEST,manifest)
    print('NEUTRAL_WB_STUDY_MANIFEST',D.sha(D.MANIFEST),flush=True)


def preflight(expected):
    m=D.load_manifest(expected)
    pairs={arm:D.Pairs(m,arm,role=None) for arm in D.ARMS}
    identity=0;changed=0
    for i,s in enumerate(m['scenes']):
        xg,yg,cg=pairs['G'].get(s['name']);xn,yn,cn=pairs['neutral'].get(s['name'])
        D.need(torch.equal(yg,yn) and torch.equal(cg,cn) and xg.shape==xn.shape,'Targets/DR/geometry differ between arms')
        same=torch.equal(xg,xn)
        if s['neutral_kind']=='G':D.need(same,'Identity input differs');identity+=1
        elif not same:changed+=1
        if (i+1)%25==0:print('NEUTRAL_WB_PREFLIGHT',i+1,571,flush=True)
    D.need(identity==38 and len(m['scenes'])==571,'Identity/count drift')
    for seed in D.SEEDS:
        model=D.fresh_model(m['model_config'],seed)
        D.need(D.state_digest(model.state_dict())==m['initial_state_sha256'][str(seed)],'Fresh initial state does not reproduce')
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'R11 changed')
    D.write_once_json(D.PREFLIGHT,{'passed':True,'manifest_sha256':expected,'pairs':571,'identity_pairs':identity,
                                 'treated_pairs_with_changed_bytes':changed,'R11_sha256':D.R11_SHA,
                                 'scope':'Both strict input contracts/hashes, identical unchanged targets/conditions/dimensions, finite arrays, seeded model initialization only. No optimiser update or source-photo inference.'})
    print('NEUTRAL_WB_PREFLIGHT_PASS',571,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','preflight'));ap.add_argument('--manifest-sha');a=ap.parse_args()
    prepare() if a.command=='prepare' else preflight(a.manifest_sha)
