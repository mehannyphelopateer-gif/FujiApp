"""Fit-input-only encoder sensitivity diagnostic; no targets, fitting or app changes.

Measures gain, capture-DR and sampling-translation responses, not accuracy or
causality. Every model is frozen; only encoder/grid prediction is executed.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')

import numpy as np
import torch
import torch.nn.functional as F

import matched_gamma_data as D
import run_matched_gamma_training as T
import run_gamma_encoded_guide as U
import diagnose_gamma_guide as Q
from fjlrg import load_fjlrg_linear
import fjlrg_v2 as V
from model import resize_for_network
import encoder_information_metrics as M

PARENT_SHA='ff0fad18b4fd6ca9e8ca36e728e526b95bf975b6dd07f88da844db50089e62b2'
COHORT_SHA='d829a30a6d400530b7bd8771a0a0a0bada8e2f6d994c8f6edeb1fcf2db76783c'
GUIDED_SHA='6f9ad80528c85d7d8a475095f51519001ac4a8782096739f791b9664a2369fe4'
MANIFEST=D.DIAG/'encoder-information-manifest.json'
REPORT=D.DIAG/'encoder-information-report.json'
CASES=D.DIAG/'encoder-information-cases'
FEATURES=D.CKPT/'encoder-information-grids'
SYNTHETIC=D.DIAG/'encoder-information-synthetic.json'
GAINS=(.5,2**-.5,2**.5,2.)
SHIFTS=((0,-2),(0,-1),(0,1),(0,2),(-2,0),(-1,0),(1,0),(2,0))
SETTINGS={'gains':list(GAINS),'shifts':[list(s) for s in SHIFTS],
          'box_kernel':7,'thumbnail_size':256,'thumbnail_trim':2,'native_trim':5,
          'conditions':[.5,1.],'device':'cpu','threads':2}


class Inputs:
    """Only the frozen fit cohort's input bytes. Never call D.Pairs or load a target."""
    def __init__(self,parent,cohort):
        by={r['name']:r for r in parent['scenes']}
        D.need(len({r['name'] for r in cohort})==len(cohort)>0,'Empty/duplicate cohort')
        self.by={}
        for row in cohort:
            D.need(row['name'] in by and row['role']=='fit' and by[row['name']]['role']=='fit',
                   'Protected/development input refused')
            s=by[row['name']]
            D.need(all(s[k]==row[k] for k in ('shoot','session','captureDynamicRange')),'Cohort identity mismatch')
            self.by[row['name']]=s

    def get(self,name,arm):
        D.need(name in self.by and arm in ('legacy','linear'),'Input not in frozen fit cohort')
        row=self.by[name];spec=row['inputs'][arm];p=D.ROOT/spec['path']
        D.need(D.sha(p)==spec['sha256'],'Input hash mismatch')
        if arm=='legacy':
            rgb,w,h=load_fjlrg_linear(p)
        else:
            D.need(D.sha(V.sidecar_path(p))==spec['sidecar_sha256'],'Input sidecar mismatch')
            frame=V.load_fjlrg_v2(p,expected_contract=V.CONTRACT_ASSHOT,
                         expected_source_sha256=spec['source_raf_sha256'],expected_wasm_sha256=spec['wasm_sha256'])
            D.need(frame.provenance['capture_dynamic_range']==row['captureDynamicRange'],'DR provenance mismatch')
            rgb,w,h=frame.to_unit_float32(),frame.width,frame.height
        D.need([w,h]==row['fjlrg_wh'] and rgb.shape==(h,w,3) and rgb.dtype==np.float32
               and np.isfinite(rgb).all() and 0<=rgb.min()<=rgb.max()<=1,'Input shape/dtype/range violation')
        x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0)
        c=torch.tensor([np.log2(row['captureDynamicRange']/100)/2],dtype=torch.float32)
        return x,c


def encode(model,low,condition):
    captured=[]
    handle=model.grid.encoder[0].register_forward_hook(lambda _m,_i,out:captured.append(out.detach().clone()))
    try:grid=model.grid(low,condition)
    finally:handle.remove()
    D.need(len(captured)==1 and bool(torch.isfinite(grid).all())
           and bool(torch.isfinite(captured[0]).all()),'Invalid encoder activation/grid')
    return grid.detach().clone(),captured[0]


def response(base,changed):
    return {'grid':M.vector_response(base[0],changed[0]),
            'block1':M.vector_response(base[1],changed[1])}


def entries(parent,guided):
    result=[]
    for seed in D.SEEDS:
        for arm in ('legacy','linear'):
            result.append({'id':f'fresh-{seed}-{arm}','family':'fresh','seed':seed,'input_arm':arm,
                           'state_sha256':parent['initial_state_sha256'][str(seed)]})
        for arm in ('legacy','linear'):
            p,ck,h=T.validate_endpoint(parent,seed,arm)
            result.append({'id':f'matched-{seed}-{arm}','family':'matched','seed':seed,'input_arm':arm,
                           'checkpoint':str(p.relative_to(D.ROOT)),'sha256':h,'state_sha256':ck['meta']['state_sha256']})
        p,ck,h=U.validate_endpoint(guided,parent,seed)
        result.append({'id':f'encoded-{seed}-linear','family':'encoded','seed':seed,'input_arm':'linear',
                       'checkpoint':str(p.relative_to(D.ROOT)),'sha256':h,
                       'state_sha256':D.state_digest(ck['model'])})
    return result


def build(parent,entry):
    model=D.initialise(parent,entry['seed'],'cpu')
    if entry['family']!='fresh':
        p=D.ROOT/entry['checkpoint'];D.need(D.sha(p)==entry['sha256'],'Checkpoint changed')
        ck=torch.load(p,map_location='cpu',weights_only=True)
        model.load_state_dict(ck['model'],strict=True)
    D.need(D.state_digest(model.state_dict())==entry['state_sha256'],'Model state identity differs')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def prepare():
    cohort,parent=Q.load(COHORT_SHA)
    guided,_=U.load(GUIDED_SHA)
    models=entries(parent,guided)
    files=[Path(__file__).resolve(),D.TRAIN/'test_diagnose_encoder_information.py',
           D.TRAIN/'encoder_information_metrics.py',D.TRAIN/'test_encoder_information_metrics.py',
           Q.MANIFEST,U.MANIFEST,D.MANIFEST]
    manifest={'scope':__doc__.strip(),'parent_manifest_sha256':PARENT_SHA,
              'cohort_manifest_sha256':COHORT_SHA,'guided_manifest_sha256':GUIDED_SHA,
              'settings':SETTINGS,'cohort':cohort['cohort'],'models':models,
              'normalization':'Median unique-pair baseline grid L2 across all13 scenes per model/domain, actual capture condition. The SAME unfiltered scale normalizes both filters; separate fixed-condition0.5 and filtered scales are context. Zero/near-zero normalizers remain null.',
              'statistics':'Stored-value luma stats, not sensor clips. Compare256 thumbnails against CPU PyTorch area resize (adaptive averaging) with2px trim, not a claim to exact continuous-area resampling. Full-frame context uses native trim5. No Fuji target or error metrics.',
              'gain_interpretation':'Unclamped thumbnail RGB amplitude scaling, DR fixed; report over-one values. Darkening k<1 is primary, k>1 is context and may leave training range. Linear G darkening .5 is a physical signal-gain counterfactual. Legacy encoded amplitude scaling is NOT a physically matched exposure intervention. No direct invariance ranking as proof of exposure identifiability.',
              'translation_interpretation':'Equal-sized reflect-padded native translations up to2px before resize; border/content registration differs. Fixed box7 prefilter before translations is OOD to these trained models, used ONLY for a sensitivity contrast, never accuracy or an automatic fix.',
              'inference':'CPU only, encoder/grid head only, no slicing/refinement/target reads, no optimisation or new model selection; checkpoints remain immutable.',
              'activations':'block1 is AFTER conv, GroupNorm and ReLU, not the GroupNorm output alone. The15 model/domain entries comprise12 distinct fresh/endpoint state identities; fresh weights are reused across2 input domains.',
              'interpretation':'Descriptive finite-input sensitivity assay. Nonzero responses rule out exact invariance at tested inputs; small responses do not prove lost information. GroupNorm with conv bias/nonzero condition is not generally scale-invariant. Absolute-luma slicing and affine RGB still carry intensity. No threshold labels a model exposure-blind or defective, and no null result proves optimisation the sole cause.',
              'boundaries':'Only13 immutable fit inputs in legacy and linear domains; all116 development images plus K/monitor/finalHeldOut/T8/dr100Eval unread; no app/CN change or follow-up training in this scope.',
              'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files}}
    D.write_once_json(MANIFEST,manifest)
    print('ENCODER_MANIFEST',D.sha(MANIFEST),flush=True)


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected
           and D.committed(MANIFEST),'Unpinned/uncommitted diagnostic')
    manifest=json.loads(MANIFEST.read_text())
    D.need(manifest['settings']==SETTINGS and manifest['parent_manifest_sha256']==PARENT_SHA
           and manifest['cohort_manifest_sha256']==COHORT_SHA and manifest['guided_manifest_sha256']==GUIDED_SHA,
           'Diagnostic contract changed')
    for rel,h in manifest['bindings'].items():
        p=D.ROOT/rel;D.need(D.sha(p)==h and D.committed(p),'Bound source/evidence changed: '+rel)
    cohort,parent=Q.load(COHORT_SHA);guided,_=U.load(GUIDED_SHA)
    D.need(manifest['cohort']==cohort['cohort'] and manifest['models']==entries(parent,guided),'Cohort/models differ')
    manifest['_manifest_sha256']=expected
    return manifest,parent


def synthetic(parent):
    generator=torch.Generator().manual_seed(20261123)
    textures={'noise':torch.rand(1,3,256,256,generator=generator),
              'smooth':torch.linspace(0,1,256).reshape(1,1,1,256).expand(1,3,256,256)}
    result=[]
    with torch.inference_mode():
        for seed in D.SEEDS:
            model=D.initialise(parent,seed,'cpu').eval()
            for p in model.parameters():p.requires_grad_(False)
            for kind,texture in textures.items():
                for level in (.05,.5):
                    low=texture*level
                    for cond in (0.,.5,1.):
                        c=torch.tensor([cond]);base=encode(model,low,c)
                        result.append({'seed':seed,'texture':kind,'level':level,'condition':cond,
                                       'half_signal_response':response(base,encode(model,low*.5,c)),
                                       'DR_response':response(encode(model,low,torch.tensor([.5])),encode(model,low,torch.tensor([1.])))})
    return result


def perturbations(x):
    versions={}
    for label,source in [('unfiltered',x),('box7',M.fixed_box_prefilter(x,7))]:
        baseline=resize_for_network(source,256)
        versions[label]={'base':baseline,'gains':{str(k):baseline*k for k in GAINS},
                         'shifts':{f'{dy},{dx}':resize_for_network(M.translated_same_shape(source,dy,dx),256)
                                   for dy,dx in SHIFTS}}
    return versions


def measure(model,versions,c):
    result={};baseline_grids={}
    for filter_name,parts in versions.items():
        base=encode(model,parts['base'],c)
        c05=encode(model,parts['base'],torch.tensor([.5]))
        c10=encode(model,parts['base'],torch.tensor([1.]))
        baseline_grids[filter_name]=base[0].numpy()
        if filter_name=='unfiltered':baseline_grids['fixed_condition05']=c05[0].numpy()
        item={'DR05_to_10':response(c05,c10),'gains':{},'shifts':{}}
        for key,low in parts['gains'].items():
            item['gains'][key]=response(base,encode(model,low,c))
        for key,low in parts['shifts'].items():
            item['shifts'][key]=response(base,encode(model,low,c))
        result[filter_name]=item
    return result,baseline_grids


def stat_comparison(versions,x):
    area=F.interpolate(x,size=(256,256),mode='area')
    result={'full_input':M.image_summary(x,trim=5),'area_reference':M.image_summary(area,trim=2)}
    for name,parts in versions.items():
        result[name]={'base':M.image_summary(parts['base'],trim=2),
                      'gains':{k:M.image_summary(v,trim=2) for k,v in parts['gains'].items()},
                      'shifts':{k:M.image_summary(v,trim=2) for k,v in parts['shifts'].items()}}
    return result


def _mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summary(cases,normalizers,models):
    result={}
    for entry in models:
        key=entry['id'];scale=normalizers[key]['unfiltered']
        group={}
        for filt in ('unfiltered','box7'):
            branch={}
            for kind,variants in [('gains',[str(k) for k in GAINS]),('shifts',[f'{dy},{dx}' for dy,dx in SHIFTS]),('DR05_to_10',[None])]:
                branch[kind]={}
                for variant in variants:
                    rows=[c['models'][key][filt][kind] if variant is None else c['models'][key][filt][kind][variant] for c in cases]
                    branch[kind][str(variant)]={'scene_count':len(rows),'mean_grid_l2':_mean([r['grid']['l2'] for r in rows]),
                                              'mean_grid_normalized_l2':_mean([r['grid']['l2']/scale if scale is not None and scale>1e-12 else None for r in rows]),
                                              'mean_block1_cosine':_mean([r['block1']['cosine'] for r in rows]),
                                              'mean_block1_relative_l2':_mean([r['block1']['relative_l2'] for r in rows])}
            group[filt]=branch
        # Same unfiltered grid scale for both filters. Per-scene/shift ratios
        # remain descriptive; tiny denominators are null, never a pass.
        ratios=[]
        for c in cases:
            for dy,dx in SHIFTS:
                k=f'{dy},{dx}';old=c['models'][key]['unfiltered']['shifts'][k]['grid']['l2']
                new=c['models'][key]['box7']['shifts'][k]['grid']['l2']
                if old>1e-12:ratios.append(new/old)
        group['box7_over_unfiltered_phase_l2_ratio']={'defined':len(ratios),'median':float(np.median(ratios)) if ratios else None}
        result[key]=group
    return result


def run(expected):
    manifest,parent=load(expected)
    T.check_r11()
    D.need(not REPORT.exists() and not CASES.exists() and not FEATURES.exists() and not SYNTHETIC.exists(),'Prior output exists')
    D.write_once_json(SYNTHETIC,{'manifest_sha256':expected,'scope':'Fresh synthetic inputs only; no checkpoint/media/target reads in this synthetic stage','cases':synthetic(parent)})
    models={entry['id']:build(parent,entry) for entry in manifest['models']}
    inputs=Inputs(parent,manifest['cohort']);cases=[]
    grids={entry['id']:{kind:[] for kind in ('unfiltered','box7','fixed_condition05')} for entry in manifest['models']}
    CASES.mkdir();FEATURES.mkdir()
    with torch.inference_mode():
        for i,row in enumerate(manifest['cohort'],1):
            case={**row,'inputs':{},'models':{}}
            for arm in ('legacy','linear'):
                x,c=inputs.get(row['name'],arm);versions=perturbations(x)
                case['inputs'][arm]=stat_comparison(versions,x)
                for entry in manifest['models']:
                    if entry['input_arm']!=arm:continue
                    key=entry['id'];stats,features=measure(models[key],versions,c)
                    case['models'][key]=stats
                    for kind,grid in features.items():grids[key][kind].append(grid)
                del x,versions
            D.write_once_json(CASES/(row['name']+'.json'),case);cases.append(case)
            print(f'ENCODER_CASE {i}/{len(manifest["cohort"])} {row["name"]} complete (no targets)',flush=True)
    normalizers={};feature_files=[]
    for entry in manifest['models']:
        key=entry['id'];normalizers[key]={k:M.scene_distance_scale(v) for k,v in grids[key].items()}
        p=FEATURES/(key+'.npz')
        np.savez_compressed(p,**{k:np.stack(v) for k,v in grids[key].items()})
        feature_files.append({'model':key,'path':str(p.relative_to(D.ROOT)),'sha256':D.sha(p)})
        D.need(D.state_digest(models[key].state_dict())==entry['state_sha256']
               and all(p.grad is None for p in models[key].parameters()),'Model changed or accumulated gradients')
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'Selected R11 changed')
    # Revalidate every source/checkpoint binding without reading additional media.
    load(expected)
    report={'manifest_sha256':expected,'completed':True,'cohort_count':len(cases),'model_domain_count':len(models),
            'normalizers':normalizers,'summary':summary(cases,normalizers,manifest['models']),
            'synthetic_file_sha256':D.sha(SYNTHETIC),'feature_files':feature_files,
            'case_files':[{'name':c['name'],'path':str((CASES/(c['name']+'.json')).relative_to(D.ROOT)),
                           'sha256':D.sha(CASES/(c['name']+'.json'))} for c in cases],
            'interpretation':manifest['interpretation'],'boundaries':manifest['boundaries'],
            'R11_sha256_unchanged':D.R11_SHA}
    D.write_once_json(REPORT,report)
    print('ENCODER_DIAGNOSTIC_COMPLETE no targets, fitting or selection',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','run'));ap.add_argument('--manifest-sha')
    args=ap.parse_args();torch.set_num_threads(2)
    if args.command=='prepare':prepare()
    else:run(args.manifest_sha)


if __name__=='__main__':main()
