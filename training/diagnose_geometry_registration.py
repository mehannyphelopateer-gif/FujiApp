"""Eight-fit-source geometry diagnostic, never a crop fix or colour/model score.

All candidates are fixed before pixels: identity, source metadata zoom,
target metadata zoom and target 180 degrees. No transform is learned.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matched_gamma_data as D
import audit_raf_geometry as A
import fjlrg_v2 as V

GEOMETRY_SHA='e041dff199b19c29aa322c60e1e5015953e641109da6184a304c02e7ef9aad5a'
PARENT_SHA='ff0fad18b4fd6ca9e8ca36e728e526b95bf975b6dd07f88da844db50089e62b2'
MANIFEST=D.DIAG/'geometry-registration-manifest.json'
REPORT=D.DIAG/'geometry-registration-report.json'
PANELS=D.CKPT/'geometry-registration-panels'
CODE=Path(__file__).resolve()
SALT='fit-geometry-registration-v1\n'


def primary_tags(row):
    dirs=[d for d in row['geometry']['directories'] if d['index']==0]
    D.need(len(dirs)==1,'Primary geometry metadata missing')
    return dirs[0]['tags']


def linear_to_display(rgb):
    a=np.clip(np.asarray(rgb,dtype=np.float64),0,1)
    return np.where(a<=0.0031308,12.92*a,1.055*np.power(a,1/2.4)-0.055)


def measure_candidates(rgb,ref,tags,orientation):
    import geometry_registration_metrics as R
    window=None;srczoom=None
    if tags['raw_zoom_active']==1:
        window=R.normalized_zoom_window(tags['raw_image_cropped_size'],tags['raw_zoom_top_left'],tags['raw_zoom_size'],orientation)
        srczoom=R.zoom_resample(rgb,window)
    pairs={'identity':(rgb,ref),'target_180':(rgb,np.ascontiguousarray(ref[::-1,::-1]))}
    if window is not None:pairs.update(source_metadata_zoom=(srczoom,ref),target_metadata_zoom=(rgb,R.zoom_resample(ref,window)))
    measured={}
    for name,(a,b) in pairs.items():
        fa=R.registration_feature(a);fb=R.registration_feature(b)
        measured[name]={'applicable':True,'feature_ncc':R.feature_ncc(fa,fb),'tiles':R.tile_phase_shifts(fa,fb)}
    for name in ('source_metadata_zoom','target_metadata_zoom'):
        measured.setdefault(name,{'applicable':False,'reason':'RawZoomActive is not1','feature_ncc':None,'tiles':[]})
    return measured,window,srczoom


def select(rows):
    by={r['name']:r for r in rows};selected=[]
    # Existing source-only anchors; no target/image/error evidence selects them.
    for name in ('DSCF0464.RAF','DSCF1068.RAF','DSCF1054.RAF','DSCF1011.RAF'):
        D.need(name in by and by[name]['role']=='fit','Missing fixed fit anchor')
        selected.append(by[name])
    def add(candidates):
        candidates=[r for r in candidates if r['name'] not in {s['name'] for s in selected}]
        D.need(candidates,'Metadata coverage SHORT; do not substitute a different class')
        selected.append(min(candidates,key=lambda r:hashlib.sha256((SALT+r['name']).encode()).hexdigest()))
    add([r for r in rows if r['target_orientation']==1 and primary_tags(r)['raw_zoom_active']==0])
    add([r for r in rows if r['target_orientation']==8 and primary_tags(r)['raw_zoom_active']==1
         and primary_tags(r)['raw_zoom_size']==[2592,3888]])
    for size in ([3648,5472],[2592,3888]):
        add([r for r in rows if r['target_orientation']==1 and primary_tags(r)['raw_zoom_active']==1
             and primary_tags(r)['raw_zoom_size']==size])
    D.need(len(selected)==8 and len({r['name'] for r in selected})==8 and all(r['role']=='fit' for r in selected),'Wrong fit cohort')
    return selected


def prepare():
    A.load(GEOMETRY_SHA);metadata=json.loads(A.REPORT.read_text());proof_path=D.DIAG/'raf-geometry-fit-verification.json';proof=json.loads(proof_path.read_text())
    D.need(metadata['completed'] and proof['verified'] and proof['report_sha256']==D.sha(A.REPORT),'Geometry metadata not verified')
    parent=D.load_manifest(PARENT_SHA);by={r['name']:r for r in parent['scenes']}
    scenes=[]
    for r in select(metadata['rows']):
        p=by[r['name']];D.need(p['role']=='fit' and p['shoot']==r['shoot'],'Role/provenance differs')
        scenes.append({**r,'input_file':p['inputs']['linear']['path'],'input_sha256':p['inputs']['linear']['sha256'],
                       'input_sidecar_sha256':p['inputs']['linear']['sidecar_sha256'],
                       'target_cache_file':p['target_path'],'target_cache_sha256':p['npy_sha256'],
                       'wasm_sha256':p['inputs']['linear']['wasm_sha256']})
    files=[CODE,D.TRAIN/'test_geometry_registration_scope.py',D.TRAIN/'geometry_registration_metrics.py',
           D.TRAIN/'test_geometry_registration_metrics.py',A.MANIFEST,A.REPORT,proof_path,D.MANIFEST,
           D.TRAIN/'fjlrg_v2.py']
    m={'scope':__doc__.strip(),'geometry_manifest_sha256':GEOMETRY_SHA,'scenes':scenes,
       'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files},
       'selection':'Four fixed metadata anchors plus salted-name-hash choices covering unzoomed/zoomed, both zoom levels, landscape/portrait and sole TIFF orientation6. FIT only; no replacements.',
       'candidates':['identity','source_metadata_zoom','target_metadata_zoom','target_180'],
       'metrics':'Display-domain luma gradient NCC and fixed3x3 Hann phase correlation; every candidate/tile retained including null/outlier. No colour L1/DeltaE, model fitting, registration search or transform selection.',
       'interpretation':'Metadata zoom mapping assumes G fullFoV corresponds to RawCroppedSize, with TIFF orientation as a proxy for LibRaw display rotation. These are hypotheses, not assertions; size tags include borders. Inactive zoom candidates are explicitly not applicable, never zero-scored. No source/data/target/metric/model changes follow automatically.',
       'bounds':'Read only8selectedFIT G containers/sidecars and8hash-bound target-cache arrays. No RAW/TIFF reread, development/K/monitor/finalHeldOut/T8/dr100Eval, learned weights or model inference.'}
    D.write_once_json(MANIFEST,m);print('GEOMETRY_REGISTRATION_MANIFEST',D.sha(MANIFEST),flush=True)


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected and D.committed(MANIFEST),'Unpinned/uncommitted registration scope')
    m=json.loads(MANIFEST.read_text())
    for rel,h in m['bindings'].items():D.need(D.sha(D.ROOT/rel)==h and D.committed(D.ROOT/rel),'Bound code/metadata drift: '+rel)
    A.load(GEOMETRY_SHA);metadata=json.loads(A.REPORT.read_text())
    chosen=select(metadata['rows']);parent=D.load_manifest(PARENT_SHA);by={r['name']:r for r in parent['scenes']}
    geometry_by={r['name']:r for r in metadata['rows']}
    D.need([r['name'] for r in chosen]==[r['name'] for r in m['scenes']],'Cohort changed')
    for r in m['scenes']:
        p=by[r['name']]
        D.need(r['role']==p['role']=='fit' and r['input_file']==p['inputs']['linear']['path'] and r['target_cache_file']==p['target_path']
               and r['input_sha256']==p['inputs']['linear']['sha256'] and r['target_cache_sha256']==p['npy_sha256']
               and r['input_sidecar_sha256']==p['inputs']['linear']['sidecar_sha256'] and r['wasm_sha256']==p['inputs']['linear']['wasm_sha256']
               and r['raf_sha256']==p['inputs']['linear']['source_raf_sha256'],'Input/target allowlist drift')
        D.need(all(r[k]==geometry_by[r['name']][k] for k in ('geometry','target_orientation','G_native_size','G_working_size','shoot','session')),
               'Verified source geometry changed')
    return m


def run(expected):
    import geometry_registration_metrics as R
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=load(expected);D.need(not REPORT.exists() and not PANELS.exists(),'Prior registration evidence exists');PANELS.mkdir()
    rows=[]
    for s in m['scenes']:
        inp=D.ROOT/s['input_file'];target=D.ROOT/s['target_cache_file']
        D.need(D.sha(inp)==s['input_sha256'] and D.sha(Path(str(inp)+'.json'))==s['input_sidecar_sha256'] and D.sha(target)==s['target_cache_sha256'],'Input/target cache changed')
        g=V.load_fjlrg_v2(inp,expected_contract=V.CONTRACT_ASSHOT,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=s['wasm_sha256'])
        rgb=g.rgb_uint16.astype(np.float32)/np.float32(65535);ref=np.load(target,allow_pickle=False)
        D.need(ref.dtype==np.float32 and ref.shape==rgb.shape and np.isfinite(ref).all(),'Cache dtype/shape/nonfinite')
        tags=primary_tags(s);measured,window,srczoom=measure_candidates(rgb,ref,tags,s['target_orientation'])
        fig,axes=plt.subplots(1,3,figsize=(12,7))
        alternatives=srczoom if srczoom is not None else np.ascontiguousarray(ref[::-1,::-1])
        for axis,image,label in zip(axes,(rgb,ref,alternatives),('G input','Fuji Provia target','Source zoom candidate' if srczoom is not None else 'Target 180 candidate')):
            axis.imshow(linear_to_display(image));axis.set_title(label);axis.axis('off')
        fig.suptitle(f"{s['name']} | source zoom={tags['raw_zoom_active']} | orientation={s['target_orientation']}\nDiagnostic only; display transfer applied, no exposure matching")
        fig.tight_layout();panel=PANELS/f"{s['name']}.png";fig.savefig(panel,dpi=140);plt.close(fig)
        D.need(D.sha(inp)==s['input_sha256'] and D.sha(target)==s['target_cache_sha256'],'Input/target changed during measurement')
        rows.append({'name':s['name'],'shoot':s['shoot'],'role':'fit','source_zoom':tags['raw_zoom_active'],'source_zoom_size':tags['raw_zoom_size'],
                     'target_orientation':s['target_orientation'],'G_native_size':s['G_native_size'],'window':window,'metrics':measured,
                     'panel_file':str(panel.relative_to(D.ROOT)),'panel_sha256':D.sha(panel)})
        print('GEOMETRY_REGISTRATION',s['name'],{k:v['feature_ncc'] for k,v in measured.items()},flush=True)
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'R11 changed')
    result={'manifest_sha256':expected,'completed':True,'rows':rows,'scope':m['scope'],'interpretation':m['interpretation'],
            'R11_sha256':D.R11_SHA,'no_transform_selected':True,'no_colour_or_model_score':True}
    D.write_once_json(REPORT,result);print('REGISTRATION_DIAGNOSTIC_COMPLETE',len(rows),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'));ap.add_argument('--manifest-sha');a=ap.parse_args()
    prepare() if a.command=='prepare' else run(a.manifest_sha)


if __name__=='__main__':main()
