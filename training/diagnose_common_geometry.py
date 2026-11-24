"""Eight-FIT diagnostic of one source-metadata-derived common-rectangle hypothesis.

The RAW header crop includes borders and is not asserted to be Fuji's exact
visible frame. No per-image fitted warp, colour score, model or data change.
Work-scale resampling is a diagnostic approximation, not a native exporter.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import matched_gamma_data as D
import diagnose_geometry_registration as A
import geometry_registration_metrics as R
import fjlrg_v2 as V

PARENT_SHA='af7ee01bb3e1a9460cc574b441d12ca3b60037d90be66cb37bd5046a0d6ead7a'
META_SHA='60e4f3292c0414a001ca677f5d8733ea487eb99e8f8a1679b46a14644028e712'
MANIFEST=D.DIAG/'common-geometry-diagnostic-manifest.json'
REPORT=D.DIAG/'common-geometry-diagnostic-report.json'
CRITERIA={'minimum_usable_tiles':6,'maximum_median_abs_dx':0.5,'maximum_median_abs_dy':0.5,
          'maximum_p95_shift_norm':1.0,'minimum_ncc_change':-0.02}


def prepare():
    import raw_common_geometry as C
    parent=A.load(PARENT_SHA)
    metadata_file=D.DIAG/'decoder-geometry-metadata-report.json'
    proof_file=D.DIAG/'decoder-geometry-metadata-verification.json'
    metadata=json.loads(metadata_file.read_text());proof=json.loads(proof_file.read_text())
    assert metadata['completed'] and proof['verified'] and proof['report_sha256']==D.sha(metadata_file)
    assert metadata['manifest_sha256']==proof['manifest_sha256']==META_SHA
    by={r['name']:r for r in metadata['rows']};scenes=[]
    for s in parent['scenes']:
        r=by[s['name']];assert s['role']==r['role']=='fit' and s['raf_sha256']==r['source_raf_sha256']
        tags=A.primary_tags(s)
        windows=C.common_sensor_windows(r['geometry'],tags['raw_image_crop_top_left'],tags['raw_image_cropped_size'],s['target_orientation'])
        scenes.append({**s,'decoder_geometry':r['geometry'],'hypothesis':windows})
    files=[Path(__file__),D.TRAIN/'test_common_geometry_diagnostic.py',D.TRAIN/'raw_common_geometry.py',D.TRAIN/'test_raw_common_geometry.py',
           D.TRAIN/'geometry_registration_metrics.py',A.MANIFEST,A.REPORT,D.DIAG/'geometry-registration-verification.json',
           D.DIAG/'decoder-geometry-metadata-manifest.json',metadata_file,proof_file]
    m={'scope':__doc__.strip(),'scenes':scenes,'parent_sha256':PARENT_SHA,'decoder_metadata_manifest_sha256':META_SHA,
       'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files},'criteria':CRITERIA,
       'measurement':'Original identity vs same fixed gradient/NCC/3x3 phase measurements after source-only common windows, both resampled to existing working dimensions. Entire tile records kept. No target-derived coordinates, search or model scores.',
       'criteria_interpretation':'Diagnostic consistency screen only; eight images already unblinded by earlier registration work. Not independent acceptance or justification to edit canonical inputs.',
       'limits':'Only8allowlistedFIT G/sidecars and hash-bound target-cache arrays, no RAW/TIFF/development/protected media, learned weights, colour/recipe metrics or fit. No canonical or model changes.'}
    D.write_once_json(MANIFEST,m);print('COMMON_GEOMETRY_MANIFEST',D.sha(MANIFEST),flush=True)


def load(expected):
    import raw_common_geometry as C
    assert expected and D.sha(MANIFEST)==expected and D.committed(MANIFEST)
    m=json.loads(MANIFEST.read_text());assert m['criteria']==CRITERIA
    for rel,h in m['bindings'].items():assert D.sha(D.ROOT/rel)==h and D.committed(D.ROOT/rel)
    parent=A.load(PARENT_SHA)
    assert len(m['scenes'])==8
    for s,p in zip(m['scenes'],parent['scenes']):
        assert all(s[k]==v for k,v in p.items()) and s['role']=='fit'
        tags=A.primary_tags(s)
        assert C.common_sensor_windows(s['decoder_geometry'],tags['raw_image_crop_top_left'],tags['raw_image_cropped_size'],s['target_orientation'])==s['hypothesis']
    return m


def summarize(tiles):
    t=[r for r in tiles if r['usable']]
    return {'usable_tiles':len(t),'median_abs_dx':float(np.median([abs(r['dx']) for r in t])) if t else None,
            'median_abs_dy':float(np.median([abs(r['dy']) for r in t])) if t else None,
            'p95_shift_norm':float(np.percentile([np.hypot(r['dx'],r['dy']) for r in t],95)) if t else None}


def measure(a,b):
    # OpenCV's phase centre can carry a half-pixel offset when its optimal
    # padded DFT size is odd, even for self-correlation. All eight approved
    # working dimensions use even optimal sizes; refuse other shapes here.
    h,w=a.shape[:2]
    for size in (h,w):
        bounds=[i*size//3 for i in range(4)]
        assert all(cv2.getOptimalDFTSize(bounds[i+1]-bounds[i]-8)%2==0 for i in range(3)), 'Odd optimal phase DFT size is not qualified in this diagnostic'
    fa,fb=R.registration_feature(a),R.registration_feature(b)
    tiles=R.tile_phase_shifts(fa,fb)
    return {'feature_ncc':R.feature_ncc(fa,fb),'tiles':tiles,'summary':summarize(tiles),
            'feature_mean_power':[float(np.mean(fa.astype(np.float64)**2)),float(np.mean(fb.astype(np.float64)**2))]}


def run(expected):
    m=load(expected);assert not REPORT.exists();rows=[]
    original=json.loads(A.REPORT.read_text());old={r['name']:r for r in original['rows']}
    for s in m['scenes']:
        ip,tp=D.ROOT/s['input_file'],D.ROOT/s['target_cache_file']
        assert D.sha(ip)==s['input_sha256'] and D.sha(Path(str(ip)+'.json'))==s['input_sidecar_sha256'] and D.sha(tp)==s['target_cache_sha256']
        g=V.load_fjlrg_v2(ip,expected_contract=V.CONTRACT_ASSHOT,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=s['wasm_sha256'])
        rgb=g.rgb_uint16.astype(np.float32)/np.float32(65535);ref=np.load(tp,allow_pickle=False)
        assert ref.dtype==np.float32 and ref.shape==rgb.shape and np.isfinite(ref).all()
        baseline=measure(rgb,ref)
        assert baseline['feature_ncc']==old[s['name']]['metrics']['identity']['feature_ncc'] and baseline['tiles']==old[s['name']]['metrics']['identity']['tiles']
        source=R.zoom_resample(rgb,s['hypothesis']['source_window'])
        target=R.zoom_resample(ref,s['hypothesis']['target_window'])
        candidate=measure(source,target);v=candidate['summary']
        ncc_change=candidate['feature_ncc']-baseline['feature_ncc'] if candidate['feature_ncc'] is not None and baseline['feature_ncc'] is not None else None
        checks={'tile_coverage':v['usable_tiles']>=CRITERIA['minimum_usable_tiles'],
                **{k:v[k] is not None and v[k]<=CRITERIA['maximum_'+k] for k in ('median_abs_dx','median_abs_dy','p95_shift_norm')},
                'ncc_noninferior':ncc_change is not None and ncc_change>=CRITERIA['minimum_ncc_change']}
        rows.append({'name':s['name'],'role':'fit','hypothesis':s['hypothesis'],'baseline':baseline,'candidate':candidate,'ncc_change':ncc_change,'checks':checks,'diagnostic_screen_passed':all(checks.values())})
        assert D.sha(ip)==s['input_sha256'] and D.sha(tp)==s['target_cache_sha256']
        print('COMMON_GEOMETRY',s['name'],'NCC',baseline['feature_ncc'],candidate['feature_ncc'],'shift',baseline['summary'],candidate['summary'],flush=True)
    assert D.sha(D.R11_PATH)==D.R11_SHA
    D.write_once_json(REPORT,{'manifest_sha256':expected,'completed':True,'rows':rows,
                            'diagnostic_screen_passed':all(r['diagnostic_screen_passed'] for r in rows),'criteria':CRITERIA,
                            'limits':m['limits'],'interpretation':m['criteria_interpretation'],'R11_sha256':D.R11_SHA})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'));ap.add_argument('--manifest-sha');a=ap.parse_args()
    prepare() if a.command=='prepare' else run(a.manifest_sha)
