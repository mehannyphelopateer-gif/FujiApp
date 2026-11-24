"""Transport-only revision of the unchanged full-native WB measurement.

Host-verified upload completion replaces dependence on a browser/CDP return.
The same three files, options, old hashes, gain equations and limits apply.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import matched_gamma_data as D
import qualify_manual_kelvin_wb_fullnative as F
import qualify_manual_kelvin_wb as Q
import eval_provia_wb_primitive as S

PARENT_SHA='22f982280dad7703f938dd51dd6c16b849f2a43cbfe23510173eec52e520a3ac'
MANIFEST=D.DIAG/'manual-kelvin-wb-fullnative-v2-manifest.json'
REPORT=D.DIAG/'manual-kelvin-wb-fullnative-v2-report.json'
OUT=D.CKPT/'manual-kelvin-wb-fullnative-v2'
ATTEMPT=D.DIAG/'manual-kelvin-wb-fullnative-attempt1.json'
CODE=Path(__file__).resolve()


def unchanged_measurement(new,old):
    for key in ('scenes','base_options','primitive_limits','selection','sampling','limits','interpretation'):
        D.need(new[key]==old[key],'Transport revision changed scientific field: '+key)
    D.need(new['transport_revision']==2 and new['parent_fullnative_manifest_sha256']==PARENT_SHA,
           'Wrong transport lineage')


def prepare():
    old,_=F.load(PARENT_SHA)
    failure=json.loads(ATTEMPT.read_text());progress=F.OUT/'decode-progress.json'
    D.need(failure['status']=='INCOMPLETE_TRANSPORT_TIMEOUT' and failure['manifest_sha256']==PARENT_SHA
           and failure['capture_progress_sha256']==D.sha(progress),'Original transport failure changed')
    m=copy.deepcopy(old)
    m.pop('_manifest_sha256',None)
    m.update(scope=__doc__.strip(),parent_fullnative_manifest_sha256=PARENT_SHA,transport_revision=2,
             output_dir=str(OUT.relative_to(D.ROOT)),
             transport='Validated current decoder metadata is sent in a small upload header; host verifies full length/hash/exclusive publication and settles an upload promise. Do not depend on a CDP return for large buffers. Dispose worker before upload, no extra page-side hash copy. Fresh owned BrowserServer per arm, bounded cleanup/kill of that own child only. No scientific variables change.')
    files=[F.MANIFEST,ATTEMPT,progress,CODE,D.TRAIN/'test_manual_kelvin_wb_fullnative_v2.py',
           D.ROOT/'scripts/capture-manual-kelvin-wb-fullnative-v2.mjs',Path(F.__file__).resolve()]
    m['bindings'].update({str(p.relative_to(D.ROOT)):D.sha(p) for p in files})
    unchanged_measurement(m,old)
    D.write_once_json(MANIFEST,m)
    print('MANUAL_KELVIN_FULLNATIVE_V2_MANIFEST',D.sha(MANIFEST),flush=True)


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected
           and D.committed(MANIFEST),'Unpinned/uncommitted transport revision')
    m=json.loads(MANIFEST.read_text())
    for rel,h in m['bindings'].items():
        p=D.ROOT/rel;D.need(D.sha(p)==h,'Bound artifact changed: '+rel)
        if not rel.startswith('node_modules/') and 'checkpoints/' not in rel:D.need(D.committed(p),'Uncommitted binding')
    old,by=F.load(PARENT_SHA);unchanged_measurement(m,old)
    D.need(m['output_dir']==str(OUT.relative_to(D.ROOT)),'Wrong transport output namespace')
    return m,by


def run(expected):
    m,old=load(expected);D.need(not REPORT.exists(),'Prior revision result exists')
    complete=OUT/'decode-complete.json';decoded=json.loads(complete.read_text())
    D.need(decoded['manifest_sha256']==expected and decoded['completed'] is True,'Incomplete/wrong capture')
    by={r['name']:r for r in decoded['scenes']}
    D.need(len(by)==len(decoded['scenes'])==3 and set(by)=={s['name'] for s in m['scenes']},'Capture coverage differs')
    results=[]
    for scene in m['scenes']:
        name=scene['name'];prior=old[name];shape=scene['native_shape'];h,w=shape['height'],shape['width']
        row=by[name];D.need(set(row['arms'])=={'D','E'},'Unknown/missing native arm')
        D.need(not row.get('stopped') and row['raf_sha256_after']==scene['raf_sha256'],
               'Source verification incomplete or source changed')
        arrays={}
        for arm in ('D','E'):
            item=row['arms'][arm];p=D.ROOT/item['file']
            D.need(item['status']=='verified','Native arm was not verified by capture')
            D.need(p.resolve().parent==OUT.resolve() and p.name==f'{name}-{arm}.native.u16le','Native path escaped')
            D.need(p.stat().st_size==h*w*3*2==item['bytes'] and D.sha(p)==item['sha256']==scene['original_native_sha256'][arm],
                   'Full-native bytes differ from original sparse experiment')
            D.need(item['shape']==shape and item['declared_options']==Q.expected_options(m['base_options'],scene,arm),
                   'Native geometry/options differ')
            md=item['returned_metadata']
            D.need(md['cam_mul']==scene['cam_mul'] and md['camera_model']=='X100VI' and md['fujiWB']==4080,'Current header readback differs')
            arrays[arm]=np.memmap(p,dtype='<u2',mode='r',shape=(h,w,3))
            sparse=Q.OUT/Path(prior['arms'][arm]['samples_file']).name
            D.need(D.sha(sparse)==prior['arms'][arm]['samples_sha256'],'Original sparse samples changed')
            expected_grid=np.fromfile(sparse,dtype='<u2')
            actual_grid=arrays[arm][S.GRID_OFFSET::S.GRID_STRIDE,S.GRID_OFFSET::S.GRID_STRIDE,:].reshape(-1)
            D.need(np.array_equal(actual_grid,expected_grid),'Original sparse sites do not reproduce')
        pattern,problems=S.check_structure(prior['arms']['D'],prior['arms']['E'])
        D.need(not problems,'Original CFA structure invalid')
        metrics=F.full_statistics(arrays['D'],arrays['E'],np.array(pattern).reshape(6,6),scene['expected_gain_RGB'])
        results.append({'name':name,'session':scene['session'],'native_shape':shape,'channels_RGB':metrics,
                        'native_hashes_match_original':True,'sparse_sites_reproduce':True,'passed':all(c['passed'] for c in metrics)})
        print('FULL_NATIVE_V2_WB',name,[(c['n'],c['mean_abs_codes'],c['max_abs_codes']) for c in metrics],flush=True)
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'Selected model changed')
    r={'manifest_sha256':expected,'decode_report_sha256':D.sha(complete),'verdict':'PASSED' if all(r['passed'] for r in results) else 'NOT_QUALIFIED',
       'rows':results,'original_sparse_verdict':'NOT_QUALIFIED unchanged','original_full_capture':'INCOMPLETE_TRANSPORT_TIMEOUT unchanged',
       'limits':m['interpretation'],'boundaries':m['boundaries'],'R11_unchanged':D.R11_SHA}
    D.write_once_json(REPORT,r)
    print('FULL_NATIVE_V2_VERDICT',r['verdict'],flush=True)
    if r['verdict']!='PASSED':raise SystemExit(1)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'))
    ap.add_argument('--manifest-sha');args=ap.parse_args()
    if args.command=='prepare':prepare()
    else:run(args.manifest_sha)


if __name__=='__main__':main()
