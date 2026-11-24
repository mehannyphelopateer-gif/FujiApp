"""Full-native coverage follow-up to a SHORT sparse manual-WB primitive.

Same three sources, same D/E bytes, same gains/mask/numerical limits. The
original NOT_QUALIFIED verdict stays immutable. No target/model/runtime work.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import matched_gamma_data as D
import qualify_manual_kelvin_wb as Q
import eval_provia_wb_primitive as S

SPARSE_SHA='3ee9476c13b6bc8a28c4bfb7306a616d492f94d4ef904c5cbe88617dfa14a0ea'
MANIFEST=D.DIAG/'manual-kelvin-wb-fullnative-manifest.json'
REPORT=D.DIAG/'manual-kelvin-wb-fullnative-report.json'
OUT=D.CKPT/'manual-kelvin-wb-fullnative'
CODE=Path(__file__).resolve()
BLOCK_ROWS=128


def sparse_state():
    old=Q.load(SPARSE_SHA)
    result=json.loads(Q.REPORT.read_text())
    proof_path=D.DIAG/'manual-kelvin-wb-primitive-verification.json'
    proof=json.loads(proof_path.read_text())
    complete_path=Q.OUT/'decode-complete.json'
    complete=json.loads(complete_path.read_text())
    D.need(result['verdict']=='NOT_QUALIFIED' and result['manifest_sha256']==SPARSE_SHA
           and proof['verified'] is True and proof['report_sha256']==D.sha(Q.REPORT)
           and proof['complete_sha256']==D.sha(complete_path)
           and complete['completed'] is True and complete['manifest_sha256']==SPARSE_SHA,
           'Original sparse/SHORT evidence differs')
    D.need(len(result['rows'])==3 and {r['name'] for r in result['rows']}=={s['name'] for s in old['scenes']},'Original coverage differs')
    short=0
    for row in result['rows']:
        D.need(row['structure_ok'] and row['controls_and_repeat_exact'],'Not a sampling-only follow-up')
        for channel in row['channels_RGB']:
            D.need(channel.get('mean_abs_codes',float('inf'))<=1 and channel.get('max_abs_codes',float('inf'))<=3,
                   'Sparse numerical failure cannot be relabelled as insufficient coverage')
            if not channel['measurable']:short+=1
            else:D.need(channel['passed'] is True,'Measurable sparse channel failed')
    D.need(short>0,'Follow-up scope requires genuinely SHORT sparse coverage')
    return old,result,complete,proof_path,complete_path


def prepare():
    old,result,complete,proof_path,complete_path=sparse_state()
    by={r['name']:r for r in complete['scenes']}
    scenes=[]
    for source in old['scenes']:
        arms=by[source['name']]['arms'];scene=dict(source)
        D.need(arms['D']['shape']==arms['E']['shape'],'Native size differs')
        scene.update(native_shape=arms['D']['shape'],
                     original_native_sha256={a:arms[a]['native_sha256'] for a in ('D','E')})
        scenes.append(scene)
    files=[CODE,D.TRAIN/'test_manual_kelvin_wb_fullnative.py',D.ROOT/'scripts/capture-manual-kelvin-wb-fullnative.mjs',
           Q.MANIFEST,Q.REPORT,proof_path,complete_path,Path(Q.__file__).resolve(),Path(S.__file__).resolve()]
    files += [D.ROOT/p for p in old['bindings'] if p.startswith('node_modules/libraw-wasm/')]
    manifest={'scope':__doc__.strip(),'original_manifest_sha256':SPARSE_SHA,
              'original_verdict':'NOT_QUALIFIED (sparse blue count); preserved, not superseded',
              'scenes':scenes,'base_options':old['base_options'],'output_dir':str(OUT.relative_to(D.ROOT)),
              'primitive_limits':old['primitive_limits'],
              'selection':'EXACT same3 frozen manual-Kelvin primitive sources, no additions/exclusions or changed WB/options. Full arrays must hash-identically to old D/E native buffers.',
              'sampling':'ALL native CFA sites; no changed intensity mask or fitted gain. Chunked128-row independent NumPy accumulation. Inactive RGB channels must remain zero, old6x6CFA pattern fixed.',
              'limits':'Same per-scene/channel >=1000 sites in BOTH-arms1–80% range, meanabs<=1code,max<=3codes. Original SHORT remains NOT_QUALIFIED. New label only extends photosite coverage.',
              'interpretation':old['interpretation'],
              'boundaries':'Read only same3fit RAFs for6decodes; persist6ignored raw sparse-CFA native arrays (~1.44GB) and hashes, no rendered photos/FJLRG/target images/model inference/development/protected media/training/runtime changes.',
              'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files}}
    D.write_once_json(MANIFEST,manifest)
    print('MANUAL_KELVIN_FULLNATIVE_MANIFEST',D.sha(MANIFEST),flush=True)


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected
           and D.committed(MANIFEST),'Unpinned/uncommitted full-native probe')
    m=json.loads(MANIFEST.read_text())
    for rel,h in m['bindings'].items():
        p=D.ROOT/rel;D.need(D.sha(p)==h,'Bound evidence/code changed: '+rel)
        if not rel.startswith('node_modules/') and 'checkpoints/' not in rel:D.need(D.committed(p),'Uncommitted evidence/code')
    old,_,complete,_,_=sparse_state();by={r['name']:r for r in complete['scenes']}
    D.need(m['base_options']==old['base_options'] and m['primitive_limits']==old['primitive_limits']
           and len(m['scenes'])==3 and m['original_manifest_sha256']==SPARSE_SHA,'Options/cohort/tolerances differ')
    for scene,prior in zip(m['scenes'],old['scenes']):
        D.need(all(scene[k]==v for k,v in prior.items()),'Source/gain selection changed')
        D.need(scene['native_shape']==by[scene['name']]['arms']['D']['shape']
               and scene['original_native_sha256']=={a:by[scene['name']]['arms'][a]['native_sha256'] for a in ('D','E')},
               'Original full-native identity changed')
    return m,by


def full_statistics(d,e,pattern,gains,block_rows=BLOCK_ROWS):
    """Independent all-site proof; bounded memory, no sampling/refitting/clipping."""
    D.need(d.shape==e.shape and len(d.shape)==3 and d.shape[2]==3,'Native array geometry differs')
    D.need(d.dtype==e.dtype==np.dtype('<u2'),'Native arrays must be uint16 little-endian')
    pat=np.asarray(pattern,dtype=np.int64)
    D.need(pat.shape==(6,6) and set(pat.ravel())=={0,1,2},'Invalid CFA pattern')
    D.need(type(block_rows) is int and block_rows>0 and len(gains)==3
           and all(np.isfinite(g) and g>0 for g in gains),'Invalid accumulation settings')
    sums=np.zeros(3);signed=np.zeros(3);maxima=np.zeros(3);counts=np.zeros(3,dtype=np.int64)
    totals=np.zeros(3,dtype=np.int64);exacts=np.zeros(3,dtype=np.int64)
    h,w,_=d.shape;xs=np.arange(w)%6
    for start in range(0,h,block_rows):
        stop=min(h,start+block_rows);ys=np.arange(start,stop)%6
        channels=pat[ys[:,None],xs[None,:]]
        for c,g in enumerate(gains):
            dv=np.asarray(d[start:stop,:,c]);ev=np.asarray(e[start:stop,:,c])
            active=channels==c
            D.need(not np.any(dv[~active]) and not np.any(ev[~active]),'Inactive CFA channel became nonzero')
            totals[c]+=np.count_nonzero(active)
            mask=(active & (dv>=S.MASK_LOW*S.FULL_SCALE) & (dv<=S.MASK_HIGH*S.FULL_SCALE)
                  & (ev>=S.MASK_LOW*S.FULL_SCALE) & (ev<=S.MASK_HIGH*S.FULL_SCALE))
            n=int(mask.sum());counts[c]+=n
            if not n:continue
            dd=dv[mask].astype(np.float64);ee=ev[mask].astype(np.float64)
            error=ee-g*dd;absolute=np.abs(error)
            sums[c]+=absolute.sum(dtype=np.float64);signed[c]+=error.sum(dtype=np.float64)
            maxima[c]=max(maxima[c],float(absolute.max()));exacts[c]+=np.count_nonzero(ee==g*dd)
    out=[]
    for c,g in enumerate(gains):
        n=int(counts[c]);measurable=n>=S.MIN_N
        mean=float(sums[c]/n) if n else None
        maximum=float(maxima[c]) if n else None
        out.append({'channel':c,'expected_gain':g,'all_active_sites':int(totals[c]),'n':n,'measurable':measurable,
                    'mean_abs_codes':mean,'max_abs_codes':maximum,
                    'signed_mean_codes':float(signed[c]/n) if n else None,
                    'exact_fraction':float(exacts[c]/n) if n else None,
                    'passed':bool(measurable and mean<=S.MEAN_LIMIT_CODES and maximum<=S.MAX_LIMIT_CODES)})
    D.need(int(totals.sum())==h*w,'Not every native site visited exactly once')
    return out


def run(expected):
    m,old=load(expected);D.need(not REPORT.exists(),'Prior full-native result exists')
    complete=OUT/'decode-complete.json';decoded=json.loads(complete.read_text())
    D.need(decoded['manifest_sha256']==expected and decoded['completed'] is True,'Incomplete/wrong capture')
    by={r['name']:r for r in decoded['scenes']}
    D.need(len(by)==len(decoded['scenes'])==3 and set(by)=={s['name'] for s in m['scenes']},'Capture coverage differs')
    results=[]
    for scene in m['scenes']:
        name=scene['name'];prior=old[name];shape=scene['native_shape'];h,w=shape['height'],shape['width']
        row=by[name];D.need(set(row['arms'])=={'D','E'},'Unknown/missing native arm')
        arrays={}
        for arm in ('D','E'):
            item=row['arms'][arm];p=D.ROOT/item['file']
            D.need(p.resolve().parent==OUT.resolve() and p.name==f'{name}-{arm}.native.u16le','Native path escaped')
            D.need(p.stat().st_size==h*w*3*2==item['bytes'] and D.sha(p)==item['sha256']==scene['original_native_sha256'][arm],
                   'New full-native data differs from frozen sparse experiment')
            D.need(item['shape']==shape and item['declared_options']==Q.expected_options(m['base_options'],scene,arm),
                   'Native geometry/options differ')
            md=item['returned_metadata']
            D.need(md['cam_mul']==scene['cam_mul'] and md['camera_model']=='X100VI' and md['fujiWB']==4080,'Source metadata differs')
            arrays[arm]=np.memmap(p,dtype='<u2',mode='r',shape=(h,w,3))
            sparse=p.parent.parent/'manual-kelvin-wb-primitive'/Path(prior['arms'][arm]['samples_file']).name
            D.need(sparse.parent==Q.OUT and D.sha(sparse)==prior['arms'][arm]['samples_sha256'],'Original samples changed')
            expected_grid=np.fromfile(sparse,dtype='<u2')
            actual_grid=arrays[arm][S.GRID_OFFSET::S.GRID_STRIDE,S.GRID_OFFSET::S.GRID_STRIDE,:].reshape(-1)
            D.need(np.array_equal(actual_grid,expected_grid),'Original sparse sites are not reproduced')
        pattern,problems=S.check_structure(prior['arms']['D'],prior['arms']['E'])
        D.need(not problems,'Original CFA structure invalid')
        metrics=full_statistics(arrays['D'],arrays['E'],np.array(pattern).reshape(6,6),scene['expected_gain_RGB'])
        result={'name':name,'session':scene['session'],'native_shape':shape,'channels_RGB':metrics,
                'native_hashes_match_original':True,'sparse_sites_reproduce':True,'passed':all(c['passed'] for c in metrics)}
        results.append(result)
        print('FULL_NATIVE_WB',name,[(c['n'],c['mean_abs_codes'],c['max_abs_codes']) for c in metrics],flush=True)
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'Selected model changed')
    report={'manifest_sha256':expected,'decode_report_sha256':D.sha(complete),'verdict':'PASSED' if all(r['passed'] for r in results) else 'NOT_QUALIFIED',
            'rows':results,'original_sparse_verdict':'NOT_QUALIFIED, unchanged','limits':m['interpretation'],
            'boundaries':m['boundaries'],'R11_unchanged':D.R11_SHA}
    D.write_once_json(REPORT,report)
    print('FULL_NATIVE_WB_VERDICT',report['verdict'],flush=True)
    if report['verdict']!='PASSED':raise SystemExit(1)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'))
    ap.add_argument('--manifest-sha');args=ap.parse_args()
    if args.command=='prepare':prepare()
    else:run(args.manifest_sha)


if __name__=='__main__':main()
