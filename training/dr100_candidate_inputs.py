"""Prepare/independently verify source-only G inputs for nine metadata candidates.

No targets, neural models, role assignment, training or protected inputs.
The existing decoder/codec/options remain unchanged; candidate provenance
uses a truthful source annotation rather than implying training admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import check_dr100_candidate_wb as B
import fjlrg_v2 as V

A=B.A;ROOT=A.ROOT
MF=ROOT/'training/diagnostics/dr100-candidate-input-manifest.json'
OUT=ROOT/'training/checkpoints/dr100-candidate-inputs'
PLAN=ROOT/'training/coordination/dr100-candidate-input-plan.md'
PHASE_B_SHA='697ce74215a49c3657c95d47414198b97dfb4a4ba3dddfad5c2edfbfe801f16b'
ORIGIN='frozen DR100 reference-candidate metadata: Auto resolved100; no training admission'
R11=ROOT/'training/checkpoints/r11-drcond-seed42-best.pt'
R11_SHA='1cfd2682dfe390b5fdb95d516889050f21bf00662c27f1f4a51f9d68005c291c'
G_PARENT=ROOT/'training/diagnostics/neutral-wb-study-manifest.json'
G_PARENT_SHA='d416942f282ec8d1fbe65fad6ad6a4e3c30d612d7938cc6052bc364885b018d0'
CODE=[Path(__file__).resolve(),ROOT/'training/test_dr100_candidate_inputs.py',
      ROOT/'scripts/export-dr100-candidate-inputs.mjs',ROOT/'scripts/test-dr100-candidate-input-host.mjs',
      ROOT/'scripts/lib/dr100-candidate-input-scope.mjs',PLAN,
      ROOT/'scripts/lib/decode-provia-v2-research.mjs',ROOT/'scripts/lib/decode-dr100-candidate-g.mjs',ROOT/'scripts/lib/fjlrg-v2.mjs',
      ROOT/'scripts/lib/raf-wb-coefficients.mjs',ROOT/'training/fjlrg_v2.py',
      ROOT/'package-lock.json',B.SCOPE,B.OUTPUT,G_PARENT]
LIBS=[ROOT/'node_modules/libraw-wasm/dist'/n for n in ('index.js','worker.js','libraw.js','libraw.wasm')]


def canaries(scenes):
    key=lambda s:hashlib.sha256(('dr100-candidate-G-input-v1\n'+s['name']).encode()).hexdigest()
    return [min((s for s in scenes if s['source_wb_mode']==mode),key=key)['name'] for mode in (1,4080)]


def metadata():
    B.load_scope(PHASE_B_SHA);A.committed(B.OUTPUT);report=A.read(B.OUTPUT)
    A.need(report['completed_phaseB'] is True and report['metadata_supported_count']==9
           and report['scope_sha256']==PHASE_B_SHA and report['training_admission'] is False,'Metadata qualification incomplete')
    A.need([r['name'] for r in report['rows']]==list(A.NAMES),'Metadata source coverage differs')
    for r in report['rows']:
        A.need(r['metadata_valid'] is True and r['decision']['policy_supported'] is True
               and r['decision']['neutral_kind_candidate']=='G'
               and r['decision']['source_policy_fields']['source_wb_fine_tune_raw']==[0,0], 'Only qualified zero-shift G candidates allowed')
    return report


def prepare():
    report=metadata();A.need(A.sha(G_PARENT)==G_PARENT_SHA,'Existing G study changed')
    parent=A.read(G_PARENT);wasm={r['inputs']['G']['wasm_sha256'] for r in parent['scenes']}
    A.need(len(wasm)==1 and A.sha(LIBS[-1]) in wasm,'Installed WASM differs from qualified G domain')
    for p in CODE:A.committed(p)
    scenes=[]
    for r in report['rows']:
        s=r['decision']['source_policy_fields']
        scenes.append({'name':r['name'],'path':r['path'],'raf_sha256':r['raf_sha256'],**s,
                       'capture_dynamic_range':100,'iso':r['metadata']['exif']['iso'],
                       'dateTimeOriginal':r['metadata']['exif']['dateTimeOriginal'],
                       'options':V.canonical_options(V.CONTRACT_ASSHOT,s['cam_mul'])})
    A.dump(MF,{'format':'dr100-candidate-G-input-v1','scope':'Input-data acquisition only. No training admission, targets, neural model, camera hardware or protected access.',
               'scenes':scenes,'canary_names':canaries(scenes),'output_dir':str(OUT.relative_to(ROOT)),
               'wasm_sha256':A.sha(LIBS[-1]),'capture_DR_annotation':ORIGIN,
               'bindings':{str(p.relative_to(ROOT)):A.sha(p) for p in CODE+LIBS},
               'training_admission':False,'repeat':'Independent page/worker G and R; exact native browser hash, working/container/sidecar/strip equality.',
               'native_limit':'Full native hashes are browser-reported, not independently rehashed from stored full arrays; stored containers/sidecars/strips are independently checked.',
               'R11_sha256':R11_SHA})
    print('DR100_CANDIDATE_INPUT_MANIFEST',A.sha(MF),flush=True)


def load(expected):
    A.need(isinstance(expected,str) and len(expected)==64 and A.sha(MF)==expected,'Pinned input manifest required')
    A.committed(MF);m=A.read(MF);report=metadata()
    A.need(m['format']=='dr100-candidate-G-input-v1' and m['training_admission'] is False
           and m['output_dir']==str(OUT.relative_to(ROOT)) and m['capture_DR_annotation']==ORIGIN,'Input scope changed')
    A.need(set(m['bindings'])=={str(p.relative_to(ROOT)) for p in CODE+LIBS},'Input bindings expanded/incomplete')
    for rel,h in m['bindings'].items():
        p=ROOT/rel;A.need(A.sha(p)==h,'Changed input binding: '+rel)
        if not rel.startswith('node_modules/'):A.committed(p)
    A.need([s['name'] for s in m['scenes']]==list(A.NAMES) and m['canary_names']==canaries(m['scenes']),'Input source/canary set differs')
    prior_wasm={r['inputs']['G']['wasm_sha256'] for r in A.read(G_PARENT)['scenes']}
    A.need(prior_wasm=={m['wasm_sha256']} and m['wasm_sha256']==m['bindings'][str(LIBS[-1].relative_to(ROOT))],
           'Declared WASM differs from served/current qualified domain')
    for s,r in zip(m['scenes'],report['rows']):
        A.need(all(s[k]==r[k] for k in ('name','path','raf_sha256')),'Input source provenance differs')
        A.need(all(s[k]==v for k,v in r['decision']['source_policy_fields'].items())
               and s['capture_dynamic_range']==100 and s['options']==V.canonical_options(V.CONTRACT_ASSHOT,s['cam_mul']),
               'Source fields/options changed')
        A.need(s['iso']==r['metadata']['exif']['iso'] and s['dateTimeOriginal']==r['metadata']['exif']['dateTimeOriginal'],
               'Source ISO/time metadata changed')
    A.need(A.sha(R11)==R11_SHA,'Selected R11 changed')
    return m


def verify(expected,only_canaries):
    m=load(expected);wanted=[s for s in m['scenes'] if not only_canaries or s['name'] in m['canary_names']]
    rows=[]
    for s in wanted:
        folder=OUT/s['name'].removesuffix('.RAF');seal_path=folder/'scene-seal.json';seal=A.read(seal_path)
        A.need(seal['name']==s['name'] and seal['manifest_sha256']==expected and seal['training_admission'] is False,'Wrong scene seal')
        checked={}
        for arm in ('G','R'):
            r=seal['arms'][arm];file=folder/f'{arm}.fjlrg';side=Path(str(file)+'.json');strips=folder/f'{arm}-strips.u16le'
            A.need(r['file']==str(file.relative_to(ROOT)) and A.sha(file)==r['sha256']
                   and A.sha(side)==r['sidecar_sha256'] and A.sha(strips)==r['strips_sha256'],'Stored artifact changed')
            frame=V.load_fjlrg_v2(file,expected_contract=V.CONTRACT_ASSHOT,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=m['wasm_sha256'])
            p=frame.provenance
            A.need(p['source_wb_mode']==s['source_wb_mode'] and p['source_kelvin']==s['source_kelvin']
                   and p['source_wb_fine_tune_raw']==[0,0] and p['capture_dynamic_range']==100
                   and p['capture_dynamic_range_source']==ORIGIN and p['wb_source']['levels_rgbg']==s['cam_mul'],
                   'Input annotation/WB/DR differs')
            A.need(r['source_coefficients']['asShotCamMul']==s['cam_mul']
                   and r['source_coefficients']['autoRgbg']==s['auto_rgbg']
                   and [frame.width,frame.height]==[r['width'],r['height']],'Source coefficient/working size mismatch')
            A.need(p['options_json']==r['options_json'] and p['options_sha256']==r['options_sha256']
                   and p['decoded_native_size']==r['native_size'],'Declared parameters/geometry differ')
            native_width,native_height=r['native_size'];strip_rows=list(range(64,native_height,128))
            A.need(r['native_channels']==native_width*native_height*3,'Native RGB channel count differs')
            A.need(r['strip_rows']==strip_rows and strips.stat().st_size==len(strip_rows)*native_width*3*2,'Native strip layout differs')
            payload=frame.rgb_uint16.astype('<u2',copy=False).tobytes()
            A.need(hashlib.sha256(payload).hexdigest()==r['working_payload_sha256'],'Working bytes differ')
            checked[arm]=(r,frame,side.read_bytes(),strips.read_bytes())
        g,r=checked['G'],checked['R']
        A.need(g[0]['sha256']==r[0]['sha256'] and g[0]['native_sha256']==r[0]['native_sha256']
               and g[0]['native_size']==r[0]['native_size'] and g[2:]==r[2:],'Independent repeat differs')
        A.need(A.sha(ROOT/s['path'])==s['raf_sha256'],'Source changed after input capture')
        rows.append({'name':s['name'],'verified':True,'container_sha256':g[0]['sha256'],
                     'scene_seal_sha256':A.sha(seal_path),'file':g[0]['file'],
                     'sidecar_sha256':g[0]['sidecar_sha256'],'native_repeat_hash':g[0]['native_sha256']})
    A.need(len(rows)==(2 if only_canaries else 9) and A.sha(R11)==R11_SHA,'Coverage/R11 changed')
    if not only_canaries:
        done=A.read(OUT/'complete.json');A.need(done['manifest_sha256']==expected and done['completed'] is True and len(done['names'])==9,'Final export receipt missing')
    output=ROOT/'training/diagnostics'/('dr100-candidate-input-canary-verification.json' if only_canaries else 'dr100-candidate-input-verification.json')
    A.dump(output,{'verified':True,'canary_only':only_canaries,'manifest_sha256':expected,'verification_script_sha256':A.sha(Path(__file__)),
                   'rows':rows,'training_admission':False,'R11_sha256':R11_SHA,
                   'limits':'Input repeat/provenance/stored-byte checks only; native full hashes browser-reported. No Fuji rendering accuracy, target read, fitting or app integration.'})
    print('DR100_CANDIDATE_G_INPUTS_VERIFIED',len(rows),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify'))
    ap.add_argument('--manifest-sha');ap.add_argument('--canaries',action='store_true');a=ap.parse_args()
    prepare() if a.command=='prepare' else verify(a.manifest_sha,a.canaries)
