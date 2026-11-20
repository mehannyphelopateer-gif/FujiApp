"""Prepare/verify source-Auto W input pilot; no target or model reads."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import fjlrg_v2 as V

ROOT=Path(__file__).resolve().parent.parent
D=ROOT/'training/diagnostics'
MANIFEST=D/'provia-v2-wb-pilot-manifest.json'
REPORT=D/'provia-v2-wb-pilot-verification.json'
OUT=ROOT/'training/checkpoints/provia-input-v2-wb-pilot'
G_MANIFEST=D/'provia-v2-fullnative-pilot-manifest.json'
G_REPORT=D/'provia-v2-fullnative-pilot-verification.json'
PRIMITIVE=D/'provia-wb-primitive-v2-report.json'
PARSER=D/'provia-runtime-wb-parser-verification.json'
SOURCE=D/'provia-source-wb-probe.json'

def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,d):
    with p.open('x')as f:f.write(json.dumps(d,indent=2,allow_nan=False)+'\n')
def committed(p):
    b=subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT)
    need(hashlib.sha256(b).hexdigest()==sha(p),'Uncommitted '+str(p))

def prepare():
    gm,gr,pr,pa,source=map(read,(G_MANIFEST,G_REPORT,PRIMITIVE,PARSER,SOURCE))
    need(gr['passed'] and gr['manifest_sha256']==sha(G_MANIFEST),'G pilot not independently passed')
    need(pr['verdict']=='PASSED' and len(pr['rows'])==11,'Primitive not passed')
    need(pa['passed'] and len(pa['rows'])==24 and source['crosschecks_passed'],'Parser gate not passed')
    gmap={r['name']:r for r in gr['scenes']}; smap={r['name']:r for r in source['rows']}; pmap={r['name']:r for r in pa['rows']}
    scenes=[]
    for scene in gm['scenes']:
        if scene['source_wb_mode']!=0:continue
        r=scene.copy();n=r['name'];s=smap[n];p=pmap[n];g=gmap[n]
        need(s['source_wb_mode']==p['source_wb_mode']==0 and p['source_raf_sha256']==s['raf_sha256']==r['raf_sha256'],'Metadata source mismatch')
        need(p['asShotCamMul']==r['cam_mul'],'Source as-shot mismatch')
        r['auto_rgbg']=p['autoRgbg']
        # Only the independently read RAF record is used, never reference WB coefficients.
        raw=[t for t in s['records'] if t['tag']=='0xf00d' and t['supported']]
        need(len(raw)==1 and r['auto_rgbg']==[raw[0]['values'][1],raw[0]['values'][0],raw[0]['values'][2],raw[0]['values'][0]],'Not a unique source Auto record')
        gfolder=ROOT/'training/checkpoints/provia-input-v2-pilot-fullnative'/r['shoot']
        sealed=read(gfolder/'scene-complete.json')['arms']['G']
        need(sealed['sha256']==g['linear_v2_sha256'] and sha(ROOT/sealed['file'])==sealed['sha256'],'G baseline changed')
        r.update(g_file=sealed['file'],g_sha256=sealed['sha256'],g_strips_file=sealed['strips']['file'],g_strips_sha256=sealed['strips']['sha256'],g_native_size=sealed['native_size'])
        need(sha(ROOT/r['g_strips_file'])==r['g_strips_sha256'],'G strips changed')
        scenes.append(r)
    need(len(scenes)==len({s['name']for s in scenes})==len({s['shoot']for s in scenes})==11,'F11 coverage differs')
    need({r['name']for r in pr['rows']}=={r['name']for r in scenes},'Primitive coverage differs')
    paths=['scripts/export-provia-input-v2-wb-pilot.mjs','scripts/lib/decode-provia-v2-research.mjs','scripts/lib/raf-wb-coefficients.mjs','scripts/lib/fjlrg-v2.mjs','training/verify_provia_v2_wb_pilot.py','training/fjlrg_v2.py']
    paths+=[str(p.relative_to(ROOT))for p in (G_MANIFEST,G_REPORT,PRIMITIVE,PARSER,SOURCE)]
    paths+=['node_modules/libraw-wasm/dist/'+n for n in ('index.js','worker.js','libraw.js','libraw.wasm')]
    dump(MANIFEST,{'scope':'F11 source-Auto research inputs only. No target/model/app/protected-tier access. Gamma remains six-element linear; WB is source0xf00d. Manual/preset/Kelvin unqualified. Original joint OLS failure unchanged.',
        'bindings':{p:sha(ROOT/p)for p in paths},'scenes':scenes,'base_options':dict(gm['base_options'],gamm=[1,1,0,0,0,0]),'wasm_sha256':gm['wasm_sha256'],
        'output_dir':str(OUT.relative_to(ROOT)),
        'gates':{'control':'Explicit as-shot userMul [R,G,B,G] vs cameraWB G: exact working payload and saved native strips. Any matrix/path difference stops.',
                 'source':'Qualified sourceWBmode0; bounded parser sourcef00d and asShot exact; immutable RAF before/after.',
                 'repeat':'W working v2 container, fullnative SHA and native strips byte-exact in independent page/worker.',
                 'codec':'Browser/Node/Python version2 strict sourceAuto contract and provenance, no old model accepts it.'},
        'limits':'Input-path qualification only. No claim of Fuji rendering, AWB-algorithm parity or better predictions. G and W remain separate paired research arms.'})
    print('W manifest SHA',sha(MANIFEST))

def verify(expected):
    need(sha(MANIFEST)==expected,'Manifest drift');m=read(MANIFEST)
    for p,h in m['bindings'].items():need(sha(ROOT/p)==h,'Bound file drift '+p)
    for p in (MANIFEST,Path(__file__).resolve(),ROOT/'scripts/export-provia-input-v2-wb-pilot.mjs'):committed(p)
    complete=read(OUT/'complete.json');need(complete['completed']and complete['manifest_sha256']==expected,'Incomplete/stale W run')
    need([r['name']for r in complete['scenes']]==[r['name']for r in m['scenes']],'Coverage mismatch')
    results=[]
    for s,r in zip(m['scenes'],complete['scenes']):
        need(r['shoot']==s['shoot'] and set(r['arms'])=={'control','W','W-repeat'},'Wrong coverage')
        need(sha(ROOT/s['g_file'])==s['g_sha256'] and sha(ROOT/s['g_strips_file'])==s['g_strips_sha256'],'G changed')
        g=V.load_fjlrg_v2(ROOT/s['g_file'],expected_contract=V.CONTRACT_ASSHOT,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=m['wasm_sha256'])
        loaded=[]
        for arm,rec in r['arms'].items():
            p=ROOT/rec['file'];strip=ROOT/rec['strips']['file']
            need(p.parent==strip.parent==OUT/s['shoot'] and sha(p)==rec['sha256'] and sha(strip)==rec['strips']['sha256'],'Output hash/path changed')
            need(rec['native_size']==s['g_native_size'],'Native shape mismatch')
            need(rec['strips']['rows']==list(range(64,rec['native_size'][1],128)) and rec['strips']['width']==rec['native_size'][0] and rec['strips']['channels']==3 and rec['strips']['dtype']=='uint16-le','Strip layout')
            need(strip.stat().st_size==len(rec['strips']['rows'])*rec['native_size'][0]*6,'Strip size')
            need(rec['source_coefficients']['autoRgbg']==s['auto_rgbg'] and rec['source_coefficients']['asShotCamMul']==s['cam_mul'],'Source coefficient report drift')
            if arm=='control':
                raw=np.fromfile(p,dtype='<u2');need(raw.size==g.rgb_uint16.size and np.array_equal(raw.reshape(g.rgb_uint16.shape),g.rgb_uint16),'Explicit as-shot does not reproduce cameraWB G')
                need(sha(strip)==s['g_strips_sha256'],'Native explicit as-shot strips differ')
            else:
                f=V.load_fjlrg_v2(p,expected_contract=V.CONTRACT_AUTOWB,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=m['wasm_sha256'])
                need(f.width==g.width and f.height==g.height,'Working geometry differs')
                need(f.provenance['wb_source']['levels_rgbg']==s['auto_rgbg'] and f.provenance['source_wb_fine_tune_raw']==s['source_wb_fine_tune_raw'] and f.provenance['source_kelvin']==s['source_kelvin'],'W source metadata mismatch')
                need(f.provenance['capture_dynamic_range']==s['capture_dynamic_range'] and f.provenance['decoded_native_size']==s['g_native_size'],'W DR/native metadata mismatch')
                need(f.provenance==rec['provenance'],'Recorded W sidecar differs');loaded.append(f)
        a,b=r['arms']['W'],r['arms']['W-repeat']
        need(a['sha256']==b['sha256']and a['native_sha256']==b['native_sha256']and a['strips']['sha256']==b['strips']['sha256'],'W repeat differs')
        need(np.array_equal(loaded[0].rgb_uint16,loaded[1].rgb_uint16),'Python repeat differs')
        results.append({'name':s['name'],'shoot':s['shoot'],'g_sha256':s['g_sha256'],'w_sha256':a['sha256'],'source_auto_rgbg':s['auto_rgbg'],'explicit_asshot_reproduces_g':True,'native_and_working_repeat_exact':True,'passed':True})
    dump(REPORT,{'passed':True,'manifest_sha256':expected,'complete_sha256':sha(OUT/'complete.json'),'scenes':results,'scope':m['scope'],'limits':m['limits']})
    print('Independent W input pilot verification PASS',len(results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify']);p.add_argument('--manifest-sha');a=p.parse_args()
    if a.command=='prepare':prepare()
    else:need(bool(a.manifest_sha),'Manifest SHA required');verify(a.manifest_sha)
