"""Prepare/verify gamma-only G exports for the existing583 train sources.
No target reads, model execution, training, W expansion or protected media.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
import fjlrg_v2 as V
R=Path(__file__).resolve().parent.parent;D=R/'training/diagnostics'
M=D/'provia-v2-train-export-manifest.json';O=D/'provia-v2-train-export-verification.json'
OUT=R/'training/checkpoints/provia-input-v2-train-g'
G_REPORT=D/'provia-v2-fullnative-pilot-verification.json'
INVENTORY=D/'provia-v2-source-inventory.json'
def need(b,m):
    if not b:raise ValueError(m)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,d):
    with p.open('x')as f:f.write(json.dumps(d,indent=2,allow_nan=False)+'\n')
def prepare():
    g=read(G_REPORT);need(g['passed']and len(g['scenes'])==24,'Independent P1b pass required')
    inventory=read(INVENTORY);need(inventory['completed']and len(inventory['rows'])==583,'Source-only inventory incomplete')
    gm=read(D/'provia-v2-fullnative-pilot-manifest.json');need(g['manifest_sha256']==sha(D/'provia-v2-fullnative-pilot-manifest.json'),'Stale P1b report')
    old_pool=read(D/'provia-color-run-manifest.json');need(sha(D/'provia-color-run-manifest.json')=='d6c04efd70df63aa38a9f2f08e0b291f858d5085a9670ffb7d3f1b3f4c6cf254','Train allowlist drift')
    need({(s['name'],s['shoot'])for s in inventory['rows']}=={(s['name'],s['shoot'])for s in old_pool['scenes']},'Non-allowlisted inventory')
    rows={s['name']:dict(s,capture_dynamic_range=s['captureDynamicRange'])for s in inventory['rows']}
    for s in g['scenes']:
        if s['name']in rows:rows[s['name']]['expected_pilot_g_sha256']=s['linear_v2_sha256']
    canaries=['DSCF0523.RAF','DSCF0964.RAF','DSCF0514.RAF']
    scenes=[rows[n]for n in canaries]+[s for n,s in rows.items()if n not in canaries]
    need(all('expected_pilot_g_sha256'in s for s in scenes[:3]),'Missing canary anchors')
    paths=['training/verify_provia_v2_train_exports.py','scripts/export-provia-input-v2-train.mjs','scripts/lib/decode-provia-v2-research.mjs','scripts/lib/raf-wb-coefficients.mjs','scripts/lib/fjlrg-v2.mjs','training/fjlrg_v2.py',
           str(G_REPORT.relative_to(R)),str(INVENTORY.relative_to(R)),'training/diagnostics/provia-v2-source-inventory-manifest.json','training/diagnostics/provia-color-run-manifest.json']
    paths+=['node_modules/libraw-wasm/dist/'+n for n in('index.js','worker.js','libraw.js','libraw.wasm')]
    dump(M,{'scope':__doc__,'bindings':{p:sha(R/p)for p in paths},'scenes':scenes,'base_options':dict(gm['base_options'],gamm=[1,1,0,0,0,0]),'wasm_sha256':gm['wasm_sha256'],'output_dir':str(OUT.relative_to(R)),
      'canaries':canaries,'gate':'First3 Auto/Kelvin/preset helper outputs exactly match independently verified P1b G containers. All other F24 overlaps also match. Every source hashed before/after, strict G codec, native/working geometry and sourceWB metadata verified. Stop on any mismatch.',
      'limits':'G gamma-only. Source as-shot WB and fine-tune retained. Not an R11 input, no model is selected or fitted by this export. W remains restricted to its separate qualified cohort.'})
    print('G train export manifest',sha(M))
def verify(expected):
    need(sha(M)==expected,'Manifest mismatch');m=read(M)
    for p,h in m['bindings'].items():need(sha(R/p)==h,'Binding drift '+p)
    c=read(OUT/'complete.json');need(c['completed']and c['manifest_sha256']==expected,'Incomplete/stale run')
    need([r['name']for r in c['scenes']]==[r['name']for r in m['scenes']],'Coverage mismatch')
    results=[]
    for s,r in zip(m['scenes'],c['scenes']):
        need(r['shoot']==s['shoot'],'Shoot mismatch');p=R/r['file'];strip=R/r['strips']['file']
        need(p.parent==strip.parent==OUT/s['shoot']and sha(p)==r['sha256']and sha(strip)==r['strips']['sha256'],'Export/strip hash or path mismatch')
        f=V.load_fjlrg_v2(p,expected_contract=V.CONTRACT_ASSHOT,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=m['wasm_sha256'])
        need([f.width,f.height]==s['fjlrg_wh'],'Canonical geometry changed')
        pr=f.provenance
        need(pr['declared_options']==m['base_options'] and pr['wb_source']['levels_rgbg']==s['cam_mul'],'Options or source gains drift')
        need(pr['source_wb_mode']==s['source_wb_mode'] and pr['source_wb_fine_tune_raw']==s['source_wb_fine_tune_raw'] and pr['source_kelvin']==s['source_kelvin'],'Source WB metadata drift')
        need(pr['capture_dynamic_range']==s['captureDynamicRange'] and pr['decoded_native_size']==r['native_size'],'DR/native metadata drift')
        need(sha(R/'calibration-input'/s['shoot']/s['name'])==s['raf_sha256'],'Current source RAF drift')

        if 'expected_pilot_g_sha256'in s:need(r['sha256']==s['expected_pilot_g_sha256'],'P1b canary drift')
        need(r['source_coefficients']['asShotCamMul']==s['cam_mul']and r['source_coefficients']['autoRgbg']==s['auto_rgbg'],'WB metadata mismatch')
        need(r['strips']['rows']==list(range(64,r['native_size'][1],128))and strip.stat().st_size==len(r['strips']['rows'])*r['native_size'][0]*6,'Strip layout mismatch')
        results.append({'name':s['name'],'shoot':s['shoot'],'file':r['file'],'sha256':r['sha256'],'sidecar_sha256':sha(V.sidecar_path(p)),'source_raf_sha256':s['raf_sha256'],'passed':True})
    dump(O,{'passed':True,'manifest_sha256':expected,'complete_sha256':sha(OUT/'complete.json'),'scenes':results,'scope':m['scope'],'limits':m['limits']});print('Independent G training export verification PASS',len(results))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify']);p.add_argument('--manifest-sha');a=p.parse_args()
    if a.command=='prepare':prepare()
    else:need(bool(a.manifest_sha),'Manifest SHA required');verify(a.manifest_sha)
