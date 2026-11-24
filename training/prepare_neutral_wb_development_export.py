"""Freeze internal-development INPUT qualification for a future paired WB study.

116 development sources from the existing session split: G37 identity,
W2 previously-qualified reuse,77novel W3/P74 with C/N/R qualification.
No target arrays, model inference, metrics or training. This is not an
acceptance set; protected K/monitor/finalHeldOut/T8/dr100Eval stay unread.
"""
import argparse
import hashlib
from pathlib import Path
import matched_gamma_data as D
import audit_target_wb_headers as H
import neutral_wb_domain_plan as N
import prepare_neutral_wb_fit_export as B

MANIFEST=D.DIAG/'neutral-wb-development-export-manifest.json'
OUT=D.CKPT/'neutral-wb-development-export'
read=B.read
HEADER_SHA=B.HEADER_SHA
SALT='neutral-wb-development-input-canary-v1\n'


def build():
    header=H.load(HEADER_SHA);alignment=read(H.A.REPORT)
    hp=D.DIAG/'target-wb-header-verification.json'
    assert read(hp)['verified'] and read(hp)['report_sha256']==D.sha(H.REPORT)
    gp=D.CKPT/'provia-input-v2-train-g/complete.json';gvp=D.DIAG/'provia-v2-train-export-verification.json'
    wp=D.CKPT/'provia-input-v2-wb-pilot/complete.json';wvp=D.DIAG/'provia-v2-wb-pilot-verification.json'
    pp=D.DIAG/'profile-rgb-sentinel-report.json';pvp=D.DIAG/'profile-rgb-sentinel-verification.json'
    g,gv,w,wv,pv=map(read,(gp,gvp,wp,wvp,pvp))
    assert g['completed'] and gv['passed'] and gv['complete_sha256']==D.sha(gp)
    assert w['completed'] and wv['passed'] and wv['complete_sha256']==D.sha(wp)
    assert pv['verified'] and pv['report_sha256']==D.sha(pp)
    source={s['name']:s for s in header['scenes'] if s['role']=='development'}
    gb={s['name']:s for s in g['scenes']};gvb={s['name']:s for s in gv['scenes']}
    wb={s['name']:s for s in w['scenes'] if s['name'] in source}
    assert len(source)==116 and len(wb)==2
    plans=N.plan_rows(list(source.values()),alignment['profiles'])
    assert N.summarize(plans)['by_role_kind']=={'development/G':37,'development/P':74,'development/W':5}
    assert not N.fit_rows(plans), 'Development must never enter the FIT export filter'
    scenes=[]
    for plan in plans:
        s=plan['source'];old=gb[s['name']];proof=gvb[s['name']]
        assert s['role']=='development' and not plan['excluded'] and not plan['export_eligible']
        assert proof['passed'] and proof['source_raf_sha256']==s['raf_sha256'] and old['shoot']==s['shoot']
        assert proof['sha256']==old['sha256'] and old['source_coefficients']['asShotCamMul']==s['cam_mul'] and old['source_coefficients']['autoRgbg']==s['auto_rgbg']
        reuse={'kind':'W','source_record':wb[s['name']]} if s['name'] in wb else {'kind':'G','source_record':old} if plan['kind']=='G' else None
        assert not reuse or reuse['kind']==plan['kind']
        scene={**s,'kind':plan['kind'],'contract':plan['contract'],'fit_export_eligible':False,
               'purpose':'development-input-qualification-only','requires_new_decode':reuse is None,
               'reuse':reuse,'declared_coeff_source':plan['declared_coeff_source'],'declared_levels_rgbg':plan['declared_levels_rgbg'],
               'capture_dynamic_range':s['captureDynamicRange'],'expected_native_size':old['native_size'],
               'expected_working_size':[old['width'],old['height']],
               'G':{k:old[k] for k in ('file','sha256','native_sha256','working_payload_sha256','strips')}}
        scene['G']['sidecar_sha256']=proof['sidecar_sha256'];scenes.append(scene)
    new=[s for s in scenes if s['requires_new_decode']];assert len(new)==77
    canaries=[min((s for s in new if s['kind']==kind),key=lambda s:hashlib.sha256((SALT+s['name']).encode()).hexdigest())['name'] for kind in ('W','P')]
    scenes.sort(key=lambda s:(0 if s['name'] in canaries else 1,s['name']))
    files=[Path(__file__),D.TRAIN/'neutral_wb_domain_plan.py',D.TRAIN/'test_neutral_wb_domain_plan.py',
           D.TRAIN/'fjlrg_v2.py',D.TRAIN/'fjnrgb_profile.py',D.TRAIN/'verify_neutral_wb_development_export.py',
           D.TRAIN/'test_neutral_wb_development_scope.py',D.ROOT/'scripts/export-neutral-wb-development.mjs',
           D.ROOT/'scripts/lib/neutral-wb-development-artifacts.mjs',D.ROOT/'scripts/test-neutral-wb-development-artifacts.mjs',
           D.ROOT/'scripts/lib/decode-provia-v2-research.mjs',D.ROOT/'scripts/lib/decode-fjnrgb-profile-development.mjs',
           D.ROOT/'scripts/test-decode-fjnrgb-profile-development.mjs',
           D.ROOT/'scripts/lib/fjlrg-v2.mjs',D.ROOT/'scripts/lib/fjnrgb-profile.mjs',D.ROOT/'scripts/lib/raf-wb-coefficients.mjs',
           H.MANIFEST,H.REPORT,hp,H.A.REPORT,gp,gvp,wp,wvp,pp,pvp]
    files += [D.ROOT/'node_modules/libraw-wasm/dist'/n for n in ('index.js','worker.js','libraw.js','libraw.wasm')]
    return {'scope':__doc__.strip(),'scenes':scenes,'canaries':canaries,'output_dir':str(OUT.relative_to(D.ROOT)),
            'counts':{'development':116,'identity_G':37,'reuse_W':2,'new_W':3,'new_P':74},
            'wasm_sha256':D.sha(D.ROOT/'node_modules/libraw-wasm/dist/libraw.wasm'),
            'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files},
            'qualification':'Same C vs old G full-native hash/working/strip identity and N/R exact repeats, strict source/WASM/options/geometry/provenance; truthful development-only role. Fresh W/P canaries must change native and working pixels; any failure stops, no replacements.',
            'reads':'Only116explicit internal-development RAFs/G/W/input artifacts. No TIFF/target-cache arrays, fit-source media, K/monitor/finalHeldOut/T8/dr100Eval, models or scoring.',
            'limits':'Input qualification only, no independent acceptance/Fuji colour accuracy or training launch. Geometry unchanged. Two canaries plus39reuses verified before77bulk continues; no role relabelling, no changed source/target/canonical files.'}


def load(expected):
    assert expected and D.sha(MANIFEST)==expected and D.committed(MANIFEST)
    m=read(MANIFEST)
    for rel,h in m['bindings'].items():
        assert D.sha(D.ROOT/rel)==h,'Binding changed: '+rel
        if not rel.startswith(('node_modules/','training/checkpoints/')):assert D.committed(D.ROOT/rel),'Uncommitted binding: '+rel
    assert m==build(), 'Development input plan drift'
    return m


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','check'));ap.add_argument('--manifest-sha');a=ap.parse_args()
    if a.command=='prepare':D.write_once_json(MANIFEST,build());print('NEUTRAL_WB_DEVELOPMENT_MANIFEST',D.sha(MANIFEST),flush=True)
    else:load(a.manifest_sha);print('NEUTRAL_WB_DEVELOPMENT_MANIFEST_VERIFIED',flush=True)
