"""Freeze455FIT source-only WB input qualification rows; no targets or training.

Revalidate G identity1, qualified W9 and profile4 reuse; capture441 new
sources with explicit as-shot C / neutral N / independent neutral R.
Geometry remains the existing G geometry in every arm. Unsupported12
presets and ALL development/protected pixel inputs are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matched_gamma_data as D
import audit_target_wb_headers as H
import neutral_wb_domain_plan as N

HEADER_SHA='16fcb23a2d4fb48aad7f12251c00015f4231c1373214618b264a21b7eb889ddc'
MANIFEST=D.DIAG/'neutral-wb-fit-export-manifest.json'
OUT=D.CKPT/'neutral-wb-fit-export'
SALT='neutral-wb-fit-export-canary-v1\n'


def read(p):return json.loads(p.read_text())


def build():
    header=H.load(HEADER_SHA);alignment=read(H.A.REPORT)
    header_proof=D.DIAG/'target-wb-header-verification.json'
    assert read(header_proof)['verified'] and read(header_proof)['report_sha256']==D.sha(H.REPORT)
    gp=D.CKPT/'provia-input-v2-train-g/complete.json';gvp=D.DIAG/'provia-v2-train-export-verification.json'
    wp=D.CKPT/'provia-input-v2-wb-pilot/complete.json';wvp=D.DIAG/'provia-v2-wb-pilot-verification.json'
    pp=D.CKPT/'profile-rgb-sentinels/complete.json';pr=D.DIAG/'profile-rgb-sentinel-report.json';pvp=D.DIAG/'profile-rgb-sentinel-verification.json'
    g,gv,w,wv,p,pv=map(read,(gp,gvp,wp,wvp,pp,pvp))
    assert g['completed'] and gv['passed'] and gv['complete_sha256']==D.sha(gp)
    assert w['completed'] and wv['passed'] and wv['complete_sha256']==D.sha(wp)
    assert p['completed'] and pv['verified'] and pv['complete_sha256']==D.sha(pp) and pv['report_sha256']==D.sha(pr)
    source={s['name']:s for s in header['scenes'] if s['role']=='fit'}
    gb={s['name']:s for s in g['scenes']};gvb={s['name']:s for s in gv['scenes']}
    wb={s['name']:s for s in w['scenes'] if s['name'] in source}
    pb={s['name']:s for s in p['scenes'] if s['name'] in source}
    assert len(source)==467 and len(wb)==9 and len(pb)==4
    plans=N.plan_rows(list(source.values()),alignment['profiles'],qualified_w=set(wb),qualified_p=set(pb))
    eligible=N.fit_rows(plans);assert len(eligible)==455 and N.summarize(plans)['fit_new_decodes']==441
    scenes=[]
    for plan in eligible:
        s=plan['source'];old=gb[s['name']];proof=gvb[s['name']]
        assert proof['passed'] and old['shoot']==s['shoot'] and proof['source_raf_sha256']==s['raf_sha256']
        assert proof['sha256']==old['sha256'] and old['source_coefficients']['asShotCamMul']==s['cam_mul'] and old['source_coefficients']['autoRgbg']==s['auto_rgbg']
        scene={**s,'kind':plan['kind'],'contract':plan['contract'],'requires_new_decode':plan['requires_new_decode'],
               'reuse_source':plan['reuse_source'],'declared_coeff_source':plan['declared_coeff_source'],
               'declared_levels_rgbg':plan['declared_levels_rgbg'],'policy_basis':plan['policy_basis'],
               'capture_dynamic_range':s['captureDynamicRange'],
               'expected_native_size':old['native_size'],'expected_working_size':[old['width'],old['height']],
               'G':{k:old[k] for k in ('file','sha256','native_sha256','working_payload_sha256','strips')}}
        scene['G']['sidecar_sha256']=proof['sidecar_sha256']
        if s['name'] in wb:scene['reuse']={'kind':'W','source_record':wb[s['name']]}
        elif s['name'] in pb:scene['reuse']={'kind':'P','source_record':pb[s['name']]}
        elif plan['kind']=='G':scene['reuse']={'kind':'G','source_record':old}
        else:scene['reuse']=None
        scenes.append(scene)
    # Source-name hash only, no pixel/metric selection. New W/P canaries run first.
    new=[s for s in scenes if s['requires_new_decode']]
    canaries=[min((s for s in new if s['kind']==kind),key=lambda s:hashlib.sha256((SALT+s['name']).encode()).hexdigest())['name'] for kind in ('W','P')]
    scenes.sort(key=lambda s:(0 if s['name'] in canaries else 1,s['name']))
    files=[Path(__file__),D.TRAIN/'neutral_wb_domain_plan.py',D.TRAIN/'test_neutral_wb_domain_plan.py',
           D.TRAIN/'fjlrg_v2.py',D.TRAIN/'fjnrgb_profile.py',D.TRAIN/'verify_neutral_wb_fit_export.py',D.TRAIN/'test_neutral_wb_fit_scope.py',
           D.ROOT/'scripts/export-neutral-wb-fit.mjs',D.ROOT/'scripts/lib/neutral-wb-artifacts.mjs',D.ROOT/'scripts/test-neutral-wb-artifacts.mjs',
           D.ROOT/'scripts/lib/decode-provia-v2-research.mjs',D.ROOT/'scripts/lib/decode-fjnrgb-profile-research.mjs',
           D.ROOT/'scripts/lib/fjlrg-v2.mjs',D.ROOT/'scripts/lib/fjnrgb-profile.mjs',D.ROOT/'scripts/lib/raf-wb-coefficients.mjs',
           H.MANIFEST,H.REPORT,header_proof,H.A.REPORT,gp,gvp,wp,wvp,pp,pr,pvp,
           D.DIAG/'fjnrgb-profile-codec-qualification.json',D.DIAG/'manual-kelvin-wb-fullnative-v2-verification.json']
    files += [D.ROOT/'node_modules/libraw-wasm/dist'/n for n in ('index.js','worker.js','libraw.js','libraw.wasm')]
    return {'scope':__doc__.strip(),'header_manifest_sha256':HEADER_SHA,'scenes':scenes,
            'canaries':canaries,'counts':N.summarize(plans),'output_dir':str(OUT.relative_to(D.ROOT)),
            'wasm_sha256':D.sha(D.ROOT/'node_modules/libraw-wasm/dist/libraw.wasm'),
            'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files},
            'qualification':'Every new C full-native hash, working payload and strips must equal old G. N/R native, payload, strips, strict container and sidecar exact. All source/camera/WB/options/geometry guards; any failure stops. Reuse13 revalidated against frozen proof and bytes; G1 byte-hash reference only.',
            'staging':'First2source-hash-selected canaries, then independent stored-byte verifier before441bulk continues. Resume only hash-verified sealed arms; never overwrite prior files/partial evidence. No skipping failed scenes.',
            'reads':'455FIT RAFs and own G/W/P artifact bytes ONLY. No target arrays/TIFFs, development/protected media, training or model inference. Parent metadata proof JSON only may describe older development artifacts; never open those files.',
            'limits':'Input provenance/determinism qualification, not Fuji RGB accuracy. Keep old geometry both arms. Native hashes are browser-reported; saved working/strip/container bytes independently rehashed. No current app/R11/canonical edits.'}


def load(expected):
    assert expected and D.sha(MANIFEST)==expected and D.committed(MANIFEST)
    m=read(MANIFEST)
    for rel,h in m['bindings'].items():
        assert D.sha(D.ROOT/rel)==h,'Binding changed: '+rel
        if not rel.startswith(('node_modules/','training/checkpoints/')):assert D.committed(D.ROOT/rel),'Uncommitted binding: '+rel
    assert m==build(), 'Source-only export plan drift'
    return m


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','check'));ap.add_argument('--manifest-sha');a=ap.parse_args()
    if a.command=='prepare':D.write_once_json(MANIFEST,build());print('NEUTRAL_WB_FIT_EXPORT_MANIFEST',D.sha(MANIFEST),flush=True)
    else:load(a.manifest_sha);print('NEUTRAL_WB_FIT_EXPORT_MANIFEST_VERIFIED',flush=True)
