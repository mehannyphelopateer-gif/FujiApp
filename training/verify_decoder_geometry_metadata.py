"""Validate stored numeric geometry readback and its eight-FIT source guards.

This checks saved provenance/coverage/geometry; it is not a second LibRaw
implementation or an independent observation of the decoder's internals.
"""
from collections import Counter
import json
from pathlib import Path
import matched_gamma_data as D
import diagnose_geometry_registration as A

SHA='60e4f3292c0414a001ca677f5d8733ea487eb99e8f8a1679b46a14644028e712'


def main():
    manifest=D.DIAG/'decoder-geometry-metadata-manifest.json'
    report=D.DIAG/'decoder-geometry-metadata-report.json'
    assert D.sha(manifest)==SHA and D.committed(manifest)
    m=json.loads(manifest.read_text());r=json.loads(report.read_text())
    for rel,h in m['bindings'].items():assert D.sha(D.ROOT/rel)==h
    parent=A.load('af7ee01bb3e1a9460cc574b441d12ca3b60037d90be66cb37bd5046a0d6ead7a')
    assert r['completed'] and r['manifest_sha256']==SHA and r['limits']==m['limits']
    assert [s['name'] for s in m['scenes']]==[s['name'] for s in parent['scenes']]==[s['name'] for s in r['rows']]
    combos=Counter()
    for s,x,p in zip(m['scenes'],r['rows'],parent['scenes']):
        assert x['role']==s['role']==p['role']=='fit'
        assert x['source_raf_sha256']==s['raf_sha256']==p['raf_sha256']
        assert x['phase']=='after-open-before-any-pixel-method'
        assert x['source_guard']=={'camera_model':'X100VI','fujiWB':s['source_wb_mode'],'cam_mul':s['cam_mul']}
        g=x['geometry'];assert not g['not_returned']
        assert [g['width'],g['height']]==s['G_native_size']==p['G_native_size']
        for key in ('width','height','raw_width','raw_height','top_margin','left_margin','flip','iwidth','iheight','raw_pitch'):
            assert type(g[key]) is int and g[key]>=0
        assert g['left_margin']+g['iwidth']<=g['raw_width']
        assert g['top_margin']+g['iheight']<=g['raw_height']
        assert g['raw_inset_crops'][1]=={'cleft':65535,'ctop':65535,'cwidth':0,'cheight':0}
        # The sentinel rectangle is retained as unset, never silently selected.
        combos[json.dumps({k:g[k] for k in ('iwidth','iheight','top_margin','left_margin','raw_width','raw_height','raw_inset_crops')},sort_keys=True)]+=1
    assert D.sha(D.R11_PATH)==D.R11_SHA
    out={'verified':True,'manifest_sha256':SHA,'report_sha256':D.sha(report),
         'verifier_sha256':D.sha(Path(__file__)),'rows':8,
         'geometry_groups':[{'geometry':json.loads(k),'n':n} for k,n in sorted(combos.items())],
         'limits':__doc__.strip()+' No crop/warp selected or applied; all eight dimensions agree with old processed G dimensions, but metadata is pre-processing readback.',
         'R11_sha256':D.R11_SHA}
    D.write_once_json(D.DIAG/'decoder-geometry-metadata-verification.json',out)
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
