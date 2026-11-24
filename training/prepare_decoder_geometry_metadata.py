"""Freeze actual LibRaw metadata queries for the SAME eight FIT geometry sources.

No imageData/rawImageData/thumbnail access, target reads, model or transform.
Metadata is captured after open, BEFORE image processing, and may therefore
not describe the eventual processed crop. Missing fields remain unknown.
"""
import json
from pathlib import Path
import matched_gamma_data as D
import diagnose_geometry_registration as A


def main():
    parent=A.load('af7ee01bb3e1a9460cc574b441d12ca3b60037d90be66cb37bd5046a0d6ead7a')
    inventory=json.loads((D.DIAG/'provia-v2-source-inventory.json').read_text())
    rows=inventory['scenes'] if 'scenes' in inventory else inventory['rows']
    by={r['name']:r for r in rows}
    scenes=[]
    for s in parent['scenes']:
        r=by[s['name']]
        assert s['role']==r['role']=='fit' and s['raf_sha256']==r['raf_sha256']
        sidecar=Path(str(D.ROOT/s['input_file'])+'.json')
        assert D.sha(sidecar)==s['input_sidecar_sha256']
        options=json.loads(json.loads(sidecar.read_text())['options_json'])
        assert options['gamm']==[1,1,0,0,0,0] and options['useCameraWb'] is True
        scenes.append({k:s[k] for k in ('name','shoot','role','session','raf_sha256','G_native_size')})
        scenes[-1].update(cam_mul=r['cam_mul'],source_wb_mode=r['source_wb_mode'],declared_options=options)
    files=[Path(__file__),D.ROOT/'scripts/query-libraw-geometry-metadata.mjs',
           D.ROOT/'scripts/lib/libraw-geometry-metadata.mjs',D.ROOT/'scripts/test-libraw-geometry-metadata.mjs',
           A.MANIFEST,D.DIAG/'provia-v2-source-inventory.json']
    files += [D.ROOT/'node_modules/libraw-wasm/dist'/n for n in ('index.js','worker.js','libraw.js','libraw.wasm','index.d.ts')]
    m={'scope':__doc__.strip(),'scenes':scenes,
       'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files},
       'output_file':'training/diagnostics/decoder-geometry-metadata-report.json',
       'limits':'Only8FIT RAFs; hash before/after. decoder.open then metadata(true), never imageData/rawImageData/thumbnail. Save numeric geometry whitelist plus camera/WB/cam source guards. No target, development/protected media, weights, crop selection or pipeline changes.'}
    D.write_once_json(D.DIAG/'decoder-geometry-metadata-manifest.json',m)
    print('DECODER_GEOMETRY_METADATA_MANIFEST',D.sha(D.DIAG/'decoder-geometry-metadata-manifest.json'),flush=True)


if __name__=='__main__':main()
