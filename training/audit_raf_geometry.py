"""FIT-only RAF geometry HEADER diagnostic; no pixel decode or model work.

Opaque source bytes are streamed only for frozen SHA identity. The parser
interprets bounded RAF directory/header fields; JPEG/TIFF/image payloads are
never interpreted. Crop-size tags include borders, not automatic Fuji FoV.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import matched_gamma_data as D
import audit_target_wb_headers as H

HEADER_SHA='16fcb23a2d4fb48aad7f12251c00015f4231c1373214618b264a21b7eb889ddc'
MANIFEST=D.DIAG/'raf-geometry-fit-manifest.json'
REPORT=D.DIAG/'raf-geometry-fit-report.json'
PROGRESS=D.CKPT/'raf-geometry-fit-progress.json'
CODE=Path(__file__).resolve()
CANARIES=('DSCF0464.RAF','DSCF0809.RAF','DSCF1011.RAF','DSCF1054.RAF')
PROTOCOL=D.DIAG/'raf-geometry-protocol-source.json'


def source_path(scene):
    D.need(scene['role']=='fit' and scene['shoot'].startswith('Shoot ') and scene['shoot'][6:].isdigit()
           and scene['name'].startswith('DSCF') and scene['name'][4:8].isdigit() and scene['name'][8:]=='.RAF','Unapproved source/path')
    return D.ROOT/'calibration-input'/scene['shoot']/scene['name']


def prepare():
    parent=H.load(HEADER_SHA)
    selected={r['name']:r for r in parent['scenes'] if r['role']=='fit'}
    D.need(len(selected)==467 and set(CANARIES)<=set(selected),'Fit/canary coverage changed')
    old_path=D.CKPT/'provia-input-v2-train-g/complete.json';old=json.loads(old_path.read_text())
    proof_path=D.DIAG/'provia-v2-train-export-verification.json';proof=json.loads(proof_path.read_text())
    D.need(proof['passed'] and proof['complete_sha256']==D.sha(old_path),'Native size source unverified')
    native={r['name']:r for r in old['scenes']}
    target_path=D.DIAG/'provia-color-data-audit.json';target=json.loads(target_path.read_text())
    targets={r['name']:r for r in target['per_scene']}
    names=list(CANARIES)+sorted(set(selected)-set(CANARIES));scenes=[]
    for name in names:
        r=selected[name];t=targets[name];n=native[name]
        D.need(t['role']=='fit' and t['shoot']==n['shoot']==r['shoot'] and t['tiff_sha256']==r['tiff_sha256'],'Static target/native provenance differs')
        scenes.append({k:r[k] for k in ('name','shoot','role','session','raf_sha256','source_wb_mode','captureDynamicRange')})
        scenes[-1].update(G_native_size=n['native_size'],G_working_size=[n['width'],n['height']],
                         target_stored_shape=t['shape'],target_orientation=t['orientation'])
    files=[CODE,D.TRAIN/'test_audit_raf_geometry.py',D.TRAIN/'raf_crop_header.py',D.TRAIN/'test_raf_crop_header.py',
           H.MANIFEST,old_path,proof_path,target_path,PROTOCOL]
    m={'scope':__doc__.strip(),'parent_header_manifest_sha256':HEADER_SHA,'scenes':scenes,'canaries':list(CANARIES),
       'bindings':{str(p.relative_to(D.ROOT)):D.sha(p) for p in files},'protocol_source':json.loads(PROTOCOL.read_text()),
       'reads':'467FIT RAFs only; full opaque SHA streams then<=4096interpreted header bytes per file. Four metadata canaries first, then all remaining. No JPEG/TIFF/image decoder, target caches/pixels, development/protected media or model inference.',
       'interpretation':'Raw tag values and size-class/compression/session/DR/WB cross-tabs only. Missing geometry remains unknown. Borders/camera crop/lens warp are not equated with visible Fuji framing. No crop/resize/metric/input/model change.',
       'stop':'Source hash drift/parser or budget error stops without a complete report. No source replacement, fallback, skipped failure or permissive retry.'}
    D.write_once_json(MANIFEST,m);print('RAF_GEOMETRY_FIT_MANIFEST',D.sha(MANIFEST),flush=True)


def load(expected):
    D.need(isinstance(expected,str) and MANIFEST.exists() and D.sha(MANIFEST)==expected and D.committed(MANIFEST),'Unpinned/uncommitted geometry scope')
    m=json.loads(MANIFEST.read_text())
    for rel,h in m['bindings'].items():
        p=D.ROOT/rel;D.need(D.sha(p)==h,'Bound metadata/code changed: '+rel)
        if 'checkpoints/' not in rel:D.need(D.committed(p),'Uncommitted binding')
    parent=H.load(HEADER_SHA);by={r['name']:r for r in parent['scenes'] if r['role']=='fit'}
    D.need(len(m['scenes'])==467 and len({r['name'] for r in m['scenes']})==467 and {r['name'] for r in m['scenes']}==set(by),'Wrong fit allowlist')
    D.need(m['canaries']==list(CANARIES) and [r['name'] for r in m['scenes'][:4]]==list(CANARIES),'Canary order changed')
    for r in m['scenes']:
        D.need(all(r[k]==by[r['name']][k] for k in ('name','shoot','role','session','raf_sha256','source_wb_mode','captureDynamicRange')),'Source metadata differs')
        source_path(r)
    return m


def summarize(rows):
    counts=Counter()
    for row in rows:
        key=(tuple(row['G_native_size']),str(row['geometry']['compression']),row['session'],row['captureDynamicRange'],row['source_wb_mode'])
        counts[key]+=1
    return [{'G_native_size':list(native),'compression_code':comp,'session':session,'captureDR':dr,'sourceWB':wb,'n':n}
            for (native,comp,session,dr,wb),n in sorted(counts.items())]


def geometry_summary(rows):
    counts=Counter()
    for row in rows:
        directories=[{'directory_index':d['index'],'raw_tags':d['tags']} for d in row['geometry']['directories']]
        signature=json.dumps(directories,sort_keys=True,separators=(',',':'))
        counts[(tuple(row['G_native_size']),str(row['geometry']['compression']),row['target_orientation'],signature)]+=1
    return [{'G_native_size':list(native),'compression_code':None if comp=='None' else int(comp),
             'target_orientation':orientation,'RAF_directories':json.loads(signature),'n':n}
            for (native,comp,orientation,signature),n in sorted(counts.items())]


def run(expected):
    import raf_crop_header as R
    m=load(expected);D.need(not PROGRESS.exists() and not REPORT.exists(),'Prior geometry output exists')
    result={'manifest_sha256':expected,'completed':False,'rows':[],'scope':m['scope'],'interpretation':m['interpretation']}
    for index,scene in enumerate(m['scenes']):
        p=source_path(scene)
        try:
            before=p.stat();h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
                D.need(h.hexdigest()==scene['raf_sha256'],'Source SHA differs')
                f.seek(0);geometry=R.read_raf_crop_header(f)
            after=p.stat();D.need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Source changed while reading')
            D.need(geometry['bytes_read']<=4096,'Metadata budget exceeded')
            row={**scene,'geometry':geometry,'source_sha256_verified':h.hexdigest()}
            result['rows'].append(row)
            PROGRESS.write_text(json.dumps(result,indent=2)+'\n')
            print('RAF_GEOMETRY',index+1,467,scene['name'],geometry['compression'],flush=True)
        except Exception as e:
            result['stopped']={'name':scene['name'],'error':str(e)};PROGRESS.write_text(json.dumps(result,indent=2)+'\n');raise
    result.update(completed=True,cross_tabs=summarize(result['rows']),geometry_cross_tabs=geometry_summary(result['rows']),R11_sha256=D.sha(D.R11_PATH))
    D.need(result['R11_sha256']==D.R11_SHA,'Selected R11 changed')
    D.write_once_json(REPORT,result);print('RAF_GEOMETRY_COMPLETE',len(result['rows']),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run'));ap.add_argument('--manifest-sha');a=ap.parse_args()
    prepare() if a.command=='prepare' else run(a.manifest_sha)


if __name__=='__main__':main()
