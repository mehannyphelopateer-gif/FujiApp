"""Freeze and read source-only metadata for the existing583 train allowlist.

No JPEG interpretation (preview SHA only), target pixels, unpack, demosaic,
model, fitting or protected media. This prepares provenance for a later G
export; it does not start that export or qualify an expanded W population.
"""
import argparse, hashlib, json, struct, subprocess
from pathlib import Path
from probe_provia_source_wb import MetadataReader
R=Path(__file__).resolve().parent.parent;D=R/'training/diagnostics'
M=D/'provia-v2-source-inventory-manifest.json';O=D/'provia-v2-source-inventory.json'
POOL=D/'provia-color-run-manifest.json';AUDIT=D/'provia-color-data-audit.json'
POOL_SHA='d6c04efd70df63aa38a9f2f08e0b291f858d5085a9670ffb7d3f1b3f4c6cf254'
def need(b,m):
    if not b:raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,d):
    with p.open('x')as f:f.write(json.dumps(d,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(p.read_text())
def prepare():
    need(sha(POOL)==POOL_SHA,'Train allowlist drift')
    pool=read(POOL);a=read(AUDIT);need(a['status']=='PASS'and a['scenes']==583,'Original audit not passed')
    amap={s['name']:s for s in a['per_scene']};scenes=[]
    for s in pool['scenes']:
        ar=amap[s['name']];need(ar['shoot']==s['shoot']and not ar['problems'],'Audit identity issue')
        scenes.append(dict(s,source_wb_mode=ar['raf_wb']['WhiteBalance'],source_wb_fine_tune_raw=ar['raf_wb']['WBFineTune'],source_kelvin=ar['raf_wb']['ColorTemperature'],raf_preview_sha256=ar['raf_preview_sha256']))
    need(len(scenes)==len({s['name']for s in scenes})==len({s['shoot']for s in scenes})==583,'Training coverage differs')
    paths=['training/inventory_provia_v2_sources.py','training/probe_provia_source_wb.py',str(POOL.relative_to(R)),str(AUDIT.relative_to(R))]
    dump(M,{'scope':__doc__,'bindings':{p:sha(R/p)for p in paths},'scenes':scenes})
    print('Metadata-only inventory manifest',sha(M))
def run(expected):
    need(sha(M)==expected,'Manifest mismatch');m=read(M)
    for p,h in m['bindings'].items():need(sha(R/p)==h,'Code/evidence drift '+p)
    for p in(M,Path(__file__).resolve()):need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(R))],cwd=R)).hexdigest()==sha(p),'Must commit before source access')
    rows=[]
    for s in m['scenes']:
        p=R/'calibration-input'/s['shoot']/s['name'];before=p.stat()
        with p.open('rb')as f:
            h=f.read(112);need(h[:16]==b'FUJIFILMCCD-RAW ','RAF header mismatch')
            off,n=struct.unpack_from('>II',h,84);need(off>=112 and 0<n<=32<<20 and off+n<=before.st_size,'Preview bounds')
            f.seek(off);preview=f.read(n)
        need(hashlib.sha256(preview).hexdigest()==s['raf_preview_sha256'],'Archived source WB preview changed')
        reader=MetadataReader(p)
        try:meta=reader.parse()
        finally:reader.close()
        coeffs={}
        for tag,key in [('0xf00d','auto_rgbg'),('0xf00e','cam_mul')]:
            vals=[r['values']for r in meta['records']if r['tag']==tag and r['supported']]
            need(vals and all(v==vals[0]for v in vals),'Missing/conflicting source WB record')
            g,r,b=vals[0];need(all(type(x)is int and 0<x<2**32 for x in (g,r,b)),'Invalid sourceWB')
            coeffs[key]=[r,g,b,g if key=='auto_rgbg'else 0]
        source_hash=sha(p);after=p.stat();need((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed during metadata inventory')
        rows.append(dict(s,raf_sha256=source_hash,**coeffs,metadata_bytes_read=meta['reader_bytes_read']))
        print(f'{len(rows)}/583 {s["name"]} source metadata only',flush=True)
    dump(O,{'completed':True,'manifest_sha256':expected,'scope':m['scope'],'rows':rows})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--manifest-sha');a=p.parse_args()
    if a.command=='prepare':prepare()
    else:need(bool(a.manifest_sha),'Manifest SHA required');run(a.manifest_sha)
