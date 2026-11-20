"""Metadata-only camera/WB qualification of the frozen583 train sources.
Pillow reads JPEG EXIF only; no image.load/array conversion, targets or models.
"""
import argparse,hashlib,io,json,struct,subprocess
from pathlib import Path
from collections import Counter
import PIL
from PIL import Image
R=Path(__file__).resolve().parent.parent;D=R/'training/diagnostics'
M=D/'provia-v2-source-qualification-manifest.json';O=D/'provia-v2-source-qualification.json';INVENTORY=D/'provia-v2-source-inventory.json'
def need(b,m):
    if not b:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,d):
    with p.open('x')as f:f.write(json.dumps(d,indent=2,allow_nan=False)+'\n')
def maker_wb(mn):
    need(isinstance(mn,bytes)and mn[:8]==b'FUJIFILM','Missing Fuji MakerNote')
    off=struct.unpack_from('<I',mn,8)[0];need(12<=off<len(mn)-2,'MakerNote IFD offset')
    n=struct.unpack_from('<H',mn,off)[0];need(n<=1024 and off+2+n*12<=len(mn),'MakerNote IFD bounds')
    found={}
    for i in range(n):
        tag,typ,count,val=struct.unpack_from('<HHII',mn,off+2+i*12)
        if tag in (0x1002,0x1005):
            need(typ==3 and count==1,'WB scalar type/count');v=val&65535
        elif tag==0x100a:
            need(typ==9 and count==2 and val+8<=len(mn),'WB shift bounds');v=list(struct.unpack_from('<ii',mn,val))
        else:continue
        need(tag not in found or found[tag]==v,'Conflicting WB tag');found[tag]=v
    need(0x1002 in found and 0x100a in found,'Missing source WB mode/shift')
    return found[0x1002],found[0x100a],found.get(0x1005)
def prepare():
    inv=read(INVENTORY);need(inv['completed']and len(inv['rows'])==583,'Inventory incomplete')
    rows=[{k:s[k]for k in('name','shoot','raf_sha256','raf_preview_sha256','source_wb_mode','source_wb_fine_tune_raw','source_kelvin')}for s in inv['rows']]
    need(len({s['name']for s in rows})==len({s['shoot']for s in rows})==583,'Duplicate identities')
    paths=['training/qualify_provia_v2_source_metadata.py',str(INVENTORY.relative_to(R))]
    dump(M,{'scope':__doc__,'bindings':{p:sha(R/p)for p in paths},'pillow_version':PIL.__version__,'scenes':rows,'required_camera':'X100VI','expected_wb_mode_counts':dict(Counter(str(s['source_wb_mode'])for s in rows))})
    print('Source camera/WB qualification manifest',sha(M))
def run(expected):
    need(sha(M)==expected,'Manifest drift');m=read(M);need(PIL.__version__==m['pillow_version'],'Pillow drift')
    for p,h in m['bindings'].items():need(sha(R/p)==h,'Binding drift '+p)
    for p in(M,Path(__file__).resolve()):need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(R))],cwd=R)).hexdigest()==sha(p),'Commit before source reads')
    rows=[]
    for s in m['scenes']:
        p=R/'calibration-input'/s['shoot']/s['name']
        with p.open('rb')as f:
            head=f.read(112);need(head[:16]==b'FUJIFILMCCD-RAW ','Bad RAF');off,n=struct.unpack_from('>II',head,84)
            need(off>=112 and 0<n<=32<<20 and off+n<=p.stat().st_size,'Preview bounds');f.seek(off);prev=f.read(n)
        need(hashlib.sha256(prev).hexdigest()==s['raf_preview_sha256'],'Source preview drift')
        with Image.open(io.BytesIO(prev))as image:
            ex=image.getexif();model=str(ex.get(272,'')).strip();make=str(ex.get(271,'')).strip();mn=ex.get_ifd(34665).get(37500)
            wb,shift,kelvin=maker_wb(mn)
        need(model==m['required_camera']and make=='FUJIFILM','Unexpected source camera')
        need(wb==s['source_wb_mode']and shift==s['source_wb_fine_tune_raw']and kelvin==s['source_kelvin'],'Archived source WB mismatch')
        rows.append({'name':s['name'],'shoot':s['shoot'],'source_raf_sha256_from_inventory':s['raf_sha256'],'preview_sha256':s['raf_preview_sha256'],'camera_model':model,'camera_make':make,'source_wb_mode':wb,'source_wb_fine_tune_raw':shift,'source_kelvin':kelvin,'passed':True})
        if len(rows)%50==0:print('Qualified source EXIF',len(rows),flush=True)
    counts=dict(Counter(str(s['source_wb_mode'])for s in rows));need(counts==m['expected_wb_mode_counts'],'WB coverage changed')
    dump(O,{'passed':True,'manifest_sha256':expected,'scope':m['scope'],'source_wb_mode_counts':counts,'camera_model_counts':dict(Counter(s['camera_model']for s in rows)),'rows':rows,'limits':'Metadata-only. Existing inventory binds full sources; this run checks preview SHA. LibRaw independently checks camera/WB on each later decode; unqualified modes are not added to W.'});print('Source camera/WB metadata qualification PASS',len(rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--manifest-sha');a=p.parse_args()
    if a.command=='prepare':prepare()
    else:need(bool(a.manifest_sha),'Manifest SHA required');run(a.manifest_sha)
