"""Verify saved metadata coverage and independently parse four fit RAF canaries.

No image/target/decoder/model access; source files are read only at bounded
RAF header/directory offsets. Native-size/zoom associations are observations.
"""
from collections import Counter
import json
from pathlib import Path
import struct
import matched_gamma_data as D
import audit_raf_geometry as A

SHA='e041dff199b19c29aa322c60e1e5015953e641109da6184a304c02e7ef9aad5a'
TAGS={0x100:'raw_image_full_size',0x110:'raw_image_crop_top_left',0x111:'raw_image_cropped_size',0x115:'raw_image_aspect_ratio',
      0x117:'raw_zoom_active',0x118:'raw_zoom_top_left',0x119:'raw_zoom_size'}


def independently_read_primary(path):
    with path.open('rb') as f:
        prefix=f.read(108);assert len(prefix)==108 and prefix[:16]==b'FUJIFILMCCD-RAW '
        jpeg,jpeg_len=struct.unpack_from('>II',prefix,84)
        start,length=struct.unpack_from('>II',prefix,92)
        tiff,tiff_len=struct.unpack_from('>II',prefix,100)
        assert start>=108 and length>=4 and not(start<jpeg+jpeg_len and jpeg<start+length)
        assert not(start<tiff+tiff_len and tiff<start+length)
        f.seek(start);count=struct.unpack('>I',f.read(4))[0];assert count<256
        values={v:None for v in TAGS.values()};pos=start+4;read_bytes=112
        for _ in range(count):
            assert pos+4<=start+length;f.seek(pos);tag,n=struct.unpack('>HH',f.read(4));read_bytes+=4;pos+=4
            assert pos+n<=start+length
            if tag in TAGS:
                assert n==4 and values[TAGS[tag]] is None
                b=f.read(n);read_bytes+=n
                values[TAGS[tag]]=struct.unpack('>I',b)[0] if tag==0x117 else list(struct.unpack('>HH',b))
            pos+=n
        assert read_bytes<=4096
        return values


def main():
    m=A.load(SHA);r=json.loads(A.REPORT.read_text());rows=r['rows'];by={s['name']:s for s in m['scenes']}
    D.need(r['completed'] and r['manifest_sha256']==SHA and len(rows)==len(by)==467 and len({z['name'] for z in rows})==467,'Wrong report coverage')
    combinations=Counter();canaries=[]
    for x in rows:
        s=by[x['name']];D.need(all(x[k]==v for k,v in s.items()) and x['source_sha256_verified']==s['raf_sha256'],'Source record differs')
        g=x['geometry'];D.need(g['bytes_read']==sum(v['length'] for v in g['reads'])<=4096,'Read budget/accounting differs')
        forbidden=[g['sections']['jpeg']]+[v for v in g['sections']['fuji_ifds'] if v]
        for read in g['reads']:
            a,b=read['offset'],read['offset']+read['length']
            D.need(all(not(a<t['offset']+t['length'] and t['offset']<b) for t in forbidden),'Image payload interpreted')
        D.need(len(g['directories'])==1,'Unexpected multiple directories; independent check requires explicit extension')
        tags=g['directories'][0]['tags'];combinations[(tuple(x['G_native_size']),tags['raw_zoom_active'])]+=1
        if x['name'] in A.CANARIES:
            D.need(independently_read_primary(A.source_path(s))==tags,'Independent canary metadata differs')
            canaries.append(x['name'])
    D.need(set(canaries)==set(A.CANARIES) and r['cross_tabs']==A.summarize(rows) and r['geometry_cross_tabs']==A.geometry_summary(rows),'Summary/canary coverage differs')
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'Selected model changed')
    result={'verified':True,'manifest_sha256':SHA,'report_sha256':D.sha(A.REPORT),'verification_script_sha256':D.sha(Path(__file__)),
            'rows':467,'independently_parsed_primary_canaries':canaries,
            'native_size_zoom_counts':[{'native_size':list(size),'raw_zoom_active':z,'n':n} for (size,z),n in sorted(combinations.items())],
            'limits':'Four source metadata canaries independently parsed, all467 saved records/coverage/read ranges checked. No target pixels/FoV proof or crop/metric/input/model change.',
            'R11_sha256':D.R11_SHA}
    D.write_once_json(D.DIAG/'raf-geometry-fit-verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
