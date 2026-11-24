"""Independent stored-byte/header/descriptor check for four FIT-only RGB cases.

Native full RGB arrays are not persisted: their browser-reported hashes are
cross-checked against original G and repeats, not independently re-hashed here.
Stored working and strip bytes ARE independently re-hashed/compared. No targets.
"""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import matched_gamma_data as D
import qualify_profile_rgb_sentinels as Q

SHA='781967dc666f7d2677a4cec9a7069f01e719d93b2ec1d95dd0f81cc2c6741bf4'
FORMAT='<8sHHIIHHIIHHHH4I8s'


def main():
    m=Q.load(SHA);report=json.loads(Q.REPORT.read_text());complete=json.loads((Q.OUT/'complete.json').read_text())
    D.need(report['passed'] is True and report['manifest_sha256']==SHA and report['complete_sha256']==D.sha(Q.OUT/'complete.json'),'Wrong result binding')
    by={r['name']:r for r in complete['scenes']};rows=[]
    for scene in m['scenes']:
        case=by[scene['name']];old=scene['G'];g=(D.ROOT/old['file']).read_bytes()
        D.need(hashlib.sha256(g).hexdigest()==old['sha256'],'Original G changed')
        gh=struct.unpack_from(FORMAT,g);w,h=scene['expected_working_size']
        D.need(gh[:3]==(b'FJLRGB16',2,64) and gh[3:9]==(w,h,3,16,w*h*6,3) and gh[9:13]==(1,1,1,1),'Original G header differs')
        original=g[64:];arrays={};strips={}
        for arm in ('C','P','R'):
            item=case['arms'][arm];b=(D.ROOT/item['file']).read_bytes();strip=(D.ROOT/item['strips_file']).read_bytes()
            D.need(hashlib.sha256(b).hexdigest()==item['sha256'] and hashlib.sha256(strip).hexdigest()==item['strips_sha256'],'Stored bytes changed')
            D.need(item['strip_rows']==list(range(64,scene['expected_native_size'][1],128)),'Strip rows changed')
            D.need(len(strip)==len(item['strip_rows'])*scene['expected_native_size'][0]*6,'Strip byte count wrong')
            if arm=='C':
                D.need(b==original and item['native_sha256']==old['native_sha256'],'Control differs from G')
                D.need(strip==(D.ROOT/old['strips']['file']).read_bytes(),'Control strips differ from G')
                payload=b
            else:
                hd=struct.unpack_from(FORMAT,b)
                D.need(hd[:3]==(b'FJNRGB16',1,64) and hd[3:9]==(w,h,3,16,w*h*6,3) and hd[9:17]==(1,1,2,3,620,302,455,302),'Profile header differs')
                D.need(hd[-1]==bytes.fromhex(item['options_sha256'])[:8],'Options prefix mismatch')
                D.need(len(b)==64+w*h*6,'Profile length wrong');payload=b[64:]
            D.need(hashlib.sha256(payload).hexdigest()==item['working_payload_sha256'],'Working digest differs')
            arrays[arm]=np.frombuffer(payload,dtype='<u2').reshape(h,w,3);strips[arm]=strip
        D.need(np.array_equal(arrays['P'],arrays['R']) and strips['P']==strips['R'],'Repeat differs')
        delta=arrays['P'].astype(float)-arrays['C'].astype(float)
        ref=next(r for r in report['rows'] if r['name']==scene['name'])
        D.need(np.allclose(np.abs(delta).mean((0,1)),ref['mean_abs_working_codes_RGB'],rtol=0,atol=1e-12)
               and np.allclose(delta.mean((0,1)),ref['working_mean_delta_RGB'],rtol=0,atol=1e-12),'Descriptive statistics differ')
        rows.append({'name':scene['name'],'verified':True,'changed_working_pixels':int(np.any(delta!=0,axis=2).sum()),
                     'working_pixels':w*h,'max_abs_working_change':float(np.abs(delta).max())})
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'R11 changed')
    result={'verified':True,'manifest_sha256':SHA,'report_sha256':D.sha(Q.REPORT),'complete_sha256':D.sha(Q.OUT/'complete.json'),
            'verification_script_sha256':D.sha(Path(__file__)),'rows':rows,'R11_sha256':D.R11_SHA,
            'limits':__doc__.strip()+' No Fuji accuracy, target images, development/protected media or model inference.'}
    D.write_once_json(D.DIAG/'profile-rgb-sentinel-verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
