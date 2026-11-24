import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'training'))
import qualify_manual_kelvin_wb_fullnative_v2 as V
import matched_gamma_data as D
import eval_provia_wb_primitive as S
sha='2896f5af1fab83b182f5d6243328930e3c047554acf8b144f2e107bed4197736'
m,old=V.load(sha); report=json.loads(V.REPORT.read_text()); rows=[]
assert report['manifest_sha256']==sha
by={r['name']:r for r in report['rows']}
for s in m['scenes']:
 h,w=s['native_shape']['height'],s['native_shape']['width']
 a={k:np.memmap(V.OUT/f"{s['name']}-{k}.native.u16le",dtype='<u2',mode='r',shape=(h,w,3)) for k in ('D','E')}
 pattern,problems=S.check_structure(old[s['name']]['arms']['D'],old[s['name']]['arms']['E']);assert not problems
 pat=np.array(pattern).reshape(6,6);result=[]
 for c,g in enumerate(s['expected_gain_RGB']):
  total=n=exact=0;abs_sum=signed=maximum=0.0
  for y in range(0,h,37):
   yy=min(y+37,h);active=pat[np.arange(y,yy)[:,None]%6,np.arange(w)[None,:]%6]==c
   d=np.array(a['D'][y:yy,:,c]);e=np.array(a['E'][y:yy,:,c])
   assert not np.any(d[~active]) and not np.any(e[~active])
   total+=int(active.sum())
   valid=active&(d>=.01*65535)&(d<=.8*65535)&(e>=.01*65535)&(e<=.8*65535)
   err=e[valid].astype(float)-g*d[valid].astype(float)
   n+=err.size;abs_sum+=float(np.abs(err).sum());signed+=float(err.sum());exact+=int(np.count_nonzero(err==0))
   if err.size:maximum=max(maximum,float(np.abs(err).max()))
  ref=by[s['name']]['channels_RGB'][c]
  assert n==ref['n'] and total==ref['all_active_sites']
  assert abs(abs_sum/n-ref['mean_abs_codes'])<1e-12 and abs(maximum-ref['max_abs_codes'])<1e-12
  assert abs(signed/n-ref['signed_mean_codes'])<1e-12
  assert abs(exact/n-ref['exact_fraction'])<1e-12
  assert ref['passed'] == (n>=1000 and abs_sum/n<=1 and maximum<=3)
  result.append({'n':n,'mean_abs_codes':abs_sum/n,'max_abs_codes':maximum,'total_active':total,'matched':True})
 assert sum(z['total_active'] for z in result)==h*w
 for k in a:assert D.sha(V.OUT/f"{s['name']}-{k}.native.u16le")==s['original_native_sha256'][k]
 rows.append({'name':s['name'],'channels_RGB':result})
 print('INDEPENDENT_VERIFIED',s['name'],flush=True)
assert D.sha(D.R11_PATH)==D.R11_SHA
out={'verified':True,'manifest_sha256':sha,'report_sha256':D.sha(V.REPORT),'decode_complete_sha256':D.sha(V.OUT/'decode-complete.json'),'verification_script_sha256':D.sha(Path(__file__)),'method':'Separate 37-row/channel accumulation vs frozen 128-row implementation, all6 native hashes and all eligible sites; no targets/models run','rows':rows,'R11_sha256':D.R11_SHA}
D.write_once_json(D.DIAG/'manual-kelvin-wb-fullnative-v2-verification.json',out)
print('ALL_VERIFIED',report['verdict'],flush=True)
