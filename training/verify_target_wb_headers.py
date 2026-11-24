"""Independent saved-record arithmetic/coverage check; never opens TIFF media."""
import json
from fractions import Fraction
from pathlib import Path
from collections import Counter
import matched_gamma_data as D
import audit_target_wb_headers as A

MANIFEST_SHA='16fcb23a2d4fb48aad7f12251c00015f4231c1373214618b264a21b7eb889ddc'


def main():
    m=A.load(MANIFEST_SHA);r=json.loads(A.REPORT.read_text())
    D.need(r['completed'] is True and r['manifest_sha256']==MANIFEST_SHA,'Wrong/incomplete report')
    rows=r['rows'];source={s['name']:s for s in m['scenes']}
    D.need(len(rows)==len(source)==583 and len({z['name'] for z in rows})==583 and {z['name'] for z in rows}==set(source),'Wrong coverage')
    archive=D.DIAG/'cn-step1-wb-reference-audit.json'
    arch={x['name']:x['provia'] for x in json.loads(archive.read_text())['scenes']}
    count=Counter();archived=0
    for x in rows:
        s=source[x['name']]
        D.need(x['status']=='checked' and x['tiff_sha256']==s['tiff_sha256'],'Invalid/unbound TIFF entry')
        D.need(all(x[k]==s[k] for k in ('shoot','role','session','source_wb_mode','source_kelvin')),'Source identity differs')
        t=x['target_settings'];levels=[t[k] for k in ('WBLevelR','WBLevelG','WBLevelB')]
        D.need(all(type(v) is int and v>0 for v in levels),'Invalid gain')
        ratios=[Fraction(levels[0],levels[1]),Fraction(levels[2],levels[1])]
        p=s['frozen_source_only_policy'];pred=p['ratios']
        match=None if pred is None else ratios==[Fraction(pred[k]['num'],pred[k]['den']) for k in ('R/G','B/G')]
        settings=(t['WhiteBalance']==s['source_wb_mode'] and t['ColorTemperature']==s['source_kelvin'] and t['WBFineTune']==[0,0])
        D.need(x['exact_policy_ratio_match'] is match and x['neutral_settings_match'] is settings,'Recorded decision differs')
        D.need(settings,'Nonneutral settings cannot be support')
        D.need(x['header_bytes_read']==sum(z['length'] for z in x['header_reads'])<=1<<20,'Read budget/accounting differs')
        D.need(all(type(z['offset']) is int and z['offset']>=0 and type(z['length']) is int and 0<z['length']<=256*1024 and z['purpose'] in ('TIFF header','IFD0 count','IFD0 entries','Exif IFD count','Exif IFD entries','MakerNote') for z in x['header_reads']),'Unapproved parsed header read')
        if x['name'] in arch:
            D.need(all(t[k]==arch[x['name']].get(k) for k in t) and x['archived_reader_crosscheck'] is True,'Archived parser comparison differs')
            archived+=1
        else:D.need(x['archived_reader_crosscheck'] is None,'Invented archive reference')
        count[(x['role'],'exact' if match is True else 'mismatch' if match is False else 'unsupported')]+=1
    D.need(archived==r['archived_reader_matches']==24,'Archive coverage differs')
    D.need(r['groups']==A.summarize(rows) and r['invalid']==0 and r['settings_mismatch']==0,'Summary differs')
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'R11 changed')
    out={'verified':True,'manifest_sha256':MANIFEST_SHA,'report_sha256':D.sha(A.REPORT),'verification_script_sha256':D.sha(Path(__file__)),
         'source_policy_unchanged':True,'rows':583,'archived_reader_matches':24,
         'role_counts':{role:{outcome:count[(role,outcome)] for outcome in ('exact','mismatch','unsupported')} for role in ('fit','development')},
         'method':'Saved records only: independent Fraction comparisons, coverage, SHA bindings, header read budgets,24 archived references; no new TIFF or image reads.',
         'R11_sha256':D.R11_SHA}
    D.write_once_json(D.DIAG/'target-wb-header-verification.json',out)
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
