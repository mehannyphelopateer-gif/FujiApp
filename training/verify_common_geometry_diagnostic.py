"""Reproduce the eight-fit screen and independently check the raw-coordinate intersections."""
import json
from pathlib import Path
import numpy as np
import matched_gamma_data as D
import diagnose_common_geometry as A
import geometry_registration_metrics as R
import fjlrg_v2 as V

SHA='a389aba20e1d5577baa2d361df08ca358c420a1362656ac67e364c6a4f69e05c'


def main():
    m=A.load(SHA);report=json.loads(A.REPORT.read_text())
    assert report['completed'] and report['manifest_sha256']==SHA and report['criteria']==A.CRITERIA
    assert [r['name'] for r in report['rows']]==[s['name'] for s in m['scenes']]
    summaries=[]
    for s,row in zip(m['scenes'],report['rows']):
        assert s['role']==row['role']=='fit' and row['hypothesis']==s['hypothesis']
        g=s['decoder_geometry'];tags=A.A.primary_tags(s)
        x,y,w,h=tags['raw_image_crop_top_left'][1],tags['raw_image_crop_top_left'][0],tags['raw_image_cropped_size'][1],tags['raw_image_cropped_size'][0]
        left,top=max(x,g['left_margin']),max(y,g['top_margin'])
        right,bottom=min(x+w,g['left_margin']+g['iwidth']),min(y+h,g['top_margin']+g['iheight'])
        assert s['hypothesis']['common_rect_xywh']==[left,top,right-left,bottom-top]
        assert s['hypothesis']['source_rect_xywh']==[g['left_margin'],g['top_margin'],g['iwidth'],g['iheight']]
        assert s['hypothesis']['target_rect_xywh']==[x,y,w,h]
        ip,tp=D.ROOT/s['input_file'],D.ROOT/s['target_cache_file']
        assert D.sha(ip)==s['input_sha256'] and D.sha(Path(str(ip)+'.json'))==s['input_sidecar_sha256'] and D.sha(tp)==s['target_cache_sha256']
        frame=V.load_fjlrg_v2(ip,expected_contract=V.CONTRACT_ASSHOT,expected_source_sha256=s['raf_sha256'],expected_wasm_sha256=s['wasm_sha256'])
        source=frame.rgb_uint16.astype(np.float32)/np.float32(65535);target=np.load(tp,allow_pickle=False)
        for name,src,ref in [('baseline',source,target),('candidate',R.zoom_resample(source,s['hypothesis']['source_window']),R.zoom_resample(target,s['hypothesis']['target_window']))]:
            assert A.measure(src,ref)==row[name]
        v=row['candidate']['summary'];delta=row['candidate']['feature_ncc']-row['baseline']['feature_ncc'];assert delta==row['ncc_change']
        checks={'tile_coverage':v['usable_tiles']>=6,'median_abs_dx':v['median_abs_dx'] is not None and v['median_abs_dx']<=0.5,
                'median_abs_dy':v['median_abs_dy'] is not None and v['median_abs_dy']<=0.5,
                'p95_shift_norm':v['p95_shift_norm'] is not None and v['p95_shift_norm']<=1.0,'ncc_noninferior':delta>=-0.02}
        assert row['checks']==checks and row['diagnostic_screen_passed']==all(checks.values())
        summaries.append({'name':s['name'],'source_zoom':tags['raw_zoom_active'],
                          'baseline_p95_shift':row['baseline']['summary']['p95_shift_norm'],
                          'candidate_p95_shift':v['p95_shift_norm'],'screen_passed':row['diagnostic_screen_passed']})
        print('VERIFIED_COMMON_GEOMETRY',s['name'],flush=True)
    assert report['diagnostic_screen_passed']==all(r['screen_passed'] for r in summaries)
    assert D.sha(D.R11_PATH)==D.R11_SHA
    D.write_once_json(D.DIAG/'common-geometry-diagnostic-verification.json',{
        'verified':True,'manifest_sha256':SHA,'report_sha256':D.sha(A.REPORT),'verification_script_sha256':D.sha(Path(__file__)),
        'rows':summaries,'checks':'Raw rectangle intersections independently reconstructed; all image hashes, measurements and frozen screen checks reproduced.',
        'limits':'Already-unblinded8FIT diagnostic, no colour improvement or generalisation claim, no canonical/model/app changes.',
        'R11_sha256':D.R11_SHA})


if __name__=='__main__':main()
