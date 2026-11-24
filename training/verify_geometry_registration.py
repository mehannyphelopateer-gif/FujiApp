"""Reproduce the eight frozen fit-only registration measurements; no new hypotheses."""
import json
from pathlib import Path
import cv2
import numpy as np
import matched_gamma_data as D
import diagnose_geometry_registration as A
import geometry_registration_metrics as R
import fjlrg_v2 as V

SHA = 'af7ee01bb3e1a9460cc574b441d12ca3b60037d90be66cb37bd5046a0d6ead7a'


def main():
    m = A.load(SHA)
    report = json.loads(A.REPORT.read_text())
    assert report['completed'] and report['manifest_sha256'] == SHA
    assert report['no_transform_selected'] and report['no_colour_or_model_score']
    assert [r['name'] for r in report['rows']] == [s['name'] for s in m['scenes']]
    summaries = []
    for s, saved in zip(m['scenes'], report['rows']):
        inp, target = D.ROOT/s['input_file'], D.ROOT/s['target_cache_file']
        assert D.sha(inp) == s['input_sha256']
        assert D.sha(Path(str(inp)+'.json')) == s['input_sidecar_sha256']
        assert D.sha(target) == s['target_cache_sha256']
        assert D.sha(D.ROOT/saved['panel_file']) == saved['panel_sha256']
        g = V.load_fjlrg_v2(inp, expected_contract=V.CONTRACT_ASSHOT,
                           expected_source_sha256=s['raf_sha256'], expected_wasm_sha256=s['wasm_sha256'])
        rgb = g.rgb_uint16.astype(np.float32)/np.float32(65535)
        ref = np.load(target, allow_pickle=False)
        measurements, window, _ = A.measure_candidates(rgb, ref, A.primary_tags(s), s['target_orientation'])
        assert measurements == saved['metrics'] and window == saved['window']
        # Separate correlation implementation cross-checks the identity scalar.
        a, b = R.registration_feature(rgb), R.registration_feature(ref)
        independent = float(np.corrcoef(a.astype(np.float64).ravel(), b.astype(np.float64).ravel())[0, 1])
        assert abs(independent-saved['metrics']['identity']['feature_ncc']) < 1e-12
        # Independently reproduce the middle tile with the recorded bounds.
        tile = saved['metrics']['identity']['tiles'][4]
        y0, y1, x0, x1 = tile['bounds_yx']
        av, bv = a[y0:y1,x0:x1].astype(np.float64), b[y0:y1,x0:x1].astype(np.float64)
        if av.std() >= 1e-4 and bv.std() >= 1e-4:
            (dx,dy), response = cv2.phaseCorrelate(av.copy(), bv.copy(), cv2.createHanningWindow((x1-x0,y1-y0),cv2.CV_64F))
            assert max(abs(dx-tile['dx']),abs(dy-tile['dy']),abs(response-tile['response'])) < 1e-12
        summaries.append({'name':s['name'], 'source_zoom':saved['source_zoom'],
                          'identity_ncc':independent, **R.summarize_shifts(saved['metrics']['identity']['tiles'])})
        print('VERIFIED_REGISTRATION', s['name'], flush=True)
    assert D.sha(D.R11_PATH) == D.R11_SHA
    result = {'verified':True, 'manifest_sha256':SHA, 'report_sha256':D.sha(A.REPORT),
              'verifier_sha256':D.sha(Path(__file__)), 'scenes':summaries,
              'checks':'All saved metrics reproduced; identity NCC separately cross-checked with numpy.corrcoef; middle-tile phase correlation independently reconstructed; eight panels and input/target/sidecar hashes checked.',
              'limits':'Same eight FIT inputs, diagnostic only. Appearance/clipping affects gradient correlation; no geometric transform or model selected.',
              'R11_sha256':D.R11_SHA}
    D.write_once_json(D.DIAG/'geometry-registration-verification.json', result)


if __name__ == '__main__':
    main()
