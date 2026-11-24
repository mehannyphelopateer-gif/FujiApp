"""Reproducible synthetic-only Python/JavaScript profile-container qualification.

No RAF/TIFF/FJLRG media, targets or learned weights are read. The only existing
research evidence read is JSON hashes for the fixed profile/primitive records.
"""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import fjnrgb_profile as P

ROOT=Path(__file__).resolve().parent.parent
REPORT=ROOT/'training/diagnostics/fjnrgb-profile-codec-qualification.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not REPORT.exists(),'Prior synthetic qualification exists'
    suite=unittest.defaultTestLoader.loadTestsFromName('test_fjnrgb_profile')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream).run(suite)
    assert result.wasSuccessful(),stream.getvalue()
    for key,name in [('sparse_primitive_report_sha256','manual-kelvin-wb-primitive-report.json'),
                     ('fullnative_v2_report_sha256','manual-kelvin-wb-fullnative-v2-report.json'),
                     ('fullnative_v2_verification_sha256','manual-kelvin-wb-fullnative-v2-verification.json')]:
        assert sha(ROOT/'training/diagnostics'/name)==P.PRIMITIVE_REPORTS[key]
    assert sha(ROOT/'training/diagnostics/wb-alignment-manifest.json')==P.PROFILE_MANIFEST_SHA256
    with tempfile.TemporaryDirectory(prefix='fjnrgb-codec-synthetic-',dir=ROOT/'training/checkpoints') as tmp:
        directory=Path(tmp)
        data=(np.arange(18,dtype=np.uint16)*np.uint16(3107)).reshape(2,3,3);data[0,0]=[0,258,65535]
        options=P.serialize_options(P.canonical_options())
        container=P.encode(3,2,data,options_sha256=P.options_sha256(options))
        provenance=P.build_provenance(container,source_raf_sha256='a'*64,wasm_sha256='b'*64,capture_dynamic_range=200,
                                      capture_dynamic_range_source='synthetic fixture only',decoded_native_size=[3,2])
        (directory/'python.fjnrgb').write_bytes(container)
        (directory/'python.fjnrgb.json').write_text(json.dumps(provenance))
        completed=subprocess.run(['node','scripts/test-fjnrgb-profile.mjs',str(directory)],cwd=ROOT,check=True,capture_output=True,text=True,timeout=90)
        js=json.loads(completed.stdout);assert js['passed'] and js['checks']>=70
        out=(directory/'javascript.fjnrgb').read_bytes();side=P.strict_json_loads((directory/'javascript.fjnrgb.json').read_text())
        decoded=P.decode(out,side,expected_source_sha256='a'*64,expected_wasm_sha256='b'*64)
        assert out==container and side==provenance and np.array_equal(decoded.rgb_uint16,data)
    files=['training/fjnrgb_profile.py','training/test_fjnrgb_profile.py','training/check_fjnrgb_profile_codec.py',
           'scripts/lib/fjnrgb-profile.mjs','scripts/test-fjnrgb-profile.mjs','training/fjlrg_v2.py',
           'scripts/lib/fjlrg-v2.mjs','scripts/lib/fjlrg-v2-fixtures.mjs']
    report={'qualification':'SYNTHETIC_CONTAINER_AND_PROVENANCE_ONLY','passed':True,'real_photo_exports':0,
            'python_tests':result.testsRun,'javascript_checks':js['checks'],'python_to_javascript_exact':True,
            'javascript_to_python_exact':True,'synthetic_container_sha256':js['container_sha256'],
            'profile_manifest_sha256':P.PROFILE_MANIFEST_SHA256,'primitive_reports':P.PRIMITIVE_REPORTS,
            'bindings':{f:sha(ROOT/f) for f in files},
            'boundaries':'No raw decode, input-domain assembly, RGB/Fuji parity, training or runtime qualification. Existing v2 codecs and canonical inputs remain unchanged.'}
    with REPORT.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
