"""Transport-only revision: the committed1.58MB manifest exceeds Node's default1MiB stdout cap.

No v1 source/artifact reads or decodes occurred. V2 raises only git-show's
bounded buffer to32MiB, retaining the v1 code/manifest as unrun evidence.
All455FIT source/arm/qualification rules and selected canaries are identical.
"""
import argparse
from pathlib import Path
import matched_gamma_data as D
import prepare_neutral_wb_fit_export as B

MANIFEST=D.DIAG/'neutral-wb-fit-export-manifest-v2.json'
OUT=B.OUT
read=B.read


def build():
    m=B.build()
    frozen=B.read(B.MANIFEST)
    assert m==frozen, 'Original unrun scope changed'
    m['bindings'].pop('scripts/export-neutral-wb-fit.mjs')
    for p in [Path(__file__),D.ROOT/'scripts/export-neutral-wb-fit-v2.mjs',
              D.TRAIN/'verify_neutral_wb_fit_export_v2.py',D.TRAIN/'test_neutral_wb_fit_scope_v2.py']:
        m['bindings'][str(p.relative_to(D.ROOT))]=D.sha(p)
    m['revision']={'reason':__doc__.strip(),'unrun_parent_manifest_sha256':D.sha(B.MANIFEST),
                   'git_show_max_buffer_bytes':32<<20,'entrypoint':'scripts/export-neutral-wb-fit-v2.mjs'}
    return m


def load(expected):
    assert expected and D.sha(MANIFEST)==expected and D.committed(MANIFEST)
    m=read(MANIFEST)
    for rel,h in m['bindings'].items():
        assert D.sha(D.ROOT/rel)==h,'Binding changed: '+rel
        if not rel.startswith(('node_modules/','training/checkpoints/')):assert D.committed(D.ROOT/rel),'Uncommitted binding: '+rel
    assert m==build(), 'Source-only export plan drift'
    return m


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','check'));ap.add_argument('--manifest-sha');a=ap.parse_args()
    if a.command=='prepare':D.write_once_json(MANIFEST,build());print('NEUTRAL_WB_FIT_EXPORT_MANIFEST_V2',D.sha(MANIFEST),flush=True)
    else:load(a.manifest_sha);print('NEUTRAL_WB_FIT_EXPORT_MANIFEST_VERIFIED_V2',flush=True)
