"""Post-hoc scalar-only DR100 seed-spread diagnostic; never trains or loads media.

Three seeds describe observed variation, not a noise distribution or a
causal test. The frozen study remains FAIL regardless of this analysis.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = 'd416942f282ec8d1fbe65fad6ad6a4e3c30d612d7938cc6052bc364885b018d0'
SEEDS = (42, 43, 44)
ARMS = ('G', 'neutral')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    need(bool(values) and all(type(v) in (int, float) and math.isfinite(v) for v in values),
         'Missing/nonfinite scalar metric')
    return math.fsum(values) / len(values)


def analyse(manifest, evals):
    scenes = manifest['scenes']
    need(len(scenes) == 571 and all(s['role'] in ('fit', 'development') for s in scenes),
         'Only the frozen FIT/development metadata is allowed')
    wanted = {s['name'] for s in scenes}
    need(len(wanted) == 571, 'Duplicate scene')
    for (seed, arm), ev in evals.items():
        need(seed in SEEDS and arm in ARMS and set(ev['scenes']) == wanted,
             'Unexpected evaluation coverage')
    need(set(evals) == {(s, a) for s in SEEDS for a in ARMS}, 'Incomplete paired seeds')
    dev = [s for s in scenes if s['role'] == 'development']
    groups = {dr: [s for s in dev if s['neutral_kind'] == 'G' and s['captureDynamicRange'] == dr]
              for dr in (100, 200)}
    need(len(groups[100]) == 12 and len(groups[200]) == 25, 'Frozen identity group changed')
    need(not any(s['role'] == 'fit' and s['captureDynamicRange'] == 100 for s in scenes),
         'DR100 unexpectedly has FIT support')

    def metric(s, seed, arm):
        value = evals[(seed, arm)]['scenes'][s['name']]['metrics']['mean_dE00']
        mean([value])
        return value

    per_scene = []
    for s in groups[100]:
        values = {a: [metric(s, seed, a) for seed in SEEDS] for a in ARMS}
        deltas = [n - g for n, g in zip(values['neutral'], values['G'])]
        per_scene.append({
            **{k: s[k] for k in ('name', 'session', 'source_wb_mode', 'source_kelvin')},
            'G_seed_values': values['G'], 'neutral_seed_values': values['neutral'],
            'G_observed_seed_range': max(values['G']) - min(values['G']),
            'neutral_observed_seed_range': max(values['neutral']) - min(values['neutral']),
            'paired_neutral_minus_G': dict(zip(map(str, SEEDS), deltas))})
    per_seed = {}
    for seed in SEEDS:
        blocks = {}
        for dr, rows in groups.items():
            deltas = [metric(s, seed, 'neutral') - metric(s, seed, 'G') for s in rows]
            blocks[str(dr)] = {
                'n': len(rows), 'mean_delta': mean(deltas),
                'contribution_to_identity37_mean': math.fsum(deltas) / 37,
                'worse': sum(v > 0 for v in deltas), 'better': sum(v < 0 for v in deltas),
                'min_delta': min(deltas), 'max_delta': max(deltas),
                'largest_absolute_scene_share_of_absolute_deltas': max(map(abs, deltas)) /
                    math.fsum(map(abs, deltas)) if any(deltas) else 0.0}
        per_seed[str(seed)] = {
            'groups': blocks,
            'identity37_mean_delta': math.fsum(b['contribution_to_identity37_mean'] for b in blocks.values())}
    return {
        'post_hoc': True, 'seeds': list(SEEDS), 'DR100_per_scene': per_scene,
        'per_seed': per_seed,
        'DR100_metadata': {k: dict(Counter(str(s.get(k, 'UNKNOWN_NOT_RECORDED')) for s in groups[100]))
                          for k in ('session', 'source_wb_mode', 'source_kelvin', 'iso')},
        'observed_seed_range_mean': {a: mean([r[a + '_observed_seed_range'] for r in per_scene]) for a in ARMS},
        'limits': [
            'Scalar-only post-hoc diagnosis, no new media, inference, fitting or selection.',
            'The original five-criterion result stays FAIL, including the DR100 cases.',
            'Three observed seeds do not define a noise distribution or establish causal equivalence.',
            'Inputs for identity cases are identical; their predictions differ because training inputs elsewhere changed.',
            'All12 DR100 cases are one development session with no DR100 FIT support; scene counts are not independent sessions.',
            'ISO is not in this manifest and is not guessed or extracted here.',
            'No protected K/monitor/finalHeldOut/T8/dr100Eval access; R11 and app unchanged.']}


def main():
    mp = ROOT / 'training/diagnostics/neutral-wb-study-manifest.json'
    rp = ROOT / 'training/runs/neutral-wb-study-report.json'
    vp = ROOT / 'training/diagnostics/neutral-wb-study-verification.json'
    m, r, v = (json.loads(p.read_text()) for p in (mp, rp, vp))
    need(digest(mp) == MANIFEST_SHA == v['manifest_sha256'], 'Manifest binding changed')
    need(v['verified'] is True and v['screen_passed'] is False and r['screening']['passed'] is False,
         'Frozen verified study result differs')
    need(digest(rp) == v['report_sha256'], 'Final report hash changed')
    fingerprints = {str(p.relative_to(ROOT)): digest(p) for p in (mp, rp, vp)}
    evals = {}
    for seed in SEEDS:
        for arm in ARMS:
            p = ROOT / f'training/runs/neutral-wb-study-seed{seed}-{arm}-eval.json'
            need(digest(p) == v['endpoint_fingerprints'][f'{seed}/{arm}']['eval_sha256'], 'Eval hash changed')
            ev = json.loads(p.read_text())
            need(ev['manifest_sha256'] == MANIFEST_SHA and ev['seed'] == seed and ev['arm'] == arm,
                 'Wrong evaluation identity')
            evals[(seed, arm)] = ev
            fingerprints[str(p.relative_to(ROOT))] = digest(p)
    result = analyse(m, evals)
    for seed in SEEDS:
        need(abs(result['per_seed'][str(seed)]['identity37_mean_delta'] -
                 r['per_seed'][str(seed)]['development_identity37']['delta_mean_dE00']) < 1e-10,
             'Identity contribution sum does not reproduce the frozen report')
    result.update(input_sha256=fingerprints, script_sha256=digest(Path(__file__)),
                  original_screen_passed=False)
    out = ROOT / 'training/diagnostics/neutral-wb-dr100-seed-spread.json'
    with out.open('x') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('POST_HOC_SCALAR_DIAGNOSTIC_COMPLETE', out, flush=True)
    print(json.dumps(result['per_seed'], indent=1))


if __name__ == '__main__':
    main()
