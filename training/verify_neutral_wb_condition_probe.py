"""Independent scalar accounting and fixed CPU canaries for the condition probe.

Supplementary diagnostic only. It cannot change the original FAIL, select
a condition/model, access protected sources, or establish causality.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
ARMS = ('G', 'neutral')
CONDITIONS = (0.0, 0.5, 1.0)
METRIC_TOLERANCES = {'mean_dE00': 1e-5, 'p95_dE00': 1e-4, 'linear_l1_unclamped': 1e-7}
BRIGHTNESS_TOLERANCES = {'meanYpred': 1e-6, 'meanYtarget': 1e-6,
                         'B_err': 1e-6, 'B_stops': 1e-5, 'B_shift': 1e-5}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(v):
    return type(v) in (int, float) and math.isfinite(v)


def average(v):
    need(bool(v) and all(number(x) for x in v), 'Empty/nonfinite scalar group')
    return math.fsum(v) / len(v)


def key(row):
    return row['seed'], row['arm'], row['name'], row['condition']


def source_rows(parent):
    rows = [s for s in parent['scenes'] if s['role'] == 'development'
            and s['neutral_kind'] == 'G' and s['policy_subgroup'] == 'identity']
    need(len(rows) == 37 and len({r['name'] for r in rows}) == 37, 'Identity count drift')
    need({s['session'] for s in rows} == {14}, 'Identity session changed')
    need(sum(s['captureDynamicRange'] == 100 for s in rows) == 12
         and sum(s['captureDynamicRange'] == 200 for s in rows) == 25, 'DR count drift')
    need(all(s['inputs']['G'] == s['inputs']['neutral'] for s in rows), 'Identity bytes differ')
    return rows


def canary_cases(parent):
    rows = source_rows(parent)
    order = lambda s: hashlib.sha256(('neutral-wb-condition-probe-verification-v1\n' + s['name']).encode()).hexdigest()
    selected = {dr: min((s for s in rows if s['captureDynamicRange'] == dr), key=order)
                for dr in (100, 200)}
    out = []
    for seed in SEEDS:
        for arm in ARMS:
            for dr, c, actual in ((100, 0.0, True), (200, 0.5, True),
                                  (100, 0.5, False), (100, 1.0, False), (200, 0.0, False)):
                out.append({'seed': seed, 'arm': arm, 'name': selected[dr]['name'],
                            'condition': c, 'is_actual': actual})
    return out


def scalar_accounting(parent, cells, evals):
    """Rebuild group arithmetic from raw scalars without the probe's summary helpers."""
    rows = source_rows(parent); by = {r['name']: r for r in rows}
    expected = {(seed, arm, r['name'], c) for seed in SEEDS for arm in ARMS
                for r in rows for c in CONDITIONS}
    need(all(number(r['condition']) for r in cells), 'Invalid condition type')
    indexed = {key(r): r for r in cells}
    need(len(cells) == len(indexed) == len(expected) == 666 and set(indexed) == expected,
         'Missing/extra/duplicate scalar cells')
    need(all(r.get('is_actual') is True for r in cells[:222])
         and all(r.get('is_actual') is False for r in cells[222:]), 'Reproduction-first order violated')
    max_repro = {k: 0.0 for k in METRIC_TOLERANCES}
    for row in cells:
        src = by[row['name']];actual = 0.0 if src['captureDynamicRange'] == 100 else 0.5
        need(row['sourceDR'] == src['captureDynamicRange']
             and row['is_actual'] is (row['condition'] == actual), 'Cell identity changed')
        for metric, tolerance in METRIC_TOLERANCES.items():
            value = row['metrics'][metric]
            need(number(value), 'Nonfinite metric')
            if row['is_actual']:
                baseline = evals[(row['seed'], row['arm'])]['scenes'][row['name']]['metrics'][metric]
                gap = abs(value - baseline);need(gap <= tolerance, 'Actual-condition reproduction failed')
                max_repro[metric] = max(max_repro[metric], gap)
        b = row['brightness']
        need(number(b['meanYpred']) and number(b['meanYtarget']) and number(b['B_err']), 'Nonfinite linear brightness')
        need(abs(b['B_err'] - (b['meanYpred'] - b['meanYtarget'])) <= 1e-10, 'Brightness subtraction differs')
        target_ok = b['meanYpred'] > 1e-6 and b['meanYtarget'] > 1e-6
        if target_ok:
            need(number(b['B_stops']) and abs(b['B_stops'] - math.log2(b['meanYpred']/b['meanYtarget'])) <= 1e-10,
                 'Brightness stops differ')
        else:
            need(b['B_stops'] is None, 'Undefined stops must be null')
        base = indexed[(row['seed'], row['arm'], row['name'], actual)]['brightness']['meanYpred']
        shift_ok = b['meanYpred'] > 1e-6 and base > 1e-6
        if shift_ok:
            need(number(b['B_shift']) and abs(b['B_shift'] - math.log2(b['meanYpred']/base)) <= 1e-10,
                 'Target-free brightness response differs')
        else:
            need(b['B_shift'] is None, 'Undefined response must be null')
        for field in ('B_stops', 'B_shift'):
            reason = b['null_reasons'][field]
            need((reason is None) if b[field] is not None else isinstance(reason, str) and bool(reason),
                 'Undefined brightness coverage reason missing/inconsistent')

    summaries = []; paired = []
    for seed in SEEDS:
        for dr in (100, 200):
            group = [s for s in rows if s['captureDynamicRange'] == dr]
            for c in CONDITIONS:
                for arm in ARMS:
                    block = [indexed[(seed, arm, s['name'], c)] for s in group]
                    actual = 0.0 if dr == 100 else 0.5
                    rec = {'seed': seed, 'arm': arm, 'sourceDR': dr, 'condition': c,
                           'n': len(group), 'metrics': {}, 'brightness': {},
                           'is_actual': c == actual,
                           'condition_fit_support': 'below_training_interval' if c == 0 else 'observed_training_code'}
                    for metric in METRIC_TOLERANCES:
                        values = [r['metrics'][metric] for r in block]
                        deltas = [r['metrics'][metric] - indexed[(seed, arm, r['name'], actual)]['metrics'][metric]
                                  for r in block]
                        rec['metrics'][metric] = {
                            'mean': average(values), 'response_mean': average(deltas),
                            'response_median': statistics.median(deltas),
                            'response_min': min(deltas), 'response_max': max(deltas),
                            'response_better': sum(x < 0 for x in deltas),
                            'response_worse': sum(x > 0 for x in deltas)}
                    for metric in BRIGHTNESS_TOLERANCES:
                        values = [r['brightness'][metric] for r in block if r['brightness'][metric] is not None]
                        rec['brightness'][metric] = {
                            'n_defined': len(values), 'partial_coverage': len(values) != len(group),
                            'mean': average(values) if values else None}
                    summaries.append(rec)
                for metric in METRIC_TOLERANCES:
                    deltas = [indexed[(seed, 'neutral', s['name'], c)]['metrics'][metric] -
                              indexed[(seed, 'G', s['name'], c)]['metrics'][metric] for s in group]
                    paired.append({'seed': seed, 'sourceDR': dr, 'condition': c, 'metric': metric,
                                   'n': len(group), 'mean_gap': average(deltas),
                                   'contribution_to_hypothetical_uniform_condition_37': math.fsum(deltas)/37,
                                   'neutral_worse': sum(x > 0 for x in deltas),
                                   'neutral_better': sum(x < 0 for x in deltas)})
    ranges = []
    for dr in (100, 200):
        for c in CONDITIONS:
            for metric in METRIC_TOLERANCES:
                selected = [r for r in paired if r['sourceDR'] == dr and r['condition'] == c and r['metric'] == metric]
                values = {str(r['seed']): r['mean_gap'] for r in selected}
                ranges.append({'sourceDR': dr, 'condition': c, 'metric': metric,
                               'three_observed_seed_gaps': values,
                               'observed_range': max(values.values())-min(values.values()),
                               'label': 'DESCRIPTIVE_THREE_VALUES_NOT_A_NOISE_DISTRIBUTION_OR_CAUSAL_TEST'})
    return {'cell_count': 666, 'reproduction_cells': 222, 'counterfactual_cells': 444,
            'reproduction_max_abs_gaps': max_repro, 'independent_groups': summaries,
            'independent_paired_gaps': paired, 'observed_arm_gap_seed_ranges': ranges}


def brightness(pred, target, actual_mean):
    import numpy as np
    p = pred.detach().cpu().numpy().astype(np.float64)
    t = target.detach().cpu().numpy().astype(np.float64)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64).reshape(1, 3, 1, 1)
    yp, yt = float((p*weights).sum(axis=1).mean()), float((t*weights).sum(axis=1).mean())
    need(number(yp) and number(yt), 'Nonfinite canary brightness')
    return {'meanYpred': yp, 'meanYtarget': yt, 'B_err': yp-yt,
            'B_stops': math.log2(yp/yt) if yp > 1e-6 and yt > 1e-6 else None,
            'B_shift': math.log2(yp/actual_mean) if yp > 1e-6 and actual_mean > 1e-6 else None}


def main():
    import torch
    import neutral_wb_study_data as D
    import run_neutral_wb_study as T
    import probe_neutral_wb_condition as P
    from provia_color_pilot import dev_metrics
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--scope-sha', required=True);args = ap.parse_args()
    scope = P.load_scope(args.scope_sha)
    parent, evals = P.load_parent_and_evals(scope)
    # Field/schema equality is checked before any source or checkpoint access.
    need(scope['conditions'] == list(CONDITIONS), 'Condition grid changed')
    torch.set_num_threads(scope['num_threads'])
    path = ROOT/'training/runs/neutral-wb-condition-probe-results.json'
    report = json.loads(path.read_text())
    need(report['scope_sha256'] == args.scope_sha and report['status'] == 'complete'
         and report['original_screen_passed'] is False, 'Probe incomplete/voided or original result changed')
    cells = report['cells']; rebuilt = scalar_accounting(parent, cells, evals)
    canaries = canary_cases(parent)
    need(scope['canary_cases'] == canaries, 'Independent canary set changed')
    indexed = {key(r): r for r in cells}; checks = []
    for seed in SEEDS:
        for arm in ARMS:
            _, ck, h = T.validate_endpoint(parent, seed, arm)
            need(h == evals[(seed, arm)]['checkpoint_sha256'], 'Endpoint differs from frozen evaluation')
            model = D.initialise(parent, seed, torch.device('cpu'))
            model.load_state_dict(ck['model'], strict=True);model.eval()
            actual_means = {}
            with torch.no_grad():
                for case in (r for r in canaries if r['seed'] == seed and r['arm'] == arm):
                    x, y, actual = P.allowlisted_pair(parent, scope, case['name'])
                    pred = D.forward(model, parent, x, torch.tensor([case['condition']], dtype=torch.float32), torch.device('cpu'))
                    need(bool(torch.isfinite(pred).all()), 'Nonfinite independent canary')
                    if case['is_actual']:
                        actual_means[case['name']] = brightness(pred, y, 1.0)['meanYpred']
                    b = brightness(pred, y, actual_means[case['name']])
                    metrics = dev_metrics(pred, y);saved = indexed[key(case)];gaps = {}
                    for metric, limit in METRIC_TOLERANCES.items():
                        gap = abs(metrics[metric]-saved['metrics'][metric]);need(gap <= limit, 'Canary metric mismatch')
                        gaps[metric] = gap
                    for metric, limit in BRIGHTNESS_TOLERANCES.items():
                        left, right = b[metric], saved['brightness'][metric]
                        need((left is None) == (right is None), 'Canary null coverage differs')
                        if left is not None:
                            gap = abs(left-right);need(gap <= limit, 'Canary brightness mismatch')
                            gaps[metric] = gap
                    checks.append({**case, 'metric_abs_gaps': gaps})
            print('CONDITION_PROBE_INDEPENDENT_ENDPOINT', seed, arm, 'five canaries passed', flush=True)
    need(len(checks) == 30 and D.sha(D.R11_PATH) == D.R11_SHA, 'Canary count/R11 drift')
    out = ROOT/'training/diagnostics/neutral-wb-condition-probe-verification.json'
    with out.open('x') as f:
        f.write(json.dumps({'verified': True, 'scope_sha256': args.scope_sha,
                           'result_sha256': digest(path), 'verifier_sha256': digest(Path(__file__)),
                           'original_screen_passed': False, 'independent_accounting': rebuilt,
                           'canary_checks': checks, 'R11_sha256': D.R11_SHA,
                           'limits': 'Sensitivity only; no causal mechanism, usable condition, safety pass or model selection.'},
                          indent=2, allow_nan=False)+'\n')
    print('CONDITION_PROBE_VERIFIED_ORIGINAL_SCREEN_STILL_FAIL', flush=True)


if __name__ == '__main__':
    main()
