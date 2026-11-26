"""Frozen CPU condition-response diagnostic. No fitting or model selection.

prepare binds the completed study, code and metadata before inference.
run reproduces every actual-condition case before any counterfactual.
Only per-cell scalars are written; the separate verifier builds summaries.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import torch
import neutral_wb_study_data as D
import run_neutral_wb_study as T
import verify_neutral_wb_condition_probe as V
from provia_color_pilot import dev_metrics

ROOT = D.ROOT
SCOPE = ROOT/'training/diagnostics/neutral-wb-condition-probe-scope.json'
RESULT = ROOT/'training/runs/neutral-wb-condition-probe-results.json'
PLAN = ROOT/'training/coordination/neutral-wb-condition-probe-proposal.md'
PARENT_SHA = 'd416942f282ec8d1fbe65fad6ad6a4e3c30d612d7938cc6052bc364885b018d0'
REPORT = ROOT/'training/runs/neutral-wb-study-report.json'
REPORT_SHA = '3e7e219463f245b86f1dd5da7437eaceb10ca267ae532f9de36ac974b3ffa6a4'
VERIFICATION = ROOT/'training/diagnostics/neutral-wb-study-verification.json'
VERIFICATION_SHA = 'e5b7cdbf82fb17e36e92c136e57d09c813af1c23696a09ef177ac1163665f059'
CODE = Path(__file__).resolve()
CODE_PATHS = [CODE, Path(V.__file__).resolve(), PLAN,
              ROOT/'training/test_probe_neutral_wb_condition.py']


def read(path):
    return json.loads(path.read_text())


def bound(path, expected):
    D.need(D.sha(path) == expected and D.committed(path), 'Unpublished/changed evidence: '+str(path))
    return read(path)


def completed_evidence():
    parent = D.load_manifest(PARENT_SHA)
    verification = bound(VERIFICATION, VERIFICATION_SHA)
    report = bound(REPORT, REPORT_SHA)
    D.need(verification['verified'] is True and verification['screen_passed'] is False
           and verification['report_sha256'] == REPORT_SHA and report['screening']['passed'] is False,
           'Original failed verified result changed')
    V.source_rows(parent)
    evals = {}; bindings = {}
    for seed in V.SEEDS:
        for arm in V.ARMS:
            proof = verification['endpoint_fingerprints'][f'{seed}/{arm}']
            path = ROOT/f'training/runs/neutral-wb-study-seed{seed}-{arm}-eval.json'
            ev = bound(path, proof['eval_sha256'])
            D.need(ev['manifest_sha256'] == PARENT_SHA and ev['seed'] == seed and ev['arm'] == arm
                   and ev['checkpoint_sha256'] == proof['checkpoint_sha256'], 'Wrong endpoint evaluation')
            ep, record = T.endpoint_path(seed, arm), T.endpoint_record_path(seed, arm)
            D.need(D.sha(ep) == proof['checkpoint_sha256'], 'Endpoint bytes changed')
            meta = read(record)
            D.need(meta['endpoint_sha256'] == proof['checkpoint_sha256'] and meta['meta']['epoch'] == 20
                   and meta['meta']['global_updates'] == 9100, 'Endpoint record differs')
            evals[(seed, arm)] = ev
            for p in (path, ep, record):bindings[str(p.relative_to(ROOT))] = D.sha(p)
    return parent, evals, bindings


def prepare():
    parent, _, endpoints = completed_evidence()
    for p in CODE_PATHS:D.need(D.committed(p), 'Commit implementation before preparing scope')
    bindings = {str(p.relative_to(ROOT)): D.sha(p) for p in CODE_PATHS}
    bindings.update(endpoints)
    for p in (D.MANIFEST, REPORT, VERIFICATION):bindings[str(p.relative_to(ROOT))] = D.sha(p)
    scope = {'format': 'neutral-wb-condition-probe-v1', 'parent_manifest_sha256': PARENT_SHA,
             'conditions': list(V.CONDITIONS), 'source_names': sorted(s['name'] for s in V.source_rows(parent)),
             'num_threads': torch.get_num_threads(), 'bindings': bindings,
             'canary_cases': V.canary_cases(parent), 'metric_tolerances': V.METRIC_TOLERANCES,
             'brightness_tolerances': V.BRIGHTNESS_TOLERANCES,
             'expected': {'actual_cells': 222, 'counterfactual_cells': 444, 'all_cells': 666, 'canaries': 30},
             'original_screen_passed': False,
             'interpretation': 'Network sensitivity only; no alternative condition, causal mechanism, model selection, screen revision or app change.'}
    D.write_once_json(SCOPE, scope)
    print('CONDITION_PROBE_SCOPE', D.sha(SCOPE), flush=True)


def load_scope(expected_sha):
    D.need(isinstance(expected_sha, str) and len(expected_sha) == 64, 'Pinned scope required')
    scope = bound(SCOPE, expected_sha)
    D.need(scope['format'] == 'neutral-wb-condition-probe-v1' and scope['parent_manifest_sha256'] == PARENT_SHA,
           'Wrong probe scope')
    D.need(scope['conditions'] == list(V.CONDITIONS) and scope['metric_tolerances'] == V.METRIC_TOLERANCES
           and scope['brightness_tolerances'] == V.BRIGHTNESS_TOLERANCES
           and scope['original_screen_passed'] is False, 'Scope constants changed')
    D.need(type(scope['num_threads']) is int and 1 <= scope['num_threads'] <= 128, 'Invalid CPU thread count')
    parent, _, expected_endpoints = completed_evidence()
    expected_paths = set(expected_endpoints) | {str(p.relative_to(ROOT)) for p in CODE_PATHS + [D.MANIFEST, REPORT, VERIFICATION]}
    D.need(set(scope['bindings']) == expected_paths, 'Scope bindings incomplete/expanded')
    for rel, h in scope['bindings'].items():
        p = ROOT/rel
        D.need(D.sha(p) == h, 'Changed probe binding: '+rel)
        if not rel.startswith('training/checkpoints/'):D.need(D.committed(p), 'Unpublished binding')
    D.need(scope['source_names'] == sorted(s['name'] for s in V.source_rows(parent))
           and scope['canary_cases'] == V.canary_cases(parent), 'Frozen source/canary allowlist differs')
    D.need(scope['expected'] == {'actual_cells': 222, 'counterfactual_cells': 444, 'all_cells': 666, 'canaries': 30},
           'Fixed case counts differ')
    scope['_scope_sha256'] = expected_sha
    return scope


def load_parent_and_evals(scope):
    parent, evals, _ = completed_evidence()
    D.need(scope['source_names'] == sorted(s['name'] for s in V.source_rows(parent)), 'Source scope differs')
    return parent, evals


def allowlisted_pair(parent, scope, name):
    D.need(name in scope['source_names'], 'Outside frozen37source allowlist')
    rows = {s['name']: s for s in V.source_rows(parent)}
    D.need(name in rows, 'Not an identity development source')
    x, y, actual = D.Pairs(parent, 'G', role='development').get(name)
    expected = torch.tensor([math.log2(rows[name]['captureDynamicRange']/100)/2], dtype=torch.float32)
    D.need(torch.equal(actual, expected) and actual.device.type == x.device.type == y.device.type == 'cpu',
           'Actual condition/device mismatch')
    return x, y, actual


def load_endpoint_model(parent, scope, seed, arm):
    D.need(seed in V.SEEDS and arm in V.ARMS, 'Unknown endpoint')
    path, ck, h = T.validate_endpoint(parent, seed, arm)
    D.need(scope['bindings'][str(path.relative_to(ROOT))] == h, 'Endpoint not frozen in scope')
    model = D.initialise(parent, seed, torch.device('cpu'))
    model.load_state_dict(ck['model'], strict=True);model.eval()
    D.need(D.state_digest(model.state_dict()) == ck['meta']['state_sha256'], 'Model state differs')
    return model


def brightness(pred, target, baseline_mean):
    w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float64).view(1, 3, 1, 1)
    yp = float((pred.detach().to(dtype=torch.float64)*w).sum(dim=1).mean())
    yt = float((target.to(dtype=torch.float64)*w).sum(dim=1).mean())
    D.need(math.isfinite(yp) and math.isfinite(yt), 'Nonfinite brightness')
    if baseline_mean is None:baseline_mean = yp
    stops_ok, shift_ok = yp > 1e-6 and yt > 1e-6, yp > 1e-6 and baseline_mean > 1e-6
    return {'meanYpred': yp, 'meanYtarget': yt, 'B_err': yp-yt,
            'B_stops': math.log2(yp/yt) if stops_ok else None,
            'B_shift': math.log2(yp/baseline_mean) if shift_ok else None,
            'null_reasons': {'B_stops': None if stops_ok else 'mean target/prediction <=1e-6',
                             'B_shift': None if shift_ok else 'mean prediction/actual <=1e-6'}}


def reproduce(metrics, saved):
    for k, limit in V.METRIC_TOLERANCES.items():
        D.need(V.number(metrics[k]) and V.number(saved[k]) and abs(metrics[k]-saved[k]) <= limit,
               'Actual-condition reproduction failed: '+k)


def phases(actual_phase, counterfactual_phase):
    """A failed actual phase cannot reach a counterfactual callback."""
    actual_phase()
    counterfactual_phase()


def run(expected_sha):
    D.need(not RESULT.exists(), 'Existing result/receipt; overwrite refused')
    scope = load_scope(expected_sha);parent, evals = load_parent_and_evals(scope)
    torch.set_num_threads(scope['num_threads'])
    rows = {s['name']: s for s in V.source_rows(parent)}
    cells = [];base_means = {};phase = 'actual';t0 = time.time()

    def do_phase(is_actual):
        nonlocal phase
        phase = 'actual' if is_actual else 'counterfactual'
        for seed in V.SEEDS:
            for arm in V.ARMS:
                model = load_endpoint_model(parent, scope, seed, arm)
                with torch.no_grad():
                    for name in scope['source_names']:
                        x, y, actual = allowlisted_pair(parent, scope, name)
                        current = float(actual[0])
                        for c in scope['conditions']:
                            if (c == current) is not is_actual:continue
                            pred = D.forward(model, parent, x, torch.tensor([c], dtype=torch.float32), torch.device('cpu'))
                            D.need(bool(torch.isfinite(pred).all()), 'Nonfinite prediction')
                            all_metrics = dev_metrics(pred, y)
                            metrics = {k: all_metrics[k] for k in V.METRIC_TOLERANCES}
                            D.need(all(V.number(v) for v in metrics.values()), 'Nonfinite metric')
                            if is_actual:reproduce(metrics, evals[(seed, arm)]['scenes'][name]['metrics'])
                            b = brightness(pred, y, None if is_actual else base_means[(seed, arm, name)])
                            if is_actual:base_means[(seed, arm, name)] = b['meanYpred']
                            cells.append({'seed': seed, 'arm': arm, 'name': name,
                                          'sourceDR': rows[name]['captureDynamicRange'], 'condition': c,
                                          'is_actual': is_actual, 'metrics': metrics, 'brightness': b})
                print('CONDITION_PROBE',phase,seed,arm,'cells',len(cells),'seconds',round(time.time()-t0),flush=True)
                del model
        if is_actual:D.need(len(cells) == 222, 'Actual-condition phase short')

    try:
        phases(lambda: do_phase(True), lambda: do_phase(False))
        D.need(len(cells) == 666 and D.sha(D.R11_PATH) == D.R11_SHA, 'Count/R11 changed')
        # Re-check all frozen source/code/endpoint hashes after processing.
        load_scope(expected_sha)
        D.write_once_json(RESULT, {'status': 'complete', 'scope_sha256': expected_sha,
                                  'original_screen_passed': False, 'cells': cells,
                                  'seconds': time.time()-t0,
                                  'note': 'Raw scalar sensitivity evidence only. The separate verifier builds all summaries; no best condition or model is selected.'})
        print('CONDITION_PROBE_COMPLETE',len(cells),flush=True)
    except BaseException as exc:
        D.write_once_json(RESULT, {'status': 'FAILED_REPRODUCTION' if phase == 'actual' else 'VOIDED',
                                  'scope_sha256': expected_sha, 'original_screen_passed': False,
                                  'completed_cells': len(cells), 'error': type(exc).__name__+': '+str(exc)})
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('command', choices=('prepare', 'run'))
    ap.add_argument('--scope-sha');args = ap.parse_args()
    prepare() if args.command == 'prepare' else run(args.scope_sha)
