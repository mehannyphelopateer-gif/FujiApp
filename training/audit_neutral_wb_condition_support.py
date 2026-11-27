"""Post-hoc metadata support audit and synthetic condition-plane wiring check.

No source pixels, trained checkpoints, protected tiers, fitting or recipe
scoring. Rejected-attempt JSON is candidate inventory only, not authority
to reuse excluded photos or change acceptance criteria.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path
import torch
import neutral_wb_study_data as D

ROOT = D.ROOT
PARENT_SHA = 'd416942f282ec8d1fbe65fad6ad6a4e3c30d612d7938cc6052bc364885b018d0'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    m = D.load_manifest(PARENT_SHA)
    fit = [s for s in m['scenes'] if s['role'] == 'fit']
    dev = [s for s in m['scenes'] if s['role'] == 'development']
    D.need(len(fit) == 455 and len(dev) == 116, 'Unexpected study roles/counts')
    identities = [s for s in dev if s['neutral_kind'] == 'G' and s['policy_subgroup'] == 'identity']
    D.need(len(identities) == 37, 'Identity cohort differs')
    fit_names = {s['name'] for s in fit};dev_names = {s['name'] for s in dev}
    session_overlap = {s['session'] for s in fit} & {s['session'] for s in dev}
    D.need(not fit_names & dev_names and not session_overlap, 'Split leakage')
    D.need(not any(s['session'] == 14 for s in fit), 'Session14 leaks into FIT')
    support = Counter(math.log2(s['captureDynamicRange']/100)/2 for s in fit)
    D.need(support == Counter({0.5:33, 1.0:422}), 'Recorded support differs')
    by_name = {s['name']:s for s in fit};per_run = {}
    for seed in D.SEEDS:
        orders = m['orders'][str(seed)]
        D.need(len(orders) == 20 and all(len(o) == 455 and set(o) == fit_names for o in orders), 'Orders incomplete')
        seen = Counter(math.log2(by_name[name]['captureDynamicRange']/100)/2 for o in orders for name in o)
        for arm in D.ARMS:
            p = ROOT/f'training/checkpoints/neutral-wb-study-seed{seed}-{arm}/endpoint.json'
            meta = json.loads(p.read_text())['meta']
            D.need(meta['epoch'] == 20 and meta['global_updates'] == 9100, 'Endpoint budget incomplete')
            per_run[f'{seed}/{arm}'] = {'unique_condition_counts':dict(sorted((str(k),v) for k,v in support.items())),
                                       'condition_update_counts':dict(sorted((str(k),v) for k,v in seen.items())),
                                       'endpoint_record_sha256':digest(p)}

    # Fresh random weights + uniform synthetic RGB, never a trained model or photo.
    model = D.fresh_model(m['model_config'], 42);model.eval()
    before = D.state_digest(model.state_dict());planes = []
    def hook(_layer, args):
        x = args[0]
        D.need(x.shape[1] == 4, 'Expected RGB plus one condition channel')
        planes.append({'shape':list(x.shape),'plane_min':float(x[:,3].min()),'plane_max':float(x[:,3].max())})
    handle = model.grid.encoder[0][0].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for c in (0.0,0.5,1.0):
                pred = D.forward(model,m,torch.full((1,3,9,13),0.2),torch.tensor([c]),torch.device('cpu'))
                D.need(bool(torch.isfinite(pred).all()) and planes[-1]['plane_min'] == planes[-1]['plane_max'] == c,
                       'Condition did not reach the actual encoder channel')
    finally:handle.remove()
    D.need(D.state_digest(model.state_dict()) == before, 'Synthetic check changed model weights')

    # Existing quarantined tag dump ONLY; RAW contents are not opened.
    dump = ROOT/'calibration-input/phase4-dr100eval-attempt1-excluded/exif.json'
    candidates = {'available':dump.exists(),'status':'NOT_APPROVED_FOR_REUSE; old acceptance exclusion unchanged'}
    if dump.exists():
        old = json.loads(dump.read_text())
        D.need(len(old) == 18 and len({s['name'] for s in old}) == 18, 'Rejected metadata coverage differs')
        allowed = {f'DSCF{i}.RAF' for i in range(1424,1442)}
        D.need({s['name'] for s in old} == allowed and not allowed & (fit_names | dev_names), 'Candidate identity/leakage issue')
        groups = {}
        for dr in (100,200):
            rows = [s for s in old if s.get('fuji',{}).get('AutoDynamicRange','').split()[0] == str(dr)]
            D.need(len(rows) == 9, 'Stored AutoDR resolution count differs')
            for s in rows:
                D.need(s['path'] == f"calibration-input/phase4-dr100eval-attempt1-excluded/{s['name']}"
                       and (ROOT/s['path']).is_file(), 'Candidate original missing/path differs')
            groups[str(dr)] = {'count':len(rows),'source_names':[s['name'] for s in rows],
                                'stored_ISO_counts':dict(Counter(str(s['iso']) for s in rows)),
                                'stored_WB_mode_counts':dict(Counter(s['fuji'].get('WhiteBalance','UNKNOWN') for s in rows))}
        candidates.update(metadata_json_sha256=digest(dump),groups=groups,
                          stored_capture_time_range=[min(s['dateTimeOriginal'] for s in old),max(s['dateTimeOriginal'] for s in old)],
                          limits='Stored metadata only, not re-extracted from originals here. RAW bytes/targets untouched. These18still fail their original fixed-DR/multisession acceptance protocol;9Auto-resolvedDR100lowISO sources are only possible future TRAIN candidates under a separately reviewed scope.')

    cross = {}
    for role,rows in [('fit',fit),('development',dev)]:
        cross[role] = {'capture_DR_counts':dict(Counter(str(s['captureDynamicRange']) for s in rows)),
                       'session_counts':dict(Counter(str(s['session']) for s in rows)),
                       'DR_and_policy_counts':dict(Counter(f"{s['captureDynamicRange']}/{s['neutral_kind']}" for s in rows)),
                       'ISO':'UNKNOWN_NOT_RECORDED_IN_FROZEN_STUDY_METADATA',
                       'lighting':'UNKNOWN_NOT_RECORDED_IN_FROZEN_STUDY_METADATA'}
    output = {'post_hoc':True,'scope':'Metadata + fresh-model synthetic RGB only; no source-image inference',
              'manifest_sha256':PARENT_SHA,'script_sha256':digest(Path(__file__)),
              'data_api_sha256':digest(Path(D.__file__)),'model_code_sha256':digest(ROOT/'training/model.py'),
              'condition_source':'captureDynamicRange, log2(captureDynamicRange/100)/2; not TIFF output DR',
              'condition_constructor_source':inspect.getsource(D.Pairs.get).split('cond=')[-1].strip(),
              'measured_condition_support':{'unique_counts':dict(sorted((str(k),v) for k,v in support.items())),
                                            'min':min(support),'max':max(support),'DR100_FIT_count':0},
              'per_run':per_run,'metadata_cross_tabs':cross,
              'leakage_checks':{'name_overlap':[],'session_overlap':[],'session14_FIT_count':0},
              'synthetic_actual_encoder_condition_planes':planes,'synthetic_weights_unchanged':True,
              'rejected_attempt_inventory':candidates,'R11_sha256':D.sha(D.R11_PATH),
              'decision':'No remap or new training. Scope a DR100 training-support/data-qualification plan first;9existing quarantined sources may reduce new-photo needs but are not admitted or exported here.',
              'limits':['Confirms missing recorded FIT support, not a causal explanation of the colour/brightness error.',
                        'Synthetic hooks establish scalar delivery to the actual first convolution, not a second independent network/colour implementation.',
                        'The old sensitivity evidence and its code are unchanged; its canaries remain shared-forward numerical reproducibility checks.',
                        'No protected K/monitor/finalHeldOut/T8/dr100Eval access, new fitting, app change or screen revision.']}
    D.need(output['R11_sha256'] == D.R11_SHA, 'R11 changed')
    out = ROOT/'training/diagnostics/neutral-wb-condition-support-audit.json'
    D.write_once_json(out,output)
    print('CONDITION_SUPPORT_AUDIT_COMPLETE',out,flush=True)


if __name__ == '__main__':main()
