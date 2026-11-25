"""One-shot local job continuation: verified inputs -> frozen study -> all six runs.

Never changes a model/data/metric setting or opens protected tiers. Any
failed integrity/process check stops. Git publication is limited to the
specific generated evidence for the already-authorized research stage.
No deployment, model selection, cloud compute or recurring scheduling.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import neutral_wb_study_data as D

LOCAL=D.TRAIN/'coordination/.local'
STATE=LOCAL/'neutral-wb-study-pipeline-state.json'
LOG=LOCAL/'neutral-wb-study-pipeline.log'


def state(stage,**extra):
    obj={'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),**extra}
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,STATE)
    print(stage,json.dumps(extra),flush=True)


def run(args):
    with LOG.open('a',buffering=1) as log:
        print('\nCOMMAND '+repr(args),file=log,flush=True)
        p=subprocess.run(args,cwd=D.ROOT,stdout=log,stderr=subprocess.STDOUT)
    D.need(p.returncode==0,'Command failed: '+repr(args)+'; see '+str(LOG))


def git(*args):
    return subprocess.check_output(['git',*args],cwd=D.ROOT,text=True).strip()


def published(path):
    rel=str(path.relative_to(D.ROOT))
    p=subprocess.run(['git','show','HEAD:'+rel],cwd=D.ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    return p.returncode==0 and hashlib.sha256(p.stdout).hexdigest()==D.sha(path)


def publish(paths,message):
    rel=[str(p.relative_to(D.ROOT)) for p in paths]
    D.need(not git('diff','--cached','--name-only'),'Unrelated staged changes; publication refused')
    for p in paths:D.need(p.is_file(),'Evidence missing before publication: '+str(p))
    run(['git','add','--',*rel])
    staged=set(filter(None,git('diff','--cached','--name-only').splitlines()))
    D.need(staged<=set(rel),'Publication escaped exact evidence allowlist')
    if staged:run(['git','commit','-m',message])
    run(['git','push','origin','master'])


def verify_development_receipt():
    p=D.DEV_REPORT;r=D.read(p)
    D.need(r.get('verified') is True and r.get('canary_only') is False and r.get('completed_development_rows')==116
           and r.get('manifest_sha256')==D.DEV_SHA,'Development input qualification not complete')
    D.need(r.get('verification_script_sha256')==D.sha(D.TRAIN/'verify_neutral_wb_development_export.py'), 'Input verifier code drift')
    m=D.E.load(D.DEV_SHA)
    D.need({s['name'] for s in r['rows']}=={s['name'] for s in m['scenes']} and len(r['rows'])==116
           and all(s.get('verified') is True for s in r['rows']),'Development qualification coverage differs')
    for s in m['scenes']:
        seal=D.E.OUT/s['shoot']/'scene-seal.json'
        D.need(D.sha(seal)==r['scene_seal_sha256'][s['name']],'Development seal changed after verification')
    D.need(D.sha(D.R11_PATH)==D.R11_SHA,'Selected R11 changed')


def wait_inputs():
    while not D.DEV_REPORT.exists():
        p=LOCAL/'neutral-wb-development-pipeline-state.json'
        if p.exists():
            obj=D.read(p);D.need('FAILED' not in obj.get('stage',''),'Development job failed: '+repr(obj))
            pid=obj.get('pipeline_pid')
            if type(pid) is int:
                try:os.kill(pid,0)
                except ProcessLookupError:D.need(False,'Development process exited without a verification receipt')
        state('WAITING_DEVELOPMENT_INPUT_VERIFICATION')
        time.sleep(20)
    verify_development_receipt()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    D.need(a.run,'Explicit --run is required for this one-shot research continuation')
    LOCAL.mkdir(parents=True,exist_ok=True)
    power=subprocess.Popen(['/usr/bin/caffeinate','-dimsu','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        # New code must already be reviewed, tested and published by the root agent.
        for name in ('neutral_wb_study_data.py','prepare_neutral_wb_study.py','run_neutral_wb_study.py',
                     'eval_neutral_wb_study.py','verify_neutral_wb_study.py','continue_neutral_wb_study.py'):
            D.need(D.committed(D.TRAIN/name),'Unpublished study implementation: '+name)
        wait_inputs();state('DEVELOPMENT_INPUTS_VERIFIED')
        if not published(D.DEV_REPORT):publish([D.DEV_REPORT],'Publish complete development WB input qualification')
        if not D.MANIFEST.exists():
            state('PREPARING_FROZEN_STUDY')
            run([sys.executable,'-u','training/prepare_neutral_wb_study.py','prepare'])
            publish([D.MANIFEST],'Freeze fresh paired WB study data, orders and fixed evaluation limits')
        D.need(published(D.MANIFEST),'Unpublished prior study manifest; investigate before continuing')
        sha=D.sha(D.MANIFEST);m=D.load_manifest(sha);state('STUDY_FROZEN',manifest_sha256=sha)
        if not D.PREFLIGHT.exists():
            state('FULL_PAIR_PREFLIGHT_STARTED',manifest_sha256=sha)
            run([sys.executable,'-u','training/prepare_neutral_wb_study.py','preflight','--manifest-sha',sha])
        p=D.read(D.PREFLIGHT)
        D.need(p.get('passed') is True and p.get('pairs')==571 and p.get('identity_pairs')==38
               and p.get('manifest_sha256')==sha,'Full pair preflight short/failed/stale')
        smoke=D.DIAG/'neutral-wb-study-smoke.json'
        if not smoke.exists():
            state('DISPOSABLE_SMOKE_STARTED',manifest_sha256=sha)
            run([sys.executable,'-u','training/run_neutral_wb_study.py','smoke','--manifest-sha',sha])
        import run_neutral_wb_study as T
        s=D.read(smoke)
        D.need(s.get('passed') is True and s.get('manifest_sha256')==sha and s.get('trainer_sha256')==D.sha(T.CODE)
               and s.get('data_api_sha256')==D.sha(T.DATA_API),'Smoke receipt invalid')
        D.need(all(s['resume'].get(k) is True for k in ('equal','stale_refused','unknown_refused')),'CPU resume smoke failed')
        D.need(set(s['arms'])==set(D.ARMS) and all(len(x['losses'])==2 and all(type(v) in (int,float) and math.isfinite(v) for v in x['losses']) for x in s['arms'].values()),'Smoke incomplete/nonfinite')
        if not published(D.PREFLIGHT) or not published(smoke):
            publish([D.PREFLIGHT,smoke],'Record full paired-data preflight and disposable local training smoke')
        T.check_proofs(m)
        state('SIX_RUN_STUDY_LAUNCHING',manifest_sha256=sha,note='Actual first optimizer update is separately marked TRAINING_STARTED in the log.')
        run([sys.executable,'-u','training/run_neutral_wb_study.py','all','--manifest-sha',sha])
        state('ALL_SIX_ENDPOINTS_AND_SCORES_COMPLETE',manifest_sha256=sha)
        verification=D.DIAG/'neutral-wb-study-verification.json'
        if not verification.exists():
            run([sys.executable,'-u','training/verify_neutral_wb_study.py','--manifest-sha',sha])
        v=D.read(verification)
        D.need(v.get('verified') is True and v.get('manifest_sha256')==sha
               and v.get('verification_script_sha256')==D.sha(D.TRAIN/'verify_neutral_wb_study.py'),'Study verification missing/changed')
        import eval_neutral_wb_study as E
        D.need(v.get('report_sha256')==D.sha(E.final_path()),'Final report changed after verification')
        evidence=[verification,E.final_path()]
        evidence += [E.eval_path(seed,arm) for seed in D.SEEDS for arm in D.ARMS]
        publish(evidence,'Record bounded paired WB study results and independent canary verification')
        state('STUDY_COMPLETE_VERIFIED_NO_MODEL_SELECTED',manifest_sha256=sha,screen_passed=v['screen_passed'])
        peer=LOCAL/'focused-claude-peer-session.json'
        if peer.exists():
            sid=D.read(peer)['session_id']
            prompt=('Codex completion handoff: the fixed six-run neutral-WB study and independent verifier completed. '
                    'Read training/runs/neutral-wb-study-report.json and training/diagnostics/neutral-wb-study-verification.json '
                    'and frozen study manifest metadata/code ONLY. No media, targets, RAW/TIFFs, checkpoints/weights, '
                    'new training, inference, git or edits. Summarize actual five criterion outcomes, per-seed G vsneutral '
                    'development means, treated79/identity37/DR groups and limits. Native input qualification is not Fuji '
                    'accuracy. Current R11 remains selected, protected K/monitor/finalHeldOut/T8/dr100Eval unread. '
                    'Recommend the next bounded diagnostic from measured evidence; do not launch/select/integrate anything. '
                    'No human relay is needed. Report clear failure concerns if any.')
            (LOCAL/'neutral-wb-study-completion-to-claude.txt').write_text(prompt)
            with (LOCAL/'neutral-wb-study-claude-review.json').open('w') as out,(LOCAL/'neutral-wb-study-claude-review.stderr').open('w') as err:
                p=subprocess.run(['/Users/phelomehanny/.vscode/extensions/anthropic.claude-code-2.1.326-darwin-arm64/resources/native-binary/claude',
                    '--print','--resume',sid,'--output-format','json','--tools','Read,Grep,Glob','--allowedTools','Read,Grep,Glob',
                    '--permission-prompts','none',prompt],cwd=D.ROOT,stdout=out,stderr=err)
            state('STUDY_COMPLETE_CLAUDE_REVIEW_SAVED' if p.returncode==0 else 'STUDY_COMPLETE_CLAUDE_REVIEW_PENDING',
                  manifest_sha256=sha,screen_passed=v['screen_passed'],claude_returncode=p.returncode)
    except BaseException as exc:
        state('STOPPED',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if power.poll() is None:power.terminate()


if __name__=='__main__':main()
