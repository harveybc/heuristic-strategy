"""Finite unattended specialist queue, publication and independent collection."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

SCHEMA='modular_controlled_campaign.v1'


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    if hasattr(x,'isoformat'):return x.isoformat()
    if hasattr(x,'item'):return clean(x.item())
    return x


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(clean(body),f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        d=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:
        if os.path.exists(name):os.unlink(name)


def identity():
    root=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(root),'diff','HEAD','--','app','tools'],text=True)
    if dirty:raise ValueError('RUNTIME_TRACKED_SOURCE_DIRTY')
    return sha


def schedule():
    return [dict(id=f'{a.lower()}-{s}',arm=a,seed=s) for s in range(20261019,20261024)
            for a in ('JOINT16','JOINT_MATCHED','FULL')]


def prepare(root,config):
    root=Path(root).resolve();cfg=read(config)
    if root.is_relative_to(Path(__file__).resolve().parents[1]):raise ValueError('PRIVATE_OUTPUT_REQUIRED')
    from app.short_horizon.expert_guided import data as DG
    pins={str(Path(cfg['admitted'])/n):DG.PINS[k] for n,k in
          [('eurusd.parquet','admitted_sha256'),('CLOSURE_INTERVALS.json','closure_sha256')]}
    pins.update({str(Path(cfg['csv_dir'])/f'{role}.csv'):DG.PINS[role+'_csv_sha256'] for role in ('train','validation')})
    for p,h in pins.items():
        if digest(p)!=h:raise ValueError('INPUT_PIN_MISMATCH')
    spec=dict(cfg,schema=SCHEMA,runtime=identity(),cells=schedule(),inputs=pins,max_attempts=2,
              max_cell_seconds=10800,created=time.time())
    if (root/'SPEC.json').exists():raise ValueError('CAMPAIGN_ALREADY_EXISTS')
    write(root/'SPEC.json',spec)
    write(root/'STATE.json',dict(spec_sha256=digest(root/'SPEC.json'),cells=[],state='READY'))
    return spec


def load_spec(root):
    spec=read(root/'SPEC.json')
    if spec['schema']!=SCHEMA or spec['runtime']!=identity() or spec['cells']!=schedule():
        raise ValueError('CAMPAIGN_IDENTITY_MISMATCH')
    if read(root/'STATE.json')['spec_sha256']!=digest(root/'SPEC.json'):raise ValueError('STATE_IDENTITY_MISMATCH')
    for p,h in spec['inputs'].items():
        if digest(p)!=h:raise ValueError('INPUT_IDENTITY_MISMATCH')
    spec['spec_sha256']=digest(root/'SPEC.json')
    return spec


def verify_close(path,cell,spec_pin):
    close=read(path/'CLOSE.json')
    if close['cell']!=cell or close['spec_sha256']!=spec_pin:raise ValueError('CLOSE_IDENTITY_MISMATCH')
    for rel,h in close['files'].items():
        p=(path/rel).resolve()
        if not p.is_relative_to(path.resolve()) or digest(p)!=h:raise ValueError('CLOSE_ARTIFACT_MISMATCH')
    return read(path/'RESULT.json')


def resources_ready(spec,root):
    mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')}
    if mem['MemAvailable']<5*1024**3:return 'WAIT_RAM'
    if shutil.disk_usage(root).free<10*1024**3:return 'WAIT_DISK'
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    if not any(spec['gpu_uuid'] in line and '5090' in line for line in rows.splitlines()):return 'WAIT_ASSIGNED_GPU'
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(spec['gpu_uuid'] in line for line in procs.splitlines()):return 'WAIT_GPU_BUSY'
    return 'READY'


def summarize(root,state):
    rows=[];durations=[]
    for rec in state['cells']:
        if rec['state']!='CLOSED':continue
        out=Path(rec['attempts'][-1]['out'])
        r=verify_close(out,rec['cell'],state['spec_sha256']);durations.append(r['wall_seconds'])
        for control,periods in r['metrics'].items():
            for period,met in periods.items():
                rows.append(dict(**rec['cell'],control=control,period=period,**met,
                                 parameters=r['architecture']['total'],config_sha256=r['config_sha256']))
    write(root/'RESULTS.json',dict(rows=rows))
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        fd,name=tempfile.mkstemp(dir=root,prefix='.results')
        with os.fdopen(fd,'w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
        os.replace(name,root/'RESULTS.csv')
    left=len(schedule())-sum(r['state'] in ('CLOSED','FAILED_EXHAUSTED') for r in state['cells'])
    eta=None if not durations or state['state'].startswith(('WAIT','BLOCKED')) else [left*min(durations),left*max(durations)]
    write(root/'STATUS.json',dict(state=state['state'],complete=len(durations),total=len(schedule()),
        failed=sum(r['state']=='FAILED_EXHAUSTED' for r in state['cells']),eta_seconds=eta,updated=time.time(),
        active=next((r for r in state['cells'] if r['state']=='RUNNING'),None)))


def work(root,execute=None,ready=None):
    root=Path(root).resolve()
    with (root/'.worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 0
        try:
            spec=load_spec(root);state=read(root/'STATE.json')
            for cell in spec['cells']:
                rec=next((r for r in state['cells'] if r['cell']==cell),None)
                if rec is None:
                    rec=dict(cell=cell,state='READY',attempts=[]);state['cells'].append(rec)
                if rec['state'] in ('CLOSED','FAILED_EXHAUSTED'):continue
                if rec['attempts']:
                    last=rec['attempts'][-1];out=Path(last['out'])
                    if (out/'CLOSE.json').exists():
                        verify_close(out,cell,state['spec_sha256']);rec['state']='CLOSED'
                        write(root/'STATE.json',state);continue
                    # Lock released with a RUNNING record means the old worker died.
                    if last.get('state')=='RUNNING':
                        last.update(state='INTERRUPTED',finished=time.time())
                        write(out/'FAILURE.json',dict(reason='PROCESS_INTERRUPTED',partial=True))
                if len(rec['attempts'])>=spec['max_attempts']:
                    rec['state']='FAILED_EXHAUSTED';write(root/'STATE.json',state);continue
                resource=(ready or resources_ready)(spec,root)
                if resource!='READY':
                    state['state']=resource;write(root/'STATE.json',state);summarize(root,state);return 0
                out=root/'runs'/cell['id']/f'a{len(rec["attempts"])+1}'
                out.mkdir(parents=True,exist_ok=False)
                attempt=dict(out=str(out),state='RUNNING',started=time.time());rec['attempts'].append(attempt)
                rec['state']=state['state']='RUNNING';write(root/'STATE.json',state);summarize(root,state)
                rc=1
                try:
                    if execute:rc=execute(spec,cell,out)
                    else:
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':spec['gpu_uuid'],'OMP_NUM_THREADS':'1',
                             'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
                        with (out/'worker.log').open('a') as log:
                            rc=subprocess.run([sys.executable,'-m','tools.modular_controlled','--root',str(root),
                                '--cell',cell['id'],'--out',str(out)],env=env,stdout=log,stderr=log,
                                timeout=spec['max_cell_seconds']).returncode
                except subprocess.TimeoutExpired:rc=124
                attempt.update(returncode=rc,finished=time.time())
                if rc==78:
                    raise ValueError('SCIENTIFIC_CHILD_INTEGRITY_FAILURE:'+cell['id'])
                if rc==0 and (out/'CLOSE.json').exists():
                    verify_close(out,cell,state['spec_sha256']);rec['state']=attempt['state']='CLOSED'
                else:
                    attempt['state']='FAILED';rec['state']='RETRY_PENDING'
                    write(out/'FAILURE.json',dict(returncode=rc,partial=True))
                state['state']=rec['state'];write(root/'STATE.json',state);summarize(root,state)
                return 0
            state['state']='SCIENTIFIC_COMPLETE' if all(r['state']=='CLOSED' for r in state['cells']) else 'COMPLETE_WITH_FAILURES'
            write(root/'STATE.json',state);summarize(root,state)
            write(root/'SCIENTIFIC_COMPLETE.json',dict(state=state['state'],state_sha256=digest(root/'STATE.json')))
            return 0
        except (ValueError,KeyError) as exc:
            write(root/'BLOCKED.json',dict(state='BLOCKED_INTEGRITY',reason=str(exc),time=time.time()))
            raise


PUBLISH_CHILD='''
import json,sys
from tools.rl_warehouse import invoke
spec=json.load(open(sys.argv[1]))
print(json.dumps(invoke(spec['publication'],sys.argv[2],sys.argv[3],sys.argv[4])))
'''


def publish(root):
    root=Path(root).resolve()
    with (root/'.publish.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        spec=load_spec(root);state=read(root/'STATE.json');pub=spec['publication']
        expected=0
        def invoke(action,request):
            p=root/'publication-request.json';write(p,request)
            r=subprocess.run([sys.executable,'-c',PUBLISH_CHILD,str(root/'SPEC.json'),str(root),action,str(p)],
                cwd=spec['agent_runtime'],capture_output=True,text=True,check=True,timeout=360,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':spec['agent_runtime']})
            return json.loads(r.stdout.strip().splitlines()[-1])
        try:
            for rec in state['cells']:
                if rec['state']!='CLOSED':continue
                out=Path(rec['attempts'][-1]['out']);result=verify_close(out,rec['cell'],state['spec_sha256'])
                for control,periods in result['metrics'].items():
                    for period,metrics in periods.items():
                        key=rec['cell']['id']+'-'+control+'-'+period
                        source_role='validation' if period=='OBS_2024' else 'train'
                        split='train' if period=='TRAIN' else 'validation'
                        report=dict(experiment_key='mc-'+state['spec_sha256'][:12]+'-'+key,experiment_set_key='modular-controlled-'+state['spec_sha256'][:12],
                            actor=pub['actor'],lake=pub['lake'],lineage='UNVERIFIED',project='heuristic-strategy',
                            phase='MODULAR_CONTROLLED_DEVELOPMENT',config_sha256=result['config_sha256'],code_commit=spec['runtime'],
                            tags=dict(period=period,control=control,arm=rec['cell']['arm'],seed=str(rec['cell']['seed']),
                                campaign_sha256=state['spec_sha256'],result_sha256=digest(out/'RESULT.json'),
                                costs='ASSUMED_NOT_BROKER_VERIFIED',algorithm='BC_DAGGER_NOT_SAC'),
                            datasets=[dict(lake=pub['lake'],resource=pub['resources'][source_role],
                                sha256=spec['inputs'][str(Path(spec['csv_dir'])/(source_role+'.csv'))],
                                role=split,lineage='UNVERIFIED',reason='Pinned admitted local source')],
                            metrics=[dict(metric=k,value=v,split=split,horizon=None,unit=('USD' if k.endswith('_usd') else None))
                                for k,v in metrics.items() if v is None or type(v) in (int,float)])
                        invoke('enqueue',dict(report=report,source_unit=key));expected+=1
            status=invoke('pump',{k:pub[k] for k in ('warehouse_url','token_env')})
            write(root/'PUBLICATION_STATUS.json',dict(expected=expected,**status))
            wh=status['warehouse']
            if (root/'SCIENTIFIC_COMPLETE.json').exists() and wh['pending']==wh['mismatch']==0 and wh['delivered']==expected:
                write(root/'PUBLICATION_COMPLETE.json',dict(expected=expected,**status))
        except (subprocess.SubprocessError,OSError) as exc:
            write(root/'PUBLICATION_STATUS.json',dict(state='RETRY_PENDING',error_type=type(exc).__name__,expected=expected))


def collect(source,remote_root,destination):
    """Runs on the independent storage machine; validates every completed run's manifest."""
    dst=Path(destination).resolve();dst.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(dst).free<5*1024**3:raise ValueError('COLLECT_SPACE_REQUIRED')
    with (dst/'.collection.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        # Metadata first; copy only closed immutable child directories.
        names=('STATE.json','SPEC.json','STATUS.json','RESULTS.json','RESULTS.csv',
               'SCIENTIFIC_COMPLETE.json','PUBLICATION_STATUS.json','PUBLICATION_COMPLETE.json')
        cmd=['rsync','-rt','--ignore-missing-args','--protect-args']
        subprocess.run(cmd+[f'{source}:{remote_root}/{n}' for n in names]+[str(dst)+'/'],check=True,timeout=120)
        state=read(dst/'STATE.json');done=0
        for rec in state['cells']:
            if rec['state']!='CLOSED':continue
            remote=rec['attempts'][-1]['out'];local=dst/'runs'/rec['cell']['id'];local.mkdir(parents=True,exist_ok=True)
            subprocess.run(cmd+[source+':'+remote+'/',str(local)+'/'],check=True,timeout=180)
            verify_close(local,rec['cell'],state['spec_sha256']);done+=1
        write(dst/'COLLECTION_STATUS.json',dict(copied=done,total=len(schedule()),time=time.time()))
        if (dst/'SCIENTIFIC_COMPLETE.json').exists() and (dst/'PUBLICATION_COMPLETE.json').exists():
            if done != sum(r['state']=='CLOSED' for r in state['cells']):raise ValueError('COLLECTION_INCOMPLETE')
            write(dst/'COLLECTION_COMPLETE.json',dict(copied=done,state=state['state'],spec_sha256=state['spec_sha256']))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','work','publish','status','collect'])
    p.add_argument('--root');p.add_argument('--config');p.add_argument('--source');p.add_argument('--remote-root');p.add_argument('--destination')
    a=p.parse_args();root=Path(a.root).resolve() if a.root else None
    if a.action=='prepare':prepare(root,a.config)
    elif a.action=='work':work(root)
    elif a.action=='publish':publish(root)
    elif a.action=='collect':collect(a.source,a.remote_root,a.destination)
    else:
        state=read(root/'STATUS.json') if (root/'STATUS.json').exists() else read(root/'STATE.json')
        active=state.get('active')
        if active:
            progress=Path(active['attempts'][-1]['out'])/'PROGRESS.json'
            state['progress']=read(progress) if progress.exists() else None
        print(json.dumps(state,indent=2))

if __name__=='__main__':main()
