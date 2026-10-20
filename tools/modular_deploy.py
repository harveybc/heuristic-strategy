"""Render the authorized finite study's private configuration and systemd units."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from tools.modular_campaign import read, write


def unit(description,command,cwd,condition,environment_file=None,memory='7G',block=None):
    lines=['[Unit]','Description='+description,'ConditionPathExists=!'+str(condition)]
    if block:lines+=['ConditionPathExists=!'+str(block)]
    lines+=['[Service]','Type=oneshot','WorkingDirectory='+str(cwd),
        'Environment=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'ExecStart='+shlex.join(command),'TimeoutStartSec=4h','KillMode=control-group',
        'MemoryMax='+memory,'MemorySwapMax=512M','Nice=5']
    if environment_file:lines+=['EnvironmentFile='+str(environment_file)]
    return '\n'.join(lines)+'\n'


def timer(name,seconds):
    return f'[Unit]\nDescription=Finite {name} scheduler\n[Timer]\nOnBootSec=45s\nOnUnitInactiveSec={seconds}s\nUnit={name}.service\n[Install]\nWantedBy=timers.target\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','render','render-collector'])
    p.add_argument('--root',required=True);p.add_argument('--baseline');p.add_argument('--admitted');p.add_argument('--agent-runtime')
    p.add_argument('--environment-file');p.add_argument('--source');p.add_argument('--remote-root');p.add_argument('--destination')
    a=p.parse_args();root=Path(a.root).resolve();cwd=Path(__file__).resolve().parents[1]
    units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','tools.modular_campaign']
    if a.action=='prepare':
        base=Path(a.baseline).resolve();publication=read(base/'CONTRACT.json')['publication']
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
        gpu=[l.split(',')[0].strip() for l in rows.splitlines() if '5090' in l]
        if len(gpu)!=1:raise ValueError('EXACT_ASSIGNED_GPU_REQUIRED')
        config=dict(csv_dir=str(base),admitted=str(Path(a.admitted).resolve()),gpu_uuid=gpu[0],
                    publication=publication,agent_runtime=str(Path(a.agent_runtime).resolve()))
        write(root/'DEPLOY.private.json',config)
        from tools.modular_campaign import prepare
        prepare(root,root/'DEPLOY.private.json')
    elif a.action=='render':
        for suffix,action,seconds,condition in [('worker','work',15,'SCIENTIFIC_COMPLETE.json'),
                                               ('publication','publish',120,'PUBLICATION_COMPLETE.json')]:
            name='rl-modular-controlled-'+suffix
            service=unit('Controlled modular study '+suffix,cmd+[action,'--root',str(root)],cwd,root/condition,
                a.environment_file if suffix=='publication' else None,
                memory='7G' if suffix=='worker' else '1500M',block=root/'BLOCKED.json')
            (units/(name+'.service')).write_text(service)
            (units/(name+'.timer')).write_text(timer(name,seconds))
    else:
        dst=Path(a.destination).resolve();name='rl-modular-controlled-collection'
        service=unit('Verified specialist artifact collection',cmd+['collect','--source',a.source,
            '--remote-root',a.remote_root,'--destination',str(dst)],cwd,dst/'COLLECTION_COMPLETE.json',memory='1500M')
        (units/(name+'.service')).write_text(service)
        (units/(name+'.timer')).write_text(timer(name,180))

if __name__=='__main__':main()
