"""Stdlib-only staging and CPU submission; no login-node numerical analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,data):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent
    if (root/'request.json').exists():raise RuntimeError('Already staged: do not duplicate jobs')
    upstream=Path(json.loads((root.parent/'line_mass_L32x64_allconf_B_20261015/latest.json').read_text())['root'])
    files=sorted(f for name in ['code','src','reference','slurm'] for f in (root/name).rglob('*')
        if f.is_file() and f.suffix in ['.py','.cpp','.hpp','.h','.slurm'])
    # Parse, but do not execute or import numerical modules on the login node.
    for f in files:
        if f.suffix=='.py':compile(f.read_text(),str(f),'exec')
        if f.suffix=='.slurm':subprocess.run(['bash','-n',str(f)],check=True)
    write(root/'request.json',dict(created=datetime.now().astimezone().isoformat(),
        upstream_root=str(upstream),upstream_manifest_sha256=digest(upstream/'manifest.json'),
        code_sha256={str(f.relative_to(root)):digest(f) for f in files},
        Nbootstrap=2000,seed=20261016,scope='common full-grid completed snapshot selected by CPU gate'))
    if not a.submit:print('STAGED',root);return
    jobs=[]
    for name in ['snapshot_cpu.slurm','bootstrap_fit_cpu.slurm','profile_verify_cpu.slurm']:
        command=['sbatch','--parsable','--chdir='+str(root),
            '--export=ALL,LOOP_ANALYSIS_ROOT='+str(root),
            '--output='+str(root/'logs'/(name+'_%j.out')),'--error='+str(root/'logs'/(name+'_%j.err'))]
        if jobs:command.append('--dependency=afterok:'+jobs[-1]['job_id'])
        command.append(str(root/'slurm'/name))
        result=subprocess.check_output(command,text=True).strip().split(';')[0]
        if not result.isdigit():raise RuntimeError('unexpected sbatch response: '+result)
        jobs.append(dict(stage=name,job_id=result,command=command))
        write(root/'submission.json',dict(jobs=jobs,created=datetime.now().astimezone().isoformat()))
        print('SUBMITTED',name,result,flush=True)


if __name__=='__main__':main()
