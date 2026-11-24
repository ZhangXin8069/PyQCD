"""Finish submission under AssocMaxSubmitJobLimit without editing frozen core."""
from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')


def main():
    ast.parse(Path(__file__).read_text());subprocess.run(['bash','-n',str(ROOT/'slurm/run_packed_tail_v2.sh')],check=True)
    source=json.loads((ROOT/'source_lock.json').read_text())
    for p,h in source['files_sha256'].items():assert sha(p)==h,p
    lock=ROOT/'packed_tail_lock_v2.json'
    files={str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'slurm/run_packed_tail_v2.sh')}
    if lock.exists():assert json.loads(lock.read_text())['files_sha256']==files
    else:dump(lock,dict(created=datetime.now().astimezone().isoformat(),files_sha256=files,reason='AssocMaxSubmitJobLimit; pack logical tasks into 5 CPU allocations; core numerical code unchanged'))
    p=ROOT/'submission.json';record=json.loads(p.read_text());jobs=record['jobs']
    record['packed_tail_lock_sha256']=sha(lock)
    record['created']='2026-11-08'  # Date-only: exact sbatch timestamps are available in Slurm.
    def submit(label,stage,key,dep,cpus=8,mem='24G'):
        if label in jobs:return jobs[label]['jobid']
        cmd=['sbatch','--parsable','--chdir='+str(ROOT),'--job-name='+label,'--cpus-per-task='+str(cpus),
            '--mem='+mem,'--dependency=afterok:'+dep,'--output='+str(ROOT/'logs/%x_%j.out'),
            '--error='+str(ROOT/'logs/%x_%j.err'),str(ROOT/'slurm/run_packed_tail_v2.sh'),stage,key]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit(),job
        jobs[label]=dict(jobid=job,stage=stage,ensemble=key,dependency=dep,command=cmd,
            logical_tasks=63 if stage=='fit' else 9 if stage in ('analyze','plot') else 1,
            packed_cpu_tasks=True,array=None,cpus_per_task=cpus)
        record['last_submission']=datetime.now().astimezone().isoformat();dump(p,record)
        print(label,job,flush=True);return job
    verify=jobs['q3a_ratio_verify']['jobid'];fits=[jobs['q3a_L32x64_fit']['jobid']]
    for key in ('L32x96','L24x72'):fits.append(submit('q3a_'+key+'_fit','fit',key,verify))
    numeric=submit('q3a_numeric','analyze','all_three',':'.join(fits))
    plot=submit('q3a_plot','plot','all_three',numeric,cpus=4,mem='16G')
    submit('q3a_final','final','all_three',plot,cpus=1,mem='8G')
    print('PACKED_CPU_TAIL_SUBMITTED',flush=True)


if __name__=='__main__':main()
