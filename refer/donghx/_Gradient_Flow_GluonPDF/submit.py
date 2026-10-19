#!/usr/bin/env python3
"""Freeze source and submit CPU-only gate -> bootstrap -> fits -> verification/plots."""
import argparse
import ast
from datetime import datetime
import subprocess
from common import HERE,sha256,atomic_json,code_check


def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    if not a.submit:
        print('DRY_RUN A scheme L32x64 Nconf30 Nboot100,8flow*3channel*3flavor; Tmin7,Q>=1e-3,chi2/dof<=1e3; CPU only')
        return
    if (HERE/'submission.json').exists():raise RuntimeError('already submitted; inspect receipts before retry')
    sources=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.slurm'))+sorted((HERE/'reference').rglob('*.py'))
    for p in sources:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        if p.suffix=='.slurm':subprocess.run(['bash','-n',str(p)],check=True)
    lock=dict(timestamp=datetime.now().astimezone().isoformat(),files_sha256={str(p.relative_to(HERE)):sha256(p) for p in sources})
    if (HERE/'code_lock.json').exists():code_check()
    else:atomic_json(HERE/'code_lock.json',lock)
    receipt=dict(state='SUBMITTING',timestamp=datetime.now().astimezone().isoformat(),
        code_lock_sha256=sha256(HERE/'code_lock.json'),jobs={},nconf=30,nboot=100,fit_Tmin=7,
        q_min=1e-3,chi2_dof_max=1e3,array_concurrency_cap=None)
    atomic_json(HERE/'submission.json',receipt)
    def submit(key,script,args=()):
        cmd=['sbatch','--parsable',f'--output={HERE}/logs/{key}_%A_%a.out',
             f'--error={HERE}/logs/{key}_%A_%a.err',*args,str(HERE/script)]
        r=subprocess.run(cmd,check=True,capture_output=True,text=True);job=r.stdout.strip().split(';')[0]
        if not job.isdigit():raise ValueError(r.stdout)
        receipt['jobs'][key]=dict(job_id=job,command=cmd);atomic_json(HERE/'submission.json',receipt)
        print('SUBMITTED',key,job,flush=True);return job
    try:
        gate=submit('gate','gate_cpu.slurm')
        ratios=submit('ratios','ratios_cpu.slurm',['--array=0-7',f'--dependency=afterok:{gate}'])
        fit=submit('fit','fit_cpu.slurm',['--array=0-71',f'--dependency=afterok:{ratios}'])
        submit('summary','summary_cpu.slurm',[f'--dependency=afterok:{fit}'])
    except BaseException as exc:
        receipt.update(state='PARTIAL_SUBMISSION_REQUIRES_AUDIT',error=str(exc));atomic_json(HERE/'submission.json',receipt);raise
    receipt['state']='SUBMITTED';atomic_json(HERE/'submission.json',receipt)


if __name__=='__main__':main()
