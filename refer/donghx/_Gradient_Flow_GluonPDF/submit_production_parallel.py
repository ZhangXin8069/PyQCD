#!/usr/bin/env python3
"""User-authorized parallel production; explicit audited confs, no afterok."""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--regression-job',default='3729470')
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--confs',nargs='+',required=True)
    a=p.parse_args()
    audit={r['conf']:r for r in json.loads(a.audit.read_text())}
    ledger=HERE/'status'/'submitted_production.json'
    rows=json.loads(ledger.read_text())
    for conf in a.confs:
        if audit.get(conf,{}).get('status')!='PASS':
            raise ValueError(f'{conf}: missing validated inputs')
        if any(r['conf']==conf and not r.get('superseded_by') for r in rows):
            raise ValueError(f'{conf}: already submitted; consult ledger before replacing')
    for conf in a.confs:
        script=HERE/'run_production_parallel_na800.slurm'
        cmd=['sbatch','--parsable','--partition=na800-sug','--reservation=donghx_223',
             f'--export=ALL,CONF={conf},REGRESSION_JOB={a.regression_job}',str(script)]
        done=subprocess.run(cmd,check=True,text=True,capture_output=True)
        job=done.stdout.strip().split(';')[0];assert job.isdigit(),done.stdout
        rows.append(dict(conf=conf,job=job,dependency=None,parallel_regression_job=a.regression_job,
            partition='na800-sug',reservation='donghx_223',submitted=datetime.now().isoformat(),
            flow_time=.5,tseps=[5,6,7,8,9,10],profile='pilot_z',script=str(script),
            regression_policy='user-authorized parallel run, validation pending'))
        ledger.write_text(json.dumps(rows,indent=2)+'\n')
        print(conf,job,flush=True)

if __name__=='__main__':main()
