#!/usr/bin/env python3
"""Submit only explicitly named, audited inputs, gated on the regression job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--regression-job',required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--confs',nargs='+',required=True)
    args=p.parse_args();audit={r['conf']:r for r in json.loads(args.audit.read_text())}
    existing=HERE/'status'/'submitted_production.json'
    records=json.loads(existing.read_text()) if existing.exists() else []
    for conf in args.confs:
        if conf not in audit or audit[conf]['status']!='PASS':
            print('NOT_SUBMITTED missing/invalid inputs:',conf,flush=True);continue
        if any(r['conf']==conf and not r.get('superseded_by') for r in records):
            print('NOT_SUBMITTED already in submission ledger:',conf,flush=True);continue
        result=subprocess.run(['sbatch','--parsable','--partition=na800-sug',
            '--reservation=donghx_223','--kill-on-invalid-dep=yes',
            '--dependency=afterok:'+args.regression_job,
            '--export=ALL,CONF='+conf+',REGRESSION_JOB='+args.regression_job,
            str(HERE/'run_production_na800.slurm')],check=True,text=True,capture_output=True)
        job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
        records.append(dict(conf=conf,job=job,dependency='afterok:'+args.regression_job,
            partition='na800-sug',reservation='donghx_223',
            submitted=datetime.now().isoformat(),flow_time=.5,tseps=[5,6,7,8,9,10],
            profile='pilot_z',script=str(HERE/'run_production_na800.slurm')))
        existing.write_text(json.dumps(records,indent=2)+'\n');print(conf,job,flush=True)

if __name__=='__main__':main()
