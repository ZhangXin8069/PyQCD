#!/usr/bin/env python3
"""Submit the reviewed ten-configuration expansion once, without a gate."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime

HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'status'/'submitted_new10.json'
    if out.exists():raise FileExistsError(f'already submitted: {out}')
    inventory=json.loads((HERE/'status'/'inventory20.json').read_text())
    confs=[r['conf'] for r in inventory['new_configurations']]
    assert confs==['11000','11100','12300','12400','12500','12600','12700','12800','12900','13000']
    assert len(set(inventory['target_confs']))==20
    assert all(r['status']=='PASS' for r in inventory['new_configurations'])
    assert all(f['exists'] for r in inventory['existing_configurations'] for f in r['files'])
    script=HERE/'run_new10.slurm'
    subprocess.run(['bash','-n',str(script)],check=True)
    done=subprocess.run(['sbatch','--parsable',str(script)],check=True,text=True,capture_output=True)
    job=done.stdout.strip().split(';')[0];assert job.isdigit(),done.stdout
    receipt=dict(array_job=job,submitted=datetime.now().isoformat(),partition='na800-sug',
        reservation='donghx_223',array_concurrency=None,gpus_per_conf=4,dependency=None,
        flow_time=.5,tseps=[5,6,7,8,9,10],profile='pilot_z',regression_job='3729470',
        regression_policy='user-authorized concurrent production; no regression dependency',
        tasks=[dict(array_index=i,conf=c,job=f'{job}_{i}') for i,c in enumerate(confs)],
        target_total_confs=inventory['target_confs'],
        flow_work_root=str(HERE/'flow_cache'),save_root=str(HERE/'results_tf0p5_pilot_z'),
        sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in list(HERE.rglob('*.py'))+[script]})
    out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['tasks'],indent=2),flush=True)

if __name__=='__main__':main()
