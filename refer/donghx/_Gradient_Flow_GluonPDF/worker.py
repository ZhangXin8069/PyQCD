#!/usr/bin/env python3
"""19 allocations, each sequentially processes three flows for one configuration."""
import argparse
import json
import os
import subprocess
import sys
from common import HERE,load_manifest,sha256,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--smoke-task',type=int);a=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('GPU allocation required')
    m=load_manifest()
    tasks=[a.smoke_task] if a.smoke_task is not None else m['workers'][a.worker]['tasks']
    for task in tasks:
        env=os.environ.copy();env['P4_TASK']=str(task)
        subprocess.run([sys.executable,'-B','-u',str(HERE/'run_pipeline.py'),'--task',str(task)],env=env,check=True)
    atomic_json(HERE/'status'/f'worker_{a.worker if a.worker is not None else "smoke"}.json',
        dict(status='PASS',tasks=tasks,job=os.environ['SLURM_JOB_ID'],manifest_sha256=sha256(HERE/'manifest.json')))


if __name__=='__main__':main()
