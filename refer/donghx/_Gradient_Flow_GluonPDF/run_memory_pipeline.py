#!/usr/bin/env python3
"""Reuse validated full-4D basis; rerun only failed B C3/C2 contractions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from common import HERE, load_manifest, check_inputs, atomic_json, sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);a=p.parse_args()
    m=load_manifest()
    if ('SLURM_JOB_ID' not in os.environ or a.task != int(os.environ['P4_TASK'])
            or a.task not in m['selected_tasks_to_compute']):
        raise RuntimeError('allocated missing-only task required')
    gate=json.loads((HERE/'status/P4_cpu_gate.json').read_text())
    if gate['status']!='PASS' or gate['manifest_sha256']!=sha256(HERE/'manifest.json'):
        raise RuntimeError('matching CPU gate not PASS')
    if (HERE/'status'/f'task{a.task}_comparison.json').exists():
        raise RuntimeError('already completed: audit before resubmission')
    check_inputs(m['tasks'][a.task])
    devices=[v for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v]
    if len(devices)!=4 or len(set(devices))!=4:
        raise RuntimeError('one MPI rank per allocated GPU required')
    scratch=Path(tempfile.mkdtemp(prefix=f'q64mem_{os.environ["SLURM_JOB_ID"]}_',dir='/tmp'))
    logs=HERE/'logs'/f'task{a.task}_{os.environ["SLURM_JOB_ID"]}'
    logs.mkdir(parents=True)
    status=HERE/'status'/f'pipeline_task{a.task}_{os.environ["SLURM_JOB_ID"]}.json'
    row=dict(state='RUNNING',task=a.task,job=os.environ['SLURM_JOB_ID'],
        flow_time_a2=m['tasks'][a.task]['flow_time_a2'],
        manifest_sha256=sha256(HERE/'manifest.json'),basis_policy='read_only_upstream_no_reflow',
        current_stage='memory_bounded_B_C3_C2_comparison',allocated_cuda_devices=devices)
    child=None;start=time.monotonic()
    def stop(signum,frame):
        raise RuntimeError(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        atomic_json(status,row)
        env=os.environ.copy();env.update(CUPY_CACHE_DIR=str(scratch/'cupy'),QUDA_RESOURCE_PATH=str(scratch/'quda'))
        with (logs/'compare_windows.log').open('a') as log:
            child=subprocess.Popen(['mpirun','--bind-to','none','-n','4',sys.executable,'-u',
                str(HERE/'compare_windows.py'),'--task',str(a.task)],env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=child.wait()
            if code:
                raise RuntimeError(f'comparison failed: {code}; see {logs}')
        check_inputs(m['tasks'][a.task])
        row['state']='COMPLETED'
    except BaseException as exc:
        row.update(state='FAILED',error=str(exc))
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        row['seconds']=time.monotonic()-start
        atomic_json(status,row)
        # Only our freshly created, exact /tmp directory; no basis/vertex deletion.
        shutil.rmtree(scratch)


if __name__=='__main__':main()
