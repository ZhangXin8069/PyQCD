"""24 immutable worker identities cover all350 tasks without 72h bundling."""
import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
from common import (HERE,NWORKERS,load_manifest,require_allocation,sha256,atomic_json)


def main():
    require_allocation()
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,required=True)
    worker=parser.parse_args().worker
    manifest=load_manifest();digest=sha256(HERE/'manifest.json')
    if not 0<=worker<NWORKERS:
        raise ValueError(worker)
    tasks=list(range(worker,len(manifest['tasks']),NWORKERS))
    pending=[];completed=[]
    for task in tasks:
        marker=HERE/'status'/f'task{task}.json'
        if not marker.exists():
            pending.append(task)
            continue
        data=json.loads(marker.read_text())
        assert data['status']=='PASS' and data['task']==task and data['case_count']==252
        assert data['manifest_sha256']==digest
        completed.append(task)
    if pending:
        task=pending[0]
        print('PACKED_WORKER_START',worker,task,'remaining',len(pending),flush=True)
        # This child verifies every published case and the source/gauge/basis
        # identities; a scientific failure exits, rather than auto-requeueing.
        subprocess.run([sys.executable,'-B','-u',str(HERE/'launch.py'),'--task',str(task)],check=True)
        data=json.loads((HERE/'status'/f'task{task}.json').read_text())
        assert data['status']=='PASS' and data['manifest_sha256']==digest and data['case_count']==252
        completed.append(task);pending=pending[1:]
    atomic_json(HERE/'status'/f'packed_worker{worker:02d}.json',dict(
        status='REQUEUE_NEEDED' if pending else 'TASK_MARKERS_COMPLETE_FINAL_AUDIT_REQUIRED',
        worker=worker,tasks=tasks,completed=completed,pending=pending,
        manifest_sha256=digest,slurm_job=os.environ['SLURM_JOB_ID'],
        restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0')),
        updated=datetime.now().astimezone().isoformat()))
    if pending:
        job=f'{os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}'
        print('REQUEUE_SAME_ARRAY_ELEMENT',job,'next_task',pending[0],flush=True)
        subprocess.run(['scontrol','requeue',job],check=True)
    else:
        print('PACKED_WORKER_FINISHED',worker,len(completed),'final audit required',flush=True)


if __name__=='__main__':
    main()
