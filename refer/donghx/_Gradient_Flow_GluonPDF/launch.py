"""Per-job scratch and GPU process binding. Keep basis, never cache vertex/VVV."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from common import HERE,load_manifest,load_request,require_allocation,sha256,atomic_json


def run_basis(task,env,smoke,scratch):
    visible=env.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    if len(visible)!=4:
        raise RuntimeError(('expected exactly four allocated CUDA devices',visible))
    children=[]
    streams=[]
    try:
        for worker in range(4):
            local=env.copy()
            local.update(CUDA_VISIBLE_DEVICES=visible[worker],GRID_SIZE='1,1,1,1',
                         QUDA_RESOURCE_PATH=str(scratch/f'quda_{worker}'))
            (scratch/f'quda_{worker}').mkdir(parents=True,exist_ok=True)
            output=(HERE/'logs'/f'basis_task{task}_worker{worker}_{os.environ["SLURM_JOB_ID"]}.out').open('a')
            streams.append(output)
            command=[sys.executable,'-B','-u',str(HERE/'generate_basis.py'),
                     '--task',str(task),'--worker',str(worker)]
            if smoke:
                command.append('--smoke')
            children.append(subprocess.Popen(command,env=local,stdout=output,stderr=subprocess.STDOUT))
        while True:
            codes=[p.poll() for p in children]
            if any(c not in (None,0) for c in codes):
                raise RuntimeError(('basis worker failed; all sibling workers terminated',task,codes))
            if all(c==0 for c in codes):
                break
            time.sleep(1)
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=15)
        for stream in streams:
            stream.close()


def main():
    require_allocation()
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=int)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    manifest=load_manifest()
    gate=json.loads((HERE/'status/cpu_gate.json').read_text())
    assert gate['status']=='PASS' and gate['manifest_sha256']==sha256(HERE/'manifest.json')
    epsilon_gate=json.loads((HERE/'status/epsilon_gate.json').read_text())
    assert epsilon_gate['status']=='PASS'
    assert epsilon_gate['manifest_sha256']==sha256(HERE/'manifest.json')
    assert epsilon_gate['tasks']==[0,6]
    tasks=(0,6) if args.smoke else (args.task,)
    if any(t is None or t<0 or t>=len(manifest['tasks']) for t in tasks):
        raise ValueError(tasks)
    if not args.smoke:
        smoke_gate=json.loads((HERE/'status/smoke_gate.json').read_text())
        assert smoke_gate['status']=='PASS' and smoke_gate['manifest_sha256']==sha256(HERE/'manifest.json')
    scratch=Path(tempfile.mkdtemp(prefix='q24_B_radius_',dir='/tmp'))
    env=os.environ.copy()
    env.update(CUPY_CACHE_DIR=str(scratch/'cupy'),QUDA_RESOURCE_PATH=str(scratch/'quda'),
               MPLCONFIGDIR=str(scratch/'mpl'),GRID_SIZE='1,1,1,1',FLOW_MODE_BATCH='16')
    (scratch/'quda').mkdir()
    try:
        for task in tasks:
            marker=HERE/'status'/f'task{task}.json'
            if marker.exists() and not args.smoke:
                # Do not call an already-complete task missing without verifying it.
                subprocess.run([sys.executable,'-B','-u',str(HERE/'audit_cpu.py'),
                                '--task',str(task)],env=env,check=True)
                continue
            run_basis(task,env,args.smoke,scratch)
            command=['mpirun','--bind-to','none','-n','4',sys.executable,'-B','-u',
                     str(HERE/'gpu_worker.py'),'--task',str(task)]
            if args.smoke:
                command.append('--smoke')
            subprocess.run(command,env=env,check=True)
        if args.smoke:
            for task in tasks:
                result=json.loads((HERE/'status'/f'smoke_task{task}.json').read_text())
                assert result['status']=='PASS' and result['manifest_sha256']==sha256(HERE/'manifest.json')
            atomic_json(HERE/'status/smoke_gate.json',dict(status='PASS',tasks=list(tasks),
                manifest_sha256=sha256(HERE/'manifest.json'),job=os.environ['SLURM_JOB_ID'],
                scope='first/last flow real basis, C2 full-spin and H/six-Wick ordering/projection gates; finite window remains an approximation'))
    finally:
        # Only this task's explicitly validated mkdtemp cache directory.
        # Shared basis/gauge/C3 and existing user files are never removed here.
        if scratch.parent==Path('/tmp') and scratch.name.startswith('q24_B_radius_') and not scratch.is_symlink():
            try:
                shutil.rmtree(scratch)
            except OSError as error:
                print('SCRATCH_CLEANUP_WARNING',error,file=sys.stderr)


if __name__=='__main__':
    main()
