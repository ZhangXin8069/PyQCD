#!/usr/bin/env python3
"""Only per-allocation temporary compilation cache; no VVV/vertex disk cache."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from common import HERE


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--task',type=int)
    a=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('GPU allocation required')
    tasks=(0,7) if a.smoke else (a.task,)
    if any(t is None or not 0<=t<160 for t in tasks):
        raise ValueError(tasks)
    scratch=Path(tempfile.mkdtemp(prefix='q64_Binsert_',dir='/tmp'))
    env=os.environ.copy()
    env.update(CUPY_CACHE_DIR=str(scratch/'cupy'),QUDA_RESOURCE_PATH=str(scratch/'quda'),
               MPLCONFIGDIR=str(scratch/'mpl'))
    try:
        for task in tasks:
            command=['mpirun','--bind-to','none','-n','4',sys.executable,'-B','-u',
                     str(HERE/'gpu_worker.py'),'--task',str(task)]
            if a.smoke:command+=['--smoke']
            subprocess.run(command,env=env,check=True)
    finally:
        if scratch.parent==Path('/tmp') and scratch.name.startswith('q64_Binsert_') and not scratch.is_symlink():
            shutil.rmtree(scratch)


if __name__=='__main__':
    main()
