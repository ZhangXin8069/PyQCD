"""Run ten independent readback shards using five CPUs in one Slurm job."""
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def main():
    assert 'SLURM_JOB_ID' in os.environ
    assert int(os.environ['SLURM_CPUS_PER_TASK'])>=5
    pending=list(range(10));active={}
    try:
        while pending or active:
            while pending and len(active)<5:
                worker=pending.pop(0)
                active[worker]=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'audit_saved.py'),'--worker',str(worker)],cwd=HERE)
            for worker,child in list(active.items()):
                result=child.poll()
                if result is not None:
                    if result:raise RuntimeError(('audit shard failed',worker,result))
                    del active[worker]
            if active:time.sleep(1)
        subprocess.run([sys.executable,'-B','-u',str(HERE/'audit_saved.py'),'--collect'],cwd=HERE,check=True)
    finally:
        for child in active.values():
            if child.poll() is None:child.terminate()
        for child in active.values():
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)

if __name__=='__main__':main()
