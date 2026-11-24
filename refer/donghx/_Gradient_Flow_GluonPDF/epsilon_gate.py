"""Independent one-GPU zero/free/step-size checks before full basis production."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from common import HERE,load_manifest,sha256,atomic_json,require_allocation


def main():
    require_allocation()
    manifest=load_manifest();digest=sha256(HERE/'manifest.json')
    for name in ('cpu_gate','tests','retry_tests'):
        result=json.loads((HERE/'status'/f'{name}.json').read_text())
        assert result['status']=='PASS'
        if 'manifest_sha256' in result:
            assert result['manifest_sha256']==digest
    scratch=Path(tempfile.mkdtemp(prefix='q24_B_eps_',dir='/tmp'))
    results=[]
    try:
        env=os.environ.copy()
        env.update(QUDA_RESOURCE_PATH=str(scratch/'quda_validation'),
                   CUPY_CACHE_DIR=str(scratch/'cupy'),GRID_SIZE='1,1,1,1')
        (scratch/'quda_validation').mkdir()
        subprocess.run([sys.executable,'-B','-u',str(HERE/'gpu_validation.py')],env=env,check=True)
        for task in (0,6):
            quda=scratch/f'quda_{task}';quda.mkdir()
            env.update(QUDA_RESOURCE_PATH=str(quda))
            subprocess.run([sys.executable,'-B','-u',str(HERE/'generate_basis.py'),
                            '--task',str(task),'--worker','0','--epsilon-only'],env=env,check=True)
            result=json.loads((HERE/'status'/f'epsilon_task{task}.json').read_text())
            row=manifest['tasks'][task]
            assert result['status']=='PASS' and result['manifest_sha256']==digest
            assert result['flow_time']==row['flow_time_a2'] and result['nstep']==row['nstep']
            results.append(result)
    finally:
        if scratch.parent==Path('/tmp') and scratch.name.startswith('q24_B_eps_') and not scratch.is_symlink():
            shutil.rmtree(scratch)
    atomic_json(HERE/'status/epsilon_gate.json',dict(status='PASS',tasks=[0,6],results=results,
        manifest_sha256=digest,job=os.environ['SLURM_JOB_ID'],
        scope='L24 APBC free heat, analytic zero-flow VDV/contractions, full4D first/last-flow step-halving'))
    print('L24_EPSILON_GATE_PASS',flush=True)


if __name__=='__main__':
    main()
