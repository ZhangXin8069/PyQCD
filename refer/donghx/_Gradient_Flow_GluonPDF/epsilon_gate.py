"""Run only the short first/last-flow GPU integration checks, before basis jobs."""
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
    manifest=load_manifest()
    digest=sha256(HERE/'manifest.json')
    for name in ('cpu_gate','retry_tests','basis_reuse'):
        result=json.loads((HERE/'status'/f'{name}.json').read_text())
        assert result['status']=='PASS'
        if 'manifest_sha256' in result:
            assert result['manifest_sha256']==digest
    scratch=Path(tempfile.mkdtemp(prefix='q96_B_eps_v2_',dir='/tmp'))
    results=[]
    try:
        for task in (0,6):
            quda=scratch/f'quda_{task}'
            quda.mkdir()
            env=os.environ.copy()
            env.update(QUDA_RESOURCE_PATH=str(quda),CUPY_CACHE_DIR=str(scratch/'cupy'),
                       GRID_SIZE='1,1,1,1')
            subprocess.run([sys.executable,'-B','-u',str(HERE/'generate_basis.py'),
                            '--task',str(task),'--worker','0','--epsilon-only'],env=env,check=True)
            result=json.loads((HERE/'status'/f'epsilon_task{task}.json').read_text())
            row=manifest['tasks'][task]
            assert result['status']=='PASS' and result['manifest_sha256']==digest
            assert result['flow_time']==row['flow_time_a2'] and result['nstep']==row['nstep']
            assert result['coarse_meas_interval']>row['nstep']
            assert result['fine_meas_interval']>2*row['nstep']
            results.append(result)
    finally:
        if scratch.parent==Path('/tmp') and scratch.name.startswith('q96_B_eps_v2_') and not scratch.is_symlink():
            shutil.rmtree(scratch)
    atomic_json(HERE/'status/epsilon_gate.json',dict(status='PASS',tasks=[0,6],
        manifest_sha256=digest,job=os.environ['SLURM_JOB_ID'],results=results,
        scope='actual full4D first/last-flow step-halving; not a C3/C2 completion gate'))
    print('EPSILON_GATE_PASS',flush=True)


if __name__=='__main__':
    main()

