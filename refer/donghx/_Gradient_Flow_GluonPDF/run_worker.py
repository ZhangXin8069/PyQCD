"""For four flows: resume missing moving C3/C2, then reuse basis for new P0."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from plan_common import *


def ready_basis(row,moving,p0):
    import numpy as np
    from generate_flowed_vertex_l32x64 import read_flowed_gauge_index
    cfg=p0.config_for(row)
    marker=json.loads((MOVING/'status'/f'task{row["upstream_task"]}.json').read_text())
    assert marker['status']=='PASS' and marker['case_count']==252
    assert marker['manifest_sha256']==moving.manifest_hash()
    identities=[]
    for key,expected in row['basis_metadata'].items():
        sigma=int(key);done=cfg.basis_done_path(sigma);meta=json.loads(done.read_text())
        assert meta['schema']==expected['schema'] and meta['manifest_sha256']==expected['manifest_sha256']
        assert meta['conf']==row['conf'] and meta['original_time']==sigma
        assert meta['flow_time']==row['flow_time_a2'] and meta['epsilon']==.01 and meta['n_steps']==row['nstep']
        targets=[tj for tj in CURRENTS if sigma in moving.support_times(tj,row['window'])]
        assert meta['stored_t_current']==targets and meta['temporal_window']==row['window']
        assert meta['window_diagnostics']['max_per_mode_squared_norm_fraction']<=moving.load_manifest()['per_column_squared_norm_gate']
        identities.append(p0.fingerprint(done))
        for tj in targets:
            path=cfg.basis_path(sigma,tj);record=p0.fingerprint(path)
            assert record['shape']==[400,32,32,32,3] and record['dtype']=='complex64'
            identities.append(record)
    read_flowed_gauge_index(p0.gauge_config(row))
    identities.append(p0.fingerprint(cfg.gauge_flow_root/'index.json'))
    atomic_json(HERE/'status'/f'basis_ready_task{row["task"]}.json',dict(status='PASS',
        conf=row['conf'],tau=row['flow_time_a2'],task=row['task'],identities=identities,
        moving_manifest_sha256=moving.manifest_hash(),P0_manifest_sha256=sha(P0/'manifest.json'),
        moving_completed_marker_sha256=sha(MOVING/'status'/f'task{row["upstream_task"]}.json'),
        verification='basis completion identities, NPY header/stat and gauge index; basis payload already guarded by moving producer',
        job=os.environ['SLURM_JOB_ID']))


def main():
    allocation();request=check_frozen()
    parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True,type=int);args=parser.parse_args()
    node=os.environ.get('SLURMD_NODENAME')
    assert node in ALLOWED_NODES,('node not explicitly allowed',node)
    gate=json.loads((HERE/'status/gate_complete.json').read_text());assert gate['status']=='PASS'
    moving=load_module('moving_B100_run',MOVING);p0=load_module('P0_B100_run',P0)
    m=moving.load_manifest();pm=p0.load_manifest();worker=pm['workers'][args.worker]
    assert worker['worker']==args.worker and len(worker['tasks'])==4
    assert len({pm['tasks'][t]['conf'] for t in worker['tasks']})==1
    state=dict(status='RUNNING',worker=args.worker,conf=worker['conf'],tasks=worker['tasks'],
        job=os.environ['SLURM_JOB_ID'],node=node,timestamp=datetime.now().astimezone().isoformat())
    path=HERE/'status'/f'worker{args.worker}.json';atomic_json(path,state)
    try:
        for task in worker['tasks']:
            row=pm['tasks'][task]
            state.update(current_task=task,stage='moving_missing_only');atomic_json(path,state)
            if task in gate['completed_moving_tasks']:
                marker=json.loads((MOVING/'status'/f'task{task}.json').read_text())
                assert marker['status']=='PASS' and marker['manifest_sha256']==moving.manifest_hash()
                print('VERIFIED_COMPLETED_MOVING_SKIP',task,row['conf'],row['flow_time_a2'],flush=True)
            else:
                subprocess.run([sys.executable,'-B','-u',str(MOVING/'launch.py'),'--task',str(task)],
                               cwd=str(MOVING),check=True)
            state['stage']='basis_dependency_guard';atomic_json(path,state)
            ready_basis(row,moving,p0)
        # All four matching moving flows are committed before P0 consumes them.
        state['stage']='P0_same_basis_four_flows';atomic_json(path,state)
        subprocess.run([sys.executable,'-B','-u',str(P0/'launch.py'),'--worker',str(args.worker)],
                       cwd=str(P0),check=True)
        state.update(status='PASS',stage='moving_and_P0_four_flows_complete',timestamp=datetime.now().astimezone().isoformat())
        atomic_json(path,state);print('CONF51_100_P0456_WORKER_PASS',args.worker,worker['conf'],flush=True)
    except BaseException as error:
        state.update(status='FAILED_OR_INTERRUPTED',error=str(error),timestamp=datetime.now().astimezone().isoformat())
        atomic_json(path,state);raise


if __name__=='__main__':main()
