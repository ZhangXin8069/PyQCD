#!/usr/bin/env python3
"""Four full-domain flow workers followed by four current-slice C3 workers."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    if args.task != int(os.environ['P4_TASK']) or 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('allocation and consistent task ID required')
    manifest = load_manifest()
    row = manifest['tasks'][args.task]
    gate = json.loads((HERE / 'status' / 'P4_cpu_gate.json').read_text())
    if gate['status'] != 'PASS' or gate['manifest_sha256'] != sha256(HERE / 'manifest.json'):
        raise RuntimeError('matching CPU gate not PASS')
    completed = HERE / 'status' / f'task{args.task}_comparison.json'
    if completed.exists():
        marker=json.loads(completed.read_text())
        if marker['status']!='PASS' or marker['manifest_sha256']!=sha256(HERE/'manifest.json'):
            raise RuntimeError('incompatible existing completion')
        check_inputs(row)
        print('MISSING_ONLY_SKIP',args.task,flush=True)
        return
    devices = [s.strip() for s in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if s.strip()]
    if len(devices) != 4 or len(set(devices)) != 4:
        raise RuntimeError(f'four allocated GPUs required: {devices}')
    check_inputs(row)
    scratch = Path(tempfile.mkdtemp(prefix=f'q64_P4B_{os.environ["SLURM_JOB_ID"]}_{args.task}_', dir='/tmp'))
    logdir = HERE / 'logs' / f'task{args.task}_{os.environ["SLURM_JOB_ID"]}'
    logdir.mkdir(parents=True, exist_ok=True)
    active, handles = [], []
    receipt = dict(state='RUNNING', task=args.task, conf=row['conf'],
        flow_time_a2=row['flow_time_a2'], windows=row['windows'], stages=[],
        allocated_cuda_devices=devices, job=os.environ['SLURM_JOB_ID'],
        weight_scheme='B', manifest_sha256=sha256(HERE / 'manifest.json'))
    status = HERE / 'status' / f'pipeline_task{args.task}_{os.environ["SLURM_JOB_ID"]}.json'

    def stop(signum, frame):
        raise RuntimeError(f'received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        atomic_json(status, receipt)
        started = time.monotonic()
        workers = []
        for worker, device in ([] if row['basis_reuse'] else enumerate(devices)):
            env = os.environ.copy()
            cache = scratch / f'worker{worker}'
            (cache / 'quda').mkdir(parents=True)
            env.update(CUDA_VISIBLE_DEVICES=device, GRID_SIZE='1,1,1,1',
                QUDA_RESOURCE_PATH=str(cache / 'quda'), CUPY_CACHE_DIR=str(cache / 'cupy'),
                QUDA_PROFILE_OUTPUT_BASE=str(cache / 'quda_profile'))
            output = (logdir / f'basis_gpu{worker}.log').open('a')
            handles.append(output)
            child = subprocess.Popen(['mpirun', '--bind-to', 'none', '-n', '1',
                sys.executable, '-u', str(HERE / 'generate_selected_basis.py'),
                '--task', str(args.task), '--worker', str(worker)], env=env,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append(child)
            active.append(child)
        pending = set(range(len(workers)))
        while pending:
            for worker in list(pending):
                code = workers[worker].poll()
                if code is None:
                    continue
                pending.remove(worker)
                if code:
                    raise RuntimeError(f'basis worker {worker} failed: {code}; {logdir}')
                active.remove(workers[worker])
            if pending:
                time.sleep(2)
        receipt['stages'].append(dict(stage='selected_full4D_basis', seconds=time.monotonic()-started))
        receipt['current_stage'] = 'B_C3_C2_temporal_window_comparison'
        atomic_json(status, receipt)
        env = os.environ.copy()
        env.update(CUPY_CACHE_DIR=str(scratch / 'cupy_contraction'),
                   QUDA_RESOURCE_PATH=str(scratch / 'quda_contraction'))
        output = (logdir / 'compare_windows.log').open('a')
        handles.append(output)
        started = time.monotonic()
        child = subprocess.Popen(['mpirun', '--bind-to', 'none', '-n', '4',
            sys.executable, '-u', str(HERE / 'compare_windows.py'),
            '--task', str(args.task)], env=env, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True)
        active.append(child)
        code = child.wait()
        active.remove(child)
        if code:
            raise RuntimeError(f'B C3/window comparison failed: {code}; {logdir}')
        receipt['stages'].append(dict(stage='B_C3_C2_window_comparison', seconds=time.monotonic()-started))
        check_inputs(row)
        receipt['state'] = 'COMPLETED'
        atomic_json(status, receipt)
        print('P4_B_PIPELINE_COMPLETED', args.task, row['flow_time_a2'], flush=True)
    except BaseException as exc:
        receipt.update(state='FAILED_OR_INTERRUPTED', error=str(exc))
        atomic_json(status, receipt)
        raise
    finally:
        for child in active:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
        for output in handles:
            output.close()
        if (scratch.parent == Path('/tmp') and scratch.name.startswith('q64_P4B_')
                and not scratch.is_symlink()):
            try:
                shutil.rmtree(scratch)
            except OSError as exc:
                print(f'SCRATCH_CLEANUP_WARNING {scratch}: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
