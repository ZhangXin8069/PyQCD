#!/usr/bin/env python3
"""Full C3 B successor: reuse original basis/gauge, regenerate only vertices.

Not automatically submitted by the window test. All disposable vertex/VVV
objects live in this job's private /tmp; original caches are never rewritten.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from common import HERE, PRODUCT, load_manifest, atomic_json
from flow_common_l32x64 import FlowConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('conf')
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('GPU allocation required')
    load_manifest()
    scratch = Path(tempfile.mkdtemp(prefix=f'q64_Bprod_{os.environ["SLURM_JOB_ID"]}_', dir='/tmp'))
    env = os.environ.copy()
    env.update(FLOW_WORK_ROOT=str(scratch / 'flow_cache'),
        SAVE_ROOT=str(HERE / 'C3_schemeB'), SPIN2_VVV_ROOT=str(scratch / 'vvv'),
        SPIN2_STATUS_ROOT=str(HERE / 'status'), RESULT_TMP_ROOT=str(scratch / 'result_tmp'),
        INPUT_STAGE_ROOT=str(scratch / 'input_stage'), FLOWED_PERAM_TMP_ROOT=str(scratch / 'flowed_peram'),
        CUPY_CACHE_DIR=str(scratch / 'cupy'), BUILD_VVV_IF_MISSING='1',
        STAGE_VVV_LOCAL='0', STAGE_PERAM_LOCAL='0', FLOWED_PERAM_STORAGE='device',
        KEEP_RESULT_TMP='0', KEEP_FLOWED_PERAM_TMP='0', VALIDATE_OUTPUT='1',
        FLOW_REQUIRE_MAX_DISCARDED_NORM='1e-6', PERAM_H2D_BATCHED='1',
        SPIN2_MOM_BATCH='1', BACKEND='cupy', VERTEX_BACKEND='cupy')
    os.environ.update(env)
    cfg = FlowConfig.from_env(args.conf)
    oldroot = Path(env.get('B_INPUT_CACHE_ROOT', str(PRODUCT / 'flow_cache'))) / args.conf / cfg.tag
    if oldroot.resolve() == cfg.root.resolve():
        raise RuntimeError('input cache must differ from private B work cache')
    cfg.root.mkdir(parents=True)
    for name in ('basis', 'flowed_gauge'):
        original = oldroot / name
        if not original.is_dir():
            raise FileNotFoundError(original)
        (cfg.root / name).symlink_to(original.resolve(), target_is_directory=True)
    currents = env.get('CURRENT_LIST', '').strip() or '0-63'
    directions = env.get('DIRS', 'z').replace(',', ' ')
    commands = [
        ['mpirun', '--bind-to', 'none', '-n', '4', sys.executable, '-u',
         str(HERE / 'reference' / 'generate_flowed_vertex_l32x64.py'), args.conf,
         '--stage', 'vertex', '--currents', currents, '--directions', *directions.split()],
        ['mpirun', '--bind-to', 'none', '-n', '4', sys.executable, '-u',
         str(HERE / 'run_spin2.py'), args.conf]]
    receipt = dict(conf=args.conf, weight_scheme='B', original_input_cache=str(oldroot),
                   state='RUNNING', job=os.environ['SLURM_JOB_ID'], commands=commands)
    status = HERE / 'status' / f'Bproduction_{args.conf}_{os.environ["SLURM_JOB_ID"]}.json'
    atomic_json(status, receipt)
    try:
        for command in commands:
            subprocess.run(command, env=env, check=True)
        receipt['state'] = 'COMPLETED'
    except BaseException as exc:
        receipt['state'] = 'FAILED'
        receipt['error'] = str(exc)
        raise
    finally:
        atomic_json(status, receipt)
        if scratch.parent == Path('/tmp') and scratch.name.startswith('q64_Bprod_') and not scratch.is_symlink():
            # rmtree unlinks the two input symlinks, not their targets.
            shutil.rmtree(scratch)


if __name__ == '__main__':
    main()
