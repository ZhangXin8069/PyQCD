#!/usr/bin/env python3
"""Isolated parity-reduced successor of the reviewed production MPI driver.

SPIN2_VERIFY=1 compares every kernel invocation with the frozen original.
All original time/MPI/save code remains in reference/, not copied ad hoc.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
import calc_flowed_c2c3_factorized_l32x64 as driver
from spin2_contractions import c2_reduced, accumulate_triplet_reduced

original_c2=driver.c2_batch
original_c3=driver.accumulate_triplet
original_batches=driver.momentum_batches
original_save=driver.expand_and_save
original_cache=driver.vvv_cache_path
checks=[]
verify=os.environ.get('SPIN2_VERIFY','0')=='1'


def compare(label, reference, reduced):
    reference=driver.to_host(reference);reduced=driver.to_host(reduced)
    if not np.isfinite(reference).all() or not np.isfinite(reduced).all():
        raise FloatingPointError(label)
    diff=reference-reduced
    norm=max(float(np.linalg.norm(reference.ravel())),1e-300)
    row=dict(label=label,max_abs=float(np.max(abs(diff))),
             relative_L2=float(np.linalg.norm(diff.ravel())/norm))
    if row['relative_L2']>2e-10:
        raise AssertionError(row)
    checks.append(row)


def c2_batch(p,vs,vr,g,q):
    value=c2_reduced(p,vs,vr,g,q,driver.contract)
    if verify: compare('C2',original_c2(p,vs,vr,g,q),value)
    return value


def accumulate(*args):
    # Isolated zero increments also catch accidentally overwritten accumulators.
    if verify:
        inputs=list(args)
        ru=driver.xp.zeros_like(args[8]);rd=driver.xp.zeros_like(args[9])
        inputs[8]=ru;inputs[9]=rd
        original_c3(*inputs)
        su=driver.xp.zeros_like(ru);sd=driver.xp.zeros_like(rd)
        inputs[8]=su;inputs[9]=sd
        accumulate_triplet_reduced(*inputs)
        for local,idx in enumerate(args[10]):
            for pol in range(4):
                compare(f'C3_u_mom{idx}_pol{pol}',ru[:,pol,idx],su[:,pol,idx])
                compare(f'C3_d_mom{idx}_pol{pol}',rd[:,pol,idx],sd[:,pol,idx])
        args[8][...]+=su;args[9][...]+=sd
    else:
        accumulate_triplet_reduced(*args)


def batches(jobs):
    maximum=int(os.environ.get('SPIN2_MOM_BATCH','1'))
    if maximum<1:raise ValueError('SPIN2_MOM_BATCH must be positive')
    return [part[i:i+maximum] for part in original_batches(jobs)
            for i in range(0,len(part),maximum)]


def cache_path(config,jobs):
    return Path(os.environ['SPIN2_VVV_ROOT'])/config.conf/original_cache(config,jobs).name


def save(config,tseps,jobs,maps,diagnostics):
    # Preserve NPZ layout and existing physics metadata; add an explicit receipt.
    original_save(config,tseps,jobs,maps,diagnostics)
    target=Path(os.environ['SAVE_ROOT'])/config.tag/f'Nev_{config.nev_src}'/config.conf
    receipt=dict(schema='spin2_production_v1',conf=config.conf,
        operator='unchanged C gamma5 gamma4; external parity rank 2 only',
        current_spins='full 4x4',momentum_batch=int(os.environ.get('SPIN2_MOM_BATCH','1')),
        verification_enabled=verify,mpi_size=driver.size,
        c2_source_list=driver.env_index_list('C2_SOURCE_LIST',list(range(64))),
        current_list=driver.env_index_list('CURRENT_LIST',list(range(64))),
        tseps=tseps,job=os.environ.get('SLURM_JOB_ID'),
        files_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in list(HERE.glob('*.py'))+list((HERE/'reference').glob('*.py'))})
    driver.atomic_json(target/'spin2_provenance.json',receipt)


def main():
    if not os.environ.get('SPIN2_VVV_ROOT') or not os.environ.get('SAVE_ROOT'):
        raise RuntimeError('explicit SPIN2_VVV_ROOT and SAVE_ROOT required')
    driver.c2_batch=c2_batch;driver.accumulate_triplet=accumulate
    driver.momentum_batches=batches;driver.vvv_cache_path=cache_path
    driver.expand_and_save=save
    started=time.perf_counter()
    result=driver.main()
    gathered=driver.comm.gather(checks,root=0)
    if driver.rank==0:
        rows=[item for group in gathered for item in group]
        if verify and not rows:raise AssertionError('no regression comparisons ran')
        receipt=dict(status='PASS' if verify else 'COMPLETED',verify=verify,
                     checks=len(rows),records=rows,wall_seconds=time.perf_counter()-started,
                     mpi_size=driver.size,job=os.environ.get('SLURM_JOB_ID'))
        path=Path(os.environ['SPIN2_STATUS_ROOT'])/f"job_{os.environ.get('SLURM_JOB_ID','local')}.json"
        path.parent.mkdir(parents=True,exist_ok=True);driver.atomic_json(path,receipt)
        print('SPIN2_SUCCESS '+str(path),flush=True)
    return result


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        if driver.size>1:driver.comm.Abort(1)
        raise
