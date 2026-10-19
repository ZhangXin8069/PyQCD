#!/usr/bin/env python3
"""GPU diagnostic: common unweighted J -> two tables -> ordered blocks/six Wick terms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'reference'))
import numpy as np
from mpi4py import MPI
import cupy as cp
from flow_common_l32x64 import (active_source_times, atomic_json, baryon_vvv,
    create_omega, phase_3pt, seq_peram, spin_objects)
from generate_flowed_vertex_l32x64 import flowed_gauge_time
from flow_common_l32x64 import generalized_vertex
from spin2_contractions import Contractor, c2_reduced
from weights_and_terms import (weight_tables, SECTOR_NAMES, SPECS,
    iter_reduced_right, current_block_h, signed_block_term, summed_flavors)
from prepare import config_for, fingerprint

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size


def metric(a, b):
    a, b = np.asarray(a), np.asarray(b)
    delta = b-a
    na, nb, nd = (float(np.linalg.norm(v.ravel())) for v in (a,b,delta))
    return dict(norm_A=na,norm_B=nb,norm_difference=nd,
                max_abs=float(np.max(abs(delta))),relative_L2_to_A=nd/max(na,1e-300),
                symmetric_relative_L2=2*nd/max(na+nb,1e-300))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def input_receipt(paths, expected):
    rows = []
    for path in sorted(set(map(Path,paths))):
        info = fingerprint(path)
        frozen = expected[str(path)]
        if any(info[key] != frozen[key] for key in ('bytes','mtime_ns')):
            raise RuntimeError(f'input changed since manifest: {path}')
        info['sha256'] = sha256(path)
        rows.append(info)
    return rows


def host(value):
    return cp.asnumpy(value)


def device_metrics(a,b):
    delta = b-a
    # Per sign/dz complex Frobenius norms and max, no phase cancellation.
    values = cp.stack([cp.sqrt(cp.sum(cp.abs(v)**2,axis=(-2,-1)))
                       for v in (a,b,delta)] +
                      [cp.max(cp.abs(delta),axis=(-2,-1))],axis=-1)
    if not bool(cp.isfinite(values).all()):
        raise FloatingPointError('vertex norm is non-finite')
    return host(values)


def atomic_npz(path, **payload):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**payload)
    os.replace(tmp,path)


def compute_current(config, row, manifest, tj, scratch, output):
    start = time.monotonic()
    taus = active_source_times(config,tj)
    dt,tsep = manifest['dt'],manifest['tsep']
    tsrc,tsink = (tj-dt)%64,(tj-dt+tsep)%64
    boundary = -1 if tsrc+tsep>=64 else 1
    expected = {info['path']:info for info in row['inputs']}
    paths = [config.basis_path(t,tj) for t in taus]
    paths += [config.basis_done_path(t) for t in taus]
    paths += [config.peram_path(t) for t in (tsrc,tsink)]
    paths += [config.eigen_path(t) for t in (tsrc,tsink)]
    paths += [config.gauge_flow_root/'index.json']
    paths += [config.gauge_flow_root/s['file'] for s in row['gauge_index']['shards']]
    # File identities are checked before reading data and again on completion.
    receipts = input_receipt(paths,expected)
    _,second,product,labels = weight_tables()
    frozen_second = create_omega(400).real
    np.testing.assert_allclose(second,frozen_second,rtol=4e-15,atol=0)
    # Branch A uses the frozen table's actual floating-point values, not an
    # algebraically equal reconstruction with a different division order.
    second = frozen_second
    table_A,table_product = cp.asarray(second),cp.asarray(product)
    # Production promotes complex64 basis with complex128 gauge, applies one
    # table and saves complex64 J. Both branches follow this exact cast order.
    gauge = cp.asarray(flowed_gauge_time(config,tj))
    bases = [cp.asarray(np.load(config.basis_path(t,tj),mmap_mode='r')) for t in taus]
    shape = (len(taus),len(taus),2,21,400,400)
    ja,jb = cp.empty(shape,dtype=cp.complex64),cp.empty(shape,dtype=cp.complex64)
    metrics = np.empty((len(taus),len(taus),2,21,4),dtype=np.float64)
    sector_metrics = np.empty((len(taus),len(taus),len(SECTOR_NAMES),len(SECTOR_NAMES),4))
    mask_unchanged = labels[:,None] != labels[None,:]
    mask_unchanged |= (labels[:,None]==0) & (labels[None,:]==0)
    mask_unchanged = cp.asarray(mask_unchanged)
    for ia,ta in enumerate(taus):
        for ib,tb in enumerate(taus):
            raw = generalized_vertex(bases[ia],bases[ib],gauge,'z',21,xp=cp)
            a = (raw*table_A[None,None]).astype(cp.complex64)
            b = a if ta==tb else (raw*table_product[None,None]).astype(cp.complex64)
            if ta==tb and not bool(cp.array_equal(a,b)):
                raise AssertionError('same-time J mismatch')
            if not bool(cp.array_equal(a[...,mask_unchanged],b[...,mask_unchanged])):
                raise AssertionError('different-sector or exact/exact J mismatch')
            ja[ia,ib],jb[ia,ib] = a,b
            metrics[ia,ib] = device_metrics(a,b)
            for r in range(len(SECTOR_NAMES)):
                ir = np.where(labels==r)[0]
                for s in range(len(SECTOR_NAMES)):
                    js = np.where(labels==s)[0]
                    sector_metrics[ia,ib,r,s] = device_metrics(
                        a[...,ir[0]:ir[-1]+1,js[0]:js[-1]+1],
                        b[...,ir[0]:ir[-1]+1,js[0]:js[-1]+1]).sum(axis=(0,1))
            del raw,a,b
        print(f'VERTEX_BLOCK_ROW conf={config.conf} tJ={tj} sigma_a={ta} complete',flush=True)
    del bases,gauge,table_A,table_product,mask_unchanged
    cp.get_default_memory_pool().free_all_blocks()
    vertex_seconds = time.monotonic()-start
    contractor = Contractor('optimal')
    gamma7,_,projectors = spin_objects(cp)
    source_peram = np.load(config.peram_path(tsrc),mmap_mode='r')
    sink_peram = np.load(config.peram_path(tsink),mmap_mode='r')
    a = seq_peram(cp.asarray(np.ascontiguousarray(sink_peram[taus,:,:,:400,:200])),cp)
    b = cp.asarray(np.ascontiguousarray(source_peram[taus,:,:,:400,:200]))
    p = cp.asarray(source_peram[tsink,:,:,:200,:200])
    momenta = manifest['momentum_labels']
    phases = cp.stack([phase_3pt([k,0,0],xp=cp) for k in (3,4,5)])
    vs = baryon_vvv(phases,np.load(config.eigen_path(tsink),mmap_mode='r')[:200],cp)
    vr = baryon_vvv(phases,np.load(config.eigen_path(tsrc),mmap_mode='r')[:200],cp).conj()
    del phases
    c2 = boundary*host(c2_reduced(p,vs,vr,gamma7,projectors,contractor))
    # host results are small enough to retain all ordered block/Wick labels.
    terms = np.empty((2,len(taus),len(taus),6,2,4,3,21,4,4),dtype=np.complex128)
    timings = []
    for ib,tb in enumerate(taus):
        begun = time.monotonic()
        ha = current_block_h(ja[:,ib],a,contractor)
        hb = current_block_h(jb[:,ib],a,contractor)
        for im in range(3):
            for it,(spec,right,q2,basis) in enumerate(iter_reduced_right(
                    p,b[ib:ib+1],gamma7,projectors,vs[im:im+1],vr[im:im+1],contractor,cp)):
                for scheme,h in enumerate((ha,hb)):
                    value = signed_block_term(h,spec,right,q2,basis,contractor)
                    terms[scheme,:,ib,it,:,:,im] = boundary*host(value[:,:, :,0])
                    del value
                print(f'WICK_BLOCKS conf={config.conf} tJ={tj} sigma_b={tb} '
                      f'mom={momenta[im]} term={spec[0]} rows={len(taus)}',flush=True)
        cp.cuda.Stream.null.synchronize()
        timings.append(dict(sigma_b=tb,seconds=time.monotonic()-begun))
        del ha,hb
        cp.get_default_memory_pool().free_all_blocks()
    if not np.isfinite(terms).all():
        raise FloatingPointError('non-finite Wick result')
    diagonal = np.arange(len(taus))
    if not np.array_equal(terms[0,diagonal,diagonal],terms[1,diagonal,diagonal]):
        raise AssertionError('same-time individual Wick terms not bitwise equal')
    c3 = summed_flavors(terms)
    production = {}
    checks = {}
    for flavor,iflav in [('u',0),('d',1)]:
        ref = np.load(scratch/f'corr_3pt_{flavor}.npy',mmap_mode='r')
        production[flavor] = np.asarray(ref[...,tsrc,tj]).copy()
        checks['production_'+flavor] = metric(production[flavor],c3[0,iflav])
        # Block decomposition changes contraction order only. A disagreement
        # above this strict normwise gate is an implementation failure.
        if checks['production_'+flavor]['relative_L2_to_A'] > 2e-9:
            raise AssertionError(checks['production_'+flavor])
    c2ref = np.load(scratch/'corr_2pt.npy',mmap_mode='r')[:,:,tsrc,tsink]
    checks['production_c2'] = metric(c2ref,c2)
    if checks['production_c2']['relative_L2_to_A'] > 2e-10:
        raise AssertionError(checks['production_c2'])
    checks['same_time_Wick_bitwise_equal'] = True
    checks['same_time_vertex_bitwise_equal'] = True
    checks['different_sector_and_exact_exact_vertex_bitwise_equal'] = True
    checks['all_6_Wick_terms_finite'] = True
    checks['weight_applied_exactly_once'] = 'raw geometric vertex -> table -> complex64, no other weights'
    wick_metrics = {spec[0]:metric(terms[0,:,:,it],terms[1,:,:,it]) for it,spec in enumerate(SPECS)}
    for info in receipts:
        now = Path(info['path']).stat()
        if now.st_size != info['bytes'] or now.st_mtime_ns != info['mtime_ns']:
            raise RuntimeError(f'input changed while computing: {info["path"]}')
    target = output / f'tJ{tj:03d}'
    target.mkdir(parents=True,exist_ok=True)
    atomic_npz(target/'comparison.npz',signed_wick_blocks=terms,c3_by_scheme_flavor=c3,
        c2_common=c2,c3_production_u=production['u'],c3_production_d=production['d'],
        vertex_metrics=metrics,sector_vertex_metrics=sector_metrics,
        metric_labels=np.asarray(['norm_A','norm_B','norm_B_minus_A','max_abs_B_minus_A']),
        sector_metric_reduction=np.asarray('sum over sign/dz of per-sign/dz norm or maximum'),
        sigma_times=np.asarray(taus),wick_names=np.asarray([s[0] for s in SPECS]),
        wick_coefficients=np.asarray([s[2] for s in SPECS]),sector_names=np.asarray(SECTOR_NAMES),
        scheme_labels=np.asarray(manifest['schemes']),flavor_labels=np.asarray(['u','d']),
        pol_labels=np.asarray(['nopol','pol15','pol25','pol35']),mom_labels=np.asarray(momenta),
        sign_labels=np.asarray(['minus_dz','plus_dz']),delta_z=np.arange(21),
        signed_wick_axes=np.asarray(['scheme','sigma_a','sigma_b','wick','sign','pol','mom','dz','m','o']),
        c3_axes=np.asarray(['scheme','flavor','sign','pol','mom','dz','m','o']),
        t_source=tsrc,t_current=tj,t_sink=tsink,tsep=tsep,dt=dt,boundary=boundary,
        flow_time_a2=config.flow_time,epsilon=config.epsilon,nstep=config.n_steps,
        temporal_window=config.temporal_window)
    receipt = dict(status='PASS',conf=config.conf,t_current=tj,t_source=tsrc,t_sink=tsink,
        boundary=boundary,checks=checks,wick_metrics=wick_metrics,
        c3_metrics={f:metric(c3[0,i],c3[1,i]) for i,f in enumerate(('u','d'))},
        vertex_seconds=vertex_seconds,per_sigma_b_timing=timings,seconds=time.monotonic()-start,
        input_content_sha256=receipts,job=os.environ.get('SLURM_JOB_ID'),
        scope='tau=0.5; one source/sink case per selected current; full 400 modes / six Wick terms',
        interpretation='A/B differences alone do not determine stochastic bias; source/sink joint-noise moments remain unproved')
    atomic_json(target/'receipt.json',receipt)
    print('OMEGA_CURRENT_PASS',config.conf,tj,receipt['c3_metrics'],flush=True)
    del ja,jb,a,b,p,vs,vr
    cp.get_default_memory_pool().free_all_blocks()
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',type=int,required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE/'manifest.json').read_text())
    if size != 4:
        raise RuntimeError(f'exactly four MPI ranks required, got {size}')
    if not 0 <= args.task < len(manifest['configurations']):
        raise ValueError('invalid array task')
    for relative,digest in manifest['files_sha256'].items():
        if sha256(HERE/relative) != digest:
            raise RuntimeError(f'comparison source changed after manifest: {relative}')
    row = manifest['configurations'][args.task]
    config = config_for(row['conf'])
    localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK',rank))
    cp.cuda.Device(localrank).use()
    cp.get_default_memory_pool().set_limit(size=70*1024**3)
    tj = manifest['currents'][rank]
    scratch = Path(os.environ['OMEGA_SCRATCH'])
    if rank == 0:
        # One shared job-local copy, discarded by the Slurm exit trap.
        with zipfile.ZipFile(row['production_file']) as archive:
            for key in ('corr_2pt','corr_3pt_u','corr_3pt_d'):
                with archive.open(key+'.npy') as src, (scratch/(key+'.npy')).open('wb') as dst:
                    shutil.copyfileobj(src,dst,8*1024*1024)
    comm.Barrier()
    output = HERE/'results'/config.conf
    receipt = compute_current(config,row,manifest,tj,scratch,output)
    all_receipts = comm.gather(receipt,root=0)
    if rank==0:
        atomic_json(HERE/'status'/f'{config.conf}.json',dict(status='PASS',conf=config.conf,
            currents=[r['t_current'] for r in all_receipts],job=os.environ.get('SLURM_JOB_ID'),
            manifest_sha256=sha256(HERE/'manifest.json')))
        print('OMEGA_CONF_PASS',config.conf,flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        comm.Abort(1)
        raise
