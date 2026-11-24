#!/usr/bin/env python3
"""L24x72: four currents on four GPUs; Pz=0,+/-4..6; streamed B C3."""
import argparse
import json
import os
import time
import numpy as np

from common import (HERE, NT,NX,NV,NE,NZ,A_FM,BASIS_SCHEMA, CURRENTS, TSEPS, MOMENTA, SCHEME, SCHEMA, PROFILE, CHANNELS, POLS,
    config_for, gauge_config, support_times, load_manifest, check_inputs, case_times, output_dir,
    read_case, sha256, atomic_json, atomic_npz, project_spin, projection_specs, require_allocation)
from insertion_kernels import one_right_time_h
from mpi4py import MPI
import cupy as cp
from flow_common import (ENSEMBLE, generalized_vertex, seq_peram, baryon_vvv, phase_3pt, spin_objects,
    create_omega, first_order_weights, create_time_pair_omega)
from flow_backend import flowed_gauge_time
from spin2_contractions import Contractor, c2_reduced, c2_reference, c3_reduced, SPECS


def metric(reference, actual):
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise FloatingPointError('Nonfinite GPU regression')
    diff = actual - reference
    return dict(relative_L2=float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(reference.ravel()), 1e-300)),
                max_abs=float(np.max(np.abs(diff))))


def memory(stage):
    cp.cuda.get_current_stream().synchronize()
    free, total = cp.cuda.runtime.memGetInfo()
    print('GPU_MEMORY', json.dumps(dict(stage=stage, rank=MPI.COMM_WORLD.rank,
        pool_used=cp.get_default_memory_pool().used_bytes(), device_free=free, device_total=total)), flush=True)


def c2_oracle(row, tj, sep, dt, manifest):
    cfg = config_for(row)
    source, sink, _, boundary = case_times(tj, sep, dt)
    con = Contractor('optimal')
    g, _, q = spin_objects(cp)
    p = cp.asarray(np.load(cfg.peram_path(source), mmap_mode='r')[sink, :, :, :NE, :NE])
    values, errors = [], []
    for momentum in MOMENTA:
        phase = phase_3pt([momentum, 0, 0], xp=cp)[None]
        if momentum == 0:
            assert float(cp.max(cp.abs(phase - 1))) == 0.
        vs = baryon_vvv(phase, np.load(cfg.eigen_path(sink), mmap_mode='r')[:NE], cp)
        vr = baryon_vvv(phase, np.load(cfg.eigen_path(source), mmap_mode='r')[:NE], cp).conj()
        got = boundary * cp.asnumpy(c2_reduced(p, vs, vr, g, q, con))
        gold = boundary * cp.asnumpy(c2_reference(p, vs, vr, g, q, con))
        error = metric(gold, got)
        if error['relative_L2'] > manifest['algebra_relative_l2_gate']:
            raise AssertionError(('C2 oracle', momentum, source, sink, error))
        values.append(got)
        errors.append(dict(Pz=momentum,**error))
        del vs,vr,phase
        cp.get_default_memory_pool().free_all_blocks()
    return np.concatenate(values,axis=1),dict(momentum_checks=errors,
        max_relative_L2=max(e['relative_L2'] for e in errors))


def load_vertices(row, tj, times):
    cfg = config_for(row)
    bases = []
    for sigma in times:
        meta = json.loads(cfg.basis_done_path(sigma).read_text())
        assert meta['schema'] == manifest_basis_schema()
        assert meta['manifest_sha256'] == sha256(HERE/'manifest.json')
        assert tj in meta['stored_t_current'] and meta['flow_time'] == row['flow_time_a2']
        assert meta['epsilon']==row['epsilon'] and meta['n_steps']==row['nstep']
        basis=cp.asarray(np.load(cfg.basis_path(sigma, tj), mmap_mode='r'))
        assert basis.shape==(NV,NX,NX,NX,3) and basis.dtype==cp.complex64 and bool(cp.isfinite(basis).all())
        bases.append(basis)
    gauge = cp.asarray(flowed_gauge_time(gauge_config(row), tj))
    second = cp.asarray(create_omega(NV)); first = cp.asarray(first_order_weights(NV))
    product = first[:, None] * first[None, :]
    vertices = cp.empty((len(times), len(times), 2, NZ, NV, NV), cp.complex64)
    for ia, sa in enumerate(times):
        for ib, sb in enumerate(times):
            raw = generalized_vertex(bases[ia], bases[ib], gauge, 'z', NZ, xp=cp)
            table = create_time_pair_omega(sa, sb, NV, second=second, product=product)
            vertices[ia, ib] = (raw * table[None, None]).astype(cp.complex64)
            del raw
        print('B_PZ0_VERTEX_ROW', row['task'], tj, sa, flush=True)
    del bases, gauge, second, first, product, table
    cp.get_default_memory_pool().free_all_blocks()
    if not bool(cp.isfinite(vertices).all()):
        raise FloatingPointError('nonfinite B-weighted vertices')
    memory('weighted_J_once_no_disk_vertex')
    return vertices


def manifest_basis_schema():
    return BASIS_SCHEMA


def contract_case(row, tj, sep, dt, times, vertices, c2_record, manifest, oracle=False):
    started = time.monotonic()
    cfg = config_for(row)
    source, sink, _, boundary = case_times(tj, sep, dt)
    con = Contractor('optimal')
    g, _, q = spin_objects(cp)
    src = np.load(cfg.peram_path(source), mmap_mode='r')
    snk = np.load(cfg.peram_path(sink), mmap_mode='r')
    a = seq_peram(cp.asarray(np.ascontiguousarray(snk[times, :, :, :NV, :NE])), cp)
    b = cp.asarray(np.ascontiguousarray(src[times, :, :, :NV, :NE]))
    p = cp.asarray(src[sink, :, :, :NE, :NE])
    # In-memory only. Reuse each momentum's VVV over the sigma_b sum, not on disk.
    vvs,vvr=[],[]
    for momentum in MOMENTA:
        phase=phase_3pt([momentum,0,0],xp=cp)[None]
        vvs.append(baryon_vvv(phase,np.load(cfg.eigen_path(sink),mmap_mode='r')[:NE],cp))
        vvr.append(baryon_vvv(phase,np.load(cfg.eigen_path(source),mmap_mode='r')[:NE],cp).conj())
    wick = np.zeros((6, 2, 4, len(MOMENTA), NZ, 4, 4), np.complex128)
    real_checks = []
    for ib, sigma_b in enumerate(times):
        h = one_right_time_h(vertices, a, ib, con)
        if oracle and ib == 0:
            # Explicit sigma_a sum, avoiding the large complete J^2/H allocation.
            gold_h = cp.zeros_like(h)
            for ia in range(len(times)):
                gold_h[0] += cp.einsum('SLnp,ambn->SLpamb', vertices[ia, ib], a[ia], optimize=True)
            error = metric(cp.asnumpy(gold_h), cp.asnumpy(h))
            if error['relative_L2'] > manifest['algebra_relative_l2_gate']:
                raise AssertionError(('real Pz0 ordered-time H oracle', error))
            real_checks.append(dict(kind='explicit_sigma_a_sum', **error))
        for im,(vs,vr) in enumerate(zip(vvs,vvr)):
            _, _, terms = c3_reduced(h,p,b[ib:ib+1],g,q,vs,vr,con,return_terms=True)
            if oracle and ib==0:
                _,_,reference_terms=c3_reduced(gold_h,p,b[ib:ib+1],g,q,vs,vr,con,return_terms=True)
            for it,spec in enumerate(SPECS):
                if oracle and ib==0:
                    error=metric(cp.asnumpy(reference_terms[spec[0]]),cp.asnumpy(terms[spec[0]]))
                    if error['relative_L2']>manifest['algebra_relative_l2_gate']:
                        raise AssertionError(('six Wick H-order oracle',MOMENTA[im],spec[0],error))
                    real_checks.append(dict(kind=spec[0],Pz=MOMENTA[im],**error))
                wick[it,:,:,im:im+1]+=boundary*spec[2]*cp.asnumpy(terms[spec[0]])
            if oracle and ib==0:
                del reference_terms
            del terms
        if oracle and ib == 0:
            del gold_h
        del h
        cp.get_default_memory_pool().free_all_blocks()
    c3 = np.stack((wick[:4].sum(axis=0), wick[4:].sum(axis=0)))
    projected = project_spin(c3)
    c2, c2check = c2_record
    if not all(np.isfinite(x).all() for x in (c3, wick, c2, projected)):
        raise FloatingPointError('Nonfinite zero-momentum correlators')
    target = output_dir(row, tj, sep, dt)
    if (target / 'receipt.json').exists():
        raise RuntimeError(('refuse overwrite', target))
    specs = projection_specs()
    atomic_npz(target / 'correlators.npz', schema=np.asarray(SCHEMA), profile=np.asarray(PROFILE),
        c3_flavor=c3, signed_wick=wick, c2_common=c2, c3_projected=projected,
        c3_projected_u_minus_d=projected[:, 0] - projected[:, 1],
        c3_axes=np.asarray(['flavor', 'sign', 'pol', 'mom', 'dz', 'm', 'o']),
        wick_axes=np.asarray(['wick', 'sign', 'pol', 'mom', 'dz', 'm', 'o']),
        c2_axes=np.asarray(['pol', 'mom']), projected_axes=np.asarray(['channel', 'flavor', 'sign', 'mom', 'dz']),
        channel_labels=np.asarray(CHANNELS), pol_labels=np.asarray(POLS), flavor_labels=np.asarray(['u', 'd']),
        wick_names=np.asarray([s[0] for s in SPECS]), wick_coefficients=np.asarray([s[2] for s in SPECS]),
        sign_labels=np.asarray(['minus_dz', 'plus_dz']), momenta_z=np.asarray(MOMENTA),
        delta_z=np.arange(NZ), sigma_times=np.asarray(times), window=row['window'],
        t_source=source, t_current=tj, t_sink=sink, tsep=sep, dt=dt, baryon_wrap_sign=boundary,
        is_contact=dt in (0, sep), weight_scheme=np.asarray(SCHEME), weight_applied_count=1,
        flow_time_a2=row['flow_time_a2'], epsilon=row['epsilon'], nstep=row['nstep'], nev_current=NV, nev_src=NE,
        matched_tau_L32x64=row['tau64'],tf_fm2=row['tf_fm2'],flow_radius_fm=row['radius_fm'],
        a_fm=A_FM,ensemble=np.asarray(ENSEMBLE),lattice_size_xyzt=np.asarray([NX,NX,NX,NT]),
        exact_full_time_reference=False, gamma_matrices=specs.gamma,
        gamma_labels=specs.gamma_labels, selected_pol_indices=specs.pol_indices,
        channel_phases=np.asarray([1+0j, -1j, 1j]),
        momentum_direction=np.asarray('z; Pz=0 once and +/-4..6 retained separately; not six directions'),
        fermion_normalization_scheme=np.asarray('ordinary_flowed_fields_no_Zchi'),
        transversity_status=np.asarray('pending_external_physics_confirmation'),
        manifest_sha256=np.asarray(sha256(HERE / 'manifest.json')))
    receipt = dict(status='PASS', schema=SCHEMA, profile=PROFILE, task=row['task'], conf=row['conf'],
        flow_time_a2=row['flow_time_a2'], t_current=tj, tsep=sep, dt=dt, t_source=source, t_sink=sink,
        baryon_wrap_sign=boundary, window=row['window'], weight_scheme=SCHEME, weight_applied_count=1,
        c2_regression=c2check, real_H_and_Wick_regressions=real_checks,
        npz_sha256=sha256(target / 'correlators.npz'), manifest_sha256=sha256(HERE / 'manifest.json'),
        job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-started,
        physical_status='finite_flow_connected_L24x72_B_correlator_no_Zchi_or_matching')
    with np.load(target / 'correlators.npz', allow_pickle=False) as saved:
        for key, value in (('c3_flavor', c3), ('signed_wick', wick), ('c2_common', c2), ('c3_projected', projected)):
            np.testing.assert_array_equal(saved[key], value)
    atomic_json(target / 'receipt.json', receipt)
    print('PZ0_CASE_PASS', row['task'], tj, sep, dt, receipt['seconds'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    require_allocation()
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        raise RuntimeError('four MPI ranks required, one current/GPU')
    cp.cuda.Device(int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', comm.rank))).use()
    cp.get_default_memory_pool().set_limit(size=70 * 1024**3)
    manifest = load_manifest(); row = manifest['tasks'][args.task]
    check_inputs(row)
    tj = CURRENTS[comm.rank]
    seps = (7, 12) if args.smoke else TSEPS
    cases = [(sep, dt) for sep in seps for dt in
        (tuple(sorted({0, 1, sep//2, sep-1, sep})) if args.smoke else range(sep + 1))]
    pending = []
    for sep, dt in cases:
        if (output_dir(row, tj, sep, dt) / 'receipt.json').exists():
            read_case(row, tj, sep, dt)
        else:
            pending.append((sep, dt))
    c2_records = {}
    for sep, dt in pending:
        c2_records[sep, dt] = c2_oracle(row, tj, sep, dt, manifest)
        cp.get_default_memory_pool().free_all_blocks()
    memory('Pz0_C2_oracles_before_J')
    if pending:
        times = support_times(tj, row['window'])
        vertices = load_vertices(row, tj, times)
        for sep, dt in pending:
            contract_case(row, tj, sep, dt, times, vertices, c2_records[sep, dt], manifest,
                oracle=(sep == 7 and dt == 3))
            cp.get_default_memory_pool().free_all_blocks()
        del vertices
    for sep, dt in cases:
        read_case(row, tj, sep, dt)
    checks = read_case(row, tj, 7, 3)[-1]['real_H_and_Wick_regressions']
    if len(checks) != 1+6*len(MOMENTA):
        raise AssertionError('Missing real H/six-Wick regression for every signed momentum')
    gathered = comm.gather(dict(t_current=tj, new_cases=len(pending), verified_cases=len(cases), regressions=checks), root=0)
    if comm.rank == 0:
        name = f'smoke_task{args.task}.json' if args.smoke else f'task{args.task}.json'
        atomic_json(HERE / 'status' / name, dict(status='PASS', task=args.task, conf=row['conf'],
            flow_time_a2=row['flow_time_a2'], smoke=args.smoke, case_count=4*len(cases), currents=gathered,
            manifest_sha256=sha256(HERE / 'manifest.json'), job=os.environ['SLURM_JOB_ID']))
        print('PZ0_TASK_PASS', args.task, 'smoke', args.smoke, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
        raise
