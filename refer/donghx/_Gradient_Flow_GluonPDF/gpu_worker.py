#!/usr/bin/env python3
"""Four MPI ranks/current anchors; supplement dt using frozen B basis only."""
import argparse
import json
import os
import time
import numpy as np
from common import (HERE, CURRENTS, TSEPS, MOMENTA, SCHEME, SCHEMA, config_for,
    gauge_config, support_times, load_manifest, check_inputs, case_times,
    midpoint_dir, output_dir, read_case, sha256, atomic_json, atomic_npz)
from insertion_kernels import one_right_time_h
from mpi4py import MPI
import cupy as cp
from flow_common_l32x64 import (generalized_vertex, seq_peram, baryon_vvv,
    phase_3pt, spin_objects, create_omega, first_order_weights, create_time_pair_omega)
from generate_flowed_vertex_l32x64 import flowed_gauge_time
from spin2_contractions import Contractor, c2_reduced, c2_reference, c3_reduced, SPECS


def metric(reference, actual):
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise FloatingPointError('non-finite regression')
    diff = actual-reference
    return dict(relative_L2=float(np.linalg.norm(diff.ravel()) /
        max(np.linalg.norm(reference.ravel()), 1e-300)), max_abs=float(np.max(abs(diff))))


def memory(stage):
    cp.cuda.get_current_stream().synchronize()
    free, total = cp.cuda.runtime.memGetInfo()
    print('GPU_MEMORY', json.dumps(dict(stage=stage, rank=MPI.COMM_WORLD.rank,
        pool_used=cp.get_default_memory_pool().used_bytes(), device_free=free,
        device_total=total)), flush=True)


def c2_oracle(row, tj, sep, dt, manifest):
    """Do full-spin oracles BEFORE the large J tensor is allocated."""
    cfg = config_for(row)
    source, sink, _, boundary = case_times(tj, sep, dt)
    contract = Contractor('optimal')
    g, _, q = spin_objects(cp)
    p = cp.asarray(np.load(cfg.peram_path(source), mmap_mode='r')[sink, :, :, :200, :200])
    result, errors = [], []
    for momentum in MOMENTA:
        phase = phase_3pt([momentum, 0, 0], xp=cp)[None]
        vs = baryon_vvv(phase, np.load(cfg.eigen_path(sink), mmap_mode='r')[:200], cp)
        vr = baryon_vvv(phase, np.load(cfg.eigen_path(source), mmap_mode='r')[:200], cp).conj()
        actual = boundary*cp.asnumpy(c2_reduced(p, vs, vr, g, q, contract))
        reference = boundary*cp.asnumpy(c2_reference(p, vs, vr, g, q, contract))
        error = metric(reference, actual)
        if error['relative_L2'] > manifest['algebra_relative_l2_gate']:
            raise AssertionError(('C2 oracle', source, sink, momentum, error))
        result.append(actual)
        errors.append(error)
        del vs, vr, phase
        cp.get_default_memory_pool().free_all_blocks()
    return np.concatenate(result, axis=1), dict(momentum_checks=errors,
        max_relative_L2=max(x['relative_L2'] for x in errors))


def load_vertices(row, tj, times):
    """Same ordered-time B tables/precision as old frozen midpoint producer."""
    cfg = config_for(row)
    bases = []
    for sigma in times:
        meta = json.loads(cfg.basis_done_path(sigma).read_text())
        expected = row['basis_metadata'][str(sigma)]
        assert meta['schema'] == expected['schema']
        assert meta['manifest_sha256'] == expected['manifest_sha256']
        assert tj in meta['stored_t_current'] and meta['flow_time'] == row['flow_time_a2']
        bases.append(cp.asarray(np.load(cfg.basis_path(sigma, tj), mmap_mode='r')))
    gauge = cp.asarray(flowed_gauge_time(gauge_config(row), tj))
    second = cp.asarray(create_omega(400))
    first = cp.asarray(first_order_weights(400))
    product = first[:, None]*first[None, :]
    vertices = cp.empty((len(times), len(times), 2, 21, 400, 400), cp.complex64)
    for ia, sa in enumerate(times):
        for ib, sb in enumerate(times):
            raw = generalized_vertex(bases[ia], bases[ib], gauge, 'z', 21, xp=cp)
            table = create_time_pair_omega(sa, sb, 400, second=second, product=product)
            vertices[ia, ib] = (raw*table[None, None]).astype(cp.complex64)
            del raw
        print('B_VERTEX_ROW', row['task'], tj, sa, flush=True)
    del bases, gauge, second, first, product, table
    cp.get_default_memory_pool().free_all_blocks()
    memory('weighted_J_loaded_once_no_disk_vertex')
    return vertices


def contract_case(row, tj, sep, dt, times, vertices, c2_record, write=True):
    started = time.monotonic()
    cfg = config_for(row)
    source, sink, _, boundary = case_times(tj, sep, dt)
    contractor = Contractor('optimal')
    g, _, q = spin_objects(cp)
    src = np.load(cfg.peram_path(source), mmap_mode='r')
    snk = np.load(cfg.peram_path(sink), mmap_mode='r')
    a = seq_peram(cp.asarray(np.ascontiguousarray(snk[times, :, :, :400, :200])), cp)
    b = cp.asarray(np.ascontiguousarray(src[times, :, :, :400, :200]))
    p = cp.asarray(src[sink, :, :, :200, :200])
    phases = cp.stack([phase_3pt([m, 0, 0], xp=cp) for m in MOMENTA])
    vs = baryon_vvv(phases, np.load(cfg.eigen_path(sink), mmap_mode='r')[:200], cp)
    vr = baryon_vvv(phases, np.load(cfg.eigen_path(source), mmap_mode='r')[:200], cp).conj()
    del phases
    wick = np.zeros((6, 2, 4, 8, 21, 4, 4), np.complex128)
    for ib, sigma_b in enumerate(times):
        h = one_right_time_h(vertices, a, ib, contractor)
        for im in range(len(MOMENTA)):
            u, d, terms = c3_reduced(h, p, b[ib:ib+1], g, q,
                vs[im:im+1], vr[im:im+1], contractor, return_terms=True)
            for it, spec in enumerate(SPECS):
                # return_terms are UNSIGNED, whereas u/d already contain signs.
                wick[it, :, :, im] += boundary*spec[2]*cp.asnumpy(terms[spec[0]][:, :, 0])
            del terms, u, d
        del h
        cp.get_default_memory_pool().free_all_blocks()
    c3 = np.stack((wick[:4].sum(axis=0), wick[4:].sum(axis=0)))
    c2, c2check = c2_record
    if not np.isfinite(c3).all() or not np.isfinite(wick).all():
        raise FloatingPointError(('C3 non-finite', row['task'], tj, sep, dt))
    if not write:
        return c3, wick, c2
    target = output_dir(row, tj, sep, dt)
    if (target / 'receipt.json').exists():
        raise RuntimeError(('refuse overwrite', target))
    atomic_npz(target / 'correlators.npz', schema=np.asarray(SCHEMA),
        c3_flavor=c3, signed_wick=wick, c2_common=c2,
        c3_axes=np.asarray(['flavor', 'sign', 'pol', 'mom', 'dz', 'm', 'o']),
        wick_axes=np.asarray(['wick', 'sign', 'pol', 'mom', 'dz', 'm', 'o']),
        c2_axes=np.asarray(['pol', 'mom']), flavor_labels=np.asarray(['u', 'd']),
        wick_names=np.asarray([s[0] for s in SPECS]),
        wick_coefficients=np.asarray([s[2] for s in SPECS]),
        pol_labels=np.asarray(['nopol', 'pol15', 'pol25', 'pol35']),
        sign_labels=np.asarray(['minus_dz', 'plus_dz']), momenta_z=np.asarray(MOMENTA),
        delta_z=np.arange(21), sigma_times=np.asarray(times), window=row['window'],
        t_source=source, t_current=tj, t_sink=sink, tsep=sep, dt=dt,
        baryon_wrap_sign=boundary, is_contact=dt in (0, sep),
        weight_scheme=np.asarray(SCHEME), weight_applied_count=1,
        flow_time_a2=row['flow_time_a2'], epsilon=.01, nstep=row['nstep'],
        nev_current=400, nev_src=200, a_fm=.0897,
        exact_full_time_reference=False,
        fermion_normalization_scheme=np.asarray('ordinary_flowed_fields_no_Zchi'),
        manifest_sha256=np.asarray(sha256(HERE/'manifest.json')))
    receipt = dict(status='PASS', schema=SCHEMA, task=row['task'], conf=row['conf'],
        flow_time_a2=row['flow_time_a2'], t_current=tj, tsep=sep, dt=dt,
        t_source=source, t_sink=sink, baryon_wrap_sign=boundary, window=row['window'],
        weight_scheme=SCHEME, weight_applied_count=1, c2_regression=c2check,
        npz_sha256=sha256(target/'correlators.npz'),
        manifest_sha256=sha256(HERE/'manifest.json'), job=os.environ['SLURM_JOB_ID'],
        seconds=time.monotonic()-started,
        physical_status='finite_flow_connected_B_correlator_no_Zchi_or_matching')
    # Read-back BEFORE the PASS receipt is published.
    with np.load(target/'correlators.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['c3_flavor'], c3)
        np.testing.assert_array_equal(saved['signed_wick'], wick)
        np.testing.assert_array_equal(saved['c2_common'], c2)
    atomic_json(target/'receipt.json', receipt)
    print('INSERTION_CASE_PASS', row['task'], tj, sep, dt, receipt['seconds'], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('GPU allocation required')
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        raise RuntimeError('four MPI ranks required, one current/GPU')
    cp.cuda.Device(int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', comm.rank))).use()
    cp.get_default_memory_pool().set_limit(size=70*1024**3)
    manifest = load_manifest()
    row = manifest['tasks'][args.task]
    check_inputs(row)
    tj = CURRENTS[comm.rank]
    seps = (7, 12) if args.smoke else TSEPS
    pending, reused = [], 0
    for sep in seps:
        dts = (0, 1, sep-1, sep) if args.smoke else range(sep+1)
        for dt in dts:
            if dt == sep//2:
                read_case(row, tj, sep, dt)
                reused += 1
            elif (output_dir(row, tj, sep, dt)/'receipt.json').exists():
                read_case(row, tj, sep, dt, verify_hash=True)
            else:
                pending.append((sep, dt))
    # Also validate each conf/flow/current in the production array, not just smoke.
    regression_cases = [(sep, sep//2) for sep in seps] if args.smoke else [(7, 3)]
    c2_records = {}
    for sep, dt in pending + regression_cases:
        c2_records[sep, dt] = c2_oracle(row, tj, sep, dt, manifest)
        cp.get_default_memory_pool().free_all_blocks()
    memory('C2_oracles_finished_before_J')
    checks = []
    if pending or regression_cases:
        times = support_times(tj, row['window'])
        vertices = load_vertices(row, tj, times)
        for sep, dt in regression_cases:
            actual = contract_case(row, tj, sep, dt, times, vertices, c2_records[sep,dt], write=False)
            gold = read_case(row, tj, sep, dt)
            check = {name: metric(ref, got) for name, ref, got in
                     zip(('c3', 'six_signed_wick', 'c2'), gold[:3], actual)}
            if max(x['relative_L2'] for x in check.values()) > manifest['midpoint_regression_relative_l2_gate']:
                raise AssertionError(('frozen midpoint regression', row['task'], tj, sep, check))
            checks.append(dict(t_current=tj, tsep=sep, checks=check))
            print('MIDPOINT_REGRESSION_PASS', row['task'], tj, sep, check, flush=True)
            cp.get_default_memory_pool().free_all_blocks()
        for sep, dt in pending:
            contract_case(row, tj, sep, dt, times, vertices, c2_records[sep,dt])
            cp.get_default_memory_pool().free_all_blocks()
        del vertices
    # Final finite/identity readback at every requested case before marking task.
    for sep in seps:
        dts = (0, 1, sep-1, sep) if args.smoke else range(sep+1)
        for dt in dts:
            read_case(row, tj, sep, dt, verify_hash=True)
    gathered = comm.gather(dict(t_current=tj, new_cases=len(pending), reused_midpoints=reused,
                               regressions=checks), root=0)
    if comm.rank == 0:
        name = f'smoke_task{args.task}.json' if args.smoke else f'task{args.task}.json'
        atomic_json(HERE/'status'/name, dict(status='PASS', task=args.task,
            conf=row['conf'], flow_time_a2=row['flow_time_a2'], smoke=args.smoke,
            case_count=32 if args.smoke else 252, currents=gathered,
            manifest_sha256=sha256(HERE/'manifest.json'), job=os.environ['SLURM_JOB_ID']))
        print('B_INSERTIONS_TASK_PASS', args.task, 'smoke', args.smoke, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
        raise
