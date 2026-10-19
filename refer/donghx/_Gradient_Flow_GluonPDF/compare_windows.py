#!/usr/bin/env python3
"""B time-pair/Wick pilot with independent finite-window factorization checks."""
import argparse
import json
import os
import time
import numpy as np
from common import (HERE, UPSTREAM, RESULT_ROOT, OLD, SCHEMA, SCHEME, WINDOWS, CURRENTS, TSEPS,
                    MOMENTA, config_for, original_config, load_manifest,
                    check_inputs, atomic_json, atomic_npz, distance, sha256,
                    momentum_labels, case_times)
from mpi4py import MPI
import cupy as cp
from flow_common_l32x64 import (support_times, generalized_vertex, seq_peram,
    baryon_vvv, phase_3pt, spin_objects, create_omega, first_order_weights,
    create_time_pair_omega)
from generate_flowed_vertex_l32x64 import flowed_gauge_time
from spin2_contractions import Contractor, c2_reduced, c2_reference, c3_reduced
from flowed_perambulator_factorization import build_flowed_sequential
from weights_and_terms import SPECS, iter_reduced_right, current_block_h, signed_block_term
from memory_bounded import selected_h_streamed


def memory_snapshot(stage):
    cp.cuda.get_current_stream().synchronize()
    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    row = dict(stage=stage, rank=MPI.COMM_WORLD.rank, pool_used=pool.used_bytes(),
               pool_reserved=pool.total_bytes(), device_free=free, device_total=total)
    print('GPU_MEMORY', json.dumps(row), flush=True)
    return row


def c2_before_vertices(cfg, tj, tsep, manifest):
    """Full-spin oracle retained at every momentum, before any J/A/B is resident."""
    source, sink, _, boundary = case_times(tj, tsep)
    contractor = Contractor('optimal')
    g, _, q = spin_objects(cp)
    p = cp.asarray(np.load(cfg.peram_path(source), mmap_mode='r')[sink,:,:,:200,:200])
    result, gold, checks = [], [], []
    for im, momentum in enumerate(MOMENTA):
        phase = phase_3pt([momentum,0,0], xp=cp)[None]
        vs = baryon_vvv(phase, np.load(cfg.eigen_path(sink), mmap_mode='r')[:200], cp)
        vr = baryon_vvv(phase, np.load(cfg.eigen_path(source), mmap_mode='r')[:200], cp).conj()
        got = boundary*cp.asnumpy(c2_reduced(p,vs,vr,g,q,contractor))
        ref = boundary*cp.asnumpy(c2_reference(p,vs,vr,g,q,contractor))
        check = metric(ref,got)
        if check['relative_L2'] > manifest['algebra_relative_l2_gate']:
            raise AssertionError(('singleton momentum full-spin C2',momentum,check))
        result.append(got); gold.append(ref); checks.append(check)
        del vs,vr,phase
        cp.get_default_memory_pool().free_all_blocks()
        memory_snapshot(f'C2_tsep{tsep}_P{momentum}_done')
    c2 = np.concatenate(result,axis=1)
    comparison = metric(np.concatenate(gold,axis=1),c2)
    comparison['momentum_checks'] = checks
    return c2,comparison


def metric(reference, value):
    diff = value-reference
    if not np.isfinite(reference).all() or not np.isfinite(value).all():
        raise FloatingPointError('non-finite comparison')
    norm = float(np.linalg.norm(reference.ravel()))
    delta = float(np.linalg.norm(diff.ravel()))
    return dict(norm_reference=norm, norm_difference=delta,
                relative_L2=delta/max(norm, 1e-300), max_abs=float(np.max(abs(diff))))


def load_vertices(cfg, oldcfg, tj, taus):
    bases = []
    row = load_manifest()['tasks'][int(os.environ['P4_TASK'])]
    expected_hash = row.get('basis_source_manifest_sha256') or sha256(HERE / 'manifest.json')
    expected_schema = row.get('basis_schema', SCHEMA)
    for sigma in taus:
        meta = json.loads(cfg.basis_done_path(sigma).read_text())
        if (meta['schema'] != expected_schema or tj not in meta['stored_t_current']
                or meta['manifest_sha256'] != expected_hash
                or meta['flow_time'] != cfg.flow_time or meta['n_steps'] != cfg.n_steps):
            raise ValueError(f'incompatible selected basis: {cfg.conf}/{sigma}/{tj}')
        bases.append(cp.asarray(np.load(cfg.basis_path(sigma, tj), mmap_mode='r')))
    second = cp.asarray(create_omega(400))
    first = cp.asarray(first_order_weights(400))
    product = first[:, None]*first[None, :]
    gauge = cp.asarray(flowed_gauge_time(oldcfg, tj))
    vertices = cp.empty((len(taus), len(taus), 2, 21, 400, 400), cp.complex64)
    norms = np.empty((len(taus), len(taus), 2, 21), float)
    for ia, sa in enumerate(taus):
        for ib, sb in enumerate(taus):
            raw = generalized_vertex(bases[ia], bases[ib], gauge, 'z', 21, xp=cp)
            # Apply Omega exactly ONCE at the geometrical current vertex.
            table = create_time_pair_omega(sa, sb, 400, second=second, product=product)
            vertices[ia, ib] = (raw*table[None, None]).astype(cp.complex64)
            norms[ia, ib] = cp.asnumpy(cp.sqrt(cp.sum(abs(vertices[ia, ib])**2, axis=(-2, -1))))
        print('P4_B_VERTEX_ROW', cfg.flow_time, tj, sa, flush=True)
    del bases, gauge, second, first, product, raw, table
    cp.get_default_memory_pool().free_all_blocks()
    return vertices, norms


def compute_case(row, manifest, tj, tsep, cfg, taus, radius, vertices, vertex_norms, c2_record):
    started = time.monotonic()
    source, sink, dt, boundary = case_times(tj, tsep)
    target = RESULT_ROOT / cfg.tag / cfg.conf / f'tJ{tj:03d}' / f'tsep{tsep:02d}'
    target.mkdir(parents=True, exist_ok=True)
    contractor = Contractor('optimal')
    g, _, q = spin_objects(cp)
    src_peram = np.load(cfg.peram_path(source), mmap_mode='r')
    snk_peram = np.load(cfg.peram_path(sink), mmap_mode='r')
    a = seq_peram(cp.asarray(np.ascontiguousarray(snk_peram[taus, :, :, :400, :200])), cp)
    b = cp.asarray(np.ascontiguousarray(src_peram[taus, :, :, :400, :200]))
    p = cp.asarray(src_peram[sink, :, :, :200, :200])
    phases = cp.stack([phase_3pt([m, 0, 0], xp=cp) for m in MOMENTA])
    vs = baryon_vvv(phases, np.load(cfg.eigen_path(sink), mmap_mode='r')[:200], cp)
    vr = baryon_vvv(phases, np.load(cfg.eigen_path(source), mmap_mode='r')[:200], cp).conj()
    del phases
    c2, c2check = c2_record
    if c2check['relative_L2'] > manifest['algebra_relative_l2_gate']:
        raise AssertionError(('rank2 C2 regression', c2check))
    memory_snapshot(f'C3_tsep{tsep}_inputs_loaded')
    old = None
    if cfg.conf == '10000' and cfg.flow_time == .5 and tsep == 7:
        with np.load(OLD / 'results' / cfg.conf / f'tJ{tj:03d}' / 'comparison.npz',
                     allow_pickle=False) as data:
            old = {k: data[k] for k in ('sigma_times', 'signed_wick_blocks',
                                       'c3_by_scheme_flavor', 'c2_common')}
        np.testing.assert_array_equal(old['sigma_times'], support_times(tj, min(WINDOWS)))
        check = metric(old['c2_common'], c2[:, [0, 2, 4]])
        if check['relative_L2'] > manifest['algebra_relative_l2_gate']:
            raise AssertionError(('historical C2', check))
        oldindex = {int(t): i for i, t in enumerate(old['sigma_times'])}
    shells = np.zeros((max(WINDOWS)+1, 6, 2, 4, len(MOMENTA), 21, 4, 4), complex)
    norms = np.zeros((len(taus), len(taus), 6, len(MOMENTA), 2), float)
    baseline_diff2 = baseline_norm2 = 0.
    for ib, sb in enumerate(taus):
        h = current_block_h(vertices[:, ib], a, contractor)
        for im in range(len(MOMENTA)):
            for it, (spec, right, q2, basis) in enumerate(iter_reduced_right(
                    p, b[ib:ib+1], g, q, vs[im:im+1], vr[im:im+1], contractor, cp)):
                value = boundary*cp.asnumpy(signed_block_term(
                    h, spec, right, q2, basis, contractor)[:, :, :, 0])
                if not np.isfinite(value).all():
                    raise FloatingPointError('non-finite ordered Wick block')
                for ia, sa in enumerate(taus):
                    r = max(int(radius[ia]), int(radius[ib]))
                    shells[r, it, :, :, im] += value[ia]
                    norms[ia, ib, it, im] = (np.linalg.norm(value[ia].ravel()),
                                                       np.max(abs(value[ia])))
                    if old is not None and r <= min(WINDOWS) and im in (0, 2, 4):
                        ref = old['signed_wick_blocks'][1, oldindex[sa], oldindex[sb],
                                                        it, :, :, im//2]
                        baseline_diff2 += float(np.sum(abs(value[ia]-ref)**2))
                        baseline_norm2 += float(np.sum(abs(ref)**2))
                del value, right
        del h
        cp.get_default_memory_pool().free_all_blocks()
        print('P4_B_WICK_COLUMN', cfg.flow_time, tj, tsep, sb, flush=True)
    wick = np.stack([shells[:w+1].sum(axis=0) for w in WINDOWS])
    c3 = np.stack((wick[:, :4].sum(axis=1), wick[:, 4:].sum(axis=1)), axis=1)
    regression = {}
    if old is not None:
        for flavor, name in enumerate(('u', 'd')):
            regression[name] = metric(old['c3_by_scheme_flavor'][1, flavor],
                                      c3[0, flavor][:, :, [0, 2, 4]])
        regression['ordered_wick_relative_L2'] = float(np.sqrt(
            baseline_diff2/max(baseline_norm2, 1e-300)))
        values = [regression[k]['relative_L2'] for k in ('u', 'd')]
        values.append(regression['ordered_wick_relative_L2'])
        if max(values) > manifest['prior_B_C3_relative_l2_gate']:
            raise AssertionError(('historical B regression', regression))
    factorization = {}
    if tj == CURRENTS[0] and tsep == TSEPS[0]:
        # First and highest momentum, BOTH signs; explicit shell assembly versus
        # production-style time-factorization. Every tau/window is checked.
        for iw, w in enumerate(WINDOWS):
            idx = np.flatnonzero(radius <= w)
            hs = selected_h_streamed(vertices, a, idx, contractor, cp)
            for im in (0, 1, 6, 7):
                uu, dd = c3_reduced(hs, p, b[idx], g, q,
                                    vs[im:im+1], vr[im:im+1], contractor)
                gold = boundary*np.stack((cp.asnumpy(uu), cp.asnumpy(dd)))
                check = metric(gold, c3[iw, :, :, :, im:im+1])
                factorization[f'w{w}_{momentum_labels()[im]}'] = check
                if check['relative_L2'] > manifest['algebra_relative_l2_gate']:
                    raise AssertionError(('physical B factorization', w, im, check))
                del uu, dd
            del hs
            cp.get_default_memory_pool().free_all_blocks()
            memory_snapshot(f'factorization_w{w}_done')
    convergence = {}
    for lo, hi in ((0, 1), (1, 2), (0, 2)):
        left = (c3[lo, 0], c3[lo, 1], c3[lo, 0]-c3[lo, 1])
        right = (c3[hi, 0], c3[hi, 1], c3[hi, 0]-c3[hi, 1])
        convergence[f'{WINDOWS[lo]}_to_{WINDOWS[hi]}'] = {
            name: metric(x, y) for name, x, y in zip(('u', 'd', 'u-d'), left, right)}
    check_inputs(row)
    atomic_npz(target / 'window_comparison.npz', c3_by_window_flavor=c3,
        signed_wick_by_window=wick, signed_wick_by_radius=shells, c2_common=c2,
        block_wick_norm_max=norms, weighted_vertex_norm=vertex_norms,
        sigma_times=np.asarray(taus), windows=np.asarray(WINDOWS), sigma_radius=radius,
        wick_names=np.asarray([s[0] for s in SPECS]),
        wick_coefficients=np.asarray([s[2] for s in SPECS]),
        weight_scheme=np.asarray(SCHEME), weight_applied_count=1,
        flavor_labels=np.asarray(['u', 'd']),
        pol_labels=np.asarray(['nopol', 'pol15', 'pol25', 'pol35']),
        mom_labels=np.asarray(momentum_labels()), momenta_z=np.asarray(MOMENTA),
        sign_labels=np.asarray(['minus_dz', 'plus_dz']), delta_z=np.arange(21),
        c3_axes=np.asarray(['window', 'flavor', 'sign', 'pol', 'mom', 'dz', 'm', 'o']),
        wick_axes=np.asarray(['window', 'wick', 'sign', 'pol', 'mom', 'dz', 'm', 'o']),
        c2_axes=np.asarray(['pol', 'mom']),
        t_source=source, t_current=tj, t_sink=sink, tsep=tsep, dt=dt,
        flow_time_a2=cfg.flow_time, epsilon=.01, nstep=cfg.n_steps,
        nev_current=400, nev_src=200, a_fm=.0897,
        fermion_normalization_scheme=np.asarray('ordinary_flowed_fields_no_Zchi'),
        exact_full_time_reference=False,
        pilot_scope=np.asarray('four currents, midpoint insertion, six separations; not full time production'))
    # Read-back gate before marking this case complete.
    with np.load(target / 'window_comparison.npz', allow_pickle=False) as data:
        if data['c3_by_window_flavor'].shape != (3, 2, 2, 4, 8, 21, 4, 4):
            raise ValueError('saved C3 shape mismatch')
        if not np.isfinite(data['c3_by_window_flavor']).all():
            raise FloatingPointError('saved C3 not finite')
        np.testing.assert_array_equal(data['c3_by_window_flavor'], c3)
        np.testing.assert_array_equal(data['c2_common'], c2)
    receipt = dict(status='PASS', task=row['task'], conf=cfg.conf,
        flow_time_a2=cfg.flow_time, t_current=tj, tsep=tsep, dt=dt,
        windows=list(WINDOWS), weight_scheme=SCHEME, c2_regression=c2check,
        old_B_subset_regression=regression, physical_factorization_checks=factorization,
        convergence=convergence, seconds=time.monotonic()-started,
        manifest_sha256=sha256(HERE / 'manifest.json'), job=os.environ['SLURM_JOB_ID'],
        interpretation='Implementation PASS only; single-conf finite-window comparison, not ensemble window-systematic acceptance.')
    atomic_json(target / 'receipt.json', receipt)
    print('P4_B_CASE_PASS', cfg.flow_time, tj, tsep, flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or args.task != int(os.environ['P4_TASK']):
        raise RuntimeError('GPU allocation and consistent P4_TASK required')
    manifest = load_manifest()
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        raise RuntimeError('four ranks required, one selected current/GPU')
    cp.cuda.Device(int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', comm.rank))).use()
    cp.get_default_memory_pool().set_limit(size=70*1024**3)
    row = manifest['tasks'][args.task]
    check_inputs(row)
    tj = CURRENTS[comm.rank]
    cfg = config_for(row['conf'])
    completion = HERE / 'status' / f'task{args.task}_comparison.json'
    if completion.exists():
        marker = json.loads(completion.read_text())
        if marker['status'] != 'PASS' or marker['manifest_sha256'] != sha256(HERE/'manifest.json'):
            raise ValueError('existing completion marker is incompatible')
        print('MISSING_ONLY_SKIP_TASK', args.task, flush=True)
        return
    taus = support_times(tj, max(WINDOWS))
    radius = np.asarray([distance(t, tj) for t in taus])
    c2_records = {}
    for tsep in TSEPS:
        c2_records[tsep] = c2_before_vertices(cfg,tj,tsep,manifest)
        cp.get_default_memory_pool().free_all_blocks()
    memory_snapshot('all_C2_oracles_done_before_J')
    vertices, norms = load_vertices(cfg, original_config(row['conf']), tj, taus)
    memory_snapshot('weighted_vertices_loaded')
    receipts = []
    for tsep in TSEPS:
        target = RESULT_ROOT / cfg.tag / cfg.conf / f'tJ{tj:03d}' / f'tsep{tsep:02d}'
        saved = target / 'receipt.json'
        if saved.exists():
            receipt = json.loads(saved.read_text())
            if (receipt['status'] != 'PASS' or receipt['manifest_sha256'] != sha256(HERE/'manifest.json')
                    or receipt['conf'] != cfg.conf or receipt['flow_time_a2'] != cfg.flow_time):
                raise ValueError(f'incompatible existing case: {target}')
            with np.load(target/'window_comparison.npz',allow_pickle=False) as d:
                if (d['c3_by_window_flavor'].shape != (3,2,2,4,8,21,4,4)
                        or not np.isfinite(d['c3_by_window_flavor']).all()):
                    raise ValueError(f'bad existing case: {target}')
            receipts.append(receipt)
        else:
            receipts.append(compute_case(row, manifest, tj, tsep, cfg, taus, radius, vertices, norms, c2_records[tsep]))
        cp.get_default_memory_pool().free_all_blocks()
    gathered = comm.gather(receipts, root=0)
    if comm.rank == 0:
        atomic_json(HERE / 'status' / f'task{args.task}_comparison.json', dict(
            status='PASS', task=args.task, conf=cfg.conf, flow_time_a2=cfg.flow_time,
            cases=[r for group in gathered for r in group],
            manifest_sha256=sha256(HERE / 'manifest.json'), job=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
        raise
