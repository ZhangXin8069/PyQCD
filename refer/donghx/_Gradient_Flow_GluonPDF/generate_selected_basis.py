#!/usr/bin/env python3
"""Full-4D flow; retain only requested output slices for a window diagnostic.

Not a full production cache: it intentionally uses a different done schema.
The full untruncated GPU response is passed to the diagnostics before saving.
"""
import argparse
import json
import os
import time
import numpy as np
from common import (HERE, WINDOWS, SCHEMA, config_for, original_config, source_times,
                    selected_currents, load_manifest, check_inputs, atomic_json,
                    atomic_npz, distance, task_row)
from flow_common_l32x64 import first_order_weights
from generate_flowed_vertex_l32x64 import initialize_pyquda


def generate(config, sigma, state, manifest):
    started = time.monotonic()
    targets = selected_currents(sigma)
    done = config.basis_done_path(sigma)
    if done.exists():
        meta = json.loads(done.read_text())
        if (meta['schema'] == SCHEMA and meta['stored_t_current'] == targets
                and meta['manifest_sha256'] == __import__('common').sha256(HERE / 'manifest.json')
                and all(config.basis_path(sigma, t).exists() for t in targets)):
            print('SELECTED_BASIS_SKIP', config.conf, sigma, flush=True)
            return
        raise RuntimeError(f'incomplete/incompatible selected basis: {done}')
    cp, li, dr = state['cp'], state['latt_info'], state['dirac']
    if state['comm'].size != 1 or state['t_start'] != 0 or state['t_stop'] != 64:
        raise RuntimeError('selected basis requires one rank, full 4D domain per GPU')
    eig = np.load(config.eigen_path(sigma), mmap_mode='r')
    maps = {}
    for tj in targets:
        path = config.basis_path(sigma, tj)
        path.parent.mkdir(parents=True, exist_ok=True)
        maps[tj] = np.lib.format.open_memmap(path, mode='w+', dtype='complex64',
                                            shape=(400, 32, 32, 32, 3))
    norm = np.zeros((400, 64), float)
    batch = int(os.environ.get('FLOW_MODE_BATCH', '32'))
    if batch < 4 or batch % 4:
        raise ValueError('FLOW_MODE_BATCH must be a positive multiple of four')
    for start in range(0, 400, batch):
        count = min(batch, 400-start)
        ns = (count+3)//4
        lex = np.zeros((ns, 64, 32, 32, 32, 4, 3), complex)
        for k in range(count):
            lex[k//4, sigma, :, :, :, k % 4, :] = eig[start+k]
        eo = li.evenodd(lex, multi=True)
        src = state['Multi'](li, ns, cp.asarray(eo))
        out = state['Multi'](li, ns)
        # initialize_pyquda sets restart=False, t0=0 and nstep=tau/.01; every batch
        # starts from the same thin gauge, not the previous batch's flowed field.
        state['perform'](out.data_ptrs, src.data_ptrs, dr.invert_param,
                         dr.smear_param, dr.obs_param, ns)
        cp.cuda.get_current_stream().synchronize()
        full = li.lexico(cp.asnumpy(out.data), multi=True)
        for k in range(count):
            v = full[k//4, :, :, :, :, k % 4, :]
            norm[start+k] = np.sum(abs(v)**2, axis=(1, 2, 3, 4))
            for tj, mmap in maps.items():
                mmap[start+k] = v[tj]
        del lex, eo, src, out, full, v
        cp.get_default_memory_pool().free_all_blocks()
        print('SELECTED_FLOW_BATCH', config.conf, sigma, start, count, flush=True)
    if not np.isfinite(norm).all() or np.any(norm.sum(axis=1) <= 0):
        raise FloatingPointError('non-finite/zero flowed column norm')
    for mmap in maps.values():
        mmap.flush()
    total = norm.sum()
    first = first_order_weights(400)
    # Aggregate diagnostics over all output times, separately for each input
    # sigma. Weighted diagnostic is a one-body norm, NOT a bound on B C3.
    diagnostics = {}
    sector_counts = (100,) + (10,)*10 + (200,)
    cuts = np.cumsum((0,) + sector_counts)
    for w in WINDOWS:
        discard = np.asarray([distance(t, sigma) > w for t in range(64)])
        tail = norm[:, discard].sum(axis=1)
        col = tail / norm.sum(axis=1)
        sector = [float(tail[l:r].sum()/norm[l:r].sum()) for l, r in zip(cuts[:-1], cuts[1:])]
        diagnostics[str(w)] = dict(
            discarded_squared_norm_fraction=float(tail.sum()/total),
            max_per_mode_squared_norm_fraction=float(col.max()),
            weighted_one_body_squared_norm_fraction=float(first @ tail / (first @ norm.sum(axis=1))),
            sector_squared_norm_fractions=sector)
    regression = {}
    old = original_config(config.conf)
    for tj in targets:
        if distance(tj, sigma) > task_row()['base_window']:
            continue
        if not old.basis_path(sigma,tj).exists():
            regression[str(tj)] = dict(status='no_archived_central_basis; full4D norm and C3 algebra checked independently')
            continue
        reference = np.load(old.basis_path(sigma, tj), mmap_mode='r')
        got = maps[tj]
        delta2 = denom2 = 0.
        maximum = 0.
        for k in range(0, 400, 16):
            ref = np.asarray(reference[k:k+16], dtype=np.complex128)
            delta = np.asarray(got[k:k+16], dtype=np.complex128)-ref
            delta2 += float(np.sum(abs(delta)**2))
            denom2 += float(np.sum(abs(ref)**2))
            maximum = max(maximum, float(np.max(abs(delta))))
        relative = float(np.sqrt(delta2 / max(denom2, 1e-300)))
        regression[str(tj)] = dict(relative_L2=relative, max_abs=maximum)
        if relative > manifest['baseline_reflow_relative_l2_gate']:
            raise AssertionError(f'central basis reflow regression failed: {sigma}, {tj}, {relative}')
    atomic_npz(done.with_name('norm_diagnostics.npz'), norm_by_mode_time=norm,
               first_order_weights=first, windows=np.asarray(WINDOWS), original_time=sigma)
    meta = dict(schema=SCHEMA, conf=config.conf, original_time=sigma,
        flow_time=config.flow_time, epsilon=config.epsilon,
        n_steps=config.n_steps, fermion_flow_dimension=4,
        temporal_window=max(WINDOWS), stored_t_current=targets, nev=400,
        dtype='complex64', exact_temporal_support=False,
        cache_scope='selected output currents only; not a complete production basis cache',
        manifest_sha256=__import__('common').sha256(HERE / 'manifest.json'),
        input_eigen_path=str(config.eigen_path(sigma)),
        window_diagnostics=diagnostics, central_basis_regression=regression,
        seconds=time.monotonic()-started, job=os.environ.get('SLURM_JOB_ID'))
    # The gate is checked before publishing the completion marker.
    if diagnostics[str(max(WINDOWS))]['max_per_mode_squared_norm_fraction'] > manifest['per_column_squared_norm_gate']:
        atomic_json(done.with_name('failed_diagnostics.json'), meta)
        raise AssertionError(f'aggregate tail gate failed: {config.conf}/{sigma}')
    atomic_json(done, meta)
    print('SELECTED_BASIS_PASS', config.conf, sigma, diagnostics, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('run full-4D flow on an allocated GPU node only')
    manifest = load_manifest()
    if args.task != int(os.environ['P4_TASK']):
        raise RuntimeError('P4_TASK differs from explicit --task')
    row = manifest['tasks'][args.task]
    check_inputs(row)
    if row['basis_reuse']:
        raise RuntimeError('read-only external basis must not be regenerated')
    cfg = config_for(row['conf'])
    state = initialize_pyquda(cfg)
    for sigma in source_times()[args.worker::4]:
        generate(cfg, sigma, state, manifest)
    check_inputs(row)


if __name__ == '__main__':
    main()
