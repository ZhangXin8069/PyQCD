#!/usr/bin/env python3
"""Read-only production-input comparison; all numerical work needs Slurm."""
from __future__ import annotations

import argparse
import collections
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
PRODUCTION = HERE.parent / 'quark_l32x96_B_expand50_matched_radius_20261028'
sys.path.insert(0, str(PRODUCTION))
sys.path.insert(0, str(PRODUCTION / 'reference'))
from spin2_contractions import c2_reduced, c3_reduced, SPECS
from insertion_kernels import one_right_time_h
from flow_common_l32x96 import spin_objects, seq_peram, phase_3pt, baryon_vvv


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(temporary, path)


class Contractor:
    """Exactly the production shape-keyed cache, with only its engine changed."""
    def __init__(self, engine, optimize='optimal'):
        if engine == 'pyqcd':
            import pyqcd.einsum as opt_einsum
        else:
            import opt_einsum
        self.module = opt_einsum
        self.engine, self.optimize = engine, optimize
        self.expressions, self.planning, self.calls, self.layouts = {}, {}, collections.Counter(), {}

    def __call__(self, equation, *arrays):
        key = (equation, tuple(a.shape for a in arrays))
        self.calls[key] += 1
        if key not in self.expressions:
            started = time.perf_counter()
            self.expressions[key] = self.module.contract_expression(equation, *key[1], optimize=self.optimize)
            self.planning[key] = time.perf_counter() - started
            self.layouts[key] = [dict(shape=a.shape, strides=a.strides, dtype=str(a.dtype),
                                     backend=type(a).__module__.split('.')[0]) for a in arrays]
        return self.expressions[key](*arrays)

    def report(self):
        return dict(engine=self.engine, planning_seconds=sum(self.planning.values()),
                    equations=[dict(equation=k[0], arrays=self.layouts[k], calls=self.calls[k],
                                    planning_seconds=self.planning[k],
                                    contraction_list=repr(e.contraction_list))
                               for k, e in self.expressions.items()])


def host(x):
    if isinstance(x, dict):
        return {k: host(v) for k, v in x.items()}
    if type(x).__module__.split('.')[0] == 'cupy':
        import cupy as cp
        if not isinstance(x, cp.ndarray) or x.device.id != cp.cuda.Device().id:
            raise AssertionError('unexpected output backend/device')
        return cp.asnumpy(x)
    return np.asarray(x)


def compare(reference, actual, gate=2e-10):
    if isinstance(reference, dict):
        if reference.keys() != actual.keys():
            raise AssertionError('output keys differ')
        return {k: compare(reference[k], actual[k], gate) for k in reference}
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise AssertionError(('shape/dtype changed', reference.shape, actual.shape,
                              str(reference.dtype), str(actual.dtype)))
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise AssertionError('nonfinite output')
    delta = actual - reference
    relative = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(reference.ravel()), 1e-300))
    maximum = float(np.max(np.abs(delta))) if delta.size else 0.
    result = dict(relative_L2=relative, max_abs=maximum, shape=list(actual.shape),
                  dtype=str(actual.dtype), gate=gate, passed=relative <= gate)
    if relative > gate:
        raise AssertionError(result)
    return result


def draw(rng, shape, xp, dtype='complex128'):
    value = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    return xp.asarray(value)


def small_validation(xp):
    rng = np.random.default_rng(20261109)
    n, nc, nt, nz = 4, 5, 3, 5
    p = draw(rng, (4, 4, n, n), xp)
    # Sequential peram has reversed eigen axes after seq_peram's dagger.
    a = draw(rng, (nt, 4, 4, n, nc), xp)
    b = draw(rng, (nt, 4, 4, nc, n), xp)
    vertex = draw(rng, (nt, nt, 2, nz, nc, nc), xp, 'complex64')
    vs, vr = (draw(rng, (1, n, n, n), xp) for _ in range(2))
    g, _, q = spin_objects(xp)
    results = {}
    for engine in ('opt_einsum', 'pyqcd'):
        con = Contractor(engine)
        h = one_right_time_h(vertex, a, 1, con)
        u, d, terms = c3_reduced(h, p, b[1:2], g, q, vs, vr, con, return_terms=True)
        results[engine] = host(dict(H=h, C2=c2_reduced(p, vs, vr, g, q, con), u=u, d=d, **terms))
    checks = compare(results['opt_einsum'], results['pyqcd'])
    # Exercise layout, repeated-index/scalar and ellipsis cases explicitly.
    x = draw(rng, (2, 4, 4), xp)
    z = draw(rng, (4, 4), xp)
    direct = {}
    for equation, operands in (
        ('...ij,jk->...ik', (x, z)),
        ('ii->', (z,)),
        ('ij,jk->ik', (z.T, z[:, ::-1])),
        ('aij,ajk->aik', (x[:1], x)),
    ):
        con0, con1 = Contractor('opt_einsum'), Contractor('pyqcd')
        direct[equation] = compare(host(con0(equation, *operands)), host(con1(equation, *operands)))
    return dict(passed=True, backend=xp.__name__, six_wick_and_C2_H=checks, API_cases=direct)


class DeviceMemorySample:
    """50-ms sampled device usage, including allocations outside CuPy's pool."""
    def __init__(self, cp):
        self.cp = cp
        self.stop = threading.Event()
        self.peak, self.samples = 0, 0
        self.device = cp.cuda.Device().id

    def sample(self):
        free, total = self.cp.cuda.runtime.memGetInfo()
        self.peak = max(self.peak, int(total - free))
        self.samples += 1

    def run(self):
        self.cp.cuda.Device(self.device).use()
        while not self.stop.wait(.05):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.sample()


def time_call(cp, function):
    cp.cuda.get_current_stream().synchronize()
    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    started = time.perf_counter()
    start_event.record()
    result = function()
    end_event.record()
    end_event.synchronize()
    wall = time.perf_counter() - started
    return result, dict(wall_seconds=wall, event_interval_seconds=cp.cuda.get_elapsed_time(start_event, end_event) / 1000.)


def benchmark_pair(cp, label, function, repeats, report, expected=None):
    print('STAGE_BEGIN', label, flush=True)
    engines = {name: Contractor(name) for name in ('opt_einsum', 'pyqcd')}
    cold, outputs, samples = {}, {}, {k: [] for k in engines}
    memory = {}
    for name, con in engines.items():
        with DeviceMemorySample(cp) as monitor:
            value, cold[name] = time_call(cp, lambda: function(con))
        memory[name] = dict(sampled_device_used_peak_bytes=monitor.peak, sample_count=monitor.samples,
                            note='50-ms sampling, not an exact peak; shared inputs/caches included')
        outputs[name] = host(value)
        del value
    checks = compare(outputs['opt_einsum'], outputs['pyqcd'])
    expected_checks = None if expected is None else {
        name: compare(expected, value) for name, value in outputs.items()}
    # One additional warmup for each engine, then alternating order per repetition.
    for con in engines.values():
        value, _ = time_call(cp, lambda: function(con))
        del value
    for repetition in range(repeats):
        order = list(engines) if repetition % 2 == 0 else list(reversed(engines))
        for name in order:
            value, timing = time_call(cp, lambda: function(engines[name]))
            samples[name].append(timing)
            del value
    medians = {name: statistics.median(x['wall_seconds'] for x in rows) for name, rows in samples.items()}
    result = dict(label=label, correctness=checks, production_output_check=expected_checks,
                  cold=cold, warm_samples=samples, median_wall_seconds=medians,
                  speedup_opt_over_pyqcd=medians['opt_einsum'] / medians['pyqcd'],
                  memory=memory, contractor={name: con.report() for name, con in engines.items()})
    report['benchmarks'].append(result)
    write_json(HERE / 'results' / 'gpu_result.json', report)
    print('STAGE_RESULT', label, json.dumps(dict(medians=medians, speedup=result['speedup_opt_over_pyqcd'])), flush=True)
    del outputs, engines
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


def gpu_run(args, report):
    import cupy as cp
    import pyqcd.einsum as einsum
    cp.cuda.Device(0).use()
    cp.get_default_memory_pool().set_limit(size=70 * 1024**3)
    report['probe'] = einsum.probe()
    report['cpu_native_validation'] = small_validation(np)
    report['gpu_small_validation'] = small_validation(cp)
    write_json(HERE / 'results' / 'gpu_result.json', report)
    print('SMALL_GPU_PASS', flush=True)

    from common import (config_for, load_manifest, support_times, case_times,
                        output_dir, MOMENTA, sha256, SCHEME)
    from gpu_worker import load_vertices
    manifest = load_manifest()
    row = manifest['tasks'][args.task]
    cfg = config_for(row)
    tj, sep, dt = args.current, args.tsep, args.dt
    source, sink, _, boundary = case_times(tj, sep, dt)
    times = support_times(tj, row['window'])
    report['real_input'] = dict(task=row['task'], conf=row['conf'], tau_L32x96=row['flow_time_a2'],
        matched_tau_L32x64=row['tau64'], current=tj, tsep=sep, dt=dt, source=source,
        sink=sink, Pz=args.momentum, sigma_times=times, weight_scheme=SCHEME,
        weight_applied_count=1, production_manifest_sha256=sha256(PRODUCTION / 'manifest.json'))
    started = time.perf_counter()
    vertices = load_vertices(row, tj, times)
    src = np.load(cfg.peram_path(source), mmap_mode='r')
    snk = np.load(cfg.peram_path(sink), mmap_mode='r')
    a = seq_peram(cp.asarray(np.ascontiguousarray(snk[times, :, :, :400, :200])), cp)
    b = cp.asarray(np.ascontiguousarray(src[times, :, :, :400, :200]))
    p = cp.asarray(src[sink, :, :, :200, :200])
    g, _, q = spin_objects(cp)
    phase = phase_3pt([args.momentum, 0, 0], xp=cp)[None]
    vs = baryon_vvv(phase, np.load(cfg.eigen_path(sink), mmap_mode='r')[:200], cp)
    vr = baryon_vvv(phase, np.load(cfg.eigen_path(source), mmap_mode='r')[:200], cp).conj()
    cp.cuda.get_current_stream().synchronize()
    report['real_input']['preparation_seconds_IO_vertex_VVV'] = time.perf_counter() - started
    report['real_input']['arrays'] = {name: dict(shape=x.shape, dtype=str(x.dtype), bytes=x.nbytes)
                                    for name, x in dict(vertices=vertices, a=a, b=b, p=p, vs=vs, vr=vr).items()}
    product = output_dir(row, tj, sep, dt) / 'correlators.npz'
    receipt_path = product.parent / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'PASS' or sha256(product) != receipt['npz_sha256']:
        raise AssertionError('production saved result/receipt is not valid')
    im = MOMENTA.index(args.momentum)
    with np.load(product, allow_pickle=False) as saved:
        expected_c2 = saved['c2_common'][:, im:im+1]
        expected_wick = saved['signed_wick'][:, :, :, im:im+1]
    report['real_input']['production_result'] = str(product)
    report['real_input']['production_result_sha256'] = receipt['npz_sha256']
    print('REAL_INPUT_PREPARED', json.dumps(report['real_input']), flush=True)

    benchmark_pair(cp, 'real_C2_reduced', lambda con: boundary * c2_reduced(p, vs, vr, g, q, con),
                   args.repeats, report, expected_c2)
    ib = len(times) // 2
    h = one_right_time_h(vertices, a, ib, Contractor('opt_einsum'))
    benchmark_pair(cp, 'real_H_one_sigma_b', lambda con: one_right_time_h(vertices, a, ib, con),
                   args.repeats, report)

    def one_c3(con):
        u, d, terms = c3_reduced(h, p, b[ib:ib+1], g, q, vs, vr, con, return_terms=True)
        return dict(u=u, d=d, **terms)
    benchmark_pair(cp, 'real_C3_six_Wick_one_sigma_b', one_c3, args.repeats, report)
    del h
    cp.get_default_memory_pool().free_all_blocks()

    def full_block(con):
        wick = cp.zeros((6, 2, 4, 1, 21, 4, 4), dtype=cp.complex128)
        for ib in range(len(times)):
            h = one_right_time_h(vertices, a, ib, con)
            _, _, terms = c3_reduced(h, p, b[ib:ib+1], g, q, vs, vr, con, return_terms=True)
            for index, spec in enumerate(SPECS):
                wick[index] += boundary * spec[2] * terms[spec[0]]
            del h, terms
        return wick
    benchmark_pair(cp, 'real_full_sigma_sum_C3_one_Pz_all_z', full_block,
                   args.repeats, report, expected_wick)
    api = importlib.import_module('pyqcd.einsum._api')
    report['native_fallback_diagnostics'] = dict(
        disabled_signatures=[repr(x) for x in getattr(api, '_NATIVE_DISABLED', ())],
        warning_emitted=bool(getattr(api, '_NATIVE_WARNING_EMITTED', False)))
    report['status'] = 'PASS'
    report['scope'] = ('Cached opt_einsum.contract_expression replacement only. '
                       'Gauge/basis flow, vertex construction, VVV, I/O and final projection unchanged. '
                       'One real conf/current/tsep/dt/Pz with full sigma_a/sigma_b and all z; '
                       'not a complete 4-GPU production wall-time benchmark.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--current', type=int, default=0)
    parser.add_argument('--tsep', type=int, default=7)
    parser.add_argument('--dt', type=int, default=3)
    parser.add_argument('--momentum', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical tests must run inside a compute allocation')
    report = dict(status='RUNNING', job=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
                  generated=time.strftime('%Y-%m-%d %H:%M:%S %Z'), arguments=vars(args),
                  versions={name: importlib.metadata.version(name) for name in
                            ('pyqcd', 'opt_einsum', 'numpy', 'cupy-cuda12x')}, benchmarks=[])
    path = HERE / 'results' / f'{args.device}_result.json'
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if args.device == 'cpu':
                report['validation'] = small_validation(np)
                report['status'] = 'PASS'
            else:
                gpu_run(args, report)
        report['warnings'] = [dict(category=x.category.__name__, message=str(x.message)) for x in caught]
    except BaseException as error:
        report['status'] = 'FAILED'
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        write_json(path, report)
        raise
    write_json(path, report)
    print('BENCHMARK_DONE', report['status'], str(path), flush=True)


if __name__ == '__main__':
    main()
