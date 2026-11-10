"""15-case four-engine benchmark: qcd / opt / matx / einsums.

The case list, sizes, warm-up rule and alternating-order protocol are the ones
in ``docs/zh/performance.md`` ("基准脚本", 15 CASES).  This module only adds the
two native bridges and a device switch; it does not change the cases.

Usage
-----
    python -m pyqcd.testing.einsum.bench_four_engines --device cpu
    python -m pyqcd.testing.einsum.bench_four_engines --device gpu --json gpu.json

Engines
-------
* ``qcd``     : ``pyqcd.einsum.contract``
* ``opt``     : ``opt_einsum.contract`` with the ``dp`` path
* ``matx``    : ``pyqcd.testing.matx._bridge`` (GPU only, cuTENSOR)
* ``einsums`` : ``pyqcd.testing.cpu_einsums._bridge`` (CPU only)

MatX pins ``CUTENSORNET_COMPUTE_32F``, so its results are compared with a
1e-4 relative tolerance and marked ``32F``.  Einsums' ``cg::einsum`` has no
unary form and no rank-0 output, and its bridge is binary-only with equal
ranks; unsupported cells are reported as ``N/A`` with the reason.

Environment used for the 2026-11-10 run (all outside the repo)::

    mamba create -y -n einsum-bench -c conda-forge \\
        python=3.12 numpy opt_einsum=3.4.0 scipy
    mamba run -n einsum-bench python -m pip install cupy-cuda13x

GPU runs need ``CUDA_PATH`` pointing at a CUDA toolkit include tree (cupy JIT)
and the MatX bridge dependencies on ``LD_LIBRARY_PATH`` (``libcutensor``,
``libcutensornet``, the ``libcuda`` stub).  CPU runs need the Einsums build
tree on ``LD_LIBRARY_PATH``.  Both bridges are loaded lazily per engine.

Pin threads when comparing across engines: importing ``pyqcd.einsum`` rewrites
``OMP_NUM_THREADS``/``OPENBLAS_NUM_THREADS`` for its own profiles, and the
Einsums library loaded later in the same process inherits them::

    PYQCD_EINSUM_THREADS=off OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
        python -m pyqcd.testing.einsum.bench_four_engines --device cpu

The driver warms each engine up adaptively (5/2/1 calls by one-shot probe) and
allocates the bridges' output buffer once per case, not once per run.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time

import numpy as np
import opt_einsum as oe

import pyqcd
import pyqcd.einsum as pe

ROUNDS = 3
TARGET_SECONDS = 0.3
TOL = 1e-10
MATX_TOL = 1e-4


def _cuda():
    import cupy as cp

    return cp


def _device_helpers(device):
    if device == 'gpu':
        cp = _cuda()

        def to_dev(a):
            return cp.asarray(a)

        def sync():
            cp.cuda.Device(0).synchronize()

        def to_host(a):
            return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)

        def release():
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        return to_dev, sync, to_host, release, name
    return (lambda a: a), (lambda: None), np.asarray, (lambda: None), 'cpu'


# (label, equation, operand maker, output shape) - verbatim from
# docs/zh/performance.md, with the output shapes needed by the bridges.
CASES = (
    ('matmul 2D (real)', 'ij,jk->ik',
     lambda d, r: [d(r.standard_normal((4096, 4096))),
                   d(r.standard_normal((4096, 4096)))], (4096, 4096)),
    ('matmul 2D (complex)', 'ij,jk->ik',
     lambda d, r: [d((r.standard_normal((2048, 2048))
                      + 1j * r.standard_normal((2048, 2048))).astype(np.complex128)),
                   d((r.standard_normal((2048, 2048))
                      + 1j * r.standard_normal((2048, 2048))).astype(np.complex128))],
     (2048, 2048)),
    ('element-wise', 'ij,ij->ij',
     lambda d, r: [d(r.standard_normal((12000, 12000))),
                   d(r.standard_normal((12000, 12000)))], (12000, 12000)),
    ('outer product', 'i,j->ij',
     lambda d, r: [d(r.standard_normal(8000)), d(r.standard_normal(8000))],
     (8000, 8000)),
    ('full reduction', 'ij,ij->',
     lambda d, r: [d(r.standard_normal((12000, 12000))),
                   d(r.standard_normal((12000, 12000)))], ()),
    ('partial reduction', 'ij->i',
     lambda d, r: [d(r.standard_normal((16000, 16000)))], (16000,)),
    ('chain 3 ops', 'ij,jk,kl->il',
     lambda d, r: [d(r.standard_normal((3072, 3072))) for _ in range(3)],
     (3072, 3072)),
    ('chain 4 ops', 'ij,jk,kl,lm->im',
     lambda d, r: [d(r.standard_normal((2048, 2048))) for _ in range(4)],
     (2048, 2048)),
    ('hyper-edge', 'ij,jk,ik->i',
     lambda d, r: [d(r.standard_normal((2048, 2048))) for _ in range(3)],
     (2048,)),
    ('batched GEMM', 'bij,bjk->bik',
     lambda d, r: [d(r.standard_normal((16, 1024, 1024))),
                   d(r.standard_normal((16, 1024, 1024)))], (16, 1024, 1024)),
    ('ellipsis', '...ij,...jk->...ik',
     lambda d, r: [d(r.standard_normal((256, 512, 512))),
                   d(r.standard_normal((256, 512, 512)))], (256, 512, 512)),
    ('batch × matrix', 'abc,cd->abd',
     lambda d, r: [d(r.standard_normal((8, 1024, 512))),
                   d(r.standard_normal((512, 1024)))], (8, 1024, 1024)),
    ('many small batches', 'bij,bjk->bik',
     lambda d, r: [d(r.standard_normal((4096, 64, 64))),
                   d(r.standard_normal((4096, 64, 64)))], (4096, 64, 64)),
    ('nucleon 2pt (kernel)', 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M',
     lambda d, r: [
         d(r.standard_normal((4, 4, 112, 112)) + 1j * r.standard_normal((4, 4, 112, 112))),
         d(r.standard_normal((4, 4, 112, 112)) + 1j * r.standard_normal((4, 4, 112, 112))),
         d(r.standard_normal((4, 4, 112, 112)) + 1j * r.standard_normal((4, 4, 112, 112))),
         d(r.standard_normal((4, 4)) + 1j * r.standard_normal((4, 4))),
         d(r.standard_normal((4, 4)) + 1j * r.standard_normal((4, 4))),
         d(r.standard_normal((4, 112, 112, 112)) + 1j * r.standard_normal((4, 112, 112, 112))),
         d(r.standard_normal((4, 112, 112, 112)) + 1j * r.standard_normal((4, 112, 112, 112))),
         d(r.standard_normal((4, 4)) + 1j * r.standard_normal((4, 4))),
     ], ()),
    ('gamma sandwich (kernel)', 'ab,...bce,cd->...dae',
     lambda d, r: [
         d(r.standard_normal((4, 4)) + 1j * r.standard_normal((4, 4))),
         d(r.standard_normal((8192, 4, 4, 1024)) + 1j * r.standard_normal((8192, 4, 4, 1024))),
         d(r.standard_normal((4, 4)) + 1j * r.standard_normal((4, 4))),
     ], (8192, 4, 4, 1024)),
)


def _ranks(ops):
    return [int(np.ndim(op)) for op in ops]


def matx_supported(ops):
    ranks = _ranks(ops)
    return 1 <= len(ops) <= 2 and max(ranks) <= 4 and (
        len(ops) == 1 or ranks[0] == ranks[1])


def einsums_supported(ops, out_shape, equation):
    ranks = _ranks(ops)
    return (len(ops) == 2 and len(set(ranks)) == 1 and 1 <= ranks[0] <= 4
            and 1 <= len(out_shape) <= 4 and '...' not in equation)


def _numpy_to_einsums(eq):
    from pyqcd.testing.cpu_einsums import _contract

    return _contract.numpy_to_einsums(eq)


def make_runners(device):
    from pyqcd.testing.matx import _bridge as matx_bridge
    from pyqcd.testing.cpu_einsums import _bridge as einsums_bridge

    def qcd(eq, ops, out_shape, out=None):
        return pe.contract(eq, *ops)

    def opt(eq, ops, out_shape, out=None):
        return oe.contract(eq, *ops, optimize='dp')

    def matx(eq, ops, out_shape, out=None):
        matx_bridge.einsum_into(out, eq, *ops)
        return out

    def einsums(eq, ops, out_shape, out=None):
        einsums_bridge.einsum_into(out, _numpy_to_einsums(eq), *ops)
        return out

    return {'qcd': qcd, 'opt': opt, 'matx': matx, 'einsums': einsums}


def one(call, inner, sync):
    sync()
    best = float('inf')
    for _ in range(inner):
        start = time.perf_counter()
        call()
        sync()
        best = min(best, time.perf_counter() - start)
    return best


def relative_error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float('inf')
    scale = max(float(np.max(np.abs(expected))) if expected.size else 0.0,
                np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected)) / scale)


def run(device, engines, rounds):
    to_dev, sync, to_host, release, device_name = _device_helpers(device)
    runners = make_runners(device)

    from pyqcd.testing.matx import _bridge as matx_bridge
    from pyqcd.testing.cpu_einsums import _bridge as einsums_bridge

    print(f'pyqcd {pyqcd.__version__} {pyqcd.__file__} | {device}: {device_name}')
    if device == 'gpu':
        cp = _cuda()
        print(f'cupy {cp.__version__} | '
              f'matx abi={matx_bridge.probe()["abi_version"]} '
              f'cutensor={matx_bridge.probe()["cutensor"]}')
    print(f'einsums backend={einsums_bridge.probe()["backend_enabled"]} | '
          f'rounds={rounds} target={TARGET_SECONDS}s | engines={",".join(engines)}')

    results = {}
    for label, equation, maker, out_shape in CASES:
        rng = np.random.default_rng(0)
        operands = maker(to_dev, rng)
        reference = runners['opt'](equation, operands, out_shape)
        reference_host = to_host(reference)
        del reference

        available = {}
        for name in engines:
            if name == 'matx':
                ok = matx_supported(operands)
                reason = '' if ok else 'bridge 仅支持 1–2 操作数/rank≤4/同 rank'
            elif name == 'einsums':
                ok = einsums_supported(operands, out_shape, equation)
                reason = '' if ok else (
                    'cg::einsum 仅二元同 rank、无 rank-0 输出/一元/省略号')
            else:
                ok = True
                reason = ''
            available[name] = (ok, reason)

        row = {'case': label, 'equation': equation,
               'n_ops': len(operands), 'out_shape': list(out_shape),
               'engines': {}}
        # The bridges take a caller-owned output. Allocate one per case (not
        # once for the whole run): holding every case's output would keep
        # gigabytes resident and distort later GPU cases.
        outs = {}
        if device == 'gpu' and available.get('matx', (False,))[0]:
            outs['matx'] = _cuda().zeros(out_shape, dtype=operands[0].dtype)
        if device == 'cpu' and available.get('einsums', (False,))[0]:
            outs['einsums'] = np.zeros(
                out_shape, dtype=np.asarray(operands[0]).dtype)

        def call(name):
            return runners[name](equation, operands, out_shape, outs.get(name))

        working = []
        for name, (ok, reason) in available.items():
            if not ok:
                row['engines'][name] = {'status': 'N/A', 'reason': reason}
                continue
            try:
                got = call(name)
                host = to_host(got)
                error = relative_error(host, reference_host)
                tolerance = MATX_TOL if name == 'matx' else TOL
                precision = '32F' if name == 'matx' else '64F'
                row['engines'][name] = {
                    'status': 'OK' if error <= tolerance else 'NUMERIC-FAIL',
                    'rel_error': error, 'tolerance': tolerance,
                    'precision': precision,
                }
                working.append(name)
                del got, host
            except Exception as exc:  # noqa: BLE001 - 记录而不是中断
                row['engines'][name] = {
                    'status': 'ERROR',
                    'reason': f'{type(exc).__name__}: {exc}'[:200]}

        if working:
            # CUDA clocks and allocator pools need more than one call to settle
            # (the MatX batched case moves ~35% between the 2nd and 10th call),
            # but a 45 s opt_einsum case must not pay five of them. Scale the
            # warm-up count with a one-shot probe.
            probe0 = {
                name: one((lambda e=name: call(e)),
                          1, sync)
                for name in working
            }
            warmups = {name: (5 if t < 0.1 else 2 if t < 1.0 else 1)
                       for name, t in probe0.items()}
            for name in working:
                for _ in range(warmups[name]):
                    call(name)
            sync()
            probe = {
                name: one((lambda e=name: call(e)),
                          1, sync)
                for name in working
            }
            inner = max(1, min(9, int(
                TARGET_SECONDS / max(min(probe.values()), 1e-6))))
            samples = {name: [] for name in working}
            for rnd in range(rounds):
                order = (working[rnd % len(working):]
                         + working[:rnd % len(working)])
                for engine in order:
                    try:
                        samples[engine].append(one(
                            (lambda e=engine: call(e)), inner, sync))
                    except Exception as exc:  # noqa: BLE001
                        samples[engine].append(None)
                        row['engines'][engine] = {
                            'status': 'ERROR',
                            'reason': f'timing: {type(exc).__name__}: {exc}'[:200]}
            for name in working:
                times = [t for t in samples[name] if t is not None]
                if len(times) != rounds or not times:
                    continue
                median = statistics.median(times)
                spread = (max(times) / min(times)) if min(times) > 0 else 0.0
                row['engines'][name].update(
                    {'median_s': median, 'inner': inner, 'spread': spread})

        results[label] = row
        row_ok = ' '.join(
            f'{n}={row["engines"][n]["median_s"]*1e3:.2f}ms'
            if row['engines'][n].get('median_s') is not None
            else f'{n}={row["engines"][n]["status"]}'
            for n in engines)
        print(f'{label:26s} {row_ok}')
        del operands, reference_host
        gc.collect()
        release()

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--engines', default=None,
                        help='逗号列表，默认按设备选择')
    parser.add_argument('--rounds', type=int, default=ROUNDS)
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)

    default = 'qcd,opt,einsums' if args.device == 'cpu' else 'qcd,opt,matx'
    engines = [name.strip() for name in (args.engines or default).split(',')
               if name.strip()]
    results = run(args.device, engines, args.rounds)

    print('\n' + '=' * 100)
    header = f'{"case":26s} ' + ' '.join(f'{name:>16s}' for name in engines)
    print(header)
    for label, row in results.items():
        cells = []
        for name in engines:
            info = row['engines'].get(name, {})
            if info.get('status') == 'OK':
                cells.append(f'{info["median_s"]*1e3:9.2f}ms{"*" if info["precision"] == "32F" else " "}')
            else:
                cells.append(f'{info.get("status", "N/A"):>16s}')
        print(f'{label:26s} ' + ' '.join(f'{cell:>16s}' for cell in cells))

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump({'device': args.device, 'engines': engines,
                       'results': results}, handle, indent=2,
                      ensure_ascii=False, default=float)
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
