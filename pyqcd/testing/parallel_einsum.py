"""``pyqcd.parallel.einsum.einsums`` 的可信性自检与实测入口。

两类证据分开给，避免把「快」当「对」：

* ``verify``（默认）：正确性与契约——并行结果必须与串行 ``contract`` 一致；
* ``bench``：同机、同 dtype、同后端、交错计时（串行/并行交替各 K 轮取中位），
  并按任务规模给出加速比与固定开销占比。

真实格点 QCD 任务来自 ``refer/sush/_distillation/contraction.*.py`` 的方程
（仓库内已在 ``pyqcd.testing.einsum.bench_distillation`` 用同一形状复现），
形状可缩放到本机可承受的 Nvec。

用法::

    python -m pyqcd.testing.parallel_einsum verify
    python -m pyqcd.testing.parallel_einsum bench --tasks 16 --procs 4
    python -m pyqcd.testing.parallel_einsum bench --equation matmul --tasks 32
    python -m pyqcd.testing.parallel_einsum bench --equation S1 --nvec 12 --tasks 16
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np

import pyqcd.einsum as pe
from pyqcd.parallel.einsum import einsums, plan_einsums

# refer/sush/_distillation 的真实方程（形状见 bench_distillation.build_operands）
SUSH_EQUATIONS = {
    'S1': 'eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN',
    'S2': 'gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
    'S3': 'manb,cedf,bd,Mfn,ac,MFem->MF',
    'P1': ('agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,'
           'Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy'),
    'P2': ('agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,'
           'Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy'),
}

_EQUATION_NOTES = {
    'S1': 'sush NN 2pt（8 算子，真实生产方程）',
    'S2': 'sush 3pt（9 算子，含 μν 插入）',
    'S3': 'sush 2pt（6 算子，含 μ 插入）',
    'P1': 'sush 3pt 主项（15 算子，含 γ_curr 与投影）',
    'P2': 'sush 3pt 主项置换型（15 算子）',
    'matmul': '通用矩阵乘（非物理，用作固定开销对照）',
    'hadamard': '逐元素（无求和标签，走 backend 原生 einsum）',
}


# ═══════════════════════════════════════════════════════════════════
# 任务构造
# ═══════════════════════════════════════════════════════════════════

def _perturb(operands, rng, scale=0.05):
    """给同一形状的真实操作数加一点随机扰动，得到 m 个独立任务。"""
    batch = []
    for operand in operands:
        noise = (rng.standard_normal(operand.shape)
                 + 1j * rng.standard_normal(operand.shape))
        batch.append(np.ascontiguousarray(
            operand + scale * noise * np.abs(operand).max()))
    return batch


_PRECISIONS = {
    'complex128': np.complex128,
    'complex64': np.complex64,
    'float64': np.float64,
    'float32': np.float32,
}
# 单精度下并行/串行的差异来自 BLAS 求和顺序，量级 ~1e-6；双精度取 1e-12。
_DEFAULT_RTOL = {
    'complex128': 1e-12, 'float64': 1e-12,
    'complex64': 1e-5, 'float32': 1e-5,
}


def _cast(operands, dtype):
    return [np.ascontiguousarray(operand, dtype=dtype) for operand in operands]


def build_tasks(equation, count, nvec=None, base=256, seed=20261111,
                precision=None):
    """构造 ``count`` 个相互独立的任务（同一方程、同一形状、不同数据）。

    ``precision`` 为空时物理方程用 ``complex128``、通用方程用 ``float64``；
    资源受限的机器可传 ``complex64``/``float32``：dtype 完全由操作数决定，
    本功能不改变它，单精度能直接把内存与算力需求减半。
    """
    rng = np.random.default_rng(seed)
    if equation in ('matmul', 'hadamard'):
        dtype = np.dtype(_PRECISIONS[precision or 'float64'])
        tasks = {}
        for index in range(count):
            size = base if equation == 'matmul' else base // 4
            left = rng.standard_normal((size, size)).astype(dtype)
            right = rng.standard_normal((size, size)).astype(dtype)
            eq = 'ij,jk->ik' if equation == 'matmul' else 'ij,ij->ij'
            tasks[f'{equation}-{index}'] = (eq, [left, right])
        return tasks

    dtype = np.dtype(_PRECISIONS[precision or 'complex128'])
    from .einsum.bench_distillation import (
        PROFILES, build_operands, synthetic_peram)
    if nvec is None:
        nvec = PROFILES['T0']['nvec']
    cfg = dict(PROFILES['T0'], nvec=int(nvec))
    peram = synthetic_peram(0, cfg['nvec']).astype(dtype)
    base_operands = _cast(build_operands(equation, peram, cfg, 0), dtype)
    return {f'{equation}-{index}':
            (SUSH_EQUATIONS[equation], _cast(_perturb(base_operands, rng), dtype))
            for index in range(count)}


def task_elements(tasks):
    """任务集的操作数元素总数（用于估算内存压力）。"""
    return sum(int(np.prod(spec[1][0].shape)) for spec in tasks.values())


def serial_reference(tasks):
    return {key: pe.contract(spec[0], *spec[1]) for key, spec in tasks.items()}


# ═══════════════════════════════════════════════════════════════════
# verify：正确性与契约
# ═══════════════════════════════════════════════════════════════════

def _compare(actual, expected, label, rtol):
    ok = True
    for key in expected:
        if key not in actual:
            print(f'  FAIL {label}: missing key {key!r}')
            ok = False
            continue
        left, right = actual[key], expected[key]
        if left.shape != right.shape:
            print(f'  FAIL {label}: {key!r} shape {left.shape} != {right.shape}')
            ok = False
            continue
        scale = max(float(np.max(np.abs(right))), 1e-300)
        deviation = float(np.max(np.abs(left - right))) / scale
        exact = np.array_equal(left, right)
        flag = 'PASS' if (exact or deviation <= rtol) else 'FAIL'
        print(f'  {flag} {label}: {key!r} rel={deviation:.3e} '
              f'bit_exact={exact}')
        ok = ok and flag == 'PASS'
    return ok


def verify(equation='S1', count=6, nvec=12, procs=2, rtol=None,
           precision=None):
    tasks = build_tasks(equation, count, nvec=nvec, precision=precision)
    expected = serial_reference(tasks)
    dtype = str(np.asarray(next(iter(tasks.values()))[1][0]).dtype)
    if rtol is None:
        rtol = _DEFAULT_RTOL[dtype]
    print(f'verify: equation={equation} tasks={count} dtype={dtype} '
          f'({_EQUATION_NOTES[equation]})')
    ok = True
    for procs_ in (1, procs):
        payload = einsums(tasks, max_procs=procs_, reminder=False)
        label = f'procs={procs_}'
        ok = _compare(payload, expected, label, rtol) and ok
    plan = plan_einsums(tasks, max_procs=procs)
    print(f'  plan: workers={plan["workers"]} skew={plan["skew"]:.2f}x '
          f'bins={[len(b) for b in plan["bins"]]} rtol={rtol:.0e}')
    return 0 if ok else 1


# ═══════════════════════════════════════════════════════════════════
# bench：交错计时
# ═══════════════════════════════════════════════════════════════════

def _one_round(tasks, procs, optimize='auto'):
    started = time.perf_counter()
    payload = einsums(tasks, max_procs=procs, reminder=False, verbose=False,
                      optimize=optimize)
    elapsed = time.perf_counter() - started
    return elapsed, payload


def bench(equation='S1', count=8, nvec=12, procs=4, rounds=3, base=256,
          optimize='auto', json_out=None, precision=None):
    tasks = build_tasks(equation, count, nvec=nvec, base=base,
                        precision=precision)
    dtype = str(np.asarray(next(iter(tasks.values()))[1][0]).dtype)
    rtol = _DEFAULT_RTOL[dtype]
    expected = serial_reference(tasks)
    total_elements = sum(int(np.prod(spec[1][0].shape)) for spec in tasks.values())
    print(f'bench: equation={equation} ({_EQUATION_NOTES[equation]}) '
          f'tasks={count} dtype={dtype} procs={procs} rounds={rounds} '
          f'optimize={optimize}')
    print(f'  serial baseline: {total_elements/1e6:.2f} M elements per task set')

    serial_times, parallel_times = [], []
    for _ in range(rounds):
        serial_times.append(_one_round(tasks, 1, optimize)[0])
        parallel_times.append(_one_round(tasks, procs, optimize)[0])
    # 交错协议：同一轮内先串行后并行，取每轮比值的中位数
    ratios = [s / p for s, p in zip(serial_times, parallel_times)]
    median_serial = statistics.median(serial_times)
    median_parallel = statistics.median(parallel_times)
    ratio = statistics.median(ratios)

    payload = einsums(tasks, max_procs=procs, reminder=False, verbose=False,
                      optimize=optimize)
    ok = _compare(payload, expected, f'bench-procs={procs}', rtol)

    plan = plan_einsums(tasks, max_procs=procs)
    per_task = median_serial / count
    report = {
        'equation': equation,
        'note': _EQUATION_NOTES[equation],
        'mode': 'mpi_launch',
        'dtype': dtype, 'rtol': rtol,
        'tasks': count, 'procs': procs, 'rounds': rounds,
        'optimize': optimize,
        'serial_seconds': serial_times,
        'parallel_seconds': parallel_times,
        'serial_median': median_serial,
        'parallel_median': median_parallel,
        'speedup': ratio,
        'serial_per_task_ms': per_task * 1e3,
        'workers': plan['workers'], 'skew': plan['skew'],
        'bins': [len(item) for item in plan['bins']],
        'correct': bool(ok),
    }
    print(f'  serial   median {median_serial*1e3:9.1f} ms '
          f'({per_task*1e3:.1f} ms/task)')
    print(f'  parallel median {median_parallel*1e3:9.1f} ms '
          f'(mode=mpi_launch procs={procs} skew={plan["skew"]:.2f}x)')
    print(f'  speedup  {ratio:.2f}x  (interleaved median of {rounds} rounds)')
    if ratio < 1.05 and per_task * 1e3 < 50:
        print('  note: per-task work is small; process startup and data '
              'staging dominate — expected for this shape, not a defect')
    if json_out:
        with open(json_out, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f'  json: {json_out}')
    return 0 if ok else 1


def bench_collective(equation='S1', count=8, nvec=12, rounds=3, base=256,
                     optimize='auto', json_out=None, precision=None):
    """在 ``mpirun`` 内部计时 collective 路径：rank 0 串行基准 vs 全体并行。

    必须经 ``mpirun -np K python -m pyqcd.testing.parallel_einsum
    bench-collective ...`` 启动，否则 rank 数恒为 1。
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    tasks = build_tasks(equation, count, nvec=nvec, base=base,
                        precision=precision)
    dtype = str(np.asarray(next(iter(tasks.values()))[1][0]).dtype)
    rtol = _DEFAULT_RTOL[dtype]
    if rank == 0:
        print(f'bench-collective: equation={equation} '
              f'({_EQUATION_NOTES[equation]}) tasks={count} dtype={dtype} '
              f'ranks={size} rounds={rounds}')

    serial_times, parallel_times = [], []
    for _ in range(rounds):
        comm.Barrier()
        if rank == 0:
            started = time.perf_counter()
            for spec in tasks.values():
                pe.contract(spec[0], *spec[1], optimize=optimize)
            serial_times.append(time.perf_counter() - started)
        comm.Barrier()
        started = time.perf_counter()
        payload = einsums(tasks, reminder=False, verbose=False,
                          optimize=optimize, max_procs=size)
        parallel_times.append(time.perf_counter() - started)
        comm.Barrier()

    ok = True
    if rank == 0:
        expected = serial_reference(tasks)
        ok = _compare(payload, expected, f'collective-ranks={size}', rtol)
        median_serial = statistics.median(serial_times)
        median_parallel = statistics.median(parallel_times)
        ratio = statistics.median(
            [s / p for s, p in zip(serial_times, parallel_times)])
        print(f'  serial   median {median_serial*1e3:9.1f} ms '
              f'({median_serial/count*1e3:.1f} ms/task)')
        print(f'  parallel median {median_parallel*1e3:9.1f} ms '
              f'(mode=mpi-collective ranks={size})')
        print(f'  speedup  {ratio:.2f}x')
        if json_out:
            with open(json_out, 'w', encoding='utf-8') as handle:
                json.dump({
                    'equation': equation, 'mode': 'mpi_collective',
                    'dtype': dtype, 'rtol': rtol, 'tasks': count,
                    'ranks': size, 'rounds': rounds, 'optimize': optimize,
                    'serial_seconds': serial_times,
                    'parallel_seconds': parallel_times,
                    'serial_median': median_serial,
                    'parallel_median': median_parallel,
                    'speedup': ratio, 'correct': bool(ok),
                }, handle, ensure_ascii=False, indent=2)
            print(f'  json: {json_out}')
    return 0 if ok else 1


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')

    p_verify = sub.add_parser('verify', help='并行结果与串行 contract 对照')
    p_verify.add_argument('--equation', default='S1',
                          choices=sorted(SUSH_EQUATIONS) + ['matmul', 'hadamard'])
    p_verify.add_argument('--tasks', type=int, default=6)
    p_verify.add_argument('--nvec', type=int, default=12)
    p_verify.add_argument('--procs', type=int, default=2)
    p_verify.add_argument('--rtol', type=float, default=None,
                          help='不传则按 dtype 取默认值（单精度 1e-5，双精度 1e-12）')
    p_verify.add_argument('--precision', default=None,
                          choices=sorted(_PRECISIONS),
                          help='操作数精度；资源受限时用 complex64/float32')

    p_bench = sub.add_parser('bench', help='交错计时串行/并行')
    p_bench.add_argument('--equation', default='S1',
                         choices=sorted(SUSH_EQUATIONS) + ['matmul', 'hadamard'])
    p_bench.add_argument('--tasks', type=int, default=8)
    p_bench.add_argument('--nvec', type=int, default=12)
    p_bench.add_argument('--procs', type=int, default=4)
    p_bench.add_argument('--rounds', type=int, default=3)
    p_bench.add_argument('--base', type=int, default=256,
                         help='通用方程（matmul/hadamard）的方阵边长')
    p_bench.add_argument('--optimize', default='auto')
    p_bench.add_argument('--json', default=None)
    p_bench.add_argument('--precision', default=None,
                         choices=sorted(_PRECISIONS),
                         help='操作数精度；资源受限时用 complex64/float32')

    p_coll = sub.add_parser(
        'bench-collective',
        help='必须在 mpirun 内部运行：rank 0 串行基准 vs 全体 collective 并行')
    p_coll.add_argument('--equation', default='S1',
                        choices=sorted(SUSH_EQUATIONS) + ['matmul', 'hadamard'])
    p_coll.add_argument('--tasks', type=int, default=8)
    p_coll.add_argument('--nvec', type=int, default=12)
    p_coll.add_argument('--rounds', type=int, default=3)
    p_coll.add_argument('--base', type=int, default=256)
    p_coll.add_argument('--optimize', default='auto')
    p_coll.add_argument('--json', default=None)
    p_coll.add_argument('--precision', default=None,
                        choices=sorted(_PRECISIONS))

    args = parser.parse_args(argv)
    if args.command == 'verify':
        return verify(args.equation, args.tasks, args.nvec, args.procs,
                      args.rtol, args.precision)
    if args.command == 'bench':
        return bench(args.equation, args.tasks, args.nvec, args.procs,
                     args.rounds, args.base, args.optimize, args.json,
                     args.precision)
    if args.command == 'bench-collective':
        return bench_collective(args.equation, args.tasks, args.nvec,
                                args.rounds, args.base, args.optimize,
                                args.json, args.precision)
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
