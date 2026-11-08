"""一条命令跑性能对照：pyqcd.einsum vs opt_einsum。

用法：
    python -m pyqcd.testing.einsum.simple.bench
    python -m pyqcd.testing.einsum.simple.bench --profile T1 --backend cpu
    python -m pyqcd.testing.einsum.simple.bench --profile L24 --backend cuda --iters 10

口径：同 profile、同 dtype、同线程、同 planner（各自规划一次，只计稳态执行），
预热后取中位数；双方中位耗时都低于 100 ms 的组标 `too_short`，不参与倍率判定。
"""

from __future__ import annotations

import argparse
import os

from . import _common

MIN_MEANINGFUL_SECONDS = 0.1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default=None)
    parser.add_argument('--backend', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--equations', default=None, help='逗号列表')
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--optimize', default='dp')
    args = parser.parse_args(argv)

    backend = args.backend
    if backend == 'cuda' and not _common.cuda_available():
        print('SKIP: CUDA 不可用')
        return 0
    os.environ.setdefault(
        'PYQCD_EINSUM_NATIVE', 'cuda' if backend == 'cuda' else 'kupl')

    cfg = _common.profile_config(args.profile)
    names = (args.equations.split(',') if args.equations
             else _common.equation_names())
    oracle = _common.oracle_module()
    import pyqcd.einsum as target

    backend_name = 'cupy' if backend == 'cuda' else 'numpy'
    print(f'profile={args.profile or _common.DEFAULT_PROFILE} '
          f'backend={backend} dtype={_common.DTYPE} '
          f'iters={args.iters} warmup={args.warmup} '
          f'optimize={args.optimize}')
    header = (f'{"eq":4s} {"opt_ms":>10s} {"qcd_ms":>10s} {"speedup":>8s} '
              f'{"rel_err":>10s}  note')
    print(header)
    print('-' * len(header))

    worst = None
    for name in names:
        meta = _common.EQUATION_BY_NAME.get(name)
        if meta is None:
            print(f'{name:4s} 未知方程')
            continue
        arrays = _common.operands(name, cfg)
        host = _common.to_backend(arrays, backend)
        # 双方各自规划一次，只计时稳态执行（避免把 opt_einsum 的重复规划开销
        # 算进倍率，那会高估我们）。
        oracle_path = list(oracle.contract_path(
            meta['eq'], *arrays, optimize=args.optimize)[0])
        target_path = list(target.contract_path(
            meta['eq'], *arrays, optimize=args.optimize)[0])
        oracle_result = oracle.contract(meta['eq'], *host,
                                        optimize=oracle_path,
                                        backend=backend_name)
        target_result = target.contract(meta['eq'], *host,
                                        optimize=target_path,
                                        backend=backend_name)
        error = _common.relative_error(
            _common.to_host(target_result, backend),
            _common.to_host(oracle_result, backend))
        opt_seconds = _common.median_seconds(
            lambda: oracle.contract(meta['eq'], *host,
                                    optimize=oracle_path,
                                    backend=backend_name),
            args.iters, args.warmup, backend)
        qcd_seconds = _common.median_seconds(
            lambda: target.contract(meta['eq'], *host,
                                    optimize=target_path,
                                    backend=backend_name),
            args.iters, args.warmup, backend)
        speedup = opt_seconds / qcd_seconds if qcd_seconds else float('inf')
        note = ''
        if min(opt_seconds, qcd_seconds) < MIN_MEANINGFUL_SECONDS:
            note = 'too_short(不作实战对照)'
        elif speedup < 3.0:
            note = 'below-3x'
        else:
            note = 'PASS(>=3x)'
        if error > _common.RELATIVE_TOLERANCE:
            note += f' NUMERIC-FAIL rel={error:.2e}'
        print(f'{name:4s} {opt_seconds * 1e3:10.2f} {qcd_seconds * 1e3:10.2f} '
              f'{speedup:8.3f} {error:10.2e}  {note}')
        if note.startswith('below-3x') or 'NUMERIC-FAIL' in note:
            worst = name
        del arrays, host, oracle_result, target_result
    return 1 if worst else 0


if __name__ == '__main__':
    raise SystemExit(main())
