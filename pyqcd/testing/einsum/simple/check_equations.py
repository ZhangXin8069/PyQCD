"""方程级正确性：7 条蒸馏方程与 opt_einsum / numpy 逐元素对照。

用法：
    python -m pyqcd.testing.einsum.simple.check_equations
    python -m pyqcd.testing.einsum.simple.check_equations --profile T1
    python -m pyqcd.testing.einsum.simple.check_equations --equations E1,S2
    python -m pyqcd.testing.einsum.simple.check_equations --backend cuda
"""

from __future__ import annotations

import argparse

import numpy as np

from . import _common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default=None,
                        help=f'profile 名（默认 {_common.DEFAULT_PROFILE}）')
    parser.add_argument('--equations', default=None, help='逗号列表')
    parser.add_argument('--backend', default='cpu', choices=('cpu', 'cuda'))
    args = parser.parse_args(argv)

    cfg = _common.profile_config(args.profile)
    names = (args.equations.split(',') if args.equations
             else _common.equation_names())
    backend = args.backend
    if backend == 'cuda' and not _common.cuda_available():
        print('SKIP: CUDA 不可用（cupy 或设备缺失）')
        return 0

    oracle = _common.oracle_module()
    import pyqcd.einsum as target

    print(f'profile={args.profile or _common.DEFAULT_PROFILE} '
          f'nvec={cfg["nvec"]} spatial={cfg["spatial"]} '
          f'n_t={cfg["n_t"]} backend={backend}')
    tally = _common.Tally('equations')
    for name in names:
        meta = _common.EQUATION_BY_NAME.get(name)
        if meta is None:
            tally.check(False, f'{name}/known', f'未知方程 {name}')
            continue
        arrays = _common.operands(name, cfg)
        reference = oracle.contract(meta['eq'], *arrays, optimize='dp')
        host = _common.to_backend(arrays, backend)
        result = target.contract(meta['eq'], *host, optimize='dp',
                                 backend='cupy' if backend == 'cuda' else 'numpy')
        error = _common.relative_error(
            _common.to_host(result, backend), np.asarray(reference))
        tally.check(error <= _common.RELATIVE_TOLERANCE,
                    f'{name}/{backend}',
                    f'rel={error:.3e} > {_common.RELATIVE_TOLERANCE:g} '
                    f'({meta["desc"]})')
        print(f'     {name}: {meta["desc"]}  rel={error:.3e}')
    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
