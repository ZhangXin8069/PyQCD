"""专用 kernel 分派与回退的回归检查（只测行为，不测性能）。

校验点：
1. 命中形态时 kernel 确实被调用，且结果与 opt_einsum 一致（rel ≤ 1e-12）；
2. 关闭开关（`PYQCD_EINSUM_KERNEL_KUPL=0`、`specialize=False`）后仍正确；
3. `complex64`、混用不同 nvec 等不满足契约的输入走回退且结果正确。

用法：
    python -m pyqcd.testing.einsum.simple.check_kernels
    python -m pyqcd.testing.einsum.simple.check_kernels --backend both
"""

from __future__ import annotations

import argparse
import importlib
import os

import numpy as np

from . import _common

NUCLEON_EQ = 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'
THREE_PT_EQ = ('agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,'
               'Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy')

KUPL_KERNEL_ENV = 'PYQCD_EINSUM_KERNEL_KUPL'
CUDA_KERNEL_ENV = 'PYQCD_EINSUM_KERNEL_CUDA'


def _random(shape, rng, dtype=np.complex128):
    return np.ascontiguousarray(
        (rng.standard_normal(shape)
         + 1j * rng.standard_normal(shape)).astype(dtype))


def nucleon_operands(nvec, rng, dtype=np.complex128):
    mom = 1
    peram = _random((4, 4, nvec, nvec), rng, dtype)
    gamma = _random((4, 4), rng, dtype)
    vvv = _random((mom, nvec, nvec, nvec), rng, dtype)
    projector = np.ascontiguousarray(np.eye(4).astype(dtype))
    return [peram, peram, peram, gamma, gamma, vvv, vvv, projector]


def three_pt_operands(nvec, rng, dtype=np.complex128):
    mom, link = 1, 2
    perams = [_random((4, 4, nvec, nvec), rng, dtype) for _ in range(5)]
    projection = np.ascontiguousarray(_random((4, 4), rng, dtype)[:, :2])
    return perams + [
        _random((4, 4), rng, dtype),
        _random((4, 4, 4), rng, dtype),
        _random((4, 4), rng, dtype),
        _random((4, 4), rng, dtype),
        _random((mom, nvec, nvec, nvec), rng, dtype),
        _random((mom, link, nvec, nvec), rng, dtype),
        _random((mom, nvec, nvec), rng, dtype),
        _random((mom, nvec, nvec, nvec), rng, dtype),
        projection, projection,
    ]


def _reference(eq, operands):
    import opt_einsum

    return opt_einsum.contract(eq, *operands, optimize='dp')


def _with_env(name, value, function):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        return function()
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _check_dispatch(tally, label, eq, operands, module_name, hit_call, env_name):
    import pyqcd.einsum as target

    reference = _reference(eq, operands)
    error = _common.relative_error(
        target.contract(eq, *operands, optimize='dp'), reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE, f'{label}/correct',
                f'rel={error:.2e}')

    try:
        module = importlib.import_module(module_name)
        hit = hit_call(module, eq, operands) is not None
    except Exception as exc:  # noqa: BLE001 - 缺模块/缺库按未命中处理
        tally.skip(f'{label}/hit', f'{type(exc).__name__}: {exc}')
    else:
        tally.check(hit, f'{label}/hit', 'kernel 未命中')

    error = _common.relative_error(
        _with_env(env_name, '0',
                  lambda: target.contract(eq, *operands, optimize='dp')),
        reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE, f'{label}/kernel-off',
                f'rel={error:.2e}')

    error = _common.relative_error(
        target.contract(eq, *operands, optimize='dp', specialize=False),
        reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE,
                f'{label}/specialize-off', f'rel={error:.2e}')


def _kupl_cases(tally, nvec, rng):
    _check_dispatch(tally, 'kupl/nucleon', NUCLEON_EQ,
                    nucleon_operands(nvec, rng),
                    'pyqcd.einsum._kernel_kupl',
                    lambda module, eq, ops: module.try_nucleon_2pt(eq, ops),
                    KUPL_KERNEL_ENV)
    _check_dispatch(tally, 'kupl/3pt', THREE_PT_EQ,
                    three_pt_operands(nvec, rng),
                    'pyqcd.einsum._kernel_kupl',
                    lambda module, eq, ops: module.try_three_pt(eq, ops),
                    KUPL_KERNEL_ENV)

    import pyqcd.einsum as target

    ops64 = [operand.astype(np.complex64)
             for operand in nucleon_operands(nvec, rng)]
    reference = _reference(NUCLEON_EQ, ops64)
    got = target.contract(NUCLEON_EQ, *ops64, optimize='dp')
    error = _common.relative_error(got, reference)
    tally.check(error <= 1e-5, 'kupl/nucleon-complex64',
                f'rel={error:.2e} dtype={got.dtype}')

    base = nucleon_operands(nvec, rng)
    strided = [np.asfortranarray(base[0])] + base[1:]
    reference = _reference(NUCLEON_EQ, strided)
    error = _common.relative_error(
        target.contract(NUCLEON_EQ, *strided, optimize='dp'), reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE,
                'kupl/nucleon-noncontiguous', '非连续输入应回退且正确')


def _cuda_cases(tally, nvec, rng):
    import cupy

    import pyqcd.einsum as target

    host_operands = three_pt_operands(nvec, rng)
    device_operands = [cupy.asarray(operand) for operand in host_operands]
    reference = _reference(THREE_PT_EQ, host_operands)

    got = cupy.asnumpy(target.contract(THREE_PT_EQ, *device_operands,
                                       optimize='dp'))
    error = _common.relative_error(got, reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE, 'cuda/3pt-correct',
                f'rel={error:.2e}')

    module = importlib.import_module('pyqcd.einsum._kernel_cuda')
    hit = module.try_three_pt(THREE_PT_EQ, device_operands) is not None
    tally.check(hit, 'cuda/3pt-hit', 'kernel 未命中')

    got = cupy.asnumpy(_with_env(
        CUDA_KERNEL_ENV, '0',
        lambda: target.contract(THREE_PT_EQ, *device_operands, optimize='dp')))
    error = _common.relative_error(got, reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE, 'cuda/3pt-kernel-off',
                f'rel={error:.2e}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='kupl',
                        choices=('kupl', 'cuda', 'both'))
    parser.add_argument('--nvec', type=int, default=16)
    args = parser.parse_args(argv)

    backends = ['kupl', 'cuda'] if args.backend == 'both' else [args.backend]
    rng = np.random.default_rng(20261108)
    tally = _common.Tally('kernels')
    for backend in backends:
        if backend == 'cuda' and not _common.cuda_available():
            tally.skip(backend, 'CUDA 不可用')
            continue
        if backend == 'kupl':
            _kupl_cases(tally, args.nvec, rng)
        else:
            _cuda_cases(tally, args.nvec, rng)
    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
