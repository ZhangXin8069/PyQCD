#!/usr/bin/env python3
"""`backends` 单函数对照：get_func / has_einsum / has_tensordot / infer_backend。

参考实现是 **opt_einsum**（torch 没有这层后端派发）；`get_func` 取到的函数
其数值正确性由 check_contract.py 的 torch 对照覆盖。
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from . import _common
except ImportError:  # 直接以脚本方式运行
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simple import _common


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    tally = _common.Tally('backends')
    target = _common.target_module()
    oracle = _common.oracle_module()

    # 1) has_einsum / has_tensordot：numpy 后端两边都应为真
    for name in ('einsum', 'tensordot'):
        getter = 'has_' + name
        target_value = getattr(target.backends, getter)('numpy')
        oracle_value = getattr(oracle.backends, getter)('numpy')
        tally.check(bool(target_value) == bool(oracle_value), f'{getter}/numpy',
                    f'{target_value} != {oracle_value}')
        tally.check(bool(target_value), f'{getter}/numpy-available',
                    'numpy 缺少该原语')

    # 2) 未知后端必须返回 False / 报错，不能假装可用
    try:
        target_unknown = target.backends.has_einsum('definitely-not-a-backend')
    except Exception:  # noqa: BLE001 - 报错也算 fail-closed
        target_unknown = False
    tally.check(not target_unknown, 'has_einsum/unknown-backend',
                '未知后端被判定为可用')

    # 3) infer_backend：opt_einsum 没有该 API（pyqcd 扩展），只能验证语义
    if hasattr(oracle.backends, 'infer_backend'):
        array = np.zeros((2, 2), dtype=np.complex128)
        target_backend = target.backends.infer_backend(array)
        oracle_backend = oracle.backends.infer_backend(array)
        tally.check(str(target_backend) == str(oracle_backend),
                    'infer_backend/numpy',
                    f'{target_backend!r} != {oracle_backend!r}')
    else:
        tally.skip('infer_backend/parity',
                   'opt_einsum 无 infer_backend（属 pyqcd 扩展 API）')
    array = np.zeros((2, 2), dtype=np.complex128)
    inventory = target.backends.infer_backend(array)
    tally.check(str(inventory) in ('numpy', 'cpu'), 'infer_backend/numpy-array',
                f'numpy 数组被推断为 {inventory!r}')
    try:
        import cupy

        device_array = cupy.zeros((2, 2), dtype=cupy.complex128)
        gpu_backend = target.backends.infer_backend(device_array)
        tally.check(str(gpu_backend) in ('cupy', 'gpu'),
                    'infer_backend/cupy-array',
                    f'cupy 数组被推断为 {gpu_backend!r}')
    except ImportError:
        tally.skip('infer_backend/cupy-array', 'cupy 未安装')

    # 4) get_func('einsum', 'numpy') 必须真的能算，且与 torch 一致
    target_einsum = target.backends.get_func('einsum', 'numpy')
    left = (np.arange(6).reshape(2, 3) + 1j).astype(np.complex128)
    right = (np.arange(12).reshape(3, 4) + 2j).astype(np.complex128)
    result = target_einsum('ij,jk->ik', left, right)
    reference = _common.torch_reference('ij,jk->ik', [left, right])
    error = _common.relative_error(result, reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE, 'get_func/einsum-works',
                f'rel={error:.2e}')

    # 5) get_func 对不存在的函数应抛错而不是返回 None
    try:
        target.backends.get_func('definitely-not-a-function', 'numpy')
    except Exception:  # noqa: BLE001 - 期望抛错
        rejected = True
    else:
        rejected = False
    tally.check(rejected, 'get_func/unknown-function',
                '未知函数未报错')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
