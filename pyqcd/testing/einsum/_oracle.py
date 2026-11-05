"""oracle 侧唯一入口：`refer/git-rep/opt_einsum`（生产代码禁止 import refer/）。

本模块只在 `pyqcd/testing/einsum/` 内使用。import 本模块不加载 opt_einsum；
调用 `oracle()` 时才把参考仓库放到 `sys.path` 首位并解析到仓库内实现。
"""
from __future__ import annotations

import functools
import os
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
ORACLE_ROOT = os.path.join(REPO_ROOT, 'refer', 'git-rep', 'opt_einsum')
ORACLE_PACKAGE = os.path.join(ORACLE_ROOT, 'opt_einsum')


def _purge_opt_einsum():
    """清掉可能来自站点包的 opt_einsum，避免两套实现混用。"""
    for name in [n for n in list(sys.modules)
                 if n == 'opt_einsum' or n.startswith('opt_einsum.')]:
        del sys.modules[name]


def _package_dir(module):
    return os.path.abspath(os.path.dirname(module.__file__))


@functools.lru_cache(maxsize=1)
def oracle():
    """把参考仓库加入 `sys.path` 后返回仓库内的 opt_einsum 模块。

    解析结果必须落在 `refer/git-rep/opt_einsum/opt_einsum`，否则报错
    （站点包 3.4.0 与参考仓库版本不同，混用会让差分测试失去意义）。
    """
    if not os.path.isdir(ORACLE_PACKAGE):
        raise FileNotFoundError(f'缺少参考实现目录: {ORACLE_PACKAGE}')
    if ORACLE_ROOT not in sys.path:
        sys.path.insert(0, ORACLE_ROOT)
    loaded = sys.modules.get('opt_einsum')
    if loaded is not None and _package_dir(loaded) != os.path.abspath(
            ORACLE_PACKAGE):
        _purge_opt_einsum()
    import opt_einsum

    resolved = _package_dir(opt_einsum)
    if resolved != os.path.abspath(ORACLE_PACKAGE):
        raise RuntimeError(
            f'oracle 解析到 {resolved}，应为 {os.path.abspath(ORACLE_PACKAGE)}')
    return opt_einsum


def oracle_version():
    """返回 (版本号, 包目录)，用于报告与 JSON 记录。"""
    module = oracle()
    return getattr(module, '__version__', 'unknown'), _package_dir(module)


def _testing_module():
    """返回 oracle 的 `opt_einsum.testing`（子模块不随包自动导入）。"""
    import importlib

    oracle()
    return importlib.import_module('opt_einsum.testing')


def rand_networks(n, seed, n_ops=None, regularity=4, n_out=2,
                  d_min=2, d_max=5, global_dim=False):
    """包装 `opt_einsum.testing.rand_equation`，返回 n 个确定性网络。

    每个网络是 `{'eq': str, 'shapes': [tuple, ...], 'size_dict': dict}`，
    由 (seed, 序号) 完全决定：同一 seed 重复调用得到逐位相同的表达式与形状。

    `n_ops=None` 时按序号在 3..7 个操作数之间轮换，`global_dim` 每 5 个
    网络打开一次（覆盖带全局广播维的网络）。
    """
    testing = _testing_module()
    if n < 0:
        raise ValueError(f'n 必须非负: {n}')
    networks = []
    for index in range(n):
        ops = int(n_ops) if n_ops is not None else 3 + index % 5
        eq, shapes, size_dict = testing.rand_equation(
            ops, regularity, n_out=n_out, d_min=d_min, d_max=d_max,
            seed=int(seed) + index, global_dim=bool(global_dim) or index % 5 == 4,
            return_size_dict=True)
        networks.append({'eq': eq,
                         'shapes': [tuple(int(d) for d in shape)
                                    for shape in shapes],
                         'size_dict': dict(size_dict)})
    return networks


def rand_arrays(network, seed=0, dtype='complex128'):
    """按网络形状生成确定性操作数（数值面差分测试用）。"""
    import numpy as np

    rng = np.random.default_rng(int(seed))
    is_real = str(dtype).startswith(('float', 'int'))
    arrays = []
    for shape in network['shapes']:
        real = rng.standard_normal(shape)
        if is_real:
            arrays.append(real.astype(dtype))
        else:
            imag = rng.standard_normal(shape)
            arrays.append((real + 1j * imag).astype(dtype))
    return arrays
