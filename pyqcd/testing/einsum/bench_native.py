"""qcd-einsum 严格基准命令行入口。

本入口在全新解释器中以默认 4 线程启动基准，避免 ``pyqcd.testing`` 包初始化
先行导入 NumPy。默认 process 隔离模式会再用本入口启动两个独立 worker，
分别预热和计时两个实现。固定 oracle 子模块缺失时，必须显式设置
``PYQCD_EINSUM_ORACLE=installed`` 才会使用站点包，产物会标记为非冻结。
"""
from __future__ import annotations

import os
import sys

DEFAULT_THREADS = 4
THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
)
_BOOTSTRAP_ENV = 'PYQCD_BENCH_BOOTSTRAP_ACTIVE'
_THREAD_PIN_ENV = 'PYQCD_BENCH_THREAD_PINNED'


def _pin_thread_environment():
    raw = os.environ.get('PYQCD_BENCH_THREADS', str(DEFAULT_THREADS))
    try:
        threads = int(raw)
    except ValueError as exc:
        raise SystemExit(
            f'PYQCD_BENCH_THREADS 必须是正整数，收到 {raw!r}') from exc
    if threads < 1:
        raise SystemExit(
            f'PYQCD_BENCH_THREADS 必须是正整数，收到 {raw!r}')
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(threads)
    os.environ[_THREAD_PIN_ENV] = '1'


def _bootstrap():
    _pin_thread_environment()
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if root not in sys.path:
        sys.path.insert(0, root)
    if os.environ.get(_BOOTSTRAP_ENV) == '1':
        return

    # `python -m` 会先导入 pyqcd.testing.__init__ 并加载 NumPy；发现后重新
    # 以本文件为脚本启动，让最终计时进程在线程变量设置后才导入 NumPy。
    if 'numpy' in sys.modules:
        environment = os.environ.copy()
        environment[_BOOTSTRAP_ENV] = '1'
        os.execve(
            sys.executable,
            [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
            environment,
        )

_bootstrap()

from pyqcd.testing.einsum.bench_distillation import main  # noqa: E402


if __name__ == '__main__':
    raise SystemExit(main())
