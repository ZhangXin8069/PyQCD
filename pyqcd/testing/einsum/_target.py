"""`pyqcd.einsum` 就绪探针：差分测试与基准共用，不改动被测包。

只做 import + `hasattr` 探测；探测失败不抛给调用方，由调用方打印 SKIP。
"""
from __future__ import annotations

import importlib

MODULE_NAME = 'pyqcd.einsum'

REQUIRED_CALLABLES = ('contract', 'contract_path', 'contract_expression')
REQUIRED_OBJECTS = ('PathInfo',)

PATHINFO_FIELDS = (
    'contraction_list', 'input_subscripts', 'output_subscript', 'path',
    'indices', 'scale_list', 'naive_cost', 'opt_cost', 'speedup',
    'size_list', 'size_dict', 'shapes', 'eq', 'largest_intermediate')

DECIMAL_FIELDS = ('naive_cost', 'opt_cost', 'speedup', 'largest_intermediate')


def probe():
    """返回 `(模块 | None, 缺失项列表)`。"""
    try:
        module = importlib.import_module(MODULE_NAME)
    except Exception as exc:  # noqa: BLE001 - 探针只报告，不抛出
        return None, [f'import {MODULE_NAME} 失败: '
                      f'{type(exc).__name__}: {exc}']
    missing = [name for name in REQUIRED_CALLABLES
               if not callable(getattr(module, name, None))]
    missing += [name for name in REQUIRED_OBJECTS
                if getattr(module, name, None) is None]
    return module, missing


def ready():
    """就绪时返回模块，否则返回 None。"""
    module, missing = probe()
    return None if missing or module is None else module


def skip_reason():
    """返回一行可直接打印的 SKIP 原因。"""
    module, missing = probe()
    if module is None:
        return missing[0]
    return f'{MODULE_NAME} 缺少 API: ' + ', '.join(missing)
