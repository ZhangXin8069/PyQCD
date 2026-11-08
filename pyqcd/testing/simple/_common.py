"""`pyqcd.testing.simple` 的共享工具。

风格对齐 `PyQCU/pyqcu/testing/simple`：每个 `check_*.py` 都是可独立运行的脚本，
失败时以非零码退出。

设计要点
--------
* 所有输入**随机生成**（固定 seed 可复现），不使用任何外部数据文件。
* 数值正确性的**参考实现是 `torch.einsum`**；`opt_einsum` 只作为"接口语义对照"，
  即同时检查 pyqcd.einsum 与 opt_einsum 是否都与 torch 一致。
* 每个脚本都可 `python -m pyqcd.testing.simple.check_xxx` 或直接
  `python pyqcd/testing/simple/check_xxx.py` 运行。
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import Any, Sequence

import numpy as np

REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SYMBOLS = 'abcdefghijklmnopqrstuvwxyz'
RELATIVE_TOLERANCE = 1e-12
DEFAULT_SEED = 20261108
_SIZE_CHOICES = (1, 2, 3, 4)


class Tally:
    """PASS / FAIL / SKIP 计数与退出码。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: list[tuple[str, str]] = []

    def check(self, condition: bool, label: str, detail: str = '') -> None:
        if condition:
            self.passed += 1
            print(f'PASS {label}')
        else:
            self.failed += 1
            self.failures.append((label, detail))
            print(f'FAIL {label}' + (f': {detail}' if detail else ''))

    def skip(self, label: str, reason: str) -> None:
        self.skipped += 1
        print(f'SKIP {label}: {reason}')

    def finish(self) -> int:
        print(f'\n{self.name}: {self.passed} passed, '
              f'{self.skipped} skipped, {self.failed} failed')
        for label, detail in self.failures:
            print(f'  FAILED {label}: {detail}')
        return 0 if self.failed == 0 else 1


def target_module() -> Any:
    """返回被测模块 `pyqcd.einsum`。"""
    import pyqcd.einsum

    return pyqcd.einsum


def oracle_module() -> Any:
    """返回接口对照模块 `opt_einsum`。"""
    import opt_einsum

    return opt_einsum


def torch_module() -> Any:
    """返回正确性参考模块 `torch`（仅 CPU）。"""
    import torch

    return torch


def random_labels(rng: np.random.Generator, count: int) -> list[str]:
    labels = list(SYMBOLS[:max(count, 1)])
    rng.shuffle(labels)
    return labels


def random_case(rng: np.random.Generator, min_operands: int = 2,
                max_operands: int = 4,
                size_choices: Sequence[int] = _SIZE_CHOICES,
                ) -> tuple[str, list[np.ndarray]]:
    """生成一个随机的、合法的 einsum 表达式与对应操作数。

    标签尺寸按标签统一分配，保证同一标签在各操作数上尺寸一致
    （否则 numpy/torch 都会报 broadcast 错误）。
    """
    label_count = int(rng.integers(2, 7))
    labels = random_labels(rng, label_count)
    sizes = {label: int(rng.choice(size_choices)) for label in labels}

    operand_count = int(rng.integers(min_operands, max_operands + 1))
    terms = []
    for _ in range(operand_count):
        width = int(rng.integers(1, min(4, label_count) + 1))
        chosen = list(rng.choice(labels, size=width, replace=False))
        if rng.random() < 0.2:          # 允许同一操作数内重复标签（对角）
            chosen.append(chosen[0])
        terms.append(''.join(chosen))

    used = sorted(set(''.join(terms)))
    output = ''.join(label for label in used if rng.random() < 0.6)
    equation = ','.join(terms) + '->' + output
    operands = [
        (rng.standard_normal(tuple(sizes[c] for c in term))
         + 1j * rng.standard_normal(tuple(sizes[c] for c in term))
         ).astype(np.complex128)
        for term in terms
    ]
    return equation, operands


def torch_reference(equation: str, operands: Sequence[np.ndarray]) -> np.ndarray:
    """用 ``torch.einsum`` 计算参考结果（complex128，CPU）。"""
    torch = torch_module()
    tensors = [torch.from_numpy(np.ascontiguousarray(operand)) for operand in operands]
    with torch.no_grad():
        result = torch.einsum(equation, *tensors)
    return np.asarray(result.detach().cpu().numpy())


def relative_error(actual: Any, expected: Any) -> float:
    """范数相对误差：``max|Δ| / max|ref|``。"""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float('inf')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return float('inf')
    scale = max(float(np.max(np.abs(expected), initial=0.0)),
                np.finfo(np.float64).tiny)
    return float(np.max(np.abs(actual - expected), initial=0.0) / scale)


def compare_three_way(equation: str, operands: Sequence[np.ndarray],
                      tally: Tally, label: str, **kwargs: Any) -> None:
    """同时校验 pyqcd 与 opt_einsum 相对 ``torch.einsum`` 的误差。"""
    target = target_module()
    oracle = oracle_module()
    reference = torch_reference(equation, operands)

    target_result = target.contract(equation, *operands, **kwargs)
    oracle_result = oracle.contract(equation, *operands, **kwargs)
    target_error = relative_error(target_result, reference)
    oracle_error = relative_error(oracle_result, reference)

    tally.check(target_error <= RELATIVE_TOLERANCE, f'{label}/pyqcd-vs-torch',
                f'rel={target_error:.2e}')
    tally.check(oracle_error <= RELATIVE_TOLERANCE, f'{label}/opt-vs-torch',
                f'rel={oracle_error:.2e}')
    tally.check(relative_error(target_result, oracle_result) <= RELATIVE_TOLERANCE,
                f'{label}/pyqcd-vs-opt',
                f'rel={relative_error(target_result, oracle_result):.2e}')


def implicit_output(equation: str) -> str:
    """按 numpy 规则推导隐式输出：只出现一次的标签，按字母序排列。"""
    terms = equation.split('->')[0].split(',')
    counts = {}
    for term in terms:
        for label in term:
            counts[label] = counts.get(label, 0) + 1
    return ''.join(sorted(label for label, n in counts.items() if n == 1))


def random_shape_case(rng: np.random.Generator, max_operands: int = 4
                      ) -> tuple[str, list[tuple[int, ...]]]:
    """只生成表达式与形状（不需要真实数据时使用）。"""
    equation, operands = random_case(rng, max_operands=max_operands)
    return equation, [tuple(operand.shape) for operand in operands]


def seeded_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def iter_seeds(count: int, seed: int = DEFAULT_SEED):
    """产生 ``count`` 个独立 RNG，便于逐例复现。"""
    return [np.random.default_rng(seed + 7919 * index) for index in range(count)]


def pairwise(items: Sequence[Any]):
    return itertools.combinations(items, 2)
