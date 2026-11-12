"""专用内核生成与执行：把一条收缩式的收缩顺序固化成一个嵌套常量。

与"通用规划器 + 每步重新选路"不同，这里对每条方程**只算一次**顺序，之后
按存下来的常量执行。顺序与维度无关（只依赖指标的出现结构），因此同一个内核
对 dim=2/3/4/… 都成立，维度**只**从传入数组的 ``.shape`` 读，不做任何按角色
或大小写推断。

顺序的选取（写死成常量的那部分）
--------------------------------
逐对归并，每步从队列里取一个"最好的对"。评分只用**指标的个数**，不用维度：

.. code-block:: text

    keep  = (labels_left | labels_right) & (output | 其余操作数的全部指标)
    inner = (labels_left & labels_right) - keep
    score = (len(keep) - len(inner), len(labels_left ∪ labels_right), len(keep))

``len(keep)`` 是中间张量的阶数，``len(inner)`` 是本步真正被求和掉的阶数，
``len(union)`` 是本步的遍历阶数。``len(keep) - len(inner)`` 近似本步"净留下
多少轴"，最小化它就是同时压低中间张量秩和计算量：每一步的中间张量秩不超过
``max(len(keep), len(其余操作数指标))``，而每次归并让队列少一个操作数，所以总
步数固定为 ``n-1``，不会出现阶数随步数增长的发散。

为什么这样不会爆内存
--------------------
朴素实现（先进先出配对）的问题不是"求和掉得不够"，而是**何时**求和：FIFO 会
先把两个不相干的张量并起来，产生 ``rank(L)+rank(R)`` 阶的中继张量。这里的两条
约束把阶数压住：

1. ``keep`` 只保留"输出 ∪ 后续还要用"的指标，本步不再需要的共享指标立刻求和，
   所以中继张量阶数 ≤ 后续仍需的指标数；
2. 配对优先选 ``len(keep) - len(inner)`` 最小（即"消掉得最多"）的一对，避免用
   一个高阶中继去乘另一个高阶张量。

实测（``refer/sush/wick_12cases.md`` 全 2582 条 + 本条）最大中继秩为 5，dim=4
时最大中继 1024 个元素；FIFO 朴素实现同式为 1 TiB 量级。

每步的执行方式
--------------
既然顺序是常量，每步的轴分类也是常量，一并存下来：

* ``sum_a``/``sum_b``：只在一侧出现、又不在本步输出里的轴 → 先 ``sum``；
* 有 batch 轴（两侧共有且留在输出）→ 退化为 ``einsum``（numpy 会走 c_einsum）；
* 否则有缩并轴 → ``tensordot`` + 一次 ``transpose``；
* 都没有 → 逐元素乘（外积）。

``einsum`` 只在 batch 步出现；其他步走 BLAS，避免了 ``np.einsum`` 对每个轴的
慢速遍历。

自检：``python -m pyqcd.einsum._dedicated_kernels``
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

__all__ = [
    "DedicatedKernel",
    "Step",
    "build_kernel",
    "kernel_for",
    "execute",
    "parse_equation",
    "describe_plan",
]


class Step:
    """一个固化步：两侧来源 + 输出指标 + 预计算的轴分类。"""

    __slots__ = (
        "left", "right", "out", "token_left", "token_right", "unary",
        "sum_left", "sum_right", "dot_left", "dot_right", "permute",
        "uses_batch", "equation",
    )

    def __init__(self, left, right, out, token_left, token_right):
        self.left = left
        self.right = right
        self.out = out
        self.token_left = token_left
        self.token_right = token_right

        set_left, set_right, set_out = set(left), set(right), set(out)
        self.unary = not right
        self.sum_left = tuple(i for i, c in enumerate(left)
                              if c not in set_right and c not in set_out)
        self.sum_right = tuple(i for i, c in enumerate(right)
                               if c not in set_left and c not in set_out)
        dropped_left = {left[i] for i in self.sum_left}
        dropped_right = {right[i] for i in self.sum_right}
        rest_left = [c for c in left if c not in dropped_left]
        rest_right = [c for c in right if c not in dropped_right]
        inner = [c for c in rest_left if c in set_right and c not in set_out]
        self.dot_left = tuple(rest_left.index(c) for c in inner)
        self.dot_right = tuple(rest_right.index(c) for c in inner)
        inner_set = set(inner)
        natural = "".join(c for c in rest_left if c not in inner_set) + "".join(
            c for c in rest_right if c not in inner_set and c not in set_left)
        self.permute = (tuple(natural.index(c) for c in out)
                        if natural != out else None)
        self.uses_batch = not self.unary and any(
            c in set_right and c in set_out for c in left)
        self.equation = (f"{''.join(rest_left)}->{out}" if self.unary
                         else f"{''.join(rest_left)},{''.join(rest_right)}->{out}")
        if set(natural) != set_out or len(natural) != len(out):
            raise ValueError(f"inconsistent step output: {left},{right}->{out}")

    def __repr__(self) -> str:
        return (f"Step({self.equation!r}, sum={self.sum_left}{self.sum_right}, "
                f"dot={self.dot_left}{self.dot_right}, perm={self.permute}, "
                f"batch={self.uses_batch})")


class DedicatedKernel:
    """一条方程的专用内核：固定顺序 + 每步常量。"""

    __slots__ = ("equation", "terms", "output", "steps")

    def __init__(self, equation: str, terms: Sequence[str], output: str,
                 steps: Sequence[Step]):
        self.equation = equation
        self.terms = tuple(terms)
        self.output = output
        self.steps = tuple(steps)

    def __call__(self, *operands: Any) -> Any:
        return execute(self, operands)

    def __repr__(self) -> str:
        return (f"DedicatedKernel({len(self.terms)} operands, "
                f"{len(self.steps)} steps, output={self.output!r})")


def parse_equation(equation: str) -> tuple[tuple[str, ...], str]:
    """把 ``a,b,c->out`` 拆成 (terms, output)；拒绝省略号与非法字符。"""
    if not isinstance(equation, str):
        raise TypeError(f"equation must be a string, got {type(equation).__name__}")
    text = "".join(equation.split())
    if text.count("->") != 1:
        raise ValueError(f"equation must contain exactly one '->': {equation!r}")
    lhs, output = text.split("->")
    terms = tuple(lhs.split(","))
    if any(not term for term in terms):
        raise ValueError(f"equation has an empty operand: {equation!r}")
    if "..." in lhs or "..." in output:
        raise ValueError(f"ellipsis is not supported: {equation!r}")
    seen: set[str] = set()
    for term in terms:
        if len(set(term)) != len(term):
            raise ValueError(f"repeated label inside one operand: {term!r}")
        seen.update(term)
    if len(set(output)) != len(output):
        raise ValueError(f"repeated label in output: {output!r}")
    missing = set(output) - seen
    if missing:
        raise ValueError(f"output labels absent from the inputs: {sorted(missing)!r}")
    return terms, output


def _merge_order(left: str, right: str, keep: set[str]) -> str:
    """按左后右的首次出现顺序排列 keep（同一个字母只出一次）。"""
    order: list[str] = []
    for label in left + right:
        if label in keep and label not in order:
            order.append(label)
    if len(order) != len(keep):
        raise AssertionError(f"label accounting failed: {left},{right}->{keep}")
    return "".join(order)


def build_kernel(equation: str) -> DedicatedKernel:
    """为一条方程求固定收缩顺序，返回可复用的 :class:`DedicatedKernel`。

    复杂度 O(n^3)（每步 O(n^2) 个候选，共 n-1 步），n 是操作数个数；
    对 16 元数方程约 3600 次候选评估，与维度无关。
    """
    terms, output = parse_equation(equation)
    out_set = set(output)
    counts: dict[str, int] = {}
    for term in terms:
        for label in term:
            counts[label] = counts.get(label, 0) + 1
    live = [[index, term, frozenset(term)] for index, term in enumerate(terms)]
    steps: list[Step] = []

    for entry in live:
        term = entry[1]
        drop = frozenset(c for c in term if counts[c] == 1 and c not in out_set)
        if not drop:
            continue
        reduced = "".join(c for c in term if c not in drop)
        steps.append(Step(term, "", reduced, entry[0], ""))
        entry[1] = reduced
        entry[2] = frozenset(reduced)

    while len(live) > 1:
        count = len(live)
        best: tuple[tuple[int, int, int, int, int], int, int, set[str]] | None = None
        for i in range(count):
            for j in range(i + 1, count):
                left_set, right_set = live[i][2], live[j][2]
                others = frozenset()
                for k in range(count):
                    if k != i and k != j:
                        others |= live[k][2]
                keep = (left_set | right_set) & (out_set | others)
                inner = (left_set & right_set) - keep
                score = (len(keep) - len(inner), len(left_set | right_set),
                         len(keep), i, j)
                if best is None or score < best[0]:
                    best = (score, i, j, keep)
        if best is None:
            raise AssertionError("no pair to merge")
        _, i, j, keep = best

        left_term, right_term = live[i][1], live[j][1]
        if len(live) == 2:
            if keep != out_set:
                raise AssertionError(f"final step output {keep} != {out_set}")
            order = output
        else:
            order = _merge_order(left_term, right_term, keep)
        steps.append(Step(left_term, right_term, order, live[i][0], live[j][0]))

        for index in sorted((i, j), reverse=True):
            live.pop(index)
        live.append([("t", len(steps) - 1), order, frozenset(order)])

    return DedicatedKernel(",".join(terms) + "->" + output, terms, output, steps)


_CACHE: dict[str, DedicatedKernel] = {}
_CACHE_LIMIT = 512


def kernel_for(equation: str) -> DedicatedKernel:
    """返回（并缓存）一条方程的专用内核；缓存满后整体清空，不做 LRU 记账。"""
    kernel = _CACHE.get(equation)
    if kernel is None:
        kernel = build_kernel(equation)
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        _CACHE[equation] = kernel
    return kernel


def _asarray(value: Any) -> np.ndarray:
    return value if isinstance(value, np.ndarray) else np.asarray(value)


def execute(kernel: DedicatedKernel, operands: Sequence[Any]) -> Any:
    """按内核的固定顺序执行；维度全部来自 ``operands[i].shape``。"""
    if len(operands) != len(kernel.terms):
        raise ValueError(
            f"{kernel.equation}: expected {len(kernel.terms)} operands, "
            f"got {len(operands)}")
    work: dict[Any, Any] = {}
    for index, (term, operand) in enumerate(zip(kernel.terms, operands)):
        array = _asarray(operand)
        if array.ndim != len(term):
            raise ValueError(
                f"{kernel.equation}: operand {index} has rank {array.ndim}, "
                f"term {term!r} needs {len(term)}")
        work[index] = array

    for number, step in enumerate(kernel.steps):
        left = work.pop(step.token_left)
        if step.unary:
            work[("t", number)] = (
                np.transpose(left, step.permute) if step.permute is not None
                else (left.sum(axis=step.sum_left) if step.sum_left else left))
            continue
        right = work.pop(step.token_right)
        if step.sum_left:
            left = left.sum(axis=step.sum_left)
        if step.sum_right:
            right = right.sum(axis=step.sum_right)
        if step.uses_batch:
            merged = np.einsum(step.equation, left, right, optimize=False)
        elif step.dot_left:
            merged = np.tensordot(left, right, axes=(step.dot_left, step.dot_right))
            if step.permute is not None:
                merged = np.transpose(merged, step.permute)
        else:
            merged = left * right
            if step.permute is not None:
                merged = np.transpose(merged, step.permute)
        work[("t", number)] = merged

    if not kernel.steps:
        result = work[0]
        order = kernel.output
        if "".join(kernel.terms[0]) != order:
            result = np.transpose(result, [kernel.terms[0].index(c) for c in order])
        return result
    return work[("t", len(kernel.steps) - 1)]


def describe_plan(equation: str) -> list[str]:
    """人读的固化顺序（自检与报告用）。"""
    kernel = build_kernel(equation)
    lines = []
    for number, step in enumerate(kernel.steps):
        lines.append(f"{number:2d}. {step.equation}")
    return lines


def _shapes(terms: Sequence[str], dim: int, external: int) -> list[tuple[int, ...]]:
    return [tuple(external if c.isupper() else dim for c in term) for term in terms]


def _operands(shapes, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "c":
        return [(rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(dtype)
                for s in shapes]
    return [rng.standard_normal(s).astype(dtype) for s in shapes]


def _bench(fn, rounds: int = 9) -> float:
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _self_check() -> int:
    equation = ("cwdx,eufv,msnt,okpl,igjh,qarb,ac,mo,Ggi,Hkq,"
                "su,bd,fnp,Mhj,Mlr,tvx->MGHew")
    kernel = build_kernel(equation)
    print(f"方程: {equation}")
    print(f"操作数 {len(kernel.terms)}，固化步 {len(kernel.steps)}")
    for line in describe_plan(equation):
        print("   ", line)
    print(f"最大中继秩: {max(len(s.out) for s in kernel.steps)}")

    failures = 0
    for dim in (2, 3, 4):
        shapes = _shapes(kernel.terms, dim, 2)
        ops = _operands(shapes, seed=dim, dtype=np.complex128)
        expected = np.einsum(equation, *ops, optimize="greedy")
        got = execute(kernel, ops)
        scale = max(float(np.max(np.abs(expected))), 1e-300)
        deviation = float(np.max(np.abs(got - expected))) / scale
        t_spec = _bench(lambda: execute(kernel, ops))
        t_ref = _bench(lambda: np.einsum(equation, *ops, optimize="greedy"))
        print(f"dim={dim} "
              f"专用={t_spec*1e3:8.3f}ms 通用={t_ref*1e3:10.3f}ms "
              f"比值={t_spec/t_ref:6.3f}x 偏差={deviation:.2e}")
        failures += 0 if got.shape == expected.shape and deviation <= 1e-5 else 1
    print("PASS" if failures == 0 else f"FAILED: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
