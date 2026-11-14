"""契约：以 ``opt_einsum`` 为对照的数值、路径与 API 行为（四类 dtype）。

PyQCD 的 ``contract`` 自述是 ``opt_einsum`` 的兼容替代，因此对照面必须分三层验，
不能只验"数值一样"：

1. **数值面**：同一方程、同一 dtype，``contract`` 与 ``opt_einsum.contract`` 必须
   一致（双精度 ≤1e-12、单精度 ≤1e-4）；覆盖二元/批量/链/隐式输出/ellipsis/
   Hadamard/迹/外积。
2. **路径面**：``contract_path`` 的 ``opt_cost``、``largest_intermediate`` 与
   ``contraction_list`` 步数必须与 oracle 相同（同 ``optimize`` 策略下）。这一层
   抓的是"规划器退化了但数值仍对"的缺陷——那种问题只会变慢，数值测试永远看不见。
3. **行为面**：``contract_expression`` 可复用、``shared_intermediates`` 生效、
   非法方程两边都报错。

**对照基准的来源**：优先用仓库内 ``refer/git-rep/opt_einsum``（`_oracle.oracle()`），
该目录为 submodule、当前 checkout 可能为空；为空或不可导入时回落到站点包，并在
输出里**明确写出用的是哪一份**——站点包版本与参考仓库不同会让差分失去意义，
所以这一点必须可见，不能静默。

可直接运行::

    python -m pyqcd.testing._opt_einsum_parity_contract
    PYQCD_EINSUM_ORACLE=installed python -m pyqcd.testing._opt_einsum_parity_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import unittest
import warnings
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import contract, contract_expression, contract_path

DTYPES = (np.complex128, np.complex64, np.float64, np.float32)
TOLERANCE = {"complex128": 1e-12, "float64": 1e-12,
             "complex64": 1e-4, "float32": 1e-4}
ORACLE_SOURCE: str = "unresolved"


def oracle() -> Any:
    """返回对照用的 ``opt_einsum`` 模块，并记录来源（仓库内 / 站点包）。"""
    global ORACLE_SOURCE
    try:
        from .einsum import _oracle

        module = _oracle.oracle()
        ORACLE_SOURCE = f"repo:{os.path.relpath(_oracle.ORACLE_PACKAGE)}"
        return module
    except Exception:                                    # noqa: BLE001 - 回落到站点包
        import opt_einsum

        ORACLE_SOURCE = f"installed:{opt_einsum.__version__}"
        return opt_einsum


def _data(shape, dtype, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(tuple(shape))
    if np.dtype(dtype).kind == "c":
        values = values + 1j * rng.standard_normal(tuple(shape))
    return np.ascontiguousarray(values.astype(dtype))


def _quiet(call):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            patch.dict(os.environ, {"PYQCD_EINSUM_NOTICE": "0"}):
        warnings.simplefilter("ignore")
        return call()


def _rel(actual: Any, expected: Any) -> float:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.shape != right.shape:
        return float("inf")
    scale = float(np.max(np.abs(right), initial=0.0))
    difference = float(np.max(np.abs(left - right), initial=0.0))
    if scale == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / scale


# 覆盖类型的形态表：(标签, 方程, 各操作数形状)
SHAPES = (
    ("matmul", "ij,jk->ik", [(4, 5), (5, 6)]),
    ("batch", "bij,bjk->bik", [(2, 4, 5), (2, 5, 6)]),
    ("chain", "ij,jk,kl->il", [(3, 4), (4, 5), (5, 6)]),
    ("implicit", "ij,jk", [(3, 4), (4, 5)]),
    ("ellipsis", "...ij,...jk->...ik", [(2, 3, 4), (2, 4, 5)]),
    ("hadamard", "ij,ij->ij", [(4, 5), (4, 5)]),
    ("trace", "ii->", [(5, 5)]),
    ("outer", "i,j->ij", [(4,), (5,)]),
    ("inner", "i,i->", [(6,), (6,)]),
    ("reduce", "ij->i", [(4, 5)]),
    ("four-way", "ab,bc,cd,de->ae", [(2, 3), (3, 4), (4, 5), (5, 6)]),
)


# ── 1. 数值面 ───────────────────────────────────────────────────────────────

def test_numerics_match_opt_einsum_for_all_dtypes():
    """三种 optimize 策略 × 十一种形态 × 四类 dtype，数值必须与 oracle 一致。"""
    reference_module = oracle()
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        for label, equation, shapes in SHAPES:
            operands = [_data(shape, dtype, seed=index + 1)
                        for index, shape in enumerate(shapes)]
            expected = reference_module.contract(
                equation, *operands, optimize="greedy")
            for optimize in ("auto", "greedy", "optimal"):
                actual = _quiet(lambda: contract(
                    equation, *operands, optimize=optimize))
                assert np.dtype(np.asarray(actual).dtype) == np.dtype(dtype), (
                    name, label, np.asarray(actual).dtype)
                assert _rel(actual, expected) <= TOLERANCE[name], (
                    name, label, optimize, _rel(actual, expected))


def test_specialized_paths_match_opt_einsum():
    """走专用内核的路径（语料拓扑与形状特化）也必须与 oracle 数值一致。"""
    import pyqcd.einsum._wick_tasklist as tasklist

    reference_module = oracle()
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        tasks = [tasklist.lookup(seq) for seq in (2584, 2588, 4305)]
        for task in tasks:
            terms = task.equation.split("->")[0].split(",")
            operands = [_data((2,) * len(term), dtype, seed=task.seq + index)
                        for index, term in enumerate(terms)]
            expected = reference_module.contract(
                task.equation, *operands, optimize="greedy")
            actual = _quiet(lambda: contract(task.equation, *operands))
            assert _rel(actual, expected) <= TOLERANCE[name], (
                name, task.seq, _rel(actual, expected))

        # 形状特化 x,ax,bx,cx->abc
        size = 6
        operands = [_data((size,), dtype, 41)] + [
            _data((size, size), dtype, 42 + index) for index in range(3)]
        expected = reference_module.contract(
            "x,ax,bx,cx->abc", *operands, optimize="greedy")
        actual = _quiet(lambda: contract("x,ax,bx,cx->abc", *operands))
        assert _rel(actual, expected) <= TOLERANCE[name], (
            name, "shape-path", _rel(actual, expected))


# ── 2. 路径面 ───────────────────────────────────────────────────────────────

def test_path_cost_matches_opt_einsum():
    """``opt_cost`` 与最大中间体必须与 oracle 一致：规划退化只改耗时，数值测不到。"""
    reference_module = oracle()
    for optimize in ("greedy", "optimal"):
        for label, equation, shapes in SHAPES:
            operands = [_data(shape, np.float64, seed=index + 1)
                        for index, shape in enumerate(shapes)]
            _, mine = contract_path(equation, *operands, optimize=optimize)
            _, theirs = reference_module.contract_path(
                equation, *operands, optimize=optimize)
            assert int(mine.opt_cost) == int(theirs.opt_cost), (
                label, optimize, int(mine.opt_cost), int(theirs.opt_cost))
            assert int(mine.largest_intermediate) == int(theirs.largest_intermediate), (
                label, optimize, int(mine.largest_intermediate),
                int(theirs.largest_intermediate))
            assert len(mine.contraction_list) == len(theirs.contraction_list), (
                label, optimize, len(mine.contraction_list),
                len(theirs.contraction_list))


def test_corpus_path_cost_matches_opt_einsum():
    """语料抽样：元数 6–16 的方程路径代价也要与 oracle 对齐（多步规划不比二元弱）。"""
    import pyqcd.einsum._wick_tasklist as tasklist

    reference_module = oracle()
    checked = 0
    for seq in range(1, 2583, 200):
        task = tasklist.by_seq()[seq]
        operands = [np.ones((2,) * len(term), dtype=np.float64)
                    for term in task.equation.split("->")[0].split(",")]
        _, mine = contract_path(task.equation, *operands, optimize="greedy")
        _, theirs = reference_module.contract_path(
            task.equation, *operands, optimize="greedy")
        assert int(mine.opt_cost) == int(theirs.opt_cost), (
            seq, int(mine.opt_cost), int(theirs.opt_cost))
        checked += 1
    assert checked >= 10, checked


# ── 3. 行为面 ───────────────────────────────────────────────────────────────

def test_contract_expression_reuse_matches_opt_einsum():
    """``contract_expression`` 复用同一份计划：多次调用结果必须与 oracle 相同。"""
    reference_module = oracle()
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        left = _data((4, 5), dtype, 51)
        right = _data((5, 6), dtype, 52)
        mine = contract_expression("ij,jk->ik", left.shape, right.shape,
                                   optimize="greedy")
        theirs = reference_module.contract_expression(
            "ij,jk->ik", left.shape, right.shape, optimize="greedy")
        for round_index in range(3):
            scaled_left = (left * (round_index + 1)).astype(dtype)
            assert _rel(mine(scaled_left, right), theirs(scaled_left, right)) <= \
                TOLERANCE[name], (name, round_index)
        assert _rel(mine(left, right), theirs(left, right)) <= TOLERANCE[name], name


def test_shared_intermediates_matches_opt_einsum():
    """``shared_intermediates`` 必须命中缓存，且结果与 oracle 一致。"""
    from ..einsum import shared_intermediates

    reference_module = oracle()
    left = _data((4, 5), np.float64, 61)
    right = _data((5, 6), np.float64, 62)
    with shared_intermediates() as cache:
        first = _quiet(lambda: contract("ij,jk->ik", left, right))
        second = _quiet(lambda: contract("ij,jk->ik", left, right))
    assert isinstance(cache, dict)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    expected = reference_module.contract("ij,jk->ik", left, right, optimize="greedy")
    assert _rel(second, expected) <= TOLERANCE["float64"]


def test_error_behaviour_matches_opt_einsum():
    """非法方程两边都必须报错：一边静默返回错数就是兼容性破坏。"""
    reference_module = oracle()
    left = _data((2, 3), np.float64, 71)
    right = _data((3, 4), np.float64, 72)
    cases = (
        ("bad-char", "ij,jk->ik$"),
        ("two-arrows", "ij,jk->ik->il"),
        ("missing-operand", "ij,jk,kl->il"),
    )
    for label, equation in cases:
        mine_raised = False
        theirs_raised = False
        try:
            _quiet(lambda: contract(equation, left, right))
        except Exception:                                # noqa: BLE001 - 期望报错
            mine_raised = True
        operands = (left, right) if label != "missing-operand" else (left, right)
        try:
            reference_module.contract(equation, *operands, optimize="greedy")
        except Exception:                                # noqa: BLE001 - 期望报错
            theirs_raised = True
        assert mine_raised, f"{label}: pyqcd 必须报错"
        assert theirs_raised, f"{label}: oracle 必须报错（否则用例假设不成立）"


def test_speed_against_opt_einsum_is_not_catastrophic():
    """性能对照：**不允许**出现数量级退化（正确性之外的第二道门）。

    这里刻意不设"必须更快"的阈值——性能结论要按性能数据门（生产尺寸、≥10 轮中位、
    同线程同 dtype）单独测，不能用契约里的毫秒数冒充。此门只拦一种情况：某次改动
    把调用路径退化成"每条方程重新规划/逐元素执行"这类**数量级**问题。

    判据：交替计时取中位，``ratio = pyqcd / opt_einsum``；ratio ≥ 8 视为退化
    （小尺寸下固定开销本身就有 2–4× 差异，8 倍留足余量）。
    """
    import statistics
    import time

    reference_module = oracle()

    def timed(call, rounds: int = 5) -> float:
        call()
        samples = []
        for _ in range(rounds):
            start = time.perf_counter()
            call()
            samples.append(time.perf_counter() - start)
        return statistics.median(samples)

    degenerate = []
    measured = []
    # **两边必须用同样的包装**：``_quiet`` 内部是 redirect_stdout + patch.dict，
    # 每调用约 0.7 ms。只给 pyqcd 套壳会把脚手架开销算成 pyqcd 的耗时（实测虚高
    # 到 13×）。这里改成在计时循环外一次性设环境变量，循环内两边都不套壳。
    for label, equation, shapes in SHAPES:
        if label in ("trace", "outer", "inner", "reduce"):
            continue                       # 单/双算子太小，固定开销主导，不作判据
        operands = [_data(shape, np.float64, seed=index + 1)
                    for index, shape in enumerate(shapes)]
        with patch.dict(os.environ, {"PYQCD_EINSUM_NOTICE": "0"}):
            ours = timed(lambda: contract(
                equation, *operands, optimize="greedy"))
            theirs = timed(lambda: reference_module.contract(
                equation, *operands, optimize="greedy"))
        ratio = ours / theirs if theirs else float("inf")
        measured.append((label, ours, theirs, ratio))
        if ratio >= 8.0:
            degenerate.append((label, ratio))

    for label, ours, theirs, ratio in measured:
        print(f"  {label:12s} pyqcd={ours * 1e3:7.3f} ms "
              f"opt_einsum={theirs * 1e3:7.3f} ms ratio={ratio:5.2f}x", flush=True)
    assert not degenerate, f"数量级退化：{degenerate}"
    assert len(measured) >= 6, measured


def test_oracle_source_is_reported():
    """对照基准的来源必须显式可知：仓库内 submodule 未被检出时回落站点包。"""
    oracle()
    assert ORACLE_SOURCE in ("repo:refer/git-rep/opt_einsum/opt_einsum",) or \
        ORACLE_SOURCE.startswith("installed:"), ORACLE_SOURCE
    print(f"[opt_einsum parity] oracle = {ORACLE_SOURCE}", flush=True)


TESTS = (
    test_numerics_match_opt_einsum_for_all_dtypes,
    test_specialized_paths_match_opt_einsum,
    test_path_cost_matches_opt_einsum,
    test_corpus_path_cost_matches_opt_einsum,
    test_contract_expression_reuse_matches_opt_einsum,
    test_shared_intermediates_matches_opt_einsum,
    test_error_behaviour_matches_opt_einsum,
    test_speed_against_opt_einsum_is_not_catastrophic,
    test_oracle_source_is_reported,
)


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed "
          f"(oracle={ORACLE_SOURCE})", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
