"""契约：``optimize`` 传 callable / ``PathOptimizer`` 时不得被静默忽略。

计划缓存（``_CONTRACT_PLAN_CACHE``）原本用 ``_optimize_cache_key(optimize)`` 建键，
而该函数对 callable 一律返回 ``None``：两个**不同**的优化器因此拿到完全相同的键，
第二个优化器一次都不会被调用，只是"结果碰巧还对"。数值测试抓不到这类缺陷，
必须钉住"谁被调用、调了几次、返回的路径是否真的被执行、缓存是否只按身份命中"：

1. 两个不同 callable 都被调用，且各占一条计划缓存；
2. 同一 callable 重复调用仍命中缓存（不重规划、只调用一次）；
3. callable 返回的路径真的被执行（不是调完就丢）；
4. ``opt_einsum.paths.PathOptimizer`` 实例同样不被忽略，且同实例可命中缓存；
5. 对照：``"dp"`` / ``"greedy"`` / ``"auto"`` 的缓存行为不变（同名只规划一次，
   不同名互不共享条目）；
6. 数值与 ``optimize=False`` 的朴素路径一致；
7. 缓存键持有优化器本体，被回收对象的 id 不会指向另一个优化器；
8. token 语义：不可变参数原样透传，callable 按对象身份相等，未知类型不缓存。

可直接运行::

    python -m pyqcd.testing._einsum_optimize_callable_contract
"""
from __future__ import annotations

import contextlib
import gc
import io
import os
import unittest
import warnings
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import _api, contract

try:                                                # 可选对照依赖
    from opt_einsum.paths import PathOptimizer as _PathOptimizerBase
except Exception:                                   # pragma: no cover - 缺依赖时退化
    class _PathOptimizerBase:                       # type: ignore[no-redef]
        pass

EXPR = "ij,jk,kl->il"
SHAPES = ((3, 4), (4, 5), (5, 6))
LEFT_FIRST = [(0, 1), (0, 1)]        # 先收 (0,1)：ij,jk->ik
RIGHT_FIRST = [(1, 2), (0, 1)]       # 先收 (1,2)：jk,kl->jl
OUTER_FIRST = [(0, 2), (0, 1)]       # 先做外积：ij,kl->ijkl（与 dp/greedy 都不同）


def _operands(dtype: Any = np.float64, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.ascontiguousarray(rng.standard_normal(shape).astype(dtype))
            for shape in SHAPES]


def _recorder(log: list[str], name: str, path: list[tuple[int, int]]):
    """返回一个记录调用并把 ``path`` 交给规划器的 callable 优化器。"""
    def optimizer(inputs, output, size_dict, memory_limit=None):
        log.append(name)
        return [tuple(step) for step in path]

    return optimizer


class _CountingOptimizer(_PathOptimizerBase):
    """``opt_einsum`` 风格的 ``PathOptimizer`` 实例：计数 + 固定路径。"""

    def __init__(self, path: list[tuple[int, int]], name: str = "") -> None:
        self.path = [tuple(step) for step in path]
        self.name = name or type(self).__name__
        self.calls = 0

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        self.calls += 1
        return list(self.path)


def _quiet(call):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            patch.dict(os.environ, {"PYQCD_EINSUM_NOTICE": "0"}):
        warnings.simplefilter("ignore")
        return call()


@contextlib.contextmanager
def _plan_cache_isolation():
    """隔离计划缓存：用例内从空缓存开始，结束后恢复原条目。"""
    saved = _api._CONTRACT_PLAN_CACHE.copy()
    _api._CONTRACT_PLAN_CACHE.clear()
    try:
        yield _api._CONTRACT_PLAN_CACHE
    finally:
        _api._CONTRACT_PLAN_CACHE.clear()
        _api._CONTRACT_PLAN_CACHE.update(saved)


def _assert_matches_off_path(got: Any, operands: list[np.ndarray]) -> None:
    expected = np.einsum(EXPR, *operands, optimize=False)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-12, atol=1e-12)


# ── 1. 两个不同 callable 都必须被调用 ───────────────────────────────────────

def test_two_distinct_callables_are_both_invoked():
    """两个不同 callable 各调用一次、各占一条计划缓存（修复前只见到 A）。"""
    operands = _operands(seed=1)
    calls: list[str] = []
    first = _recorder(calls, "A", LEFT_FIRST)
    second = _recorder(calls, "B", RIGHT_FIRST)
    with _plan_cache_isolation() as cache:
        got_first = _quiet(lambda: contract(EXPR, *operands, optimize=first))
        got_second = _quiet(lambda: contract(EXPR, *operands, optimize=second))
        assert calls == ["A", "B"], (
            f"第二个 callable 被静默忽略（计划缓存键不含优化器身份）: {calls}")
        assert len(cache) == 2, f"两个不同 callable 共享了同一条计划: {list(cache)}"
    _assert_matches_off_path(got_first, operands)
    _assert_matches_off_path(got_second, operands)


# ── 2. 同一 callable 仍然命中缓存 ───────────────────────────────────────────

def test_same_callable_reuses_the_cached_plan():
    """同一 callable 重复调用只规划一次、只被调用一次（不能修成不缓存）。"""
    operands = _operands(seed=2)
    calls: list[str] = []
    optimizer = _recorder(calls, "A", RIGHT_FIRST)
    with _plan_cache_isolation() as cache:
        with patch.object(_api, "contract_path", wraps=_api.contract_path) as planning:
            first = _quiet(lambda: contract(EXPR, *operands, optimize=optimizer))
            second = _quiet(lambda: contract(EXPR, *operands, optimize=optimizer))
        assert planning.call_count == 1, (
            f"同一 callable 的第二次调用重复规划了 {planning.call_count} 次")
        assert calls == ["A"], f"同一 callable 重复调用不应再次执行优化器: {calls}"
        assert len(cache) == 1, f"同一 callable 应只占一条计划: {list(cache)}"
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    _assert_matches_off_path(first, operands)


# ── 3. callable 返回的路径必须真的被执行 ────────────────────────────────────

def test_callable_path_is_actually_adopted():
    """执行用的收缩列表必须等于 callable 给出的路径，而不是被换成默认路径。"""
    operands = _operands(seed=3)
    forced = _recorder([], "forced", OUTER_FIRST)
    with _plan_cache_isolation():
        with patch.object(_api, "_core_contract", wraps=_api._core_contract) as spy:
            _quiet(lambda: contract(EXPR, *operands, optimize=forced))
        executed = [call.args[1] for call in spy.call_args_list]
    assert len(executed) == 1, f"应恰好执行一次收缩: {executed}"
    used = [step[2] for step in executed[0]]
    _, expected = _api.contract_path(EXPR, *operands, optimize=forced, einsum_call=True)
    assert used == [step[2] for step in expected], (
        f"执行的路径不是 callable 给的路径: {used} != {[step[2] for step in expected]}")
    for name in ("dp", "greedy"):
        _, other = _api.contract_path(EXPR, *operands, optimize=name, einsum_call=True)
        assert used != [step[2] for step in other], (
            f"强制路径与 {name} 相同，这条断言区分不出路径是否被采用")


# ── 4. PathOptimizer 实例同样不被忽略 ───────────────────────────────────────

def test_path_optimizer_instances_are_not_ignored():
    """两个 ``PathOptimizer`` 实例各调用一次，同一实例重复调用命中缓存。"""
    operands = _operands(seed=4)
    first = _CountingOptimizer(RIGHT_FIRST, "first")
    second = _CountingOptimizer(OUTER_FIRST, "second")
    assert isinstance(first, _PathOptimizerBase)
    with _plan_cache_isolation() as cache:
        got_first = _quiet(lambda: contract(EXPR, *operands, optimize=first))
        got_second = _quiet(lambda: contract(EXPR, *operands, optimize=second))
        again = _quiet(lambda: contract(EXPR, *operands, optimize=first))
        assert first.calls == 1, f"同一实例的第二次调用应命中计划缓存: {first.calls}"
        assert second.calls == 1, "第二个 PathOptimizer 实例被静默忽略"
        assert len(cache) == 2, f"两个实例应各占一条计划: {list(cache)}"
    np.testing.assert_allclose(
        np.asarray(got_first), np.asarray(got_second), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(got_first), np.asarray(again))
    _assert_matches_off_path(got_first, operands)


# ── 5. 对照组：字符串优化器的缓存行为不变 ───────────────────────────────────

def test_builtin_optimizer_caching_is_unchanged():
    """``"dp"``/``"greedy"``/``"auto"``：同名只规划一次，不同名各占一条。"""
    operands = _operands(seed=5)
    for name in ("dp", "greedy", "auto"):
        with _plan_cache_isolation() as cache:
            with patch.object(_api, "contract_path", wraps=_api.contract_path) as planning:
                first = _quiet(lambda: contract(EXPR, *operands, optimize=name))
                second = _quiet(lambda: contract(EXPR, *operands, optimize=name))
            assert planning.call_count == 1, (name, planning.call_count)
            assert len(cache) == 1, (name, list(cache))
            key = next(iter(cache))
            assert key[5] == name, f"键里的 optimize token 应仍原样保留字符串: {key}"
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        _assert_matches_off_path(first, operands)
    with _plan_cache_isolation() as cache:
        for name in ("dp", "greedy", "auto"):
            _quiet(lambda: contract(EXPR, *operands, optimize=name))
        assert len(cache) == 3, f"不同优化器名不得共享计划: {list(cache)}"


# ── 6. 数值与 off 路径一致 ──────────────────────────────────────────────────

def test_callable_results_match_the_off_path():
    """callable / PathOptimizer 路径的数值必须与 ``optimize=False`` 一致。"""
    for dtype in (np.float64, np.complex128):
        operands = _operands(dtype, seed=6)
        expected = _quiet(lambda: contract(
            EXPR, *operands, optimize=False, specialize=False))
        optimizers = (
            _recorder([], "left", LEFT_FIRST),
            _recorder([], "right", RIGHT_FIRST),
            _CountingOptimizer(OUTER_FIRST),
        )
        for optimizer in optimizers:
            got = _quiet(lambda: contract(
                EXPR, *operands, optimize=optimizer, specialize=False))
            np.testing.assert_allclose(
                np.asarray(got), expected, rtol=1e-12, atol=1e-12)
            assert np.asarray(got).dtype == np.dtype(dtype), (
                np.dtype(dtype).name, np.asarray(got).dtype)


# ── 7. 身份不会被回收后的 id 复用 ───────────────────────────────────────────

def test_cached_identity_cannot_be_recycled():
    """计划缓存键持优化器本体：对象不被回收，其 id 不可能指向另一个优化器。"""
    operands = _operands(seed=7)
    calls: list[str] = []
    first = _recorder(calls, "A", RIGHT_FIRST)
    with _plan_cache_isolation() as cache:
        _quiet(lambda: contract(EXPR, *operands, optimize=first))
        keys = list(cache)
        assert len(keys) == 1, keys
        token = keys[0][5]
        assert isinstance(token, _api._OptimizeIdentity), token
        assert token.optimizer is first, "计划缓存键没有持有优化器本体"
        del first
        gc.collect()
        assert token.optimizer is not None, "键里的优化器被回收，id 可能被复用"
        assert id(token.optimizer) == token.token, (id(token.optimizer), token.token)
        second = _recorder(calls, "B", LEFT_FIRST)
        _quiet(lambda: contract(EXPR, *operands, optimize=second))
        assert calls == ["A", "B"], f"新优化器命中了别人留下的计划: {calls}"
        assert len(cache) == 2, f"新优化器必须占用自己的条目: {list(cache)}"


# ── 8. token 语义 ──────────────────────────────────────────────────────────

def test_plan_cache_optimize_token_semantics():
    """不可变参数原样透传；callable 按对象身份相等；未知类型不缓存。"""
    assert _api._plan_cache_optimize_key("dp", "dp") == "dp"
    assert _api._plan_cache_optimize_key(True, True) is True
    assert _api._plan_cache_optimize_key([(0, 1)], ((0, 1),)) == ((0, 1),)
    first = _recorder([], "A", LEFT_FIRST)
    second = _recorder([], "B", LEFT_FIRST)
    first_token = _api._plan_cache_optimize_key(first, None)
    same_again = _api._plan_cache_optimize_key(first, None)
    other_token = _api._plan_cache_optimize_key(second, None)
    assert first_token == same_again, "同一 optimizer 的 token 必须相等（否则命中不了）"
    assert hash(first_token) == hash(same_again)
    assert first_token != other_token, "不同 callable 的 token 不得相等"
    assert _api._plan_cache_optimize_key(object(), None) is None, \
        "未知的 optimize 类型必须保持不缓存"


TESTS = (
    test_two_distinct_callables_are_both_invoked,
    test_same_callable_reuses_the_cached_plan,
    test_callable_path_is_actually_adopted,
    test_path_optimizer_instances_are_not_ignored,
    test_builtin_optimizer_caching_is_unchanged,
    test_callable_results_match_the_off_path,
    test_cached_identity_cannot_be_recycled,
    test_plan_cache_optimize_token_semantics,
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
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
