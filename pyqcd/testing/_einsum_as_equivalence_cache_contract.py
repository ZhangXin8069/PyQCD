"""契约：等价映射自动施加与 einsum 缓存开关的集成行为。

本轮新增的 ``as_equivalence``、``cache`` 与 ``kernel_force`` 都可能在结果正确时
仍然选错路径：显式参数没有压过环境、缓存关闭后仍被读写、或强制内核错误地覆盖
自动分派。本模块只断言这些可观察语义，不做计时：

1. ``as_equivalence`` 的 ``None``/``True``/``False`` 三态与 spy 调用证据；
2. 操作数顺序、输出轴序及二者复合映射返回的仍是原方程答案；
3. ``contract(..., cache=False)`` 不改变任一缓存统计字段；
4. ``specialize.match`` / ``equivalence.normalize`` 的负结果缓存命中；
5. ``contract_path(..., cache=...)`` 的规划缓存与统一清空语义；
6. 环境强制 CUDA 时，显式 ``kernel_force='auto'`` 仍走自动分派；
7. ``kernel_force='generic'`` 与 ``out=`` / ``specialize=False`` 的非冲突组合。

可直接运行::

    python -B -m pyqcd.testing._einsum_as_equivalence_cache_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import unittest
import warnings
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

import pyqcd.einsum as pe
from ..einsum import _api, _cache, _equivalence, _kernel_force, _specialize


# 同一语料族的三种查询：只换操作数位置、只换输出轴序、两者同时发生。
CORPUS_EQUATION = "cadb,menf,ac,em,Mbd,Nfn->MN"
OPERAND_ORDER_EQUATION = "menf,cadb,ac,em,Mbd,Nfn->MN"
OUTPUT_AXES_EQUATION = "cadb,menf,ac,em,Mbd,Nfn->NM"
COMPOSITE_EQUATION = "menf,cadb,ac,em,Mbd,Nfn->NM"

# 负结果测试分别使用两条语料外方程，避免共享一个缓存键。
SPECIALIZE_MISS_EQUATION = "ij,jk->i"
EQUIVALENCE_MISS_EQUATION = "ii->i"

CACHE_EQUATION = "ij,ij->ij"
PATH_EQUATION = "ij,jk->ik"
CACHE_NAMES = ("specialize.match", "equivalence.normalize", "api.find_path")
REQUIRED_CACHE_FIELDS = {
    "size", "maxsize", "hits", "misses", "stores", "evictions", "hit_rate",
}

LABEL_SIZES = {
    "i": 2, "j": 3, "k": 4,
    "c": 2, "a": 3, "d": 4, "b": 5,
    "m": 6, "e": 7, "n": 2, "f": 3,
    "M": 2, "N": 3,
}


@contextlib.contextmanager
def _isolated_state() -> Iterator[None]:
    """恢复本模块会触碰的环境与最近内核记录，并清空统一缓存。"""
    env_names = (
        _cache.CACHE_ENV,
        _equivalence.AS_EQUIVALENCE_ENV,
        _equivalence.EQUIVALENCE_ENV,
        _kernel_force.KERNEL_FORCE_ENV,
        _specialize.DISABLE_ENV,
        _specialize.NOTICE_ENV,
        _specialize.FUSED_DISABLE_ENV,
    )
    saved_env = {name: os.environ.get(name) for name in env_names}
    saved_last_kernel = _api._LAST_KERNEL
    saved_fused_owned = _api._FUSED_ENV_OWNED
    pe.cache_clear()
    try:
        yield
    finally:
        _api._LAST_KERNEL = saved_last_kernel
        _api._FUSED_ENV_OWNED = saved_fused_owned
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        pe.cache_clear()


def _prepare_contract_environment() -> None:
    """打开本模块依赖的开关；各用例外层仍由 ``_isolated_state`` 负责恢复。"""
    os.environ[_cache.CACHE_ENV] = "1"
    os.environ[_equivalence.EQUIVALENCE_ENV] = "1"
    os.environ[_specialize.DISABLE_ENV] = "1"
    os.environ[_specialize.NOTICE_ENV] = "0"
    os.environ.pop(_kernel_force.KERNEL_FORCE_ENV, None)


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """屏蔽本契约不关心的提示与原生回落 warning。"""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


def _operands(equation: str, *, seed: int = 20261116) -> list[np.ndarray]:
    """按方程左侧生成标签长度互不相同的 complex128 操作数。"""
    rng = np.random.default_rng(seed)
    result = []
    for term in equation.split("->", 1)[0].split(","):
        shape = tuple(LABEL_SIZES[label] for label in term)
        value = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        result.append(np.ascontiguousarray(value.astype(np.complex128)))
    return result


def _relative_error(actual: Any, expected: Any) -> float:
    """相对最大偏差；全零答案用 1 作尺度。"""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    scale = float(np.max(np.abs(expected_array))) or 1.0
    return float(np.max(np.abs(actual_array - expected_array))) / scale


def _assert_original_answer(actual: Any, expected: Any, label: str) -> None:
    """分别断言原方程答案的形状、dtype 与 rel <= 1e-12。"""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape, (
        label, actual_array.shape, expected_array.shape)
    assert actual_array.dtype == expected_array.dtype, (
        label, actual_array.dtype, expected_array.dtype)
    error = _relative_error(actual_array, expected_array)
    assert error <= 1e-12, (label, error)


def _cache_snapshot() -> dict[str, dict[str, Any]]:
    """复制全部缓存字段，避免后续调用原地改变快照。"""
    return {name: dict(info) for name, info in pe.cache_stats()["caches"].items()}


@contextlib.contextmanager
def _normalize_spy() -> Iterator[list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    """包住真实 ``normalize_equation``，记录它是否被自动施加路径调用。"""
    original = _equivalence.normalize_equation
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    with patch.object(_equivalence, "normalize_equation", new=spy):
        yield calls


def test_as_equivalence_three_state_controls_the_normalization_path():
    """``None`` 跟随环境，显式 True/False 分别压过环境 0/1，并由 spy 计数。"""
    with _isolated_state():
        _prepare_contract_environment()
        operands = _operands(OPERAND_ORDER_EQUATION)
        expected = np.einsum(OPERAND_ORDER_EQUATION, *operands)
        cases = (
            ("env=0 follows", None, "0", 0),
            ("env=1 follows", None, "1", 1),
            ("explicit True overrides env=0", True, "0", 1),
            ("explicit False overrides env=1", False, "1", 0),
        )

        for label, value, env_value, expected_calls in cases:
            os.environ[_equivalence.AS_EQUIVALENCE_ENV] = env_value
            pe.cache_clear()
            with _normalize_spy() as calls, _quiet():
                actual = pe.contract(
                    OPERAND_ORDER_EQUATION,
                    *operands,
                    as_equivalence=value,
                    cache=True,
                )
            assert len(calls) == expected_calls, (label, len(calls), calls)
            if expected_calls:
                assert calls[0][0][0] == OPERAND_ORDER_EQUATION, label
            _assert_original_answer(actual, expected, label)


def test_equivalence_only_returns_the_answer_to_the_original_equation():
    """三条映射分别只含顺序、输出轴序及二者复合，均与原方程 numpy 答案一致。"""
    with _isolated_state():
        _prepare_contract_environment()
        os.environ[_equivalence.AS_EQUIVALENCE_ENV] = "1"
        cases = (
            (OPERAND_ORDER_EQUATION, "operand-order"),
            (OUTPUT_AXES_EQUATION, "output-axes"),
            (COMPOSITE_EQUATION, "operand-order+output-axes"),
        )

        for index, (equation, expected_kind) in enumerate(cases):
            mapping = _equivalence.normalize_equation(equation, use=False)
            assert mapping is not None, equation
            assert mapping.kind == expected_kind, (equation, mapping.kind)
            operands = _operands(equation, seed=20261116 + index)
            expected = np.einsum(equation, *operands)
            with _quiet():
                actual = pe.contract(
                    equation, *operands, as_equivalence=True, cache=True)
            _assert_original_answer(actual, expected, expected_kind)


def test_contract_cache_false_leaves_every_cache_field_unchanged():
    """预热后以 ``cache=False`` 重跑同一方程，所有缓存字段逐项完全相等。"""
    with _isolated_state():
        _prepare_contract_environment()
        left = _operands(CACHE_EQUATION)[0]
        right = _operands(CACHE_EQUATION, seed=20261117)[0]
        pe.cache_clear()

        with _quiet():
            pe.contract(CACHE_EQUATION, left, right, cache=True)
        before = _cache_snapshot()
        assert before["specialize.match"]["size"] == 1, before

        for info in before.values():
            assert set(info) == REQUIRED_CACHE_FIELDS, info

        with _quiet():
            pe.contract(CACHE_EQUATION, left, right, cache=False)
        after = _cache_snapshot()
        assert after == before, (before, after)


def test_negative_results_are_cached_with_miss_then_hit():
    """两个匹配器的负结果都遵循 miss+store、再 hit，且 size 不增长。"""
    with _isolated_state():
        _prepare_contract_environment()
        cases = (
            ("specialize.match", _specialize.match, SPECIALIZE_MISS_EQUATION),
            ("equivalence.normalize", _equivalence.normalize_equation,
             EQUIVALENCE_MISS_EQUATION),
        )

        for cache_name, lookup, equation in cases:
            pe.cache_clear()
            first = lookup(equation, use=True)
            first_info = dict(pe.cache_stats()["caches"][cache_name])
            second = lookup(equation, use=True)
            second_info = dict(pe.cache_stats()["caches"][cache_name])

            assert first is None and second is None, (cache_name, first, second)
            assert first_info["misses"] == 1, (cache_name, first_info)
            assert first_info["stores"] == 1, (cache_name, first_info)
            assert first_info["hits"] == 0, (cache_name, first_info)
            assert second_info["misses"] == 1, (cache_name, second_info)
            assert second_info["stores"] == 1, (cache_name, second_info)
            assert second_info["hits"] == 1, (cache_name, second_info)
            assert second_info["size"] == first_info["size"] == 1, (
                cache_name, first_info, second_info)


def test_contract_path_cache_switch_and_global_clear_semantics():
    """``cache=True`` 第二次命中；``False`` 零变化；清空后三缓存归零。"""
    with _isolated_state():
        _prepare_contract_environment()
        shapes = ((2, 3), (3, 4))
        pe.cache_clear()

        first_path, _ = pe.contract_path(
            PATH_EQUATION, *shapes, shapes=True, optimize="greedy", cache=True)
        before_hit = dict(pe.cache_stats()["caches"]["api.find_path"])
        second_path, _ = pe.contract_path(
            PATH_EQUATION, *shapes, shapes=True, optimize="greedy", cache=True)
        after_hit = dict(pe.cache_stats()["caches"]["api.find_path"])
        assert second_path == first_path, (first_path, second_path)
        assert after_hit["hits"] == before_hit["hits"] + 1, (
            before_hit, after_hit)

        before_false = _cache_snapshot()
        pe.contract_path(
            PATH_EQUATION, *shapes, shapes=True, optimize="greedy", cache=False)
        after_false = _cache_snapshot()
        assert after_false == before_false, (before_false, after_false)

        _specialize.match(SPECIALIZE_MISS_EQUATION, use=True)
        _equivalence.normalize_equation(EQUIVALENCE_MISS_EQUATION, use=True)
        before_clear = pe.cache_stats()
        enabled_before = before_clear["enabled"]
        assert enabled_before is True, before_clear
        for name in CACHE_NAMES:
            assert before_clear["caches"][name]["size"] > 0, (name, before_clear)

        pe.cache_clear()
        cleared = pe.cache_stats()
        assert cleared["enabled"] == enabled_before, (enabled_before, cleared)
        for name in CACHE_NAMES:
            info = cleared["caches"][name]
            assert info["size"] == 0, (name, info)
            assert info["hits"] == 0, (name, info)
            assert info["misses"] == 0, (name, info)
            assert info["stores"] == 0, (name, info)
            assert info["evictions"] == 0, (name, info)
            assert info["hit_rate"] is None, (name, info)


def test_explicit_kernel_force_auto_overrides_cuda_environment():
    """环境写 ``cuda_topology`` 时，显式 auto 仍走 numpy 后端自动分派。"""
    with _isolated_state():
        _prepare_contract_environment()
        os.environ[_kernel_force.KERNEL_FORCE_ENV] = "cuda_topology"
        operands = _operands(CORPUS_EQUATION)
        expected = np.einsum(CORPUS_EQUATION, *operands)

        try:
            with _quiet():
                actual = pe.contract(
                    CORPUS_EQUATION, *operands, kernel_force="auto")
        except _kernel_force.KernelUnavailableError as exc:
            raise AssertionError(
                "显式 kernel_force='auto' 不得继承 cuda_topology 环境强制") from exc

        record = _api.last_kernel()
        assert record is not None, record
        assert record["requested"] is None, record
        assert all(label.startswith("kupl_") for label in record["attempted"]), record
        _assert_original_answer(actual, expected, "kernel_force=auto")


def test_generic_kernel_force_composes_with_out_and_specialize_false():
    """generic 与 ``out=`` / ``specialize=False`` 均不冲突，数值与 numpy 一致。"""
    with _isolated_state():
        _prepare_contract_environment()
        rng = np.random.default_rng(20261118)
        left = np.ascontiguousarray(
            rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4)))
        right = np.ascontiguousarray(
            rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5)))
        expected = np.einsum("ij,jk->ik", left, right)

        out = np.empty_like(expected)
        with _quiet():
            returned = pe.contract(
                "ij,jk->ik", left, right, out=out, kernel_force="generic")
        assert returned is out, (type(returned), type(out))
        _assert_original_answer(returned, expected, "generic with out")

        with _quiet():
            no_specialize = pe.contract(
                "ij,jk->ik", left, right,
                kernel_force="generic", specialize=False)
        _assert_original_answer(no_specialize, expected, "generic without specialize")


TESTS = (
    test_as_equivalence_three_state_controls_the_normalization_path,
    test_equivalence_only_returns_the_answer_to_the_original_equation,
    test_contract_cache_false_leaves_every_cache_field_unchanged,
    test_negative_results_are_cached_with_miss_then_hit,
    test_contract_path_cache_switch_and_global_clear_semantics,
    test_explicit_kernel_force_auto_overrides_cuda_environment,
    test_generic_kernel_force_composes_with_out_and_specialize_false,
)


def test_einsum_as_equivalence_cache_contracts() -> None:
    """注册入口：逐条执行本模块契约；全跳过时也显式 SkipTest。"""
    passed = skipped = 0
    failures = []
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    if failures:
        raise AssertionError(
            f"{len(failures)} 条等价映射/缓存契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"as_equivalence/cache contracts did not run ({skipped} skipped)")


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
