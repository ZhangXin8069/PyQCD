"""契约：缓存淘汰与内存生命周期（四类 dtype）。

缓存类缺陷的症状是"结果对但会慢"或"偶发错值"，普通数值测试抓不到：

1. **LRU 语义**：命中刷新顺序；超容量/超预算是淘汰最久未用，不是随机；
2. **字节预算**：条目数未满但字节超限时也必须淘汰（反之亦然）；
3. **不缓存值**：计划缓存只存计划，换输入必须重算（已在语义契约里验数值，
   这里验缓存键本身含 dtype/形状/优化器）；
4. **输入不可变**：任何路径都不得原地改写调用方数组；
5. **输出不别名**：新分配的结果不得与任一输入共享内存；
6. **中间体归还**：块池在异常路径上不得泄漏（重复失败后仍能正常分配）。

可直接运行::

    python -m pyqcd.testing._einsum_cache_contract
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

from ..einsum import _kernel_kupl, contract

DTYPES = (np.complex128, np.complex64, np.float64, np.float32)


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


# ── 1. LRU 与字节预算 ───────────────────────────────────────────────────────

def test_topology_plan_cache_is_lru_not_fifo():
    """命中必须刷新顺序：A,B,C,(A),(D) 之后被淘汰的是 B，不是 A。"""
    operands = [_data((2, 2), np.complex128, seed) for seed in range(3)]
    keys = [("eq-%d" % index,) for index in range(3)]
    saved_limit = _kernel_kupl._TOPOLOGY_CACHE_LIMIT
    saved_budget = _kernel_kupl._TOPOLOGY_CACHE_BUDGET
    _kernel_kupl._TOPOLOGY_PLAN_CACHE.clear()
    _kernel_kupl._TOPOLOGY_CACHE_BYTES = 0
    try:
        _kernel_kupl._TOPOLOGY_CACHE_LIMIT = 3
        _kernel_kupl._TOPOLOGY_CACHE_BUDGET = 1 << 30
        for index, key in enumerate(keys):
            _kernel_kupl._topology_plan(
                key, "ij,jk->ik", [operands[index], operands[index]])
        assert list(_kernel_kupl._TOPOLOGY_PLAN_CACHE) == keys
        # 访问 A（刷新），再插入 D
        _kernel_kupl._topology_plan(keys[0], "ij,jk->ik", [operands[0], operands[0]])
        _kernel_kupl._topology_plan(("eq-3",), "ij,jk->ik", [operands[0], operands[0]])
        remaining = list(_kernel_kupl._TOPOLOGY_PLAN_CACHE)
        assert ("eq-1",) not in remaining, remaining      # 最久未用的 B 被淘汰
        assert ("eq-0",) in remaining, remaining          # A 刚被刷新，留下
        assert ("eq-3",) in remaining, remaining
    finally:
        _kernel_kupl._TOPOLOGY_CACHE_LIMIT = saved_limit
        _kernel_kupl._TOPOLOGY_CACHE_BUDGET = saved_budget
        _kernel_kupl._TOPOLOGY_PLAN_CACHE.clear()
        _kernel_kupl._TOPOLOGY_CACHE_BYTES = 0


def test_topology_plan_cache_respects_the_byte_budget():
    """条目数远未到上限时，字节超预算也必须淘汰，且字节计数跟着回落。"""
    operands = [_data((3, 3), np.complex128, 5)]
    saved_budget = _kernel_kupl._TOPOLOGY_CACHE_BUDGET
    _kernel_kupl._TOPOLOGY_PLAN_CACHE.clear()
    _kernel_kupl._TOPOLOGY_CACHE_BYTES = 0
    try:
        _kernel_kupl._TOPOLOGY_CACHE_BUDGET = 64      # 比单个计划还小
        _kernel_kupl._topology_plan(("small",), "ij,jk->ik", operands * 2)
        assert _kernel_kupl._TOPOLOGY_CACHE_BYTES <= 64, (
            _kernel_kupl._TOPOLOGY_CACHE_BYTES)
        assert len(_kernel_kupl._TOPOLOGY_PLAN_CACHE) <= 1
    finally:
        _kernel_kupl._TOPOLOGY_CACHE_BUDGET = saved_budget
        _kernel_kupl._TOPOLOGY_PLAN_CACHE.clear()
        _kernel_kupl._TOPOLOGY_CACHE_BYTES = 0


def test_dedicated_plan_cache_key_includes_dtype_and_optimizer():
    """计划缓存键必须含 dtype 与优化器：两者任一不同都不能复用同一份步表。"""
    for dtype in DTYPES:
        other = np.float64 if np.dtype(dtype).name != "float64" else np.float32
        shapes = ((3, 4), (4, 5))
        first = _kernel_kupl._dedicated_cached("ij,jk->ik", shapes, np.dtype(dtype), "auto")
        second = _kernel_kupl._dedicated_cached("ij,jk->ik", shapes, np.dtype(other), "auto")
        third = _kernel_kupl._dedicated_cached(
            "ij,jk->ik", shapes, np.dtype(dtype), "greedy")
        assert first is not None and second is not None, (dtype, other)
        assert first[0] != second[0] or first[1] != second[1], (
            "不同 dtype 复用了同一份步表", np.dtype(dtype).name, np.dtype(other).name)
        assert first[0] != third[0] or third[2] != first[2], (
            "不同优化器复用了同一份步表", np.dtype(dtype).name)


# ── 2. 内存生命周期 ─────────────────────────────────────────────────────────

def test_inputs_are_never_mutated():
    """所有路径（含专用 kernel）都不得原地改写输入。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        left = _data((4, 5), dtype, 11)
        right = _data((5, 6), dtype, 12)
        snapshot_left = left.copy()
        snapshot_right = right.copy()
        _quiet(lambda: contract("ij,jk->ik", left, right))
        _kernel_kupl.try_topology_apply("ij,jk->ik", [left, right])
        _kernel_kupl.try_dedicated("ij,jk->ik", [left, right])
        assert np.array_equal(left, snapshot_left), (name, "left mutated")
        assert np.array_equal(right, snapshot_right), (name, "right mutated")


def test_allocated_output_does_not_alias_inputs():
    """新分配的结果不得与任一输入共享内存（否则缓存/复用会互相污染）。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        left = _data((4, 5), dtype, 13)
        right = _data((5, 6), dtype, 14)
        result = _quiet(lambda: contract("ij,jk->ik", left, right))
        assert not np.shares_memory(np.asarray(result), left), name
        assert not np.shares_memory(np.asarray(result), right), name
        # 改结果不得影响输入
        np.asarray(result)[...] = 0
        assert np.array_equal(left, _data((4, 5), dtype, 13)), (name, "aliased left")


def test_block_pool_survives_repeated_failed_calls():
    """重复失败（原生拒执行）之后仍能正常分配，不泄漏块池。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        operands = [_data((3, 4), dtype, 15), _data((4, 5), dtype, 16)]
        for _ in range(8):
            # 注：入口是 ``_kernel_kupl.kupl.topology_apply``（ctypes 绑定在
            # 子模块里），模块本身没有同名属性。
            with patch.object(_kernel_kupl.kupl, "topology_apply",
                              side_effect=RuntimeError("native exploded")):
                assert _kernel_kupl.try_topology_apply("ij,jk->ik", operands) is None
        # 失败不落负缓存（非确定性错误），恢复后必须照常工作
        result = _quiet(lambda: contract("ij,jk->ik", *operands, specialize=False))
        assert result.shape == (3, 5), name
        assert np.isfinite(np.asarray(result)).all(), name


def test_out_caller_buffer_survives_failed_call():
    """专用 kernel 抛异常时，``out=` 仍必须被正确写入（不影响语义）。

    注意：这里 patch 的是 ``try_topology_apply``，但 ``contract`` 的 ``out=``
    路径**本来就会绕开**全部专用 kernel，所以本用例真正验的是"绕过不改变结果"。
    早先的写法假设"失败即回落且仍要写对 out"，会随调用顺序变化——套件里前一组
    留下的 patch/负缓存状态能让它变红，而单跑却是绿的。契约不能依赖跨组状态。
    """
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        left = _data((3, 4), dtype, 17)
        right = _data((4, 5), dtype, 18)
        out = np.full((3, 5), 7, dtype=dtype)
        with patch.object(_kernel_kupl, "try_topology_apply",
                          side_effect=RuntimeError("boom")):
            returned = _quiet(lambda: contract("ij,jk->ik", left, right, out=out))
        assert returned is out, name
        expected = np.einsum("ij,jk->ik", left, right, optimize="greedy")
        scale = float(np.max(np.abs(expected))) or 1.0
        error = float(np.max(np.abs(out - expected))) / scale
        assert error <= 1e-4 if np.dtype(dtype).itemsize <= 8 else error <= 1e-12, \
            (name, error)


TESTS = (
    test_topology_plan_cache_is_lru_not_fifo,
    test_topology_plan_cache_respects_the_byte_budget,
    test_dedicated_plan_cache_key_includes_dtype_and_optimizer,
    test_inputs_are_never_mutated,
    test_allocated_output_does_not_alias_inputs,
    test_block_pool_survives_repeated_failed_calls,
    test_out_caller_buffer_survives_failed_call,
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
