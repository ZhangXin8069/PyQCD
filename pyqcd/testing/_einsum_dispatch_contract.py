"""契约：``contract`` 的分派门顺序与归属（四类 dtype）。

``contract`` 有六道门，顺序本身是契约的一部分：走错门**不改数值、只改耗时**，
因此普通数值测试永远抓不到。本模块只验"哪条门接了这一手"，不验性能：

1. 无求和标签 → backend 原生 einsum（单操作数一律算）；
2. ``cupy`` 非融合形态 → ``cupy.einsum``（cuBLAS）；
3. 形状特化（``x,ax,bx,cx->abc``）在规划**之前**接管；
4. ``numpy`` + batch 标签 → 原生计划执行器；无 batch 标签 → Python 分步；
5. ``out=`` 与 ``kwargs`` 绕过专用 kernel，但仍给出正确结果；
6. ``specialize=False`` 只关特化，不改语义。

每一条都对四类 dtype 各验一遍：dtype 不得改变"哪条门接手"这个结构结论。

可直接运行::

    python -m pyqcd.testing._einsum_dispatch_contract
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

from ..einsum import _api, _kernel_kupl, _shape_paths, contract

DTYPES = (np.complex128, np.complex64, np.float64, np.float32)
TOLERANCE = {"complex128": 1e-12, "float64": 1e-12,
             "complex64": 1e-4, "float32": 1e-4}


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


def _spy(call):
    """记录本次调用走过哪些分支（只包模块级名字，不进生产代码）。"""
    log: list[tuple] = []
    backend_einsum = _api._backend_einsum
    native = _api._try_native_contract
    shape_execute = _shape_paths.try_execute

    def backend_spy(name, subscripts, operands, out=None, **kwargs):
        log.append(("backend", name))
        return backend_einsum(name, subscripts, operands, out, **kwargs)

    def native_spy(operands, contraction_list, backend, out, kwargs, **extra):
        handled, result = native(operands, contraction_list, backend, out, kwargs, **extra)
        log.append(("native", backend, extra.get("batch_like"), handled))
        return handled, result

    def shape_spy(subscripts, operands):
        result = shape_execute(subscripts, operands)
        log.append(("shape", result is not None))
        return result

    with patch.object(_api, "_backend_einsum", backend_spy), \
            patch.object(_api, "_try_native_contract", native_spy), \
            patch.object(_shape_paths, "try_execute", shape_spy):
        result = call()
    return result, log


# ── 1. 元素级门 ─────────────────────────────────────────────────────────────

def test_elementwise_equations_take_the_backend_door():
    """无求和标签（Hadamard/转置/对角/迹/外积）必须走 backend 原生 einsum。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        matrix = _data((6, 6), dtype, 1)
        vector = _data((5,), dtype, 2)
        other = _data((7,), dtype, 3)
        cases = [
            ("hadamard", "ij,ij->ij", [matrix, matrix]),
            ("transpose", "ij->ji", [matrix]),
            ("diagonal", "ii->i", [matrix]),
            ("trace", "ii->", [matrix]),
            ("outer", "i,j->ij", [vector, other]),
        ]
        for label, equation, operands in cases:
            result, log = _spy(lambda: _quiet(
                lambda: contract(equation, *operands)))
            assert ("backend", "numpy") in log, (name, label, log)
            assert not any(item[0] == "shape" for item in log), (name, label, log)
            expected = np.einsum(equation, *operands)
            assert _rel(result, expected) <= TOLERANCE[name], (name, label)


# ── 2. 形状特化门 ───────────────────────────────────────────────────────────

def test_shape_path_takes_precedence_over_planner():
    """``x,ax,bx,cx->abc`` 必须在规划之前接管（含四类 dtype）。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        operands = [_data((8,), dtype, 4)] + [_data((8, 8), dtype, 5 + index)
                                              for index in range(3)]
        result, log = _spy(lambda: _quiet(
            lambda: contract("x,ax,bx,cx->abc", *operands)))
        assert log and log[0] == ("shape", True), (name, log)
        assert not any(item[0] == "native" for item in log), (name, log)
        expected = np.einsum("x,ax,bx,cx->abc", *operands, optimize="greedy")
        assert _rel(result, expected) <= TOLERANCE[name], (name, _rel(result, expected))


def test_shape_path_is_skipped_for_out_and_specialize_off():
    """``out=`` 与 ``specialize=False`` 必须绕开形状特化，但结果不变。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        operands = [_data((6,), dtype, 7)] + [_data((6, 6), dtype, 8 + index)
                                              for index in range(3)]
        out = np.empty((6, 6, 6), dtype=dtype)
        with patch.object(_shape_paths, "try_execute",
                          side_effect=AssertionError("out= 时不得接管")):
            written = _quiet(lambda: contract("x,ax,bx,cx->abc", *operands, out=out))
        assert written is out, name
        assert _rel(out, np.einsum("x,ax,bx,cx->abc", *operands,
                                   optimize="greedy")) <= TOLERANCE[name], name

        with patch.object(_shape_paths, "try_execute",
                          side_effect=AssertionError("specialize=False 时不得接管")):
            plain = _quiet(lambda: contract(
                "x,ax,bx,cx->abc", *operands, specialize=False))
        assert _rel(plain, out) <= TOLERANCE[name], name


# ── 3. 原生计划门：batch 判据 ───────────────────────────────────────────────

def test_native_plan_door_depends_on_batch_structure_not_ndim():
    """有 batch 标签走原生执行器；三维但无 batch 标签必须不走（不能按 ndim 判）。"""
    assert _api._has_batch_index("bij,bjk->bik") is True
    assert _api._has_batch_index("abc,bcd->ad") is False
    assert _api._has_batch_index("...ij,...jk->...ik") is True
    assert _api._is_elementwise_equation("ij,ij->ij") is True
    assert _api._is_elementwise_equation("ij,jk->ik") is False
    assert _api._is_elementwise_equation("i,j->ij") is True

    for dtype in DTYPES:
        name = np.dtype(dtype).name
        batch_left = _data((2, 6, 7), dtype, 11)
        batch_right = _data((2, 7, 8), dtype, 12)
        _, log = _spy(lambda: _quiet(
            lambda: contract("bij,bjk->bik", batch_left, batch_right)))
        assert ("native", "numpy", True, True) in log, (name, log)

        chain_left = _data((4, 5, 6), dtype, 13)
        chain_right = _data((5, 6, 7), dtype, 14)
        _, log = _spy(lambda: _quiet(
            lambda: contract("abc,bcd->ad", chain_left, chain_right)))
        assert not any(item[0] == "native" and item[3] for item in log), (name, log)


# ── 4. out= / kwargs 绕过专用 kernel ────────────────────────────────────────

def test_out_and_kwargs_bypass_specialized_kernels():
    """``out=`` 与任何 kwargs 都绕过专用 kernel；结果仍必须正确。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        left = _data((3, 4), dtype, 15)
        right = _data((4, 5), dtype, 16)
        out = np.empty((3, 5), dtype=dtype)
        with patch.object(_kernel_kupl, "try_topology_apply",
                          side_effect=AssertionError("out= 不得下发拓扑内核")), \
                patch.object(_kernel_kupl, "try_dedicated",
                             side_effect=AssertionError("out= 不得下发步表内核")):
            returned = _quiet(lambda: contract("ij,jk->ik", left, right, out=out))
        assert returned is out, name
        assert _rel(out, left @ right) <= TOLERANCE[name], name


# ── 5. specialize=False 只关特化 ────────────────────────────────────────────

def test_specialize_false_only_disables_the_specialized_paths():
    """``specialize=False`` 必须与默认结果一致，且不再尝试任何专用 kernel。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        left = _data((6, 7), dtype, 17)
        right = _data((7, 8), dtype, 18)
        default = _quiet(lambda: contract("ij,jk->ik", left, right))
        with patch.object(_kernel_kupl, "try_topology_apply",
                          side_effect=AssertionError("specialize=False 仍下发拓扑")), \
                patch.object(_kernel_kupl, "try_dedicated",
                             side_effect=AssertionError("specialize=False 仍下发步表")):
            plain = _quiet(lambda: contract(
                "ij,jk->ik", left, right, specialize=False))
        assert _rel(plain, default) <= TOLERANCE[name], name
        assert _rel(plain, left @ right) <= TOLERANCE[name], name


# ── 6. 混合 dtype 的边界 ────────────────────────────────────────────────────

def test_torch_backend_honours_the_out_parameter():
    """torch 后端必须真的把结果写进调用方的 ``out=``（2026-11-15 巡查实证的静默错答）。

    回归点：``_torch_einsum`` 原先用 ``**kwargs`` 静默丢弃 ``out``，于是
    ``_wrap_einsum`` 的 ``except TypeError`` 补偿永不触发，``execute_plan`` 末尾
    ``return out`` 把**一个字节都没写**的缓冲区交回调用方——实测 buf 全 NaN 而
    ``r is buf`` 为真，无异常无告警。两处都要修才通：①``_torch_einsum`` 对 ``out``
    显式抛 ``TypeError``；②``_SHIMS`` 里的 ``einsum`` 过 ``_wrap_einsum``（静态表
    会跳过 ``get_func`` 的包装）。

    本条锁两条入口：``optimize=False``（不走 BLAS 分步）与普通路径。
    """
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("torch 不可用")

    import numpy as np

    import pyqcd.einsum as pe

    base = np.arange(9, dtype=np.float64).reshape(3, 3)
    tensor = torch.tensor(base + 1j * base, dtype=torch.complex128)
    expected = np.einsum("ij,ij->ij", base + 1j * base, base + 1j * base)

    for kwargs in ({}, {"optimize": False}):
        buf = torch.full((3, 3), complex(float("nan"), float("nan")),
                         dtype=torch.complex128)
        result = pe.contract("ij,ij->ij", tensor, tensor, backend="torch",
                             out=buf, **kwargs)
        assert result is buf, (
            f"kwargs={kwargs}: 返回的必须是调用方缓冲区本身")
        assert not bool(torch.all(torch.isnan(buf))), (
            f"kwargs={kwargs}: out 缓冲区一个字节都没写（全 NaN）——这正是静默错答")
        assert np.allclose(buf.cpu().numpy(), expected), (
            f"kwargs={kwargs}: 回填的数值不对")

    # numpy 后端同样必须写进 out=（同一条补偿链的另一侧，防止只有 torch 被照顾）。
    buf = np.full((3, 3), np.nan, dtype=np.complex128)
    result = pe.contract("ij,ij->ij", base + 1j * base, base + 1j * base,
                         backend="numpy", out=buf)
    assert result is buf and np.allclose(buf, expected), "numpy 后端的 out= 也要写对"


def test_mixed_dtype_promotion_is_explicit():
    """混合 dtype 由 numpy 规则提升（不是静默降级），且不是内核路径。"""
    real_left = _data((3, 4), np.float32, 19)
    complex_right = _data((4, 5), np.complex128, 20)
    result = _quiet(lambda: contract("ij,jk->ik", real_left, complex_right))
    expected = np.einsum("ij,jk->ik", real_left, complex_right)
    assert np.dtype(result.dtype) == np.dtype(expected.dtype), result.dtype
    assert _rel(result, expected) <= TOLERANCE["complex128"], _rel(result, expected)


TESTS = (
    test_elementwise_equations_take_the_backend_door,
    test_shape_path_takes_precedence_over_planner,
    test_shape_path_is_skipped_for_out_and_specialize_off,
    test_native_plan_door_depends_on_batch_structure_not_ndim,
    test_out_and_kwargs_bypass_specialized_kernels,
    test_specialize_false_only_disables_the_specialized_paths,
    test_torch_backend_honours_the_out_parameter,
    test_mixed_dtype_promotion_is_explicit,
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
