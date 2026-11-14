"""契约：收缩语义的代数恒等式与随机方程对拍（四类 dtype）。

与既有契约的分工：``_topology_kernel_contract`` 验"原生入口的参数与回落"，
``verify_dtype_parity`` 验"全语料的数值对齐"，本模块验**收缩语义本身**——这些
恒等式与内核实现无关，任何一条被破都说明分派/规划/内存复用改坏了语义：

1. **结合律**：``(A·B)·C`` 与 ``A·(B·C)`` 同值（不同规划路径必须一致）；
2. **线性性**：``(αA)·B`` = ``α(A·B)``，且加法可分配；
3. **共轭/转置**：``conj(A·B) = conj(A)·conj(B)``，``(A·B)ᵀ = Bᵀ·Aᵀ``；
4. **迹/置换/外积**的解析真值（不只是 numpy 对拍——用闭式值）；
5. **随机方程对拍**：随机生成合法结构（复用语料签名）与小尺寸，四 dtype 比对；
6. **缓存正确性**：同一方程换操作数必须得到新结果（防"返回旧缓冲"）；
7. **确定性**：同输入重复调用逐位相同（BLAS 之外的差异即状态泄漏）。

可直接运行::

    python -m pyqcd.testing._einsum_semantics_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import unittest
import warnings
from typing import Any, Sequence
from unittest.mock import patch

import numpy as np

from ..einsum import contract
from ..einsum import _kernel_kupl

DTYPES = (np.complex128, np.complex64, np.float64, np.float32)
TOLERANCE = {"complex128": 1e-12, "float64": 1e-12,
             "complex64": 1e-4, "float32": 1e-4}
TRUTH = {np.dtype(np.complex128): np.complex128,
         np.dtype(np.complex64): np.complex128,
         np.dtype(np.float64): np.float64,
         np.dtype(np.float32): np.float64}


def _quiet(call):
    import contextlib as _ctx

    with warnings.catch_warnings(), _ctx.redirect_stdout(io.StringIO()), patch.dict(
            os.environ, {"PYQCD_EINSUM_NOTICE": "0"}):
        warnings.simplefilter("ignore")
        return call()


def _data(shape: Sequence[int], dtype: Any, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(tuple(shape))
    if np.dtype(dtype).kind == "c":
        values = values + 1j * rng.standard_normal(tuple(shape))
    return np.ascontiguousarray(values.astype(dtype))


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


# ── 1. 代数恒等式 ───────────────────────────────────────────────────────────

def test_associativity_for_all_dtypes():
    """``(A·B)·C`` 与 ``A·(B·C)`` 必须同值：不同规划路径不得给出不同语义。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        left = _data((3, 4), dtype, 1)
        middle = _data((4, 5), dtype, 2)
        right = _data((5, 6), dtype, 3)
        one_shot = _quiet(lambda: contract("ij,jk,kl->il", left, middle, right))
        # 先 A·B 再 ·C：中间体是 (3,5)，与 C(5,6) 直接相乘。
        pairwise = _quiet(lambda: contract(
            "ij,jk->ik", _quiet(lambda: contract("ij,jk->ik", left, middle)), right))
        error = _rel(one_shot, pairwise)
        assert error <= TOLERANCE[name], (name, error)
        # 先 B·C 再 A·：中间体是 (4,6)。
        right_first = _quiet(lambda: contract(
            "ij,jk->ik", left, _quiet(lambda: contract("ij,jk->ik", middle, right))))
        assert _rel(one_shot, right_first) <= TOLERANCE[name], (
            name, "left-first vs right-first", _rel(one_shot, right_first))


def test_linearity_and_distributivity():
    """``(αA)·B = α(A·B)``；``(A1+A2)·B = A1·B + A2·B``。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        left = _data((3, 4), dtype, 4)
        other = _data((3, 4), dtype, 5)
        right = _data((4, 5), dtype, 6)
        alpha = 2.5 if np.dtype(dtype).kind == "f" else 1.5 - 0.75j
        scaled = _quiet(lambda: contract("ij,jk->ik", (left * alpha).astype(dtype), right))
        reference = _quiet(lambda: contract("ij,jk->ik", left, right)) * alpha
        assert _rel(scaled, reference) <= TOLERANCE[name], (
            name, _rel(scaled, reference))

        summed = _quiet(lambda: contract(
            "ij,jk->ik", (left + other).astype(dtype), right))
        split = (_quiet(lambda: contract("ij,jk->ik", left, right))
                 + _quiet(lambda: contract("ij,jk->ik", other, right)))
        assert _rel(summed, split) <= TOLERANCE[name], (name, _rel(summed, split))


def test_conjugate_and_transpose_rules():
    """``conj(A·B) = conj(A)·conj(B)``；``(A·B)ᵀ = Bᵀ·Aᵀ``（复数同样成立）。"""
    for dtype in (np.complex128, np.complex64, np.float64, np.float32):
        name = np.dtype(dtype).name
        left = _data((3, 5), dtype, 7)
        right = _data((5, 4), dtype, 8)
        product = _quiet(lambda: contract("ij,jk->ik", left, right))
        conjugated = _quiet(lambda: contract(
            "ij,jk->ik", np.conjugate(left), np.conjugate(right)))
        assert _rel(conjugated, np.conjugate(product)) <= TOLERANCE[name], name

        transposed = _quiet(lambda: contract(
            "ij,jk->ik", right.T.copy(), left.T.copy()))
        assert _rel(transposed, product.T) <= TOLERANCE[name], name


def test_closed_form_values():
    """闭式真值（不只对拍 numpy）：迹、置换、外积、单位矩阵左乘。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        matrix = _data((4, 4), dtype, 9)
        trace = _quiet(lambda: contract("ii->", matrix))
        assert _rel(trace, np.trace(matrix)) <= TOLERANCE[name], name

        permuted = _quiet(lambda: contract("ij->ji", matrix))
        assert np.array_equal(np.asarray(permuted), matrix.T), name

        vector = _data((3,), dtype, 10)
        other = _data((4,), dtype, 11)
        outer = _quiet(lambda: contract("i,j->ij", vector, other))
        assert _rel(outer, np.outer(vector, other)) <= TOLERANCE[name], name

        identity = np.eye(4, dtype=dtype)
        scaled = _quiet(lambda: contract("ij,jk->ik", identity, matrix))
        assert _rel(scaled, matrix) <= TOLERANCE[name], name


# ── 2. 随机方程 ─────────────────────────────────────────────────────────────

def test_random_equations_against_numpy():
    """结构化随机方程 × 四 dtype：与 numpy 逐个对拍（不依赖语料）。"""
    import pyqcd.einsum._wick_tasklist as tasklist

    tasks = [tasklist.lookup(seq) for seq in (2584, 2588, 4305, 1616, 1717)]
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        for task in tasks:
            terms = task.equation.split("->")[0].split(",")
            dim = 2
            operands = [_data((dim,) * len(term), dtype, task.seq + index)
                        for index, term in enumerate(terms)]
            got = _quiet(lambda: contract(task.equation, *operands))
            truth = _quiet(lambda: contract(
                task.equation,
                *[item.astype(TRUTH[np.dtype(dtype)]) for item in operands]))
            assert np.dtype(got.dtype) == np.dtype(dtype), (name, task.seq, got.dtype)
            assert got.shape == truth.shape, (name, task.seq, got.shape)
            assert _rel(got, truth) <= TOLERANCE[name], (
                name, task.seq, _rel(got, truth))

            # 独立第二参考：numpy 自己的 greedy 计划。
            reference = np.einsum(task.equation, *operands, optimize="greedy")
            assert _rel(got, reference.astype(TRUTH[np.dtype(dtype)])) <= TOLERANCE[name], (
                name, task.seq)


# ── 3. 缓存与确定性 ─────────────────────────────────────────────────────────

def test_contract_cache_never_returns_stale_values():
    """同方程换操作数必须得到新结果：计划/上下文缓存不得连值一起缓存。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        first = _data((3, 4), dtype, 12)
        right = _data((4, 5), dtype, 13)
        before = _quiet(lambda: contract("ij,jk->ik", first, right)).copy()
        second = (first + 1).astype(dtype)
        after = _quiet(lambda: contract("ij,jk->ik", second, right))
        expected = before + _quiet(lambda: contract("ij,jk->ik", np.ones_like(first), right))
        assert _rel(after, expected) <= TOLERANCE[name], (name, "stale value")
        assert not np.array_equal(np.asarray(after), np.asarray(before)), (
            name, "换了输入却返回旧值")
        # 原输入不得被改写
        assert np.array_equal(first, _data((3, 4), dtype, 12)), (name, "input mutated")


def test_repeated_calls_are_bitwise_deterministic():
    """同一输入重复调用逐位相同：任何差异都指向状态泄漏（例如第二次换了计划）。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        left = _data((5, 6), dtype, 14)
        right = _data((6, 7), dtype, 15)
        results = [_quiet(lambda: contract("ij,jk->ik", left, right)) for _ in range(4)]
        for item in results[1:]:
            assert np.array_equal(item, results[0]), (name, "non-deterministic")


def test_negative_cache_does_not_hide_recoverable_calls():
    """负缓存只复用确定性 UNSUPPORTED；恢复后必须立刻重新尝试原生。"""
    _kernel_kupl._negative_cache_clear()
    saved = dict(_kernel_kupl._TOPOLOGY_DTYPE_SUPPORT)
    # 方程必须来自语料：语料外一律 status=4（去兜底后的设计），那样"恢复后仍回落"
    # 就是预期行为，验不出负缓存语义。
    equation = "cadb,menf,ac,em,bd,fn->"
    operands = [_data((2,) * len(term), np.complex128, 16 + index)
                for index, term in enumerate(equation.split("->")[0].split(","))]
    try:
        # 显式把探测判负 → 首次回落并可能入负缓存
        _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT[np.dtype(np.complex128)] = False
        assert _kernel_kupl.try_topology_apply(equation, operands) is None
        # 恢复探测：同一 (方程, 形状, dtype) 必须重新尝试，而不是永远回落
        _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT[np.dtype(np.complex128)] = True
        recovered = _kernel_kupl.try_topology_apply(equation, operands)
        assert recovered is not None, (
            f"恢复后仍回落：{_kernel_kupl.LAST_FAILURE}")
        assert _rel(recovered, np.einsum(equation, *operands)) <= TOLERANCE["complex128"]
    finally:
        _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.clear()
        _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.update(saved)
        _kernel_kupl._negative_cache_clear()


# ── 4. 并发 ─────────────────────────────────────────────────────────────────

def test_concurrent_contracts_agree_with_serial():
    """8 线程并发跑同一批方程：结果必须与串行逐位一致（共享缓存线程安全）。

    **不做预热**：入口绑定（dlopen + ctypes argtypes）与线程档初始化都在首次调用
    里完成，如果不加锁，并发首调会各自 dlopen 一遍并交叉改写同一份 argtypes ——
    实测会静默终止进程（退出码 0、无 traceback，只有 ``faulthandler`` 能看到停在
    首次调用内部）。``_kernel_kupl`` / ``_kernel_cuda`` 的入口绑定现在都持
    ``_ENTRY_LOCK``，因此"冷启动即并发"本身就是契约的一部分，必须直接验。
    """
    from concurrent.futures import ThreadPoolExecutor

    cases = []
    for dtype in (np.complex128, np.float64):
        left = _data((6, 7), dtype, 18)
        right = _data((7, 8), dtype, 19)
        cases.append((dtype, left, right))

    def run(case):
        _dtype, left, right = case
        return np.asarray(contract("ij,jk->ik", left, right))

    with patch.dict(os.environ, {"PYQCD_EINSUM_NOTICE": "0"}):
        serial = [run(case) for case in cases]
        with ThreadPoolExecutor(max_workers=8) as pool:
            parallel = list(pool.map(run, cases * 8))
    expected = (serial * 8)
    for index, (left, right) in enumerate(zip(expected, parallel)):
        assert np.array_equal(left, right), f"并发结果与串行不一致 #{index}"


def test_cold_start_concurrency_on_specialized_entries():
    """专用入口在**未预热**状态下并发首调也必须安全（含锁定后仅 dlopen 一次）。"""
    from concurrent.futures import ThreadPoolExecutor

    # 方程必须来自语料：拓扑入口只服务 `topology_kernels.inc` 的编译期步表，
    # 语料外一律 status=4（去兜底后的设计意图）。用 ``ij,jk->ik`` 会恒 miss。
    equation = "cadb,menf,ac,em,bd,fn->"
    operands = [_data((2,) * len(term), np.complex128, 30 + index)
                for index, term in enumerate(equation.split("->")[0].split(","))]
    saved = (_kernel_kupl._entry_topology, _kernel_kupl._failed_topology,
             _kernel_kupl._entry_dedicated, _kernel_kupl._failed_dedicated,
             dict(_kernel_kupl._TOPOLOGY_DTYPE_SUPPORT))
    _kernel_kupl._entry_topology = None
    _kernel_kupl._failed_topology = False
    _kernel_kupl._entry_dedicated = None
    _kernel_kupl._failed_dedicated = False
    # 还要清负缓存：前面组可能留下 status=4 的"该计划不支持"记录，那会让
    # 本用例的首次调用直接走回落分支，测不到冷启动绑定。
    _kernel_kupl._negative_cache_clear()
    # dtype 探测缓存也要清：别的用例可能把 complex128 标成 False（例如 patch 了
    # 原生入口的用例），那会让本用例的门控直接拒绝下发，测不到冷启动绑定。
    _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.clear()
    try:
        def hit(_index):
            return (_kernel_kupl._function_topology() is not None,
                    _kernel_kupl._function_dedicated() is not None)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(hit, range(16)))
        assert len(set(results)) == 1, results      # 所有线程看到同一结论
        assert results[0][0] and results[0][1], results[0]

        # 冷启动并发下专用内核仍必须给出正确结果
        with ThreadPoolExecutor(max_workers=8) as pool:
            outputs = list(pool.map(
                lambda _index: _kernel_kupl.try_topology_apply(equation, operands),
                range(8)))
        expected = np.einsum(equation, *operands)
        for item in outputs:
            assert item is not None
            assert np.allclose(item, expected, atol=1e-12)
    finally:
        (_kernel_kupl._entry_topology, _kernel_kupl._failed_topology,
         _kernel_kupl._entry_dedicated, _kernel_kupl._failed_dedicated,
         dtype_support) = saved
        _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.clear()
        _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.update(dtype_support)
        _kernel_kupl._negative_cache_clear()


def test_gpu_side_concurrency_matches_host():
    """CUDA 侧并发：同一方程多线程各算一次，结果必须与 host 端逐位一致。

    GPU 的并发面与 CPU 不同：这里共享的是 CUDA context、workspace 池与 cuBLAS
    句柄，缺陷表现常是"偶发错数"而不是崩溃，所以断言取**逐位相等**（同一内核、
    同一 stream 语义下没有理由不同）。无 cupy/无设备时跳过。
    """
    try:
        import cupy
    except ImportError:
        raise unittest.SkipTest("cupy is not installed")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            raise unittest.SkipTest("no CUDA device")
    except unittest.SkipTest:
        raise
    except Exception as error:                            # noqa: BLE001 - 驱动缺失
        raise unittest.SkipTest(f"CUDA unavailable: {error}")

    from concurrent.futures import ThreadPoolExecutor

    host_left = _data((48, 48), np.complex128, 40)
    host_right = _data((48, 48), np.complex128, 41)
    device_left = cupy.asarray(host_left)
    device_right = cupy.asarray(host_right)
    expected = contract("ij,jk->ik", host_left, host_right, specialize=False)

    def run(_index):
        with patch.dict(os.environ, {"PYQCD_EINSUM_NOTICE": "0"}):
            result = contract("ij,jk->ik", device_left, device_right)
        cupy.cuda.Device(0).synchronize()
        return cupy.asnumpy(result)

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, range(8)))
    # **逐位相等是错的口径**：cuBLAS 与 CPU BLAS 的累加顺序不同，两个正确实现本来
    # 就不会逐位相同（实测 rel≈1e-16）。这里要验的是"并发之间一致"，所以与 host 用
    # 容差比，而**并发结果彼此**必须逐位相同（同一设备、同一内核、同样输入）。
    scale = float(np.max(np.abs(expected))) or 1.0
    for index, item in enumerate(outputs):
        assert item.shape == expected.shape, index
        error = float(np.max(np.abs(item - expected))) / scale
        assert error <= TOLERANCE["complex128"], (index, error)
        assert np.array_equal(item, outputs[0]), (
            f"GPU 并发结果彼此不一致 #{index}")


# ── 5. 边界与错误 ───────────────────────────────────────────────────────────

def test_zero_extent_and_scalar_output():
    """0 长度轴与全归约到标量的语义：形状对、无 NaN、标量是 numpy 标量或 0-d。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        empty = np.zeros((0, 3), dtype=dtype)
        zero = _quiet(lambda: contract("ij,jk->ik", empty, _data((3, 2), dtype, 20)))
        assert np.asarray(zero).shape == (0, 2), (name, np.asarray(zero).shape)

        squares = _data((4, 4), dtype, 21)
        total = _quiet(lambda: contract("ij,ij->", squares, squares))
        assert np.asarray(total).shape == (), (name, np.asarray(total).shape)
        assert _rel(total, np.sum(squares * squares)) <= TOLERANCE[name], name


def test_invalid_equations_raise_instead_of_returning_garbage():
    """非法输入必须报错，绝不返回静默错数。"""
    left = _data((2, 3), np.float64, 22)
    right = _data((3, 4), np.float64, 23)
    for equation in ("ij,jk->ik$", "ij,jk->ik->il", "ij,jk->iij"):
        try:
            _quiet(lambda: contract(equation, left, right))
        except Exception:
            continue
        raise AssertionError(f"{equation!r} 必须报错")
    try:
        _quiet(lambda: contract("ij,jk->ik", left))
    except Exception:
        pass
    else:
        raise AssertionError("操作数个数不符必须报错")
    try:
        _quiet(lambda: contract("ij,jk->ik", left, _data((4, 4), np.float64, 24)))
    except Exception:
        pass
    else:
        raise AssertionError("shape 不符必须报错")


TESTS = (
    test_associativity_for_all_dtypes,
    test_linearity_and_distributivity,
    test_conjugate_and_transpose_rules,
    test_closed_form_values,
    test_random_equations_against_numpy,
    test_contract_cache_never_returns_stale_values,
    test_repeated_calls_are_bitwise_deterministic,
    test_negative_cache_does_not_hide_recoverable_calls,
    test_concurrent_contracts_agree_with_serial,
    test_cold_start_concurrency_on_specialized_entries,
    test_gpu_side_concurrency_matches_host,
    test_zero_extent_and_scalar_output,
    test_invalid_equations_raise_instead_of_returning_garbage,
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
