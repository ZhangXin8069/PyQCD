"""契约：形状特化路径（``_shape_paths``）的匹配、数值、回落与查询口径。

四件事必须同时成立：

1. **匹配**：按规范签名命中，标签改名、加空格、换输出宽度都不影响；
   结构不同的方程必须 miss 并回落。
2. **数值**：与 numpy 参考一致；单精度用高精度真值判（池内固有散射，见
   ``verify_dtype_parity`` 的口径），双精度 ≤1e-12。
3. **保真**：结果 dtype 与输入逐位相同、形状 ``(n,n,n)``，绝不隐式升降精度。
4. **回落**：形状不匹配、混合 dtype、``out=``、``specialize=False``、
   中间体超预算、非数组输入，都必须安静回落且结果仍正确。

可直接运行::

    python -m pyqcd.testing._shape_paths_contract
"""
from __future__ import annotations

import os
import unittest
import warnings
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import _shape_paths
from ..einsum import contract
from ..einsum import query_specialization

EQUATION = "x,ax,bx,cx->abc"
DTYPES = (np.complex128, np.complex64, np.float64, np.float32)
TRUTH = {np.dtype(np.complex128): np.complex128,
         np.dtype(np.complex64): np.complex128,
         np.dtype(np.float64): np.float64,
         np.dtype(np.float32): np.float64}
TOLERANCE = {"complex128": 1e-12, "float64": 1e-12,
             "complex64": 1e-4, "float32": 1e-4}


def _operands(size: int, dtype: Any, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    values = []
    for shape in ((size,), (size, size), (size, size), (size, size)):
        data = rng.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            data = data + 1j * rng.standard_normal(shape)
        values.append(np.ascontiguousarray(data.astype(dtype)))
    return values


def _quiet(call):
    import contextlib
    import io

    with warnings.catch_warnings(), patch.dict(
            os.environ, {"PYQCD_EINSUM_NOTICE": "0"}):
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            return call()


def _relative(actual: Any, expected: Any) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float("inf")
    scale = float(np.max(np.abs(expected))) or 1.0
    return float(np.max(np.abs(actual - expected))) / scale


# ── 1. 匹配 ─────────────────────────────────────────────────────────────────

def test_registry_matches_by_canonical_signature():
    """规范签名命中：改名/加空格等价；结构不同的方程必须 miss。"""
    assert _shape_paths.registered() == (EQUATION,), _shape_paths.registered()
    for equation in (EQUATION, "y,ay,by,cy->abc", "x, ax, bx, cx -> abc",
                     "q,aq,bq,cq->abc"):
        found = _shape_paths.lookup(equation)
        assert found is not None, equation
        assert found["name"] == "shape:abc-sandwich", found["name"]
        assert found["kind"] == "kernel(shape)", found["kind"]
    for equation in ("ij,jk->ik", "x,ax,bx->ab", "ab,ab->", "", None, 123):
        assert _shape_paths.lookup(equation) is None, equation


def test_query_reports_the_shape_path():
    """公共查询入口必须报 HIT（退出码 0 的依据），并给出内核标签。"""
    report = query_specialization(EQUATION)
    assert report["matched"] is True, report
    assert report["name"] == "shape:abc-sandwich", report
    assert report["kernels"] == ["shape_abc_sandwich"], report
    assert "corpus" not in report, "非语料路径不得伪造语料出处"
    assert query_specialization("y,ay,by,cy->abc")["matched"] is True
    assert query_specialization("ij,jk->ik")["matched"] is False


# ── 2. 数值与保真 ───────────────────────────────────────────────────────────

def test_numerics_match_reference_for_four_dtypes():
    """四类 dtype：双精度对 numpy ≤1e-12，单精度对高精度真值 ≤1e-4。"""
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        for size in (1, 2, 7, 32):
            operands = _operands(size, dtype, seed=size)
            got = _quiet(lambda: contract(EQUATION, *operands))
            truth = _quiet(lambda: contract(
                EQUATION, *[item.astype(TRUTH[np.dtype(dtype)]) for item in operands]))
            assert got is not None, (name, size)
            assert np.dtype(got.dtype) == np.dtype(dtype), (name, got.dtype)
            assert got.shape == (size, size, size), (name, got.shape)
            error = _relative(got, truth)
            assert error <= TOLERANCE[name], (name, size, error)


def test_kernel_is_used_and_matches_numpy():
    """确实走了形状特化（而不是碰巧结果对）：直接调内核并对照 numpy。"""
    size = 9
    operands = _operands(size, np.float64, seed=5)
    direct = _shape_paths.try_execute(EQUATION, operands)
    assert direct is not None, "结构完全匹配时必须由形状特化接管"
    expected = np.einsum(EQUATION, *operands, optimize="greedy")
    assert _relative(direct, expected) <= 1e-12
    # 非连续输入：内部复制后仍必须正确，且不改写调用方的数组
    slab = np.zeros((size, size * 2))
    slab[:, ::2] = operands[1]
    view = slab[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    scaled = [operands[0], view, operands[2], operands[3]]
    got = _shape_paths.try_execute(EQUATION, scaled)
    assert got is not None and _relative(got, np.einsum(EQUATION, *scaled)) <= 1e-12
    assert np.array_equal(slab[:, ::2], operands[1]), "不得原地改写输入"


# ── 3. 回落 ─────────────────────────────────────────────────────────────────

def test_shape_mismatch_falls_back_correctly():
    """轴长不一致 / 秩不对时必须返回 None 并回落，结果仍与 numpy 一致。"""
    rng = np.random.default_rng(3)

    def rel(operands):
        got = _quiet(lambda: contract(EQUATION, *operands))
        return _relative(got, np.einsum(EQUATION, *operands, optimize="greedy"))

    # 轴长一致但 n=4 与 5 混用：内核必须拒绝（否则会维数报错）
    mixed = [rng.standard_normal(5)] + [rng.standard_normal((5, 5)),
                                        rng.standard_normal((4, 4)),
                                        rng.standard_normal((5, 5))]
    assert _shape_paths.try_execute(EQUATION, mixed) is None
    # 内核拒绝后走通用路径：numpy 自身会因形状不符报错，这里只断言"内核不接管"
    non_matrix = [rng.standard_normal(4), rng.standard_normal(4),
                  rng.standard_normal((4, 4)), rng.standard_normal((4, 4))]
    assert _shape_paths.try_execute(EQUATION, non_matrix) is None
    assert rel(_operands(4, np.float64, seed=1)) <= 1e-12


def test_out_and_specialize_switches_are_respected():
    """``out=`` 与 ``specialize=False`` 都不得被形状特化接管。"""
    size = 6
    operands = _operands(size, np.complex128, seed=2)
    out = np.empty((size, size, size), dtype=np.complex128)
    with patch.object(_shape_paths, "try_execute",
                      side_effect=AssertionError("out= 时不得调用形状特化")):
        got = contract(EQUATION, *operands, out=out)
    assert got is out
    assert _relative(out, np.einsum(EQUATION, *operands)) <= 1e-12

    with patch.object(_shape_paths, "try_execute",
                      side_effect=AssertionError("specialize=False 时不得接管")):
        got = contract(EQUATION, *operands, specialize=False)
    assert _relative(got, np.einsum(EQUATION, *operands)) <= 1e-12


def test_memory_guard_and_mixed_dtype_fall_back():
    """中间体超预算或混合 dtype 时必须安静回落，绝不抛异常。"""
    size = 8
    operands = _operands(size, np.float64, seed=4)
    with patch.dict(os.environ, {"PYQCD_EINSUM_SHAPE_PATH_MAX_BYTES": "1"}):
        assert _shape_paths.try_execute(EQUATION, operands) is None
        assert _relative(_quiet(lambda: contract(EQUATION, *operands)),
                         np.einsum(EQUATION, *operands, optimize="greedy")) <= 1e-12

    mixed = _operands(size, np.complex128, seed=4)
    mixed[2] = mixed[2].astype(np.complex64)
    assert _shape_paths.try_execute(EQUATION, mixed) is None

    # 非数组输入：直接交给通用路径，绝不接管
    assert _shape_paths.try_execute(EQUATION, [1, 2, 3, 4]) is None


def test_kernel_exceptions_are_contained():
    """内核自身抛异常时 ``try_execute`` 必须返回 None（fail closed）。"""
    def boom(_operands):
        raise RuntimeError("kernel exploded")

    operands = _operands(4, np.float64, seed=6)
    found = _shape_paths.lookup(EQUATION)
    saved = found["run"]
    found["run"] = boom
    try:
        assert _shape_paths.try_execute(EQUATION, operands) is None
        assert _relative(_quiet(lambda: contract(EQUATION, *operands)),
                         np.einsum(EQUATION, *operands, optimize="greedy")) <= 1e-12
    finally:
        found["run"] = saved


TESTS = (
    test_registry_matches_by_canonical_signature,
    test_query_reports_the_shape_path,
    test_numerics_match_reference_for_four_dtypes,
    test_kernel_is_used_and_matches_numpy,
    test_shape_mismatch_falls_back_correctly,
    test_out_and_specialize_switches_are_respected,
    test_memory_guard_and_mixed_dtype_fall_back,
    test_kernel_exceptions_are_contained,
)


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"{passed} passed, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
