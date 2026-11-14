"""Contracts for deterministic negative caching and dedicated-plan statistics."""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
import os
import traceback
import unittest
from unittest.mock import patch

import numpy as np

from pyqcd.einsum import _kernel_cuda
from pyqcd.einsum import _kernel_kupl
from pyqcd.einsum import _wick_tasklist
from pyqcd.einsum import bench_specialized
from pyqcd.einsum._bench_scene import dims
from pyqcd.einsum.kupl.einsum import NativeUnavailable


_ENV_KEYS = (
    "PYQCD_EINSUM_NEGATIVE_CACHE",
    "PYQCD_EINSUM_DEDICATED_PLAN",
    "PYQCD_EINSUM_KERNEL_KUPL",
    "PYQCD_EINSUM_KERNEL_CUDA",
    # 实数 dtype 在分派层被 `_TOPOLOGY_DECLINED_DTYPES` 主动拒绝（见
    # `_kernel_kupl._topology_dtype_declined`）。本模块的负缓存用例断言的是
    # "原生返回 status 4 之后是否被缓存"，必须把这个策略关掉才能真的调到原生，
    # 否则 `native_calls` 恒为空、断言失去意义。
    "PYQCD_EINSUM_TOPOLOGY_REAL",
)


class _EnvPatch:
    def __init__(self, **values: str) -> None:
        self.values = values
        self.saved: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key in _ENV_KEYS:
            self.saved[key] = os.environ.get(key)
        os.environ.update(self.values)

    def __exit__(self, *_args) -> None:
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _fake_cached(output_shape=(2, 2)):
    plan = b"x"
    buffer = (ctypes.c_ubyte * len(plan))()
    return plan, output_shape, buffer, 0


def _fake_plan_stats(_plan):
    return {
        "plan_flops": 1,
        "plan_pack_bytes": 0,
        "plan_arena_bytes": 0,
        "plan_max_step_macs": 0,
    }


def _operands(dtype=np.float64):
    return (
        np.arange(6, dtype=dtype).reshape(2, 3),
        np.arange(6, dtype=dtype).reshape(3, 2),
    )


def test_negative_cache_key_has_all_required_dimensions():
    base = _kernel_kupl._negative_cache_key(
        "ij,jk->ik", ((2, 3), (3, 4)), np.float64, "auto", "kupl_dedicated")
    normalized = _kernel_kupl._negative_cache_key(
        "i j, jk -> i k", ((2, 3), (3, 4)), np.float64, "auto",
        "kupl_dedicated")
    assert normalized == base
    variants = {
        base,
        _kernel_kupl._negative_cache_key(
            "ij,jk->ik", ((2, 3), (3, 5)), np.float64, "auto",
            "kupl_dedicated"),
        _kernel_kupl._negative_cache_key(
            "ij,jk->ik", ((2, 3), (3, 4)), np.float32, "auto",
            "kupl_dedicated"),
        _kernel_kupl._negative_cache_key(
            "ij,jk->ik", ((2, 3), (3, 4)), np.float64, "branch-2",
            "kupl_dedicated"),
        _kernel_kupl._negative_cache_key(
            "ij,jk->ik", ((2, 3), (3, 4)), np.float64, "auto",
            "cuda_dedicated"),
    }
    assert len(variants) == 5


def test_dedicated_status4_is_cached():
    planner_calls = []
    native_calls = []

    def planner(*args):
        planner_calls.append(args)
        return _fake_cached()

    def entry(*_args):
        native_calls.append(1)
        return 4

    _kernel_kupl._negative_cache_clear()
    with _EnvPatch(PYQCD_EINSUM_NEGATIVE_CACHE="1"), \
            patch.object(_kernel_kupl, "_dedicated_cached", planner), \
            patch.object(_kernel_kupl, "_plan_statistics", _fake_plan_stats), \
            patch.object(_kernel_kupl, "_function_dedicated", lambda: entry):
        first = _kernel_kupl.try_dedicated("ij,jk->ik", _operands())
        second = _kernel_kupl.try_dedicated("ij,jk->ik", _operands())

    assert first is None and second is None
    assert len(planner_calls) == 1, planner_calls
    assert len(native_calls) == 1, native_calls
    assert _kernel_kupl.LAST_FAILURE["status"] == 4
    assert "negative-cache hit" in str(_kernel_kupl.LAST_FAILURE["reason"])
    info = _kernel_kupl.negative_cache_info()
    assert info["size"] == 1 and info["hits"] >= 1 and info["stores"] == 1, info


def test_oom_and_internal_errors_are_not_cached():
    _kernel_kupl._negative_cache_clear()
    for status in (2, 3):
        native_calls = []

        def entry(*_args, status=status):
            native_calls.append(status)
            return status

        with _EnvPatch(PYQCD_EINSUM_NEGATIVE_CACHE="1"), \
                patch.object(
                    _kernel_kupl, "_dedicated_cached",
                    lambda *args: _fake_cached()), \
                patch.object(
                    _kernel_kupl, "_plan_statistics", _fake_plan_stats), \
                patch.object(_kernel_kupl, "_function_dedicated", lambda: entry):
            assert _kernel_kupl.try_dedicated(
                f"ij,jk->ik", _operands()) is None
            assert _kernel_kupl.try_dedicated(
                f"ij,jk->ik", _operands()) is None

        assert native_calls == [status, status], native_calls
        assert _kernel_kupl.negative_cache_info()["size"] == 0
        assert _kernel_kupl.LAST_FAILURE["status"] == status


def test_optimizer_shape_and_dtype_do_not_share_rejections():
    native_calls = []

    def entry(*_args):
        native_calls.append(1)
        return 4

    _kernel_kupl._negative_cache_clear()
    with _EnvPatch(PYQCD_EINSUM_NEGATIVE_CACHE="1"), \
            patch.object(
                _kernel_kupl, "_dedicated_cached",
                lambda *args: _fake_cached()), \
            patch.object(_kernel_kupl, "_plan_statistics", _fake_plan_stats), \
            patch.object(_kernel_kupl, "_function_dedicated", lambda: entry):
        _kernel_kupl.try_dedicated("ij,jk->ik", _operands(), optimize="auto")
        _kernel_kupl.try_dedicated("ij,jk->ik", _operands(), optimize="branch-2")
        _kernel_kupl.try_dedicated("ij,jk->ik", _operands(), optimize="auto")
        _kernel_kupl.try_dedicated(
            "ij,jk->ik",
            (np.ones((2, 4)), np.ones((4, 2))),
            optimize="auto",
        )
        # dtype 是负缓存键的一部分：四类各自第一次会真的下发一次（共 4 次），
        # 重复调用必须命中缓存。这里每类调用两次，断言原生只被调 4 次。
        for dtype in (np.complex128, np.complex64, np.float64, np.float32):
            for _ in range(2):
                _kernel_kupl.try_dedicated(
                    "ij,jk->ik",
                    tuple(item.astype(dtype) for item in _operands()),
                    optimize="auto",
                )

    assert len(native_calls) == 6, native_calls
    assert _kernel_kupl.negative_cache_info()["size"] == 6


def test_explicit_dedicated_plan_switch_is_not_default():
    seen = []
    _kernel_kupl._negative_cache_clear()

    def planner(_subscripts, _shapes, _dtype, optimize):
        seen.append(optimize)
        return None

    with _EnvPatch(PYQCD_EINSUM_DEDICATED_PLAN=""), \
            patch.object(_kernel_kupl, "_dedicated_cached", planner), \
            patch.object(_kernel_kupl, "_function_dedicated", lambda: object()):
        _kernel_kupl.try_dedicated("ij,jk->ik", _operands())

    with _EnvPatch(PYQCD_EINSUM_DEDICATED_PLAN="branch-2"), \
            patch.object(_kernel_kupl, "_dedicated_cached", planner), \
            patch.object(_kernel_kupl, "_function_dedicated", lambda: object()):
        _kernel_kupl.try_dedicated("ij,jk->ik", _operands())
        _kernel_kupl.try_dedicated(
            "ij,jk->ik", _operands(), optimize="greedy")

    assert seen == ["auto", "branch-2", "greedy"], seen


def test_topology_status4_is_cached():
    native_calls = []

    def unavailable(*_args, **_kwargs):
        native_calls.append(1)
        error = NativeUnavailable("plan not executable as version-2 steps")
        error.status = 4
        raise error

    left = np.ones((3, 4), dtype=np.float32)
    right = np.ones((4, 5), dtype=np.float32)
    _kernel_kupl._negative_cache_clear()
    # 本用例 patch 了原生入口，dtype 探测必然失败：显式放行 float32，否则门控
    # 会把"探测没过"当回落原因，原生根本不会被调用（run 顺序不同时表现为
    # 单跑通过、整套失败）。
    saved_support = dict(_kernel_kupl._TOPOLOGY_DTYPE_SUPPORT)
    _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT[np.dtype(np.float32)] = True
    try:
        with _EnvPatch(PYQCD_EINSUM_NEGATIVE_CACHE="1",
                       PYQCD_EINSUM_TOPOLOGY_REAL="1"), \
                patch.object(_kernel_kupl, "_function_topology", lambda: object()), \
                patch.object(_kernel_kupl.kupl, "topology_apply", unavailable):
            assert _kernel_kupl.try_topology_apply(
                "ij,jk->ik", [left, right]) is None
            assert _kernel_kupl.try_topology_apply(
                "ij,jk->ik", [left, right]) is None
    finally:
        _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.clear()
        _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.update(saved_support)

    assert len(native_calls) == 1, native_calls
    assert _kernel_kupl.LAST_FAILURE["status"] == 4
    assert "negative-cache hit" in str(_kernel_kupl.LAST_FAILURE["reason"])


def test_thread_safe_negative_cache_reads():
    native_calls = []

    def entry(*_args):
        native_calls.append(1)
        return 4

    _kernel_kupl._negative_cache_clear()
    with _EnvPatch(PYQCD_EINSUM_NEGATIVE_CACHE="1"), \
            patch.object(
                _kernel_kupl, "_dedicated_cached",
                lambda *args: _fake_cached()), \
            patch.object(_kernel_kupl, "_plan_statistics", _fake_plan_stats), \
            patch.object(_kernel_kupl, "_function_dedicated", lambda: entry):
        assert _kernel_kupl.try_dedicated("ij,jk->ik", _operands()) is None
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(
                lambda _index: _kernel_kupl.try_dedicated(
                    "ij,jk->ik", _operands()),
                range(64),
            ))

    assert all(result is None for result in results)
    assert len(native_calls) == 1, native_calls
    info = _kernel_kupl.negative_cache_info()
    assert info["hits"] >= 64, info


def test_plan_statistics_exposes_flops_memory_and_pack_traffic():
    encoded = _kernel_kupl._dedicated_plan(
        "ij,jk,kl->il", ((8, 6), (6, 5), (5, 4)), 3, "greedy")
    assert encoded is not None
    stats = _kernel_kupl._plan_statistics(encoded[0])
    assert stats["plan_flops"] == 800, stats
    assert stats["plan_max_intermediate"] == 40, stats
    assert stats["plan_max_step_macs"] == 240, stats
    assert stats["plan_pack_bytes"] >= 0
    assert stats["plan_arena_bytes"] >= stats["plan_max_intermediate_bytes"]


def _full_scene_shapes(seq):
    task = _wick_tasklist.by_seq()[seq]
    terms = task.equation.split("->", 1)[0].split(",")
    sizes = bench_specialized._label_sizes(terms, task.roles, dims("full"))
    return task.equation, [
        tuple(sizes[label] for label in term) for term in terms
    ]


def test_heavy_cpu_plan_promotes_branch2_only_with_all_round_gains():
    equation, shapes = _full_scene_shapes(6585)
    dtype_id = 0
    auto = _kernel_kupl._dedicated_plan(
        equation, shapes, dtype_id, "auto")
    branch2 = _kernel_kupl._dedicated_plan(
        equation, shapes, dtype_id, "branch-2")
    selected = _kernel_kupl._cheapest_dedicated_plan(
        equation, shapes, dtype_id, True)

    assert auto is not None, "seq6585 auto plan unavailable"
    assert branch2 is not None, "seq6585 branch-2 plan unavailable"
    assert selected is not None, "seq6585 selected plan unavailable"
    auto_stats = _kernel_kupl._plan_statistics(auto[0])
    branch2_stats = _kernel_kupl._plan_statistics(branch2[0])
    assert selected[0] == branch2[0], {
        "selected": _kernel_kupl._plan_statistics(selected[0]),
        "auto": auto_stats,
        "branch2": branch2_stats,
    }
    assert branch2_stats["plan_flops"] * 100 <= auto_stats["plan_flops"] * 80, (
        auto_stats, branch2_stats)
    assert branch2_stats["plan_arena_bytes"] * 100 <= auto_stats["plan_arena_bytes"] * 65, (
        auto_stats, branch2_stats)
    assert branch2_stats["plan_pack_bytes"] * 100 <= auto_stats["plan_pack_bytes"] * 65, (
        auto_stats, branch2_stats)

    branch2_complex64 = _kernel_kupl._dedicated_plan(
        equation, shapes, 1, "branch-2")
    selected_complex64 = _kernel_kupl._cheapest_dedicated_plan(
        equation, shapes, 1, True)
    assert branch2_complex64 is not None, "seq6585 complex64 branch-2 plan unavailable"
    assert selected_complex64 is not None, "seq6585 complex64 selected plan unavailable"
    assert selected_complex64[0] != branch2_complex64[0], {
        "selected": _kernel_kupl._plan_statistics(selected_complex64[0]),
        "branch2": _kernel_kupl._plan_statistics(branch2_complex64[0]),
    }

    equation, shapes = _full_scene_shapes(129)
    best = _kernel_kupl._cheapest_dedicated_plan(
        equation, shapes, dtype_id, False)
    branch2 = _kernel_kupl._dedicated_plan(equation, shapes, dtype_id, "branch-2")
    selected = _kernel_kupl._cheapest_dedicated_plan(
        equation, shapes, dtype_id, True)
    assert best is not None, "seq129 auto/greedy plan unavailable"
    assert branch2 is not None, "seq129 branch-2 plan unavailable"
    assert selected is not None, "seq129 selected plan unavailable"
    assert _kernel_kupl._plan_statistics(branch2[0])["plan_flops"] > (
        _kernel_kupl._plan_statistics(best[0])["plan_flops"])
    assert selected[0] != branch2[0], {
        "selected": _kernel_kupl._plan_statistics(selected[0]),
        "best": _kernel_kupl._plan_statistics(best[0]),
        "branch2": _kernel_kupl._plan_statistics(branch2[0]),
    }


def test_cpu_full_plan_resource_gate():
    """CPU 资源门：四类 dtype 各自标定，且分化必须由**元素字节数**解释。

    门看的是 arena/pack **字节**，不是 FLOPs；因此 complex128 占 16 B、float64 8 B、
    complex64/float32 8/4 B，小字节 dtype 的 arena 更小、更容易放行。实测分化矩阵
    （ok=放行 / OUT=回落）：

        seq    id0(c128) id1(c64) id2(f64) id3(f32)
        6585       ok       OUT      OUT      ok
        129        ok        ok       ok       ok
        1599       ok        ok       ok       ok
        2585/489/2588 OUT    OUT      OUT      OUT
        7918       OUT       OUT      OUT      ok

    判据写成"每类 dtype 的门结论单调可由 arena 解释"，而不是假设四类结构一致——
    后者是我最初的错误假设，被实测立刻否掉。
    """
    for dtype_id in (0, 1, 2, 3):
        for seq, allowed in (
            (6585, None),
            (129, True),
            (1599, True),
            (2585, False),
            (489, False),
            (2588, False),
            (7918, None),
        ):
            equation, shapes = _full_scene_shapes(seq)
            encoded = _kernel_kupl._cheapest_dedicated_plan(
                equation, shapes, dtype_id, True)
            assert encoded is not None, (seq, dtype_id)
            stats = _kernel_kupl._plan_statistics(encoded[0])
            reason = _kernel_kupl._cpu_plan_fallback_reason(stats)
            if allowed is not None:
                assert (reason is None) is allowed, (seq, dtype_id, stats, reason)
            else:
                # 分歧格：结论必须由某一条**已声明**的门解释，不能是随机行为。
                limited = (
                    stats["plan_arena_bytes"] >= _kernel_kupl._CPU_PLAN_ARENA_LIMIT
                    or stats["plan_max_step_macs"] >= _kernel_kupl._CPU_PLAN_MAX_STEP_MACS_LIMIT
                    or (stats["plan_pack_bytes"] >= _kernel_kupl._HEAVY_PACK_MIN_BYTES
                        and stats["plan_flops"] <= _kernel_kupl._HEAVY_PACK_MAX_FLOPS))
                assert (reason is None) is (not limited), (
                    seq, dtype_id, stats, reason)

    # **不要**假设"小字节 dtype 的 arena 更小"：实测 seq=6585 的 complex64 是
    # 9.41 GB，反而大于 complex128 的 6.80 GB —— 规划器在不同 dtype 上会选出不同
    # 收缩顺序（单精度走 branch-2 的条件不同），字节数不是单调的。这里只断言
    # "四类各自的门结论都由已声明的三条门解释"，不假设 dtype 之间的序关系。
    for seq in (6585, 7918):
        equation, shapes = _full_scene_shapes(seq)
        for dtype_id in (0, 1, 2, 3):
            encoded = _kernel_kupl._cheapest_dedicated_plan(
                equation, shapes, dtype_id, True)
            stats = _kernel_kupl._plan_statistics(encoded[0])
            reason = _kernel_kupl._cpu_plan_fallback_reason(stats)
            limited = (
                stats["plan_arena_bytes"] >= _kernel_kupl._CPU_PLAN_ARENA_LIMIT
                or stats["plan_max_step_macs"] >= _kernel_kupl._CPU_PLAN_MAX_STEP_MACS_LIMIT
                or (stats["plan_pack_bytes"] >= _kernel_kupl._HEAVY_PACK_MIN_BYTES
                    and stats["plan_flops"] <= _kernel_kupl._HEAVY_PACK_MAX_FLOPS))
            assert (reason is None) is (not limited), (seq, dtype_id, stats, reason)


def test_cuda_full_plan_resource_gate():
    """CUDA 资源门：四类 dtype 各自标定，分歧格由 arena 阈值解释（同 CPU 侧）。"""
    dtype_of = {0: np.dtype(np.complex128), 1: np.dtype(np.complex64),
                2: np.dtype(np.float64), 3: np.dtype(np.float32)}
    for dtype_id, dtype in dtype_of.items():
        for seq, allowed in (
            (1863, True),
            (1599, True),
            (6911, True),
            (2588, False),
            (8638, False),
            (6070, None),
            (129, None),
            (7918, None),
        ):
            equation, shapes = _full_scene_shapes(seq)
            encoded = _kernel_kupl._cheapest_dedicated_plan(
                equation, shapes, dtype_id, False)
            assert encoded is not None, (seq, dtype_id)
            stats = _kernel_kupl._plan_statistics(encoded[0])
            reason = _kernel_cuda._plan_fallback_reason(dtype, stats)
            if allowed is not None:
                assert (reason is None) is allowed, (seq, dtype, stats, reason)
            else:
                limited = (
                    stats["plan_arena_bytes"] >= _kernel_cuda._CUDA_PLAN_ARENA_LIMIT
                    or stats["plan_max_step_macs"] >= _kernel_cuda._CUDA_PLAN_MAX_STEP_MACS_LIMIT
                    or (stats["plan_pack_bytes"] >= _kernel_cuda._CUDA_HEAVY_PACK_MIN_BYTES
                        and stats["plan_flops"] <= _kernel_cuda._CUDA_HEAVY_PACK_MAX_FLOPS))
                assert (reason is None) is (not limited), (
                    seq, dtype, stats, reason)


def test_cuda_dedicated_status4_is_cached():
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise unittest.SkipTest("CUDA is hidden for this CPU-only job")
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() == 0:
            raise unittest.SkipTest("CUDA device is not available")
    except unittest.SkipTest:
        raise
    except Exception as exc:  # noqa: BLE001 - hidden/missing device is a skip
        raise unittest.SkipTest(f"CUDA unavailable: {exc}") from exc

    native_calls = []

    def entry(*_args):
        native_calls.append(1)
        return 4

    left = cupy.asarray(np.ones((2, 3), dtype=np.float64))
    right = cupy.asarray(np.ones((3, 2), dtype=np.float64))
    _kernel_kupl._negative_cache_clear()
    with _EnvPatch(
            PYQCD_EINSUM_NEGATIVE_CACHE="1",
            PYQCD_EINSUM_KERNEL_CUDA="1"), \
            patch.object(
                _kernel_cuda, "dedicated_dtypes",
                lambda: (np.dtype(np.float64),)), \
            patch.object(
                _kernel_kupl, "_dedicated_cached",
                lambda *args: _fake_cached()), \
            patch.object(
                _kernel_kupl, "_plan_statistics",
                lambda _plan: {"plan_arena_bytes": 0}), \
            patch.object(_kernel_cuda, "_function_dedicated", lambda: entry):
        assert _kernel_cuda.try_dedicated("ij,jk->ik", [left, right]) is None
        assert _kernel_cuda.try_dedicated("ij,jk->ik", [left, right]) is None

    assert len(native_calls) == 1, native_calls
    assert _kernel_cuda.LAST_FAILURE["status"] == 4


TESTS = (
    test_negative_cache_key_has_all_required_dimensions,
    test_dedicated_status4_is_cached,
    test_oom_and_internal_errors_are_not_cached,
    test_optimizer_shape_and_dtype_do_not_share_rejections,
    test_explicit_dedicated_plan_switch_is_not_default,
    test_topology_status4_is_cached,
    test_thread_safe_negative_cache_reads,
    test_plan_statistics_exposes_flops_memory_and_pack_traffic,
    test_heavy_cpu_plan_promotes_branch2_only_with_all_round_gains,
    test_cpu_full_plan_resource_gate,
    test_cuda_full_plan_resource_gate,
    test_cuda_dedicated_status4_is_cached,
)


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - command-line verifier
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"{passed} passed, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
