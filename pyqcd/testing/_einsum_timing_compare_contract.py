"""契约：计时与参照的**三处失真**（2026-11-19 用户实测）。

三件事都属于"报告看起来正常、数字却是错的"那一类，普通数值测试一个都抓不到：

1. **``--compare opt_einsum`` 只走了 numpy**。``oe.contract`` 的 ``backend`` 缺省
   ``'auto'``，按**数组类型**选后端；旧实现只传 ``host``（numpy 副本），于是
   ``--device gpu --compare opt_einsum`` 的"参照"跑在 CPU 上，与主路径的 GPU 秒数
   并列展示却不可比。
2. **异步后端只量到入队时间**。``cupy``/``torch`` 是异步下发，``perf_counter`` 停在
   ``contract()`` 返回那一刻时 GEMM 可能还在队列里。实测同一条方程脚本里 ~900 ms、
   ``--tune`` 写进配置表 **0.32 ms**，那个数还被当成 ``best``。
3. **``--warmup 1`` 盖不住 GPU 冷启动**。首调要建 cuBLAS handle / workspace / 步表，
   一次丢弃不够，最快样本仍可能混入启动成本。

本模块钉住修复后的**可观测证据**，不依赖本机有没有 GPU（无设备时走 SKIP）：
参照行必须写明后端；``_timed`` 必须在计时区前后调用同步器；加速设备上 warmup 至少
2 且报告里如实标出"被抬过"。

可直接运行::

    python -B -m pyqcd.testing._einsum_timing_compare_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import pathlib
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import _sync
from ..einsum import test as cli

#: 小方程：参照与主路径都在毫秒级，不需要 GPU 也能跑完。
EQUATION = "ij,jk,kl,lm,mn->in"
SHAPES = [(8, 8)] * 5


def _has_cupy() -> bool:
    try:
        import cupy
    except Exception:                           # noqa: BLE001 - 没装就是没装
        return False
    try:
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:                           # noqa: BLE001 - 驱动/CUDA 不可用
        return False


@contextlib.contextmanager
def _quiet():
    """静音库内提示与画像（不读仓库根的 ``./.cache``）。"""
    saved = {name: os.environ.get(name)
             for name in ("PYQCD_EINSUM_NOTICE", "PYQCD_EINSUM_PROFILE")}
    os.environ["PYQCD_EINSUM_NOTICE"] = "0"
    os.environ["PYQCD_EINSUM_PROFILE"] = "0"
    try:
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ── 1. 参照必须跑在与主路径同一个后端上 ─────────────────────────────────────

def test_opt_einsum_reference_declares_and_uses_its_backend() -> None:
    """``--compare opt_einsum`` 的说明行必须含 ``backend=``，且与操作数一致。

    CPU 上 ``backend='numpy'``；cupy 可用时参照必须走 ``cupy``（同一块设备、同一份
    数据），而不是拿 CPU 数字去比 GPU 数字。
    """
    with _quiet():
        cpu = cli.run(equation=EQUATION, shapes=SHAPES, out_shape=(8, 8),
                      device="cpu", dtype="float64", backend="numpy",
                      compare="opt_einsum", rounds=1, warmup=1)
    assert cpu.get("ok"), cpu
    assert "backend='numpy'" in cpu["reference"], (
        "参照说明行必须报出实际后端（否则 GPU 上的 numpy 参照看不出来）",
        cpu["reference"])

    if not _has_cupy():
        raise unittest.SkipTest("没有可用的 CUDA 设备，GPU 侧参照无法验证")
    with _quiet():
        gpu = cli.run(equation=EQUATION, shapes=SHAPES, out_shape=(8, 8),
                      device="gpu", dtype="float64", backend="cupy",
                      compare="opt_einsum", rounds=1, warmup=1)
    assert gpu.get("ok"), gpu
    assert "backend='cupy'" in gpu["reference"], (
        "cupy 操作数上的 opt_einsum 参照必须显式走 cupy 后端", gpu["reference"])


def test_backend_choice_follows_the_operands_not_the_flag() -> None:
    """后端名只能按**操作数类型**推：命令行 ``--device`` 不是可信来源。"""
    assert cli._operand_backend([np.zeros(2)]) == "numpy"
    if _has_cupy():
        import cupy

        assert cli._operand_backend([cupy.zeros(2)]) == "cupy"


# ── 2. 计时区前后必须有设备同步 ─────────────────────────────────────────────

def test_timed_synchronises_around_each_sample() -> None:
    """``_timed`` 必须在每个样本**前后**各调一次同步器（CPU 上不调）。

    用假同步器数调用次数：``warmup + 2 × rounds`` 是精确值——warmup 不需要同步
    （它不进计时），而每个样本进出各一次。
    """
    calls: list[str] = []

    def fake_sync() -> None:
        calls.append("sync")

    result = cli._timed(lambda: None, 1, 3, fake_sync)
    assert result >= 0.0, result
    assert len(calls) == 2 * 3, ("每个样本前后各一次", len(calls))

    calls.clear()
    cli._timed(lambda: None, 1, 2)              # sync=None：CPU 语义，不产生同步
    assert calls == [], calls


def test_synchronize_is_a_noop_on_host_arrays() -> None:
    """CPU 数组上 ``synchronize`` 不导入任何加速库、不抛异常。"""
    _sync.synchronize(np.zeros(4), [np.zeros(2), np.zeros(3)])
    assert _sync.tensor_backend(np.zeros(2)) == "numpy"
    assert _sync.tensor_backend(object()) is None


def test_worker_measures_with_synchronisation() -> None:
    """``_tune_worker`` 的受测调用同样要同步（否则 tune 写进表的 wall_ms 失真）。"""
    from ..einsum import _tune_worker

    source = _tune_worker.run_job.__code__.co_consts
    rendered = repr(source)
    assert "_sync" in rendered or "synchronize" in rendered, (
        "run_job 里必须能看到同步调用（它是 GPU 上 wall_ms 的唯一来源）")


def test_worker_summary_uses_the_minimum_sample() -> None:
    """The tune worker reports the fastest observation, not a median."""
    from ..einsum import _tune_worker

    assert _tune_worker._sample_minimum([0.30, 0.10, 0.20]) == 0.10


def test_wall_time_has_no_legacy_median_alias() -> None:
    """The worker and tune rows expose ``minimum_s``/``wall_ms`` only."""
    from ..einsum import _tune_worker

    source = pathlib.Path(_tune_worker.__file__).read_text(encoding="utf-8")
    assert "median_s" not in source, "worker still emits the deprecated median_s alias"
    assert "minimum_s" in source, "worker must expose the canonical minimum_s field"

    rows = [{"wall_ms": 1.0}, {"wall_ms": None}]
    assert cli.tune_best(rows) is rows[0]
    # A legacy-only measurement must not be treated as a usable wall-time result.
    assert cli.tune_best([{"median_ms": 0.1}]) is None


# ── 3. 加速设备上 warmup 至少 2，并且如实标注 ───────────────────────────────

def test_accelerated_device_raises_warmup_and_says_so() -> None:
    """加速设备把 warmup 抬到 ≥2 并如实标注；CPU 上原样不动。"""
    effective, raised = cli._effective_warmup(1, [np.zeros(2)])
    assert (effective, raised) == (1, False), ("CPU 上不许动 warmup", effective, raised)

    if not _has_cupy():
        raise unittest.SkipTest("没有可用的 CUDA 设备，加速侧 warmup 无法验证")
    import cupy

    effective, raised = cli._effective_warmup(1, [cupy.zeros(2)])
    assert effective >= cli.ACCELERATED_MIN_WARMUP and raised, (effective, raised)
    # 已经给得更大时不许压低
    effective, raised = cli._effective_warmup(5, [cupy.zeros(2)])
    assert (effective, raised) == (5, False), (effective, raised)


def test_gpu_run_reports_the_effective_warmup() -> None:
    """真跑一次 GPU：报告里的 ``warmup`` 是**实际用的**，且标出被抬过。"""
    if not _has_cupy():
        raise unittest.SkipTest("没有可用的 CUDA 设备")
    with _quiet():
        report = cli.run(equation=EQUATION, shapes=SHAPES, out_shape=(8, 8),
                         device="gpu", dtype="float64", backend="cupy",
                         compare="none", rounds=2, warmup=1)
    assert report.get("ok"), report
    assert report.get("warmup") == 1, ("用户给的取值要原样保留", report.get("warmup"))
    assert report.get("warmup_effective", 0) >= cli.ACCELERATED_MIN_WARMUP, (
        "加速设备上的实际 warmup 必须被抬高", report.get("warmup_effective"))
    assert report.get("warmup_raised") is True, report.get("warmup_raised")
    assert report["seconds"] > 0.0, report["seconds"]


def test_tune_groups_use_the_effective_warmup() -> None:
    """tune 的组级预算按**实际** warmup 算，子进程作业也收实际值。"""
    cli_module = cli
    captured: list[dict[str, Any]] = []
    original = cli_module._tune_runner.run_group

    def spy(job, **kwargs):
        captured.append(dict(job))
        return original(job, **kwargs)

    combos = (("generic", "false", "greedy"),)
    with _quiet(), patch.object(cli_module, "tune_combos",
                                lambda *a, **k: list(combos)), \
            patch.object(cli_module._tune_runner, "run_group", spy):
        cli_module.run_tune(equation=EQUATION, shapes=[(4, 4)] * 5, warmup=1,
                            rounds=1, profile=False, isolate=True)
    assert captured, "这一轮组根本没有下发"
    # CPU 上 warmup 不被抬：下发的必须还是 1（若这里变成 2，说明判定把 CPU 当成了 GPU）
    assert captured[0]["warmup"] == 1, captured[0].get("warmup")


TESTS = (
    test_backend_choice_follows_the_operands_not_the_flag,
    test_opt_einsum_reference_declares_and_uses_its_backend,
    test_timed_synchronises_around_each_sample,
    test_synchronize_is_a_noop_on_host_arrays,
    test_worker_measures_with_synchronisation,
    test_worker_summary_uses_the_minimum_sample,
    test_wall_time_has_no_legacy_median_alias,
    test_accelerated_device_raises_warmup_and_says_so,
    test_gpu_run_reports_the_effective_warmup,
    test_tune_groups_use_the_effective_warmup,
)


def test_einsum_timing_compare_contracts() -> None:
    """套件入口：全过，否则汇总失败；一条都没跑时显式 SkipTest。"""
    failures: list[str] = []
    passed = skipped = 0
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
            f"{len(failures)} 条计时/参照契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("计时/参照契约全部跳过")


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
