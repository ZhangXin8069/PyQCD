"""契约：``--gpu-probe`` 的 dtype 诊断必须**如实**（2026-11-19 用户实测）。

用户在 A100 上 ``pip install pyqcd`` 后跑 ``python -m pyqcd --gpu-probe``，读到::

    kernel status : fallback — topology entry did not run: dtype complex64 is inside
                    the topology dtype contract but its native probe failed
                    (2x2 matmul disagreed with numpy); ...

那句话是**编的**：``_probe_dtype`` 的 ``except Exception: return False`` 把六种完全
不同的失败（缺符号 / cupy 版本不符 / 形状不符 / dtype 不符 / 数值超差 / NVRTC 编译
失败）一起吞掉，上层于是写死"数值不一致"。缺符号时这句话直接指错方向。

本模块钉住修复后的**可观测证据**，不依赖本机有没有 GPU（无设备时走 SKIP）：

1. 探测失败会把真实 ``stage`` + ``detail``（异常文本或实测数字）写进 ``LAST_PROBE``；
2. ``_topology_dtype_reason`` 不再出现那句写死文案，失败原因逐字引自探测事实；
3. ``probe_dtypes()`` 覆盖 ``TOPOLOGY_DTYPES`` 四类、键是 dtype 名，且已缓存的
   dtype 不重探；
4. ``_probe.py`` 的端到端容差取自 ``_kernel_cuda._PROBE_TOLERANCE``（源码里没有
   写死的 ``1e-6``）；
5. ``--gpu-probe`` 报告带 ``dtypes`` 字典，``kernel_status.detail`` 含逐 dtype 结论，
   "部分可用"的 verdict 既不谎称 active 也不谎称 NOT active。

可直接运行::

    python -B -m pyqcd.testing._einsum_gpu_probe_diagnosis_contract
"""
from __future__ import annotations

import inspect
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import _kernel_cuda as kernel
from ..einsum import _probe

#: 用户实测里那句**写死**的、把五种原因统一说成数值不一致的文案。
FIXED_PHRASE = "2x2 matmul disagreed with numpy"

#: 缺符号时的真实异常文本（``_program.topology_apply`` 实际抛的就是这一句）。
MISSING_SYMBOL = "qcd_cuda_topology_apply is not exported by the archive"

DTYPE_NAMES = tuple(dtype.name for dtype in kernel.TOPOLOGY_DTYPES)


def _has_cuda() -> bool:
    """本进程能否真的跑 cupy 设备代码（无设备时相关用例 SKIP）。"""
    try:
        import cupy
    except Exception:                           # noqa: BLE001 - 没装就是没装
        return False
    try:
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:                           # noqa: BLE001 - 驱动/CUDA 不可用
        return False


def _snapshot() -> tuple[dict, dict, dict]:
    """探测相关的三份可变状态（bool 缓存 / 逐 dtype 事实 / 最近一次事实）。"""
    return (dict(kernel._TOPOLOGY_DTYPE_SUPPORT),
            dict(kernel._TOPOLOGY_DTYPE_PROBE),
            dict(kernel.LAST_PROBE))


def _restore(saved: tuple[dict, dict, dict]) -> None:
    support, probes, last = saved
    kernel._TOPOLOGY_DTYPE_SUPPORT.clear()
    kernel._TOPOLOGY_DTYPE_SUPPORT.update(support)
    kernel._TOPOLOGY_DTYPE_PROBE.clear()
    kernel._TOPOLOGY_DTYPE_PROBE.update(probes)
    kernel.LAST_PROBE.clear()
    kernel.LAST_PROBE.update(last)


def _fake_record(name: str, ok: bool, stage: str = "native",
                 detail: str = "TypeError: fake") -> dict[str, Any]:
    return {"dtype": name, "ok": ok, "stage": "ok" if ok else stage,
            "detail": detail, "max_abs_diff": None, "tolerance": None,
            "reference_max": None}


def _fake_dtypes(failing: tuple[str, ...] = ()) -> dict[str, dict[str, Any]]:
    return {name: _fake_record(name, name not in failing)
            for name in DTYPE_NAMES}


# ── 1. 探测失败必须留下真实 stage / detail ───────────────────────────────────

def test_failed_probe_records_stage_and_detail() -> None:
    """探测失败时 ``LAST_PROBE`` 带 ``stage`` 与真实 ``detail``，判断仍只返回 ``False``。

    用非法输出标签把探测钉在 **plan** 阶段失败：不依赖原生库缺不缺符号，也不依赖
    真实数值超差。无设备时会更早失败（``build``），但"失败必须留证据"这条不变。
    """
    saved = _snapshot()
    try:
        kernel._TOPOLOGY_DTYPE_SUPPORT.clear()
        kernel._TOPOLOGY_DTYPE_PROBE.clear()
        kernel.LAST_PROBE.clear()
        with patch.object(kernel, "_PROBE_EQUATION", "ij,jk->iz"):
            ok = kernel._probe_dtype(np.dtype(np.complex64))
        record = dict(kernel.LAST_PROBE)
    finally:
        _restore(saved)

    assert ok is False, "探测失败必须返回 False（契约：探测器绝不抛、绝不改返回值形状）"
    assert record, "探测失败必须留下 LAST_PROBE，否则上层只能再编一次原因"
    assert record.get("dtype") == "complex64", record
    assert record.get("ok") is False, record
    assert record.get("stage") in kernel.PROBE_STAGES, record
    assert record.get("stage") != "ok", ("失败的探测不许记成 ok 阶段", record)
    assert isinstance(record.get("detail"), str) and record["detail"], record
    for field in ("max_abs_diff", "tolerance", "reference_max"):
        assert field in record, (f"LAST_PROBE 缺 {field}（无法核对数值）", record)
    if _has_cuda():
        # 有设备时必须推进到规划阶段：证明确实是方程/规划出的错，而不是笼统失败。
        assert record["stage"] == "plan", record
        assert "ValueError" in record["detail"], record


def test_native_failure_is_recorded_verbatim() -> None:
    """原生入口抛错时 ``stage == 'native'``、``detail`` 是**原文**（缺符号那一种）。"""
    if not _has_cuda():
        raise unittest.SkipTest("没有可用的 CUDA 设备，原生阶段无法被触发")

    from ..einsum.cuda import _program

    def exploding(*args, **kwargs):
        raise TypeError(MISSING_SYMBOL)

    saved = _snapshot()
    try:
        kernel._TOPOLOGY_DTYPE_SUPPORT.clear()
        kernel._TOPOLOGY_DTYPE_PROBE.clear()
        kernel.LAST_PROBE.clear()
        with patch.object(_program, "topology_apply", exploding):
            ok = kernel._probe_dtype(np.dtype(np.complex64))
        record = dict(kernel.LAST_PROBE)
    finally:
        _restore(saved)

    assert ok is False, record
    assert record["stage"] == "native", ("必须指出卡在下发阶段", record)
    assert record["detail"] == f"TypeError: {MISSING_SYMBOL}", record


# ── 2. 拒绝原因不得再复述写死文案 ─────────────────────────────────────────────

def test_probe_failure_reason_has_no_fixed_wording() -> None:
    """``_topology_dtype_reason`` 的失败原因里不许出现那句写死文案。

    两种情形都验：有探测事实时（原因逐字引自 ``stage``/``detail``）和没有事实时
    （只能如实说"没有记录"）。任一情形出现固定措辞都算又编了一次原因。
    """
    dtype = np.dtype(np.complex64)
    saved = _snapshot()
    try:
        # (a) 有事实：原因必须带上真实的 stage 与 detail。
        kernel._TOPOLOGY_DTYPE_SUPPORT.clear()
        kernel._TOPOLOGY_DTYPE_PROBE.clear()
        kernel._TOPOLOGY_DTYPE_SUPPORT[dtype] = False
        kernel._TOPOLOGY_DTYPE_PROBE[dtype] = _fake_record(
            dtype.name, False, stage="native", detail=f"TypeError: {MISSING_SYMBOL}")
        with_fact = kernel._topology_dtype_reason(dtype)

        # (b) 没有事实（探测函数被替换）：必须如实说"没有记录"。
        kernel._TOPOLOGY_DTYPE_PROBE.clear()
        with patch.object(kernel, "_probe_dtype", lambda item: False):
            without_fact = kernel._topology_dtype_reason(dtype)
    finally:
        _restore(saved)

    for reason in (with_fact, without_fact):
        assert reason is not None, "探测没过必须给出原因"
        assert FIXED_PHRASE not in reason, (
            "那句写死的『2x2 matmul disagreed with numpy』会误导诊断", reason)
        # 三段既有语义必须保留：契约内但没过 / 无隐式降级 / 重建提示。
        assert "inside the topology dtype contract" in reason, reason
        assert "no implicit downcast" in reason, reason
        assert kernel._REBUILD_HINT in reason, reason
    assert "stage=native" in with_fact, ("有事实时必须点名真实阶段", with_fact)
    assert f"TypeError: {MISSING_SYMBOL}" in with_fact, (
        "有事实时原因必须逐字引用真实异常文本", with_fact)
    assert "no recorded stage/detail" in without_fact, (
        "没有事实时必须如实说没有记录，不许猜一个阶段", without_fact)


# ── 3. probe_dtypes() 覆盖四类 dtype 且不重探 ────────────────────────────────

def test_probe_dtypes_covers_the_whole_contract() -> None:
    """键是 dtype 名字符串、顺序与 ``TOPOLOGY_DTYPES`` 一致、已缓存则不重探。"""
    saved = _snapshot()
    try:
        kernel._TOPOLOGY_DTYPE_SUPPORT.clear()
        kernel._TOPOLOGY_DTYPE_PROBE.clear()
        for dtype in kernel.TOPOLOGY_DTYPES:
            kernel._TOPOLOGY_DTYPE_SUPPORT[dtype] = dtype.name in ("complex128",
                                                                   "float32")
            kernel._TOPOLOGY_DTYPE_PROBE[dtype] = _fake_record(
                dtype.name, kernel._TOPOLOGY_DTYPE_SUPPORT[dtype])

        def must_not_probe(item):                # pragma: no cover - 触发即失败
            raise AssertionError(f"已缓存的 {item} 被重探了")

        with patch.object(kernel, "_probe_dtype", must_not_probe):
            report = kernel.probe_dtypes()
    finally:
        _restore(saved)

    assert isinstance(report, dict), report
    assert tuple(report) == DTYPE_NAMES, ("必须覆盖四类且按契约序", tuple(report))
    assert set(report) == {dtype.name for dtype in kernel.TOPOLOGY_DTYPES}
    for name, record in report.items():
        assert isinstance(name, str), name
        assert isinstance(record, dict), record
        assert record.get("dtype") == name, record
        assert record.get("ok") is (name in ("complex128", "float32")), record
        assert record.get("stage") in kernel.PROBE_STAGES + ("unrecorded",), record


# ── 4. 端到端容差必须复用同一份映射，不许写死 ────────────────────────────────

def test_probe_tolerance_comes_from_the_kernel_mapping() -> None:
    """``_probe.py`` 的对照容差取自 ``_kernel_cuda._PROBE_TOLERANCE``（源码断言）。

    写死 ``atol=1e-6`` 会让 complex128/float64 的对照失去意义（比原生探测宽 1000×），
    并且和 ``_PROBE_TOLERANCE`` 变成两把尺子。
    """
    source = inspect.getsource(_probe)
    assert "1e-6" not in source, (
        "端到端对照容差里不许再有写死的 1e-6", source.count("1e-6"))
    assert "_PROBE_TOLERANCE" in source, "容差必须复用 _kernel_cuda 的那份映射"
    assert "rtol=tolerance" in source or "rtol=" in source, (
        "对照必须按相对容差比较（与原生探测同口径）")

    tolerance = kernel._PROBE_TOLERANCE
    assert set(tolerance) == set(DTYPE_NAMES), (
        "四类 dtype 都要有相对容差（缺一类就会 KeyError）", sorted(tolerance))
    assert tolerance["complex128"] == tolerance["float64"] == 1e-9, tolerance
    assert tolerance["complex64"] == tolerance["float32"] == 1e-4, tolerance


# ── 5. 报告的逐 dtype 结论（有 GPU 才真跑）──────────────────────────────────

def test_gpu_probe_report_carries_per_dtype_conclusions() -> None:
    """``probe()`` 的 ``dtypes`` 是字典，``kernel_status.detail`` 含逐类结论。"""
    if not _has_cuda():
        raise unittest.SkipTest("没有可用的 CUDA 设备，逐 dtype 结论无法实测")
    report = _probe.probe()
    assert isinstance(report.get("dtypes"), dict), report.get("dtypes")
    assert tuple(report["dtypes"]) == DTYPE_NAMES, report["dtypes"]
    detail = report["kernel_status"]["detail"]
    for name in DTYPE_NAMES:
        assert name in detail, ("kernel_status.detail 必须逐 dtype 点名", name, detail)
    assert "=ok" in detail or "=fail(stage=" in detail, detail
    text = _probe.format_report(report)
    assert "dtypes" in text, text


def test_format_report_shows_dtypes_right_after_kernel_status() -> None:
    """``dtypes`` 行插在 ``kernel status`` 之后，既有行不动（无 GPU 也能验）。"""
    report = {
        "build_info": {"backend": "cuda", "native_available": True,
                       "array_backend": "numpy"},
        "cupy": None,
        "device": "synthetic",
        "kernel_enabled": None,
        "kernel_status": {"state": "fallback", "detail": "synthetic detail"},
        "dtypes": {"complex128": _fake_record("complex128", True),
                   "complex64": _fake_record("complex64", False, stage="native",
                                             detail=f"TypeError: {MISSING_SYMBOL}")},
        "verdict": "synthetic verdict",
        "hints": [],
    }
    lines = _probe.format_report(report).splitlines()
    labels = [line.strip().split(":")[0].strip() for line in lines]
    assert labels[:6] == ["GPU probe", "build_info", "cupy", "device",
                          "kernel enabled", "kernel status"], labels
    assert labels[6] == "dtypes", ("dtypes 行必须紧跟 kernel status", labels)
    assert labels[7] == "verdict", labels
    dtypes_line = lines[6]
    assert "complex128=ok" in dtypes_line, dtypes_line
    assert "complex64=fail(stage=native" in dtypes_line, dtypes_line
    assert MISSING_SYMBOL in dtypes_line, dtypes_line


def test_partially_active_verdict_is_honest() -> None:
    """部分 dtype 回落时 verdict 既不说 active 也不说 NOT active。

    本机四类全部通过，所以这条分支只能靠伪造：把 ``probe_dtypes`` 换成"三类 ok、
    complex64 在原生阶段失败"，端到端对照用真实的参考结果顶替（否则验的是回落）。
    """
    if not _has_cuda():
        raise unittest.SkipTest("没有可用的 CUDA 设备，端到端对照无法伪造")

    import cupy

    left = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    right = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.complex128)
    reference = np.einsum(_probe._PROBE_EQUATION, left, right)
    failing = ("complex64",)

    def fake_apply(subscripts, operands, **kwargs):
        return cupy.asarray(reference.astype(np.dtype(operands[0].dtype)))

    with patch.object(kernel, "probe_dtypes",
                      lambda: _fake_dtypes(failing)), \
            patch.object(kernel, "try_topology_apply", fake_apply):
        report = _probe.probe()

    verdict = report["verdict"]
    assert verdict.startswith("CUDA fast path is partially active"), (
        "部分可用时必须如实说 partially active", verdict)
    assert "complex64" in verdict, verdict
    assert "NOT active" not in verdict, ("有 dtype 真的在跑就不许说 NOT active", verdict)
    assert report["kernel_status"]["state"] != "ok", report["kernel_status"]
    detail = report["kernel_status"]["detail"]
    assert "complex128=ok" in detail and "complex64=fail(stage=native" in detail, detail


def test_all_dtypes_failing_keeps_the_original_verdict() -> None:
    """四类全部失败时 verdict 文案**逐字不变**，但 detail 必须逐类点名。"""
    if not _has_cuda():
        raise unittest.SkipTest("没有可用的 CUDA 设备，四类全失败分支无法进入")

    def fake_apply(*args, **kwargs):            # pragma: no cover - 不该被调用
        raise AssertionError("四类全失败时不得下发原生")

    with patch.object(kernel, "probe_dtypes", lambda: _fake_dtypes(DTYPE_NAMES)), \
            patch.object(kernel, "try_topology_apply", fake_apply):
        report = _probe.probe()

    assert report["verdict"] == "CUDA fast path is NOT active — results fall back", (
        report["verdict"])
    detail = report["kernel_status"]["detail"]
    for name in DTYPE_NAMES:
        assert f"{name}=fail(stage=" in detail, (name, detail)


TESTS = (
    test_failed_probe_records_stage_and_detail,
    test_native_failure_is_recorded_verbatim,
    test_probe_failure_reason_has_no_fixed_wording,
    test_probe_dtypes_covers_the_whole_contract,
    test_probe_tolerance_comes_from_the_kernel_mapping,
    test_gpu_probe_report_carries_per_dtype_conclusions,
    test_format_report_shows_dtypes_right_after_kernel_status,
    test_partially_active_verdict_is_honest,
    test_all_dtypes_failing_keeps_the_original_verdict,
)


def test_einsum_gpu_probe_diagnosis_contracts() -> None:
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
            f"{len(failures)} 条 GPU 探测诊断契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("GPU 探测诊断契约全部跳过")


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
