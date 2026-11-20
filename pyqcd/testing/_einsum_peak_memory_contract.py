"""契约：``--tune`` 每组都要记录"峰值内存/显存"，并在进下一组前释放上一组的占用。

为什么单独立契约
----------------
这条需求里有三件事**全都不会让数值测试变红**：

1. **重置**：``ru_maxrss`` 是**进程生命周期**的峰值、降不下来。如果拿它直接当
   "本组占用"，第一组之后每一组都会顶着前几组的历史峰值，报告写成"第 7 组吃了
   6.2 GB"而实际上那 6.2 GB 是第 3 组留下的。所以本契约的第 2、3 条专门钉住
   "``reset_peak_memory()`` 之后读数确实变小/归零"与"分配 200 MB 必须读得出来"。
2. **释放**：cupy / torch 的池把显存攥在手里——``used_bytes()``/``memory_allocated()``
   归零**不等于**把显存还给驱动（本机实测：cupy 32 MiB 的块释放后
   ``total_bytes()`` 仍是 32 MiB，直到 ``free_all_blocks()``）。不清池就等于把上一组
   的显存算进下一组，第 4 条盯这一路。
3. **不崩**：测量代码在 tune 的记账路径上，任何一次异常都会让一组配置记账失败。
   第 7 条用 monkeypatch 把 ``resource.getrusage`` 打挂，要求函数仍返回 ``0.0``、
   且原因写进 ``LAST_ERROR``——"失败但不静默"。

七条断言
--------
1. ``peak_memory_mb()`` 返回 ``float`` 且 ``>= 0``，不抛；
2. ``reset_peak_memory()`` 之后读数变小（或为 0）；
3. 分配 200 MB ``uint8`` 后 ``peak_memory_mb()`` 至少涨 100 MB
   （``ru_maxrss`` 历史峰值已经很高时用 ``LAST_NOTE`` 说明并断言"瞬时 RSS 采样峰值"
   这一路——两路都在 ``peak_memory_mb()`` 里取 ``max``，所以断言同一个数即可）；
4. ``release_memory()`` 后 cupy 池归零（无 GPU → ``unittest.SkipTest``）；
5. ``memory_probe_available()`` 的键恰好是 ``{"cpu","cupy","torch"}``，值是 ``bool``；
6. ``format_peak_mb()`` 的三种渲染（含 ``0.0 -> "-"``）；
7. ``LAST_ERROR`` 是 dict；伪造 ``resource.getrusage`` 异常后函数仍返回 ``0.0``
   且原因被记下。

隔离口径：临时文件一律落 ``data/tmp/``（本仓库的临时区），跑完删干净；
本契约**不碰**配置表、不跑 ``contract()``、不依赖机器快慢；单条断言峰值
不超过 400 MB（第 3 条那块 200 MB 的数组用完立刻 ``del`` + ``gc.collect()``）。

可直接运行::

    python -B -m pyqcd.testing._einsum_peak_memory_contract
"""
from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
import unittest
from typing import Any
from unittest.mock import patch

__all__ = ["test_einsum_peak_memory_contracts"]

#: 第 3 条那块数组的大小：200 MB（``uint8`` 1 字节/元素，不必乘 itemsize）。
ALLOC_MB = 200

#: 第 3 条的判定门槛：留 100 MB 余量给"分配器已经持有这么多页、RSS 不涨"的情形。
ALLOC_MIN_GROWTH_MB = 100.0

#: 第 4 条那次 cupy 分配（MiB）：够大到池一定会向驱动要块，又不至于占卡。
CUPY_ALLOC_MB = 48

#: 临时区：``data/tmp/``（相对仓库根），与仓库既有约定一致。
_TMP_ENV = "PYQCD_TEST_TMP"


def _repo_root() -> str:
    """仓库根目录（本文件在 ``<root>/pyqcd/testing/`` 下）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tmp_root() -> str:
    """临时根：``data/tmp/``；不可写时退回系统临时目录（并在报告里能看出来）。"""
    override = os.environ.get(_TMP_ENV)
    candidates = [override] if override else []
    candidates.append(os.path.join(_repo_root(), "data", "tmp"))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = tempfile.mkdtemp(prefix="memprobe-", dir=candidate)
            os.rmdir(probe)
            return candidate
        except Exception:                             # noqa: BLE001 - 换下一个
            continue
    return tempfile.gettempdir()


class _TempDir:
    """``with _TempDir() as path:`` —— 在 ``data/tmp/`` 下建一个用完即删的目录。"""

    def __init__(self, prefix: str = "peakmem-") -> None:
        self._prefix = prefix
        self.path = ""

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix=self._prefix, dir=_tmp_root())
        return self.path

    def __exit__(self, *exc_info: Any) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        return None


def _memory_module():
    """懒取被测模块（避免 import 期就去碰 ``resource`` / CUDA）。"""
    from ..einsum import _memory

    return _memory


def _rss_mb() -> float:
    """当前瞬时 RSS（MB）；拿不到返回 0.0（本契约不依赖它，只用于解释）。"""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:                                 # noqa: BLE001
        return 0.0
    return 0.0


def _cleanup(memory: Any, *extra: Any) -> None:
    """收尾：丢掉本契约造的对象、还池、清峰值（顺序与生产调用一致）。"""
    del extra
    gc.collect()
    memory.release_memory()


# ---------------------------------------------------------------------------
# 1. 返回类型与下界
# ---------------------------------------------------------------------------
def test_peak_memory_mb_is_non_negative_float() -> None:
    """``peak_memory_mb()`` 返回 ``float`` 且 ``>= 0``，且不抛异常。"""
    memory = _memory_module()
    try:
        value = memory.peak_memory_mb()
    except Exception as exc:                          # noqa: BLE001 - 就是断言它不抛
        raise AssertionError(
            f"peak_memory_mb() 抛了 {type(exc).__name__}: {exc}——测量失败必须返回 0.0") from exc
    assert isinstance(value, float), (
        f"peak_memory_mb() 应返回 float，实际 {type(value).__name__}")
    assert value >= 0.0, f"peak_memory_mb() 不该为负：{value}"
    assert value == value, "peak_memory_mb() 返回了 NaN"
    # 未重置过的进程也必须能读出数（不依赖 reset 先被调用）。
    assert isinstance(memory.format_peak_mb(value), str), "format_peak_mb 必须返回 str"


# ---------------------------------------------------------------------------
# 2. 重置生效
# ---------------------------------------------------------------------------
def test_reset_lowers_the_reading() -> None:
    """先撑起一块峰值，重置后读数必须变小（或为 0）——证明重置真的生效。

    这里刻意用一块 64 MB 的 ``uint8``：够大到 ``VmRSS``/``VmHWM`` 一定看得见，
    又不至于把机器打出余量。分配、读峰值、``del``、``gc``，然后重置再读。
    """
    memory = _memory_module()
    previous = memory.peak_memory_mb()
    big = bytearray(64 * 1024 * 1024)
    try:
        grown = memory.peak_memory_mb()
        assert grown >= previous, (
            f"分配 64 MB 后峰值反而变小：{previous} -> {grown}")
    finally:
        del big
        gc.collect()
        memory.release_memory()
    memory.reset_peak_memory()
    after = memory.peak_memory_mb()
    note = "; ".join(f"{k}={v}" for k, v in memory.LAST_NOTE.items()) or "无"
    assert after < grown or after == 0.0, (
        f"reset_peak_memory() 之后读数没有下降：{grown} -> {after}"
        f"（LAST_ERROR={memory.LAST_ERROR}; LAST_NOTE={note}）")


# ---------------------------------------------------------------------------
# 3. 200 MB 分配必须读得出来
# ---------------------------------------------------------------------------
def test_known_allocation_is_visible() -> None:
    """分配 200 MB 后 CPU 这一路至少涨 100 MB，且总峰值不低于该路。

    为什么门槛是 100 而不是 200：``ru_maxrss`` 是**生命周期**峰值，本进程在跑到这条
    断言之前可能已经冲得更高（``import torch`` 一项就 ~540 MB），此时
    ``current_peak - baseline`` 读不出增量。真正能反映"本组涨了多少"的是
    ``VmRSS`` 抽样峰值那一路，所以增量在 **CPU 分量**（``peak_memory_breakdown()
    ["cpu"]``）上量。

    为什么不直接看 ``peak_memory_mb()`` 的总增量：它是各路（CPU / cupy / torch /
    device）的**最大值**。同一个进程里如果还握着 128 MB 的 torch 显存，那条路的
    读数本身就是 128 MB（它是绝对值，不是增量），总增量的基线被抬到 128，200 MB
    的 numpy 分配只能显出 72——这不代表 CPU 那一路没测到，只是被另一路垫高了。
    所以这里量 CPU 分量的增量（这条断言真正要保证的事），再断言总峰值
    ``>=`` CPU 分量（顶层 API 一定不会比它自己的某一路小），并在本来就是干净进程
    （基线 ≈ 0）的常见情形下顺带钉住"总增量也到 100 MB"。
    """
    memory = _memory_module()
    memory.reset_peak_memory()
    baseline_total = memory.peak_memory_mb()
    baseline = memory.peak_memory_breakdown()
    baseline_rss = _rss_mb()
    payload = bytearray(ALLOC_MB * 1024 * 1024)
    # 真写一遍：``bytearray`` 的页可能被延迟映射，只有写了才进 RSS。
    payload[::4096] = b"\x01" * len(payload[::4096])
    try:
        peak = memory.peak_memory_mb()
        breakdown = memory.peak_memory_breakdown()
    finally:
        del payload
        gc.collect()
        memory.release_memory()
    growth = breakdown["cpu"] - baseline["cpu"]
    if growth < ALLOC_MIN_GROWTH_MB:
        raise AssertionError(
            f"分配 {ALLOC_MB} MB 后 CPU 峰值只涨了 {growth:.1f} MB"
            f"（baseline={baseline}, peak={breakdown},"
            f" RSS {baseline_rss:.1f}->{_rss_mb():.1f}MB）"
            f"——瞬时 RSS 采样那一路没抓到这 {ALLOC_MB} MB；"
            f"LAST_ERROR={memory.LAST_ERROR}")
    assert peak >= breakdown["cpu"] - 1e-9, (
        f"peak_memory_mb()={peak} 小于它自己的 CPU 分量 {breakdown['cpu']}"
        "——总峰值必须取各路最大值")
    assert peak >= 0.0
    # 干净进程（总基线 < 1 MB）里，顶层读数本身也必须看到这块分配。
    if baseline_total < 1.0:
        assert peak - baseline_total >= ALLOC_MIN_GROWTH_MB, (
            f"进程本来没有历史峰值，但 peak_memory_mb() 只涨了 "
            f"{peak - baseline_total:.1f} MB（baseline={baseline_total:.2f}MB, "
            f"peak={peak:.1f}MB, breakdown={breakdown}）")
    # 上界自检：本组只分配了这一块，读数不该远超它（留 2 倍 + 256 MB 余量给
    # 解释器/allocator 的固有波动）。超过说明把别的历史峰值算进了本组增量。
    assert peak <= baseline_total + 2 * ALLOC_MB + 256.0, (
        f"峰值 {peak:.1f}MB 远超本次分配的 {ALLOC_MB}MB（baseline={baseline_total:.1f}MB），"
        f"疑似把生命周期历史峰值算成了本组增量；LAST_ERROR={memory.LAST_ERROR}")


# ---------------------------------------------------------------------------
# 4. release_memory 真的还池
# ---------------------------------------------------------------------------
def test_release_returns_cupy_pool() -> None:
    """``release_memory()`` 后 cupy 池的 ``used_bytes()`` 必须归零/下降。

    无 GPU（或 cupy 不可用）时 ``unittest.SkipTest``：这条断言的意义是"把显存还给
    驱动"，没有 GPU 就没有可验的对象，跳过比伪造一个通过更诚实。

    ``used_bytes()`` 只算"正在使用"的块，所以这里刻意先 ``del`` 再 ``release``：
    ``del`` 之后块回到池里（``used_bytes()`` 已归零，但 ``total_bytes()`` 还占着卡），
    ``free_all_blocks()`` 才把 ``total_bytes()`` 也清掉——两个都断言。
    """
    memory = _memory_module()
    available = memory.memory_probe_available()
    if not available.get("cupy"):
        raise unittest.SkipTest("cupy 不可用（无 GPU 或未安装），跳过池释放断言")
    import cupy                                          # noqa: PLC0415 - 有 GPU 才 import

    pool = cupy.get_default_memory_pool()
    errors_before = dict(memory.LAST_ERROR)
    holder: list[Any] = []
    try:
        holder.append(cupy.ones(CUPY_ALLOC_MB * 1024 * 1024 // 8, dtype=cupy.float64))
        cupy.cuda.runtime.deviceSynchronize()
        used_live = pool.used_bytes()
        assert used_live > 0, "cupy 分配之后池的 used_bytes() 仍为 0，这条断言无从验证"
        holder.clear()
        gc.collect()
        used_after_del = pool.used_bytes()
        reserved_after_del = pool.total_bytes()
        memory.release_memory()
        used_after_release = pool.used_bytes()
        reserved_after_release = pool.total_bytes()
    finally:
        holder.clear()
        gc.collect()
        memory.release_memory()
    assert reserved_after_del > 0, (
        "del 之后池的 total_bytes() 已经是 0——这条断言失去意义（分配没走池？）")
    assert used_after_release <= used_after_del, (
        f"release_memory() 后 used_bytes() 没降：{used_after_del} -> {used_after_release}")
    # ``free_all_blocks()`` 的定义是"把**空闲**块还给驱动"，所以判据是
    # "预留量退回到只剩仍在使用的那些"（``total == used``）。不用 ``total == 0``：
    # 同一进程里如果别的用例还握着 cupy 数组，它们本来就该继续占卡。
    assert reserved_after_release == used_after_release, (
        f"release_memory() 后池里仍缓存着空闲块：total_bytes={reserved_after_release} "
        f"!= used_bytes={used_after_release}（free_all_blocks() 没生效？）")
    if used_after_release == 0:
        assert reserved_after_release == 0, (
            f"release_memory() 后 total_bytes() 仍占着显存：{reserved_after_release}")
    # 只断言"这次释放没新增错误"：别的用例（例如第 7 条）可能在同一个进程里
    # 先往账本里记过东西，用绝对空 dict 断言会变成对运行顺序的隐性依赖。
    new_errors = {key: value for key, value in memory.LAST_ERROR.items()
                  if key not in errors_before}
    assert not new_errors, f"释放过程中记了新错误：{new_errors}"


# ---------------------------------------------------------------------------
# 5. 可用性字典的形状
# ---------------------------------------------------------------------------
def test_memory_probe_available_shape() -> None:
    """键恰好 ``{"cpu","cupy","torch"}``，值是 ``bool``（不是 numpy bool 之类）。"""
    memory = _memory_module()
    available = memory.memory_probe_available()
    assert isinstance(available, dict), (
        f"memory_probe_available() 应返回 dict，实际 {type(available).__name__}")
    assert set(available) == {"cpu", "cupy", "torch"}, (
        f"键集合不对：{sorted(available)}")
    for key, value in available.items():
        assert type(value) is bool, (
            f"{key} 的值必须是 bool，实际 {type(value).__name__}={value!r}")
    assert available["cpu"] is True, (
        f"本机 CPU 这一路必须可用（POSIX 的 getrusage 一定在）：{available}")
    # 第二次调用必须走缓存、不抛、结果一致（torch 的 import 只该发生一次）。
    again = memory.memory_probe_available()
    assert again == available, f"两次探测结果不一致：{available} vs {again}"


# ---------------------------------------------------------------------------
# 6. 渲染格式
# ---------------------------------------------------------------------------
def test_format_peak_mb() -> None:
    """``format_peak_mb``：整数、无小数、无千分位；``0.0 -> "-"``。"""
    memory = _memory_module()
    cases = {
        0.0: "-",
        1234.4: "1234MB",
        1e6: "1000000MB",
        0.4: "0MB",
        12.5: "12MB",
        2048.0: "2048MB",
        -3.0: "-",
        float("nan"): "-",
    }
    for value, expected in cases.items():
        got = memory.format_peak_mb(value)
        assert got == expected, f"format_peak_mb({value!r}) = {got!r}，期望 {expected!r}"
    # 机器可读：不含空格、千分位逗号、小数点。
    rendered = memory.format_peak_mb(1234567.89)
    assert rendered.endswith("MB") and rendered[:-2].isdigit(), (
        f"渲染不是 <整数>MB 形式：{rendered!r}")


# ---------------------------------------------------------------------------
# 7. 失败不抛、但要记账
# ---------------------------------------------------------------------------
def test_failure_is_recorded_not_raised() -> None:
    """打挂 ``resource.getrusage``：函数仍返回 ``0.0`` 量级，且原因进 ``LAST_ERROR``。

    打挂的必须是**计时路径上真会调用**的那一个：``peak_memory_mb()`` 经
    ``_cpu_peak_locked`` 调 ``_maxrss_mb`` → ``resource.getrusage``。这里把
    ``resource.getrusage`` 换成抛 ``OSError``，然后要求：

    * 不抛异常；
    * 返回 ``float`` 且 ``>= 0``（``VmRSS`` 那一路还能工作，所以不强制等于 0——
      真正要钉的是"没崩"与"记了账"）；
    * ``LAST_ERROR`` 是 dict，且有一个键的值里带 ``OSError`` 与完整的异常信息。
    """
    memory = _memory_module()
    assert isinstance(memory.LAST_ERROR, dict), (
        f"LAST_ERROR 必须是 dict，实际 {type(memory.LAST_ERROR).__name__}")

    def boom(*args: Any, **kwargs: Any):
        raise OSError("synthetic getrusage failure for the peak-memory contract")

    memory.reset_peak_memory()
    with patch("resource.getrusage", boom):
        try:
            value = memory.peak_memory_mb()
            breakdown = memory.peak_memory_breakdown()
        except Exception as exc:                      # noqa: BLE001 - 就是断言它不抛
            raise AssertionError(
                f"getrusage 失败时被测函数抛了 {type(exc).__name__}: {exc}"
                "——测量失败必须降级为 0.0 而不是让 tune 崩") from exc
    assert isinstance(value, float) and value >= 0.0, (
        f"降级后返回值不是非负 float：{value!r}")
    assert isinstance(breakdown, dict) and set(breakdown) >= {"cpu", "total"}, (
        f"peak_memory_breakdown() 的分路结构不对：{breakdown}")
    recorded = dict(memory.LAST_ERROR)
    assert recorded, "getrusage 被打挂了，但 LAST_ERROR 里什么都没记（静默失败）"
    assert any("OSError" in str(message) and "synthetic getrusage failure" in str(message)
               for message in recorded.values()), (
        f"LAST_ERROR 里没有那条 OSError 的原因：{recorded}")
    assert any("1024" in str(message) for message in recorded.values()), (
        f"LAST_ERROR 的原因里没写单位口径（MB = 1024² 字节）：{recorded}")
    # 恢复之后必须能重新读出真实的 CPU 峰值（打挂只影响那一次调用）。
    memory.reset_peak_memory()
    restored = memory.peak_memory_mb()
    assert isinstance(restored, float) and restored >= 0.0, (
        f"getrusage 恢复后仍读不出峰值：{restored!r}")


# ---------------------------------------------------------------------------
# 额外：临时目录口径（``data/tmp/`` 下落盘、跑完清理）
# ---------------------------------------------------------------------------
def test_tmp_area_is_under_data_tmp() -> None:
    """临时文件落 ``data/tmp/``（仓库临时区），离开 ``with`` 后目录必须消失。

    本契约本身不需要落盘，但"在 ``data/tmp/`` 下落临时文件、跑完清理"是任务口径，
    这里用一个 1 KiB 的探针文件证明这条口径成立（也证明该目录可写）。
    """
    root = _tmp_root()
    data_tmp = os.path.join(_repo_root(), "data", "tmp")
    if os.path.abspath(root) != os.path.abspath(data_tmp):
        raise unittest.SkipTest(
            f"data/tmp/ 不可写，临时区退回 {root}（本条不适用）")
    with _TempDir() as path:
        assert os.path.abspath(path).startswith(os.path.abspath(data_tmp) + os.sep), (
            f"临时目录不在 data/tmp/ 下：{path}")
        probe = os.path.join(path, "probe.txt")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("x" * 1024)
        assert os.path.getsize(probe) == 1024
        kept = path
    assert not os.path.exists(kept), f"临时目录没被清理：{kept}"


TESTS = (
    test_peak_memory_mb_is_non_negative_float,
    test_reset_lowers_the_reading,
    test_known_allocation_is_visible,
    test_release_returns_cupy_pool,
    test_memory_probe_available_shape,
    test_format_peak_mb,
    test_failure_is_recorded_not_raised,
    test_tmp_area_is_under_data_tmp,
)


# ---------------------------------------------------------------------------
# 汇总入口
# ---------------------------------------------------------------------------
def _run() -> tuple[int, int, list[str]]:
    """跑全部断言；返回 ``(passed, skipped, failures)``。"""
    passed = skipped = 0
    failures: list[str] = []
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except Exception as exc:                      # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    return passed, skipped, failures


def test_einsum_peak_memory_contracts() -> None:
    """``--tune`` 每组峰值内存/显存测量的七条契约（另加一条临时区口径）。

    失败时抛 :class:`AssertionError` 并列出每条失败；全部跳过时抛
    :class:`unittest.SkipTest`（不留一个"看起来通过"的空套件）。

    收尾把测量模块的内部状态**还原到干净态**：``release_memory()`` 停掉采样线程，
    ``reset_peak_memory()`` 重新立基线并清空 ``LAST_ERROR`` / ``LAST_NOTE``——否则
    本契约在第 7 条伪造的那条 ``OSError`` 会跟着留在账本里，下一组 tune 报告会莫名
    多出一行错误。
    """
    memory = _memory_module()
    started = time.perf_counter()
    passed, skipped, failures = _run()
    try:
        _cleanup(memory)
        memory.reset_peak_memory()
        memory.release_memory()
    except Exception as exc:                          # noqa: BLE001 - 收尾失败也要报
        failures.append(f"teardown: {type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started
    if failures:
        raise AssertionError(
            f"{len(failures)} 条峰值内存契约失败（{passed} passed, {skipped} skipped, "
            f"{elapsed:.2f}s）：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"峰值内存契约全部跳过（{skipped} skipped）：没有可验证的来源")


def main() -> int:
    """独立运行入口：``python -B -m pyqcd.testing._einsum_peak_memory_contract``。"""
    passed, skipped, failures = _run()
    print(f"{passed} passed, {skipped} skipped, {len(failures)} failed", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
