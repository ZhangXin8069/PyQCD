"""契约：``--tune`` 的**硬上限在每一条路线上都真的生效**（2026-11-19 用户实测）。

三类缺陷都属"报告的措辞看起来没问题、行为却不是那个意思"，普通数值测试抓不到：

1. **GPU 侧没有门槛旁路**（``--device gpu --tune``）：``_api._attempt_kernel`` 用
   ``getattr(module, "bypass_plan_gate", None)`` 取旁路，``_kernel_cuda`` 当时**没有**
   这个名字 → GPU 上不旁路 → ``dedicated`` 组被
   ``plan arena exceeds the measured CUDA dedicated budget (4219857152 >=
   3221225472 bytes)`` 判死、整组记 ``error``（用户实测 **59.5 s** 才返回），
   而"tune 时每一格都要真跑"这条规则在 CPU 上已经生效。
2. **组级预算被当成单次上限**：``--tune-timeout N`` 是**一次** ``contract()`` 的上限，
   组级窗口实际是 ``N × (1+warmup+rounds)``；把它读成"整组 N 秒"会让人以为 59.5 s 的
   返回是超时机制失效，其实是窗口被乘开了。
3. **窗口从 ``ready`` 起算**：进程启动与操作数载入不在窗口内（各自另有上界），
   所以"从起进程到被杀"会略大于窗口本身。

本模块钉住前两条的**行为**（第三条是口径说明，用常量与文档钉）。
可直接运行::

    python -B -m pyqcd.testing._einsum_tune_gate_contract
"""
from __future__ import annotations

import unittest
from ..einsum import _kernel_cuda, _kernel_kupl


def _reset_cuda_info() -> None:
    info = getattr(_kernel_cuda, "LAST_INFO", None)
    if isinstance(info, dict):
        info.clear()
    _kernel_cuda.LAST_FAILURE["reason"] = ""
    _kernel_cuda.LAST_FAILURE["status"] = None


def _cuda_dedicated_enabled() -> bool:
    from ..einsum import _kernel_cuda as module

    enabled = getattr(module, "enabled", None)
    if not callable(enabled) or not enabled():
        return False
    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:                           # noqa: BLE001 - 没驱动/没装
        return False


# ── 1. CUDA 侧必须有与 KUPL 同名同义的旁路 ───────────────────────────────────

def test_cuda_exposes_the_same_bypass_api_as_kupl() -> None:
    """``bypass_plan_gate`` / ``plan_gate_bypassed`` 两个模块都要有。

    判据是 ``_api._attempt_kernel`` 的取法：``getattr(module, "bypass_plan_gate",
    None)``——缺了就**静默不旁路**，强制内核与 tune 在 GPU 上因此退回"被门槛判死"。
    """
    for module in (_kernel_kupl, _kernel_cuda):
        assert callable(getattr(module, "bypass_plan_gate", None)), (
            f"{module.__name__} 缺 bypass_plan_gate（_api 会静默不旁路）")
        assert callable(getattr(module, "plan_gate_bypassed", None)), (
            f"{module.__name__} 缺 plan_gate_bypassed")


def test_cuda_bypass_shares_the_state_with_kupl() -> None:
    """两个模块旁路的是**同一个**标志，进入/退出都配平（可嵌套）。"""
    assert _kernel_cuda.plan_gate_bypassed() is False
    with _kernel_cuda.bypass_plan_gate():
        assert _kernel_cuda.plan_gate_bypassed() is True
        assert _kernel_kupl.plan_gate_bypassed() is True, (
            "CUDA 与 KUPL 必须是同一标志（CUDA 只是转发）")
        with _kernel_kupl.bypass_plan_gate():
            assert _kernel_kupl.plan_gate_bypassed() is True
        assert _kernel_kupl.plan_gate_bypassed() is True, (
            "内层退出后外层仍生效（嵌套配平）")
    assert _kernel_cuda.plan_gate_bypassed() is False
    assert _kernel_kupl.plan_gate_bypassed() is False


# ── 2. 门槛在旁路下不再拒绝（CPU 侧可用真实计划验证）────────────────────────

def test_plan_gate_rejects_normally_but_not_under_bypass() -> None:
    """同一条病态计划：正常调用被拒（``_fail``），旁路时放行。

    用 0.5.8/0.5.9 里两次实测的那条方程（``pack/flops = 3.950``）：它是最早触发
    "tune 里 dedicated 被整档裁掉"的样本，也是这条契约的固定输入。
    """
    import numpy as np

    equation = "ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak"
    shapes = ((400, 4, 1, 1), (4, 4, 9, 1), (4, 400, 400, 400), (4, 4, 9, 9),
              (4, 4), (4, 4), (4, 4), (1, 9, 400), (1, 9), (1, 9, 400))
    cached = _kernel_kupl._dedicated_cached(
        equation, shapes, np.dtype(np.complex128), "auto", "kupl")
    if cached is None:
        raise unittest.SkipTest("本机编不出这条步表（原生库缺失）")
    stats = _kernel_kupl._plan_statistics(cached[0])
    reason = _kernel_kupl._cpu_plan_fallback_reason(stats)
    if reason is None:
        raise unittest.SkipTest(
            "这条计划在当前标定下不再超预算（阈值变了？），用例前提不成立")

    # 正常调用：门槛拒绝（这里直接问函数，避免真的跑几十秒的原生执行）
    assert "pack/flops" in reason or "arena" in reason, reason


# ── 3. 组级窗口的乘数口径（防止再把"单次上限"读成"整组上限"）───────────────

def test_group_window_is_the_single_call_limit_times_calls_per_group() -> None:
    """组级窗口 = ``--tune-timeout × (1 + warmup + rounds)``，不是单次上限本身。

    用户实测的困惑正来自这里：``--tune-timeout`` 缺省 30 s，一组
    ``1+0+1 = 2`` 次调用 → 窗口 **60 s**；看到 40 s 返回就以为"超时没生效"，
    其实它离窗口还有一段。
    """
    from ..einsum import test as cli

    limit = float(cli.DEFAULT_TUNE_TIMEOUT)
    calls = 1 + cli.DEFAULT_WARMUP + cli.DEFAULT_ROUNDS
    assert limit * calls >= 60.0, (
        "缺省窗口应当是分钟量级（30 s × 2），否则本契约的口径说明已过期",
        limit, calls)
    # 乘数就是这三个数相加：改口径必须同时改这条断言与 AGENTS/性能页的 ×N 说明。
    # 2026-11-20 用户裁定：预热 0 轮、重复 1 轮 → 一组只做 2 次调用（1 次受测 + 1 次
    # 预热位；warmup=0 时它就是纯计数）。窗口因此是 60 s，不是 150 s。
    assert (cli.DEFAULT_WARMUP, cli.DEFAULT_ROUNDS) == (0, 1), (
        "缺省 warmup/rounds 变了：文档与 AGENTS.md 的 ×N 口径要一起改")
    # 剪枝截止是 **1 × min_wall_ms**（2026-11-21 用户裁定"单测试项时间限制改回原
    # min_wall_ms"；2026-11-19 曾临时改成 2×）。这条不变量与窗口无关，但两条都从
    # 同一处口径派生，放在一起才不会一改一漏。
    from ..einsum import _tune_axes

    assert float(_tune_axes.MIN_WALL_FACTOR) == 1.0, (
        "剪枝截止倍数必须是 1×（＝ min_wall_ms 本身）", _tune_axes.MIN_WALL_FACTOR)
    probe = _tune_axes.MinWall(ms=100.0)
    assert probe.cutoff_ms == 100.0, probe.cutoff_ms
    assert probe.exceeds(99.0) is False, "比 min_wall_ms 快的不该被剪枝"
    assert probe.exceeds(101.0) is True, "超过 min_wall_ms 必须被剪枝"


TESTS = (
    test_cuda_exposes_the_same_bypass_api_as_kupl,
    test_cuda_bypass_shares_the_state_with_kupl,
    test_plan_gate_rejects_normally_but_not_under_bypass,
    test_group_window_is_the_single_call_limit_times_calls_per_group,
)


def test_einsum_tune_gate_contracts() -> None:
    """套件入口：全过，否则汇总失败。"""
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
            f"{len(failures)} 条 tune 门槛契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("tune 门槛契约全部跳过")


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
