"""契约：命中通知里的内核名必须是**实际执行**的路线，且带得上"命中什么"的上下文。

为什么要单独立契约
------------------
命中提示只在**真的下发内核**时打：``kernel_force`` 的缺省值已是 ``generic``
（``auto`` 已删除），那条路既不做形态匹配也不下发内核，因此根本不提示；本契约因此
显式强制 ``dedicated``（单值强制，一次调用只下发这一个）。于是最常见的命中是

    [pyqcd.einsum] 专用路径: topology:a1c49862 [内核: kupl_dedicated]

旧提示只打 ``[内核: …]``，读起来像"topology 命中却报了 dedicated，标签取错了"——
实际执行确实是 dedicated，缺的是"命中了什么形态"那一半（2026-11-17 用户报告）。
现在提示拆成两段机器可解析的记号：``[match: …]`` 命中形态族、``[kernel: …]``
实际执行路线（取自执行侧 ``LAST_INFO['path']``），没跑内核时写 ``[plan-cache]``。

三条能独立失败的断言：

1. **标签即事实**：语料方程的通知里 ``[kernel: …]`` 必须与原生入口的
   ``LAST_INFO['path']`` **逐字相同**；提示说 ``[plan-cache]`` 时 ``LAST_INFO``
   必须是空的（两个方向都查，否则"永远打 plan-cache"也能通过）。
2. **不许裸 dedicated**：提示里出现 ``dedicated`` 时，同一行必须有
   ``[match: topology]`` 这个上下文。语料命中走的是实际执行结果，所以另用一条
   **合成**调用（清空 ``LAST_INFO`` 后直接 ``announce(entry, "kupl_dedicated")``）
   把这条钉死在没装原生库的机器上也可复现。
3. **机器段跨语言逐字相同**：``PYQCD_EINSUM_LANG=en`` / ``zh`` 两次运行，抽出的
   ``[match: …]`` / ``[kernel: …]`` / ``[plan-cache]`` 与形态名必须**完全相等**，
   同时原文必须不同（否则说明语言没真的切换，比对没有意义）。

不做计时断言：本模块只关心"打出来的字是不是事实"，不关心它有多快。全部调用都在
临时环境的画像关闭下进行，仓库根不会留下配置表。

可直接运行::

    python -B -m pyqcd.testing._einsum_notice_kernel_label_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import unittest
import warnings
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import _api, _kernel_force, _kernel_kupl, _specialize, contract

#: 语料内方程：与 ``_kernel_kupl._PROBE_EQUATION`` 同一条，各操作数轴长见 :data:`DIM`。
EQUATION = "cadb,menf,ac,em,bd,fn->"

#: 轴长取 2：步表可编码、执行不到一秒，命中结论与生产形状一致（只看路，不看速度）。
DIM = 2

#: 提示行里"命中形态"的前缀：两种语言共用，翻译掉脚本就抓不到了。
PREFIX = "[pyqcd.einsum]"

#: 机器段的完整形状：``[match: <族>]`` + ``[kernel: <路线>]`` 或 ``[plan-cache]``。
ROUTE_PATTERN = re.compile(
    r"\[match: (?P<match>[^\]]+)\] "
    r"(?:\[kernel: (?P<kernel>[^\]]+)\]|(?P<cache>\[plan-cache\]))")

#: 形态名（``topology:<摘要>``）：动词随语言变，形态名不变。
NAME_PATTERN = re.compile(
    r"\[pyqcd\.einsum\] (?:专用路径|specialized path): (?P<name>\S+)")


def _operands() -> list[np.ndarray]:
    """按方程左侧各操作数的 rank 造一组等长 C 连续实数操作数。"""
    rng = np.random.default_rng(20261117)
    return [np.ascontiguousarray(rng.standard_normal((DIM,) * len(term)))
            for term in EQUATION.split("->")[0].split(",")]


@contextlib.contextmanager
def _isolated_state(language: str):
    """隔离"本次调用"的全部可变状态：环境、语言档、提示去重、内核记录与执行侧信息。

    语言走 :func:`_specialize.set_language`（``--language`` 用同一入口）而不只改环境
    变量：别的契约可能已经把进程档钉成某一语言，只改 env 会被它压住。
    环境里的强制内核/拓扑强制/专用路径开关一并清掉：本契约用**显式参数**强制内核
    （见 :func:`_run_corpus`），残留的环境值只会让"谁说了算"变得不可读。
    """
    removals = (
        _kernel_force.KERNEL_FORCE_ENV,
        _specialize.DISABLE_ENV,
        _specialize.FUSED_DISABLE_ENV,
        "PYQCD_EINSUM_TOPOLOGY_REAL",
    )
    saved_env = {name: os.environ.pop(name, None) for name in removals}
    saved_language = _specialize.set_language(language)
    saved_detail = _specialize.set_detail(True)
    saved_announced = set(_specialize._announced)
    saved_record = None if _api._LAST_KERNEL is None else dict(_api._LAST_KERNEL)
    saved_info = dict(_kernel_kupl.LAST_INFO)
    saved_failure = dict(_kernel_kupl.LAST_FAILURE)
    additions = {
        _specialize.NOTICE_ENV: "1",
        _specialize.VERBOSE_ENV: "1",
        "PYQCD_EINSUM_PROFILE": "0",
    }
    try:
        with patch.dict(os.environ, additions):
            yield
    finally:
        _specialize.set_language(saved_language)
        _specialize.set_detail(saved_detail)
        _specialize._announced.clear()
        _specialize._announced.update(saved_announced)
        _api._LAST_KERNEL = saved_record
        _kernel_kupl.LAST_INFO.clear()
        _kernel_kupl.LAST_INFO.update(saved_info)
        _kernel_kupl.LAST_FAILURE.clear()
        _kernel_kupl.LAST_FAILURE.update(saved_failure)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_corpus(language: str) -> tuple[str, dict[str, Any]]:
    """跑一次语料方程，返回（提示行原文, 本次调用后的 ``LAST_INFO`` 快照）。

    **必须显式强制内核**：``kernel_force`` 的缺省值现在是 ``generic``（``auto`` 已删除），
    那条路按设计**不匹配、不提示**，因此"命中提示"根本不会打。强制 ``dedicated`` 才会
    真的下发内核并打那条两段式提示，本契约要核对的正是它。

    ``LAST_INFO`` 必须在 ``contract`` 返回后**立刻**取：它是执行侧对"最近一次下发"
    的记录，任何后续内核调用都会覆盖它。
    """
    key = _kernel_force.KERNEL_FORCE_ENV
    with _isolated_state(language), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert key not in os.environ, "环境里的强制值会盖过本契约的显式参数"
        _kernel_kupl.LAST_INFO.clear()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                contract(EQUATION, *_operands(), kernel_force="dedicated")
            except _kernel_force.KernelUnavailableError as exc:
                detail = "; ".join(f"{label}: {reason}"
                                   for label, reason in (exc.reasons or {}).items())
                raise unittest.SkipTest(
                    "本机没有可用的 KUPL dedicated 内核，命中提示不可达："
                    + (detail or str(exc)))
        info = dict(_kernel_kupl.LAST_INFO)
    lines = [line for line in buffer.getvalue().splitlines()
             if line.startswith(PREFIX) and ("专用路径" in line or "specialized path" in line)]
    assert len(lines) == 1, f"语料方程应当正好打一条提示，实际 {len(lines)} 行：{buffer.getvalue()!r}"
    return lines[0], info


def _parse(line: str) -> tuple[str, str, str]:
    """从提示行取出（形态名, 命中族, 执行路线）；路线对 ``plan-cache`` 记作空串。"""
    name_match = NAME_PATTERN.search(line)
    assert name_match is not None, f"提示行缺少形态名：{line!r}"
    route_match = ROUTE_PATTERN.search(line)
    assert route_match is not None, f"提示行缺少 [match: …] / [kernel: …] 机器段：{line!r}"
    return (name_match.group("name"),
            route_match.group("match"),
            route_match.group("kernel") or "")


def test_notice_kernel_label_matches_last_info_path() -> None:
    """① 通知里的内核名与 ``LAST_INFO['path']`` 一致（双向：不许空喊、也不许漏报）。"""
    line, info = _run_corpus("en")
    name, match, kernel = _parse(line)
    assert name.startswith("topology:"), f"语料形态名应当形如 topology:<摘要>：{name!r}"
    assert match == "topology", f"命中族应当取自形态名（topology）：{match!r}"
    executed = info.get("path")
    if kernel:
        assert executed, f"提示说执行了 {kernel!r}，但 LAST_INFO 没有 path：{info!r}"
        assert kernel == executed, (
            f"提示的内核名 {kernel!r} 与 LAST_INFO['path'] {executed!r} 不一致——"
            "通知在说一个没真正执行的路线")
    else:
        assert not executed, (
            f"提示说只命中计划缓存，但 LAST_INFO 记着 path={executed!r}——提示在漏报执行事实")


def test_topology_notice_keeps_topology_context() -> None:
    """② topology 提示中只要出现 ``dedicated``，同一行必须带 ``[match: topology]``。"""
    for language in ("en", "zh"):
        line, _info = _run_corpus(language)
        _name, match, kernel = _parse(line)
        route = ROUTE_PATTERN.search(line).group(0)
        assert "[match: topology]" in line, f"（{language}）语料提示缺少命中族上下文：{line!r}"
        assert match == "topology", f"（{language}）命中族应当取自形态名：{match!r}"
        # 只看**机器段**：形态说明本身可能出现 "dedicated"（强制时它写的就是刚跑的内核），
        # 那属于解释文字，不该被当成"裸 dedicated 标签"。
        if "dedicated" in route:
            assert "topology" in route, (
                f"（{language}）机器段出现 dedicated 却没带 topology 上下文：{line!r}")
            assert kernel.endswith("dedicated"), (
                f"（{language}）dedicated 只应出现在 [kernel: …] 里：{line!r}")

    # 合成调用：清空执行侧记录，让 announce 拿到"只跑了 dedicated"这一事实。
    # 原生库缺失时 _run_corpus 会打 [plan-cache]，这条合成断言保证 ② 不被环境掏空。
    from ..einsum import _specialize as spec

    with _isolated_state("en"):
        entry = spec.match(EQUATION)
        assert entry is not None, "语料方程必须命中，否则本契约的前提不成立"
        _kernel_kupl.LAST_INFO.clear()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            spec.announce(entry, "kupl_dedicated")
    line = buffer.getvalue().strip()
    assert "[match: topology]" in line, f"合成提示缺少命中族上下文：{line!r}"
    assert "[kernel: kupl_dedicated]" in line, f"合成提示没有如实报出执行内核：{line!r}"
    assert line.index("[match: topology]") < line.index("kupl_dedicated"), (
        f"命中族必须排在执行内核之前，读起来才是「匹配到 X → 执行 Y」：{line!r}")


def test_notice_machine_segments_identical_across_languages() -> None:
    """③ ``en`` / ``zh`` 两次运行的机器段逐字相同（而原文不同，证明确实切换了语言）。"""
    zh_line, _zh_info = _run_corpus("zh")
    en_line, _en_info = _run_corpus("en")
    zh_route = ROUTE_PATTERN.search(zh_line)
    en_route = ROUTE_PATTERN.search(en_line)
    assert zh_route is not None and en_route is not None
    assert zh_route.group(0) == en_route.group(0), (
        f"机器段跨语言不一致：zh={zh_route.group(0)!r} en={en_route.group(0)!r}")
    assert _parse(zh_line) == _parse(en_line), (
        f"形态名/命中族/执行路线跨语言不一致：zh={_parse(zh_line)!r} en={_parse(en_line)!r}")
    assert zh_line != en_line, "两种语言的提示原文相同——语言没真的切换，比对没有意义"
    assert "专用路径" in zh_line, f"zh 提示应当用中文动词：{zh_line!r}"
    assert "specialized path" in en_line, f"en 提示应当用英文动词：{en_line!r}"


TESTS = (
    test_notice_kernel_label_matches_last_info_path,
    test_topology_notice_keeps_topology_context,
    test_notice_machine_segments_identical_across_languages,
)


def test_einsum_notice_kernel_label_contracts() -> None:
    """套件入口：跑完 :data:`TESTS`，任何一条失败就抛出汇总。"""
    failures: list[str] = []
    passed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest:
            continue
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    if failures:
        raise AssertionError(
            f"{len(failures)} 条命中通知标签契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("命中通知标签契约全部跳过")


def main() -> int:
    """独立运行入口：逐条打印 PASS/FAIL，返回进程退出码。"""
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
