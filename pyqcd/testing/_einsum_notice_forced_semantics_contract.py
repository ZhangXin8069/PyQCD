"""契约：命中提示的形态说明必须与 ``kernel_force`` 的**单值强制**语义一致。

为什么要单独立契约
-----------------
``kernel_force`` 现在只剩**强制值**（``backend`` / ``dedicated`` / ``generic`` /
``topology``，缺省 ``generic``；``auto`` 已删除、逗号链式已取消），所以一次调用**至多**
下发一个专用内核：默认 ``generic`` 时一个都不试，强制了哪个就只试哪个。旧文案还是
``auto`` 时代的说法——"先试展开式 dedicated 步表、失败才试拓扑步表"——它同时说错两件
事：默认调用根本不试专用内核，强制调用也不会连试两个（2026-11-18 用户报告）。

现在条目里只写"命中了什么形态"，强制与否由 ``_specialize._execution_semantics`` 按
**现场事实**补上，分三档：

============  ==================================================================
档            触发事实（``announce`` 看得到的）
============  ==================================================================
``forced``    ``kernel`` 非空：被强制的那条真的执行了（``_api`` 唯一带标签调它的路）
``generic``   ``kernel`` 为空且 ``_LAST_KERNEL['requested']`` 为空：只是形态命中
``fallback``  ``kernel`` 为空而 ``requested`` 非空：强制没跑成，已回落通用路径
============  ==================================================================

四条能独立失败的断言：

1. **默认 ``generic`` 不许出现"自动尝试"的说法**：真实调用里语料方程**不**打专用路径
   提示（``generic`` 分支不匹配、不下发内核）；任何实际出现的一行都不得含
   "先试 / 失败才试"这类措辞，直接 ``announce(entry, "")`` 时必须写"未强制内核"。
2. **画像建议 ``generic`` 的那一行**（真实打印路径）：命中形态但执行通用，说明必须写
   "仅形态匹配"，执行路线是 ``[plan-cache]``。
3. **强制 ``dedicated`` / ``topology``**：提示点名刚执行的内核
   （``kupl_dedicated`` / ``kupl_topology``…），并明确是**强制**语义。
4. **机器段跨语言逐字相同**：``zh`` / ``en`` 三档里形态名、``[match: …]`` /
   ``[kernel: …]`` / ``[plan-cache]`` 完全相等，而原文必须不同（否则语言没切换，
   比对没有意义）。

补充一条（不在四项内，覆盖需求里的第三档文案）：强制失败回落的提示必须写
"已回落 generic"，且不得声称某个内核执行过——那一行只有 ``[plan-cache]``。

不做计时断言：本契约只关心"打出来的字是不是事实"。全部调用都在临时环境的画像关闭下
进行（画像那一项用临时目录显式开启并自建表），仓库根不会留下配置表。

可直接运行::

    python -B -m pyqcd.testing._einsum_notice_forced_semantics_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import tempfile
import unittest
import warnings
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import (
    _api,
    _kernel_force,
    _kernel_kupl,
    _profile_cache,
    _specialize,
    contract,
)

#: 语料内方程：与 ``_kernel_kupl._PROBE_EQUATION`` 同一条（写一次，不改字母）。
EQUATION = "cadb,menf,ac,em,bd,fn->"

#: 轴长取 2：步表可编码、执行不到一秒，命中结论与生产形状一致（只看路，不看速度）。
DIM = 2

#: 提示行前缀：两种语言共用，翻译掉脚本就抓不到了。
PREFIX = "[pyqcd.einsum]"

#: 旧文案的措辞：默认 ``generic`` 那条路上**一条都不许出现**。
STALE = ("先试", "失败才试", "tries the expanded", "then the v3 topology")

#: 强制档**不许**出现的负向说明（"没有自动尝试"那类）。用户 2026-11-18 裁定：
#: 提示只写实际执行的路线，解释"代码没有尝试什么"是多余输出，已从
#: ``_specialize._NOTE_ZH/_NOTE_EN["forced"]`` 删除；本契约反向钉住它别再回来。
NEGATIVE_META = ("自动尝试", "automatic", "未做", "did not attempt")

#: 形态名（``topology:<摘要>``）：动词随语言变，形态名不变。
NAME_PATTERN = re.compile(
    r"\[pyqcd\.einsum\] (?:专用路径|specialized path): (?P<name>\S+)")

#: 机器段的完整形状：``[match: <族>]`` + ``[kernel: <路线>]`` 或 ``[plan-cache]``。
ROUTE_PATTERN = re.compile(
    r"\[match: (?P<match>[^\]]+)\] "
    r"(?:\[kernel: (?P<kernel>[^\]]+)\]|(?P<cache>\[plan-cache\]))")


def _operands() -> list[np.ndarray]:
    """按方程左侧各操作数的 rank 造一组等长 C 连续实数操作数。"""
    rng = np.random.default_rng(20261118)
    return [np.ascontiguousarray(rng.standard_normal((DIM,) * len(term)))
            for term in EQUATION.split("->")[0].split(",")]


@contextlib.contextmanager
def _isolated_state(language: str, *, profile: bool = False,
                    topology_real: bool = False):
    """隔离"本次调用"的全部可变状态：环境、语言档、提示去重、内核记录与配置表目录。

    语言走 :func:`_specialize.set_language`（``--language`` 用同一入口）而不只改环境
    变量：别的契约可能已经把进程档钉成某一语言，只改 env 会被它压住。
    环境里的强制内核/专用路径开关一并清掉：本契约用**显式参数**强制内核，
    残留的环境值只会让"谁说了算"变得不可读。
    """
    removals = (
        _kernel_force.KERNEL_FORCE_ENV,
        _specialize.DISABLE_ENV,
        _specialize.FUSED_DISABLE_ENV,
        _profile_cache.PROFILE_ENV,
        _profile_cache.DIR_ENV,
    )
    saved_env = {name: os.environ.pop(name, None) for name in removals}
    saved_language = _specialize.set_language(language)
    saved_detail = _specialize.set_detail(True)
    saved_announced = set(_specialize._announced)
    saved_record = None if _api._LAST_KERNEL is None else dict(_api._LAST_KERNEL)
    saved_info = dict(_kernel_kupl.LAST_INFO)
    saved_failure = dict(_kernel_kupl.LAST_FAILURE)
    saved_dir = _profile_cache._dir_text()
    additions = {
        _specialize.NOTICE_ENV: "1",
        _specialize.VERBOSE_ENV: "1",
        _profile_cache.PROFILE_ENV: "1" if profile else "0",
    }
    if topology_real:
        additions["PYQCD_EINSUM_TOPOLOGY_REAL"] = "1"
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
        _profile_cache.configure(saved_dir)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _capture(call: Any, *args: Any, **kwargs: Any) -> str:
    """在提示语言/去重状态已隔离的前提下跑一次，返回 stdout 全文。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        call(*args, **kwargs)
    return buffer.getvalue()


def _notices(text: str) -> list[str]:
    """抽出提示行（``[pyqcd.einsum]`` 前缀 + 专用路径动词）。"""
    return [line for line in text.splitlines()
            if line.startswith(PREFIX)
            and ("专用路径" in line or "specialized path" in line)]


def _parse(line: str) -> tuple[str, str, str]:
    """从提示行取出（形态名, 命中族, 执行路线）；``plan-cache`` 记作空串。"""
    name_match = NAME_PATTERN.search(line)
    assert name_match is not None, f"提示行缺少形态名：{line!r}"
    route_match = ROUTE_PATTERN.search(line)
    assert route_match is not None, f"提示行缺少 [match: …] / [kernel: …] 机器段：{line!r}"
    return (name_match.group("name"),
            route_match.group("match"),
            route_match.group("kernel") or "")


def _run_contract(language: str, *, kernel_force: Any = None,
                  profile_dir: str | None = None,
                  topology_real: bool = False
                  ) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """跑一次 ``contract(EQUATION, ...)``，返回（stdout 全文, ``LAST_INFO``, 内核记录）。

    强制内核在本机不可用时抛 ``SkipTest``：那是环境缺失（原生库/策略拒绝），不是文案
    错误，不该把契约判成失败。``LAST_INFO`` 与 :func:`_api.last_kernel` 必须在
    ``contract`` 返回后**立刻**取：它们是"最近一次调用"的记录，任何后续内核调用都会
    覆盖它，而状态隔离一退出就会把它们复原。
    """
    with _isolated_state(language, profile=profile_dir is not None,
                         topology_real=topology_real):
        _kernel_kupl.LAST_INFO.clear()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                contract(EQUATION, *_operands(), kernel_force=kernel_force,
                         profile=profile_dir is not None, profile_dir=profile_dir)
            except _kernel_force.KernelUnavailableError as exc:
                detail = "; ".join(f"{label}: {reason}"
                                   for label, reason in (exc.reasons or {}).items())
                raise unittest.SkipTest(
                    f"本机没有可用的 {kernel_force} 内核，命中提示不可达："
                    + (detail or str(exc)))
        info = dict(_kernel_kupl.LAST_INFO)
        record = _api.last_kernel() or {}
    return buffer.getvalue(), info, record


def _profile_key() -> dict[str, str]:
    """按 ``_api`` 同一口径算出画像键（形状/dtype/设备/方程/后端）。"""
    operands = _operands()
    device, _user = _api._profile_devices(operands, "numpy")
    return {
        "shape": _profile_cache.format_shape([o.shape for o in operands]),
        "dtype": _api._profile_dtype(operands),
        "device": device,
        "equation": "".join(EQUATION.split()),
        "backend": "numpy",
    }


@contextlib.contextmanager
def _profile_generic_table():
    """临时配置表：只含"``kernel=generic`` 胜出"这一行；退出即复原目录并删除。

    这张表是**唯一**能真实打印"命中语料形态但没强制内核"那条提示的路径：调用方不传
    ``kernel_force`` 时 ``_api`` 的 ``generic_only`` 分支会直接跳过匹配；只有画像建议
    ``generic``（``forced_by_profile``）才会匹配并提示，而执行仍走通用路线。

    写表必须在 ``PYQCD_EINSUM_PROFILE=1`` 下进行：测试入口（``conftest.main``）为了
    不污染仓库根，整个进程把总开关设成 ``0``，而 :func:`_profile_cache.write_winner`
    在总开关关闭时**静默不写**。无论成败都要把配置表目录复原，否则后面依赖"缺表提示"
    的契约会读到半张表或一个已删除的目录。
    """
    saved_dir = _profile_cache._dir_text()
    directory = tempfile.mkdtemp(prefix="pyqcd-notice-forced-")
    try:
        with patch.dict(os.environ, {_profile_cache.PROFILE_ENV: "1"}):
            _profile_cache.configure(directory)
            _profile_cache.write_winner(_profile_key(),
                                        {"kernel": "generic", "wall_ms": 1.0})
            assert os.path.exists(_profile_cache.profile_path()), (
                "临时配置表没写出来，本契约的前提不成立")
        yield directory
    finally:
        _profile_cache.configure(saved_dir)
        shutil.rmtree(directory, ignore_errors=True)


def test_default_generic_notice_has_no_automatic_attempt_wording() -> None:
    """① 默认 ``generic`` + 语料命中：真实调用不提示，任何一行都不许说"自动尝试"。"""
    text, info, record = _run_contract("zh")
    assert _notices(text) == [], (
        f"默认 kernel_force=generic 时不匹配、不下发内核，不该有专用路径提示：{text!r}")
    for line in text.splitlines():
        for stale in STALE:
            assert stale not in line, f"默认 generic 的提示里出现了过时措辞 {stale!r}：{line!r}"
    assert info == {}, f"默认 generic 不该下发任何内核，LAST_INFO 应为空：{info!r}"
    assert not record["requested"], (
        f"默认 generic 的 requested 必须为空（否则提示会误判成强制后回落）：{record!r}")

    # 直接调 announce（``_api`` 里唯一的空标签调用点）：必须写"未强制"而不是"自动尝试"。
    entry = _specialize.match(EQUATION)
    assert entry is not None, "语料方程必须命中，否则本契约的前提不成立"
    with _isolated_state("zh"):
        _api._LAST_KERNEL = {"requested": (), "ran": None, "attempted": (), "reasons": {}}
        _kernel_kupl.LAST_INFO.clear()
        zh_line = _notices(_capture(_specialize.announce, entry, ""))
    with _isolated_state("en"):
        _api._LAST_KERNEL = {"requested": (), "ran": None, "attempted": (), "reasons": {}}
        _kernel_kupl.LAST_INFO.clear()
        en_line = _notices(_capture(_specialize.announce, entry, ""))
    assert len(zh_line) == 1 and len(en_line) == 1, (zh_line, en_line)
    assert "未强制内核" in zh_line[0], f"zh 提示必须写明没强制内核：{zh_line[0]!r}"
    assert "no kernel forced" in en_line[0], f"en 提示必须写明没强制内核：{en_line[0]!r}"
    assert "[plan-cache]" in zh_line[0], f"没跑内核时必须写 plan-cache：{zh_line[0]!r}"
    for line in (zh_line[0], en_line[0]):
        for stale in STALE:
            assert stale not in line, f"提示里仍有过时措辞 {stale!r}：{line!r}"

    # 源码层面也钉死：过时措辞不该再回到这个模块里。
    source = os.path.join(os.path.dirname(_specialize.__file__), "_specialize.py")
    with open(source, encoding="utf-8") as handle:
        body = handle.read()
    for stale in ("先试", "失败才试"):
        assert stale not in body, f"_specialize.py 里仍有过时措辞 {stale!r}"


def test_profile_generic_notice_says_shape_match_only() -> None:
    """② 画像建议 ``generic`` 的那一行：命中形态但执行通用（真实打印路径）。"""
    with _profile_generic_table() as directory:
        zh_text, zh_info, _zh_record = _run_contract("zh", profile_dir=directory)
        en_text, en_info, _en_record = _run_contract("en", profile_dir=directory)
    zh_line = _notices(zh_text)
    en_line = _notices(en_text)
    assert len(zh_line) == 1 and len(en_line) == 1, (zh_text, en_text)
    assert "仅形态匹配" in zh_line[0], f"zh 提示必须写明只是形态命中：{zh_line[0]!r}"
    assert "shape match only" in en_line[0], f"en 提示必须写明只是形态命中：{en_line[0]!r}"
    for line in (zh_line[0], en_line[0]):
        name, match, kernel = _parse(line)
        assert name.startswith("topology:") and match == "topology", (name, match)
        assert kernel == "", f"没强制内核就不该报执行内核：{line!r}"
        for stale in STALE:
            assert stale not in line, f"提示里仍有过时措辞 {stale!r}：{line!r}"
    assert zh_info == {} and en_info == {}, (zh_info, en_info)


def test_forced_dedicated_notice_names_the_forced_kernel() -> None:
    """③ 强制 ``dedicated`` + 语料命中：点名实际执行的内核，并写明是强制。"""
    for language, marker in (("zh", "强制"), ("en", "forced")):
        line_text, info, _record = _run_contract(language, kernel_force="dedicated")
        lines = _notices(line_text)
        assert len(lines) == 1, f"（{language}）语料方程应当正好打一条提示：{line_text!r}"
        name, match, kernel = _parse(lines[0])
        assert name.startswith("topology:") and match == "topology", (name, match)
        assert kernel in ("kupl_dedicated", "cuda_dedicated"), (
            f"（{language}）强制 dedicated 的提示必须点名 dedicated 内核：{lines[0]!r}")
        assert info.get("path") == kernel, (
            f"（{language}）提示的内核与 LAST_INFO['path'] 不一致：{kernel!r} vs {info!r}")
        assert marker in lines[0], (
            f"（{language}）强制语义必须写在提示里（{marker!r}）：{lines[0]!r}")
        # **不许**补"没有自动尝试"这类负向说明：用户明确要求删掉那句尾巴
        # （``未做"先 dedicated 再 topology"的自动尝试``），理由是"多余的输出"——
        # 提示只陈述**实际跑了什么**，不去解释代码选择**没有**尝试什么
        # （2026-11-18 用户裁定；``_NOTE_ZH/_NOTE_EN["forced"]`` 已按此精简）。
        for negative in NEGATIVE_META:
            assert negative not in lines[0], (
                f"（{language}）提示只陈述实际执行的路线，不写负向说明"
                f"（{negative!r} 属被删除的『多余输出』）：{lines[0]!r}")


def test_forced_topology_notice_names_topology_kernel() -> None:
    """④ 强制 ``topology`` + 语料命中：提示点名 ``kupl_topology``（或 v2 回落的路线）。"""
    for language, marker in (("zh", "强制"), ("en", "forced")):
        line_text, info, _record = _run_contract(language, kernel_force="topology",
                                                 topology_real=True)
        lines = _notices(line_text)
        assert len(lines) == 1, f"（{language}）语料方程应当正好打一条提示：{line_text!r}"
        name, match, kernel = _parse(lines[0])
        assert name.startswith("topology:") and match == "topology", (name, match)
        assert "kupl_topology" in kernel or "cuda_topology" in kernel, (
            f"（{language}）强制 topology 的提示必须点名拓扑内核：{lines[0]!r}")
        assert info.get("path") == kernel, (
            f"（{language}）提示的内核与 LAST_INFO['path'] 不一致：{kernel!r} vs {info!r}")
        assert marker in lines[0], (
            f"（{language}）强制语义必须写在提示里（{marker!r}）：{lines[0]!r}")


def test_notice_machine_segments_identical_across_languages() -> None:
    """⑤ ``en`` / ``zh`` 三档的机器段逐字相同（而原文不同，证明确实切换了语言）。"""
    with _profile_generic_table() as directory:
        scenarios = {
            "generic": {"profile_dir": directory},
            "dedicated": {"kernel_force": "dedicated"},
            "topology": {"kernel_force": "topology", "topology_real": True},
        }
        for label, kwargs in scenarios.items():
            zh_line = _notices(_run_contract("zh", **kwargs)[0])
            en_line = _notices(_run_contract("en", **kwargs)[0])
            assert len(zh_line) == 1 and len(en_line) == 1, (label, zh_line, en_line)
            assert _parse(zh_line[0]) == _parse(en_line[0]), (
                f"（{label}）形态名/命中族/执行路线跨语言不一致："
                f"zh={_parse(zh_line[0])!r} en={_parse(en_line[0])!r}")
            assert ROUTE_PATTERN.search(zh_line[0]).group(0) == \
                ROUTE_PATTERN.search(en_line[0]).group(0), (
                    f"（{label}）机器段跨语言不一致")
            assert zh_line[0] != en_line[0], (
                f"（{label}）两种语言的提示原文相同——语言没真的切换，比对没有意义")


def test_forced_failure_notice_says_it_fell_back_to_generic() -> None:
    """补充：强制没跑成而回落的那条提示必须写"已回落 generic"，且不谎称跑过内核。

    ``_api`` 的强制路径失败时**抛异常**（不回落到提示），所以这一档只能这样合成：
    把执行侧记录摆成"要求了 dedicated、但 ``ran`` 为空"，再让 ``announce`` 拿到空标签。
    这样即使将来某条路真的回落了，文案也已经钉在契约里。
    """
    entry = _specialize.match(EQUATION)
    assert entry is not None, "语料方程必须命中，否则本契约的前提不成立"
    expected = {"zh": "已回落 generic", "en": "fell back to generic"}
    for language, marker in expected.items():
        with _isolated_state(language):
            _api._LAST_KERNEL = {
                "requested": ("dedicated",), "ran": None,
                "attempted": ("kupl_dedicated",),
                "reasons": {"kupl_dedicated": "boom"},
            }
            _kernel_kupl.LAST_INFO.clear()
            lines = _notices(_capture(_specialize.announce, entry, ""))
        assert len(lines) == 1, (language, lines)
        assert marker in lines[0], f"（{language}）回落文案缺失：{lines[0]!r}"
        assert "[plan-cache]" in lines[0] and "[kernel:" not in lines[0], (
            f"（{language}）回落后没有内核执行，不该出现 [kernel: …]：{lines[0]!r}")
        assert "dedicated" in lines[0], (
            f"（{language}）回落文案要说清被强制的是什么：{lines[0]!r}")


TESTS = (
    test_default_generic_notice_has_no_automatic_attempt_wording,
    test_profile_generic_notice_says_shape_match_only,
    test_forced_dedicated_notice_names_the_forced_kernel,
    test_forced_topology_notice_names_topology_kernel,
    test_notice_machine_segments_identical_across_languages,
    test_forced_failure_notice_says_it_fell_back_to_generic,
)


def test_einsum_notice_forced_semantics_contracts() -> None:
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
            f"{len(failures)} 条命中提示语义契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("命中提示语义契约全部跳过")


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
