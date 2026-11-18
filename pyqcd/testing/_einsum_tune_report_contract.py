"""契约：``--tune`` 报告的三处**用户实测**缺陷（2026-11-18）。

三处都属于"报告看起来正常、结论却是错的"那一类，所以各有一条断言钉住：

1. **提示参数保真**：缺表时 ``需要先 tune：`` 给出的命令必须是**用户刚跑的那条**。
   用户实测在 GPU 上跑，提示里却是 ``--device cpu``（键里的 ``cupy:0`` 被读成 CPU），
   并且 ``--shape`` 少了 ``->[40,400]`` 输出段——照着提示重跑一遍等于换了配置。
2. **不可用内核被跳过而不是报错**：``dedicated`` 的计划 arena 超出内核自己的预算时，
   22 组里有 7 组是同一句错误的副本，``topology`` 再补一组 "status 4"，真正的异常被
   淹掉。这类"该内核对这个形状没有路"必须在**遍历前**探测并整档跳过（状态 ``skip``）。
3. **组数用裁剪后的实际值**：``1/N`` 的 ``N`` 是**实际会被遍历**的组数；照旧写 22 会让
   "哪几组没跑"永远对不上账。三条分支（``B1``/``B2``/``B3``）分别计算。
4. **末行带 ``min_wall_ms`` 与 ``best``**：用户要在 ``PASS`` 那一行直接看到剪枝上界与
   最终配置——两个数相同也要都显示，那是"上界确实来自最优组"的唯一证据。

隔离口径：配置表一律写临时目录（``--profile-dir``），**不碰 ``~/.cache``**；本契约不做
任何"机器够快"的假设（没有任何计时阈值断言）。

可直接运行::

    python -B -m pyqcd.testing._einsum_tune_report_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import tempfile
import unittest
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import numpy as np

import pyqcd.einsum as pe
from ..einsum import _profile_cache

#: 小方程：跑满一整张网格也不心疼（``B3`` = 21 组）。
EQUATION = "ij,jk,kl->il"
SHAPES = [(6, 6), (6, 6), (6, 6)]

#: ``--shape`` 里带输出段的小方程：``ab,bc->ac`` + ``[40,4],[4,400]->[40,400]`` 的输出
#: 正是用户报告里缺掉的那一段，而算量只有 40×4×400（够小，CPU 也能跑）。
HINT_EQUATION = "ab,bc->ac"
HINT_SHAPE = "[40,4],[4,400]->[40,400]"
HINT_OUTPUT = "->[40,400]"

#: 用户报告里的 arena 超预算原文（阈值 3 GiB = :data:`_kernel_cuda._CUDA_PLAN_ARENA_LIMIT`）。
#: 本契约**不新造数字**：只把内核自己会给出的那句话照抄过来，好断言它被逐字转述。
ARENA_REASON = ("plan arena exceeds the measured CUDA dedicated budget "
                "(4375047168 >= 3221225472 bytes)")

#: ``PYQCD_EINSUM_PROFILE`` 的取值：1 = 开启（缺表时才会有提示可验）。
_ENV_NAMES = (_profile_cache.PROFILE_ENV, _profile_cache.DIR_ENV)


# -- 隔离 --------------------------------------------------------------------
@contextlib.contextmanager
def _isolated() -> Iterator[str]:
    """临时 CWD + 干净的配置表开关；退出时原样还回去（仓库根不留 ``.cache``）。

    **必须一并复位"缺表警告"的一次性开关**（``_api._PROFILE_MISSING_NOTICED``）：
    那条警告按设计是"每进程一次"，全套里任何一个先撞上缺表的契约（例如
    ``_einsum_profile_readonly_contract.test_missing_profile_warns_with_tune_command``）
    都会把它置位，本契约再问就一个字都收不到——单独跑必过、进全套必失败。这与
    ``PYQCD_EINSUM_PROFILE=0``（``conftest.main`` 的隔离）是两件事，后者只关开关。
    """
    from ..einsum import _api

    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    saved_state = (_profile_cache._TABLE, _profile_cache._CONFIGURED_DIR,
                   _profile_cache._LAST_ERROR)
    saved_noticed = _api._PROFILE_MISSING_NOTICED
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-tune-report-") as root:
            os.chdir(root)
            os.environ[_profile_cache.PROFILE_ENV] = "1"
            os.environ.pop(_profile_cache.DIR_ENV, None)
            _profile_cache._TABLE = None
            _profile_cache._CONFIGURED_DIR = ""
            _profile_cache._LAST_ERROR = None
            _api._PROFILE_MISSING_NOTICED = False
            try:
                yield root
            finally:
                os.chdir(saved_cwd)
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        (_profile_cache._TABLE, _profile_cache._CONFIGURED_DIR,
         _profile_cache._LAST_ERROR) = saved_state
        _api._PROFILE_MISSING_NOTICED = saved_noticed
        pe.cache_clear()


def _cli() -> Any:
    """CLI 模块（懒取，避免本模块 import 期就拖进 argparse）。"""
    from ..einsum import test as cli

    return cli


@contextlib.contextmanager
def _capture() -> Iterator[tuple[io.StringIO, io.StringIO]]:
    """同时收 stdout / stderr（``PASS`` 在 stdout，``tune`` 表在 stderr）。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


def _main(argv: Sequence[str], *, cpu_arrays: bool = False) -> tuple[int, str, str]:
    """进程内跑一次 ``main()``，返回 ``(退出码, stdout, stderr)``。

    ``cpu_arrays=True`` 时把 ``_xp_for_device`` 换成 numpy：``--device gpu`` 只是
    **命令行上的取值**（本契约验的是提示保真，不该依赖本机有没有 GPU）。
    """
    cli = _cli()
    with _capture() as (out, err):
        if cpu_arrays:
            with patch.object(cli, "_xp_for_device", lambda device, language="zh": np):
                code = cli.main(list(argv))
        else:
            code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def _run_tune(work: str, tag: str, *, unavailable: dict[str, str] | None
              ) -> tuple[dict[str, Any], str]:
    """在临时子目录里跑一次 ``run_tune`` + ``print_tune``，返回 ``(tune, 表文本)``。

    ``unavailable`` 直接顶掉**准入探测**（本契约要验的是"探测判不可用之后报告怎么写"，
    不是内核本身跑不跑得动；真值由内核自己的准入函数决定）。
    """
    cli = _cli()
    directory = os.path.join(work, tag)
    os.makedirs(directory, exist_ok=True)
    cache_dir = os.path.join(directory, "cache")
    tune: dict[str, Any] = {}
    printed = ""
    with _capture() as (_out, err):
        with patch.object(cli, "tune_unavailable_kernels",
                          lambda **kwargs: dict(unavailable or {})):
            tune = cli.run_tune(equation=EQUATION, shapes=SHAPES, warmup=1, rounds=1,
                                profile=True,
                                profile_kwargs={"profile": True,
                                                "profile_dir": cache_dir})
        cli.print_tune(tune, "zh-cn", verbose=False)
        printed = err.getvalue()
    return tune, printed


# ── 1. 提示参数保真 ──────────────────────────────────────────────────────────
def test_hint_repeats_the_users_own_invocation():
    """契约 1/4：``需要先 tune：`` 的命令用**本次实参**：GPU 还是 GPU、形状带输出段。

    断言的三件事分别对应三个真实症状：``--device gpu`` 必须原样保留（原先按
    ``cupy:0`` 重建，输出成了 ``--device cpu``）、``--shape`` 必须带 ``->[40,400]``
    （原先只有操作数）、用户没给 ``--compare`` 就不许替他加上（``--tune`` 缺省关闭）。
    """
    cli = _cli()
    with _isolated() as work:
        cache_dir = os.path.join(work, "cache")
        code, _stdout, stderr = _main(
            ["--equation", HINT_EQUATION, "--shape", HINT_SHAPE,
             "--device", "gpu", "--dtype", "complex128", "--warmup", "1",
             "--rounds", "1",
             "--profile-dir", cache_dir, "--language", "zh-cn"],
            cpu_arrays=True)
    assert code == 0, ("这次调用本身必须成功", code, stderr[-800:])
    assert "需要先 tune" in stderr, ("缺表必须有提示", stderr[-800:])
    hint = [line for line in stderr.splitlines()
            if "python -m pyqcd.einsum.test" in line]
    assert len(hint) == 1, ("提示里必须恰好一条可复制的命令", hint, stderr[-800:])
    command = hint[0]
    assert "--device gpu" in command, ("提示必须保留用户的 --device", command)
    assert "--device cpu" not in command, (
        "GPU 调用不许被提示成 CPU（cupy:0 不是 cpu）", command)
    assert HINT_OUTPUT in command, ("--shape 必须带输出段", command)
    # 另外三轴也必须逐字来自本次实参：照提示重跑要等价于"再跑一次同样的配置"。
    assert f"--equation {HINT_EQUATION}" in command or (
        f'--equation "{HINT_EQUATION}"' in command), (
        "提示必须保留用户的 --equation", command)
    assert "--dtype complex128" in command, ("提示必须保留用户的 --dtype", command)
    assert "--tune" in command, command
    # 用户没给 --compare：tune 缺省不跑参照基准，提示也就不能替他加。
    assert "--compare" not in command, (
        "用户原命令没有 --compare 时不许写进去", command)


# ── 2. 不可用内核被跳而非报错 ────────────────────────────────────────────────
def test_unavailable_kernel_is_skipped_not_errored():
    """契约 2/4：``dedicated`` 准入不过 → 整档 ``skip``，``error`` 一份都不许有。

    "该内核对这个形状本质上不可用"是**判定**，不是异常：逐组跑一遍只会得到同一句
    错误（用户实测 7 组）的副本。报告必须把这一档写成 ``skip`` 并在表尾点名
    ``skipped-kernels:``，原因保持内核原文（含两个字节数，用户能与内核日志对上）。
    """
    cli = _cli()
    with _isolated() as work:
        tune, printed = _run_tune(work, "skip", unavailable={"dedicated": ARENA_REASON})
    rows = tune["rows"]
    skipped = [dict(item) for item in (tune.get("skipped_kernels") or [])]

    # ① 没有任何一组是 error（这一轮里根本没有"意外"）。
    errored = [row for row in rows if row.get("error")]
    assert not errored, ("准入不可是 skip，不是 error", errored[:1])
    assert "状态=error" not in printed, printed[-1200:]
    assert "tune skip :" not in printed, (
        "没有 error/mismatch/timeout 时不该出现 tune skip 汇总行", printed[-1200:])

    # ② 该内核的**所有组**都被 skip 覆盖（组数与完整清单逐项相等）。
    branch = cli.tune_branch(EQUATION)["branch"]
    planned = cli.tune_plan(branch)
    dedicated = [combo for combo in planned if combo[0] == "dedicated"]
    assert dedicated, ("B3 的清单里也有 dedicated 组", planned[:3])
    assert [item["kernel_force"] for item in skipped] == ["dedicated"], skipped
    entry = skipped[0]
    assert entry["status"] == "skip", entry
    assert int(entry["combos"]) == len(dedicated), (entry, len(dedicated))
    assert ARENA_REASON in str(entry["reason"]), (
        "原因必须是内核自己的原文（含两个字节数）", entry["reason"])

    # ③ 报告里看得见：逐组区一行 ``状态=skip``，表尾 ``skipped-kernels:`` 点名。
    skip_lines = [line for line in printed.splitlines() if "状态=skip" in line]
    assert len(skip_lines) == 1, ("整档一行，不是每组一行", skip_lines)
    assert "kernel_force=dedicated" in skip_lines[0], skip_lines[0]
    assert "skipped-kernels: kernel_force=dedicated" in printed, printed[-1200:]
    assert ARENA_REASON in printed, ("表尾必须带原因原文", printed[-1200:])


# ── 3. 组数裁剪 ─────────────────────────────────────────────────────────────
def test_trimmed_group_counts_are_computed_per_branch():
    """契约 3/4：``N`` 是**裁剪后**的组数，三条分支各算各的（不许共用一个数）。

    冻结口径是 ``B1``/``B2`` = 22、``B3`` = 21；``dedicated`` 占 7 组（共同前缀里只有
    ``as_equivalence=false`` 那一遍），``topology`` 只在 ``B1``/``B2`` 各占 1 组。
    于是裁剪后的期望值可以直接写死：``dedicated`` 不可用 → 15/15/14，
    ``topology`` 不可用 → 21/21/21（``B3`` 本来就没有它），两者都不可用 → 14/14/14。
    """
    cli = _cli()
    assert cli.tune_plan_sizes_trimmed(()) == {"B1": 22, "B2": 22, "B3": 21}
    assert cli.tune_plan_sizes_trimmed(["dedicated"]) == {"B1": 15, "B2": 15, "B3": 14}
    assert cli.tune_plan_sizes_trimmed(["topology"]) == {"B1": 21, "B2": 21, "B3": 21}
    assert cli.tune_plan_sizes_trimmed(["dedicated", "topology"]) == {
        "B1": 14, "B2": 14, "B3": 14}
    # 裁剪只动 ``kernel_force`` 那一轴：保留项的顺序必须与完整清单一致。
    kept = cli.tune_plan_trimmed("B1", ["dedicated"])
    assert kept == [combo for combo in cli.tune_plan("B1") if combo[0] != "dedicated"]

    # 真跑一轮对账：报告里的 ``N`` 就是裁剪后的实际组数（不是 21 也不是 22）。
    with _isolated() as work:
        tune, printed = _run_tune(work, "counts",
                                  unavailable={"dedicated": ARENA_REASON})
    branch = cli.tune_branch(EQUATION)["branch"]
    expected = cli.tune_plan_sizes_trimmed(["dedicated"])[branch]
    assert int(tune["planned"]) == len(cli.tune_plan(branch)), tune["planned"]
    assert int(tune["trimmed"]) == 7, tune["trimmed"]
    assert int(tune["total"]) == len(tune["rows"]) == expected, (
        "报告里的 N 必须是裁剪后的组数", tune["total"], len(tune["rows"]), expected)
    assert f"计划 {int(tune['planned'])} → 实际 {expected}" in printed, (
        "tune hdr 必须说明组数被裁剪", printed[-1200:])


# ── 4. 末行含 min_wall_ms ───────────────────────────────────────────────────
def test_pass_line_reports_min_wall_ms_and_best():
    """契约 4/4：末行 ``PASS`` 同时给 ``min_wall_ms=`` 与 ``best=`` 的最终配置。

    ``min_wall_ms`` 与 ``best.wall_ms`` 数值相同也要并排显示：前者是**剪枝上界**，
    后者是**被选中的组**，两个数相等正是"上界确实来自最优组"的证据。
    """
    cli = _cli()
    with _isolated() as work:
        cache_dir = os.path.join(work, "cache")
        code, stdout, stderr = _main(
            ["--equation", HINT_EQUATION, "--shape", "[4,4],[4,4]->[4,4]",
             "--device", "cpu", "--dtype", "float64", "--warmup", "1",
             "--rounds", "1", "--tune", "--profile-dir", cache_dir,
             "--language", "zh-cn"])
    assert code == 0, (code, stderr[-800:])
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert lines, ("stdout 必须给出结果行", stdout)
    last = lines[-1]
    assert last.startswith("PASS "), ("末行必须是 PASS", last)
    assert re.search(r"min_wall_ms=\d", last), (
        "末行必须带 min_wall_ms 的数值", last)
    assert "best=" in last, ("末行必须带最终 best 配置", last)
    assert "kernel_force=" in last and "as_equivalence=" in last \
        and "optimize=" in last, ("best= 后的三个轴都要给", last)
    # ``min_wall_ms`` 与 ``best.wall_ms`` 是同一个数：从末行里取出来逐位比较。
    bound = re.search(r"min_wall_ms=([0-9.]+)ms", last)
    assert bound, last
    assert float(bound.group(1)) > 0.0, ("上界必须是实测值，不是 0", last)
    assert cli._tune_pass_detail({}) == "  min_wall_ms=-ms  best=none", (
        "没有可用组时也要如实写出来", cli._tune_pass_detail({}))


TESTS = (
    test_hint_repeats_the_users_own_invocation,
    test_unavailable_kernel_is_skipped_not_errored,
    test_trimmed_group_counts_are_computed_per_branch,
    test_pass_line_reports_min_wall_ms_and_best,
)


def test_einsum_tune_report_contracts() -> None:
    """注册入口：逐条执行本模块契约；一条都没跑时显式 SkipTest（不许静默通过）。"""
    passed = skipped = 0
    failures: list[str] = []
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
            f"{len(failures)} 条 tune 报告契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"tune report contracts did not run ({skipped} skipped)")


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
