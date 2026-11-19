"""契约：``--tune`` 的**分支化有序遍历**与 ``min_wall_ms`` 剪枝（2026-11-18 规则）。

这条规则替换了两件事，所以必须有契约盯住——两件事都属于"报告看起来正常、结论却是
错的"那一类：

1. **清单是有序的，不再是笛卡尔积**：三个分支各有**固定组数**（``B1`` = 22、
   ``B2`` = 22、``B3`` = 21）且**逐项顺序**固定（``dedicated`` → ``generic`` →
   ``topology`` → ``backend``，每个内核内 ``greedy/branch_1/branch_2/dp/branch_all/
   optimal/none``）。顺序即执行顺序，而执行顺序决定剪枝的上界何时被压低——换一次
   顺序，结论就换一套，而报告里看不出来。
2. **``min_wall_ms`` 剪枝**：初值 = ``--tune-timeout``；任一组的单次 ``contract()``
   耗时超过当前上界即**终止该组**（原文 ``耗时大于min_wall_ms,此组被终止``）、继续下
   一组、**不写缓存**；某组测到更小耗时则立刻把上界降为该值。``best`` 就是记录的最小
   耗时那一组。

计时怎么注入
------------
本契约**不做任何"机器够快"的假设**：每组耗时由 :func:`_delayed_contract` 用
``time.sleep`` **注入**（按组下标取睡眠时长），于是"哪一组应当被剪枝"完全由注入表
决定，而不是由本机快慢决定。``--tune-timeout`` 也显式给出（缺省 30 s → 初值
30000 ms），所以第 2 条的"初值 = --tune-timeout"是**恒等式**，不是近似。

可直接运行::

    python -B -m pyqcd.testing._einsum_tune_plan_contract
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
import time
import unittest
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import pyqcd.einsum as pe
from ..einsum import _profile_cache, _tune_axes

#: 本契约自己用的小清单：四个格子，睡眠时长逐项给定（见 :data:`DELAYS`）。
#: 用注入的小清单而不是 22 组：这里要验的是**机制**，"22 组本身有多快"与规则无关。
COMBOS = (("generic", "false", "greedy"),
          ("generic", "false", "branch_1"),
          ("generic", "false", "dp"),
          ("generic", "false", "branch_2"))

#: 注入的耗时序列（秒）：第 1 组 20 ms 立起上界，第 2 组 5 ms 压低它，
#: 第 3 组 10 ms 与第 4 组 30 ms 都超过 5 ms → 必须被剪枝。
DELAYS = (0.020, 0.005, 0.010, 0.030)

#: ``--tune-timeout`` 的显式取值（秒）= ``min_wall_ms`` 的初值来源。
TIMEOUT = 30.0

EQUATION = "ij,jk,kl->il"
SHAPES = [(6, 6), (6, 6), (6, 6)]

#: 规格 §3 的**逐项顺序**（契约自己照抄一遍，不复用实现的常量：复用就验不出顺序）。
SPEC_PREFIX = tuple(
    (kernel, "false", optimize)
    for kernel in ("dedicated", "generic")
    for optimize in ("greedy", "branch_1", "branch_2", "dp", "branch_all",
                     "optimal", "none"))
SPEC_TAIL = tuple(("backend", "false", optimize)
                  for optimize in ("greedy", "branch_1", "branch_2", "dp",
                                   "branch_all", "optimal", "none"))
SPEC_EXTRA = {
    _tune_axes.TUNE_BRANCH_RAW: (("topology", "false", "none"),),
    _tune_axes.TUNE_BRANCH_EQUIVALENT: (("topology", "true", "none"),),
    _tune_axes.TUNE_BRANCH_NONE: (),
}
SPEC_SIZES = {_tune_axes.TUNE_BRANCH_RAW: 22,
              _tune_axes.TUNE_BRANCH_EQUIVALENT: 22,
              _tune_axes.TUNE_BRANCH_NONE: 21}

#: 剪枝时必须逐字出现的原文（用户口径）。
TERMINATION_TEXT = "耗时大于min_wall_ms,此组被终止"

_ENV_NAMES = (_profile_cache.PROFILE_ENV, _profile_cache.DIR_ENV)


# -- 隔离 --------------------------------------------------------------------
@contextlib.contextmanager
def _isolated() -> Iterator[str]:
    """临时 CWD + 干净的配置表开关；退出时原样还回去（仓库根不留 ``.cache``）。"""
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    saved_state = (_profile_cache._TABLE, _profile_cache._CONFIGURED_DIR,
                   _profile_cache._LAST_ERROR)
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-tune-plan-") as root:
            os.chdir(root)
            os.environ[_profile_cache.PROFILE_ENV] = "1"
            os.environ.pop(_profile_cache.DIR_ENV, None)
            _profile_cache._TABLE = None
            _profile_cache._CONFIGURED_DIR = ""
            _profile_cache._LAST_ERROR = None
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
        pe.cache_clear()


def _cli() -> Any:
    """CLI 模块（懒取，避免本模块 import 期就拖进 argparse）。"""
    from ..einsum import test as cli

    return cli


def _require_plan() -> Any:
    """新规则是否已落地（清单入口 + 剪枝记账）。未落地时**显式跳过**，不静默通过。"""
    cli = _cli()
    missing = [name for name in ("tune_plan", "tune_branch", "TERMINATION_MESSAGE",
                                 "tune_combos", "run_tune")
               if getattr(cli, name, None) is None]
    if missing:
        raise unittest.SkipTest(f"等待实现：tune 新规则尚未落地（缺 {missing}）")
    return cli


def _delayed_contract(delays: Sequence[float], counter: dict[str, int]):
    """按组下标睡眠的假 ``contract``：耗时由 :data:`DELAYS` 注入，不看本机快慢。

    组下标 = ``counter["index"] // calls_per_group``——``warmup + rounds`` 次调用算
    同一组，因此每次调用都必须睡眠（真实 ``contract`` 也是每次都花时间）。
    """
    real = pe.contract

    def fake(equation: str, *operands: Any, **kwargs: Any) -> Any:
        index = counter["index"] // counter["calls_per_group"]
        counter["index"] += 1
        delay = delays[min(index, len(delays) - 1)]
        if delay:
            time.sleep(delay)
        counter["seen"].append(index)
        return real(equation, *operands, **kwargs)

    return fake


def _run(work: str, tag: str, *, delays: Sequence[float],
         timeout: float = TIMEOUT) -> dict[str, Any]:
    """在临时子目录里跑一次小清单 ``run_tune``（注入耗时、注入小清单）。

    ``isolate=False`` 是**必须**的：本契约用 ``patch.object(pe, "contract", ...)`` 注入
    耗时，而子进程路线（``--tune`` 的缺省）根本不在本进程里调 ``contract``，注入会失效
    ——那样"哪一组该被剪枝"就由本机快慢决定，断言全成废纸。``min_wall_ms`` 的规则两条
    路都实现（只是执行位置不同），所以进程内足够验机制。
    """
    cli = _cli()
    directory = os.path.join(work, tag)
    os.makedirs(directory, exist_ok=True)
    counter = {"index": 0, "calls_per_group": 2, "seen": []}
    stderr = io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(
            cli, "tune_combos", lambda *args, **kwargs: list(COMBOS)))
        stack.enter_context(patch.object(
            pe, "contract", _delayed_contract(delays, counter)))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(stderr))
        tune = cli.run_tune(equation=EQUATION, shapes=SHAPES,
                            warmup=1, rounds=1, profile=True,
                            profile_kwargs={"profile": True,
                                            "profile_dir": os.path.join(directory,
                                                                        "cache")},
                            timeout=timeout, isolate=False, language="zh-cn")
        # 报告文本也一起收：本契约要断言"逐组那一行 + 表尾"的原文，那是
        # ``print_tune`` 的产物（``run_tune`` 只返回字典、不打印）。
        cli.print_tune(tune, "zh-cn", verbose=False)
    tune["_stderr"] = stderr.getvalue()
    return tune


def _disk_rows() -> list[dict[str, str]]:
    """配置表的**数据行**（表头不计；缺文件就是空表）。"""
    path = _profile_cache.summary().get("file")
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _optimize_column(rows: Sequence[dict[str, Any]]) -> list[str]:
    return [str(row.get("optimize")) for row in rows]


# ── 1. 三支组数与逐项顺序 ────────────────────────────────────────────────────
def test_tune_plan_sizes_and_order_per_branch():
    """契约 1/5：``B1``/``B2`` = 22 组、``B3`` = 21 组，且顺序与清单逐项一致。

    顺序是**执行顺序**（前缀 → 分支专属 → 尾部）：报告里的"哪一组先压低了上界"
    直接由它决定，所以逐项比对，而不是只比集合或组数。
    """
    cli = _require_plan()
    for branch, size in SPEC_SIZES.items():
        expected = list(SPEC_PREFIX) + list(SPEC_EXTRA[branch]) + list(SPEC_TAIL)
        actual = list(cli.tune_plan(branch))
        assert actual == expected, (
            f"{branch} 的有序清单必须逐项一致", actual, expected)
        assert len(actual) == size, (f"{branch} 的组数", len(actual), size)
        assert cli._tune_axes.tune_plan_size(branch) == size, branch
    # 三个分支合起来覆盖了这两条 axis 取值（清单不许漏档）。
    sizes = [len(cli.tune_plan(branch)) for branch in cli.TUNE_BRANCHES]
    assert sizes == [22, 22, 21], ("三个分支的组数", sizes)
    # ``topology`` 只在能特化的分支出现，且 ``B2`` 那一组必须带 as_equivalence=true。
    assert ("topology", "false", "none") in cli.tune_plan(
        cli.TUNE_BRANCH_RAW), "B1 必须试 topology（原方程能特化）"
    assert ("topology", "true", "none") in cli.tune_plan(
        cli.TUNE_BRANCH_EQUIVALENT), "B2 的 topology 组必须让等价改写生效"
    assert not [item for item in cli.tune_plan(cli.TUNE_BRANCH_NONE)
                if item[0] == "topology"], "B3 不该有 topology 组"
    # 分支判定本身：语料方程 → B1；只有等价写法能命中 → B2；都不行 → B3。
    assert cli.tune_branch("cadb,menf,ac,em,bd,fn->")["branch"] == "B1"
    equiv = cli.tune_branch("cbda,menf,ac,em,bd,fn->")
    assert equiv["branch"] == "B2" and equiv["raw_ok"] is False, equiv
    none = cli.tune_branch("ij,jk->ik")
    assert none["branch"] == "B3" and not none["raw_ok"] and not none["equiv_ok"], none


# ── 2. min_wall_ms 初值 = --tune-timeout，且被更小值替换 ────────────────────
def test_tune_min_wall_starts_at_timeout_and_shrinks():
    """契约 2/5：初值 = ``--tune-timeout``（秒→毫秒）；中途被更小的实测中位替换。

    注入的耗时序列是 20 ms → 5 ms → 10 ms → 30 ms：上界因此必须是
    ``30000 ms → 20 ms → 5 ms``，最后停在 5 ms（那之后的两组都超过它）。
    """
    cli = _require_plan()
    with _isolated() as work:
        tune = _run(work, "shrink", delays=DELAYS)
    assert tune["min_wall_source"] == "--tune-timeout", tune
    assert float(tune["min_wall_ms_initial"]) == TIMEOUT * 1000.0, (
        "min_wall_ms 初值必须是 --tune-timeout 换算成毫秒",
        tune["min_wall_ms_initial"], TIMEOUT)
    rows = tune["rows"]
    assert len(rows) == len(COMBOS), rows
    # 第 1、2 组是干净的 ok：它们依次把上界压到 20 ms、5 ms。
    assert [row["status"] for row in rows[:2]] == ["ok", "ok"], rows[:2]
    first, second = rows[0]["median_ms"], rows[1]["median_ms"]
    final = tune["min_wall_ms"]
    assert final is not None and float(final) <= float(second), (
        "上界必须已降到第 2 组的实测耗时", final, second)
    assert float(final) < float(first), (
        "上界必须被更小的实测值替换过", final, first)
    assert final == second, ("上界就是记录到的最小中位", final, second)
    assert tune["min_wall_config"] == list(COMBOS[1]), (
        "压低上界的那一组必须被记下来", tune["min_wall_config"], list(COMBOS[1]))
    # --tune-timeout 0 = 不设上界：初值与最终值都必须是 None，且一组都不剪。
    with _isolated() as work:
        free = _run(work, "unbounded", delays=DELAYS, timeout=0.0)
    assert free["min_wall_ms_initial"] is None and free["min_wall_ms"] is None, free
    assert not [row for row in free["rows"] if row.get("pruned")], free["rows"]


# ── 3./4./5. 剪枝：终止、不写缓存、原文、best 是最小耗时组 ───────────────────
def test_tune_prunes_slower_combinations_without_writing_them():
    """契约 3/5 + 4/5 + 5/5：终止、不写缓存、原文逐字、``best`` = 最小耗时组。

    三条共用同一次注入运行（同一份证据不该跑三遍）：``write_winner`` 只被调用 1 次且
    写的是 ``best``（= 3）、``原因=`` 里逐字含 :data:`TERMINATION_TEXT`（= 4）、
    ``best`` 是可用组里最小的 ``median_ms``（= 5）。
    """
    cli = _require_plan()
    written: list[dict[str, Any]] = []
    real_write = _profile_cache.write_winner

    def spy(key, config, **kwargs):
        written.append({"key": dict(key), "config": dict(config),
                        "kwargs": dict(kwargs)})
        return real_write(key, config, **kwargs)

    with _isolated() as work:
        with patch.object(_profile_cache, "write_winner", spy):
            tune = _run(work, "prune", delays=DELAYS)
        rows = tune["rows"]
        disk = _disk_rows()
        stderr = tune["_stderr"]

    pruned = [row for row in rows if row.get("pruned")]
    assert pruned, ("后两组必须被剪枝", [(r["optimize"], r["median_ms"]) for r in rows])
    # 第 3、4 组的注入耗时（10 ms / 30 ms）都超过当时的 5 ms 上界。
    assert [row["optimize"] for row in pruned] == ["dp", "branch_2"], pruned
    assert all(row["median_ms"] is None for row in pruned), pruned
    assert all(row["status"] == "pruned" for row in pruned), pruned

    # 原文逐字出现（含两个数值），且不折进任何其它分类。
    for row in pruned:
        reason = str(row.get("reason") or "")
        assert TERMINATION_TEXT in reason, ("终止原文必须逐字出现", reason)
        assert "wall_ms=" in reason and "min_wall_ms=" in reason, reason
    assert cli.TERMINATION_MESSAGE == TERMINATION_TEXT, cli.TERMINATION_MESSAGE
    assert TERMINATION_TEXT in stderr, ("逐组那一行也要打出来", stderr[-800:])
    assert "tune prune:" in stderr, stderr[-800:]

    # **不写缓存**：``write_winner`` 只被调用 1 次，且写的是 ``best``。
    best = tune["best"]
    assert best is not None, tune["best"]
    assert len(written) == 1, ("被剪枝的组不许落盘", [item["config"] for item in written])
    assert written[0]["config"]["optimize"] == best["optimize"], (written, best)
    assert "dp" not in _optimize_column(disk), ("被剪枝组的配置行不得出现在表里", disk)
    assert len(disk) == 1, ("表里只该有胜出那一行", disk)

    # ``best`` = 记录的最小耗时组（第 2 组，5 ms）。
    measured = [row for row in rows if row["median_ms"] is not None]
    assert best is min(measured, key=lambda row: row["median_ms"]), (best, measured)
    assert best["optimize"] == COMBOS[1][2], best
    assert float(best["median_ms"]) == float(tune["min_wall_ms"]), (
        "best 与 min_wall_ms 必须是同一个数（否则两套账）", best, tune["min_wall_ms"])
    assert "tune best :" in stderr, stderr[-800:]


TESTS = (
    test_tune_plan_sizes_and_order_per_branch,
    test_tune_min_wall_starts_at_timeout_and_shrinks,
    test_tune_prunes_slower_combinations_without_writing_them,
)


def test_einsum_tune_plan_contracts() -> None:
    """注册入口：逐条执行本模块契约；全跳过时也显式 SkipTest（不许静默通过）。"""
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
            f"{len(failures)} 条 tune 有序清单/剪枝契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"tune plan contracts did not run ({skipped} skipped)")


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
