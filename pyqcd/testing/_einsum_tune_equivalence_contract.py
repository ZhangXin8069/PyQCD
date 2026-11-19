"""契约：``--tune`` 遍历与"只有 tune 能写配置表"的口径。

这些事都是"结果正确却白花时间/选错配置"的类型：遍历选错了只是报了个更慢的配置，
坏组合留下了一行则会**让下一次调用照着一条没跑成的配置走**。普通数值测试永远碰不到
它们，所以把它们固化成能独立失败的断言：

1. **``--tune`` 忽略它自己遍历的三维**：``--tune --kernel-force dedicated`` 与
   ``--tune --as-equivalence force`` 都必须**正常退出（0）**，同时在 stderr 给出
   "已忽略"警告，且遍历网格仍是完整的三轴笛卡尔积（``dedicated`` 组合照样在表里）——
   悄悄吞掉用户写下的取值、或把它变成"只跑一组"，报告里都看不出来。
2. **``--tune`` 组合数上限**：把某一维加宽到超过 30 组时必须报用法错误（退出码 2，
   消息里给出实际组数与上限），**绝不静默截断**；且报错发生在构造张量/跑任何一组之前。
3. **``--tune`` 最优选择**：注入假耗时让"最快"的那一组是数值不符（``rel`` 超
   ``atol + rtol``）的——它必须被判 ``mismatch`` 且**不得**成为 ``best``；
   跑不成（``KernelUnavailableError``）的组合只记 ``error``、不中断遍历。
4. **坏组合不留配置行**：被放弃的组合有三类——**超时**（注入 ``TimeoutError``）、
   **抛异常**（注入 ``RuntimeError``）、**数值不符**（``rel`` 超 ``atol``）。
   §3.1 的 ``tune_session`` 让遍历期间**一个字都不落盘**，§3.3 只在最后写胜出的那一行，
   所以三种坏组合结束后表里都只有**胜出的那一行**：坏组合既不成为 ``tune best``，
   也不留下"它跑过"的痕迹（含"更快的坏组合"这种最危险的情形）。每类都配一条正对照
   （同一网格不注入时那一行确实会写出来，且行数仍为 1），因为"没写"只有在
   "该写的地方写过"之后才有意义。

TV 重设计删除的旧口径（本模块不再断言）
--------------------------------------
* 已被废除的**等价负标记表**（``equivalence_marker`` /
  ``record_equivalence`` / ``as_equivalence="force"`` 刷新标记）——规格 §0/§5 已废除那
  张表；`auto` 的"有没有等价信息"改由主表（``pyqcd.einsum.profile.csv``）的等价行
  回答（§10）。
* tune 期间的**强制写整张地图**开关（``_api.set_profile_tune`` /
  ``_api._PROFILE_TUNE`` / ``_profile_cache.drop()``）——被 §1/§3.3 "只有 tune 能写，
  且只写胜出的那一行"取代。
对应的四条旧用例已随实现一并删除，**不留 skip 空壳**。

计时口径
--------
本模块**不做任何计时断言**：

* 第 3 条把 :func:`pyqcd.einsum.test._timed` 换成"调一次被测可调用对象、返回注入值"的
  假实现（:func:`_fake_timed`），排序结论只来自注入的假耗时；
* 第 4 条的超时/异常同样来自注入（:func:`_scripted_timed` 会先真跑一次再抛，否则
  "没写配置行"是空断言），跑的是 6×6 的小方程而不是生产尺寸。

临时目录 + 恢复
---------------
每个用例都在临时树里跑（CWD 也换过去），退出时把环境变量与两个模块的进程内状态原样
放回；仓库根不会留下 ``.cache``。

可直接运行::

    python -B -m pyqcd.testing._einsum_tune_equivalence_contract
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import warnings
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import numpy as np

import pyqcd.einsum as pe
from ..einsum import (_api, _cache, _equivalence, _kernel_force, _profile_cache,
                      _specialize, _threads)

#: ``--tune`` 的方程与形状：小到跑 30 组也不心疼（且耗时全部来自注入）。
TUNE_EQUATION = "ij,jk,kl->il"
TUNE_SHAPES = [(6, 6), (6, 6), (6, 6)]

#: 本模块自己用的小网格：同一个键、同一个通用内核，**只有 ``optimize`` 不同**。
#: 用它跑第 4 条（不碰生产网格）。取值必须是配置表的 7 个合法叶子之一
#: （``_profile_cache.OPTIMIZE_LEAVES``）：``auto`` 不是叶子，写进表里会被清空，
#: 于是"写下的那一行"读回来 .optimize 为空，正对照就无从比对。
SMALL_TUNE_COMBOS = (("generic", "false", "none"),
                     ("generic", "false", "greedy"),
                     ("generic", "false", "optimal"))

#: 各标签的轴长；同一个标签在不同操作数里必须等长。
LABEL_SIZES = {
    "i": 2, "j": 3, "k": 4, "l": 4,
    "c": 2, "a": 3, "d": 4, "b": 5,
    "m": 6, "e": 7, "n": 2, "f": 3,
    "M": 2, "N": 3,
}

#: 本模块会改写/还原的环境变量（含"别的契约可能设过"的开关）。
_ENV_NAMES = (
    _cache.CACHE_ENV,
    _equivalence.EQUIVALENCE_ENV,
    _equivalence.AS_EQUIVALENCE_ENV,
    _kernel_force.KERNEL_FORCE_ENV,
    _specialize.DISABLE_ENV,
    _specialize.NOTICE_ENV,
    _specialize.FUSED_DISABLE_ENV,
    _profile_cache.PROFILE_ENV,
    _profile_cache.DIR_ENV,
    _threads.STANDARD_OMP_ENV,
    _threads.STANDARD_BLAS_ENV,
)

#: ``_profile_cache`` 的进程内状态（``_TABLE`` 是解析缓存，容器由
#: :func:`_clear_mutable_state` 统一清）。``_LOCK`` 与常量不碰。
_PROFILE_SCALARS = {
    "_TUNING": 0, "_CONFIGURED_DIR": "", "_LAST_ERROR": None, "_HOST": None,
}

#: 仓库根：跑 ``python -m pyqcd.einsum.test`` 子进程时既做 CWD 也做 ``PYTHONPATH``。
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# -- 环境与进程内状态 ------------------------------------------------------------
def _prepare_env(profile: bool = True) -> None:
    """把本模块依赖的开关摆好（``_isolated`` 的 ``finally`` 负责还原）。"""
    os.environ[_cache.CACHE_ENV] = "1"
    os.environ[_equivalence.EQUIVALENCE_ENV] = "1"
    os.environ[_equivalence.AS_EQUIVALENCE_ENV] = "1"
    # ``PYQCD_EINSUM_SPECIALIZE`` 的名字是"是否禁用"，取值 ``1`` 表示**没**禁用：
    # 等价门在 ``specialize=False`` 时提前返回，所以这里必须让它开着。
    os.environ[_specialize.DISABLE_ENV] = "1"
    os.environ[_specialize.NOTICE_ENV] = "0"
    os.environ.pop(_specialize.FUSED_DISABLE_ENV, None)
    os.environ[_profile_cache.PROFILE_ENV] = "1" if profile else "0"
    os.environ.pop(_profile_cache.DIR_ENV, None)
    os.environ.pop(_kernel_force.KERNEL_FORCE_ENV, None)


def _clear_mutable_state(module: Any) -> None:
    """清空模块里的可变容器（字典/列表/集合），跳过全大写常量。

    不能只清"当前已知的那几个字段"：标记表是**新加**的状态，漏掉它会让上一条用例的
    标记漏进下一条（表现为"明明没写标记却跳过了"）。按类型清空才能覆盖得住；全大写
    名字（``_CLIP_LIMITS`` 之类的配置常量）明确跳过。
    """
    for name, value in vars(module).items():
        if name.startswith("__") or name.isupper():
            continue
        if isinstance(value, dict):
            value.clear()
        elif isinstance(value, list):
            value[:] = []
        elif isinstance(value, set):
            value.clear()


def _reset_state() -> None:
    """把两个模块的进程内状态清到"没配置、没解析缓存、不处于 tune 窗口"的初始态。"""
    for module in (_profile_cache, _api, _threads):
        _clear_mutable_state(module)
    for name, initial in _PROFILE_SCALARS.items():
        setattr(_profile_cache, name, initial)
    _profile_cache._TABLE = None
    _api._LAST_CONTRACT = None
    _api._LAST_KERNEL = None
    reset_threads = getattr(_threads, "reset", None)
    if callable(reset_threads):
        reset_threads()
    pe.cache_clear()


def _snapshot(module: Any) -> dict[str, tuple[str, Any]]:
    """快照模块状态：容器拷内容、标量记值（锁之类的对象不碰）。"""
    snapshot: dict[str, tuple[str, Any]] = {}
    for name, value in vars(module).items():
        if isinstance(value, dict):
            snapshot[name] = ("dict", {
                key: (dict(item) if isinstance(item, dict) else item)
                for key, item in value.items()})
        elif isinstance(value, list):
            snapshot[name] = ("list", [
                dict(item) if isinstance(item, dict) else item for item in value])
        elif isinstance(value, set):
            snapshot[name] = ("set", set(value))
        elif isinstance(value, (bool, int, float, str)) or value is None:
            snapshot[name] = ("value", value)
    return snapshot


def _restore(module: Any, snapshot: dict[str, tuple[str, Any]]) -> None:
    """按 :func:`_snapshot` 的副本原地放回（容器对象不换，外部引用继续有效）。"""
    for name, (kind, saved) in snapshot.items():
        current = getattr(module, name, None)
        if kind == "dict" and isinstance(current, dict):
            current.clear()
            current.update(saved)
        elif kind == "list" and isinstance(current, list):
            current[:] = saved
        elif kind == "set" and isinstance(current, set):
            current.clear()
            current.update(saved)
        elif kind == "value":
            setattr(module, name, saved)


@contextlib.contextmanager
def _isolated(profile: bool = True) -> Iterator[str]:
    """临时 CWD + 干净开关 + 干净的进程内状态；退出时全部还原。

    CWD 换到临时树里，是为了让缺省的 ``./.cache`` 落在那里而不是仓库根（画像默认目录
    相对 CWD 解析）。画像开关按参数给：验 ``profile=False`` 的零 IO 要用 ``True``
    （否则关掉之后"没写"什么也证明不了）。
    """
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    snapshots = [(module, _snapshot(module))
                 for module in (_profile_cache, _api, _threads)]
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-tune-equiv-") as root:
            work = os.path.join(root, "work")
            os.makedirs(work)
            os.chdir(work)
            try:
                _prepare_env(profile=profile)
                _reset_state()
                yield work
            finally:
                os.chdir(saved_cwd)
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for module, snapshot in snapshots:
            _restore(module, snapshot)
        pe.cache_clear()


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """屏蔽本契约不关心的提示与原生回落 warning（输出与数值无关）。"""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


# -- CLI 能力探测 ----------------------------------------------------------------
def _cli() -> Any:
    """CLI 模块（懒取，避免本模块 import 期就拖进 argparse/opt_einsum）。"""
    from ..einsum import test as cli

    return cli


def _require_tune() -> Any:
    """``--tune`` 是否已接入 CLI（选项 + 四个入口函数）。"""
    cli = _cli()
    help_text = cli.build_parser().format_help()
    missing = [name for name in ("run_tune", "tune_combos", "tune_conflicts",
                                 "print_tune", "tune_best")
               if not callable(getattr(cli, name, None))]
    has_option = "--tune" in help_text
    if missing or not has_option:
        raise unittest.SkipTest(
            f"等待实现：--tune 尚未落地（--help 里有该选项={has_option}，缺函数={missing}）")
    return cli


def _run_cli(argv: list[str], *, timeout: float = 600.0
             ) -> tuple[int, str, str]:
    """真起一个子进程跑 ``python -m pyqcd.einsum.test``，返回 ``(退出码, stdout, stderr)``。

    用子进程而不是进程内 ``main()``：本组验的是"**退出码** + 有提示"，那是进程级
    契约；进程内调用只能拿到 ``SystemExit.code``，看不到 ``__main__`` 打印的文案。
    两路输出**分开返回**：警告按口径走 stderr，合并之后就分不清它是警告还是正常输出。
    画像开关显式关掉，免得 30 组遍历往仓库根写 ``./.cache``。
    """
    env = dict(os.environ)
    env[_profile_cache.PROFILE_ENV] = "0"
    for name in (_profile_cache.DIR_ENV, _kernel_force.KERNEL_FORCE_ENV,
                 _equivalence.AS_EQUIVALENCE_ENV, _equivalence.EQUIVALENCE_ENV):
        env.pop(name, None)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (
        os.pathsep + existing if existing else "")
    process = subprocess.run(
        [sys.executable, "-B", "-m", "pyqcd.einsum.test", *argv],
        cwd=str(_REPO_ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    return process.returncode, process.stdout, process.stderr


def _fake_timed(durations: dict[int, float], counter: dict[str, int]):
    """假 ``_timed``：调一次被测可调用对象，返回**注入**的秒数（不做真实计时）。

    排序结论只认注入值，所以本契约能断言"哪一组该最优"却完全不测时间。
    """
    def fake(call: Any, warmup: int, rounds: int, sync: Any = None) -> float:
        # ``sync`` 是 0.5.9 起 ``_timed`` 的第 4 个参数（设备同步）。替身必须收下它：
        # 不收会在真实现调用时抛 TypeError，而报错看起来像"tune 全挂了"。
        index = counter["index"]
        counter["index"] = index + 1
        call()
        return durations[index]

    return fake


# -- 坏组合不留配置行（2026-11-17 口径）--------------------------------------
@contextlib.contextmanager
def _tune_grid(combos: Sequence[tuple[str, str, str]]
               ) -> Iterator[list[tuple[str, str, str]]]:
    """把 ``cli.tune_combos`` 换成**给定的小网格**（其余实现细节一律不碰）。

    换成固定网格而不是收窄某个轴，是因为"哪几组是坏的"必须与"留下哪一行"一一对应；
    网格本身（三轴取值、上限）由第 1/2 条单独钉，这里只借它当输入。
    """
    cli = _cli()
    # 替身必须收 ``*args, **kwargs``：``run_tune`` 走 ``_tune_combos_for(...)``，
    # 它按新签名 ``tune_combos(language, equation=...)`` 调用（2026-11-18 起）。
    with patch.object(cli, "tune_combos", lambda *args, **kwargs: list(combos)):
        yield list(combos)


def _scripted_timed(seconds: dict[int, float],
                    errors: dict[int, BaseException] | None = None):
    """假 ``_timed``：每组**先真跑一次**，再按注入抛异常或返回注入的秒数。

    ``call()`` **必须在抛之前**跑：本组问的是"失败的那一组有没有留下配置行"，只有
    "先真的跑过、再失败"才测得到"不许留下"。返回值只来自注入表，因此仍然没有计时断言。
    """
    counter = {"index": 0}

    def fake(call: Any, warmup: int, rounds: int, sync: Any = None) -> float:
        # 同 ``_fake_timed``：``sync`` 必须收下（0.5.9 的 ``_timed`` 有第 4 个参数）。
        index = counter["index"]
        counter["index"] = index + 1
        call()
        error = (errors or {}).get(index)
        if error is not None:
            raise error
        return seconds.get(index, 0.001)

    return fake


def _profile_disk_rows(path: str | None) -> list[dict[str, str]]:
    """读配置表的**数据行**（文件不存在就是空表，表头不计）。"""
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _run_tune_small(work: str, tag: str, combos: Sequence[tuple[str, str, str]], *,
                    timed: Any, contract: Any = None
                    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """在临时子目录里跑一次小网格 ``run_tune``，返回 ``(tune, summary, 盘上行)``。

    每组用独立目录：配置目录一换就换一张表，同一条用例里的几次运行（正对照/注入组）
    不会互相读到对方的行。``profile=True`` 由本模块显式打开，不依赖环境变量。

    ``isolate=False`` 是**必须**的：本模块用 ``patch.object(cli, "_timed", ...)`` 与
    ``patch.object(pe, "contract", ...)`` 注入假耗时与假数值，而子进程路线（``--tune``
    的缺省）根本不在本进程里调它们——注入会静默失效，断言随之失去区分力。
    """
    cli = _cli()
    directory = os.path.join(work, tag)
    os.makedirs(directory, exist_ok=True)
    cache_dir = os.path.join(directory, "cache")
    with contextlib.ExitStack() as stack:
        stack.enter_context(_tune_grid(combos))
        stack.enter_context(patch.object(cli, "_timed", timed))
        if contract is not None:
            stack.enter_context(patch.object(pe, "contract", contract))
        stack.enter_context(_quiet())
        tune = cli.run_tune(equation=TUNE_EQUATION, shapes=TUNE_SHAPES,
                            warmup=1, rounds=1, profile=True,
                            profile_kwargs={"profile": True,
                                            "profile_dir": cache_dir},
                            isolate=False)
    summary = _profile_cache.summary()
    return tune, summary, _profile_disk_rows(summary.get("file"))


def _assert_only_the_winner_row_was_written(
        tune: dict[str, Any], summary: dict[str, Any],
        rows: Sequence[dict[str, str]], *, abandoned: str) -> None:
    """三条"坏组合不留配置行"共用的断言：表里只有**胜出**的那一行。

    ``tune_session`` 让遍历期间一个字都不落盘（§3.1），``write_winner`` 在最后只写
    胜出的那一组并删同键旧行（§3.3/§7）——所以"坏组合没留下"与"表里恰好 1 行"是同一件事。
    """
    assert summary["keys"] == 1, ("配置表只该有 1 个键（原方程）", summary)
    assert summary["rows"] == 1, ("遍历结束只该写下胜出的那一行", summary)
    assert len(rows) == 1, ("磁盘上必须恰好 1 行", rows)
    best = tune["best"]
    assert best is not None, tune
    row = rows[0]
    assert str(row.get("optimize")) == str(best.get("optimize")), (
        "写下的必须是胜出的那一组", row, best)
    assert str(row.get("wall_ms") or ""), ("胜出行必须带 wall_ms", row)
    if abandoned != str(best.get("optimize")):
        assert abandoned not in {str(item.get("optimize")) for item in rows}, (
            "被放弃的组合的配置行仍在表里", rows)



# ── 5. --tune 忽略它自己遍历的三维（警告，不报错）────────────────────────────
#: ``--tune`` 冲突参数用例的基线命令：小方程、关画像、固定中文——只为看"警告 + 网格"。
_TUNE_CLI_BASE = ("--equation", "ij,jk->ik", "--shape", "[4,4],[4,4]->[4,4]",
                  "--device", "cpu", "--compare", "none", "--warmup", "1",
                  "--rounds", "1", "--language", "zh-cn")


def _tune_rows_of(report_path: str) -> list[dict[str, Any]]:
    """读 CLI 的 ``--json`` 报告里的 tune 行（表格本身只打前 5 行 + 最优行）。

    "网格里还有没有 ``dedicated`` 组合"问的是**跑了哪些组**，不是"打印了哪几行"；
    非 verbose 的表天然截断，用 JSON 里的完整行才问得准。
    """
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    rows = report.get("tune")
    return [dict(row) for row in rows] if isinstance(rows, list) else []


def test_tune_warns_and_ignores_the_axes_it_sweeps_itself():
    """``--tune --kernel-force …`` / ``--tune --as-equivalence force``：警告 + 忽略。

    退出码仍是 0：这是**可用性**问题（用户多写了一个自己会给的维度），不是数值或
    参数错误。但"忽略"必须说出来，且遍历网格一格都不许少——否则"最优"是在一张
    被悄悄改过的网格上选的，报告里看不出来。
    """
    cli = _require_tune()
    grid = list(cli.tune_combos())
    cases = (
        (("--kernel-force", "dedicated"), "kernel-force", "kernel_force",
         {combo[0] for combo in grid}),
        (("--as-equivalence", "force"), "as-equivalence", "as_equivalence",
         {combo[1] for combo in grid}),
    )
    for extra, label, column, expected in cases:
        with tempfile.TemporaryDirectory(prefix="pyqcd-tune-cli-") as root:
            report_path = os.path.join(root, "report.json")
            code, stdout, stderr = _run_cli(
                [*_TUNE_CLI_BASE, "--tune", *extra, "--json", report_path])
            # 旧口径把撞车当用法错误（退出码 2、文案里有"会自行遍历"、但**没有**
            # "已忽略"）。只有这一种才跳过；一旦出现了"已忽略"，就是新口径在手，
            # 后面所有断言都必须真跑（别让"警告文案里也有会自行遍历"把失败洗成跳过）。
            if (code != 0 and "已忽略" not in stderr
                    and "会自行遍历" in stdout + stderr):
                raise unittest.SkipTest(
                    f"等待实现：--tune 仍把 {label} 当用法错误拒绝"
                    "（退出码非 0 且没有'已忽略'警告），「警告 + 忽略」尚未落地")
            assert code == 0, (
                label, "冲突参数必须降级成警告：退出码仍为 0", code,
                (stdout + stderr)[-800:])
            warning = [line for line in stderr.splitlines() if "已忽略" in line]
            assert warning, (
                label, "stderr 必须说明该取值已被忽略", stderr[-1200:])
            assert any(label in line or label.replace("-", "_") in line
                       for line in warning), (label, warning)
            # 表格本身照旧要打出来（警告不是"这次不遍历了"的另一种说法）。
            assert "tune hdr" in stderr and "tune best :" in stderr, (
                label, "遍历表仍必须打印", stderr[-1200:])
            rows = _tune_rows_of(report_path)
            assert len(rows) == len(grid), (
                "被忽略的取值不得改变网格：--tune 仍遍历它自己那三轴",
                len(rows), len(grid))
            values = {str(row.get(column)) for row in rows}
            assert values == expected, (
                "网格必须原样保留（含被忽略取值对应的那一档）", label, values, expected)


# ── 6. --tune 的组合数上限：超了报错，不许静默截断 ──────────────────────────
def test_tune_refuses_more_than_the_combination_cap():
    """把某一维加宽到超过 30 组：必须报用法错误（退出码 2），且不跑任何一组。"""
    cli = _require_tune()
    limit = int(getattr(cli, "TUNE_MAX_COMBOS", 30))
    assert limit == 30, ("冻结口径：--tune 的硬上限是 30 组", limit)

    combos = list(cli.tune_combos())
    bundle = (len(cli.TUNE_KERNEL_FORCE) * len(cli.TUNE_AS_EQUIVALENCE)
              * len(cli.TUNE_OPTIMIZERS))
    # 2026-11-18 起缺省清单**不再**是三轴笛卡尔积：它按分支给**有序清单**
    # （B1/B2 = 22、B3 = 21，逐项顺序见 `_einsum_tune_plan_contract`）。
    # 这里只钉"无重复、不超过上限、且真的比笛卡尔积短"，组数与顺序由那条契约负责
    # ——两处都断言同一件事，早晚会打架。
    assert len(combos) == len(set(combos)) <= limit, (
        "缺省清单不得重复、且不超过上限", len(combos), len(set(combos)), limit)
    assert 0 < len(combos) < bundle, (
        "有序清单必须真的比笛卡尔积短（否则'完全替换'没有发生）",
        len(combos), bundle)

    # 构造一个**必然**超过上限的清单：把 ``_tune_axes.tune_plan`` 换成返回超长清单的
    # 假实现——注入点必须是那个纯函数，否则验的是替身自己而不是真正的上限校验。
    base = list(cli.tune_combos(cli.TUNE_BRANCH_NONE))
    huge = [base[index % len(base)] for index in range(limit + 5)]
    expected = len(huge)
    assert expected > limit, ("本用例必须构造出超限清单", expected, limit)

    with patch.object(cli._tune_axes, "tune_plan",
                      lambda *args, **kwargs: list(huge)):
        try:
            got = list(cli.tune_combos())
        except SystemExit as exc:
            code = getattr(exc, "code", None)
            message = getattr(exc, "message", None) or str(exc)
            assert code == 2, ("超上限是用法错误（退出码 2）", code, message)
            assert str(limit) in message, ("必须报出上限", message)
            assert str(expected) in message, (
                "必须报出实际组数（否则看不出有没有被截断）", message)
        else:
            raise AssertionError(
                f"超过 {limit} 组必须报用法错误，不得静默截断；"
                f"实际返回了 {len(got)} 组")

        # 上限必须在**构造操作数/跑任何一组之前**生效：否则用户白等一轮才发现写错。
        seen: list[dict[str, Any]] = []
        with patch.object(pe, "contract", lambda *a, **k: seen.append(k)):
            try:
                cli.run_tune(equation=TUNE_EQUATION, shapes=TUNE_SHAPES,
                             warmup=1, rounds=1)
            except SystemExit as exc:
                assert getattr(exc, "code", None) == 2, exc
            else:
                raise AssertionError("run_tune 在超限网格上本应报用法错误")
        assert seen == [], ("超限必须在跑任何一组之前拦下", len(seen))


# ── 7. --tune 最优选择：最快但 MISMATCH 的不得当选 ──────────────────────────
def test_tune_excludes_a_mismatching_combination_from_best():
    """注入假耗时：跑最快的那组数值不符 → 它必须被标 MISMATCH 且不参与选优。"""
    cli = _require_tune()
    if not callable(getattr(cli, "_timed", None)):
        raise unittest.SkipTest(
            "等待实现：缺 pyqcd.einsum.test._timed（无法注入假耗时，"
            "而本契约不许用真实计时做排序结论）")
    combos = list(cli.tune_combos())
    if not combos:
        raise unittest.SkipTest("--tune 的清单为空，最优选择不可验")
    # 从实现自己的清单里挑两组：内核不是"本机可能不可用"的强制档（否则注入的数值
    # 不符会被更早的 KernelUnavailableError 顶掉），且 ``optimize`` 不同。2026-11-18
    # 起清单是**分支化的有序清单**——``as_equivalence=true`` 只出现在 B2 分支
    # （``tune_combos()`` 的缺省清单是 B3，里面全是 false），拿它当区分列会无解。
    available = [index for index, combo in enumerate(combos)
                 if combo[0] not in ("dedicated", "topology")]
    bad = available[0] if available else None
    winner = next((index for index in available
                   if index != bad and combos[index][2] != combos[bad][2]), None)
    if bad is None or winner is None:
        raise unittest.SkipTest(
            "清单里没有 optimize 两档，无法构造数值不符的对照组："
            f"{sorted({combo[2] for combo in combos})}")
    assert bad != winner, (bad, winner)
    bad_combo = combos[bad]

    # 注入的假耗时：MISMATCH 那组最快、最优那组第二快，其余慢一个量级。
    durations = {index: 0.1 + 0.001 * index for index in range(len(combos))}
    durations[bad] = 0.001
    durations[winner] = 0.002

    real_contract = pe.contract
    calls: list[dict[str, Any]] = []

    def fake_contract(equation: str, *operands: Any, **kwargs: Any) -> Any:
        calls.append({"equation": equation, "kwargs": dict(kwargs)})
        label = str(kwargs.get("kernel_force") or "")
        if label in ("dedicated", "topology"):
            # 注入"这个内核在本机不可用"：遍历**不得**因为一组跑不成而中断。
            raise _kernel_force.KernelUnavailableError(
                (label,), equation, {"kupl_dedicated": "injected"})
        result = real_contract(equation, *operands, **kwargs)
        combo = (label, "true" if kwargs.get("as_equivalence") else "false",
                 str(kwargs.get("optimize")))
        if combo == bad_combo:
            return result + 1.0            # rel ≫ atol：注入一次数值不符
        return result

    counter = {"index": 0}
    stderr = io.StringIO()
    with _isolated(profile=False), \
            patch.object(cli, "_timed", _fake_timed(durations, counter)), \
            patch.object(pe, "contract", fake_contract), \
            contextlib.redirect_stderr(stderr):
        tune = cli.run_tune(equation=TUNE_EQUATION, shapes=TUNE_SHAPES,
                            warmup=1, rounds=1,
                            atol=cli.DEFAULT_ATOL, rtol=cli.DEFAULT_RTOL,
                            isolate=False)
        cli.print_tune(tune, verbose=False)
    printed = stderr.getvalue()

    rows = tune["rows"]
    assert len(rows) == len(combos) == counter["index"], (
        "每组都该跑一次（含跑不成的那几组）", len(rows), counter["index"])
    assert tune["compare"] == "none", ("--tune 把 compare 固定成 none", tune["compare"])

    # 不变量：方程/后端不随组合变化（tune 只遍历三个轴）。
    assert {item["equation"] for item in calls} == {TUNE_EQUATION}, (
        "方程必须固定", {item["equation"] for item in calls})
    assert len({str(item["kwargs"].get("backend")) for item in calls}) == 1, (
        "后端必须固定", {str(item["kwargs"].get("backend")) for item in calls})

    bad_row, winner_row = rows[bad], rows[winner]
    assert bad_row["median_ms"] < winner_row["median_ms"], (bad_row, winner_row)
    assert bad_row.get("mismatch") is True, ("rel 超 atol+rtol 必须判 MISMATCH", bad_row)
    assert bad_row["rel"] is not None and bad_row["rel"] > (
        cli.DEFAULT_ATOL + cli.DEFAULT_RTOL), bad_row
    assert winner_row.get("mismatch") in (None, False), winner_row
    assert [row for row in rows if row.get("mismatch")] == [bad_row], (
        "MISMATCH 判定只该挑出注入的那一组（其余 rel 都在容差内）")

    # 若把"最快"当唯一判据，当选的正是那个 MISMATCH 组——本用例因此有区分力。
    measured = [row for row in rows if row.get("median_ms") is not None]
    assert min(measured, key=lambda row: row["median_ms"]) is bad_row, (
        "注入的假耗时必须让 MISMATCH 组成为最快的那一组")

    assert tune["best"] is not None, tune["best"]
    assert tune["best"] is not bad_row, ("MISMATCH 组不得被选为 best", tune["best"])
    assert tune["best"]["median_ms"] == winner_row["median_ms"] == 2.0, (
        "best 必须是可用组里最快的那一个（注入的假耗时）", tune["best"])
    assert (tune["best"]["kernel_force"], tune["best"]["as_equivalence"],
            tune["best"]["optimize"]) == combos[winner], tune["best"]

    # 跑不成的组合：记 error、不中断遍历（注入的 dedicated/topology 全不可用）。
    errored = [row for row in rows if row.get("error")]
    assert len(errored) == sum(1 for combo in combos
                               if combo[0] in ("dedicated", "topology")), (
        "KernelUnavailableError 只该记 error，不得中断遍历", len(errored))
    assert all(row.get("median_ms") is None for row in errored), errored[:1]

    # 非 verbose：只给表头 + 最优行（不出明细行），且最优行点名的是 winner。
    best_lines = [line for line in printed.splitlines()
                  if line.startswith("tune best :")]
    assert len(best_lines) == 1, printed[-1200:]
    best_line = best_lines[0]
    assert f"kernel_force={combos[winner][0]}" in best_line, best_line
    assert f"as_equivalence={combos[winner][1]}" in best_line, best_line
    assert f"optimize={combos[winner][2]}" in best_line, best_line
    assert combos[bad][2] != combos[winner][2], (
        "对照组与最优组必须在打印字段上可区分", combos[bad], combos[winner])
    assert f"optimize={combos[bad][2]}" not in best_line, (
        "MISMATCH 组不得出现在最优行", best_line)
    assert not [line for line in printed.splitlines()
                if line.startswith("tune row")], (
        "非 verbose 不该打明细行", printed[-1200:])
    mismatched = [row for row in rows if row.get("mismatch")]
    assert f"MISMATCH={len(mismatched)}" in printed, (
        "被排除的组合数必须如实给出", len(mismatched), printed[-1200:])


# ── 4. 坏组合不留配置行（超时 / 异常 / MISMATCH，三类都是注入）──────────────
def test_tune_keeps_no_row_for_a_timed_out_combination():
    """注入超时：那一组只被标出来，表里只写**胜出**的一行（§3.1/§3.3）。

    注入点是 ``_timed``：先 ``call()`` 一次（这一组因此真的跑过），再抛
    ``TimeoutError``——这正是真实超时的形态（算了一半/算完但没在预算内）。
    ``tune_session`` 让遍历期间一个字都不落盘，所以"它跑过"本身不会留下痕迹；
    最后 ``write_winner`` 只写胜出的那一组。
    """
    _require_tune()
    bad = len(SMALL_TUNE_COMBOS) - 1
    abandoned = SMALL_TUNE_COMBOS[bad][2]
    with _isolated(profile=True) as work:
        # 正对照：同一网格不注入时也只写一行（"每键一行"是 §2.2/§7 的硬约束）。
        control_tune, control_summary, control_rows = _run_tune_small(
            work, "control", SMALL_TUNE_COMBOS, timed=_scripted_timed({}))
        _assert_only_the_winner_row_was_written(
            control_tune, control_summary, control_rows,
            abandoned=control_tune["best"]["optimize"])

        tune, summary, rows = _run_tune_small(
            work, "timeout", SMALL_TUNE_COMBOS,
            timed=_scripted_timed({}, {bad: TimeoutError("injected: per-combo timeout")}))

        assert len(tune["rows"]) == len(SMALL_TUNE_COMBOS), (
            "超时只该放弃那一组，不得整段作废", len(tune["rows"]))
        row = tune["rows"][bad]
        # 超时可以是它自己的分类（``timeout=True``），也可以落进 ``error``；两者都
        # 表示"这一组没有可用的实测"，但必须**留下痕迹**，不许静默当成没跑过。
        assert row.get("timeout") is True or row.get("error") == "TimeoutError", (
            "超时的组合必须被标出来", row)
        assert row["median_ms"] is None, row
        assert tune["best"] is not None and tune["best"] is not row, (
            "超时的组合不得被选为 best", tune["best"])
        _assert_only_the_winner_row_was_written(
            tune, summary, rows, abandoned=abandoned)


def test_tune_keeps_no_row_for_a_raising_combination():
    """注入 ``RuntimeError``：同上——异常组合也不许留下半条记录。"""
    _require_tune()
    bad = len(SMALL_TUNE_COMBOS) - 1
    abandoned = SMALL_TUNE_COMBOS[bad][2]
    with _isolated(profile=True) as work:
        control_tune, control_summary, control_rows = _run_tune_small(
            work, "control", SMALL_TUNE_COMBOS, timed=_scripted_timed({}))
        _assert_only_the_winner_row_was_written(
            control_tune, control_summary, control_rows,
            abandoned=control_tune["best"]["optimize"])

        tune, summary, rows = _run_tune_small(
            work, "raise", SMALL_TUNE_COMBOS,
            timed=_scripted_timed({}, {bad: RuntimeError("injected: combo blew up")}))

        assert len(tune["rows"]) == len(SMALL_TUNE_COMBOS), (
            "单组异常不该中断遍历", len(tune["rows"]))
        row = tune["rows"][bad]
        assert row["error"] == "RuntimeError", row
        assert row["median_ms"] is None, row
        assert tune["best"] is not None and tune["best"] is not row, tune["best"]
        _assert_only_the_winner_row_was_written(
            tune, summary, rows, abandoned=abandoned)


def test_tune_keeps_no_row_for_a_mismatching_combination():
    """``rel`` 超 ``atol``：留不下配置行，且**不得**成为 ``tune best``。

    注入假耗时让 MISMATCH 那一组成为最快的一组——若"最快即最优"，当选的正是它；
    这条因此有区分力（否则"不选它"只是因为别人更快）。
    """
    cli = _require_tune()
    bad = len(SMALL_TUNE_COMBOS) - 1
    abandoned = SMALL_TUNE_COMBOS[bad][2]
    real_contract = pe.contract

    def falsify(equation: str, *operands: Any, **kwargs: Any) -> Any:
        result = real_contract(equation, *operands, **kwargs)
        if str(kwargs.get("optimize")) == abandoned:
            return result + 1.0                # rel ≫ atol：注入一次数值不符
        return result

    # 注入的假耗时让坏组最快：index 0 = 10 ms、1 = 6 ms、2（坏组）= 2 ms。
    seconds = {index: 0.010 - 0.004 * index for index in range(len(SMALL_TUNE_COMBOS))}
    with _isolated(profile=True) as work:
        control_tune, control_summary, control_rows = _run_tune_small(
            work, "control", SMALL_TUNE_COMBOS, timed=_scripted_timed({}))
        _assert_only_the_winner_row_was_written(
            control_tune, control_summary, control_rows,
            abandoned=control_tune["best"]["optimize"])

        tune, summary, rows = _run_tune_small(
            work, "mismatch", SMALL_TUNE_COMBOS,
            timed=_scripted_timed(seconds), contract=falsify)

        tested = tune["rows"]
        row = tested[bad]
        assert row.get("mismatch") is True, ("rel 超容差必须判 MISMATCH", row)
        assert row["rel"] is not None and row["rel"] > (
            cli.DEFAULT_ATOL + cli.DEFAULT_RTOL), row
        assert [item for item in tested if item.get("mismatch")] == [row], (
            "MISMATCH 只该挑出注入的那一组", tested)
        measured = [item for item in tested if item.get("median_ms") is not None]
        assert min(measured, key=lambda item: item["median_ms"]) is row, (
            "注入的假耗时必须让 MISMATCH 组成为最快的那一组")
        assert tune["best"] is not None and tune["best"] is not row, (
            "MISMATCH 组不得被选为 best", tune["best"])
        assert tune["best"]["optimize"] == SMALL_TUNE_COMBOS[len(seconds) - 2][2], (
            "最优必须落到可用组里最快的那一组", tune["best"])
        _assert_only_the_winner_row_was_written(
            tune, summary, rows, abandoned=abandoned)


TESTS = (
    test_tune_warns_and_ignores_the_axes_it_sweeps_itself,
    test_tune_refuses_more_than_the_combination_cap,
    test_tune_excludes_a_mismatching_combination_from_best,
    test_tune_keeps_no_row_for_a_timed_out_combination,
    test_tune_keeps_no_row_for_a_raising_combination,
    test_tune_keeps_no_row_for_a_mismatching_combination,
)


def test_einsum_tune_equivalence_contracts() -> None:
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
            f"{len(failures)} 条等价负标记/--tune 契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"tune/equivalence contracts did not run ({skipped} skipped)")


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
