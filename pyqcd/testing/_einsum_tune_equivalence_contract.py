"""契约：等价负标记（``profile_equivalence.csv``）与 ``--tune`` 遍历口径。

这两件事都是"结果正确却白花时间/选错配置"的类型：标记写错了只是多查一次，遍历选错了
只是报了个更慢的配置。普通数值测试永远碰不到它们，所以把它们固化成能独立失败的断言：

1. **负标记真的会被跳过**：``record_equivalence(key, "none")`` 之后，同一条
   ``contract()`` 调用**一次都不许**碰 ``_equivalence.normalize_equation``，且答案
   仍与 ``np.einsum`` 一致（跳过只该省时间，不许改数值）。
2. **没有标记时照常查**：同一个键不写标记时 spy 必须 >= 1 —— 否则第 1 条说的"跳过"
   就不是门在起作用，而是整条链本来就没查。
3. **``as_equivalence="force"`` 无视负标记**：标记 ``none`` + force 时 spy >= 1；
   结束后标记按**真实结论**刷新——能改写必须是 ``ok``（并带 ``kind``），真的不能
   改写则仍是 ``none``。两条分支都由 :func:`_rewrite_possible` 判定，都会被执行到。
4. **``profile=False`` 零 IO**：既不是读也不是写，连 ``profile_equivalence.csv``
   所在的目录都不建，``equivalence_marker`` 仍为 ``None``。本条自带**正对照**
   （``profile=True`` 同一次调用必须留下标记），否则"没写"是空断言。
5. **``--tune`` 忽略它自己遍历的三维**：``--tune --kernel-force dedicated`` 与
   ``--tune --as-equivalence force`` 都必须**正常退出（0）**，同时在 stderr 给出
   "已忽略"警告，且遍历网格仍是完整的三轴笛卡尔积（``dedicated`` 组合照样在表里）——
   悄悄吞掉用户写下的取值、或把它变成"只跑一组"，报告里都看不出来。
6. **``--tune`` 组合数上限**：把某一维加宽到超过 30 组时必须报用法错误（退出码 2，
   消息里给出实际组数与上限），**绝不静默截断**；且报错发生在构造张量/跑任何一组之前。
7. **``--tune`` 最优选择**：注入假耗时让"最快"的那一组是数值不符（``rel`` 超
   ``atol + rtol``）的——它必须被判 ``mismatch`` 且**不得**成为 ``best``；
   跑不成（``KernelUnavailableError``）的组合只记 ``error``、不中断遍历。
8. **坏组合不留画像行**：被放弃的组合有三类——**超时**（注入 ``TimeoutError``）、
   **抛异常**（注入 ``RuntimeError``）、**数值不符**（``rel`` 超 ``atol``）。
   三者的 ``summary()["rows"]`` 与"网格里根本没有这一组"时**相同**，画像里留下的配置
   集合也相同；MISMATCH 那组还不许成为 ``tune best``。每类都配一条正对照（同一网格
   不注入时那一行确实会留下），因为"没写"只有在对的地方写过才有意义。
9. **tune 期间强制写**：``_api.set_profile_tune(True)``（库若只给属性，就是
   ``_api._PROFILE_TUNE``——实现每次记录都现读它）之后，同一个键上的多组配置都会落盘
   （``summary()["rows"] > 1``，``flush()`` 后 ``profile.csv`` 里也有多行）；
   ``False`` 时回到稀疏口径——同键多组**只落盘一行**。用临时目录 + 小方程，
   不碰生产尺寸。

计时口径
--------
本模块**不做任何计时断言**：

* 第 7 条把 :func:`pyqcd.einsum.test._timed` 换成"调一次被测可调用对象、返回注入值"的
  假实现（:func:`_fake_timed`），排序结论只来自注入的假耗时；
* 第 8 条的超时/异常同样来自注入（:func:`_scripted_timed` 会先真跑一次再抛，否则
  "没写画像"是空断言），跑的是 6×6 的小方程而不是生产尺寸；
* 第 9 条把 ``_api._profile_clock`` 钉成"没有起点"（``compute_ms`` 恒为 0），只为让
  "5% 稀疏门"可预测——小方程的真实耗时只差几微秒，靠墙钟判断落几行必然偶发。

实现未落地时的口径
-----------------
允许 :class:`unittest.SkipTest`，但**每条跳过都写明"等待实现"并给出缺失的能力**；
不许让断言静默通过：凡是"没写/没查"的负向断言都配一条正向对照（第 4、8、9 条最典型）。

临时目录 + 恢复
---------------
每个用例都在临时树里跑（CWD 也换过去），退出时把环境变量与两个模块的进程内状态原样
放回；仓库根不会留下 ``.cache``。第 9 条额外用 ``try/finally`` 把 tune 强制写开关
关回去——它是过程级状态，留在 ``True`` 会让后面的调用（以及别的用例）一直强制写盘。

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

#: 冻结的文件名：负标记表。实现里也有同名常量（见 :func:`_require_markers`）。
MARKER_NAME = "profile_equivalence.csv"

#: 语料外、`有`等价写法的方程（实测 ``kind="operand-order"``）：用来验"跳过"与
#: "force 刷新成 ok"。既不是逐元素、也不走形状特化，因此真的会进等价门。
EQUIV_EQUATION = "menf,cadb,ac,em,Mbd,Nfn->MN"
#: 语料外、`没有`等价写法的方程（实测 ``normalize_equation`` 返回 ``None``）：
#: 用来验"force 之后仍必须是 none"这一支。
NO_EQUIV_EQUATION = "ij,jk,kl->il"

#: ``--tune`` 的方程与形状：小到跑 30 组也不心疼（且耗时全部来自注入）。
TUNE_EQUATION = "ij,jk,kl->il"
TUNE_SHAPES = [(6, 6), (6, 6), (6, 6)]

#: 本模块自己用的小网格：同一个键、同一个通用内核，**只有 ``optimize`` 不同**，
#: 于是三组对应三条互不相同的配置行。用它跑第 8/9 条（不碰 30 组的生产网格）。
SMALL_TUNE_COMBOS = (("generic", "false", "auto"),
                     ("generic", "false", "greedy"),
                     ("generic", "false", "optimal"))

#: ``--tune`` 期间强制写的进程内标志（:func:`pyqcd.einsum._api.set_profile_tune`
#: 的落点）。第 9 条必须**经 setter** 切换，直接改写模块属性不算契约。
TUNE_FLAG_ATTR = "_PROFILE_TUNE"

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

#: ``_profile_cache`` 的已知标量状态（容器由 :func:`_clear_mutable_state` 统一清）。
_PROFILE_SCALARS = {
    "_configured_dir": "", "_ready": False, "_dir": "", "_dir_ok": False,
    "_last_error": None, "_notice_done": False, "_loaded": False,
    "_loaded_rows": 0, "_skipped_rows": 0, "_truncated_fields": 0,
    "_dropped_rows": 0, "_disk_rows": -1, "_rewrite_rows": -1,
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
    """把两个模块的进程内状态清到"没配置、没加载、没标记"的初始态。"""
    for module in (_profile_cache, _api, _threads):
        _clear_mutable_state(module)
    for name, initial in _PROFILE_SCALARS.items():
        setattr(_profile_cache, name, initial)
    _api._PROFILE_READY = None
    # ``--tune`` 的强制写标志也是跨用例会累积的进程状态：上一条用例若（哪怕异常地）
    # 把它留在 True，下一条的"稀疏口径"就会量到别人的设置。
    if hasattr(_api, TUNE_FLAG_ATTR):
        setattr(_api, TUNE_FLAG_ATTR, False)
    if hasattr(_api, "_LAST_CONTRACT"):
        _api._LAST_CONTRACT = None
    if hasattr(_api, "_LAST_KERNEL"):
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


# -- 方程语料与断言基元 ----------------------------------------------------------
def _operands(equation: str, *, seed: int = 20261117) -> list[np.ndarray]:
    """按方程左侧生成标签长度互不相同的 complex128 操作数。"""
    rng = np.random.default_rng(seed)
    operands: list[np.ndarray] = []
    for term in equation.split("->", 1)[0].split(","):
        shape = tuple(LABEL_SIZES[label] for label in term)
        value = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        operands.append(np.ascontiguousarray(value.astype(np.complex128)))
    return operands


def _relative_error(actual: Any, expected: Any) -> float:
    """相对最大偏差；全零答案用 1 作尺度（与库内同口径）。"""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    scale = float(np.max(np.abs(expected_array))) or 1.0
    return float(np.max(np.abs(actual_array - expected_array))) / scale


def _assert_matches_numpy(actual: Any, expected: Any, label: str) -> None:
    """逐项断言形状、dtype 与 rel <= 1e-12：跳过/改写都不许改答案。"""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape, (
        label, actual_array.shape, expected_array.shape)
    assert actual_array.dtype == expected_array.dtype, (
        label, actual_array.dtype, expected_array.dtype)
    error = _relative_error(actual_array, expected_array)
    assert error <= 1e-12, (label, error)


@contextlib.contextmanager
def _normalize_spy() -> Iterator[list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    """包住真实 ``normalize_equation``，记录等价步骤到底有没有被查。"""
    original = _equivalence.normalize_equation
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    with patch.object(_equivalence, "normalize_equation", new=spy):
        yield calls


def _profile_key(equation: str, operands: list[np.ndarray],
                 backend: str = "numpy") -> dict[str, Any]:
    """按 ``_api._profile_begin`` 的**同一套**口径构造画像键。

    标记表与主表共用 :data:`._profile_cache.KEY_FIELDS`，所以键必须逐字段一致，
    否则标记写在一处、查在另一处，"跳过"永远不会发生。这里直接借用实现自己的三个
    键构造器（设备、dtype、形状归一化），不另写一套归一化。
    """
    for name in ("_profile_devices", "_profile_dtype"):
        if not callable(getattr(_api, name, None)):
            raise unittest.SkipTest(
                f"等待实现：_api.{name}（画像键构造器）缺失，标记键无法精确构造")
    device, user_device = _api._profile_devices(operands, backend)
    return {
        "user_device": user_device,
        "backend": backend,
        "dtype": _api._profile_dtype(operands),
        "equation": "".join(equation.split()),
        "shape": _profile_cache.format_shape([operand.shape for operand in operands]),
        "device": device,
    }


def _rewrite_possible(equation: str) -> bool:
    """该方程**真的**能被改写成等价形式吗（与 ``_contract_via_equivalence`` 同判据）。

    :func:`._equivalence.normalize_equation` 返回 ``identity`` 或没有三个置换字段之一
    时，等价门会直接返回 ``None`` —— 那种方程查过之后结论就是"不能改写"，标记必须
    如实停在 ``none``。
    """
    mapping = _equivalence.normalize_equation(equation, use=False)
    if mapping is None or mapping.kind == "identity":
        return False
    return bool(mapping.operand_order or mapping.operand_permutations
                or mapping.output_permutation)


def _marker_state(key: dict[str, Any]) -> str | None:
    """标记的 ``state``（没有标记返回 ``None``；缺 ``state`` 字段也算没有）。"""
    marker = _profile_cache.equivalence_marker(key)
    if not isinstance(marker, dict):
        return None
    state = marker.get("state")
    return None if state is None else str(state).strip().lower()


# -- 能力探测：实现没落地时明确 Skip，而不是让断言静默通过 ----------------------
def _require_markers() -> None:
    """标记 API 是否已落地；顺带钉住冻结的文件名。"""
    missing = [name for name in ("equivalence_marker", "record_equivalence")
               if not callable(getattr(_profile_cache, name, None))]
    if missing:
        raise unittest.SkipTest(
            f"等待实现：_profile_cache 缺 {missing}（等价负标记表尚未落地）")
    declared = getattr(_profile_cache, "EQUIVALENCE_NAME", MARKER_NAME)
    assert declared == MARKER_NAME, (
        "标记表文件名是冻结口径", declared, MARKER_NAME)


def _skip_is_effective(work: str) -> bool:
    """行为探测：写 "none" 之后同一条调用是否真的不再查（名字不作为判据）。"""
    directory = os.path.join(work, "probe-gate")
    operands = _operands(EQUIV_EQUATION)
    _profile_cache.configure(directory)
    key = _profile_key(EQUIV_EQUATION, operands)
    if not _profile_cache.record_equivalence(key, "none"):
        return False
    with _normalize_spy() as calls, _quiet():
        pe.contract(EQUIV_EQUATION, *operands, profile=True, profile_dir=directory)
    return not calls


def _require_skip_gate(work: str) -> None:
    """``state=none`` 的廉价门是否已接进 ``_api``。"""
    named = (callable(getattr(_api, "_equivalence_marked_none", None))
             and callable(getattr(_api, "_record_equivalence", None)))
    if named or _skip_is_effective(work):
        return
    raise unittest.SkipTest(
        "等待实现：_api 尚未接入等价负标记门"
        "（state=none 时仍会调用 _equivalence.normalize_equation）")


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
    def fake(call: Any, warmup: int, rounds: int) -> float:
        index = counter["index"]
        counter["index"] = index + 1
        call()
        return durations[index]

    return fake


# -- 坏组合不写画像 / tune 期间强制写（2026-11-17 口径）----------------------
def _tune_switch() -> Any:
    """取"tune 期间强制记录"的开关：优先库的 ``set_profile_tune``，再退回它读的属性。

    ``_api._profile_record`` 每次记录时**现读** :data:`TUNE_FLAG_ATTR`，所以两种写法
    作用在同一条路径上；实现若提供了 setter 就必须用它（那是公开口径），只有实现只给
    属性时才直接改属性——直接改不是"绕过契约"，而是沿着实现自己读的那一处切。
    """
    setter = getattr(_api, "set_profile_tune", None)
    if callable(setter):
        return setter
    if hasattr(_api, TUNE_FLAG_ATTR):
        return lambda on: setattr(_api, TUNE_FLAG_ATTR, bool(on))
    raise unittest.SkipTest(
        f"等待实现：_api.set_profile_tune() / _api.{TUNE_FLAG_ATTR} 都不存在，"
        "tune 期间的强制写开关无法切换")


def _require_bad_combo_drop() -> None:
    """坏组合"不留画像行"的落地标志：``drop()``（或上面那个开关）存在。

    行为本身仍由断言兜着：只要这批能力露了头，本组就真跑，坏组合被留下来即失败。
    两个入口都缺席才跳过——那种情况下实现连抹掉一行的手段都还没有。
    """
    if callable(getattr(_profile_cache, "drop", None)):
        return
    if callable(getattr(_api, "set_profile_tune", None)):
        return
    if hasattr(_api, TUNE_FLAG_ATTR):
        return
    raise unittest.SkipTest(
        "等待实现：_profile_cache.drop() 与 tune 强制写开关均未落地，"
        "坏组合（超时/异常/MISMATCH）目前仍会留下画像行")


@contextlib.contextmanager
def _tune_grid(combos: Sequence[tuple[str, str, str]]
               ) -> Iterator[list[tuple[str, str, str]]]:
    """把 ``cli.tune_combos`` 换成**给定的小网格**（其余实现细节一律不碰）。

    换成固定网格而不是收窄某个轴，是因为"哪几组是坏的"必须与"哪几行该消失"一一对应；
    网格本身（三轴取值、上限）由第 5/6 条单独钉，这里只借它当输入。
    """
    cli = _cli()
    with patch.object(cli, "tune_combos", lambda language="zh": list(combos)):
        yield list(combos)


@contextlib.contextmanager
def _frozen_compute_ms() -> Iterator[None]:
    """把画像记下的 ``compute_ms`` 钉成 0，让"5% 稀疏门"可预测（**不读墙钟**）。

    稀疏口径要求"新键，或比该键当前最优快过 5%"才落盘；小方程的真实耗时只差几微秒，
    快慢由噪声决定——那样的断言会偶发。``_profile_clock`` 返回 ``None`` 时
    ``_profile_end`` 不写 ``compute_ms``（保持初值 0），于是除新键外没有一行"够格"。
    它只影响画像记录，不参与任何数值、排序或选优结论。
    """
    if not callable(getattr(_api, "_profile_clock", None)):
        raise unittest.SkipTest(
            "等待实现：_api._profile_clock 缺失，无法把 compute_ms 钉成 0"
            "（本模块不做计时断言）")
    with patch.object(_api, "_profile_clock", lambda info: None):
        yield


def _scripted_timed(seconds: dict[int, float],
                    errors: dict[int, BaseException] | None = None):
    """假 ``_timed``：每组**先真跑一次**，再按注入抛异常或返回注入的秒数。

    ``call()`` **必须在抛之前**跑：坏组合的画像行是 ``contract()`` 成功返回时记下的，
    只有"先记下、后失败"才测得到"失败要把它抹掉"。一上来就抛的话，``summary()["rows"]``
    本来就不会多一行——那种断言是空的。返回值只来自注入表，因此仍然没有计时断言。
    """
    counter = {"index": 0}

    def fake(call: Any, warmup: int, rounds: int) -> float:
        index = counter["index"]
        counter["index"] = index + 1
        call()
        error = (errors or {}).get(index)
        if error is not None:
            raise error
        return seconds.get(index, 0.001)

    return fake


def _profile_configs() -> set[tuple[str, str]]:
    """画像索引里现存的 ``(kernel, optimize)`` 配置集合。

    只看 ``summary()["rows"]`` 的**条数**会被"每键最快 K 条"的裁剪盖住（坏组合挤掉
    别人、条数不变）；按配置集合比对才能说清"留下的到底是哪几行"。
    """
    index = getattr(_profile_cache, "_index", None)
    if not isinstance(index, dict):
        raise unittest.SkipTest(
            "等待实现：_profile_cache._index 缺失，无法核对坏组合留下了哪几行")
    return {(str(kernel), str(optimize))
            for slot in index.values()
            for (kernel, optimize) in slot}


def _profile_disk_rows(path: str | None) -> list[dict[str, str]]:
    """读 ``profile.csv`` 的**数据行**（文件不存在就是空表，表头不计）。"""
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle) if row.get("kernel")]


def _run_tune_small(work: str, tag: str, combos: Sequence[tuple[str, str, str]], *,
                    timed: Any, contract: Any = None
                    ) -> tuple[dict[str, Any], dict[str, Any], set[tuple[str, str]]]:
    """在临时子目录里跑一次小网格 ``run_tune``，返回 ``(tune, summary, 配置集合)``。

    每组用独立目录：画像目录一换就换一张表，同一条用例里的三次运行（正对照/对照组/
    注入组）不会互相读到对方的行。``profile=True`` 由本模块显式打开，不依赖环境变量。
    """
    cli = _cli()
    directory = os.path.join(work, tag)
    os.makedirs(directory, exist_ok=True)
    cache_dir = os.path.join(directory, "cache")
    with contextlib.ExitStack() as stack:
        stack.enter_context(_tune_grid(combos))
        stack.enter_context(_frozen_compute_ms())
        stack.enter_context(patch.object(cli, "_timed", timed))
        if contract is not None:
            stack.enter_context(patch.object(pe, "contract", contract))
        stack.enter_context(_quiet())
        tune = cli.run_tune(equation=TUNE_EQUATION, shapes=TUNE_SHAPES,
                            warmup=1, rounds=1, profile=True,
                            profile_kwargs={"profile": True,
                                            "profile_dir": cache_dir})
    return tune, _profile_cache.summary(), _profile_configs()


def _assert_only_the_abandoned_combo_disappeared(
        full_summary: dict[str, Any], full_configs: set[tuple[str, str]],
        control_summary: dict[str, Any], control_configs: set[tuple[str, str]],
        summary: dict[str, Any], configs: set[tuple[str, str]],
        abandoned: str) -> None:
    """三条"坏组合不写画像"共用的断言：留下的是"网格里没有它"那一套行。

    ``full`` 是**同一网格不注入**的正对照（那一行确实会留下），``control`` 是
    "网格里根本没有这一组"，``summary/configs`` 是注入后的结果。
    """
    assert full_summary["rows"] == len(SMALL_TUNE_COMBOS), (
        "正对照：不注入时整张地图都该留在画像里", full_summary["rows"])
    assert abandoned in {optimize for _kernel, optimize in full_configs}, (
        "正对照：被放弃的那一组在不注入时本来会留下", full_configs)
    assert len(full_configs) == len(SMALL_TUNE_COMBOS), full_configs

    assert summary["rows"] == control_summary["rows"], (
        "被放弃的组合不许留下画像行", summary["rows"], control_summary["rows"])
    assert configs == control_configs, (
        "留下的配置集合必须与「网格里没有这一组」时一致", configs, control_configs)
    assert abandoned not in {optimize for _kernel, optimize in configs}, (
        "被放弃的组合的配置行仍在画像里", configs)


# ── 1. 负标记 → 跳过等价步骤 ─────────────────────────────────────────────────
def test_negative_marker_skips_the_equivalence_step():
    """``state=none`` 的标记必须让等价步骤**一次都不查**，答案仍与 numpy 一致。"""
    _require_markers()
    with _isolated() as work:
        _require_skip_gate(work)
        # 该方程本来**有**等价写法：所以"没查"才是跳过，而不是"查了也没有"。
        assert _rewrite_possible(EQUIV_EQUATION), (
            "本用例的方程必须有等价写法，否则跳过无法观测")

        operands = _operands(EQUIV_EQUATION)
        expected = np.einsum(EQUIV_EQUATION, *operands)
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _profile_key(EQUIV_EQUATION, operands)
        assert _profile_cache.record_equivalence(key, "none"), (
            "首次写 none 标记必须真的排队落盘", key)
        assert _marker_state(key) == "none", (
            "写进去的标记必须能被同一套键读回来", _profile_cache.equivalence_marker(key))

        with _normalize_spy() as calls, _quiet():
            actual = pe.contract(EQUIV_EQUATION, *operands,
                                 profile=True, profile_dir=directory)

        assert calls == [], (
            "state=none 时不得调用 _equivalence.normalize_equation", len(calls), calls[:1])
        _assert_matches_numpy(actual, expected, "negative-marker skip")


# ── 2. 无标记 → 照常查 ──────────────────────────────────────────────────────
def test_without_a_marker_the_equivalence_step_still_runs():
    """同一个键不写标记时 spy 必须 >= 1（否则第 1 条的"跳过"与门无关）。"""
    _require_markers()
    with _isolated() as work:
        operands = _operands(EQUIV_EQUATION)
        expected = np.einsum(EQUIV_EQUATION, *operands)
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _profile_key(EQUIV_EQUATION, operands)
        # 前提：这个键此刻确实没有任何标记（否则下面的 >=1 会被门挡住）。
        assert _profile_cache.equivalence_marker(key) is None, (
            "新目录里不该有标记", _profile_cache.equivalence_marker(key))

        with _normalize_spy() as calls, _quiet():
            actual = pe.contract(EQUIV_EQUATION, *operands,
                                 profile=True, profile_dir=directory)

        assert len(calls) >= 1, ("没有标记时必须照常查等价写法", len(calls))
        assert calls[0][0][0] == EQUIV_EQUATION, calls[0][0][:1]
        _assert_matches_numpy(actual, expected, "no-marker lookup")


# ── 3. force → 无视负标记并刷新标记 ─────────────────────────────────────────
def test_force_ignores_the_negative_marker_and_refreshes_it():
    """``as_equivalence="force"``：spy >= 1，且标记按真实结论刷新（ok / 仍 none）。"""
    _require_markers()
    cli = _cli()
    if getattr(_api, "_AS_EQ_FORCE", None) != "force" and \
            "force" not in getattr(cli, "AS_EQUIVALENCE_MODES", ()):
        raise unittest.SkipTest(
            "等待实现：as_equivalence='force' 尚未落地（负标记不会被忽略）")

    observed: list[bool] = []
    with _isolated() as work:
        for index, equation in enumerate((EQUIV_EQUATION, NO_EQUIV_EQUATION)):
            operands = _operands(equation, seed=20261117 + index)
            expected = np.einsum(equation, *operands)
            directory = os.path.join(work, f"force-{index}")
            _profile_cache.configure(directory)
            key = _profile_key(equation, operands)
            assert _profile_cache.record_equivalence(key, "none"), (
                "前置条件：标记 none 必须写成功", equation)
            assert _marker_state(key) == "none", (_profile_cache.equivalence_marker(key))

            rewritable = _rewrite_possible(equation)
            observed.append(rewritable)

            with _normalize_spy() as calls, _quiet():
                actual = pe.contract(equation, *operands, profile=True,
                                     profile_dir=directory,
                                     as_equivalence="force")

            assert len(calls) >= 1, (
                "force 必须无视负标记、真的再查一次", equation, len(calls))
            _assert_matches_numpy(actual, expected, f"force {equation}")

            marker = _profile_cache.equivalence_marker(key)
            assert isinstance(marker, dict), (
                "force 之后必须有标记行（查过就该有结论）", equation, marker)
            if rewritable:
                assert marker.get("state") == "ok", (
                    "能改写时标记必须刷新成 ok", equation, marker)
                assert str(marker.get("kind") or "").strip(), (
                    "ok 的标记必须带 kind（哪一类改写）", equation, marker)
            else:
                assert marker.get("state") == "none", (
                    "真的不能改写时不得谎报 ok", equation, marker)

    # 两条分支都必须真的被执行到，否则"两种情况都要断言"只是写在注释里。
    assert observed == [True, False], (
        "两个样本必须分别覆盖「可改写」与「不可改写」两支", observed)


# ── 4. profile=False → 零 IO ────────────────────────────────────────────────
def test_profile_false_writes_no_marker_and_creates_no_directory():
    """``profile=False`` 不读不写标记：不建目录、不建 CSV、查不到标记。"""
    _require_markers()
    with _isolated() as work:
        operands = _operands(EQUIV_EQUATION)
        expected = np.einsum(EQUIV_EQUATION, *operands)
        key = _profile_key(EQUIV_EQUATION, operands)
        off_dir = os.path.join(work, "off")

        with _quiet():
            actual = pe.contract(EQUIV_EQUATION, *operands,
                                 profile=False, profile_dir=off_dir)
        _assert_matches_numpy(actual, expected, "profile=False")

        # 零 IO：连目录都不该建（显式给了 profile_dir 也一样，因为它被忽略）。
        assert not os.path.exists(off_dir), (
            "profile=False 不得创建 profile_dir", os.listdir(work))
        assert not os.path.exists(os.path.join(off_dir, MARKER_NAME)), MARKER_NAME
        assert not os.path.exists(os.path.join(work, _profile_cache.DEFAULT_DIR)), (
            "profile=False 连缺省的 ./.cache 都不该建", os.listdir(work))
        # 此刻索引还是空的：读不到标记（这一步会按缺省目录探测，所以放在上面之后）。
        assert _marker_state(key) is None, (
            "profile=False 之后不该出现标记", _profile_cache.equivalence_marker(key))

        # **正对照**：同一次调用在 profile=True 下必须留下标记，否则上面的"没有"
        # 只是"这个功能根本不存在"。
        on_dir = os.path.join(work, "on")
        with _quiet():
            pe.contract(EQUIV_EQUATION, *operands,
                        profile=True, profile_dir=on_dir)
        marker = _profile_cache.equivalence_marker(key)
        if marker is None:
            raise unittest.SkipTest(
                "等待实现：_api 尚未把等价查询结论写成标记，"
                "profile=False 的零 IO 断言暂不可证")
        assert str(marker.get("state")).strip().lower() in ("ok", "none"), marker


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
    assert len(combos) == bundle == len(set(combos)) <= limit, (
        "缺省网格必须是三轴笛卡尔积、无重复、且不超过上限",
        len(combos), bundle, limit)

    # 构造一个**必然**超过上限的网格：给 optimize 轴加两个取值。
    widened = tuple(cli.TUNE_OPTIMIZERS) + ("dp", "branch-all")
    expected = (len(cli.TUNE_KERNEL_FORCE) * len(cli.TUNE_AS_EQUIVALENCE)
                * len(widened))
    assert expected > limit, ("本用例必须构造出超限网格", expected, limit)

    with patch.object(cli, "TUNE_OPTIMIZERS", widened):
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
        raise unittest.SkipTest("--tune 的网格为空，最优选择不可验")
    # 从实现自己的网格里挑两组：只差 ``as_equivalence``、且内核不是"本机可能不可用"
    # 的强制档（否则注入的数值不符会被更早的 KernelUnavailableError 顶掉）。
    available = [index for index, combo in enumerate(combos)
                 if combo[0] not in ("dedicated", "topology")]
    bad = next((index for index in available if combos[index][1] == "false"), None)
    winner = next((index for index in available
                   if index != bad and combos[index][1] == "true"), None)
    if bad is None or winner is None:
        raise unittest.SkipTest(
            "网格里没有 as_equivalence=true/false 两档，无法构造数值不符的对照组："
            f"{sorted({combo[1] for combo in combos})}")
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
                            atol=cli.DEFAULT_ATOL, rtol=cli.DEFAULT_RTOL)
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
    assert combos[bad][1] != combos[winner][1], (
        "对照组与最优组必须在打印字段上可区分", combos[bad], combos[winner])
    assert f"as_equivalence={combos[bad][1]}" not in best_line, (
        "MISMATCH 组不得出现在最优行", best_line)
    assert not [line for line in printed.splitlines()
                if line.startswith("tune row")], (
        "非 verbose 不该打明细行", printed[-1200:])
    mismatched = [row for row in rows if row.get("mismatch")]
    assert f"MISMATCH={len(mismatched)}" in printed, (
        "被排除的组合数必须如实给出", len(mismatched), printed[-1200:])


# ── 8. 坏组合不留画像行（超时 / 异常 / MISMATCH，三类都是注入）──────────────
def test_tune_keeps_no_row_for_a_timed_out_combination():
    """注入超时：被放弃的那一组**不留画像行**，其余组合照原样记录。

    注入点是 ``_timed``：先 ``call()`` 一次（``contract()`` 因此真的把这一组记进画像），
    再抛 ``TimeoutError``——这正是真实超时的形态（算了一半/算完但没在预算内）。
    若实现只在"跑不成就不记"这一层兜着，本用例会失败：那一行已经记下了。
    """
    _require_tune()
    _require_bad_combo_drop()
    bad = len(SMALL_TUNE_COMBOS) - 1
    abandoned = SMALL_TUNE_COMBOS[bad][2]
    with _isolated(profile=True) as work:
        _, full_summary, full_configs = _run_tune_small(
            work, "full", SMALL_TUNE_COMBOS, timed=_scripted_timed({}))
        _, control_summary, control_configs = _run_tune_small(
            work, "control", SMALL_TUNE_COMBOS[:bad], timed=_scripted_timed({}))
        tune, summary, configs = _run_tune_small(
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
        _assert_only_the_abandoned_combo_disappeared(
            full_summary, full_configs, control_summary, control_configs,
            summary, configs, abandoned)


def test_tune_keeps_no_row_for_a_raising_combination():
    """注入 ``RuntimeError``：同上——异常组合也不许在画像里留下半条记录。"""
    _require_tune()
    _require_bad_combo_drop()
    bad = len(SMALL_TUNE_COMBOS) - 1
    abandoned = SMALL_TUNE_COMBOS[bad][2]
    with _isolated(profile=True) as work:
        _, full_summary, full_configs = _run_tune_small(
            work, "full", SMALL_TUNE_COMBOS, timed=_scripted_timed({}))
        _, control_summary, control_configs = _run_tune_small(
            work, "control", SMALL_TUNE_COMBOS[:bad], timed=_scripted_timed({}))
        tune, summary, configs = _run_tune_small(
            work, "raise", SMALL_TUNE_COMBOS,
            timed=_scripted_timed({}, {bad: RuntimeError("injected: combo blew up")}))

        assert len(tune["rows"]) == len(SMALL_TUNE_COMBOS), (
            "单组异常不该中断遍历", len(tune["rows"]))
        row = tune["rows"][bad]
        assert row["error"] == "RuntimeError", row
        assert row["median_ms"] is None, row
        _assert_only_the_abandoned_combo_disappeared(
            full_summary, full_configs, control_summary, control_configs,
            summary, configs, abandoned)


def test_tune_keeps_no_row_for_a_mismatching_combination():
    """``rel`` 超 ``atol``：不留画像行，且**不得**成为 ``tune best``。

    注入假耗时让 MISMATCH 那一组成为最快的一组——若"最快即最优"，当选的正是它；
    这条因此有区分力（否则"不选它"只是因为别人更快）。
    """
    cli = _require_tune()
    _require_bad_combo_drop()
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
        _, full_summary, full_configs = _run_tune_small(
            work, "full", SMALL_TUNE_COMBOS, timed=_scripted_timed({}))
        _, control_summary, control_configs = _run_tune_small(
            work, "control", SMALL_TUNE_COMBOS[:bad], timed=_scripted_timed({}))
        tune, summary, configs = _run_tune_small(
            work, "mismatch", SMALL_TUNE_COMBOS,
            timed=_scripted_timed(seconds), contract=falsify)

        rows = tune["rows"]
        row = rows[bad]
        assert row.get("mismatch") is True, ("rel 超容差必须判 MISMATCH", row)
        assert row["rel"] is not None and row["rel"] > (
            cli.DEFAULT_ATOL + cli.DEFAULT_RTOL), row
        assert [item for item in rows if item.get("mismatch")] == [row], (
            "MISMATCH 只该挑出注入的那一组", rows)
        measured = [item for item in rows if item.get("median_ms") is not None]
        assert min(measured, key=lambda item: item["median_ms"]) is row, (
            "注入的假耗时必须让 MISMATCH 组成为最快的那一组")
        assert tune["best"] is not None and tune["best"] is not row, (
            "MISMATCH 组不得被选为 best", tune["best"])
        assert tune["best"]["optimize"] == SMALL_TUNE_COMBOS[len(seconds) - 2][2], (
            "最优必须落到可用组里最快的那一组", tune["best"])
        _assert_only_the_abandoned_combo_disappeared(
            full_summary, full_configs, control_summary, control_configs,
            summary, configs, abandoned)


# ── 9. tune 期间强制写：整张地图 vs 稀疏口径 ────────────────────────────────
def _sweep_one_key(work: str, tag: str) -> tuple[dict[str, Any], int, list[dict[str, str]]]:
    """在**同一个画像键**上跑三组不同 ``optimize``，返回 ``(summary, flush 行数, 盘上行)``。

    直接调 ``contract()`` 而不是 ``run_tune``：这里验的是**开关本身**的口径
    （``set_profile_tune`` / 它实际读的那个属性），与 tune 怎么安排遍历无关——
    ``run_tune`` 自己就会把开关打开，那样 "False 分支" 永远量不到。
    """
    directory = os.path.join(work, tag)
    os.makedirs(directory, exist_ok=True)
    cache_dir = os.path.join(directory, "cache")
    operands = _operands(TUNE_EQUATION)
    with _frozen_compute_ms(), _quiet():
        for _kernel, _as_equiv, optimize in SMALL_TUNE_COMBOS:
            pe.contract(TUNE_EQUATION, *operands, backend="numpy", optimize=optimize,
                        kernel_force="generic", as_equivalence=False,
                        profile=True, profile_dir=cache_dir)
        written = _profile_cache.flush()
    summary = _profile_cache.summary()
    return summary, written, _profile_disk_rows(summary.get("file"))


def test_tune_flag_writes_the_whole_map_and_sparse_keeps_a_single_row():
    """tune 期间的强制写开关：打开时多组配置一起落盘，关掉回到稀疏口径。"""
    switch = _tune_switch()
    with _isolated(profile=True) as work:
        try:
            switch(True)
            on_summary, on_written, on_rows = _sweep_one_key(work, "forced")
        finally:
            switch(False)                        # 无论断言怎么退出，都别把开关留下
        try:
            off_summary, off_written, off_rows = _sweep_one_key(work, "sparse")
        finally:
            switch(False)

        assert on_summary["enabled"] is True, on_summary
        assert on_summary["rows"] > 1, (
            "tune 期间必须留下整张地图（同键多组）", on_summary["rows"])
        # 每键还有"只留最快 K 条"的上限：期望条数是两者的较小值，但必须 > 1。
        per_key = int(on_summary.get("max_rows_per_key")
                      or len(SMALL_TUNE_COMBOS))
        expected_on_disk = min(len(SMALL_TUNE_COMBOS), per_key)
        assert expected_on_disk > 1, (per_key, expected_on_disk)
        assert on_summary["rows"] == expected_on_disk, (
            on_summary["rows"], expected_on_disk)
        assert on_written == expected_on_disk == len(on_rows), (
            "强制写必须让每一组配置都落到磁盘", on_written, len(on_rows),
            expected_on_disk)
        assert {row["optimize"] for row in on_rows} <= {
            optimize for _kernel, _as_equiv, optimize in SMALL_TUNE_COMBOS}, on_rows
        assert len(on_rows) > 1, on_rows

        # ``False``：同键多组只留**一行**（稀疏口径是"新键或快过 5% 才写"，
        # 而本用例把 compute_ms 钉成 0，于是只有新键那行够格）。
        assert off_summary["enabled"] is True, off_summary
        assert off_written == len(off_rows) == 1, (
            "稀疏口径：同键多组只该落盘一行", off_written, off_rows)
        assert len(off_rows) < len(on_rows), (len(off_rows), len(on_rows))


TESTS = (
    test_negative_marker_skips_the_equivalence_step,
    test_without_a_marker_the_equivalence_step_still_runs,
    test_force_ignores_the_negative_marker_and_refreshes_it,
    test_profile_false_writes_no_marker_and_creates_no_directory,
    test_tune_warns_and_ignores_the_axes_it_sweeps_itself,
    test_tune_refuses_more_than_the_combination_cap,
    test_tune_excludes_a_mismatching_combination_from_best,
    test_tune_keeps_no_row_for_a_timed_out_combination,
    test_tune_keeps_no_row_for_a_raising_combination,
    test_tune_keeps_no_row_for_a_mismatching_combination,
    test_tune_flag_writes_the_whole_map_and_sparse_keeps_a_single_row,
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
