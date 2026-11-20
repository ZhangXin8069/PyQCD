"""端到端验收契约：用户 2026-11-19 / 11-20 提的**九项**需求，逐条钉死。

为什么要有这一份"总契约"
------------------------
这九项需求有一个共同点：**它们都不会让数值测试变红**。探测失败仍返回 ``False``、
``original`` 与 ``opt_einsum`` 逐位一致、配置表少一列也照样命中、峰值内存记成 ``-``
只是报告少一格——正常路径上一切看起来都对，只有"用户当时抱怨的那件事"会悄悄回来。
所以这里按需求逐条写成一条断言函数，失败信息直接点名"哪一条需求、现在是什么值"。

九条与对应需求
--------------
1. ``test_gpu_diagnosis_reports_real_probe_stage``
   需求 #1：GPU 诊断不许再写死"2x2 matmul disagreed with numpy"，两个内核模块都要导出
   ``probe_dtypes()``，失败原因要点名真实 ``stage=``。
2. ``test_original_kernel_runs_all_seven_optimizers``
   需求 #2：``kernel_force="original"``（交给 ``opt_einsum.contract``）必须可用，
   七种 ``optimize`` 全跑得通，且四组转型（``branch_1``/``branch_all``/``none``/``None``）
   一一对上。
3. ``test_default_profile_dir_lives_in_home_cache``
   需求 #3：缺省配置目录必须是**家目录绝对路径**（``$XDG_CACHE_HOME`` 或 ``~/.cache``
   下的 ``pyqcd.einsum``），不是相对 CWD 的 ``./.cache``。
4. ``test_peak_memory_is_measured_and_rendered_per_group``
   需求 #4：``--tune`` 每组都要有峰值内存，四个公开函数可用、150 MB 分配测得出来、
   ``format_peak_mb`` 的边界与 ``tune_comb_lines`` 的 ``peak=`` 列都在。
5. ``test_prune_cutoff_is_twice_the_best_wall``
   需求 #5：剪枝截止是 **2 × min_wall_ms**，``原因=`` 里同时给出两个数。
6. ``test_record_table_replaces_and_stays_out_of_dispatch``
   需求 #6：记录表文件名/列集合/整表重写语义，且 ``_api`` 里只有"更省内存"提示读它。
7. ``test_cli_defaults_are_zero_warmup_and_one_round``
   需求 #7：``DEFAULT_WARMUP == 0`` / ``DEFAULT_ROUNDS == 1``（argparse 与函数缺省同步），
   且"0 合法、负数非法"这条边界真的成立（显式 ``--warmup 0`` 跑得通、``-1`` 退出码 2、
   加速设备上 0 不被抬）。
8. ``test_cheaper_memory_hint_requires_half_the_peak``
   需求 #8：命中时要提示"存在更省内存的配置"（含两张表的文件名），候选只省 30% 时**不**提示。
9. ``test_autotune_runs_once_per_key_on_first_call``
   需求 #9：首次调用自动 tune，开关口径（``"0"`` 关、``"1"`` 开）、``reset()`` 清账、
   同键每进程只试一次。

隔离口径（硬要求）
------------------
* **绝不写 ``~/.cache/pyqcd.einsum``**：凡是碰表路径的用例，一律把
  ``PYQCD_EINSUM_PROFILE_DIR`` 与 ``_profile_cache._CONFIGURED_DIR`` 钉到 ``data/tmp/``
  下的临时目录（见 :func:`_isolated`），退出时把环境变量与进程级状态逐字还原（照抄
  ``_einsum_profile_robustness_contract.py`` 的 snapshot/restore）。
* 第 3 条刻意用**子进程**验缺省目录：``DEFAULT_DIR`` 是 import 期求值的，在本进程里改
  ``XDG_CACHE_HOME`` 只会得到旧值；子进程还顺便证明"import 不建目录"。
* 每条用例独立可跑；需要 GPU 才有意义的分支用 ``unittest.SkipTest``（本契约本身不要求
  GPU——第 1 条用非法探测方程把失败逼出来，任何机器都能跑）。
* 收尾：删临时目录、``_autotune.reset()``、还原 ``_ATTEMPTED``、还内存池。

可直接运行::

    python -B -m pyqcd.testing._einsum_nine_items_acceptance_contract
"""
from __future__ import annotations

import ast
import contextlib
import csv
import gc
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

__all__ = ["test_einsum_nine_items_acceptance_contracts"]

#: 用户实测里那句**写死**的、把六种不同失败统一说成数值不一致的文案。
FIXED_PROBE_PHRASE = "2x2 matmul disagreed with numpy"

#: 第 4 条那块数组的大小：150 MB（``uint8`` 1 字节/元素，不必乘 itemsize）。
ALLOC_MB = 150

#: 第 4 条的判定门槛：留 50 MB 余量给"分配器已持有这么多页、RSS 不涨"的情形。
ALLOC_MIN_PEAK_MB = 100.0

#: 第 2 条那个"pow2 小方程"：三个 4×4 操作数（全是 2 的幂），收缩序的三种规划器
#: （greedy / dp / optimal）都真的会分叉，`branch_1/2/all` 也各有一步可合并。
ORIGINAL_EQUATION = "ab,bc,cd->ad"

#: 第 2 条的 rel 门槛（complex128、4×4、三步收缩，实测 ~2e-16）。
ORIGINAL_MAX_REL = 1e-14

#: 七种 ``optimize`` 叶子（``_profile_cache.OPTIMIZE_LEAVES`` 的同一份清单）。
SEVEN_OPTIMIZERS = ("greedy", "branch_1", "branch_2", "dp", "branch_all",
                    "optimal", "none")

#: 临时区：``data/tmp/``（相对仓库根），与仓库既有约定一致。
_TMP_ENV = "PYQCD_TEST_TMP"

#: ``"模块上没有这个内部名"`` 的哨兵：隔离层不该因为实现少一个内部名就把整条契约拖崩。
_MISSING = object()

#: 需要按用例隔离的 ``_profile_cache`` 进程级状态（前五项照抄 robustness 契约，
#: 后四项是本契约会碰到的：旧位置说明 + 警告去重）。
_PROFILE_STATE = ("_CONFIGURED_DIR", "_LAST_ERROR", "_HOST", "_TABLE", "_TUNING",
                  "_SOURCE_NOTE", "_SOURCE_PATH", "_WARNED_CELLS",
                  "_PENDING_WARNINGS")

#: ``_api`` 的进程级状态：最近一次内核/画像记录 + 命中提示去重。
_API_STATE = ("_LAST_KERNEL", "_LAST_PROFILE", "_PROFILE_NOTICED",
              "_PROFILE_MISSING_NOTICED")

#: 本契约会临时改写的环境变量（退出时逐字还原）。
_ENV_NAMES = ("PYQCD_EINSUM_PROFILE", "PYQCD_EINSUM_PROFILE_DIR",
              "PYQCD_EINSUM_AUTOTUNE", "PYQCD_EINSUM_KERNEL_FORCE")


# ── 懒取被测模块（import 期不碰 numpy/CLI 之外的东西）──────────────────────
def _profile_cache():
    from ..einsum import _profile_cache as module
    return module


def _api_module():
    from ..einsum import _api as module
    return module


def _specialize():
    from ..einsum import _specialize as module
    return module


def _autotune_module():
    from ..einsum import _autotune as module
    return module


def _kernel_force():
    from ..einsum import _kernel_force as module
    return module


def _kernel_cuda():
    from ..einsum import _kernel_cuda as module
    return module


def _kernel_kupl():
    from ..einsum import _kernel_kupl as module
    return module


def _tune_axes():
    from ..einsum import _tune_axes as module
    return module


def _memory_module():
    from ..einsum import _memory as module
    return module


def _cli():
    from ..einsum import test as module
    return module


# ── 临时区 ──────────────────────────────────────────────────────────────────
def _repo_root() -> str:
    """仓库根目录（本文件在 ``<root>/pyqcd/testing/`` 下）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def _tmp_root() -> str:
    """临时根：``data/tmp/``；不可写时退回系统临时目录（报告里能看出来）。"""
    override = os.environ.get(_TMP_ENV)
    candidates = [override] if override else []
    candidates.append(os.path.join(_repo_root(), "data", "tmp"))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = tempfile.mkdtemp(prefix="nine-items-probe-", dir=candidate)
            os.rmdir(probe)
            return candidate
        except Exception:                          # noqa: BLE001 - 换下一个
            continue
    return tempfile.gettempdir()


# ── 状态隔离（照抄 _einsum_profile_robustness_contract 的 _snapshot/_restore）──
def _copy_state(value: Any) -> Any:
    """容器型状态按内容拷贝（``_TABLE`` 的解析缓存要连行一起拷）。"""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, set):
        return set(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple) and len(value) == 4 and isinstance(value[3], list):
        return (value[0], value[1], value[2], [dict(row) for row in value[3]])
    return value


def _snapshot(module: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """把 ``module`` 上 :data:`names` 这些进程级状态抄一份（缺的记 ``_MISSING``）。"""
    saved: dict[str, Any] = {}
    for name in names:
        value = getattr(module, name, _MISSING)
        saved[name] = _MISSING if value is _MISSING else _copy_state(value)
    return saved


def _restore(module: Any, saved: dict[str, Any]) -> None:
    """把 :func:`_snapshot` 抄下来的状态放回去（``_MISSING`` = 原来没有，跳过）。"""
    for name, value in saved.items():
        if value is _MISSING:
            continue
        setattr(module, name, value)


def _reset_profile_state(module: Any) -> None:
    """把配置表模块清到"没配置、没解析缓存、不处于 tune 窗口"的初始态。"""
    for name, value in (("_TABLE", None), ("_TUNING", 0), ("_CONFIGURED_DIR", ""),
                        ("_LAST_ERROR", None), ("_SOURCE_NOTE", None),
                        ("_SOURCE_PATH", None)):
        if hasattr(module, name):
            setattr(module, name, value)
    for name in ("_WARNED_CELLS", "_PENDING_WARNINGS"):
        container = getattr(module, name, None)
        if isinstance(container, (set, list)):
            container.clear()


def _reset_api_state(module: Any) -> None:
    """把最近一次内核/画像记录与命中提示去重清掉（否则上一条用例的提示会被静音）。"""
    for name, value in (("_LAST_KERNEL", None), ("_LAST_PROFILE", None),
                        ("_PROFILE_MISSING_NOTICED", False)):
        if hasattr(module, name):
            setattr(module, name, value)
    noticed = getattr(module, "_PROFILE_NOTICED", None)
    if isinstance(noticed, set):
        noticed.clear()


@contextlib.contextmanager
def _isolated(prefix: str = "nine-items-") -> Iterator[str]:
    """临时目录 + 干净的环境变量与进程级状态；退出时逐字还原。

    进入时：

    * 在 ``data/tmp/`` 下建一个临时目录，把 ``PYQCD_EINSUM_PROFILE_DIR`` 指向它；
    * 总开关 ``PYQCD_EINSUM_PROFILE`` **打开**（本契约要验表的行为，套件里的 ``=0``
      必须在用例内被顶掉），自动 tune 关掉（``PYQCD_EINSUM_AUTOTUNE=0``——第 9 条自己
      用 monkeypatch 验，其余用例不该为一次 ``contract()`` 跑一整轮 tune）；
    * ``_profile_cache`` / ``_api`` 的进程级状态清到初始态。

    退出时：删目录 → 还环境变量 → 还进程级状态 → ``_autotune.reset()`` 并还原
    ``_ATTEMPTED`` / ``_RUNNING``。**任何一条用例都不会碰到 ``~/.cache/pyqcd.einsum``**
    —— 目录要么是本函数的临时树，要么（第 3 条）是子进程里临时 ``HOME`` 下的一条路径。
    """
    pc = _profile_cache()
    api = _api_module()
    autotune = _autotune_module()
    specialize = _specialize()

    saved_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    saved_pc = _snapshot(pc, _PROFILE_STATE)
    saved_api = _snapshot(api, _API_STATE)
    saved_attempted = set(getattr(autotune, "_ATTEMPTED", set()))
    # ``_RUNNING`` 在实现里换过形状（全局布尔 → 按键记录的集合），两种都照抄：
    # 契约只负责"进出时一模一样"，不该假定实现的内部表示。
    saved_running = _copy_state(getattr(autotune, "_RUNNING", _MISSING))
    saved_local = getattr(getattr(autotune, "_LOCAL", None), "active", _MISSING)
    saved_language = specialize.set_language(None)      # 进入时回到"按 env/locale"

    tmp = tempfile.mkdtemp(prefix=prefix, dir=_tmp_root())
    try:
        os.environ[pc.DIR_ENV] = tmp
        os.environ.pop(pc.PROFILE_ENV, None)
        os.environ[autotune.AUTOTUNE_ENV] = "0"
        os.environ.pop("PYQCD_EINSUM_KERNEL_FORCE", None)
        _reset_profile_state(pc)
        _reset_api_state(api)
        autotune.reset()
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _restore(pc, saved_pc)
        _restore(api, saved_api)
        autotune.reset()
        attempted = getattr(autotune, "_ATTEMPTED", None)
        if isinstance(attempted, set):
            attempted.clear()
            attempted.update(saved_attempted)
        if hasattr(autotune, "_RUNNING"):
            autotune._RUNNING = saved_running
        specialize.set_language(saved_language)


# ── 小工具 ──────────────────────────────────────────────────────────────────
def _key(**overrides: Any) -> dict[str, Any]:
    """配置表/记录表键的稀疏写法（``shape`` 用列表写法，与实现同口径归一化）。"""
    key = {
        "backend": "numpy",
        "dtype": "complex128",
        "equation": "ij,jk->ik",
        "shape": [(4, 4), (4, 4)],
        "device": "cpu",
    }
    key.update(overrides)
    return key


def _record(**overrides: Any) -> dict[str, Any]:
    """记录表一行的稀疏写法（只给关心的列，其余由 ``write_records`` 补空）。"""
    row: dict[str, Any] = dict(_key())
    row.update({
        "kernel": "generic",
        "optimize": "greedy",
        "wall_ms": 1.25,
        "peak_memory_mb": 12.5,
        "note": "recorded-by-nine-items-contract",
    })
    row.update(overrides)
    return row


def _enclosing_functions(tree: ast.AST, lineno: int) -> list[str]:
    """``lineno`` 那一行落在哪些函数里（最内层在前）。"""
    owners: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or node.lineno
            if node.lineno <= lineno <= end:
                owners.append((end - node.lineno, node.name))
    owners.sort()
    return [name for _, name in owners]


def _kernel_snapshot(module: Any) -> tuple[dict, dict, dict]:
    """逐 dtype 的 bool 缓存 / 探测事实 / 最近一次事实（第 1 条会清空重探）。"""
    return (dict(getattr(module, "_TOPOLOGY_DTYPE_SUPPORT", {})),
            dict(getattr(module, "_TOPOLOGY_DTYPE_PROBE", {})),
            dict(getattr(module, "LAST_PROBE", {})))


def _kernel_restore(module: Any, saved: tuple[dict, dict, dict]) -> None:
    support, probes, last = saved
    for name, values in (("_TOPOLOGY_DTYPE_SUPPORT", support),
                         ("_TOPOLOGY_DTYPE_PROBE", probes),
                         ("LAST_PROBE", last)):
        container = getattr(module, name, None)
        if isinstance(container, dict):
            container.clear()
            container.update(values)


def _source_of(module: Any) -> str:
    """模块源码原文（用文件而不是 ``inspect``：被替换过的对象也能读到真实文件）。"""
    with open(module.__file__, "r", encoding="utf-8") as handle:
        return handle.read()


def _subprocess_env(home: str, xdg: str | None) -> dict[str, str]:
    """子进程环境：临时 ``HOME``（+ 可选 ``XDG_CACHE_HOME``），去掉本仓库的表开关。"""
    drop = set(_ENV_NAMES) | {"HOME", "XDG_CACHE_HOME"}
    env = {name: value for name, value in os.environ.items() if name not in drop}
    env["HOME"] = home
    if xdg is not None:
        env["XDG_CACHE_HOME"] = xdg
    # ``cwd`` 可能被换到临时树里，所以显式把仓库根放进 ``PYTHONPATH``。
    repo_root = _repo_root()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root + (os.pathsep + existing if existing else "")
    return env


def _default_dir_from_subprocess(env: dict[str, str], cwd: str) -> str:
    """在子进程里 import ``_profile_cache`` 并打印 ``DEFAULT_DIR``（不建目录）。"""
    code = ("from pyqcd.einsum import _profile_cache as p; print(p.DEFAULT_DIR)")
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=env, cwd=cwd, capture_output=True, text=True, timeout=300,
        check=False)
    assert completed.returncode == 0, (
        "子进程 import _profile_cache 失败：",
        completed.returncode, completed.stdout, completed.stderr)
    return completed.stdout.strip()


# ── 1. GPU 诊断不再写死原因 ─────────────────────────────────────────────────
def test_gpu_diagnosis_reports_real_probe_stage() -> None:
    """需求 #1：GPU 诊断不许再写死"2x2 matmul disagreed with numpy"。

    钉住三件事：两个内核模块都**导出** ``probe_dtypes()``（无参、定义在自己模块里）；
    两边的源码里都 grep 不到那句写死文案；``_kernel_cuda._topology_dtype_reason`` 在
    探测失败时给出的原因里带 ``stage=``（真实阶段），而不是一句"数值不一致"。

    为什么它会回归：探测器的返回值契约是 ``bool``，把 ``except Exception: return False``
    写成一行最省事——那句写死文案当初就是这么来的。文案不进数值、不进退出码，普通测试
    永远看不见；只有拿到缺符号/版本不符的机器上，用户才会看到**指错方向**的诊断。

    本机有 GPU 时走真实探测；没有 GPU 时探测在 ``import`` 阶段就失败，``stage=`` 仍在
    ——所以**任何机器都能跑**，不需要 SkipTest。
    """
    cuda = _kernel_cuda()
    kupl = _kernel_kupl()

    for module, name in ((cuda, "_kernel_cuda"), (kupl, "_kernel_kupl")):
        assert hasattr(module, "probe_dtypes"), (
            f"{name} 必须导出 probe_dtypes()：--gpu-probe 要逐 dtype 的真实结论")
        assert callable(module.probe_dtypes), (name, module.probe_dtypes)
        assert module.probe_dtypes.__module__ == f"pyqcd.einsum.{name}", (
            f"{name}.probe_dtypes 必须定义在本模块里，不能是别处借来的",
            module.probe_dtypes.__module__)
        assert not inspect.signature(module.probe_dtypes).parameters, (
            f"{name}.probe_dtypes() 必须无参（四个 dtype 由模块自己决定）",
            inspect.signature(module.probe_dtypes))
        source = _source_of(module)
        assert FIXED_PROBE_PHRASE not in source, (
            f"{name} 的源码里仍有那句写死文案 {FIXED_PROBE_PHRASE!r}："
            "缺符号/版本不符都会被告知成'数值不一致'")
        assert "probe_dtypes" in source, (name, "probe_dtypes 必须出现在源码里")

    dtype = np.dtype(np.complex64)
    saved = _kernel_snapshot(cuda)
    try:
        for name in ("_TOPOLOGY_DTYPE_SUPPORT", "_TOPOLOGY_DTYPE_PROBE", "LAST_PROBE"):
            container = getattr(cuda, name, None)
            if isinstance(container, dict):
                container.clear()
        # 用**非法输出标签**把探测钉在规划阶段失败：不依赖原生库缺不缺符号、不依赖
        # 有没有 GPU，也不依赖真实数值超差。
        with contextlib.suppress(Exception):
            from unittest.mock import patch as _patch
            with _patch.object(cuda, "_PROBE_EQUATION", "ij,jk->iz"):
                probed_ok = cuda._topology_dtype_enabled(dtype)
        reason = cuda._topology_dtype_reason(dtype)
        record = dict(getattr(cuda, "LAST_PROBE", {}))
    finally:
        _kernel_restore(cuda, saved)

    assert probed_ok is False, (
        f"非法探测方程 {cuda._PROBE_EQUATION!r} 之下 complex64 的探测必须失败，实际 {probed_ok}")
    assert reason is not None, "探测没过必须给出拒绝原因（None = 被当成可用）"
    assert "stage=" in reason, (
        "拒绝原因里必须点名真实阶段（stage=import/build/plan/native/…），"
        f"否则又是一句无信息文案：{reason!r}")
    assert FIXED_PROBE_PHRASE not in reason, (
        f"拒绝原因里仍有写死文案：{reason!r}")
    if record:
        assert record.get("stage") in getattr(cuda, "PROBE_STAGES", ()), (
            "LAST_PROBE 的 stage 必须是契约里的阶段名", record)
        assert record.get("stage") != "ok", ("失败的探测不许记成 ok 阶段", record)


# ── 2. original 内核 + 七种 optimize ─────────────────────────────────────────
def test_original_kernel_runs_all_seven_optimizers() -> None:
    """需求 #2：``kernel_force="original"``（交给 ``opt_einsum.contract``）端到端可用。

    ``ALLOWED_LABELS`` 含 ``original``、``parse("original") == ("original",)``；
    在 pow2 小方程上与 ``np.einsum`` 的 rel ≤ 1e-14（complex128），
    ``last_kernel()["ran"] == "original"``；七种 ``optimize`` 全部跑得通；四组转型
    （``branch_1→branch-1``、``branch_all→branch-all``、``none→False``、``None→greedy``）
    逐条对上。

    为什么它会回归：``original`` 是"把收缩序与执行都交给外部实现"的那条路，它不进专用
    内核分派，所以**任何**针对专用内核的改动都测不到它；而 ``optimize`` 的两侧拼写不同
    （本库下划线、``opt_einsum`` 连字符/``False``），写错就是 ``KeyError``——只有真的
    跑一遍七种叶子才会暴露。转型表一旦被"顺手统一"成同一套拼写，这个契约立刻红。
    """
    try:
        import opt_einsum  # noqa: F401,PLC0415 - 没有它就没有 original 这条路
    except Exception as exc:                       # noqa: BLE001 - 没装就跳过
        raise unittest.SkipTest(
            f"opt_einsum 不可用（{type(exc).__name__}: {exc}）：original 无从验证")

    kernel_force = _kernel_force()
    api = _api_module()

    assert "original" in kernel_force.ALLOWED_LABELS, (
        "kernel_force 必须接受 original", sorted(kernel_force.ALLOWED_LABELS))
    assert kernel_force.parse("original") == ("original",), (
        kernel_force.parse("original"))
    assert kernel_force.parse("ORIGINAL") == ("original",), "解析必须大小写不敏感"
    assert kernel_force.forced_labels("original") == ("original",), (
        kernel_force.forced_labels("original"))

    mappings = {"branch_1": "branch-1", "branch_all": "branch-all",
                "none": False, None: "greedy"}
    measured_mappings = {}
    for leaf, expected in mappings.items():
        got = kernel_force.opt_einsum_path_name(leaf)
        measured_mappings[leaf] = got
        assert got == expected, (
            f"opt_einsum_path_name({leaf!r}) 必须是 {expected!r}，实际 {got!r}")
        assert type(got) is type(expected), (
            f"{leaf!r} 的转型结果类型不对：{type(got).__name__} != {type(expected).__name__}")
    # 另外两个也顺手钉一下（拼写相同，但同样只能交给 opt_einsum）。
    assert kernel_force.opt_einsum_path_name("greedy") == "greedy"
    assert kernel_force.opt_einsum_path_name("optimal") == "optimal"

    rng = np.random.default_rng(20261120)
    operands = [(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
                 ).astype(np.complex128) for _ in range(3)]
    reference = np.einsum(ORIGINAL_EQUATION, *operands)
    assert float(np.linalg.norm(reference)) > 0.0

    measured_rel: dict[str, float] = {}
    with _isolated():
        for optimize in SEVEN_OPTIMIZERS:
            result = api.contract(ORIGINAL_EQUATION, *operands,
                                  kernel_force="original", optimize=optimize,
                                  profile=False)
            assert isinstance(result, np.ndarray), (optimize, type(result))
            assert result.dtype == np.dtype(np.complex128), (optimize, result.dtype)
            assert result.shape == reference.shape, (optimize, result.shape)
            rel = float(np.linalg.norm(result - reference)
                        / np.linalg.norm(reference))
            measured_rel[optimize] = rel
            assert rel <= ORIGINAL_MAX_REL, (
                f"original + optimize={optimize!r} 与 np.einsum 的 rel={rel:.3e} "
                f"超过 {ORIGINAL_MAX_REL:.0e}")
            record = api.last_kernel()
            assert isinstance(record, dict), (optimize, record)
            assert record.get("ran") == "original", (
                f"optimize={optimize!r} 实际跑的不是 original：{record}")
            assert record.get("requested") == ("original",), (optimize, record)


# ── 3. 缺省目录在家目录 ─────────────────────────────────────────────────────
def test_default_profile_dir_lives_in_home_cache() -> None:
    """需求 #3：缺省配置目录必须是**家目录下的绝对路径**，与 CWD 无关。

    断言 ``DEFAULT_DIR`` 是绝对路径、以 ``pyqcd.einsum`` 结尾，并且（在**子进程**里
    分别设/不设 ``XDG_CACHE_HOME``）等于 ``$XDG_CACHE_HOME/pyqcd.einsum`` 或
    ``~/.cache/pyqcd.einsum``。子进程还证明 import **不建目录**。

    为什么它会回归：``DEFAULT_DIR`` 曾是一个**相对当前工作目录**的目录名
    （``./.cache``），于是"在 A 目录 tune 出来的表，换到 B 目录就读不到"，用户会以为
    tune 白跑了。这种回归不会让任何数值变红：表照样能读，只是永远读不到——所以只能
    钉住"解析出来的绝对路径"本身。

    为什么用子进程：``DEFAULT_DIR`` 是 import 期求值的，在本进程里改 ``HOME`` /
    ``XDG_CACHE_HOME`` 只会得到旧值，还会把本进程的模块状态搅脏。
    """
    pc = _profile_cache()
    assert os.path.isabs(pc.DEFAULT_DIR), (
        f"DEFAULT_DIR 必须是绝对路径（相对 CWD 的目录名正是用户抱怨的那个回归）："
        f"{pc.DEFAULT_DIR!r}")
    assert pc.DEFAULT_DIR.endswith("pyqcd.einsum"), (
        f"DEFAULT_DIR 的最后一段必须是 pyqcd.einsum：{pc.DEFAULT_DIR!r}")

    with _isolated() as tmp:
        home = os.path.join(tmp, "home")
        xdg = os.path.join(tmp, "xdg")
        elsewhere = os.path.join(tmp, "elsewhere")
        for path in (home, xdg, elsewhere):
            os.makedirs(path, exist_ok=True)

        # (a) 不设 XDG_CACHE_HOME → ~/.cache/pyqcd.einsum（HOME 被临时改到临时树里，
        #     所以这一条既验了公式，又完全不碰用户真实家目录）。
        env_no_xdg = _subprocess_env(home=home, xdg=None)
        no_xdg = _default_dir_from_subprocess(env_no_xdg, cwd=_repo_root())
        expected_no_xdg = os.path.abspath(
            os.path.join(home, ".cache", "pyqcd.einsum"))
        assert no_xdg == expected_no_xdg, (
            f"未设 XDG_CACHE_HOME 时 DEFAULT_DIR 应为 ~/.cache/pyqcd.einsum："
            f"{no_xdg!r} != {expected_no_xdg!r}")
        # CWD 无关：换到临时树里再 import 一次，结果必须逐字相同。
        from_elsewhere = _default_dir_from_subprocess(env_no_xdg, cwd=elsewhere)
        assert from_elsewhere == no_xdg, (
            "DEFAULT_DIR 随当前工作目录变了（这正是当初的回归）：",
            no_xdg, from_elsewhere)
        assert not os.path.exists(no_xdg), (
            f"只是 import 就建出了缺省目录（读路径不建目录）：{no_xdg}")

        # (b) 设了 XDG_CACHE_HOME → $XDG_CACHE_HOME/pyqcd.einsum，且 XDG 优先。
        env_xdg = _subprocess_env(home=home, xdg=xdg)
        with_xdg = _default_dir_from_subprocess(env_xdg, cwd=_repo_root())
        expected_with_xdg = os.path.abspath(os.path.join(xdg, "pyqcd.einsum"))
        assert with_xdg == expected_with_xdg, (
            f"设了 XDG_CACHE_HOME 时 DEFAULT_DIR 应为 $XDG_CACHE_HOME/pyqcd.einsum："
            f"{with_xdg!r} != {expected_with_xdg!r}")
        assert with_xdg != no_xdg, (
            "XDG_CACHE_HOME 没生效（它必须压过 ~/.cache）", with_xdg, no_xdg)
        assert not os.path.exists(with_xdg), (
            f"只是 import 就建出了缺省目录：{with_xdg}")


# ── 4. 峰值内存逐组 ─────────────────────────────────────────────────────────
def test_peak_memory_is_measured_and_rendered_per_group() -> None:
    """需求 #4：``--tune`` 每组都要有峰值内存，且报告里看得见。

    四个公开函数（``reset_peak_memory`` / ``peak_memory_mb`` / ``release_memory`` /
    ``format_peak_mb``）都存在且可调用；``reset`` 之后分配 150 MB ``np.uint8``，
    峰值至少 100 MB；``release_memory()`` 不抛；``format_peak_mb(0.0) == "-"``、
    ``format_peak_mb(1234.4) == "1234MB"``；``tune_comb_lines()`` 的输出里含 ``peak=``
    （未测到时是 ``peak=-``，不是整列消失）。

    为什么它会回归：``ru_maxrss``/``VmHWM`` 是**进程生命周期**峰值、降不下来。一旦
    有人把"本组峰值"直接读成 ``ru_maxrss``，第一组之后每一组都会顶着前几组的历史峰值，
    报告里写"第 7 组吃了 6 GB"而那 6 GB 是第 3 组留下的——数值全对，报告全错。
    ``format_peak_mb`` 的 ``0.0 -> "-"`` 是"没测到"与"真的用满 0 MB"的唯一区别。
    """
    memory = _memory_module()

    for name in ("reset_peak_memory", "peak_memory_mb", "release_memory",
                 "format_peak_mb"):
        function = getattr(memory, name, None)
        assert callable(function), (
            f"_memory.{name} 必须存在且可调用（这是 --tune 每组的唯一测量口径）")

    memory.reset_peak_memory()
    baseline = memory.peak_memory_mb()
    assert isinstance(baseline, float), (type(baseline), baseline)
    payload = np.ones(ALLOC_MB * 1024 * 1024, dtype=np.uint8)
    assert payload.nbytes == ALLOC_MB * 1024 * 1024, payload.nbytes
    try:
        peak = memory.peak_memory_mb()
        breakdown = memory.peak_memory_breakdown()
    finally:
        del payload
        gc.collect()
        released = memory.release_memory()

    assert released is None, ("release_memory() 必须返回 None", released)
    assert isinstance(peak, float), (type(peak), peak)
    assert peak >= ALLOC_MIN_PEAK_MB, (
        f"分配 {ALLOC_MB} MB 后 peak_memory_mb() 只有 {peak:.1f} MB"
        f"（baseline={baseline:.1f} MB, breakdown={breakdown}）——"
        "瞬时 RSS 采样那一路没抓到这块分配")
    assert peak >= baseline, (
        f"峰值比 reset 之后的基线还小：{baseline:.1f} -> {peak:.1f} MB")
    assert isinstance(breakdown, dict), breakdown
    assert breakdown.get("cpu", 0.0) >= ALLOC_MIN_PEAK_MB - 1e-9, (
        f"CPU 分量没看到 {ALLOC_MB} MB 的分配：{breakdown}")

    assert memory.format_peak_mb(0.0) == "-", memory.format_peak_mb(0.0)
    assert memory.format_peak_mb(1234.4) == "1234MB", memory.format_peak_mb(1234.4)

    cli = _cli()
    row = {"kernel_force": "generic", "as_equivalence": "false",
           "optimize": "greedy", "backend": "numpy", "dtype": "complex128",
           "device": "cpu", "median_ms": 1.5, "peak_memory_mb": 12.0,
           "reason": "ok"}
    lines = cli.tune_comb_lines(1, 1, row)
    text = "\n".join(lines)
    assert "peak=" in text, ("每组的完整行必须带 peak= 列", lines)
    assert "peak=12MB" in text, ("已测到的组要写出数值", lines)
    row_unmeasured = dict(row, peak_memory_mb=None)
    text_unmeasured = "\n".join(cli.tune_comb_lines(1, 1, row_unmeasured))
    assert "peak=-" in text_unmeasured, (
        "没测到峰值时写 peak=-（与 format_peak_mb 同一口径），不能整列消失",
        text_unmeasured)


# ── 5. 剪枝截止 2× ──────────────────────────────────────────────────────────
def test_prune_cutoff_is_twice_the_best_wall() -> None:
    """需求 #5：剪枝截止是 **2 × min_wall_ms**，且 ``原因=`` 里两个数都给。

    ``MIN_WALL_FACTOR == 2.0``；``MinWall(ms=100.0).cutoff_ms == 200.0``；
    ``exceeds(199.0) is False``、``exceeds(201.0) is True``；
    ``prune_reason(2100, 1000)`` 同时含 ``min_wall_ms=1000.000000`` 与
    ``cutoff_ms=2000.000000``。

    为什么它会回归：``min_wall_ms`` 是"目前最快那一组的中位"，而真正比较用的是它的
    2 倍。把 1× 当截止会让第一组之后几乎所有组都被剪掉（"最优"退化成"第一个跑完的"），
    而报告里只写 ``wall_ms=… min_wall_ms=…`` 时读者拿 ``wall_ms > min_wall_ms`` 自己去
    对，会得出"这个组本该被剪却没被剪"的错误结论。两个数都不进数值，只能靠断言钉住。
    """
    axes = _tune_axes()

    assert float(axes.MIN_WALL_FACTOR) == 2.0, (
        f"剪枝截止倍数必须是 2×，实际 {axes.MIN_WALL_FACTOR!r}")
    probe = axes.MinWall(ms=100.0)
    assert probe.cutoff_ms == 200.0, (
        f"MinWall(ms=100.0).cutoff_ms 应为 200.0，实际 {probe.cutoff_ms!r}")
    assert probe.exceeds(199.0) is False, (
        "1×~2× 之间的组不该被剪枝（199.0 ms < 200.0 ms 截止）")
    assert probe.exceeds(201.0) is True, (
        "超过 2× 截止必须被剪枝（201.0 ms > 200.0 ms）")
    assert probe.exceeds(200.0) is False, (
        "恰好等于截止不算超界（判据是严格大于）")

    reason = axes.prune_reason(2100, 1000)
    assert "min_wall_ms=1000.000000" in reason, (
        f"原因里必须给出 min_wall_ms 的六位小数原值：{reason!r}")
    assert "cutoff_ms=2000.000000" in reason, (
        f"原因里必须给出真正的截止（2 × min_wall_ms）：{reason!r}")

    unlimited = axes.MinWall(ms=None)
    assert unlimited.cutoff_ms is None, unlimited.cutoff_ms
    assert unlimited.exceeds(10 ** 9) is False, "ms=None 表示不设上界，一组都不剪"


# ── 6. 记录表 ───────────────────────────────────────────────────────────────
def test_record_table_replaces_and_stays_out_of_dispatch() -> None:
    """需求 #6：记录表 ``pyqcd.einsum.record.csv`` 的文件名/列/写入语义 + 不参与选路。

    ``RECORD_FILENAME == "pyqcd.einsum.record.csv"``；``record_path()`` 与
    ``profile_path()`` 同目录；``RECORD_COLUMNS`` 是 ``PROFILE_COLUMNS`` 的超集且多出的
    恰好是 ``peak_memory_mb``；在 ``data/tmp`` 下 ``write_records([r1,r2])`` 后
    ``read_records()`` 长度 2、再 ``write_records([r3])`` 后长度 1（**整表替换**，不是
    追加）；``_api`` 源码里 ``read_records`` 只出现在 :func:`_cheaper_memory_hint` 里。

    为什么它会回归：记录表是"这一次 tune 的全量参考"，**不是缓存**。一旦有人把
    ``read_records`` 接到选路上（或把追加改成"每次都写一份"），命中判定就会从"没被选中的
    组"里挑答案——"tune 的结论"变成"tune 的过程"，而数值一位都不会变。列集合那一条同理：
    往 ``PROFILE_COLUMNS`` 里塞 ``peak_memory_mb`` 会改掉配置表的冻结格式，命中侧与多条
    老契约一起失配。
    """
    pc = _profile_cache()
    api = _api_module()

    assert pc.RECORD_FILENAME == "pyqcd.einsum.record.csv", pc.RECORD_FILENAME
    assert os.path.basename(pc.record_path()) == pc.RECORD_FILENAME, pc.record_path()
    assert os.path.dirname(pc.record_path()) == os.path.dirname(pc.profile_path()), (
        "记录表必须与配置表同目录（用户只记一个目录）：",
        pc.record_path(), pc.profile_path())
    assert set(pc.PROFILE_COLUMNS) <= set(pc.RECORD_COLUMNS), (
        "记录表必须是配置表的列超集：",
        sorted(set(pc.PROFILE_COLUMNS) - set(pc.RECORD_COLUMNS)))
    extra = set(pc.RECORD_COLUMNS) - set(pc.PROFILE_COLUMNS)
    assert extra == {"peak_memory_mb"}, (
        f"记录表多出的列必须恰好是 peak_memory_mb，实际 {sorted(extra)}")
    assert len(set(pc.RECORD_COLUMNS)) == len(pc.RECORD_COLUMNS), (
        "RECORD_COLUMNS 里有重复列名", pc.RECORD_COLUMNS)

    with _isolated() as tmp:
        directory = os.path.join(tmp, "cache")
        pc.configure(directory)
        first_batch = [_record(optimize="greedy", kernel="generic",
                              wall_ms=1.0, peak_memory_mb=10.0),
                       _record(optimize="dp", kernel="backend",
                              wall_ms=2.0, peak_memory_mb=3.0)]
        written = pc.write_records(first_batch)
        assert written == len(first_batch), (written, len(first_batch))
        first = pc.read_records()
        assert len(first) == 2, ("两行必须真的读回来", len(first), first)
        for row in first:
            assert set(pc.RECORD_COLUMNS) <= set(row), (
                "读回来的每一行都要补齐全部列（含 peak_memory_mb）",
                sorted(set(pc.RECORD_COLUMNS) - set(row)))

        second_batch = [_record(optimize="optimal", kernel="generic",
                                wall_ms=3.0, peak_memory_mb=1.0)]
        assert pc.write_records(second_batch) == 1
        second = pc.read_records()
        assert len(second) == 1, (
            "记录表必须整表替换（不是追加）：第二次写入后应只剩 1 行",
            len(second), second)
        assert second[0].get("optimize") == "optimal", second[0]
        assert second[0].get("peak_memory_mb") == "1.000", second[0]

        path = pc.record_path()
        assert os.path.isfile(path), path
        assert os.path.dirname(path) == os.path.abspath(directory), path
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows_on_disk = list(csv.reader(handle))
        assert len(rows_on_disk) == 2, (
            "磁盘上也只该剩表头 + 1 行数据（整表替换）", rows_on_disk)
        assert rows_on_disk[0] == list(pc.RECORD_COLUMNS), (
            "表头必须逐字等于 RECORD_COLUMNS",
            rows_on_disk[0], list(pc.RECORD_COLUMNS))

    source = inspect.getsource(api)
    tree = ast.parse(source)
    hits = [(lineno, text) for lineno, text in
            enumerate(source.splitlines(), start=1) if "read_records" in text]
    assert hits, "_api 里找不到 read_records——'更省内存'提示的数据源断了"
    for lineno, text in hits:
        owners = _enclosing_functions(tree, lineno)
        assert owners, (
            f"_api:{lineno} 的 read_records 在模块级（不在任何函数里）：{text!r}")
        assert owners[0] == "_cheaper_memory_hint", (
            f"_api:{lineno} 的 read_records 落在 {owners[0]!r} 里，"
            f"但记录表只许用于 _cheaper_memory_hint 这条提示：{text!r}")


# ── 7. warmup=1 / rounds=3 ──────────────────────────────────────────────────
def test_cli_defaults_are_zero_warmup_and_one_round() -> None:
    """需求 #7：``DEFAULT_WARMUP == 0`` 且 ``DEFAULT_ROUNDS == 1``（2026-11-20 用户裁定）。

    除了两个模块常量，还要钉住 argparse 与 ``run()`` / ``run_tune()`` 的函数缺省——
    三处必须同时是 0 / 1。

    为什么它会回归：这是"一次 tune 每组做几次调用"的唯一来源，它同时决定
    ``组级窗口 = --tune-timeout × (1 + warmup + rounds)``。用户在生产形状上实测
    单组 ``dedicated/false/greedy`` 在 1/3 下要 **40.1 s**、29 组就是十几分钟，
    因此缺省改成"不预热、只测一次"。反过来也一样要钉：有人为了"更稳"把它调回
    1/3，遍历立刻慢 4 倍，而报告字面上看不出来。
    """
    cli = _cli()

    assert cli.DEFAULT_WARMUP == 0, (
        f"DEFAULT_WARMUP 必须是 0，实际 {cli.DEFAULT_WARMUP!r}"
        "（用户裁定：预热缺省零次；要稳态数字就显式给值）")
    assert cli.DEFAULT_ROUNDS == 1, (
        f"DEFAULT_ROUNDS 必须是 1，实际 {cli.DEFAULT_ROUNDS!r}"
        "（用户裁定：重复轮次缺省一次，即每格只测一次）")
    assert isinstance(cli.DEFAULT_WARMUP, int) and isinstance(cli.DEFAULT_ROUNDS, int), (
        type(cli.DEFAULT_WARMUP), type(cli.DEFAULT_ROUNDS))

    parser = cli.build_parser(language="en")
    args = parser.parse_args(["--equation", "ij,jk->ik",
                              "--shape", "[4,4],[4,4]"])
    assert args.warmup == 0, ("argparse 的 --warmup 缺省没跟上（应为 0）", args.warmup)
    assert args.rounds == 1, ("argparse 的 --rounds 缺省没跟上（应为 1）", args.rounds)

    for function, label in ((cli.run, "run"), (cli.run_tune, "run_tune")):
        parameters = inspect.signature(function).parameters
        assert parameters["warmup"].default == 0, (
            f"{label}() 的 warmup 缺省必须也是 0", parameters["warmup"].default)
        assert parameters["rounds"].default == 1, (
            f"{label}() 的 rounds 缺省必须也是 1", parameters["rounds"].default)


# ── 8. "存在更省内存的配置"提示 ──────────────────────────────────────────────
def test_cheaper_memory_hint_requires_half_the_peak() -> None:
    """需求 #8：命中时要提示"存在更省内存的配置"，但候选只省 30% 时必须**不**提示。

    直接调 ``_api._cheaper_memory_hint(row)``：用 ``write_records`` 写同键两行
    （当前 40 MB / 候选 8 MB，``wall_ms`` 5.0 / 5.2），``write_winner`` 写当前那条，
    断言返回的字符串非空、含 ``更省内存``（zh）或 ``lower-memory``（en）、含
    ``record.csv`` 与 ``profile.csv`` 两个文件名；把候选峰值改成 28 MB（只省 30%）后
    返回 ``None``。

    为什么它会回归：这条提示是"tune 的全量快照"唯一的**用户可见出口**，但它只是一行
    打印——判据（同键、峰值 < 当前一半、耗时 ≤ 当前 1.2 倍）写松一点就成了噪音：每个
    命中都会建议用户"换一条更省的"，而那一条可能慢一倍。反过来判据写严到永不触发，
    用户永远不知道记录表里躺着一个省 80% 内存的方案，两者都不会让任何数值测试变红。
    """
    pc = _profile_cache()
    api = _api_module()
    specialize = _specialize()

    with _isolated() as tmp:
        directory = os.path.join(tmp, "cache")
        pc.configure(directory)
        key = _key()
        current = dict(key, kernel="generic", optimize="greedy",
                       wall_ms=5.0, peak_memory_mb=40.0)
        candidate = dict(key, kernel="generic", optimize="dp",
                         wall_ms=5.2, peak_memory_mb=8.0)
        assert pc.write_records([current, candidate]) == 2
        pc.write_winner(key, {"kernel": "generic", "optimize": "greedy",
                              "wall_ms": 5.0})
        row = pc.read_winner(key)
        assert row is not None, "write_winner 之后必须能读回那一行"
        assert row.get("kernel") == "generic", row
        assert row.get("optimize") == "greedy", row

        saved_language = specialize.set_language("zh")
        try:
            zh = api._cheaper_memory_hint(row)
            specialize.set_language("en")
            en = api._cheaper_memory_hint(row)
        finally:
            specialize.set_language(saved_language)

        assert isinstance(zh, str) and zh, (
            f"候选省了一半内存（40 → 8 MB）时必须给出提示，实际 {zh!r}")
        assert "更省内存" in zh, ("中文提示必须含『更省内存』", zh)
        assert "record.csv" in zh, ("提示必须点名记录表文件名", zh)
        assert "profile.csv" in zh, ("提示必须点名配置表文件名", zh)
        assert "optimize=dp" in zh, ("提示必须点名候选那一格", zh)

        assert isinstance(en, str) and en, (
            f"英文下同样必须给出提示，实际 {en!r}")
        assert "lower-memory" in en, ("英文提示必须含 lower-memory", en)
        assert "record.csv" in en, ("英文提示必须点名记录表文件名", en)
        assert "profile.csv" in en, ("英文提示必须点名配置表文件名", en)

        # 候选只省 30%（40 → 28 MB）：不满足"一半"的门槛，必须静默。
        weak_candidate = dict(key, kernel="generic", optimize="optimal",
                              wall_ms=5.2, peak_memory_mb=28.0)
        assert pc.write_records([current, weak_candidate]) == 2
        assert api._cheaper_memory_hint(row) is None, (
            "候选只省 30% 时不许提示（门槛是严格小于当前峰值的一半）")

        # 只省一半但慢一倍：同样不提示（耗时门槛 1.2×）。
        slow_candidate = dict(key, kernel="generic", optimize="branch_all",
                              wall_ms=50.0, peak_memory_mb=8.0)
        assert pc.write_records([current, slow_candidate]) == 2
        assert api._cheaper_memory_hint(row) is None, (
            "更省但慢 10 倍的候选不是建议（耗时门槛是当前 1.2×）")


# ── 9. 首次调用自动 tune ────────────────────────────────────────────────────
def test_autotune_runs_once_per_key_on_first_call() -> None:
    """需求 #9：首次调用（表里没有这次调用的行）先自动 tune 一次，且同键只试一次。

    ``_autotune.AUTOTUNE_ENV == "PYQCD_EINSUM_AUTOTUNE"``；``enabled("0") is False``、
    ``enabled("1") is True``、``os.environ[AUTOTUNE_ENV]="0"`` 时 ``enabled() is False``；
    ``reset()`` 后 ``_ATTEMPTED`` 为空；把 ``_run_locked`` 换成计数器（返回 ``False``）
    后连调三次 ``run_for`` 同一 key，计数器必须是 **1**；``_api._profile_begin`` 的源码里
    含 ``_autotune.run_for``。

    为什么它会回归：开关解析这里有两个经典坑——``bool("0")`` 是 ``True``（字符串原样
    交给布尔判定，"关掉"变成"强制打开"），以及"每进程只试一次"的账本。少了账本，一次
    跑不出可用组合的 tune 会在**每一次** ``contract()`` 上重演（CPU 上几十组 × 每组
    ``1+warmup+rounds`` 次调用），表现为"程序卡死"而不是"结果不对"；账本记错了（只在
    成功时记）则同样的灾难换个触发条件复活。这条链路没有一条会改数值。
    """
    autotune = _autotune_module()
    api = _api_module()

    assert autotune.AUTOTUNE_ENV == "PYQCD_EINSUM_AUTOTUNE", autotune.AUTOTUNE_ENV

    with _isolated():
        os.environ.pop(autotune.AUTOTUNE_ENV, None)
        assert autotune.enabled() is True, "缺省必须开启自动 tune"
        assert autotune.enabled("1") is True, (
            "enabled('1') 必须是 True（字符串 '1' 不是布尔真值以外的任何东西）")
        assert autotune.enabled("0") is False, (
            "enabled('0') 必须是 False——bool('0') 是 True，"
            "这正是'关掉自动 tune'会静默失效的那个坑")
        assert autotune.enabled("false") is False
        assert autotune.enabled("off") is False
        assert autotune.enabled("no") is False

        os.environ[autotune.AUTOTUNE_ENV] = "0"
        assert autotune.enabled() is False, (
            "PYQCD_EINSUM_AUTOTUNE=0 时 enabled() 必须为 False")
        os.environ[autotune.AUTOTUNE_ENV] = "1"
        assert autotune.enabled() is True, (
            "PYQCD_EINSUM_AUTOTUNE=1 时 enabled() 必须为 True")
        os.environ.pop(autotune.AUTOTUNE_ENV, None)

        autotune.reset()
        assert getattr(autotune, "_ATTEMPTED", None) == set(), (
            "reset() 之后 _ATTEMPTED 必须是空集合",
            getattr(autotune, "_ATTEMPTED", _MISSING))

        calls: list[dict[str, Any]] = []

        def counting_run_locked(**kwargs: Any) -> bool:
            calls.append(kwargs)
            return False                      # "tune 没跑出可用组合"也要记账

        key = _key()
        with contextlib.suppress(Exception):
            from unittest.mock import patch as _patch
            with _patch.object(autotune, "_run_locked", counting_run_locked):
                for _ in range(3):
                    result = autotune.run_for(
                        key=key, subscripts="ij,jk->ik",
                        operands=[np.zeros((4, 4)), np.zeros((4, 4))])
                    assert result is False, (
                        "_run_locked 返回 False 时 run_for 必须原样返回 False", result)
                # 换一个键：允许再试一次（账本按 key 记）。
                other = _key(equation="ab,bc->ac", shape=[(4, 4), (4, 4)])
                autotune.run_for(
                    key=other, subscripts="ab,bc->ac",
                    operands=[np.zeros((4, 4)), np.zeros((4, 4))])

        assert len(calls) == 2, (
            "同一个键在一个进程里只许试一次：三次 run_for 同一 key + 一次新 key，"
            f"期望 2 次 _run_locked，实际 {len(calls)} 次")
        assert len(getattr(autotune, "_ATTEMPTED", set())) == 2, (
            "_ATTEMPTED 必须按 key 记账（失败也要记）", autotune._ATTEMPTED)

    source = inspect.getsource(api._profile_begin)
    assert "_autotune.run_for" in source, (
        "_api._profile_begin 必须真的调用 _autotune.run_for——"
        "缺行时先 tune 再执行的接线点在这里；不接线就只是多了一个没人用的模块")


TESTS = (
    test_gpu_diagnosis_reports_real_probe_stage,
    test_original_kernel_runs_all_seven_optimizers,
    test_default_profile_dir_lives_in_home_cache,
    test_peak_memory_is_measured_and_rendered_per_group,
    test_prune_cutoff_is_twice_the_best_wall,
    test_record_table_replaces_and_stays_out_of_dispatch,
    test_cli_defaults_are_zero_warmup_and_one_round,
    test_cheaper_memory_hint_requires_half_the_peak,
    test_autotune_runs_once_per_key_on_first_call,
)


# ── 收尾：把本契约碰过的进程级状态还给后面的用例 ─────────────────────────────
def _teardown() -> None:
    """``_autotune.reset()`` + 还内存池 + 把配置表模块清到干净态。"""
    try:
        autotune = _autotune_module()
        autotune.reset()
    except Exception:                              # noqa: BLE001 - 收尾尽力而为
        pass
    try:
        memory = _memory_module()
        memory.release_memory()
        memory.reset_peak_memory()
        memory.release_memory()
    except Exception:                              # noqa: BLE001 - 收尾尽力而为
        pass
    try:
        _reset_profile_state(_profile_cache())
        _reset_api_state(_api_module())
    except Exception:                              # noqa: BLE001 - 收尾尽力而为
        pass


# ── 汇总入口（骨架照抄 _einsum_peak_memory_contract）────────────────────────
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
        except Exception as exc:                   # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    return passed, skipped, failures


def test_einsum_nine_items_acceptance_contracts() -> None:
    """九项需求的端到端验收契约（逐条运行 ``TESTS``）。

    失败时抛 :class:`AssertionError` 并列出每条失败；全部跳过时抛
    :class:`unittest.SkipTest`（不留一个"看起来通过"的空套件）。

    收尾（即使有用例失败也会跑）：``_autotune.reset()``、``release_memory()`` +
    ``reset_peak_memory()``、把 ``_profile_cache`` / ``_api`` 的进程级状态清回干净态——
    本契约造过的表、语言覆盖与"已尝试"账本都不该漏进后面的用例。
    """
    started = time.perf_counter()
    passed, skipped, failures = _run()
    try:
        _teardown()
    except Exception as exc:                       # noqa: BLE001 - 收尾失败也要报
        failures.append(f"teardown: {type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started
    if failures:
        raise AssertionError(
            f"{len(failures)} 条九项验收契约失败（{passed} passed, {skipped} skipped,"
            f" {elapsed:.2f}s）：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"九项验收契约全部跳过（{skipped} skipped）：没有可验证的来源")


def main() -> int:
    """独立运行入口：``python -B -m pyqcd.testing._einsum_nine_items_acceptance_contract``。"""
    started = time.perf_counter()
    passed, skipped, failures = _run()
    try:
        _teardown()
    except Exception as exc:                       # noqa: BLE001 - 报告用
        failures.append(f"teardown: {type(exc).__name__}: {exc}")
        print(f"FAIL teardown: {type(exc).__name__}: {exc}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {len(failures)} failed "
          f"({time.perf_counter() - started:.2f}s)", flush=True)
    for failure in failures:
        print(f"  - {failure}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
