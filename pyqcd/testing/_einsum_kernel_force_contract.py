"""契约：强制指定专用内核（``kernel_force=`` / ``PYQCD_EINSUM_KERNEL_FORCE``）。

强制内核是一层**纯选路**语义，数值上永远等价于通用路径，因此普通数值测试抓不到
它的缺陷：标签解析错了只是"强制没生效"，分派错了只是"跑成了别的内核"，两者都不
影响结果。本模块把下面八条固化成可独立失败的断言：

1. **取值矩阵**：``parse`` / ``forced_labels`` / ``generic_only`` 对
   ``None``/``auto``/``generic``/单标签/多标签/带空格/混合大小写的返回；非法标签与
   ``generic`` 混写必须抛 ``ValueError``，且消息里能查到合法集合。本条按"纯解析"与
   "三态判定 + 环境/参数优先级"拆成两个用例（所以 ``TESTS`` 是 9 条对应下面 8 点）。
2. **``generic`` 与 ``specialize=False`` 等价**：同一条语料方程两者逐元素一致，
   且都不发"specialised kernel did not run"告警。
3. **逻辑名按后端解析**：``dedicated`` / ``topology`` 在 numpy 操作数上必须解析成
   ``kupl_*``、在 cupy 操作数上必须解析成 ``cuda_*``（解析只看操作数模块名，不需要
   真 GPU）。
4. **真跑一次**：强制 ``dedicated`` 跑语料方程，``last_kernel()['ran']`` 必须是它解析
   出的具体标签 ``kupl_dedicated``，结果与 ``numpy.einsum`` 相对误差 ≤ 1e-12，且
   **零告警**。
5. **负缓存旁路**：确定性 status=4 拒绝连续两次调用 → 第二次因负缓存命中；在
   ``bypass_negative_cache()`` 内必须真试一次（原因不含 ``negative-cache hit``，
   且 ``negative_cache_info()`` 与进入前逐项相同）；``contract`` 的强制路径自带
   旁路，两次调用都不许报负缓存命中。
6. **参数优先于环境变量**：``PYQCD_EINSUM_KERNEL_FORCE=topology`` 时传
   ``kernel_force='generic'`` 生效（``requested == ()``），不传则环境里的逻辑名生效
   （``requested == ('topology',)``）。
7. **冲突早失败**：``out=`` / ``specialize=False`` 与具体标签组合必须抛
   ``ValueError``，且**一次内核都没试**（``_LAST_KERNEL`` 未被写过）。
8. **回归**：不强制时仍是"尝试 → 失败 → 一次告警 → 回落"，结果与 numpy 一致；
   进程内的形态告警去重不得让自动分派变成"静默回落"。

不做任何计时/性能断言：本模块只关心"选了哪条路"，不关心它有多快。

可直接运行::

    python -m pyqcd.testing._einsum_kernel_force_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from ..einsum import (_api, _kernel_force, _kernel_kupl, _profile_cache,
                      _specialize, _threads, contract)

#: 强制内核的环境变量名（与实现同源，避免测试里写第二份字面量）。
FORCE_ENV = _kernel_force.KERNEL_FORCE_ENV

#: ``_api`` 的集合型进程状态：都是"只报一次"的去重闸门。第 4 条要求"零告警"，
#: 第 8 条要求"自动分派失败必须告警一次"——两个方向都会被别人的残留判反。
_API_SET_FIELDS = (
    "_NATIVE_DISABLED", "_NATIVE_WARNED_REASONS", "_SPECIALIZED_KERNEL_WARNED",
    "_KERNEL_SKIP_WARNED", "_PROFILE_NOTICED",
)

#: ``_profile_cache`` 的进程状态：``contract()`` 缺省开着画像，一次调用就会改它们。
#: 值即模块初值（TV 重设计后只剩"钉住的目录 / 解析缓存 / 遍历计数"）。
_CACHE_PLAIN_FIELDS = {
    "_CONFIGURED_DIR": "", "_LAST_ERROR": None, "_HOST": None,
    "_TABLE": None, "_TUNING": 0,
}

#: 语料内方程：与 ``_kernel_kupl._PROBE_EQUATION`` 同一条，dim=2 下 proposed
#: dedicated 步表与拓扑步表都可表达，且专用内核**接受这一字面拼写**。
EQUATION = "cadb,menf,ac,em,bd,fn->"

#: 语料方程各操作数的轴长；小规模让步表可编码、结果又好对拍。
DIM = 2

#: 专用 kernel 没跑成时 ``_api`` 发的告警原文片段（契约要盯的那一条）。
SKIP_WARNING = "specialised kernel did not run"


def _operands(dtype=np.complex128, seed: int = 20261116):
    """按方程各操作数的 rank 造一组等长 ``DIM`` 的 C 连续操作数。"""
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    values = []
    for term in EQUATION.split("->")[0].split(","):
        shape = (DIM,) * len(term)
        data = rng.standard_normal(shape)
        if dtype.kind == "c":
            data = data + 1j * rng.standard_normal(shape)
        values.append(np.ascontiguousarray(data.astype(dtype)))
    return values


def _reference(operands):
    """参考答案：numpy 自己的 n 元 einsum。"""
    return np.einsum(EQUATION, *operands)


def _rel_max(result, expected) -> float:
    """相对最大偏差（按参考答案幅度归一；全零时用 1 作尺度）。"""
    actual = np.asarray(result)
    scale = float(np.max(np.abs(expected))) or 1.0
    return float(np.max(np.abs(actual - expected))) / scale


def _capture(call):
    """安静地执行一次调用：关闭首次命中提示，并收集本次的全部 warning。"""
    with warnings.catch_warnings(record=True) as caught, \
            contextlib.redirect_stdout(io.StringIO()), \
            patch.dict(os.environ, {_specialize.NOTICE_ENV: "0"}):
        warnings.simplefilter("always")
        value = call()
    return value, caught


def _skip_warnings(caught):
    """只挑出"专用 kernel 该跑却没跑"那一条告警的正文。"""
    return [str(item.message) for item in caught if SKIP_WARNING in str(item.message)]


def _require_specialized_path():
    """专用路径被环境关掉时整条用例不可达 —— 跳过必须写明原因，不许静默通过。"""
    if not _specialize.enabled():
        raise unittest.SkipTest(
            f"环境关闭了专用内核路径（{_specialize.DISABLE_ENV}=0）；"
            "强制分支不可达，本用例无意义")


@contextlib.contextmanager
def _isolated_state():
    """快照并恢复强制内核涉及的全部进程状态。

    覆盖四块：

    * ``os.environ``：强制开关、fused 开关、画像开关/目录与两个标准线程变量；
    * ``_api``：``_LAST_KERNEL`` / ``_FUSED_ENV_OWNED`` 与五个
      "只报一次"的去重集合。第 4 条要"零告警"、第 8 条要"告警一次"——两个方向都
      会被上一条用例留下的去重记录判反；
    * ``_profile_cache``：``contract()`` 缺省开着画像，一次调用就会改索引、目录与
      计数器（``_disk_rows`` 还参与 ``flush`` 的整表重写判断）；
    * ``_kernel_kupl`` 的负缓存/计数/最近失败与 ``LAST_INFO``；
    * ``_threads``：``reset()`` 只复位 applied/omp/blas/source，``exported_env`` /
      ``runtime_applied`` / ``runtime_attempts`` / ``profile`` 会一直留着。

    进入前快照、退出后恢复（不是清空了事）：任何一条用例失败都不会把状态漏给下一条。

    画像与目录也一并隔离：``contract()`` 缺省走画像（``profile=None``），而画像给出的
    ``kernel`` 建议会**抢在**强制语义之前改写选路——不把目录指到临时树，
    "强制 dedicated 必须真跑成"这类断言读到的可能是别人留下的建议。
    """
    saved = {
        "last_kernel": _api._LAST_KERNEL,
        "fused_owned": _api._FUSED_ENV_OWNED,
        "api_sets": {name: set(getattr(_api, name)) for name in _API_SET_FIELDS},
        "env": {name: os.environ.get(name) for name in (
            FORCE_ENV, _specialize.FUSED_DISABLE_ENV,
            _profile_cache.PROFILE_ENV, _profile_cache.DIR_ENV,
            _threads.STANDARD_OMP_ENV, _threads.STANDARD_BLAS_ENV)},
        "failure": dict(_kernel_kupl.LAST_FAILURE),
        "info": dict(_kernel_kupl.LAST_INFO),
        "cache_plain": {name: getattr(_profile_cache, name)
                        for name in _CACHE_PLAIN_FIELDS},
        "threads": dict(_threads._state),
    }
    with _kernel_kupl._NEGATIVE_CACHE_LOCK:
        saved["cache"] = dict(_kernel_kupl._NEGATIVE_CACHE)
        saved["stats"] = dict(_kernel_kupl._NEGATIVE_CACHE_STATS)
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-kernel-force-") as root:
            os.environ[_profile_cache.DIR_ENV] = os.path.join(root, "profile")
            _reset_profile_state()
            yield
    finally:
        _api._LAST_KERNEL = saved["last_kernel"]
        _api._FUSED_ENV_OWNED = saved["fused_owned"]
        for name, value in saved["api_sets"].items():
            getattr(_api, name).clear()
            getattr(_api, name).update(value)
        for name, value in saved["env"].items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for name, value in saved["cache_plain"].items():
            setattr(_profile_cache, name, value)
        _threads._state.clear()
        _threads._state.update(saved["threads"])
        _kernel_kupl.LAST_FAILURE.clear()
        _kernel_kupl.LAST_FAILURE.update(saved["failure"])
        _kernel_kupl.LAST_INFO.clear()
        _kernel_kupl.LAST_INFO.update(saved["info"])
        with _kernel_kupl._NEGATIVE_CACHE_LOCK:
            _kernel_kupl._NEGATIVE_CACHE.clear()
            _kernel_kupl._NEGATIVE_CACHE.update(saved["cache"])
            _kernel_kupl._NEGATIVE_CACHE_STATS.clear()
            _kernel_kupl._NEGATIVE_CACHE_STATS.update(saved["stats"])


def _reset_profile_state() -> None:
    """把配置表清到"没配置、没缓存、没遍历"的初始态；不改环境变量。

    ``_TABLE`` 是解析缓存（按 ``(路径, mtime, size)`` 失效），必须置回 ``None``：
    留着上一条用例的表就等于把别人的行当成本用例的结果。
    """
    for name, initial in _CACHE_PLAIN_FIELDS.items():
        setattr(_profile_cache, name, initial)


def _expect_value_error(call, token: str, what: str) -> str:
    """要求 ``call`` 抛 ``ValueError`` 且消息含 ``token``；返回消息正文。"""
    try:
        call()
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError(f"{what} 必须抛 ValueError")
    assert token in message, (what, token, message)
    return message


# ── 1. 取值矩阵 ─────────────────────────────────────────────────────────────

def test_kernel_force_parse_matrix():
    """``parse`` 的取值矩阵：缺省/auto/generic/单标签/多标签/空格/大小写/去重。"""
    parse = _kernel_force.parse
    generic = _kernel_force.GENERIC_LABEL
    backend = _kernel_force.BACKEND_LABEL

    assert parse(None) == ()
    assert parse("") == ()
    assert parse("   ") == ()
    assert parse("auto") == ()
    assert parse("AUTO") == ()
    assert parse("generic") == (generic,)
    assert parse(" Generic ") == (generic,)
    # 单标签 + 混合大小写 + 去空白：返回的是**逻辑名**，不是具体内核标签。
    assert parse("dedicated") == ("dedicated",)
    assert parse("  DEDICATED ") == ("dedicated",)
    assert parse("backend") == (backend,)
    # 多标签：按书写次序（尝试次序就是这个次序）
    assert parse("topology,dedicated") == ("topology", "dedicated")
    assert parse(" TOPOLOGY , dedicated ") == ("topology", "dedicated")
    # 重复标签保序去重（同一次调用不该重复试同一入口）
    assert parse("topology,topology") == ("topology",)
    # 多标签里写 auto 等价于没写
    assert parse("auto,topology") == ("topology",)
    assert parse("auto,backend") == (backend,)

    # 非法标签：必须抛 ValueError，且消息里能查到完整合法集合。旧的具体内核标签
    # 已被移除，必须是非法值——静默接受会让"标签改窄"名存实亡。
    for bad in ("kupl", "cuda_topolog", "kupl-dedicated", "generic,nope",
                "kupl_dedicated", "kupl_topology", "cuda_dedicated",
                "cuda_topology"):
        try:
            parse(bad)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError(f"未知标签 {bad!r} 必须抛 ValueError")
        for label in sorted(_kernel_force.ALLOWED_LABELS):
            assert label in message, (bad, label, message)
        assert generic in message and _kernel_force.AUTO_LABEL in message, (
            bad, message)
        assert backend in message, (bad, message)

    # generic 与具体标签混写没有"半强制"解释，必须早失败而不是悄悄挑一个。
    for bad in ("generic,dedicated", "topology,generic"):
        try:
            parse(bad)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError(f"{bad!r}（generic 与具体标签混写）必须抛 ValueError")
        assert generic in message, (bad, message)

    # backend 是"别用专用内核"，与"只准跑某个专用内核"互斥，同样早失败。
    for bad in ("backend,dedicated", "topology,backend"):
        try:
            parse(bad)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError(f"{bad!r}（backend 与其它标签混写）必须抛 ValueError")
        assert backend in message, (bad, message)


def test_kernel_force_forced_labels_and_generic_only():
    """``forced_labels``/``generic_only`` 的三态判定与"显式参数优先于环境"。"""
    with _isolated_state():
        os.environ.pop(FORCE_ENV, None)
        # 未设环境 → 自动分派
        assert _kernel_force.forced_labels() is None
        assert _kernel_force.generic_only() is False
        # 显式取值压过环境（环境此刻未设）
        assert _kernel_force.forced_labels("auto") is None
        assert _kernel_force.generic_only("auto") is False
        assert _kernel_force.forced_labels("generic") == ()
        assert _kernel_force.generic_only("generic") is True
        # 逻辑名原样返回：解析成具体标签是 ``_api`` 的职责，不在这里。
        assert _kernel_force.forced_labels("dedicated") == ("dedicated",)
        assert _kernel_force.generic_only("dedicated") is False
        assert _kernel_force.forced_labels("backend") == (
            _kernel_force.BACKEND_LABEL,)
        assert _kernel_force.generic_only("backend") is False

        # 环境变量驱动同一套语义（调用时解析，改完即时生效）
        os.environ[FORCE_ENV] = "generic"
        assert _kernel_force.forced_labels() == ()
        assert _kernel_force.generic_only() is True
        # 显式 auto 必须压过环境里的 generic
        assert _kernel_force.forced_labels("auto") is None
        assert _kernel_force.generic_only("auto") is False

        os.environ[FORCE_ENV] = " Topology "
        assert _kernel_force.forced_labels() == ("topology",)
        assert _kernel_force.generic_only() is False
        # 显式 generic 必须压过环境里的具体标签
        assert _kernel_force.forced_labels("generic") == ()
        assert _kernel_force.generic_only("generic") is True

        # 环境里的拼写错误同样要早失败，不能静默当 auto；旧的具体内核标签也在此列。
        os.environ[FORCE_ENV] = "topolog"
        _expect_value_error(
            _kernel_force.forced_labels, "topology",
            f"{FORCE_ENV} 里的非法标签")
        os.environ[FORCE_ENV] = "kupl_topology"
        _expect_value_error(
            _kernel_force.forced_labels, "topology",
            f"{FORCE_ENV} 里的旧标签")


# ── 2. generic ≡ specialize=False ───────────────────────────────────────────

def test_kernel_force_generic_matches_specialize_false():
    """``kernel_force='generic'`` 与 ``specialize=False`` 逐元素一致，且都不发告警。

    需要专用分支可达：``PYQCD_EINSUM_SPECIALIZE=0`` 时 ``_api`` 根本不进该分支，
    也就没有 ``last_kernel()`` 记录可验 —— 那时两条调用的语义完全相同，本契约退化。
    """
    _require_specialized_path()
    operands = _operands()
    expected = _reference(operands)
    with _isolated_state():
        generic, caught_generic = _capture(
            lambda: contract(EQUATION, *operands, kernel_force="generic"))
        record = _api.last_kernel()
        # generic 是"完全跳过专用内核"：记录里 requested 是空元组，一次都没试。
        assert record is not None, "generic 也必须写 last_kernel() 记录"
        assert record["requested"] == (), record
        assert record["ran"] is None, record
        assert record["attempted"] == (), record

        off, caught_off = _capture(
            lambda: contract(EQUATION, *operands, specialize=False))

    assert not _skip_warnings(caught_generic), [str(w.message) for w in caught_generic]
    assert not _skip_warnings(caught_off), [str(w.message) for w in caught_off]
    for name, value in (("kernel_force='generic'", generic), ("specialize=False", off)):
        rel = _rel_max(value, expected)
        assert rel <= 1e-12, (name, rel)
    # 两条通用路径之间也必须逐元素一致（实测差异只来自累加次序，约 1e-14）。
    assert _rel_max(generic, off) <= 1e-12, _rel_max(generic, off)
    assert np.asarray(generic).shape == np.asarray(expected).shape


# ── 3. 逻辑名按后端解析 ─────────────────────────────────────────────────────

def test_kernel_force_resolves_labels_per_backend():
    """``dedicated`` / ``topology`` 只解析到**当前后端**的具体标签。

    用户写的是逻辑名，具体标签由操作数后端决定：numpy → ``kupl_*``、cupy →
    ``cuda_*``。后者不需要真 GPU——解析只看操作数的模块名，用 ``_cupy_operands``
    的桩把两个方向都钉住。
    """
    _require_specialized_path()
    operands = _operands()
    resolved = _api._resolve_forced_labels(("dedicated", "topology"), operands)
    assert resolved == {"dedicated": ("kupl_dedicated",),
                        "topology": ("kupl_topology",)}, resolved

    with patch.object(_api, "_cupy_operands", return_value=True):
        gpu = _api._resolve_forced_labels(("topology", "dedicated"), operands)
    assert gpu == {"topology": ("cuda_topology",),
                   "dedicated": ("cuda_dedicated",)}, gpu

    # backend 不是专用内核角色，不产生任何 attempt 映射。
    assert _api._resolve_forced_labels(("backend",), operands) == {}


# ── 4. 真跑一次 dedicated ───────────────────────────────────────────────────

def test_kernel_force_runs_kupl_dedicated():
    """强制 ``dedicated``：解析成 ``kupl_dedicated`` 真命中、对拍 ≤ 1e-12、零告警。"""
    _require_specialized_path()
    if not _kernel_kupl.enabled():
        raise unittest.SkipTest(
            "环境关闭了 KUPL 内核（PYQCD_EINSUM_KERNEL_KUPL=0），强制入口不可达")
    if _kernel_kupl._function_dedicated() is None:
        raise unittest.SkipTest(
            "本机没有可用的 KUPL 原生库（qcd_kupl_dedicated_apply 缺失），"
            "无法真跑强制内核；这是环境问题，不是契约失败")

    operands = _operands()
    expected = _reference(operands)
    with _isolated_state():
        result, caught = _capture(
            lambda: contract(EQUATION, *operands, kernel_force="dedicated"))
        record = _api.last_kernel()
        path = _kernel_kupl.LAST_INFO.get("path")

    assert record is not None, "强制命中必须留下 last_kernel() 记录"
    assert record["requested"] == ("dedicated",), record
    # ran/attempted 是**具体标签**：requested 与它们不同源，正是本契约要钉的分层。
    assert record["ran"] == "kupl_dedicated", record
    assert record["attempted"] == ("kupl_dedicated",), record
    # 内核自己也要如实报告路线，否则"ran"只是 record 的一面之词。
    assert path == "kupl_dedicated", path
    # 强制命中是正常路径，不该有任何告警（尤其不该有"没跑成"）。
    assert not caught, [str(w.message) for w in caught]
    rel = _rel_max(result, expected)
    assert rel <= 1e-12, ("kupl_dedicated 与 numpy.einsum 不一致", rel)


# ── 5. 负缓存旁路 ───────────────────────────────────────────────────────────

def test_kernel_force_bypasses_negative_cache():
    """强制即真试一次：负缓存不得把强制挡回去，旁路也不得改动缓存计数。

    失败形态用 **int64 操作数**构造：``kupl_dedicated`` 的 dtype 契约不含 int64，
    第一次下发即以 status=4 拒绝并入负缓存 —— 确定性、与本机原生库是否可执行无关。
    """
    _require_specialized_path()
    if not _kernel_kupl._negative_cache_enabled():
        raise unittest.SkipTest(
            "环境关闭了负缓存（PYQCD_EINSUM_NEGATIVE_CACHE=0），本契约无对象可验")

    left = np.ascontiguousarray(np.arange(6, dtype=np.int64).reshape(2, 3))
    right = np.ascontiguousarray(np.arange(15, dtype=np.int64).reshape(3, 5))
    operands = [left, right]
    equation = "ij,jk->ik"

    with _isolated_state():
        _kernel_kupl._negative_cache_clear()

        first = _kernel_kupl.try_dedicated(equation, operands)
        first_reason = str(_kernel_kupl.LAST_FAILURE.get("reason", ""))
        second = _kernel_kupl.try_dedicated(equation, operands)
        second_reason = str(_kernel_kupl.LAST_FAILURE.get("reason", ""))
        assert first is None and second is None, (first, second)
        assert "int64" in first_reason, first_reason
        assert "negative-cache hit" not in first_reason, first_reason
        # 第二次同形调用必须由负缓存给出拒绝理由（否则缓存没生效）。
        assert "negative-cache hit" in second_reason, second_reason

        before = _kernel_kupl.negative_cache_info()
        with _kernel_kupl.bypass_negative_cache():
            assert _kernel_kupl._negative_cache_enabled() is False, (
                "旁路内必须忽略负缓存")
            bypassed = _kernel_kupl.try_dedicated(equation, operands)
            bypassed_reason = str(_kernel_kupl.LAST_FAILURE.get("reason", ""))
            inside = _kernel_kupl.negative_cache_info()
        after = _kernel_kupl.negative_cache_info()

        # 退出旁路后恢复原状（含异常路径）。
        assert _kernel_kupl._negative_cache_enabled() is True
        assert bypassed is None, bypassed
        assert "int64" in bypassed_reason, bypassed_reason
        assert "negative-cache hit" not in bypassed_reason, bypassed_reason
        # 旁路既不读也不写缓存：计数与条目逐项不变（只有 enabled 在上下文内为 False）。
        assert inside["enabled"] is False, inside
        assert after == before, (before, after)

        # contract 的强制路径自带旁路：连续两次都必须真试，都不许报负缓存命中。
        messages = []
        for _ in range(2):
            try:
                _capture(lambda: contract(equation, *operands,
                                          kernel_force="dedicated"))
            except _kernel_force.KernelUnavailableError as exc:
                messages.append(" ".join(str(exc).split()))
            else:
                raise AssertionError(
                    "int64 没有 dedicated 步表，强制必须抛 KernelUnavailableError")
        for message in messages:
            assert "negative-cache hit" not in message, message
            assert "int64" in message, message
            # 消息里逻辑名与具体标签必须对照出现：用户对得上自己写的取值，
            # 也看得见到底是哪个入口被下发过。
            assert "kupl_dedicated（dedicated）" in message, message


# ── 6. 显式参数优先于环境变量 ───────────────────────────────────────────────

def test_kernel_force_parameter_beats_environment():
    """环境写 ``topology`` 时：传 ``kernel_force='generic'`` 生效，不传则环境里的逻辑名生效。

    环境值是逻辑名，所以"不传"时 ``requested`` 必须是 ``('topology',)``；它能不能
    真跑成取决于本机原生库（跑不成就是 ``KernelUnavailableError``，同样证明环境生效）。
    """
    _require_specialized_path()
    operands = _operands()
    expected = _reference(operands)
    with _isolated_state():
        os.environ[FORCE_ENV] = "topology"

        generic, caught = _capture(lambda: contract(EQUATION, *operands,
                                                   kernel_force="generic"))
        record = _api.last_kernel()
        assert record is not None, record
        assert record["requested"] == (), ("显式 generic 必须压过环境", record)
        assert record["ran"] is None and record["attempted"] == (), record
        assert not _skip_warnings(caught), [str(w.message) for w in caught]
        rel = _rel_max(generic, expected)
        assert rel <= 1e-12, rel

        # 同一环境、不传 kernel_force=：环境变量生效，requested 是逻辑名。
        try:
            env_result, _ = _capture(lambda: contract(EQUATION, *operands))
        except _kernel_force.KernelUnavailableError as exc:
            message = str(exc)
            env_record = _api.last_kernel()
            assert "topology" in message, message
            assert "kupl_topology（topology）" in message, message
        else:
            env_record = _api.last_kernel()
            assert env_record["ran"] == "kupl_topology", env_record
            rel = _rel_max(env_result, expected)
            assert rel <= 1e-12, rel
        assert env_record["requested"] == ("topology",), env_record


# ── 7. 冲突早失败 ───────────────────────────────────────────────────────────

def test_kernel_force_conflicts_fail_early():
    """``out=`` / ``specialize=False`` 与具体标签组合：ValueError 且一次都没试。"""
    _require_specialized_path()
    operands = _operands()
    out = np.zeros_like(_reference(operands))
    with _isolated_state():
        # 早失败的判据：专用分支的 record 一次都不该被写（_LAST_KERNEL 保持 None）。
        _api._LAST_KERNEL = None

        _expect_value_error(
            lambda: contract(EQUATION, *operands, kernel_force="dedicated", out=out),
            "out", "out= 与强制具体标签")
        assert _api._LAST_KERNEL is None, ("out= 冲突必须在内核尝试之前失败",
                                           _api._LAST_KERNEL)

        _expect_value_error(
            lambda: contract(EQUATION, *operands, kernel_force="dedicated",
                             specialize=False),
            "specialize", "specialize=False 与强制具体标签")
        assert _api._LAST_KERNEL is None, ("specialize=False 冲突必须在内核尝试之前失败",
                                           _api._LAST_KERNEL)

        # 对照：同一方程 + 同样操作数，换成语义等价的非冲突写法不会抛 —— 证明上面的
        # ValueError 只来自冲突开关，而不是方程/形状本身有问题。
        result, _ = _capture(
            lambda: contract(EQUATION, *operands, kernel_force="generic"))
        rel = _rel_max(result, _reference(operands))
        assert rel <= 1e-12, rel


# ── 8. 回归：自动分派仍是"告警 + 回落" ─────────────────────────────────────

def test_kernel_force_auto_mode_keeps_warning_and_fallback():
    """不强制时行为与改动前一致：仍可能告警 + 回落，结果始终正确。"""
    _require_specialized_path()
    operands = _operands()
    expected = _reference(operands)

    with _isolated_state():
        # (a) 真实自动分派：记录结构不变、requested 为 None、结果与 numpy 一致。
        result, _caught = _capture(lambda: contract(EQUATION, *operands))
        record = _api.last_kernel()
        assert record is not None, record
        assert record["requested"] is None, ("auto 的 requested 必须是 None", record)
        assert record["ran"] in (None, "kupl_dedicated", "kupl_topology"), record
        assert isinstance(record["attempted"], tuple), record
        assert isinstance(record["reasons"], dict), record
        rel = _rel_max(result, expected)
        assert rel <= 1e-12, rel

        # (b) 两个入口都失败：必须仍发一次"没跑成"的告警并回落到通用路径。
        calls = []
        with patch.object(_kernel_kupl, "try_dedicated", return_value=None), \
                patch.object(_kernel_kupl, "try_topology_apply", return_value=None), \
                patch.object(_api, "_warn_kernel_skipped_once",
                             side_effect=lambda *args, **kwargs: calls.append(args)):
            fallback, _ = _capture(lambda: contract(EQUATION, *operands))
        record = _api.last_kernel()

    assert len(calls) == 1, ("自动分派失败必须告警一次", len(calls))
    entry, reasons = calls[0]
    assert str(entry.get("kind", "")).startswith("kernel"), entry
    assert isinstance(reasons, list) and len(reasons) == 2, reasons
    assert record is not None and record["ran"] is None, record
    assert record["attempted"] == ("kupl_dedicated", "kupl_topology"), record
    rel = _rel_max(fallback, expected)
    assert rel <= 1e-12, rel


TESTS = (
    test_kernel_force_parse_matrix,
    test_kernel_force_forced_labels_and_generic_only,
    test_kernel_force_generic_matches_specialize_false,
    test_kernel_force_resolves_labels_per_backend,
    test_kernel_force_runs_kupl_dedicated,
    test_kernel_force_bypasses_negative_cache,
    test_kernel_force_parameter_beats_environment,
    test_kernel_force_conflicts_fail_early,
    test_kernel_force_auto_mode_keeps_warning_and_fallback,
)


def test_einsum_kernel_force_contracts():
    """注册入口：逐条运行 ``TESTS``；全跳过时也 SkipTest（不许静默通过）。"""
    passed = skipped = 0
    failures = []
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
            f"{len(failures)} 条强制内核契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"kernel force contracts did not run ({skipped} skipped)")


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
