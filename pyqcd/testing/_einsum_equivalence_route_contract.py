"""契约：等价路由表 v2（``profile_equivalence.csv`` 的 ``state=ok`` 载荷）。

为什么值得单独立契约
--------------------
``profile_equivalence.csv`` 自 schema v2 起不再只是"能不能改写"的标记，而是一张
**可直接照着执行的路由表**：``state=ok`` 的行除 ``kind`` 之外还带 ``target_equation``
（语料写法，即特化支持的那条 equation）与三个置换字段（``operand_order`` /
``operand_permutations`` / ``output_permutation``，JSON 字符串，``null``/空串表示
"这一步不需要动"）。命中这样的行时上层**直接构造**改写并施加，不再调
:func:`pyqcd.einsum._equivalence.normalize_equation`；``schema_version`` 用来区分
"这条结论是否带了可复用的载荷"，同键多行时**先版本、后状态**。

这些性质全都属于"结果照样正确、只是白花时间或白白丢掉一条已知可用的改写"：普通
数值测试永远碰不到它们（改错了数值也一样逐位对）。本模块把它们固化成能独立失败的
断言：

1. **首次写入带载荷**：无文件 → ``contract(..., profile=True, profile_dir=tmp)``
   （用**语料外但可改写**的方程）→ ``profile_equivalence.csv`` 出现
   ``state=ok`` 的行，其 ``target_equation`` 非空、等于
   :func:`~pyqcd.einsum._equivalence.normalize_equation` 给出的语料写法，
   ``schema_version == 2``；三个置换字段非空时必须是**合法 JSON**（不是 Python repr），
   且 ``output_permutation`` 与真实映射逐位一致（本方程只有输出轴序要动）。
2. **二次命中零匹配**：同一条调用再走两遍——**首次按载荷施加**允许一次数值自检
   （与"重新匹配一次"对拍，见 ``_api._payload_result``；这也是"相信表里的载荷"的
   唯一代价），**第二次命中**（稳态）必须一次都不调
   ``_equivalence.normalize_equation``，且两遍结果 rel ≤ 1e-12（省掉匹配不许改答案）。
3. **方案 A 副产物**：施加改写时内层递归按 ``target_equation`` 画像，因此
   ``profile.csv`` 必须出现 ``equation == target_equation`` 的行——它是"这条路由真的
   被执行过"的旁证，也是后续 `--tune`/画像选优的输入。
4. **版本优先**：手工造 v1(``none``，更新时间更新) 与 v2(``ok``，更旧) →
   ``equivalence_marker`` 返回 **v2**。另有一条**判别式对照**
   （v1 ``ok`` 更新 vs v2 ``none`` 更旧），否则"状态排序恰好也让 ok 赢"会把
   "版本根本没参与排序"这件事盖住。
5. **坏 JSON 不抛**：把载荷字段改成非法 JSON 后，调用照常成功、结果与 ``np.einsum``
   一致，并**退回重查**（spy ≥ 1）——坏行只该丢一次捷径，不该把整条等价步骤关掉。
6. **``state=none`` 仍然是门**：同键写上 ``none`` 后，第二次调用 spy 必须为 0
   （v2 的路由能力不许把负标记这条廉价门挤掉）。
7. **``as_equivalence="force"``**：无视标记与路由，spy ≥ 1，并把该键刷新成
   ``ok`` 行（写入是追加式的，旧 ``none`` 行不会消失，但新证据必须落盘）。
8. **``profile=False`` 零 IO**：一个文件都不许建（含目录里的探针），配一条正对照
   ——同一次调用开着画像时确实会留下路由行。

口径与边界
----------
* 每个用例都在临时树里跑（CWD 也换过去），退出时把环境变量与两个模块的进程内状态
  原样放回；**仓库根不会留下 ``.cache``**。
* **不做任何计时断言**：本模块只关心"写了什么、查了谁、结果还对不对"。
* 不做实现探测式 Skip：这是 v2 的**验收**契约，缺载荷就该红。断言消息里写明缺的是
  哪一项（``target_equation`` / ``schema_version`` / 零匹配 / 方案 A 副产物），
  便于对照实现缺口（见 ``_api`` 的等价路由分支与 ``_profile_cache._marker_pick``）。

可直接运行::

    python -B -m pyqcd.testing._einsum_equivalence_route_contract
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import socket
import tempfile
import unittest
import warnings
from typing import Any, Iterator, Mapping, Sequence
from unittest.mock import patch

import numpy as np

import pyqcd.einsum as pe
from ..einsum import (_api, _cache, _equivalence, _kernel_force, _profile_cache,
                      _specialize)

#: 冻结的文件名（实现里也有同名常量，这里断言它们没被改名）。
ROUTE_NAME = "profile_equivalence.csv"
PROFILE_NAME = "profile.csv"

#: 语料外**但可改写**的方程：``specialize.match`` 为 ``None``，而
#: :func:`~pyqcd.einsum._equivalence.normalize_equation` 给出 ``output-axes`` 映射
#: （``...->ga`` 与 ``...->ag`` 只差输出轴序）。用它验"首次写入载荷"与"二次零匹配"。
ROUTE_EQUATION = "ambn,codp,egfh,ce,mo,bdf,nph->ga"

#: 上一条方程的语料写法（映射的 ``equation``）；只作断言基准，不硬编码到实现里。
TARGET_EQUATION = "ambn,codp,egfh,ce,mo,bdf,nph->ag"

#: 语料外、**不可**改写的方程：用来验 ``state=none`` 的廉价门在 v2 之后仍然有效。
PLAIN_EQUATION = "ij,jk,kl->il"

#: 路由载荷的三个置换字段（JSON 字符串；空串/``null`` = "这一步不需要动"）。
PAYLOAD_FIELDS = ("operand_order", "operand_permutations", "output_permutation")

#: 各标签的轴长；同一个标签在不同操作数里必须等长。
LABEL_SIZES = {
    "i": 2, "j": 3, "k": 4, "l": 4,
    "a": 2, "b": 3, "c": 2, "d": 4, "e": 3, "f": 2, "g": 5, "h": 2,
    "m": 2, "n": 3, "o": 3, "p": 4,
}

#: 本模块会改写/还原的环境变量（含别的契约可能设过的开关）。
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
)


# -- 环境与进程内状态 ------------------------------------------------------------
def _prepare_env(profile: bool = True) -> None:
    """把本模块依赖的开关摆好（``_isolated`` 的 ``finally`` 负责还原）。"""
    os.environ[_cache.CACHE_ENV] = "1"
    os.environ[_equivalence.EQUIVALENCE_ENV] = "1"
    os.environ[_equivalence.AS_EQUIVALENCE_ENV] = "1"
    # ``PYQCD_EINSUM_SPECIALIZE`` 是"是否禁用"：``1`` 表示**没**禁用。等价门在
    # ``specialize=False`` 时提前返回，所以这里必须让它开着。
    os.environ[_specialize.DISABLE_ENV] = "1"
    os.environ[_specialize.NOTICE_ENV] = "0"
    os.environ.pop(_specialize.FUSED_DISABLE_ENV, None)
    os.environ[_profile_cache.PROFILE_ENV] = "1" if profile else "0"
    os.environ.pop(_profile_cache.DIR_ENV, None)
    os.environ.pop(_kernel_force.KERNEL_FORCE_ENV, None)


def _clone(value: Any) -> Any:
    """深一层拷贝模块状态（字典/列表/集合递归，其余按值）。"""
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, set):
        return set(value)
    return value


def _snapshot(module: Any) -> dict[str, tuple[str, Any]]:
    """快照模块状态：容器拷内容、标量/元组记值（锁与函数对象不碰）。"""
    snapshot: dict[str, tuple[str, Any]] = {}
    for name, value in vars(module).items():
        if name.startswith("__") or name.isupper():
            continue
        if isinstance(value, dict):
            snapshot[name] = ("dict", _clone(value))
        elif isinstance(value, list):
            snapshot[name] = ("list", _clone(value))
        elif isinstance(value, set):
            snapshot[name] = ("set", set(value))
        elif value is None or isinstance(value, (bool, int, float, str, tuple)):
            snapshot[name] = ("value", value)
    return snapshot


def _restore(module: Any, snapshot: Mapping[str, tuple[str, Any]]) -> None:
    """按 :func:`_snapshot` 的副本原地放回（容器对象不换，外部引用继续有效）。"""
    for name, (kind, saved) in snapshot.items():
        current = getattr(module, name, None)
        if kind == "dict" and isinstance(current, dict):
            current.clear()
            current.update(_clone(saved))
        elif kind == "list" and isinstance(current, list):
            current[:] = _clone(saved)
        elif kind == "set" and isinstance(current, set):
            current.clear()
            current.update(saved)
        elif kind == "value":
            setattr(module, name, saved)


def _reset_state() -> None:
    """把两个模块的进程内状态清到"没配置、没加载、没标记"的初始态。

    上一条用例留下的目录索引最危险：``_save`` / ``_PROFILE_READY`` 记的是**绝对
    路径**，不清就会把别人的表当成本用例的表（"没写标记却命中了"就是这么来的）。
    """
    module = _profile_cache
    for name in ("_index", "_pending", "_equiv_pending", "_compare_recent",
                 "_equivalence"):
        container = getattr(module, name, None)
        if isinstance(container, dict):
            container.clear()
        elif isinstance(container, list):
            container[:] = []
    module._configured_dir = ""
    module._ready = False
    module._dir = ""
    module._dir_ok = False
    module._last_error = None
    module._notice_done = False
    module._loaded = False
    module._loaded_rows = 0
    module._skipped_rows = 0
    module._truncated_fields = 0
    module._dropped_rows = 0
    module._disk_rows = -1
    module._rewrite_rows = -1
    _api._PROFILE_READY = None
    _api._PROFILE_NOTICED.clear()
    if hasattr(_api, "_PROFILE_TUNE"):
        _api._PROFILE_TUNE = False
    for name in ("_LAST_CONTRACT", "_LAST_KERNEL", "_FUSED_ENV_OWNED"):
        if hasattr(_api, name):
            setattr(_api, name, None)


@contextlib.contextmanager
def _isolated(profile: bool = True) -> Iterator[str]:
    """临时 CWD + 干净开关 + 干净的进程内状态；退出时全部还原。

    CWD 换到临时树里：画像默认目录是**相对 CWD** 的 ``./.cache``，换过去之后即使某条
    路径漏了显式 ``profile_dir`` 也落不进仓库根。``finally`` 里先还原 CWD，再放回
    环境变量与模块状态，最后清 einsum 的结果缓存。
    """
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    snapshots = [(module, _snapshot(module))
                 for module in (_profile_cache, _api)]
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-equiv-route-") as root:
            work = os.path.join(root, "work")
            os.makedirs(work)
            os.chdir(work)
            try:
                _prepare_env(profile=profile)
                _reset_state()
                pe.cache_clear()
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


# -- 方程、结果与键 --------------------------------------------------------------
def _operands(equation: str, *, seed: int = 20261117) -> list[np.ndarray]:
    """按方程左侧生成标签长度互不相同的 complex128 操作数（C 连续）。"""
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


def _assert_matches(actual: Any, expected: Any, label: str) -> None:
    """逐项断言形状、dtype 与 rel ≤ 1e-12：路由命中/回退都不许改答案。"""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape, (
        label, actual_array.shape, expected_array.shape)
    assert actual_array.dtype == expected_array.dtype, (
        label, actual_array.dtype, expected_array.dtype)
    error = _relative_error(actual_array, expected_array)
    assert error <= 1e-12, (label, error)


def _marker_key(equation: str, operands: Sequence[np.ndarray]) -> dict[str, Any]:
    """按 ``_api._profile_begin`` 的**同一套**口径构造画像键。

    路由表与主表共用 :data:`~pyqcd.einsum._profile_cache.KEY_FIELDS`；键必须逐字段
    一致，否则"写在一处、查在另一处"，二次命中的零匹配断言就永远是因为**没命中**而
    通过的（空断言）。这里直接借用实现自己的三个键构造器，不另写归一化。
    """
    devices = getattr(_api, "_profile_devices", None)
    dtype_of = getattr(_api, "_profile_dtype", None)
    parse = getattr(_api, "parse_backend", None)
    if not callable(devices) or not callable(dtype_of) or not callable(parse):
        raise unittest.SkipTest(
            "等待实现：_api 的画像键构造器缺失（_profile_devices/_profile_dtype/"
            "parse_backend），路由键无法精确构造")
    backend = parse(list(operands), "auto")
    device, user_device = devices(operands, backend)
    return {
        "user_device": user_device,
        "backend": backend,
        "dtype": dtype_of(operands),
        "equation": "".join(equation.split()),
        "shape": _profile_cache.format_shape(
            [operand.shape for operand in operands]),
        "device": device,
    }


def _expected_mapping() -> Any:
    """例方程的等价映射（与实现同一条路径算出来，用作断言基准）。

    不硬编码三个置换字段：映射层将来换一条同桶写法（例如多一个操作数位置置换）时，
    契约仍应成立——它要钉的是"载荷与真实映射一致"，不是"某条具体写法"。
    """
    mapping = _equivalence.normalize_equation(ROUTE_EQUATION, use=False)
    assert mapping is not None, (
        "例方程必须可改写（它是本契约的前提）", ROUTE_EQUATION)
    assert mapping.kind != "identity", ("例方程必须不在语料里", mapping.kind)
    return mapping


def _is_no_op(value: Any) -> bool:
    """载荷里的"不需要动"：``null`` / 空串 / 空列表都算（文档允许这三种编码）。"""
    return value is None or value == [] or value == ()


# -- 路由表读写 ------------------------------------------------------------------
def _fresh_dir(work: str, name: str) -> str:
    """在临时树里新建一个空目录（本用例独占的画像目录）。"""
    directory = os.path.join(work, name)
    os.makedirs(directory, exist_ok=True)
    return directory


def _read_table(path: str) -> tuple[list[str], list[dict[str, str]]]:
    """读一张 CSV：返回 ``(表头, 行)``；文件不存在时抛 ``AssertionError``。"""
    assert os.path.exists(path), ("路由表/画像表没有建出来", path)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return header, rows


def _write_table(path: str, header: Sequence[str],
                 rows: Sequence[Mapping[str, Any]]) -> None:
    """整表重写（追加写的表没有删改接口，坏行只能靠重写构造）。"""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header),
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header})


def _route_rows(rows: Sequence[Mapping[str, str]],
                equation: str) -> list[Mapping[str, str]]:
    """某条方程的全部路由行（``equation`` 按 ``_profile_begin`` 的口径去空白）。"""
    wanted = "".join(equation.split())
    return [row for row in rows
            if "".join((row.get("equation") or "").split()) == wanted]


def _route_row(rows: Sequence[Mapping[str, str]],
               equation: str) -> Mapping[str, str] | None:
    """该方程的路由行：``ok`` 优先，其次最后一条（与查表口径一致）。"""
    candidates = _route_rows(rows, equation)
    for row in candidates:
        if str(row.get("state") or "").strip().lower() == "ok":
            return row
    return candidates[-1] if candidates else None


def _payload(row: Mapping[str, str], name: str) -> Any:
    """把一个载荷字段解析成 Python 值；空串按 ``None``（"这一步不需要动"）。

    非空值**必须是合法 JSON**：CSV 里写 ``(1, 0)`` / ``1;0`` 这类 Python 侧编码虽然
    人看着像置换，读回来却只能靠猜——那正是 v2 要消灭的东西。
    """
    text = str(row.get(name) or "").strip()
    if not text:
        return None
    return json.loads(text)


def _route_row_template(key: Mapping[str, Any], *, state: str,
                        schema_version: str, updated_at: str,
                        kind: str = "output-axes") -> dict[str, str]:
    """手造一条路由行（列序照 :data:`EQUIVALENCE_COLUMNS`，缺列留空）。"""
    row = {name: "" for name in _profile_cache.EQUIVALENCE_COLUMNS}
    row.update({
        "host": socket.gethostname(),
        "user_device": str(key.get("user_device", "")),
        "backend": str(key.get("backend", "")),
        "dtype": str(key.get("dtype", "")),
        "equation": str(key.get("equation", "")),
        "shape": _profile_cache.format_shape(key.get("shape")),
        "device": str(key.get("device", "")),
        "state": state,
        "kind": kind,
        "updated_at": updated_at,
        "schema_version": schema_version,
    })
    return row


@contextlib.contextmanager
def _normalize_spy() -> Iterator[list[Any]]:
    """包住真实 ``normalize_equation``，记录路由命中时它到底有没有被调用。"""
    original = _equivalence.normalize_equation
    calls: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0] if args else kwargs.get("equation"))
        return original(*args, **kwargs)

    with patch.object(_equivalence, "normalize_equation", new=spy):
        yield calls


def _warm(directory: str, equation: str,
          operands: Sequence[np.ndarray]) -> None:
    """跑一次带画像的调用并 flush（把路由行与内层画像都落到盘上）。"""
    with _quiet():
        pe.contract(equation, *operands, profile=True, profile_dir=directory)
        _profile_cache.flush()


# -- 契约 1：首次写入带 v2 载荷 --------------------------------------------------
def test_first_call_writes_a_route_row_with_the_v2_payload() -> None:
    """无文件 → ``contract(..., profile=True, profile_dir=tmp)`` → 行带完整载荷。"""
    with _isolated() as work:
        directory = _fresh_dir(work, "route-first")
        operands = _operands(ROUTE_EQUATION)
        _warm(directory, ROUTE_EQUATION, operands)

        path = os.path.join(directory, ROUTE_NAME)
        header, rows = _read_table(path)
        assert tuple(header) == tuple(_profile_cache.EQUIVALENCE_COLUMNS), (
            "路由表表头必须与 EQUIVALENCE_COLUMNS 逐列一致", header)
        assert int(getattr(_profile_cache, "EQUIVALENCE_SCHEMA_VERSION", 0)) == 2, (
            "schema 版本常量必须是 2",
            getattr(_profile_cache, "EQUIVALENCE_SCHEMA_VERSION", None))

        row = _route_row(rows, ROUTE_EQUATION)
        assert row is not None, (
            "第一次调用必须留下路由行", [dict(item) for item in rows])
        assert str(row.get("state") or "").strip().lower() == "ok", (
            "例方程可改写，state 必须是 ok", dict(row))
        assert str(row.get("schema_version") or "").strip() == "2", (
            "v2 载荷必须带 schema_version=2", dict(row))

        mapping = _expected_mapping()
        target = str(row.get("target_equation") or "").strip()
        assert target, (
            "v2 路由载荷缺失：state=ok 的行没有 target_equation（等价路由表 v2 "
            "尚未写入载荷）", dict(row))
        assert target == mapping.equation, (
            "target_equation 必须是可以直接照着执行的语料写法",
            target, mapping.equation)
        assert target != "".join(ROUTE_EQUATION.split()), (target, ROUTE_EQUATION)
        assert _specialize.match(target) is not None, (
            "target_equation 必须是特化支持的那条 equation", target)

        for name in PAYLOAD_FIELDS:
            text = str(row.get(name) or "").strip()
            if name == "output_permutation":
                assert text, (
                    "本方程的输出轴序要动，output_permutation 不能为空", dict(row))
            if text:
                _payload(row, name)              # 非 JSON 会在这里抛：正是要抓的缺陷

        assert _is_no_op(_payload(row, "operand_order")), (
            "本方程不动操作数位置", row.get("operand_order"))
        assert _is_no_op(_payload(row, "operand_permutations")), (
            "本方程不动操作数内轴序", row.get("operand_permutations"))
        assert _payload(row, "output_permutation") == list(
            mapping.output_permutation or ()), (
            "output_permutation 必须与真实映射逐位一致（方向错了形状就会变）",
            row.get("output_permutation"), mapping.output_permutation)


# -- 契约 2：二次命中零匹配 ------------------------------------------------------
def test_second_call_hits_the_route_without_re_matching() -> None:
    """载荷**第二次**命中时零匹配；首次施放只允许那一次数值自检。

    为什么不是"再调一次就零匹配"：表里的载荷是别的进程写下的，第一次照它跑出来的
    结果必须与"现在重新匹配一次再施加"对拍（``_api._payload_result``），这是"相信
    这张表"的唯一代价，且同一份载荷只付一次。因此本用例把两个时刻分开断言：

    * **首次施放**：``<= 1`` 次匹配（自检），多一次就说明每次都在重查；
    * **第二次命中**（稳态）：**0** 次，且两遍结果都与 ``np.einsum`` 一致。
    """
    with _isolated() as work:
        directory = _fresh_dir(work, "route-second")
        operands = _operands(ROUTE_EQUATION)
        expected = np.einsum(ROUTE_EQUATION, *operands)
        _warm(directory, ROUTE_EQUATION, operands)

        key = _marker_key(ROUTE_EQUATION, operands)
        marker = _profile_cache.equivalence_marker(key)
        assert isinstance(marker, dict) and (
            str(marker.get("state") or "").strip().lower() == "ok"), (
            "正对照：第一次必须留下 ok 标记，否则下面的零匹配无从谈起", marker)

        with _normalize_spy() as first_calls, _quiet():
            first = pe.contract(ROUTE_EQUATION, *operands,
                                profile=True, profile_dir=directory)
            _profile_cache.flush()
        _assert_matches(first, expected, "first payload application")
        assert len(first_calls) <= 1, (
            "首次按载荷施放最多只做一次数值自检，不该反复重查", first_calls)

        with _normalize_spy() as calls, _quiet():
            result = pe.contract(ROUTE_EQUATION, *operands,
                                 profile=True, profile_dir=directory)
            _profile_cache.flush()
        assert not calls, (
            "路由表稳态命中时不该再调 _equivalence.normalize_equation"
            "（v2 的意义就是省掉这次匹配）", calls)
        _assert_matches(result, expected, "route hit")


# -- 契约 3：方案 A 副产物（内层画像） -------------------------------------------
def test_applying_the_route_leaves_the_target_equation_in_the_profile() -> None:
    """施加路由时内层递归按 ``target_equation`` 画像 → ``profile.csv`` 有那一行。

    副产物只在**载荷**路径上出现（``_apply_equivalence(..., profile=True)``）：第一次
    调用是"查完再施加"（原本就 profile=False，不该凭空多写一行），所以这里先 warm 出
    路由行，再按载荷施放一次，然后看画像表。这也是"这条路由真的被执行过"的旁证。
    """
    with _isolated() as work:
        directory = _fresh_dir(work, "route-plan-a")
        operands = _operands(ROUTE_EQUATION)
        _warm(directory, ROUTE_EQUATION, operands)

        _, route_rows = _read_table(os.path.join(directory, ROUTE_NAME))
        row = _route_row(route_rows, ROUTE_EQUATION)
        target = "" if row is None else str(row.get("target_equation") or "").strip()
        assert row is not None and target, (
            "方案 A 的前提：路由行必须带 target_equation", dict(row or {}))

        _warm(directory, ROUTE_EQUATION, operands)      # 第二次：按载荷施放

        _, profile_rows = _read_table(os.path.join(directory, PROFILE_NAME))
        equations = {"".join((item.get("equation") or "").split())
                     for item in profile_rows}
        assert target in equations, (
            "方案 A 副产物缺失：profile.csv 里没有以 target_equation 为 equation 的行"
            "（内层递归必须 profile=True）", sorted(equations), target)


# -- 契约 4：版本优先 ------------------------------------------------------------
def test_version_wins_over_state_when_both_rows_exist() -> None:
    """同键两行：v1 ``none`` 更新 vs v2 ``ok`` 更旧 → 必须返回 v2。

    再配一条**判别式对照**：v1 ``ok`` 更新 vs v2 ``none`` 更旧。若排序里根本没有
    ``schema_version``（只剩 ``ok`` 先于 ``none``），第一条会偶然通过、这条必然失败
    ——"先版本、后状态"才被真的钉住。
    """
    with _isolated() as work:
        directory = _fresh_dir(work, "route-version")
        _profile_cache.configure(directory)
        operands = _operands(PLAIN_EQUATION)
        key = _marker_key(PLAIN_EQUATION, operands)
        path = os.path.join(directory, ROUTE_NAME)
        header = list(_profile_cache.EQUIVALENCE_COLUMNS)
        newer, older = "2026-11-17T23:00:00+08:00", "2026-11-01T00:00:00+08:00"

        _write_table(path, header, [
            _route_row_template(key, state="none", schema_version="1",
                                updated_at=newer),
            _route_row_template(key, state="ok", schema_version="2",
                                updated_at=older),
        ])
        _profile_cache.load(force=True)
        marker = _profile_cache.equivalence_marker(key)
        assert isinstance(marker, dict), ("手工造的两行必须能读进索引", key)
        assert str(marker.get("state") or "").strip().lower() == "ok", (
            "v2 的 ok 行必须压过 v1 的 none 行", dict(marker))
        assert str(marker.get("schema_version") or "").strip() == "2", (
            "返回的必须是 v2 那一行", dict(marker))

        _write_table(path, header, [
            _route_row_template(key, state="ok", schema_version="1",
                                updated_at=newer),
            _route_row_template(key, state="none", schema_version="2",
                                updated_at=older),
        ])
        _profile_cache.load(force=True)
        marker = _profile_cache.equivalence_marker(key)
        assert isinstance(marker, dict), ("判别式对照的两行必须能读进索引", key)
        assert str(marker.get("schema_version") or "").strip() == "2", (
            "版本优先：v2 必须压过更新的 v1（排序键漏了 schema_version）",
            dict(marker))
        assert str(marker.get("state") or "").strip().lower() == "none", (
            "v2 说 none 就返回 none——这条对照专门区分「版本优先」与「状态优先」",
            dict(marker))


# -- 契约 5：坏 JSON 不抛、退回重查 ----------------------------------------------
def test_broken_payload_falls_back_to_matching_without_raising() -> None:
    """载荷字段不是合法 JSON → 当作未命中，重查一次，结果照旧正确。"""
    with _isolated() as work:
        directory = _fresh_dir(work, "route-broken")
        operands = _operands(ROUTE_EQUATION)
        expected = np.einsum(ROUTE_EQUATION, *operands)
        _warm(directory, ROUTE_EQUATION, operands)

        path = os.path.join(directory, ROUTE_NAME)
        header, rows = _read_table(path)
        row = _route_row(rows, ROUTE_EQUATION)
        assert row is not None and (
            str(row.get("state") or "").strip().lower() == "ok"), (
            "正对照：先得有一条 ok 路由行，才谈得上把它弄坏", dict(row or {}))
        for name in PAYLOAD_FIELDS:
            row[name] = "{not-json"
        _write_table(path, header, rows)
        _profile_cache.load(force=True)

        marker = _profile_cache.equivalence_marker(
            _marker_key(ROUTE_EQUATION, operands))
        assert isinstance(marker, dict) and (
            str(marker.get("schema_version") or "").strip() == "2"), (
            "正对照：坏行仍然进索引（它坏的是载荷，不是 schema）", marker)

        with _normalize_spy() as calls, _quiet():
            result = pe.contract(ROUTE_EQUATION, *operands,
                                 profile=True, profile_dir=directory)
        _assert_matches(result, expected, "broken payload")
        assert calls, (
            "载荷解析失败必须退回重新匹配；一次都没查等于把可用映射丢了", calls)


# -- 契约 6：state=none 仍是廉价门 -----------------------------------------------
def test_none_marker_still_skips_the_equivalence_step() -> None:
    """同键 ``none`` → 第二次调用 spy = 0（v2 不许挤掉这条门）。"""
    with _isolated() as work:
        directory = _fresh_dir(work, "route-none")
        operands = _operands(PLAIN_EQUATION)
        expected = np.einsum(PLAIN_EQUATION, *operands)
        _profile_cache.configure(directory)
        key = _marker_key(PLAIN_EQUATION, operands)
        assert _profile_cache.record_equivalence(key, "none"), (
            "none 标记必须真的排队写盘（正对照）")
        _profile_cache.flush()
        marker = _profile_cache.equivalence_marker(key)
        assert isinstance(marker, dict) and (
            str(marker.get("state") or "").strip().lower() == "none"), (
            "标记写进表后必须能按同键查回来", marker)

        with _normalize_spy() as calls, _quiet():
            result = pe.contract(PLAIN_EQUATION, *operands,
                                 profile=True, profile_dir=directory)
        assert not calls, ("state=none 时不该再查等价映射", calls)
        _assert_matches(result, expected, "none gate")


# -- 契约 7：force 重查并刷新 ----------------------------------------------------
def test_force_ignores_the_marker_and_refreshes_the_row() -> None:
    """``as_equivalence="force"`` → spy ≥ 1，且把该键刷新成 ``ok`` 行。"""
    with _isolated() as work:
        directory = _fresh_dir(work, "route-force")
        operands = _operands(ROUTE_EQUATION)
        expected = np.einsum(ROUTE_EQUATION, *operands)
        _profile_cache.configure(directory)
        key = _marker_key(ROUTE_EQUATION, operands)
        assert _profile_cache.record_equivalence(key, "none"), (
            "正对照：先写一条 none 标记", key)
        _profile_cache.flush()
        marker = _profile_cache.equivalence_marker(key)
        assert isinstance(marker, dict) and (
            str(marker.get("state") or "").strip().lower() == "none"), (
            "标记键必须与 contract 的画像键一致", marker)

        with _normalize_spy() as calls, _quiet():
            result = pe.contract(ROUTE_EQUATION, *operands,
                                 profile=True, profile_dir=directory,
                                 as_equivalence="force")
            _profile_cache.flush()
        assert calls, ("force 必须无视标记与路由、真的重查一次", calls)
        _assert_matches(result, expected, "force")

        _, rows = _read_table(os.path.join(directory, ROUTE_NAME))
        ok_rows = [item for item in _route_rows(rows, ROUTE_EQUATION)
                   if str(item.get("state") or "").strip().lower() == "ok"]
        assert ok_rows, (
            "force 之后必须刷新出 ok 行（写入是追加式的，旧 none 行会保留）",
            [dict(item) for item in rows])
        assert any(str(item.get("target_equation") or "").strip()
                   for item in ok_rows), (
            "刷新出来的 ok 行同样要带 v2 载荷", [dict(item) for item in ok_rows])


# -- 契约 8：profile=False 零 IO -------------------------------------------------
def test_profile_false_creates_no_table() -> None:
    """``profile=False`` 一个文件都不建；配一条 ``profile=True`` 的正对照。"""
    with _isolated() as work:
        directory = _fresh_dir(work, "route-off")
        operands = _operands(ROUTE_EQUATION)
        expected = np.einsum(ROUTE_EQUATION, *operands)

        with _quiet():
            result = pe.contract(ROUTE_EQUATION, *operands,
                                 profile=False, profile_dir=directory)
        _assert_matches(result, expected, "profile off")
        assert os.listdir(directory) == [], (
            "profile=False 不许建任何文件（连探针都不行）", os.listdir(directory))

        _warm(directory, ROUTE_EQUATION, operands)
        assert os.path.exists(os.path.join(directory, ROUTE_NAME)), (
            "正对照：同一方程开着画像时必须留下路由行，否则「没写」是空断言",
            os.listdir(directory))


TESTS = (
    test_first_call_writes_a_route_row_with_the_v2_payload,
    test_second_call_hits_the_route_without_re_matching,
    test_applying_the_route_leaves_the_target_equation_in_the_profile,
    test_version_wins_over_state_when_both_rows_exist,
    test_broken_payload_falls_back_to_matching_without_raising,
    test_none_marker_still_skips_the_equivalence_step,
    test_force_ignores_the_marker_and_refreshes_the_row,
    test_profile_false_creates_no_table,
)


def test_einsum_equivalence_route_contracts() -> None:
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
            f"{len(failures)} 条等价路由表 v2 契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"equivalence route contracts did not run ({skipped} skipped)")


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
