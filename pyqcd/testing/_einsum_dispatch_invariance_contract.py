"""契约：换路不换答案（R4）——分派路径换了，数值与字段不许跟着换。

配置画像（``_profile_cache``）会把下一次调用**换到另一条路上**：命中一条
``kernel=backend`` 的记录就直接调后端自己的 ``einsum``，命中 ``*_dedicated`` /
``*_topology`` 就强制专用内核。这几条路本来就该给出同一个答案，所以**普通数值测试
永远抓不到画像层的缺陷**——某条路被换坏时，错答会被安安静静地交出来，只有跟另一条
路对拍才看得见。本模块把下面五条固化成能独立失败的断言：

1. **三路对拍**：同一方程 / 同一 dtype / 同一批操作数，``kernel_force`` 取
   ``generic``（基线）、``dedicated``、``backend`` 三路，``shape`` 与 ``dtype``
   必须完全相同，相对偏差 ``max|a-b| / max|ref|`` ≤ :data:`RTOL`。``dedicated``
   在本机不可用（原生库缺失、被 ``PYQCD_EINSUM_KERNEL_KUPL=0`` 关掉、或 dtype 被
   策略拒绝）时抛 ``SkipTest`` 并写明原因与已经对上的那两路，**不伪造通过**。
2. **画像命中不改答案**：临时目录写一条 ``kernel=backend`` 的记录（``wall_ms``
   极小）→ ``contract(..., profile=True, profile_dir=tmp)`` 与
   ``kernel_force="backend"`` 直接调用**逐元素全等**，且
   ``last_kernel()["ran"] == "backend"``：既证明答案没变，也证明路真的换了。
3. **画像命中不改 dtype/形状**：同一次命中调用的 ``dtype`` / ``shape`` 与基线一致。
4. **关闭路径无副作用**：``profile=False`` 时记录里那条 ``kernel=backend`` 不被施加
   —— 结果仍与通用基线一致，且 ``last_kernel()["ran"]`` 绝不是 ``backend``。
5. **降级安全**：记录里写一个本机**不可用**的 kernel（numpy 上写 ``cuda_topology``，
   并把这个后端的原生入口关掉）→ 调用**不抛异常**、结果与基线一致（回退到自动
   分派）。同一条强制路径直连必须真抛 ``KernelUnavailableError``，否则本用例没有
   牙齿，改为 ``SkipTest`` 说明本机构造不出这个场景。

容差口径（为什么是 1e-12）
--------------------------
三条路算的是同一个收缩，差别只在求和次序 / 是否走 BLAS / 是否分块，误差来源单一：
浮点结合律的舍入。``complex128`` 的机器 epsilon 是 2.2e-16，本条方程 6 个操作数、
16 个指标槽、68 个输入元素（输出是标量，``max|ref|≈211``），最坏累积量级仍在几个
epsilon 内：本机实测两两偏差 3.0e-16 / 3.4e-16 / 4.1e-16。取 ``1e-12``≈2.5e3 个
epsilon：比任何真实舍入大三四个数量级、又比任何真实错答（≥1e-6，通常是符号或索引
错位）小六个量级。归一化用 ``max|ref|`` 而不是逐元素绝对阈值——接近零的元素会被
后者的绝对阈值误判（与本库唯一单方程入口 ``python -m pyqcd.einsum.test --atol``
同一口径）。

第 2 条用 ``array_equal``（逐元素全等）而不是相对容差：``profile`` 命中的 backend
与直接强制 backend 是**同一次** ``backend.einsum`` 调用，任何可见差异都说明分派中
间被插了别的东西。相对容差只用于"跨不同执行路径"的比较。

不做任何计时断言：本模块只关心"换了路之后答案与字段还是不是同一个"，不关心多快。

可直接运行::

    python -B -m pyqcd.testing._einsum_dispatch_invariance_contract
"""
from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import unittest
from typing import Any

import numpy as np

from ..einsum import (_api, _kernel_force, _kernel_kupl, _profile_cache,
                      _specialize, _threads, contract)

#: 语料内方程（seq=1，6 个操作数、16 个指标槽、标量输出）。用它而不是等价写法：
#: 后者要靠 ``as_equivalence`` 改写才命中，会把"本层被对拍的是哪条方程"搅浑。
EQUATION = "cadb,menf,ac,em,bd,fn->"

#: 各操作数轴长：``c=2,a=2,d=2,b=3 / m=2,e=2,n=2,f=3 / ac / em / bd / fn``。
#: 全部小规模：让步表可编码、让每条用例都跑得完，且对拍只用同一批张量。
SHAPES = ((2, 2, 2, 3), (2, 2, 2, 3), (2, 2), (2, 2), (3, 2), (3, 2))

#: 主精度：三路对拍只钉它（实数档的跨路径界不同，另有专门的 dtype 策略契约）。
DTYPE = np.complex128

#: 跨路径相对容差（按 ``max|ref|`` 归一）。理由见模块文档。
RTOL = 1e-12

#: 隔离上下文要复原的环境变量：这些都参与"画像是施加还是只记录"的判定。
_ENV_NAMES = (
    _kernel_force.KERNEL_FORCE_ENV,      # 调用方已强制 → 画像命中只记录不施加
    _specialize.DISABLE_ENV,             # specialize=False → 同理
    _specialize.FUSED_DISABLE_ENV,       # _api 会自己置位/清位，必须连值一起复原
    _specialize.NOTICE_ENV,              # 关掉提示行，输出只留断言结论
    _profile_cache.PROFILE_ENV,          # 总开关（缺省开）
    _profile_cache.DIR_ENV,              # 画像目录（指到临时树）
    _kernel_kupl._DISABLE_ENV,           # 原生入口（第 5 条要能确定性地关掉它）
)

#: ``_profile_cache`` 的进程级状态字段：用例之间不串味的关键。
#: TV 重设计后配置表只保留"钉住的目录 / 解析缓存 / 遍历计数"三样状态，
#: 旧的索引、待写队列与计数器随 append/merge/flush 写路径一并删除。
_CACHE_FIELDS = (
    "_CONFIGURED_DIR", "_LAST_ERROR", "_HOST", "_TABLE", "_TUNING",
)

#: ``_api`` 的集合型进程状态：去重告警闸门，留着就等于让下一条用例"已经警告过"。
_API_SET_FIELDS = (
    "_NATIVE_DISABLED", "_NATIVE_WARNED_REASONS", "_SPECIALIZED_KERNEL_WARNED",
    "_KERNEL_SKIP_WARNED", "_PROFILE_NOTICED",
)

#: ``_threads`` 写进 ``os.environ`` 的两个标准线程变量：``reset()`` 只复位
#: applied/omp/blas/source，``exported_env`` / ``runtime_applied`` /
#: ``runtime_attempts`` / ``profile`` 与这两个变量都会一直留着。
_THREAD_ENV_NAMES = (_threads.STANDARD_OMP_ENV, _threads.STANDARD_BLAS_ENV)


# ── 数据与会话 ──────────────────────────────────────────────────────────────

def _operands(seed: int = 20261117):
    """按方程各操作数的 rank 造一组 C 连续 ``complex128`` 操作数（同一批给三路用）。"""
    rng = np.random.default_rng(seed)
    values = []
    for shape in SHAPES:
        data = rng.standard_normal(shape)
        data = data + 1j * rng.standard_normal(shape)
        values.append(np.ascontiguousarray(data.astype(DTYPE)))
    return values


def _signature(result: Any) -> tuple[tuple[int, ...], str]:
    """``(shape, dtype)``：换路之后这两样一个字都不许变。"""
    array = np.asarray(result)
    return tuple(int(n) for n in array.shape), str(array.dtype)


def _rel_max(actual: Any, expected: Any) -> float:
    """相对最大偏差：``max|a-b| / max|ref|``（与单方程入口的相对容差同口径）。"""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0, (
        "参考结果全零（或幅度为零），相对误差没有意义——本用例的操作数不该造出这种结果")
    return float(np.max(np.abs(actual - expected))) / scale


def _reason_text(exc: BaseException) -> str:
    """``KernelUnavailableError`` 逐标签的原因拼成一行，skip 时直接贴给人看。"""
    reasons = getattr(exc, "reasons", None) or {}
    text = "; ".join(f"{label}: {reason}" for label, reason in reasons.items())
    return text or str(exc)


def _snapshot_cache() -> dict[str, Any]:
    """深拷一份配置表的进程状态（钉住的目录 / 解析缓存 / 遍历计数）。"""
    return {name: getattr(_profile_cache, name) for name in _CACHE_FIELDS}


def _restore_cache(state: dict[str, Any]) -> None:
    """按 :func:`_snapshot_cache` 的副本原样放回。"""
    for name, value in state.items():
        setattr(_profile_cache, name, value)


def _reset_cache() -> None:
    """清到"没钉目录、没有解析缓存、不在遍历中"的初始态。

    ``_TABLE`` 必须清：它是按 ``(路径, mtime, size)`` 失效的解析缓存，留着上一条用例
    的表就等于把别人的行当成本用例的结果。
    """
    _profile_cache._CONFIGURED_DIR = ""
    _profile_cache._LAST_ERROR = None
    _profile_cache._HOST = None
    _profile_cache._TABLE = None
    _profile_cache._TUNING = 0


@contextlib.contextmanager
def _isolated():
    """临时树 + 干净环境 + 干净画像状态；退出时原样复原并删掉临时树。

    四条隔离各自都必要：

    * **临时树**：记录写在这里，``profile_dir`` 也指这里，仓库根不会多出 ``.cache``；
    * **``chdir`` 进临时树**：画像目录默认相对 CWD，这是"绝不落到仓库根"的兜底；
    * **复原 ``_api`` 的进程级状态**：``_PROFILE_MISSING_NOTICED`` 是"缺表只提示
      一次"的闸门；不复位就等于让本模块的"缺表要提示"读到别人的结论。四个去重告警
      集合同理：别人打过的告警会让本模块的"零告警"断言变成空断言；
    * **复原 ``_threads`` 的进程状态**：``contract()`` 会按形态切线程档位并写
      ``os.environ``，``reset()`` 只复位 applied/omp/blas/source，其余字段和两个标准
      线程变量会一直留着。
    """
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    saved_cache = _snapshot_cache()
    saved_api = {name: set(getattr(_api, name)) for name in _API_SET_FIELDS}
    saved_api.update({
        "missing_noticed": getattr(_api, "_PROFILE_MISSING_NOTICED", False),
        "last": _api._LAST_KERNEL,
        "fused": _api._FUSED_ENV_OWNED,
    })
    saved_threads = {
        "state": dict(_threads._state),
        "env": {name: os.environ.get(name) for name in _THREAD_ENV_NAMES},
    }
    root = tempfile.mkdtemp(prefix="pyqcd-dispatch-")
    work = os.path.join(root, "work")
    directory = os.path.join(root, "profile")
    try:
        os.makedirs(work)
        os.chdir(work)
        for name in _ENV_NAMES:
            os.environ.pop(name, None)
        os.environ[_profile_cache.DIR_ENV] = directory
        os.environ[_specialize.NOTICE_ENV] = "0"
        _reset_cache()
        _api._PROFILE_MISSING_NOTICED = False   # "这次还没提示过缺表"，不是 False
        yield directory
    finally:
        os.chdir(saved_cwd)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _restore_cache(saved_cache)
        _api._PROFILE_MISSING_NOTICED = saved_api["missing_noticed"]
        _api._LAST_KERNEL = saved_api["last"]
        _api._FUSED_ENV_OWNED = saved_api["fused"]
        for name in _API_SET_FIELDS:
            getattr(_api, name).clear()
            getattr(_api, name).update(saved_api[name])
        _threads._state.clear()
        _threads._state.update(saved_threads["state"])
        for name, value in saved_threads["env"].items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(root, ignore_errors=True)


def _key(operands) -> dict[str, Any]:
    """``_profile_begin`` 查表用的键（numpy / CPU / 单一 dtype 的固定写法）。

    与实现同源：``KEY_FIELDS`` 由被测模块给出，``dtype`` 按操作数次序去重后用 ``+``
    连接、``shape`` 走 :func:`_profile_cache.format_shape`。键对不上，下面的用例会静默
    变成空断言，所以 :func:`_seed` 必须再用 :func:`_profile_cache.read_winner` 反查一次。
    """
    names: list[str] = []
    for operand in operands:
        text = str(operand.dtype)
        if text not in names:
            names.append(text)
    return {
        "shape": _profile_cache.format_shape([o.shape for o in operands]),
        "dtype": "+".join(names),
        "device": "cpu",
        "equation": "".join(EQUATION.split()),
        "backend": "numpy",
    }


def _seed(kernel: str, wall_ms: float, operands) -> dict[str, Any]:
    """往当前（已隔离的）配置表写**唯一**一行，并确认它能按同一个键查回。"""
    key = _key(operands)
    _profile_cache.write_winner(
        key, {"kernel": kernel, "optimize": "auto", "wall_ms": wall_ms})
    picked = _profile_cache.read_winner(key)
    assert picked is not None, ("记录必须能被同一个键查回（键算错了）", key)
    assert picked["kernel"] == kernel, (kernel, picked)
    return key


# ── 1. 三路对拍 ─────────────────────────────────────────────────────────────

def test_three_paths_agree_on_shape_dtype_and_values():
    """``generic`` / ``dedicated`` / ``backend``：shape、dtype 全同，rel ≤ 1e-12。"""
    operands = _operands()
    with _isolated():
        baseline = contract(EQUATION, *operands, kernel_force="generic", profile=False)
        backend = contract(EQUATION, *operands, kernel_force="backend", profile=False)

        # 基线自证：标量输出、complex128（后面两路都拿它当参照）。
        assert _signature(baseline) == ((), "complex128"), _signature(baseline)
        # 两路先钉死：即使 dedicated 在本机不可用，这一对也永远对得上。
        assert _signature(backend) == _signature(baseline), (
            _signature(backend), _signature(baseline))
        assert _rel_max(backend, baseline) <= RTOL, (
            "backend 路与通用路的偏差超出 " f"{RTOL:g}", _rel_max(backend, baseline))

        try:
            dedicated = contract(EQUATION, *operands, kernel_force="dedicated",
                                 profile=False)
        except _kernel_force.KernelUnavailableError as exc:
            raise unittest.SkipTest(
                "dedicated 在本机不可用，三路对拍不可完成：" + _reason_text(exc)
                + f"；generic↔backend 两路已逐项对上（shape/dtype 相同、rel≤{RTOL:g}）")

        assert _signature(dedicated) == _signature(baseline), (
            _signature(dedicated), _signature(baseline))
        assert _rel_max(dedicated, baseline) <= RTOL, (
            "dedicated 路与通用路的偏差超出 " f"{RTOL:g}", _rel_max(dedicated, baseline))


# ── 2. 画像命中不改答案 ─────────────────────────────────────────────────────

def test_profile_hit_switches_to_backend_without_changing_the_answer():
    """命中 ``kernel=backend``：逐元素全等，且 ``last_kernel()['ran']`` 真是 backend。"""
    operands = _operands()
    with _isolated() as directory:
        direct = contract(EQUATION, *operands, kernel_force="backend", profile=False)
        _seed("backend", wall_ms=0.001, operands=operands)

        hit = contract(EQUATION, *operands, profile=True, profile_dir=directory)

        record = _api.last_kernel()
        assert record is not None, "画像命中也要留下分派记录（否则本用例没有观测点）"
        assert record["ran"] == _kernel_force.BACKEND_LABEL, (
            "命中 backend 记录必须真的改走 backend.einsum，否则'换路'没发生", record)
        assert np.array_equal(np.asarray(hit), np.asarray(direct)), (
            "换路不许换答案：与直接强制 backend 必须逐元素相同",
            float(np.max(np.abs(np.asarray(hit) - np.asarray(direct)))))


# ── 3. 画像命中不改 dtype/形状 ──────────────────────────────────────────────

def test_profile_hit_keeps_dtype_and_shape():
    """同一次命中调用：``dtype`` 与 ``shape`` 都回到基线，不许被后端选择带偏。"""
    operands = _operands()
    with _isolated() as directory:
        baseline = contract(EQUATION, *operands, kernel_force="generic", profile=False)
        _seed("backend", wall_ms=0.001, operands=operands)

        hit = contract(EQUATION, *operands, profile=True, profile_dir=directory)

        # 先证明这次调用真的换了路，否则"字段没变"可能只是"什么都没发生"。
        record = _api.last_kernel()
        assert record is not None and record["ran"] == _kernel_force.BACKEND_LABEL, record
        assert _signature(hit) == _signature(baseline), (
            _signature(hit), _signature(baseline))
        assert _signature(hit) == ((), "complex128"), _signature(hit)


# ── 4. 关闭路径无副作用 ─────────────────────────────────────────────────────

def test_profile_disabled_leaves_the_dispatch_untouched():
    """``profile=False``：记录仍在表里但不被施加，结果与通用基线一致。"""
    operands = _operands()
    with _isolated():
        baseline = contract(EQUATION, *operands, kernel_force="generic", profile=False)
        key = _seed("backend", wall_ms=0.001, operands=operands)

        off = contract(EQUATION, *operands, profile=False)

        record = _api.last_kernel()
        assert record is not None, "关闭画像的调用也要有分派记录"
        assert record["ran"] != _kernel_force.BACKEND_LABEL, (
            "profile=False 时记录里的 backend 绝不能被施加", record)
        # 记录是"画像的建议"，不是"调用方的要求"：这次没传 kernel_force，缺省即
        # ``generic``，所以 requested 是**解析后的空尝试列表** ``()``（``auto`` 已删除，
        # 没有 ``None`` 这种"未指定"的哨兵）。
        assert record["requested"] == (), record
        assert _signature(off) == _signature(baseline), (
            _signature(off), _signature(baseline))
        assert _rel_max(off, baseline) <= RTOL, (
            "profile=False 的结果必须与通用基线一致", _rel_max(off, baseline))
        # 记录确实一直在（否则上面几条就是空断言）。
        assert _profile_cache.read_winner(key) is not None, "签名用的记录不见了"
        # 关闭就是关闭：连默认的 ./.cache 都不许造出来。
        assert not os.path.exists(os.path.join(os.getcwd(), _profile_cache.DEFAULT_DIR)), (
            "profile=False 不许在 CWD 下建缓存目录", os.listdir(os.getcwd()))


# ── 5. 降级安全 ─────────────────────────────────────────────────────────────

def test_unavailable_profile_kernel_degrades_without_raising():
    """记录里的内核本机不可用：不抛异常，回退自动分派，答案与基线一致。"""
    operands = _operands()
    with _isolated() as directory:
        baseline = contract(EQUATION, *operands, kernel_force="generic", profile=False)
        # ``cuda_topology`` 在 numpy 操作数上按角色解析成逻辑名 ``topology``；把本后端
        # 的原生入口关掉，它就成了确定不可用的建议。
        _seed("cuda_topology", wall_ms=0.0001, operands=operands)

        os.environ[_kernel_kupl._DISABLE_ENV] = "0"
        try:
            try:
                hit = contract(EQUATION, *operands, profile=True, profile_dir=directory)
            except _kernel_force.KernelUnavailableError as exc:
                raise AssertionError(
                    "画像建议不可用时必须回退自动分派，不许把异常抛给调用方："
                    + _reason_text(exc))

            # "牙齿"：同一条路径直连必须真抛，否则本机没构造出不可用场景，
            # 上面的"没抛"可能只是"根本没试"。
            try:
                contract(EQUATION, *operands, kernel_force="topology", profile=False)
            except _kernel_force.KernelUnavailableError:
                pass
            else:
                raise unittest.SkipTest(
                    f"{_kernel_kupl._DISABLE_ENV}=0 时 topology 仍跑成了；"
                    "本机构造不出「记录里的内核不可用」这一场景，降级路径不可验")
        finally:
            os.environ.pop(_kernel_kupl._DISABLE_ENV, None)

        assert _signature(hit) == _signature(baseline), (
            _signature(hit), _signature(baseline))
        assert _rel_max(hit, baseline) <= RTOL, (
            "回退后的结果必须与基线一致", _rel_max(hit, baseline))


TESTS = (
    test_three_paths_agree_on_shape_dtype_and_values,
    test_profile_hit_switches_to_backend_without_changing_the_answer,
    test_profile_hit_keeps_dtype_and_shape,
    test_profile_disabled_leaves_the_dispatch_untouched,
    test_unavailable_profile_kernel_degrades_without_raising,
)


def test_einsum_dispatch_invariance_contracts():
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
            f"{len(failures)} 条换路不换答案契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"dispatch invariance contracts did not run ({skipped} skipped)")


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
