"""契约：per-拓扑原生执行入口（``try_topology_apply``）的 dtype、参数、内存与回落。

四件事必须同时成立，缺一不可：

1. **dtype**：入口只下发**契约内**的 dtype（``TOPOLOGY_DTYPES`` 的四类：
   ``complex128`` / ``complex64`` / ``float64`` / ``float32``）。四类**完全平权**
   ——没有主精度、没有默认精度；每一类都必须由原生 2×2 实测（``_probe_dtype``）
   自证正确后才启用，否则返回 ``None`` 回落。任何情况下都**不得**把一类 dtype
   转成另一类精度，也不得把异常抛给调用方。"探测没过"与"契约外 dtype"必须是两条
   可区分、可行动的原因。
2. **参数与内存**：交给原生的 plan 字节流、操作数**顺序**、``operand_offsets``、
   ``out`` 缓冲区必须与 numpy/cupy 数组逐项一致；非连续输入要么复制成连续
   （下标记进 ``LAST_INFO['copied']``）要么回落，非连续 ``out`` 一律回落。
3. **数值**：语料抽检 ≥100 条（``dim=4``）与 ``contract(..., specialize=False)``
   一致（按 dtype 容差），且回落路径本身也必须给出正确结果。
4. **失败回落**：原生返回错误码时结果仍正确，并且只发**一条**警告。

可直接运行::

    python -m pyqcd.testing._topology_kernel_contract
"""
from __future__ import annotations

import ctypes
import unittest
import warnings
from typing import Any, Callable, Sequence
from unittest.mock import patch

import numpy as np

from ..einsum import _kernel_cuda, _kernel_kupl
from ..einsum import contract
from ..einsum._native_plan import dtype_id as native_dtype_id
from ..einsum.kupl import einsum as kupl_einsum
from ..einsum import _wick_tasklist as tasklist
from ._topology_equations_contract import _parse_equations

# 原生支持的四类 dtype（id 0..3），一个都不能漏。
ALL_DTYPES = (np.complex128, np.complex64, np.float64, np.float32)
TOLERANCES = {
    "complex64": 1e-5,
    "float32": 1e-5,
    "complex128": 1e-12,
    "float64": 1e-12,
}
SAMPLE_SIZE = 120          # 语料抽检下限是 100，留 20 条余量
DIM = 4                    # per-拓扑 kernel 的固定维数
# 拓扑入口只服务 wick 语料的编译期步表（语料外一律 status=4，不兜底）。因此凡是
# 想验"真的下发成功"的用例，方程必须来自语料；``ij,jk->ik`` 这类语料外方程只能
# 用来验"应当回落"。这两个常量就是"语料内最小元数"的那一条。
PROBE_EQUATION = "cadb,menf,ac,em,bd,fn->"
PROBE_TERMS = PROBE_EQUATION.split("->")[0].split(",")


def probe_operands(dim: int, dtype: Any, seed: int = 0) -> list[np.ndarray]:
    """``PROBE_EQUATION`` 的操作数（所有轴长 = dim，满足等长超立方体前提）。"""
    rng = np.random.default_rng(seed)
    values = []
    for term in PROBE_TERMS:
        shape = (dim,) * len(term)
        data = rng.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            data = data + 1j * rng.standard_normal(shape)
        values.append(np.ascontiguousarray(data.astype(dtype)))
    return values
_PROBE_KEYS = (
    "dtype", "dtype_id", "n_operands", "plan_bytes",
    "steps", "output_shape", "out", "copied", "offsets", "path",
)


# ── 工具 ────────────────────────────────────────────────────────────────────

def _cupy_ready() -> bool:
    """GPU 路径是否可测（缺 cupy 或没有设备时返回 False）。"""
    try:
        import cupy
    except ImportError:
        return False
    try:
        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:                                    # noqa: BLE001 - 驱动缺失
        return False


def _make_operands(shapes: Sequence[tuple[int, ...]], dtype: Any,
                   seed: int = 0) -> list[np.ndarray]:
    """确定性随机操作数（复数 dtype 才有虚部）。"""
    rng = np.random.default_rng(seed)
    kind = np.dtype(dtype).kind
    values = []
    for shape in shapes:
        real = rng.standard_normal(shape)
        data = real + 1j * rng.standard_normal(shape) if kind == "c" else real
        values.append(np.ascontiguousarray(data.astype(dtype)))
    return values


def _quiet(call: Callable[[], Any]) -> Any:
    """屏蔽 ``_specialize.announce`` 的一次性提示，保持门禁输出干净。"""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return call()


def _relative_error(actual: Any, expected: Any) -> float:
    """按最大输出元归一的最大绝对偏差；形状不一致返回 ``inf``。"""
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.shape != right.shape:
        return float("inf")
    if right.size == 0:
        return 0.0
    scale = float(np.max(np.abs(right)))
    difference = float(np.max(np.abs(left - right)))
    if scale == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / scale


def _equations() -> list[str]:
    return _parse_equations()


def _sample_equations(count: int = SAMPLE_SIZE) -> list[str]:
    """跨全部 case 等距抽样，保证抽检覆盖 6/7/13/14/16 元数的各类拓扑。"""
    equations = _equations()
    step = max(1, len(equations) // count)
    selected = equations[::step][:count]
    if len(selected) < count:                            # 语料很小也不重复取
        selected = equations[:count]
    return selected


def _shapes(equation: str, dim: int = DIM) -> list[tuple[int, ...]]:
    terms = equation.split("->")[0].split(",")
    return [tuple(dim for _ in term) for term in terms]


def _reset_kernel_globals() -> tuple[Any, ...]:
    """隔离进程级缓存：入口缓存、dtype 探测缓存与负缓存都会把别的用例的结论
    带到本模块。

    返回快照，调用方在 ``finally`` 里用 :func:`_restore_kernel_globals` 还回去，
    避免反过来污染后续用例。
    """
    saved = (
        _kernel_kupl._entry_topology,
        _kernel_kupl._failed_topology,
        dict(_kernel_kupl._TOPOLOGY_DTYPE_SUPPORT),
        dict(_kernel_cuda._TOPOLOGY_DTYPE_SUPPORT),
    )
    _kernel_kupl._entry_topology = None
    _kernel_kupl._failed_topology = False
    _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.clear()
    _kernel_cuda._TOPOLOGY_DTYPE_SUPPORT.clear()
    _kernel_kupl._negative_cache_clear()
    return saved


def _restore_kernel_globals(saved: tuple[Any, ...]) -> None:
    (_kernel_kupl._entry_topology, _kernel_kupl._failed_topology,
     kupl_support, cuda_support) = saved
    _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.clear()
    _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.update(kupl_support)
    _kernel_cuda._TOPOLOGY_DTYPE_SUPPORT.clear()
    _kernel_cuda._TOPOLOGY_DTYPE_SUPPORT.update(cuda_support)
    _kernel_kupl._negative_cache_clear()


def _allow_dtype(module: Any, dtype: Any) -> None:
    """显式放行某 dtype，避免"宿主机探测健康度"劫持被测分支。

    只在**patch 了原生入口**的用例里用：那里探测本身不可能成功（探测也会走
    被替换的入口），再让门控去探测就变成测宿主机而不是测契约。
    """
    module._TOPOLOGY_DTYPE_SUPPORT[np.dtype(dtype)] = True


def _require_kupl() -> None:
    if _kernel_kupl._function_topology() is None:
        raise unittest.SkipTest(
            f"native KUPL topology entry unavailable: "
            f"{_kernel_kupl.LAST_FAILURE.get('reason', '')}")


def _require_cuda() -> None:
    from ..einsum.cuda._program import topology_binding

    if not _cupy_ready():
        raise unittest.SkipTest("cupy/CUDA device is not available")
    if topology_binding() is None:
        raise unittest.SkipTest(
            "qcd_cuda_topology_apply is not exported by the archive")


def _entry_status_recorder(
        status: int = 1) -> tuple[Callable[..., int], dict[str, Any]]:
    """记录一次原生调用的入参并返回固定状态码（只验参数，不验数值）。"""
    record: dict[str, Any] = {}

    def entry(plan_ptr, plan_len, pointers, n_operands, offsets, out_ptr, dtype_id):
        record["plan"] = ctypes.string_at(plan_ptr, int(plan_len))
        record["plan_len"] = int(plan_len)
        record["pointers"] = [int(pointers[index]) for index in range(int(n_operands))]
        record["n_operands"] = int(n_operands)
        record["offsets"] = (
            None if offsets is None else [int(value) for value in offsets])
        record["out_ptr"] = int(out_ptr)
        record["dtype_id"] = int(dtype_id)
        return status

    return entry, record


def _clear_kernel_warnings() -> None:
    from ..einsum import _api

    _api._KERNEL_SKIP_WARNED.clear()


# ── 1. dtype 声明 ───────────────────────────────────────────────────────────

def test_dtype_contract_is_uniform_across_all_four_dtypes():
    """两个入口的 dtype 契约必须是四类**平权**，不设主精度或默认精度。

    ``TOPOLOGY_DTYPES`` 是契约本身（四类都在），``topology_dtypes()`` 只是它的
    只读视图且**不得**顺手触发探测；``topology_enabled_dtypes()`` 才回答"本进程
    实测通过哪些"，四类走同一把尺子。
    """
    for module in (_kernel_kupl, _kernel_cuda):
        declared = tuple(module.TOPOLOGY_DTYPES)
        assert all(isinstance(item, np.dtype) for item in declared), declared
        assert set(declared) == {np.dtype(item) for item in ALL_DTYPES}, (
            f"{module.__name__}: 契约必须是四类平权 dtype，实际 {declared}")
        # 顺序与 QCD_DTYPE_* 同值，Python/C++ 两侧才能直接互传 dtype id。
        assert declared == tuple(np.dtype(item) for item in ALL_DTYPES), declared

        saved = dict(module._TOPOLOGY_DTYPE_SUPPORT)
        module._TOPOLOGY_DTYPE_SUPPORT.clear()
        try:
            # 只读视图：未探测时不把任何一类报成可用，也不触发探测。
            assert tuple(module.topology_dtypes()) == declared
            assert module._TOPOLOGY_DTYPE_SUPPORT == {}, \
                f"{module.__name__}: topology_dtypes() 不得触发探测"
        finally:
            module._TOPOLOGY_DTYPE_SUPPORT.clear()
            module._TOPOLOGY_DTYPE_SUPPORT.update(saved)


# ── 2. dtype 矩阵 ───────────────────────────────────────────────────────────

def _check_dtype_matrix(module: Any, path: str, to_array: Callable[[Any], Any],
                        to_host: Callable[[Any], Any]) -> list[str]:
    """四类 dtype 的完整行为矩阵；返回被接受的 dtype 名字列表。

    四类 dtype 完全平权，判定口径也只有一条：命中要求数值/元数据全对；回落要求
    留下原因、不写 ``LAST_INFO``、不改写操作数 dtype，并且原因必须是
    "探测没过"（该 dtype 不在 ``_TOPOLOGY_DTYPE_SUPPORT`` 的 True 集合里）或
    ``status == 4``（这条计划本入口不认）。1/3（参数或内部错误）一律失败。
    """
    accepted = []
    for dtype in ALL_DTYPES:
        numpy_operands = probe_operands(2, dtype, seed=11)
        operands = [to_array(item) for item in numpy_operands]
        module.LAST_INFO.clear()
        module.LAST_FAILURE.update({"reason": "", "status": None})
        result = module.try_topology_apply(PROBE_EQUATION, operands)
        expected = np.einsum(
            PROBE_EQUATION, *(to_host(item) for item in operands))

        if result is None:
            reason = str(module.LAST_FAILURE.get("reason", ""))
            status = module.LAST_FAILURE.get("status")
            name = np.dtype(dtype).name
            assert reason, f"{path}: {name} 回落必须留下原因"
            assert not module.LAST_INFO, \
                f"{path}: 回落时不得留下 LAST_INFO（说明根本没跑原生）"
            # 回落绝不等于降级：原数组的 dtype 必须原封不动。
            assert all(np.dtype(item.dtype) == np.dtype(dtype) for item in operands)
            probed_out = module._TOPOLOGY_DTYPE_SUPPORT.get(np.dtype(dtype)) is False
            assert probed_out or status == 4, (
                f"{path}: {name} 回落必须给出可分辨的原因（探测没过或 status=4），"
                f"实际 status={status}: {reason}")
            continue

        accepted.append(np.dtype(dtype).name)
        assert result.dtype == np.dtype(dtype), \
            f"{path}: {np.dtype(dtype)} 的结果被改成了 {result.dtype}"
        assert module._TOPOLOGY_DTYPE_SUPPORT.get(np.dtype(dtype)) is True, \
            f"{path}: {np.dtype(dtype)} 命中却不在实测通过的集合里"
        error = _relative_error(to_host(result), expected)
        assert error <= TOLERANCES[np.dtype(dtype).name], (
            f"{path}: {np.dtype(dtype)} 数值错误 rel={error:.3e}")

        info = module.LAST_INFO
        missing = [key for key in _PROBE_KEYS if key not in info]
        assert not missing, f"{path}: LAST_INFO 缺 {missing}"
        assert info["dtype"] == str(np.dtype(dtype))
        assert info["dtype_id"] == native_dtype_id(dtype)
        assert info["n_operands"] == len(PROBE_TERMS)
        assert info["steps"] >= 1
        assert tuple(info["output_shape"]) == tuple(expected.shape)
        assert info["out"] == "allocated"
        assert tuple(info["copied"]) == ()
        assert info["offsets"] is None
        assert info["path"] == path, info["path"]

    if not accepted:
        raise unittest.SkipTest(
            f"{path}: 这个原生构建没有可执行 dtype（四类全部 status=4，待实现）")
    return accepted


def test_dtype_matrix_kupl():
    """KUPL：四类 dtype 的命中/回落矩阵与 LAST_INFO 全字段一致。"""
    _require_kupl()
    _check_dtype_matrix(_kernel_kupl, "kupl_topology", lambda item: item,
                        lambda item: np.asarray(item))


def test_dtype_matrix_cuda():
    """CUDA：四类 dtype 的命中/回落矩阵与 LAST_INFO 全字段一致。"""
    _require_cuda()
    import cupy

    _check_dtype_matrix(_kernel_cuda, "cuda_topology", cupy.asarray, cupy.asnumpy)


def test_probe_failed_dtype_is_never_narrowed_to_another():
    """探测没过的 dtype 必须直接回落，且绝不把数组降级成别的精度。"""
    module = _kernel_kupl
    _require_kupl()
    dtype = np.dtype(np.complex128)
    left, right = _make_operands([(3, 4), (4, 5)], dtype, seed=13)
    calls = []
    saved = dict(module._TOPOLOGY_DTYPE_SUPPORT)
    module._TOPOLOGY_DTYPE_SUPPORT[dtype] = False
    try:
        with patch.object(kupl_einsum, "topology_apply",
                          side_effect=lambda *a, **k:
                          calls.append(a) or np.zeros((3, 5))):
            result = module.try_topology_apply("ij,jk->ik", [left, right])
    finally:
        module._TOPOLOGY_DTYPE_SUPPORT.clear()
        module._TOPOLOGY_DTYPE_SUPPORT.update(saved)

    assert result is None, "探测没过的 dtype 必须返回 None"
    assert calls == [], "不可用的 dtype 不得下发原生，更不得降级成别的精度后下发"
    assert left.dtype == dtype and right.dtype == dtype, "输入 dtype 被改动"
    reason = str(module.LAST_FAILURE.get("reason", ""))
    assert "complex128" in reason and "downcast" in reason, reason
    assert module.LAST_INFO == {}, "回落时不得记录任何调用 info"


# ── 3. 参数与内存 ───────────────────────────────────────────────────────────

def test_native_arguments_carry_plan_order_and_out_buffer():
    """plan 字节流、操作数顺序、out 指针、dtype id 必须逐项对上。"""
    module = _kernel_kupl
    _require_kupl()
    big = np.zeros((6, 4), dtype=np.complex64)
    big[1:4] = np.arange(12, dtype=np.float32).reshape(3, 4)
    slab = big[1:4]                      # 连续切片：指针自带偏移，必须零拷贝
    assert slab.flags["C_CONTIGUOUS"] and np.shares_memory(slab, big)
    right = np.arange(20, dtype=np.complex64).reshape(4, 5)
    out = np.empty((3, 5), dtype=np.complex64)

    entry, record = _entry_status_recorder(status=1)
    _allow_dtype(module, np.complex64)
    with patch.object(kupl_einsum, "load_topology", lambda: entry):
        assert module.try_topology_apply("ij,jk->ik", [slab, right], out=out) is None
    assert "status 1" in str(module.LAST_FAILURE["reason"]), module.LAST_FAILURE

    plan = module._topology_plan(
        module._topology_key("ij,jk->ik", [slab, right]), "ij,jk->ik", [slab, right])
    assert record["plan"] == plan, "下发的 plan 字节流与计划缓存不一致"
    assert record["plan_len"] == len(plan)
    assert record["n_operands"] == 2
    assert record["pointers"] == [int(slab.ctypes.data), int(right.ctypes.data)], \
        "操作数指针顺序必须与 plan 的 input_shapes 一致（且切片零拷贝）"
    assert record["out_ptr"] == int(out.ctypes.data), "out 缓冲区指针不一致"
    assert record["dtype_id"] == native_dtype_id(np.complex64)
    assert record["offsets"] is None, "operand_offsets 必须保持 NULL"

    # 真跑一遍：同样的切片必须零拷贝命中且数值正确。
    module.LAST_INFO.clear()
    result = module.try_topology_apply("ij,jk->ik", [slab, right])
    if result is None:
        # 参数契约已在上面验完；这一半只要求"未实现时必须合法回落"。
        assert module.LAST_FAILURE.get("status") == 4, module.LAST_FAILURE
        assert _relative_error(contract("ij,jk->ik", slab, right),
                               np.einsum("ij,jk->ik", slab, right)) <= 1e-5
        return
    assert tuple(module.LAST_INFO["copied"]) == (), "连续切片不应被复制"
    assert _relative_error(result, np.einsum("ij,jk->ik", slab, right)) <= 1e-5


def test_cuda_arguments_carry_plan_order_and_out_buffer():
    """CUDA：设备指针顺序、out 指针、dtype id 与 plan 字节流逐项对上。"""
    module = _kernel_cuda
    _require_cuda()
    import cupy

    from ..einsum.cuda import _program

    _allow_dtype(_kernel_cuda, np.complex64)
    left = cupy.asarray(np.arange(12, dtype=np.float32).reshape(3, 4)
                        + 1j * np.ones((3, 4), dtype=np.float32))
    right = cupy.asarray((np.arange(20, dtype=np.float32).reshape(4, 5)
                          * (1.0 + 0.0j)).astype(np.complex64))
    out = cupy.empty((3, 5), dtype=cupy.complex64)

    record: dict[str, Any] = {}

    def apply(plan_ptr, plan_len, pointers, n_operands, offsets, out_ptr,
              dtype_id, stream):
        record["plan"] = ctypes.string_at(plan_ptr, int(plan_len))
        record["n_operands"] = int(n_operands)
        record["pointers"] = [int(pointers[index]) for index in range(int(n_operands))]
        record["offsets"] = None if offsets is None else [int(v) for v in offsets]
        record["out_ptr"] = int(out_ptr)
        record["dtype_id"] = int(dtype_id)
        record["stream"] = stream
        return 1

    binding = _program.topology_binding()
    with patch.object(binding, "apply", apply):
        assert module.try_topology_apply("ij,jk->ik", [left, right], out=out) is None
    assert "status 1" in str(module.LAST_FAILURE["reason"]), module.LAST_FAILURE

    plan = module._topology_plan(
        module._topology_key("ij,jk->ik", [left, right]), "ij,jk->ik", [left, right])
    assert record["plan"] == plan
    assert record["n_operands"] == 2
    assert record["pointers"] == [int(left.data.ptr), int(right.data.ptr)]
    assert record["out_ptr"] == int(out.data.ptr)
    assert record["dtype_id"] == native_dtype_id(np.complex64)
    assert record["offsets"] is None, "operand_offsets 必须保持 NULL"
    assert record["stream"] is None, "stream 只支持默认流（NULL）"


def test_offsets_are_never_forwarded():
    """非 None 的 offsets 会与视图指针双重偏移，必须 fail closed。"""
    module = _kernel_kupl
    _require_kupl()
    left, right = _make_operands([(3, 4), (4, 5)], np.complex64, seed=17)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("operand_offsets 非 None 时不得调用原生")

    _allow_dtype(module, np.complex64)   # 探测也会走被替换的 topology_apply
    with patch.object(kupl_einsum, "topology_apply", forbidden):
        assert module.try_topology_apply(
            "ij,jk->ik", [left, right], offsets=[4, 0]) is None
    reason = str(module.LAST_FAILURE.get("reason", ""))
    assert "operand_offsets" in reason, reason
    assert module.LAST_INFO == {}


# ── 4. 非连续输入、out 缓冲区、不可编码计划 ─────────────────────────────────

def test_noncontiguous_operands_are_copied_and_recorded():
    """非连续操作数：命中则必须登记复制下标，否则必须给出回落原因。"""
    module = _kernel_kupl
    _require_kupl()
    base = np.ascontiguousarray(
        np.arange(24, dtype=np.float32).reshape(6, 4)
        + 1j * np.arange(24, dtype=np.float32).reshape(6, 4))
    view = base[::2]                                     # 非连续（行步长 2）
    assert not view.flags["C_CONTIGUOUS"]
    right = np.ascontiguousarray(
        np.arange(20, dtype=np.float32).reshape(4, 5)
        * (1.0 + 0.5j)).astype(np.complex64)
    expected = np.einsum("ij,jk->ik", view, right)

    module.LAST_INFO.clear()
    result = module.try_topology_apply("ij,jk->ik", [view, right])
    if result is None:
        assert str(module.LAST_FAILURE.get("reason", "")), "回落必须说明原因"
        assert _relative_error(contract("ij,jk->ik", view, right), expected) <= 1e-5
        return
    assert tuple(module.LAST_INFO["copied"]) == (0,), \
        f"非连续输入必须登记复制下标，实际 {module.LAST_INFO['copied']}"
    assert _relative_error(result, expected) <= 1e-5


def test_out_buffer_contract():
    """out：连续同 dtype/shape 必须复用，其余一律回落且不改缓冲区。"""
    module = _kernel_kupl
    _require_kupl()
    left, right = _make_operands([(3, 4), (4, 5)], np.complex64, seed=19)
    expected = np.einsum("ij,jk->ik", left, right)

    buffer = np.full((3, 5), np.nan + 1j * np.nan, dtype=np.complex64)
    result = module.try_topology_apply("ij,jk->ik", [left, right], out=buffer)
    if result is None:
        reason = str(module.LAST_FAILURE.get("reason", ""))
        assert module.LAST_FAILURE.get("status") == 4, (reason, module.LAST_FAILURE)
        assert np.isnan(buffer).all(), "回落时不得写调用方的 out 缓冲区"
        raise unittest.SkipTest(f"topology entry cannot execute this plan yet: {reason}")
    assert result is buffer, "提供 out 时必须把同一个缓冲区返回"
    assert module.LAST_INFO["out"] == "provided"
    assert not np.isnan(buffer).any(), "原生必须写满整个 out 缓冲区"
    assert _relative_error(buffer, expected) <= 1e-5

    rejected = {
        "dtype": np.zeros((3, 5), dtype=np.float32),
        "contiguous": np.zeros((3, 10), dtype=np.complex64)[:, ::2],
        "numpy array": [(0.0,) * 5] * 3,
    }
    for label, buffer_value in rejected.items():
        filler = buffer_value if isinstance(buffer_value, np.ndarray) else buffer_value
        if isinstance(filler, np.ndarray):
            filler[...] = np.nan
        module.LAST_INFO.clear()
        assert module.try_topology_apply(
            "ij,jk->ik", [left, right], out=filler) is None, label
        reason = str(module.LAST_FAILURE.get("reason", ""))
        assert label in reason, (label, reason)
        assert module.LAST_INFO == {}
        if isinstance(filler, np.ndarray) and filler.dtype == np.complex64:
            assert np.isnan(filler).all(), f"{label}: 被拒绝的 out 不得被写入"

    shape_bad = np.zeros((5, 3), dtype=np.complex64)
    assert module.try_topology_apply("ij,jk->ik", [left, right], out=shape_bad) is None
    assert "output shape" in str(module.LAST_FAILURE.get("reason", ""))
    assert not shape_bad.any(), "shape 不匹配的 out 不得被写入"


def test_allocated_output_does_not_alias_operands():
    """不指定 out 时必须新分配结果，不能把输入别名交给调用方。"""
    module = _kernel_kupl
    _require_kupl()
    left, right = _make_operands([(4, 4), (4, 4)], np.float32, seed=23)
    result = module.try_topology_apply("ij,jk->ik", [left, right])
    if result is None:
        raise unittest.SkipTest(f"topology entry declined: {module.LAST_FAILURE}")
    assert module.LAST_INFO["out"] == "allocated"
    assert not np.shares_memory(result, left) and not np.shares_memory(result, right)
    assert _relative_error(result, np.einsum("ij,jk->ik", left, right)) <= 1e-5


def test_unencodable_plan_falls_back_without_raising():
    """单操作数内重复下标（对角类）计划不可编码，必须回落而不是抛异常。"""
    for module in (_kernel_kupl, _kernel_cuda):
        if module is _kernel_kupl:
            _require_kupl()
            operand = np.eye(4, dtype=np.complex64)
        else:
            _require_cuda()
            import cupy

            operand = cupy.asarray(np.eye(4, dtype=np.complex64))
            _allow_dtype(_kernel_cuda, np.complex64)
        assert module.try_topology_apply("ii->i", [operand]) is None
        reason = str(module.LAST_FAILURE.get("reason", ""))
        assert "repeated" in reason or "unavailable" in reason, reason


# ── 5. 语料抽检 ─────────────────────────────────────────────────────────────

def test_corpus_sample_matches_generic_path():
    """≥100 条 dim=4 真实拓扑：命中与回落两条路都必须给出正确结果。"""
    equations = _sample_equations()
    assert len(equations) >= 100, f"抽检条数 {len(equations)} < 100"
    arities = {len(equation.split("->")[0].split(",")) for equation in equations}
    assert {6, 13}.issubset(arities), f"抽检未覆盖多元数拓扑: {sorted(arities)}"

    _require_kupl()
    module = _kernel_kupl
    hits = 0
    misses: list[tuple[str, str]] = []
    worst = 0.0
    for index, equation in enumerate(equations):
        dtype = np.complex64
        operands = _make_operands(
            _shapes(equation), dtype, seed=20261111 + index)
        expected = contract(equation, *operands, specialize=False)
        module.LAST_INFO.clear()
        result = module.try_topology_apply(equation, operands)
        if result is None:
            reason = str(module.LAST_FAILURE.get("reason", ""))
            assert reason, f"{equation}: 回落必须留下原因"
            misses.append((equation, reason))
        else:
            hits += 1
            assert result.dtype == dtype, f"{equation}: dtype {result.dtype} 被改动"
            error = _relative_error(result, expected)
            worst = max(worst, error)
            assert error <= TOLERANCES["complex64"], (
                f"{equation}: 与 contract(specialize=False) 不一致 rel={error:.3e}")

        # 回落契约：无论命中与否，通用路径都必须给出同一个结果。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fallback = _quiet(
                lambda: contract(equation, *operands, specialize=True))
        assert _relative_error(fallback, expected) <= TOLERANCES["complex64"], (
            f"{equation}: 回落路径数值错误")

    if hits == 0:
        raise unittest.SkipTest(
            "native topology entry declined all sampled equations: "
            f"{misses[0][1] if misses else 'no reason'}")
    assert hits >= len(equations) // 2, (
        f"命中率异常偏低 {hits}/{len(equations)}："
        f"{[reason for _, reason in misses[:3]]}")
    print(f"corpus sample: {hits}/{len(equations)} hits, max rel {worst:.3e}")


# ── 6. 失败回落 ─────────────────────────────────────────────────────────────

def _topology_equation() -> str:
    """挑一条由拓扑入口服务的真实语料方程。

    字面注册表（gauge-propagator/gamma-sandwich/sush-*）已随新版拓扑内核删除，
    现在**所有**语料方程都由拓扑入口服务，因此直接取第一条即可。
    """
    from ..einsum import _specialize

    for equation in _equations():
        entry = _specialize.match(equation) or {}
        if entry.get("kind") == "kernel(topology)":
            return equation
    raise unittest.SkipTest("语料里找不到拓扑入口服务的方程")


def test_native_failure_falls_back_with_one_warning():
    """原生返回错误码：结果仍正确，且同一形态只发一条警告。"""
    _require_kupl()
    equation = _topology_equation()
    operands = _make_operands(_shapes(equation, dim=2), np.complex64, seed=29)
    expected = contract(equation, *operands, specialize=False)

    entry, record = _entry_status_recorder(status=1)
    _allow_dtype(_kernel_kupl, np.complex64)
    _clear_kernel_warnings()
    # 分派链里 `kupl_dedicated` 排在 `kupl_topology` 之前：不屏蔽它的话，
    # dedicated 会先成功接手，`load_topology` 根本不会被调用，本契约就测不到
    # "拓扑入口失败后的回落"。这里把 dedicated 显式禁用，只留 topology 路径。
    with patch.object(kupl_einsum, "load_topology", lambda: entry), \
            patch.object(_kernel_kupl, "try_dedicated",
                         lambda *args, **kwargs: None), \
            patch.object(_kernel_cuda, "enabled", lambda: False), \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = _quiet(lambda: contract(equation, *operands, specialize=True))
        second = _quiet(lambda: contract(equation, *operands, specialize=True))

    assert record, "原生入口必须真的被调用过（否则这条契约没测到回落）"
    assert _relative_error(first, expected) <= TOLERANCES["complex64"]
    assert _relative_error(second, expected) <= TOLERANCES["complex64"]
    runtime = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert len(runtime) == 1, (
        f"同一形态的回落只应警告一次，实际 {len(runtime)}: "
        f"{[str(item.message)[:80] for item in runtime]}")
    message = str(runtime[0].message)
    assert "falls back" in message and "topology:" in message, message
    assert "status 1" in message, message


# ── 5. dtype 探测：容器化、缓存与原因可分辨（2026-11-14 补） ────────────────

def _reset_dtype_support(module: Any) -> dict[np.dtype, bool]:
    """清空探测缓存；返回快照供调用方恢复（不得污染后续用例）。"""
    saved = dict(module._TOPOLOGY_DTYPE_SUPPORT)
    module._TOPOLOGY_DTYPE_SUPPORT.clear()
    return saved


def _restore_dtype_support(module: Any, saved: dict[np.dtype, bool]) -> None:
    module._TOPOLOGY_DTYPE_SUPPORT.clear()
    module._TOPOLOGY_DTYPE_SUPPORT.update(saved)


def test_dtype_probe_contains_exceptions_and_is_cached():
    """探测体任何异常都只返回 ``False``；失败结果被缓存，第二次不再重探。

    回归点：CUDA 侧 ``cupy.arange`` 曾在 ``try`` 之外，NVRTC 架构不匹配时把
    ``CompileException`` 抛给调用方 —— 四类 dtype 同等受害，且失败结果无法缓存、
    每次调用都重探。四类 dtype 一起验，不做任何一类豁免。
    """
    from ..einsum import _api

    for module in (_kernel_kupl, _kernel_cuda):
        with patch.object(_api, "contract_path", side_effect=RuntimeError("boom")):
            assert module._probe_dtype(np.dtype(np.complex128)) is False, \
                f"{module.__name__}: 探测抛异常时必须返回 False，不得外泄异常"

        saved = _reset_dtype_support(module)
        calls = []

        def exploding_probe(dtype, _calls=calls):
            _calls.append(dtype)
            raise RuntimeError("probe exploded")

        try:
            with patch.object(module, "_probe_dtype", exploding_probe):
                first = module._topology_dtype_enabled(np.dtype(np.complex128))
                second = module._topology_dtype_enabled(np.dtype(np.complex128))
            assert first is False and second is False, module.__name__
            assert len(calls) == 1, (
                f"{module.__name__}: 探测失败必须缓存，实际探测 {len(calls)} 次")
            assert module._TOPOLOGY_DTYPE_SUPPORT[np.dtype(np.complex128)] is False
            assert np.dtype(np.complex128) not in set(
                module.topology_enabled_dtypes())
        finally:
            _restore_dtype_support(module, saved)


def test_topology_dtype_reasons_are_distinguishable():
    """契约外 dtype 与"探测没过"必须给出不同且可行动的原因。

    回归点：两者原先共用一句 ``dtype ... is not enabled ... (supported:
    complex64, float32)``，用户读成"complex128 永远不支持"。
    """
    for module, hint in ((_kernel_kupl, "cpp/kupl/einsum/make.sh"),
                         (_kernel_cuda, "cpp/cuda/einsum/make-nv.sh")):
        saved = _reset_dtype_support(module)
        try:
            with patch.object(module, "_probe_dtype", lambda dtype: False):
                probe_failed = module._topology_dtype_reason(np.dtype(np.complex128))
                real_double = module._topology_dtype_reason(np.dtype(np.float64))
            outside = module._topology_dtype_reason(np.dtype(np.int64))
            single = module._topology_dtype_reason(np.dtype(np.float32))
        finally:
            _restore_dtype_support(module, saved)

        # 单精度不豁免探测：本机探测通过才可下发，没通过就报同款可恢复原因。
        if module._topology_dtype_enabled(np.dtype(np.float32)):
            assert single is None, f"{module.__name__}: 探测通过时 float32 可下发"
        else:
            assert single is not None and "probe failed" in single, single
        assert probe_failed is not None, module.__name__
        assert "probe failed" in probe_failed, probe_failed
        assert "complex128" in probe_failed and hint in probe_failed, probe_failed
        assert outside is not None, module.__name__
        assert "outside the topology dtype contract" in outside, outside
        assert probe_failed != outside, "两种拒绝必须可分辨"
        assert real_double is not None and "probe failed" in real_double, real_double

        # 四类 dtype 平权：任一类探测失败都必须报同一类"可恢复"原因。
        saved = _reset_dtype_support(module)
        try:
            with patch.object(module, "_probe_dtype", lambda dtype: False):
                for dtype in ALL_DTYPES:
                    reason = module._topology_dtype_reason(np.dtype(dtype))
                    assert reason is not None and "probe failed" in reason, (
                        module.__name__, dtype, reason)
        finally:
            _restore_dtype_support(module, saved)


def test_try_topology_apply_reports_probe_failure_without_raising():
    """探测失败时入口只回落并记真实原因，绝不抛异常、绝不报成"dtype 不支持"。"""
    _require_kupl()
    dtype = np.dtype(np.complex128)
    operands = _make_operands([(2, 2), (2, 2)], dtype, seed=5)
    saved = _reset_dtype_support(_kernel_kupl)
    _kernel_kupl.LAST_INFO.clear()
    _kernel_kupl.LAST_FAILURE.update({"reason": "", "status": None})
    try:
        with patch.object(_kernel_kupl, "_probe_dtype", lambda item: False):
            result = _kernel_kupl.try_topology_apply("ij,jk->ik", operands)
        reason = str(_kernel_kupl.LAST_FAILURE.get("reason", ""))
        assert not _kernel_kupl.LAST_INFO, "回落时不得留下 LAST_INFO（说明根本没跑原生）"
    finally:
        _restore_dtype_support(_kernel_kupl, saved)
    assert result is None, "探测失败时必须回落通用路径"
    assert "probe failed" in reason, reason
    assert "not enabled" not in reason, reason


def test_wide_output_topologies_are_not_transposed():
    """输出秩 >= 4 的拓扑必须逐元素对上：分块转置的指数曾写反过。

    回归点：``topology_gemm.h`` 的"二维互换"快捷路径把两侧分块指数写反
    （``dim^split`` 应为 ``dim^(rank-split)``），只有 ``split == rank/2`` 时才
    恰好等效。120 条等距抽样与 2582 条基础表契约都覆盖不到，全语料对拍在 16 条
    4 输出轴拓扑上给出 rel≈1（值对但被错误转置）。这里按语料里真实的 4 输出轴
    方程直接钉住，抽样再少也不会漏。
    """
    _require_kupl()
    # 只取**基础表**（seq <= base_table_count）：附录条目没有编译期步表，
    # 去兜底后它们一律 status=4，那是设计意图而不是回归。
    # 步表只由基础表前 2582 条生成（见 `_topology_kernel_gen.MAX_KERNELS`）。
    wide = [task for task in tasklist.TASKS
            if task.seq <= 2582 and task.equation
            and len(task.equation.split("->")[-1]) >= 4]
    assert wide, "语料里应当存在 4 输出轴的拓扑"
    checked = 0
    for task in wide[::max(1, len(wide) // 12)]:
        for dim in (2, 3):
            operands = _make_operands(
                [(dim,) * len(term)
                 for term in task.equation.split("->")[0].split(",")],
                np.complex128, seed=task.seq + dim)
            expected = contract(task.equation, *operands, specialize=False)
            _kernel_kupl.LAST_INFO.clear()
            result = _kernel_kupl.try_topology_apply(task.equation, operands)
            assert result is not None, (
                f"seq={task.seq} dim={dim} 拓扑入口应当命中："
                f"{_kernel_kupl.LAST_FAILURE.get('reason')}")
            error = _relative_error(result, expected)
            assert error <= TOLERANCES["complex128"], (
                f"seq={task.seq} dim={dim} 4 输出轴拓扑转置错 rel={error:.3e}")
            checked += 1
    assert checked >= 12, checked


def test_higher_dims_and_larger_extents_hold_for_all_dtypes():
    """dim ∈ {5,6,8,12} × 四 dtype：命中、dtype 保真、数值在容差内。

    ``dim=4`` 是语料基准与 ``_dedicated_allowlist`` 的标定尺寸，但内核的轴长是
    ``dim`` 这个标量而不是常量，越界的尺寸最容易暴露"按 4 写死"的假设。这里不
    追求生产尺寸（那是 ``bench_dtype_matrix --scene`` 的职责），只钉住结构正确性。
    """
    _require_kupl()
    checked = 0
    for dtype in ALL_DTYPES:
        name = np.dtype(dtype).name
        # 步表只由**基础表 2582 条**生成（`_topology_kernel_gen.MAX_KERNELS=2582`,
        # seq 1..2582）；附录（2583+）没有步表，去兜底后恒 status=4。
        for seq in (2, 2000):
            task = tasklist.lookup(seq)
            for dim in (5, 6, 8, 12):
                operands = _make_operands(
                    [(dim,) * len(term)
                     for term in task.equation.split("->")[0].split(",")],
                    dtype, seed=task.seq + dim)
                expected = contract(task.equation, *operands, specialize=False)
                _kernel_kupl.LAST_INFO.clear()
                result = _kernel_kupl.try_topology_apply(task.equation, operands)
                assert result is not None, (
                    f"{name} seq={seq} dim={dim} 拓扑入口应当命中："
                    f"{_kernel_kupl.LAST_FAILURE.get('reason')}")
                assert np.dtype(result.dtype) == np.dtype(dtype), (name, dim)
                error = _relative_error(result, expected)
                assert error <= TOLERANCES[name], (
                    f"{name} seq={seq} dim={dim} rel={error:.3e}")
                checked += 1
    assert checked == len(ALL_DTYPES) * 2 * 4, checked


def test_every_dtype_is_native_when_its_probe_passes():
    """四类 dtype 各自的活证据：探测通过 ⇒ 必须真的走原生拓扑入口。

    判定对四类完全一样——``LAST_INFO['path']`` 命中、dtype 保真、数值在容差内。
    没有"某一类只要不崩就算支持"的豁免。
    """
    _require_kupl()
    checked = []
    for dtype in ALL_DTYPES:
        dtype = np.dtype(dtype)
        if not _kernel_kupl._topology_dtype_enabled(dtype):
            continue                      # 这个原生构建缺该类 GEMM，跳过即可
        assert dtype in set(_kernel_kupl.topology_dtypes())

        operands = probe_operands(2, dtype, seed=17)
        expected = np.einsum(PROBE_EQUATION, *operands)
        _kernel_kupl.LAST_INFO.clear()
        _kernel_kupl.LAST_FAILURE.update({"reason": "", "status": None})
        result = _kernel_kupl.try_topology_apply(PROBE_EQUATION, operands)
        assert result is not None, (
            f"{dtype} 探测通过却回落：{_kernel_kupl.LAST_FAILURE.get('reason')}")
        assert result.dtype == dtype, f"{dtype}: 结果 dtype 被改成 {result.dtype}"
        assert _relative_error(result, expected) <= TOLERANCES[dtype.name], dtype
        assert _kernel_kupl.LAST_INFO.get("path") == "kupl_topology", \
            (dtype, _kernel_kupl.LAST_INFO.get("path"))
        checked.append(dtype.name)

    if not checked:
        raise unittest.SkipTest("native KUPL build cannot execute any dtype")


# ── 入口 ────────────────────────────────────────────────────────────────────

def _isolated(test: Callable[[], None]) -> Callable[[], None]:
    """每个用例都从干净的入口缓存/负缓存开始，跑完还原，避免跨模块串味。"""

    def runner() -> None:
        saved = _reset_kernel_globals()
        try:
            test()
        finally:
            _restore_kernel_globals(saved)

    runner.__name__ = test.__name__
    runner.__doc__ = test.__doc__
    return runner


TESTS = tuple(_isolated(test) for test in (
    test_dtype_contract_is_uniform_across_all_four_dtypes,
    test_dtype_matrix_kupl,
    test_dtype_matrix_cuda,
    test_probe_failed_dtype_is_never_narrowed_to_another,
    test_dtype_probe_contains_exceptions_and_is_cached,
    test_topology_dtype_reasons_are_distinguishable,
    test_try_topology_apply_reports_probe_failure_without_raising,
    test_wide_output_topologies_are_not_transposed,
    test_higher_dims_and_larger_extents_hold_for_all_dtypes,
    test_every_dtype_is_native_when_its_probe_passes,
    test_native_arguments_carry_plan_order_and_out_buffer,
    test_cuda_arguments_carry_plan_order_and_out_buffer,
    test_offsets_are_never_forwarded,
    test_noncontiguous_operands_are_copied_and_recorded,
    test_out_buffer_contract,
    test_allocated_output_does_not_alias_operands,
    test_unencodable_plan_falls_back_without_raising,
    test_corpus_sample_matches_generic_path,
    test_native_failure_falls_back_with_one_warning,
))


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"{passed} passed, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
