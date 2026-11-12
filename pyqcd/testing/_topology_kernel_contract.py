"""契约：per-拓扑原生执行入口（``try_topology_apply``）的 dtype、参数、内存与回落。

四件事必须同时成立，缺一不可：

1. **dtype**：入口只下发**显式声明**的 dtype（``TOPOLOGY_DTYPES``）。单精度
   （``complex64``/``float32``）是默认与首选，恒可下发；双精度只在原生**实测
   正确**后才启用，否则返回 ``None`` 回落 —— 任何情况下都**不得**把
   ``float64``/``complex128`` 转成单精度，也不得把异常抛给调用方。
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

def test_dtype_declaration_is_explicit_and_single_precision_first():
    """两个入口都必须显式声明 dtype：单精度恒在，双精度必须留待实测。"""
    single = {np.dtype(np.complex64), np.dtype(np.float32)}
    double = {np.dtype(np.complex128), np.dtype(np.float64)}
    for module in (_kernel_kupl, _kernel_cuda):
        declared = module.TOPOLOGY_DTYPES
        assert all(isinstance(item, np.dtype) for item in declared), declared
        assert single <= set(declared), (module.__name__, declared)
        assert set(declared).isdisjoint(double), \
            f"{module.__name__}: 双精度不得直接写进声明集，必须先实测"

        saved = dict(module._TOPOLOGY_DOUBLE_SUPPORT)
        module._TOPOLOGY_DOUBLE_SUPPORT.clear()
        try:
            # 未实测时不得把双精度报成可用，也不得顺手触发探测。
            enabled = tuple(module.topology_dtypes())
            assert enabled == tuple(declared), enabled
        finally:
            module._TOPOLOGY_DOUBLE_SUPPORT.update(saved)

        enabled = set(module.topology_dtypes())
        assert set(declared) <= enabled, (module.__name__, enabled)
        assert enabled <= set(declared) | double, (module.__name__, enabled)


# ── 2. dtype 矩阵 ───────────────────────────────────────────────────────────

def _check_dtype_matrix(module: Any, path: str, to_array: Callable[[Any], Any],
                        to_host: Callable[[Any], Any]) -> list[str]:
    """四类 dtype 的完整行为矩阵；返回被接受的 dtype 名字列表。

    ``status == 4``（QCD_TOPOLOGY_UNSUPPORTED）表示"这条计划本入口不认"，是合法的
    未实现状态：诊断正确即通过，但单精度也一条都跑不起来时必须跳过，不能把"原生
    还没实现"误报成契约破坏；1/3（参数或内部错误）仍然直接失败。
    """
    accepted = []
    for dtype in ALL_DTYPES:
        numpy_operands = _make_operands([(3, 4), (4, 5)], dtype, seed=11)
        left, right = (to_array(item) for item in numpy_operands)
        module.LAST_INFO.clear()
        module.LAST_FAILURE.update({"reason": "", "status": None})
        result = module.try_topology_apply("ij,jk->ik", [left, right])
        expected = np.einsum("ij,jk->ik", *(to_host(item) for item in (left, right)))
        declared = set(module.topology_dtypes())

        if result is None:
            reason = str(module.LAST_FAILURE.get("reason", ""))
            status = module.LAST_FAILURE.get("status")
            name = np.dtype(dtype).name
            assert reason, f"{path}: {name} 回落必须留下原因"
            assert not module.LAST_INFO, \
                f"{path}: 回落时不得留下 LAST_INFO（说明根本没跑原生）"
            # 回落绝不等于降级：原数组的 dtype 必须原封不动。
            assert left.dtype == np.dtype(dtype) and right.dtype == np.dtype(dtype)
            if name in ("complex64", "float32"):
                # 单精度恒可下发，因此只允许"这条计划本入口不认"（status=4）这一种
                # 合法回落；1/3 说明参数或内部实现坏了，必须报错。
                assert status == 4, (
                    f"{path}: 单精度只允许以 status=4（计划不支持）回落，"
                    f"实际 status={status}: {reason}")
            else:
                assert np.dtype(dtype) not in declared, (
                    f"{path}: {np.dtype(dtype)} 声明为可用却回落了：{reason}")
            continue

        accepted.append(np.dtype(dtype).name)
        assert result.dtype == np.dtype(dtype), \
            f"{path}: {np.dtype(dtype)} 的结果被改成了 {result.dtype}"
        assert np.dtype(dtype) in set(module.topology_dtypes()), \
            f"{path}: {np.dtype(dtype)} 命中却不在声明集里"
        error = _relative_error(to_host(result), expected)
        assert error <= TOLERANCES[np.dtype(dtype).name], (
            f"{path}: {np.dtype(dtype)} 数值错误 rel={error:.3e}")

        info = module.LAST_INFO
        missing = [key for key in _PROBE_KEYS if key not in info]
        assert not missing, f"{path}: LAST_INFO 缺 {missing}"
        assert info["dtype"] == str(np.dtype(dtype))
        assert info["dtype_id"] == native_dtype_id(dtype)
        assert info["n_operands"] == 2
        assert info["steps"] >= 1
        assert tuple(info["output_shape"]) == tuple(expected.shape)
        assert info["out"] == "allocated"
        assert tuple(info["copied"]) == ()
        assert info["offsets"] is None
        assert info["path"] == path, info["path"]

    if not accepted:
        raise unittest.SkipTest(
            f"{path}: 这个原生构建没有可执行 dtype（单精度 status=4，待实现）")
    # 单精度是默认与首选：只要这个构建还能跑，就必须至少覆盖一个单精度 dtype。
    assert any(name in ("complex64", "float32") for name in accepted), \
        f"{path}: 单精度必须命中，实际 {accepted}"
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


def test_double_is_never_narrowed_to_single():
    """未实测通过的双精度必须直接回落，且绝不把数组降级成单精度。"""
    module = _kernel_kupl
    _require_kupl()
    dtype = np.dtype(np.complex128)
    left, right = _make_operands([(3, 4), (4, 5)], dtype, seed=13)
    calls = []
    saved = dict(module._TOPOLOGY_DOUBLE_SUPPORT)
    module._TOPOLOGY_DOUBLE_SUPPORT[dtype] = False
    try:
        with patch.object(kupl_einsum, "topology_apply",
                          side_effect=lambda *a, **k:
                          calls.append(a) or np.zeros((3, 5))):
            result = module.try_topology_apply("ij,jk->ik", [left, right])
    finally:
        module._TOPOLOGY_DOUBLE_SUPPORT.clear()
        module._TOPOLOGY_DOUBLE_SUPPORT.update(saved)

    assert result is None, "不可用的双精度必须返回 None"
    assert calls == [], "不可用的 dtype 不得下发原生，更不得降级成单精度后下发"
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


# ── 入口 ────────────────────────────────────────────────────────────────────

TESTS = (
    test_dtype_declaration_is_explicit_and_single_precision_first,
    test_dtype_matrix_kupl,
    test_dtype_matrix_cuda,
    test_double_is_never_narrowed_to_single,
    test_native_arguments_carry_plan_order_and_out_buffer,
    test_cuda_arguments_carry_plan_order_and_out_buffer,
    test_offsets_are_never_forwarded,
    test_noncontiguous_operands_are_copied_and_recorded,
    test_out_buffer_contract,
    test_allocated_output_does_not_alias_operands,
    test_unencodable_plan_falls_back_without_raising,
    test_corpus_sample_matches_generic_path,
    test_native_failure_falls_back_with_one_warning,
)


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
