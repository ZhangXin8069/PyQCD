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
from ..einsum import contract, contract_path
from ..einsum._bench_scene import dims as scene_dims
from ..einsum._native_plan import dtype_id as native_dtype_id
from ..einsum._native_plan import encode_execution_plan
from ..einsum.bench_specialized import _label_sizes
from ..einsum.kupl import einsum as kupl_einsum
from ..einsum import _wick_tasklist as tasklist
from ._topology_equations_contract import _parse_equations
from ._topology_kernel_gen import parse

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
# KUPL 侧多一项路线标记（``specialized``）：命中编译期 v3 步表与回落 v2 通用执行器
# 的返回码与数值完全相同，只有这个字段能区分——「拓扑命中」的一切判定都靠它。
_KUPL_EXTRA_PROBE_KEYS = ("specialized",)


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
        _kernel_kupl._entry_topology_route,
        _kernel_kupl._failed_topology_route,
        dict(_kernel_kupl._TOPOLOGY_DTYPE_SUPPORT),
        dict(_kernel_cuda._TOPOLOGY_DTYPE_SUPPORT),
    )
    _kernel_kupl._entry_topology = None
    _kernel_kupl._failed_topology = False
    _kernel_kupl._entry_topology_route = None
    _kernel_kupl._failed_topology_route = False
    _kernel_kupl._TOPOLOGY_DTYPE_SUPPORT.clear()
    _kernel_cuda._TOPOLOGY_DTYPE_SUPPORT.clear()
    _kernel_kupl._negative_cache_clear()
    return saved


def _restore_kernel_globals(saved: tuple[Any, ...]) -> None:
    (_kernel_kupl._entry_topology, _kernel_kupl._failed_topology,
     _kernel_kupl._entry_topology_route, _kernel_kupl._failed_topology_route,
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
    if _kernel_kupl._function_topology_route() is None:
        raise unittest.SkipTest(
            "this native KUPL build cannot report which route it took "
            "(qcd_kupl_topology_last_specialized missing), so v3 and v2 are "
            f"indistinguishable: {_kernel_kupl._REBUILD_HINT}")


def _assert_route(info: dict, label: str) -> bool | None:
    """路线标记必须与 ``path`` 自洽；CUDA 侧还没有该标记时返回 ``None``。

    ``path`` 是本契约的对外口径：``kupl_topology`` 只在**真的走了 v3 编译期步表**
    时出现，未命中步表而由 v2 通用执行器完成时写 ``kupl_topology_v2``。两者返回码
    与数值都一样，**只有**这个标记能区分，所以任何"命中"断言都必须先过这里。
    """
    if "specialized" not in info:
        return None
    specialized = info["specialized"]
    assert isinstance(specialized, bool), (label, specialized)
    expected = "kupl_topology" if specialized else "kupl_topology_v2"
    assert info.get("path") == expected, (label, info.get("path"), specialized)
    return specialized


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
        keys = _PROBE_KEYS + (_KUPL_EXTRA_PROBE_KEYS if module is _kernel_kupl else ())
        missing = [key for key in keys if key not in info]
        assert not missing, f"{path}: LAST_INFO 缺 {missing}"
        assert info["dtype"] == str(np.dtype(dtype))
        assert info["dtype_id"] == native_dtype_id(dtype)
        assert info["n_operands"] == len(PROBE_TERMS)
        assert info["steps"] >= 1
        assert tuple(info["output_shape"]) == tuple(expected.shape)
        assert info["out"] == "allocated"
        assert tuple(info["copied"]) == ()
        assert info["offsets"] is None
        if module is _kernel_kupl:
            specialized = _assert_route(info, path)
            if specialized is False:
                raise unittest.SkipTest(
                    "this native KUPL build has no compile-time step table "
                    "(topology_kernels.inc absent), so every plan falls back to v2: "
                    f"{_kernel_kupl._REBUILD_HINT}")
            # PROBE_EQUATION 是语料 seq=1，必须在 2582 条基础步表里。
            assert specialized is True, (path, info.get("specialized"))
        assert info["path"] == path, info["path"]

    if not accepted:
        raise unittest.SkipTest(
            f"{path}: 这个原生构建没有可执行 dtype（四类全部 status=4，待实现）")
    return accepted


def test_cuda_descriptor_reaches_the_step_table_for_high_arity():
    """CUDA：元数 >6 的语料拓扑必须真的查到 ``topology_steps.h`` 的步表。

    回归点（2026-11-15 修复）：``_kernel_cuda._descriptor_words`` 原先把
    ``_specialize._signature_digest`` 的 **96 位**摘要零填充成 128 位，而步表的 key
    是 blake2b-**128** 的前 8 字节。前 12 字节相同、后 4 字节不同 ⇒ 原生算出的 key
    恒查不到表，于是退到 ``topology_plan_chain`` 的「左到右链式」兜底；该兜底只接受
    ``num_inputs <= 6``，而语料 99.9% 的元数是 7..16，所以**除 arity=6 的 2 条外整张
    表都取不到**，一律 ``status=4`` 回落。

    旧用例没抓到它，是因为 ``PROBE_EQUATION`` 恰好就是 arity=6 的那条、而
    ``_sample_equations`` 的 arity 集合断言只要求 ``{6, 13}`` 是子集 —— arity=6 单独
    命中就会让"命中数 > 0"成立。这里显式钉住**高元数**：arity 必须 > 6，命中还必须
    走原生步表（``LAST_INFO['path'] == 'cuda_topology'`` 且 ``steps`` 等于表里的步数）。
    """
    _require_cuda()
    import cupy

    # arity 13/14/16 各取一条：全部超出链式兜底的 num_inputs <= 6 上限。
    wanted = []
    seen = set()
    for task in tasklist.TASKS:
        if task.seq > 2582 or task.arity <= 6 or task.arity in seen:
            continue
        seen.add(task.arity)
        wanted.append(task)
        if len(wanted) >= 3:
            break
    assert len(wanted) >= 3 and all(task.arity > 6 for task in wanted), wanted

    for task in wanted:
        dim = 2
        operands = _make_operands(
            [(dim,) * len(term)
             for term in task.equation.split("->")[0].split(",")],
            np.float64, seed=task.seq)
        expected = np.einsum(task.equation, *operands)
        devices = [cupy.asarray(item) for item in operands]
        _allow_dtype(_kernel_cuda, np.float64)
        _kernel_kupl._NEGATIVE_CACHE.clear()
        _kernel_cuda.LAST_INFO.clear()
        _kernel_cuda.LAST_FAILURE.update({"reason": "", "status": None})
        result = _kernel_cuda.try_topology_apply(task.equation, devices)
        assert result is not None, (
            f"seq={task.seq} arity={task.arity}: 高元数语料拓扑必须命中原生步表，"
            f"实际回落：{_kernel_cuda.LAST_FAILURE.get('reason')}")
        info = _kernel_cuda.LAST_INFO
        assert info.get("path") == "cuda_topology", info
        assert int(info.get("steps", 0)) == task.arity - 1, (
            f"seq={task.seq}: 步数 {info.get('steps')} 与 arity-1={task.arity - 1} 不符，"
            f"说明命中的不是这条拓扑的步表")
        error = _relative_error(cupy.asnumpy(result), expected)
        assert error <= TOLERANCES["float64"], (
            f"seq={task.seq} arity={task.arity}: rel={error:.3e}")


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

    实数档在**分派层**被 ``_TOPOLOGY_DECLINED_DTYPES`` 主动挡掉（性能选路，不是
    能力缺失，见 ``_kernel_kupl`` 模块注释），所以本用例在作用域内打开
    ``PYQCD_EINSUM_TOPOLOGY_REAL=1``——它要验的是"原生步表对四类 dtype 都正确"，
    而不是当前的选路偏置；选路本身由 ``test_dedicated_routing_contracts`` 负责。
    """
    _require_kupl()
    import os

    override = "PYQCD_EINSUM_TOPOLOGY_REAL"
    saved_override = os.environ.get(override)
    os.environ[override] = "1"
    checked = 0
    try:
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
    finally:
        if saved_override is None:
            os.environ.pop(override, None)
        else:
            os.environ[override] = saved_override
    assert checked == len(ALL_DTYPES) * 2 * 4, checked


def test_every_dtype_is_native_when_its_probe_passes():
    """四类 dtype 各自的活证据：探测通过 ⇒ 必须真的走原生拓扑入口。

    判定对四类完全一样——``LAST_INFO['path']`` 命中、dtype 保真、数值在容差内。
    没有"某一类只要不崩就算支持"的豁免。

    与 ``test_higher_dims_and_larger_extents_hold_for_all_dtypes`` 同理：实数档的
    回落是**分派策略**（``_TOPOLOGY_DECLINED_DTYPES``，性能选路），不是探测失败，
    所以本用例在作用域内打开 ``PYQCD_EINSUM_TOPOLOGY_REAL=1`` 后验"探测通过即
    下发"这条契约本身。
    """
    _require_kupl()
    import os

    override = "PYQCD_EINSUM_TOPOLOGY_REAL"
    saved_override = os.environ.get(override)
    os.environ[override] = "1"
    checked = []
    try:
        for dtype in ALL_DTYPES:
            dtype = np.dtype(dtype)
            if not _kernel_kupl._topology_dtype_enabled(dtype):
                continue                  # 这个原生构建缺该类 GEMM，跳过即可
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
            info = dict(_kernel_kupl.LAST_INFO)
            specialized = _assert_route(info, dtype.name)
            if specialized is False:
                raise unittest.SkipTest(
                    "this native KUPL build has no compile-time step table "
                    f"(topology_kernels.inc absent): {_kernel_kupl._REBUILD_HINT}")
            assert specialized is True, (dtype, info.get("specialized"))
            assert info.get("path") == "kupl_topology", (dtype, info.get("path"))
            checked.append(dtype.name)
    finally:
        if saved_override is None:
            os.environ.pop(override, None)
        else:
            os.environ[override] = saved_override

    if not checked:
        raise unittest.SkipTest("native KUPL build cannot execute any dtype")


# ``topology-specialized`` 只由 v3 步表命中时打印（``topology_apply.cpp`` 的
# ``run_specialized`` 尾部）。子进程里跑，因为 ``trace_enabled()`` 的结果在首次
# 调用后就被静态缓存，同进程内改环境变量已经晚了。
_TRACE_CHILD = """
import numpy as np
from pyqcd.einsum import _kernel_kupl as kk

equation = "cadb,menf,ac,em,bd,fn->"
for dtype in (np.complex128, np.complex64):
    rng = np.random.default_rng(11)
    operands = []
    for term in equation.split("->")[0].split(","):
        shape = (2,) * len(term)
        values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        operands.append(np.ascontiguousarray(values.astype(dtype)))
    expected = np.einsum(equation, *operands)
    got = kk.try_topology_apply(equation, operands)
    assert got is not None, (dtype, kk.LAST_FAILURE.get("reason"))
    scale = float(np.max(np.abs(expected))) or 1.0
    error = float(np.max(np.abs(got - expected))) / scale
    print("RESULT", np.dtype(dtype).name, error)
"""


def test_complex_dtypes_reach_the_v3_step_table():
    """复数必须真的走进 v3 步表（cgemm/zgemm 执行），而不是回落 v2 通用执行器。

    回归点：``topology_apply.cpp::run_specialized`` 曾把 ``elem_kind`` 只赋给
    ``QCD_DTYPE_FLOAT32`` / ``QCD_DTYPE_FLOAT64``，其余一律 ``return
    kSpecializedMiss``，于是复数一路落到 ``execute_parsed_contract``（**v2**）。
    这条回落对上层是不可见的：``LAST_INFO['path']`` 仍写 ``kupl_topology``（那是
    Python 侧的标记），数值同样正确，只是慢。唯一可区分处是原生 trace——只有
    v3 命中才打印 ``topology-specialized``，于是这里用子进程抓它，并要求两类
    复数（dtype_id 0/1）各有一条，且没有"缺 cgemm/zgemm"的 miss 行。

    复数能力缺失的构建（BLAS 不导出 cgemm/zgemm）跳过：契约是"能跑就必须走 v3"，
    不是"每台机器都必须能跑复数"。
    """
    _require_kupl()
    import os
    import pathlib
    import subprocess
    import sys

    if not _kernel_kupl.enabled():
        raise unittest.SkipTest("KUPL kernel dispatch is disabled by environment")
    if not all(_kernel_kupl._topology_dtype_enabled(np.dtype(dtype))
               for dtype in (np.complex128, np.complex64)):
        raise unittest.SkipTest(
            "native KUPL build cannot execute complex dtypes "
            "(cblas cgemm/zgemm missing), so no v3 step table can be reached")

    root = str(pathlib.Path(__file__).resolve().parents[2])
    environment = dict(os.environ)
    environment["PYQCD_EINSUM_TOPOLOGY_TRACE"] = "1"
    environment.pop("PYQCD_EINSUM_TOPOLOGY_REAL", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [root] + [item for item in [environment.get("PYTHONPATH")] if item])
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _TRACE_CHILD], cwd=root, env=environment,
        capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, (
        f"complex topology child failed: {completed.stderr[-800:]}")
    assert "RESULT complex128" in completed.stdout, completed.stdout
    assert "RESULT complex64" in completed.stdout, completed.stdout

    trace = completed.stderr
    for dtype_id, name in ((0, "complex128"), (1, "complex64")):
        assert "topology-specialized:" in trace and \
            f" dtype={dtype_id} steps=" in trace, (
            f"{name} 没有命中 v3 步表；原生 trace 只有：\n{trace}")
    assert "needs cblas cgemm/zgemm" not in trace, (
        f"复数在缺 cgemm/zgemm 的构建里不应被下发：\n{trace}")


#: 语料内抽样（基础步表 seq ≤ 2582）：必须命中 v3。
_CORPUS_CASES = (1, 2, 9)
#: 语料外的生产方程：sush 8 算子 2pt（S1）与 15 算子 3pt（P1），步表里没有它们。
_OUTSIDE_CASES = (
    "eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN",
    "agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy",
)

# 子进程里跑：``trace_enabled()`` 首次调用后就被静态缓存，同进程内改环境变量已经
# 太晚。每条方程前后打 ``CASE`` 标记（flush 过），父进程据此把原生 trace 切段，
# 从而能逐条对账"标记为 v3 的必须有 topology-specialized，标记为 v2 的必须没有"。
_ROUTE_CHILD = """
import sys

import numpy as np

from pyqcd.einsum import _kernel_kupl as kk

CASES = %(cases)r


def operands(equation, dim=2):
    rng = np.random.default_rng(3)
    values = []
    for term in equation.split("->")[0].split(","):
        shape = (dim,) * len(term)
        data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        values.append(np.ascontiguousarray(data.astype(np.complex128)))
    return values


# 预热：dtype 探测自己会命中 v3 并打一行 trace，它落在第一个 CASE 之前，
# 因此不会被算到任何一条方程头上。
kk.try_topology_apply(CASES[0][1], operands(CASES[0][1]))

for index, (kind, equation) in enumerate(CASES):
    print("CASE", index, kind, flush=True, file=sys.stderr)
    result = kk.try_topology_apply(equation, operands(equation))
    assert result is not None, (equation, kk.LAST_FAILURE)
    flag = kk.LAST_INFO.get("specialized")
    print("ROUTE", index, -1 if flag is None else int(bool(flag)),
          kk.LAST_INFO.get("path"), flush=True)
"""


def test_v2_fallback_is_reported_and_matches_the_native_trace():
    """语料内方程必须标 v3，语料外（无步表）必须标 v2，且与原生 trace 逐条一致。

    回归点（2026-11-15）：``apply_topology`` 在步表未命中时把整条计划静默交给
    ``execute_parsed_contract``（v2），返回码却仍是 0，而 Python 侧无条件写
    ``LAST_INFO['path'] = 'kupl_topology'``。于是**任何**成功执行都被报成"拓扑
    特化"：S1/P1 这类不在 2582 条步表里的生产方程也成了命中，77 格蒸馏基准里有
    52 格（seq>2582 的附录）因此把 v2 的耗时算到了 v3 头上。

    现在路线由原生 ``qcd_kupl_topology_last_specialized()`` 报告：v3 命中写
    ``kupl_topology`` / ``specialized=True``，v2 回落写 ``kupl_topology_v2`` /
    ``specialized=False``。原生只在 v3 命中时打印 ``topology-specialized:``，所以
    这个标记能与 trace 逐条对账——这是本条用例作为"保险丝"的核心。
    """
    _require_kupl()
    import os
    import pathlib
    import subprocess
    import sys

    cases = ([("corpus", tasklist.lookup(seq).equation) for seq in _CORPUS_CASES]
             + [("outside", equation) for equation in _OUTSIDE_CASES])
    assert len(cases) == len(_CORPUS_CASES) + len(_OUTSIDE_CASES), cases

    root = str(pathlib.Path(__file__).resolve().parents[2])
    environment = dict(os.environ)
    environment["PYQCD_EINSUM_TOPOLOGY_TRACE"] = "1"
    environment.pop("PYQCD_EINSUM_TOPOLOGY_REAL", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [root] + [item for item in [environment.get("PYTHONPATH")] if item])
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _ROUTE_CHILD % {"cases": cases}],
        cwd=root, env=environment, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, (
        f"route child failed: {completed.stderr[-800:]}")

    flags: dict[int, tuple[int, str]] = {}
    for line in completed.stdout.splitlines():
        if line.startswith("ROUTE "):
            _, index, flag, path = line.split()
            flags[int(index)] = (int(flag), path)
    assert len(flags) == len(cases), completed.stdout

    segments: list[list[str]] = []
    for line in completed.stderr.splitlines():
        if line.startswith("CASE "):
            segments.append([])
        elif segments:
            segments[-1].append(line)
    assert len(segments) == len(cases), completed.stderr[-1200:]

    for index, (kind, equation) in enumerate(cases):
        flag, path = flags[index]
        expected = 1 if kind == "corpus" else 0
        assert flag == expected, (
            f"{kind} 方程 {equation!r}: specialized 标记={flag}，期望 {expected}")
        assert path == ("kupl_topology" if expected else "kupl_topology_v2"), (
            f"{kind} 方程 {equation!r}: path={path!r}")
        hits = sum("topology-specialized:" in line for line in segments[index])
        assert hits == expected, (
            f"{kind} 方程 {equation!r}: 原生 trace 出现 {hits} 行 "
            f"topology-specialized，与 specialized={bool(flag)} 不一致"
            f"（v3 命中才打印这一行）：\n{''.join(segments[index])}")


# ── 7. pack-bound 守卫的流量模型：只计真正要 pack 的那一侧（2026-11-15） ────
#
# 守卫的**判据**没变（pack 流量 ≥ 1 GiB 且 FLOPs ≤ 1e9 即拒），变的是**流量模型**：
# 输入侧逐侧判直连（``v3_run_step``）、输出侧按 ``c_axes`` 是否规范序决定 unpack
# （``run_typed_v3`` 的 ``direct``）。两条都能被原生 ``PYQCD_EINSUM_TOPOLOGY_TRACE=1``
# 的逐步诊断行逐条证实，所以下面既验决策、也验"Python 侧判定 == 原生判定"。

#: 13 步全部两侧直连、输出轴序全恒等：旧守卫算 3,265,082,400 B 判死，真实 0.7 MiB。
_RELEASE_SEQ = 379
#: 真病态：末步输出 1e8 元素且 ``c_axes=[2,3,1,0]``（必须 unpack）= 3.05 GiB。
_PATHOLOGICAL_SEQ = 646


class _ShapedOperand:
    """只带 ``shape``/``dtype`` 的占位操作数：编码 plan 不需要真分配生产尺寸。"""

    def __init__(self, shape: Sequence[int], dtype: Any) -> None:
        self.shape = tuple(int(extent) for extent in shape)
        self.dtype = np.dtype(dtype)


def _production_plan(seq: int, scene: str = "distill"):
    """按语料 roles 的生产尺寸编码该 seq 的 plan（零分配）。"""
    task = tasklist.lookup(seq)
    assert task is not None, f"seq {seq} 不在语料里"
    terms, _out = parse(task.equation)
    sizes = _label_sizes(terms, task.roles, scene_dims(scene))
    shapes = [tuple(sizes[label] for label in term) for term in terms]
    dtype = np.dtype(np.complex128)
    _, info = contract_path(task.equation, *shapes, shapes=True)
    plan = encode_execution_plan(
        info.contraction_list, [_ShapedOperand(shape, dtype) for shape in shapes])
    return task.equation, plan, shapes


def test_pack_bound_guard_only_counts_the_sides_that_pack():
    """守卫逐侧计 pack：单侧直连 + 规范序输出不该被当成 GiB 级流量。

    回归点（2026-11-15）：旧守卫对**每一步的两侧**都记 2× 流量，再无条件是给最终
    输出记 2×。于是 seq=379（``distill`` 尺寸）——13 步里两侧全部恒等/组交换序可
    直连、每步输出 ``c_axes`` 全是恒等（原生不 unpack）——被算成
    3,265,082,400 B 判死，而原生真实布局流量只有 0.7 MiB。这条用例同时把同一个
    plan 喂给"旧口径"计账，证明它确实越过了 1 GiB 门槛（否则这条回归就失效了）；
    另一侧再钉死真病态 seq=646 仍必须被拒。
    """
    _require_kupl()
    equation, plan, _shapes = _production_plan(_RELEASE_SEQ)
    pathological_equation, pathological, _bad_shapes = _production_plan(_PATHOLOGICAL_SEQ)

    # 1. 单侧/双侧直连 + 规范序输出：放行。
    reason = _kernel_kupl._topology_pack_bound_reason(plan, 16)
    assert reason is None, (
        f"seq={_RELEASE_SEQ} 的每一步两侧都可直连、输出又是规范序，"
        f"不应被判 pack-bound：{reason}\n{equation}")

    # 2. 同一份 plan 用旧口径（两侧都 pack + 最终输出无条件 2×）确实越过 1 GiB：
    #    否则这条回归用例就名不副实。
    legacy_bytes = _legacy_pack_bytes(plan, 16)
    assert legacy_bytes >= _kernel_kupl._HEAVY_PACK_MIN_BYTES, (
        f"seq={_RELEASE_SEQ} 旧口径只有 {legacy_bytes} B，构不成回归用例")

    # 3. 真病态不能因为放宽而漏掉：末步输出 1e8 元素且 c_axes 非恒等。
    for itemsize, dtype in ((16, "complex128"), (8, "complex64")):
        bad = _kernel_kupl._topology_pack_bound_reason(pathological, itemsize)
        assert bad is not None, (
            f"seq={_PATHOLOGICAL_SEQ} 在 {dtype} 下末步要 unpack 1e8 元素，必须仍被拒")
        assert "pack-bound on CPU" in bad, bad
    assert pathological_equation != equation

    # 4. 放行不等于改数值：同一入口在小尺寸上仍与通用路径一致（四类 dtype 的数值
    #    矩阵由 ``test_dtype_matrix_kupl`` 覆盖，这里只钉住本用例涉及的方程）。
    terms = equation.split("->")[0].split(",")
    operands = _make_operands(
        [(DIM,) * len(term) for term in terms], np.complex64, seed=20261115)
    expected = contract(equation, *operands, specialize=False)
    actual = _quiet(lambda: _kernel_kupl.try_topology_apply(equation, operands))
    assert actual is not None, (
        f"seq={_RELEASE_SEQ} 在小尺寸下也应放行：{_kernel_kupl.LAST_FAILURE}")
    error = _relative_error(actual, expected)
    assert error <= TOLERANCES["complex64"], (
        f"seq={_RELEASE_SEQ}: 放行后数值与通用路径不一致 rel={error:.3e}")


def _legacy_pack_bytes(plan: Any, itemsize: int) -> int:
    """旧守卫的流量口径：每一步两侧都 pack，最终输出无条件再记 2×。"""
    from ..einsum._native_plan import decode_execution_plan

    decoded = decode_execution_plan(plan)
    sizes = {index: int(extent) for index, extent in enumerate(decoded.sizes)}

    def count(labels: Sequence[int]) -> int:
        total = 1
        for label in labels:
            total *= sizes[label]
        return total

    pack_bytes = 0
    for step in decoded.steps:
        if len(step.inputs) != 2:
            continue
        left, right = step.inputs
        pack_bytes += 2 * itemsize * (count(left) + count(right))
    return pack_bytes + 2 * itemsize * count(decoded.output_labels)


# 子进程里跑：原生 trace 的开关在首次调用后静态缓存，同进程内改环境变量已经太晚。
# 每条方程先打 ``CASE`` 标记与 Python 侧的逐步判定（``PYC``），随后调**原生入口**
# 触发 trace —— 标记与 trace 都写 **stderr**，父进程才能按段把两者对齐。这里不用
# ``try_topology_apply``：路径用例要在守卫会拒绝的 seq 上也能看到原生逐步行。
_LAYOUT_CHILD = """
import sys

import numpy as np

from pyqcd.einsum import _kernel_kupl as kk
from pyqcd.einsum._api import contract_path
from pyqcd.einsum._native_plan import encode_execution_plan
from pyqcd.einsum.kupl import einsum as native

CASES = __CASES__


def emit(text):
    print(text, file=sys.stderr, flush=True)


# 先跑一遍 dtype 探测：它自己会做一次原生 apply 并打出 trace，落在第一个 CASE
# 之前，因此不会被算到任何一条方程头上。
kk._topology_dtype_enabled(np.dtype(np.complex128))
entry = kk._function_topology_route()

for equation, shapes in CASES:
    rng = np.random.default_rng(11)
    operands = []
    for shape in shapes:
        extent = tuple(shape)
        values = rng.standard_normal(extent) + 1j * rng.standard_normal(extent)
        operands.append(np.ascontiguousarray(values.astype(np.complex128)))
    _, info = contract_path(equation, *operands)
    plan = encode_execution_plan(info.contraction_list, operands)
    records = kk._topology_step_records(plan)
    emit("CASE " + equation)
    for record in records:
        emit("PYC %d %d %d %d %s %s %s %d" % (
            record["batch"], record["m"], record["k"], record["n"],
            record["a_kind"], record["b_kind"], record["path"],
            int(record["output_canonical"])))
    native.topology_apply(plan, operands)
    flag = None if entry is None else kk._topology_specialized(entry)
    emit("ROUTE %d" % (-1 if flag is None else int(bool(flag))))
    del operands, plan
"""

#: 便宜且结构互补的三条基础语料（生产 ``distill`` 尺寸下 v3 分别 28 ms / 0.5 ms /
#: 0.3 ms，峰值内存 < 0.3 GiB）：它们合起来覆盖 ``path`` 的四种取值与一次真实
#: unpack（seq=855 末步 ``c_axes=[1,0]``）。**必须用生产尺寸**：v3 步表的收缩序是
#: 生成期按生产尺寸定型的，等长 dim=3 下 greedy 的次序不同（实测 seq=379 在 dim=3
#: 的原生逐步行与本 plan 的推演就不一致），那样比对的是两条不同的路径。
_LAYOUT_SEQS = (3, 855, 2)


def test_pack_bound_guard_matches_the_native_step_layout_trace():
    """守卫的逐步判定必须与原生 ``topology-gemm-step`` 诊断行逐条相同。

    守卫的账全靠 Python 侧推演，唯一的可信证据是原生在**同一条 plan** 上打出的
    ``batch=/M=/K=/N= a=plain|trans|pack b=.. path=direct|direct-a|direct-b|pack``。
    只要有一处判定与原生不一致，pack 估算就与真实流量脱节 —— 那正是"静态高估"
    的来源（旧守卫把单侧直连也计成两侧 pack）。这里逐行对账，并同时钉住路线：
    ``ROUTE 1`` 表示这些行确实来自编译期 v3 步表，不是 v2 通用执行器。
    """
    _require_kupl()
    import os
    import pathlib
    import subprocess
    import sys

    cases = []
    for seq in _LAYOUT_SEQS:
        equation, _plan, shapes = _production_plan(seq)
        cases.append((equation, [list(shape) for shape in shapes]))
    assert len({equation for equation, _ in cases}) == len(cases), cases

    root = str(pathlib.Path(__file__).resolve().parents[2])
    environment = dict(os.environ)
    environment["PYQCD_EINSUM_TOPOLOGY_TRACE"] = "1"
    environment.pop("PYQCD_EINSUM_TOPOLOGY_REAL", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [root] + [item for item in [environment.get("PYTHONPATH")] if item])
    completed = subprocess.run(
        [sys.executable, "-B", "-c",
         _LAYOUT_CHILD.replace("__CASES__", repr(cases))],
        cwd=root, env=environment, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, (
        f"layout child failed: {completed.stderr[-800:]}")

    segments: list[dict[str, Any]] = []
    for line in completed.stderr.splitlines():
        if line.startswith("CASE "):
            segments.append({"equation": line[len("CASE "):], "records": [],
                             "native": [], "route": None})
        elif line.startswith("PYC ") and segments:
            batch, m, k, n, a_kind, b_kind, path, canonical = line.split()[1:]
            segments[-1]["records"].append({
                "batch": int(batch), "m": int(m), "k": int(k), "n": int(n),
                "a_kind": a_kind, "b_kind": b_kind, "path": path,
                "output_canonical": bool(int(canonical)),
            })
        elif line.startswith("ROUTE ") and segments:
            segments[-1]["route"] = int(line.split()[1])
        elif "topology-gemm-step" in line and segments:
            segments[-1]["native"].append(line)
    assert len(segments) == len(cases), (
        f"子进程只打了 {len(segments)} 个 CASE 分段：{completed.stderr[-1200:]}")

    total = 0
    for case, segment in zip(cases, segments):
        equation = case[0]
        assert segment["equation"] == equation, (segment["equation"], equation)
        records = segment["records"]
        assert records, f"{equation}: Python 侧没有推演出任何 GEMM 步"
        assert segment["route"] == 1, (
            f"{equation}: 原生没有命中 v3 步表（route={segment['route']}），"
            f"这条用例要求的是步表执行器")
        assert len(segment["native"]) == len(records), (
            f"{equation}: 原生 {len(segment['native'])} 行逐步诊断 vs "
            f"Python 推演 {len(records)} 步（步序不同就说明 plan 与步表不是同一条"
            f"路径）：\n{chr(10).join(segment['native'])}")
        for record, line in zip(records, segment["native"]):
            fields = {}
            for token in line.split("] ", 1)[1].split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value
            wanted = {
                "batch": str(record["batch"]), "M": str(record["m"]),
                "K": str(record["k"]), "N": str(record["n"]),
                "a": record["a_kind"], "b": record["b_kind"],
                "path": record["path"],
            }
            assert fields == wanted, (
                f"{equation} 的逐步判定与原生不一致：\n  原生 {fields}\n"
                f"  Python {wanted}")
        total += len(records)
    print(f"layout trace: {len(cases)} equations, {total} steps matched")


# ── 7b. topology 侧的 pack/flops 比值判据（2026-11-15） ─────────────────────
#
# 绝对判据（>=1 GiB 且 <=1e9 FLOPs）漏掉"活干得也多"的病态计划：seq=856 在
# ``distill_curr`` 尺寸下 pack 8.28 GB、FLOPs 2.10e9，绝对判据放行，而它在 topology
# 上实测 3014 ms、通用路径 699 ms。曾为它加过比值门（阈值 2.5），但实测那批里
# 3 快 3 慢、对数口径打平，且根因是步表与运行期 plan 换序（见函数 docstring），
# 故**已撤销**：比值只由 ``_topology_pack_ratio`` 暴露给诊断，不参与拒绝。

#: 比值判据的验收 plan：``distill_curr`` 尺寸下 pack 8,281,146,368 B、
#: FLOPs 2,096,413,216、比值 3.950 —— 绝对判据要求 ``flops <= 1e9``（不满足），
#: 所以它只可能被比值那一条拒。
#: 病态样例 seq=856 的方程与生产形状（CUDA 与 topology 两侧契约共用）：
#: ``distill_curr`` 尺寸下 pack 8,281,198,592 B、FLOPs 2,096,413,216、比值 3.950。
_PACK_RATIO_PATHOLOGICAL_EQUATION = "ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak"
_PACK_RATIO_PATHOLOGICAL_SHAPES = (
    (400, 4, 1, 1), (4, 4, 9, 1), (4, 400, 400, 400), (4, 4, 9, 9),
    (4, 4), (4, 4), (4, 4), (1, 9, 400), (1, 9), (1, 9, 400))

_RATIO_PATHOLOGICAL_SEQ = 856
#: 反例一（"比值高但规模小"）：比值 12.65 远超阈值，pack 却只有 10,752 B。
_RATIO_TINY_SEQ = 2
#: 反例二（"规模大但算术密度正常"）：pack 1.03 GiB 已过绝对底，比值只有 0.028。
_RATIO_DENSE_SEQ = 2584


def test_topology_pack_ratio_is_diagnostic_only():
    """topology 侧的 ``pack/flops`` **只作诊断，不参与拒绝**（2026-11-15 实测后回退）。

    回归点：曾按 dedicated 的形态加了比值门（阈值 2.5，与 1 GiB 绝对底并列）。
    标定显示它在 ``distill`` 上零误拒、只在 ``distill_curr`` 拒 213 格；但**那批
    里好坏混杂**——同一口径的配对测（``pyqcd.einsum.test``，``--rounds 5`` 中位）：

    =========  ===============  ============  ==============
    seq        topology ms      generic ms    拒掉的效果
    =========  ===============  ============  ==============
    856        3014             699           快 4.3x
    14         3030-3224        736-780       快 4.1x
    20         2659-2713        830           快 3.3x
    9          291-317          698-706       **慢 2.3x**
    3          144-169          756-819       **慢 5.0x**
    5          138-179          792-878       **慢 5.3x**
    =========  ===============  ============  ==============

    3 快 3 慢、量级相当，属"用三格换三格"，不是净收益。

    根因不在比值：**非 ``distill`` 尺寸下编译期步表的收缩序与运行期 plan 不是同
    一条路径**（``PYQCD_EINSUM_TOPOLOGY_TRACE=1`` 实测：seq=3 原生
    ``M=9 K=400 N=640000``，而 Python 推演 ``M=4 K=640000 N=400``；seq=9/856
    同样对不上）。步表按 ``distill`` 定型，尺寸一变 ``auto`` 就换序，于是守卫在
    那个口径下量的是**另一条计划**。而在模型可信的 ``distill`` 口径上该门一格都
    不拒（3802/3802 全放行）——即它只会在不可信的口径上开火。

    因此现在：``_topology_pack_bound_reason`` 只保留绝对判据；比值由
    ``_topology_pack_ratio`` 暴露给诊断。本条同时钉住"不拒绝"与"仍能算出比值"，
    防止有人不加标定就把比值门装回去。
    """
    _require_kupl()

    # 1) 病态样例（比值 3.95）在**模型不可信**的 distill_curr 口径上也不得被拒。
    _equation, plan, _plan_shapes = _production_plan(
        _RATIO_PATHOLOGICAL_SEQ, "distill_curr")
    reason = _kernel_kupl._topology_pack_bound_reason(plan, 16)
    assert reason is None, (
        f"seq={_RATIO_PATHOLOGICAL_SEQ} 的比值门已撤销，不该再拒绝：{reason}")

    # 2) 比值仍必须能算出来（诊断价值保留）。
    ratio = _kernel_kupl._topology_pack_ratio(plan, 16)
    assert ratio is not None, "诊断用的 pack/flops 应当可算"
    assert 3.5 < ratio < 4.5, (
        f"seq={_RATIO_PATHOLOGICAL_SEQ} 的比值应约 3.95，得到 {ratio}")

    # 3) 小 plan 与正常大 plan 都不拒绝（绝对判据的门槛没被误动）。
    for seq, scene in ((9, "distill"), (_RATIO_DENSE_SEQ, "distill_curr"),
                       (2, "distill_curr")):
        _eq, other, _s = _production_plan(seq, scene)
        assert _kernel_kupl._topology_pack_bound_reason(other, 16) is None, (seq, scene)

    # 4) 绝对判据仍然有效：seq=646 是它标定的病态格，必须仍被拒。
    _eq, heavy, _s = _production_plan(646, "distill")
    absolute = _kernel_kupl._topology_pack_bound_reason(heavy, 16)
    assert absolute is not None, "绝对 pack 判据被误删了"
    assert "pack-bound on CPU" in absolute, absolute


def _cuda_dedicated_stats(equation: str, shapes: Sequence[Sequence[int]]) -> dict[str, int]:
    """按 **CUDA 口径**编码步表并取统计（``allow_heavy_branch2=False``，与
    ``_kernel_cuda.try_dedicated`` 的 ``_dedicated_cached(..., "cuda")`` 一致）。

    纯计划统计、不分配 GPU 张量，所以无设备也能跑。
    """
    cached = _kernel_kupl._cheapest_dedicated_plan(
        equation, tuple(tuple(int(extent) for extent in shape) for shape in shapes),
        0, False)
    assert cached is not None, f"{equation} 无法编码成步表"
    return _kernel_kupl._plan_statistics(cached[0])


def _scene_dedicated_stats(seq: int, scene: str = "distill_curr") -> dict[str, int]:
    """按语料 roles 的生产尺寸取该 seq 的 CUDA 口径计划统计（零分配）。"""
    task = tasklist.lookup(seq)
    assert task is not None, f"seq {seq} 不在语料里"
    terms = task.equation.split("->", 1)[0].split(",")
    sizes = _label_sizes(terms, task.roles, scene_dims(scene))
    shapes = [tuple(sizes[label] for label in term) for term in terms]
    return _cuda_dedicated_stats(task.equation, shapes)


def test_cuda_plan_gate_rejects_pack_traffic_per_flop():
    """CUDA 计划守卫：pack/flops 超限被拒，GEMM 主导与生产边界放行。

    靶子是 seq=856（``ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak``）：CUDA 计划
    ``pack_bytes=8,281,198,592`` / ``flops=2,096,413,216`` = **3.950 B/FLOP**，
    绝对判据因 ``flops > 1e9`` 放行（CPU 侧同一条计划实测 dedicated 8662.7 ms vs
    通用路径 2330 ms）。

    **与 CPU 侧的一处实测差异（不要照抄 CPU 用例的断言）**：CUDA 的 arena 底只有
    3 GiB，而该计划的 CUDA arena 是 4,219,857,152 B，所以它在本条判据加进来之前
    **已经**被第一条门（arena）拒掉了。比值判据的作用是让结论不再依赖"arena 恰好
    超了"：把 arena 与单步 MACs 归零后，只剩比值门能拒它。两条判据都独立成立。

    标定（complex128，语料 arity 6..16、``distill``/``distill_curr`` 各 8638 条）：
    旧三条门放行且 ``pack >= 1 GiB`` 的 1734 条里，``distill`` 最高 0.617、
    ``distill_curr`` 中位 1.728 / p90 1.813 / p99 2.101 / 最高 2.388。3.0 落在空档
    (2.388, 3.950) 内；把阈值降到 CPU 的 1.0 会新拒 ``distill_curr`` 的 500 条生产
    计划（pack 2.1–3.1 GB），2.0 仍新拒 30 条——而这两族里 GPU 实测过的两格都是
    **好计划**（seq=8513 比值 2.388 → ``cuda_dedicated`` 1.249 ms vs opt_einsum
    531.2 ms；seq=7900 比值 2.101 → 1.133 ms vs 338.0 ms），所以不取。
    """
    dtype = np.dtype(np.complex128)
    pathological = _cuda_dedicated_stats(
        _PACK_RATIO_PATHOLOGICAL_EQUATION, _PACK_RATIO_PATHOLOGICAL_SHAPES)
    pack_bytes = pathological["plan_pack_bytes"]
    flops = pathological["plan_flops"]
    ratio = pack_bytes / flops
    assert ratio >= _kernel_cuda._CUDA_HEAVY_PACK_RATIO_LIMIT, (
        f"seq=856 的比值 {ratio:.3f} 低于阈值 "
        f"{_kernel_cuda._CUDA_HEAVY_PACK_RATIO_LIMIT}，本条用例失去靶子：{pathological}")

    # 1. 真门今天给的是 arena 理由（实测差异见 docstring），但计划确实被拒。
    reason = _kernel_cuda._plan_fallback_reason(dtype, pathological)
    assert reason is not None, ("seq=856 的 CUDA 计划必须被拒", pathological)
    assert "arena" in reason, reason

    # 2. 归零 arena / 单步 MACs 后只剩比值门：证明该判据独立成立、不是搭便车。
    isolated = dict(pathological, plan_arena_bytes=0, plan_max_step_macs=0)
    reason = _kernel_cuda._plan_fallback_reason(dtype, isolated)
    assert reason is not None, (
        "只剩比值门时必须仍然拒绝 seq=856", isolated)
    for token in ("pack/flops", str(pack_bytes), str(flops)):
        assert token in reason, (token, reason)

    # 3. 生产边界不许误伤：distill_curr 里"旧门放行且 pack >= 1 GiB"比值最高的几格
    #    （8513/8499 = 2.388，7900 = 2.101，21 = 1.813）必须仍然放行。
    for seq in (8513, 8499, 7900, 21):
        stats = _scene_dedicated_stats(seq)
        allowed_ratio = stats["plan_pack_bytes"] / stats["plan_flops"]
        assert allowed_ratio < _kernel_cuda._CUDA_HEAVY_PACK_RATIO_LIMIT, (seq, stats)
        assert _kernel_cuda._plan_fallback_reason(dtype, stats) is None, (
            f"seq={seq}（比值 {allowed_ratio:.3f}）是生产计划，不得被新判据误拒",
            stats)

    # 4. 正常 GEMM / 小规模收缩必须放行（否则新判据会把通用收缩也打回外围路径）。
    for equation, shapes in (
            ("bij,bjk->bik", ((64, 512, 512), (64, 512, 512))),
            ("ij,jk->ik", ((1000, 1000), (1000, 1000))),
            ("am,mb->ab", ((256, 256), (256, 256)))):
        stats = _cuda_dedicated_stats(equation, shapes)
        gemm_ratio = stats["plan_pack_bytes"] / stats["plan_flops"]
        assert gemm_ratio < 0.2, (equation, stats)
        assert _kernel_cuda._plan_fallback_reason(dtype, stats) is None, (
            f"{equation} 是 GEMM 主导计划，不得被拒", stats)

    # 5. 绝对底是**并列项**而不是装饰：seq=1599（``full`` 尺寸）pack 只有 1.1 MB 但
    #    比值 17.0，没有 1 GiB 底就会被比值判据误杀（CPU 侧在语料上实测误拒 56% 的
    #    同一种病）。注意场景必须写 ``full``：同一 seq 在 ``distill_curr`` 下是小计划，
    #    比值只有 2.6，构不成反例。
    tiny = _scene_dedicated_stats(1599, "full")
    assert tiny["plan_pack_bytes"] < _kernel_cuda._CUDA_HEAVY_PACK_MIN_BYTES, tiny
    assert tiny["plan_pack_bytes"] >= (
        _kernel_cuda._CUDA_HEAVY_PACK_RATIO_LIMIT * tiny["plan_flops"]), tiny
    assert _kernel_cuda._plan_fallback_reason(dtype, tiny) is None, (
        "pack 不到 1 GiB 的小计划不得被比值判据拒", tiny)

    # 6. flops == 0 时比值无意义：只走绝对判据，且不得除零。
    zero_work = {"plan_flops": 0, "plan_pack_bytes": 0,
                 "plan_arena_bytes": 0, "plan_max_step_macs": 0}
    assert _kernel_cuda._plan_fallback_reason(dtype, zero_work) is None
    zero_work["plan_pack_bytes"] = _kernel_cuda._CUDA_HEAVY_PACK_MIN_BYTES
    reason = _kernel_cuda._plan_fallback_reason(dtype, zero_work)
    assert reason is not None and "pack-bound" in reason, reason
    assert "pack/flops" not in reason, reason


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


def test_appendix_equations_reach_the_v3_step_table():
    """附录形态（动量标签打头，seq ≥ 2583）必须命中 v3 步表，而不是回落 v2。

    回归点（2026-11-15）：编译表曾只含基础表 2582 条（``MAX_KERNELS``），附录
    6056 条命中签名后 ``run_specialized`` 返回 miss、静默走 v2。现在
    ``MAX_KERNELS = 8638``（= 整个语料），这些方程应当由 v3 执行。

    判据用 ``LAST_INFO['path']``：附录格必须是 ``kupl_topology`` 且
    ``specialized`` 为真。若有人把 ``MAX_KERNELS`` 改回 2582 并重生成步表，
    本条立刻失败 —— 这是"扩表没被退回去"的保险丝。
    """
    _require_kupl()
    from pyqcd.einsum import _bench_scene
    from pyqcd.einsum.bench_specialized import _label_sizes
    from pyqcd.testing._topology_kernel_gen import parse as parse_equation

    #: 附录前两条（arity=6，单条 <50 ms），保证用例快速、可进常规套件。
    cases = (2583, 2584)
    table = _bench_scene.dims("distill")
    checked = 0
    for seq in cases:
        task = tasklist.lookup(seq)
        assert task is not None, seq
        terms, _out = parse_equation(task.equation)
        sizes = _label_sizes(terms, task.roles, table)
        shapes = [tuple(sizes[ch] for ch in term) for term in terms]
        rng = np.random.default_rng(seq)
        operands = [
            np.ascontiguousarray(
                (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                .astype(np.complex128))
            for shape in shapes]
        _kernel_kupl.LAST_INFO.clear()
        _kernel_kupl.LAST_FAILURE.update({"reason": "", "status": None})
        result = _kernel_kupl.try_topology_apply(task.equation, operands)
        if result is None:
            raise AssertionError(
                f"seq={seq}（附录）在扩表后应命中 v3，实际回落："
                f"{_kernel_kupl.LAST_FAILURE.get('reason')}")
        path = _kernel_kupl.LAST_INFO.get("path")
        assert path == "kupl_topology", (seq, path)
        assert _kernel_kupl.LAST_INFO.get("specialized") is True, seq
        expected = np.einsum(task.equation, *operands, optimize="greedy")
        scale = float(np.max(np.abs(expected))) or 1.0
        error = float(np.max(np.abs(result - expected))) / scale
        assert error <= 1e-9, (seq, error)
        checked += 1
    assert checked == len(cases), checked


TESTS = tuple(_isolated(test) for test in (
    test_appendix_equations_reach_the_v3_step_table,
    test_dtype_contract_is_uniform_across_all_four_dtypes,
    test_cuda_descriptor_reaches_the_step_table_for_high_arity,
    test_dtype_matrix_kupl,
    test_dtype_matrix_cuda,
    test_probe_failed_dtype_is_never_narrowed_to_another,
    test_dtype_probe_contains_exceptions_and_is_cached,
    test_topology_dtype_reasons_are_distinguishable,
    test_try_topology_apply_reports_probe_failure_without_raising,
    test_wide_output_topologies_are_not_transposed,
    test_higher_dims_and_larger_extents_hold_for_all_dtypes,
    test_every_dtype_is_native_when_its_probe_passes,
    test_complex_dtypes_reach_the_v3_step_table,
    test_v2_fallback_is_reported_and_matches_the_native_trace,
    test_native_arguments_carry_plan_order_and_out_buffer,
    test_cuda_arguments_carry_plan_order_and_out_buffer,
    test_offsets_are_never_forwarded,
    test_noncontiguous_operands_are_copied_and_recorded,
    test_out_buffer_contract,
    test_allocated_output_does_not_alias_operands,
    test_unencodable_plan_falls_back_without_raising,
    test_corpus_sample_matches_generic_path,
    test_native_failure_falls_back_with_one_warning,
    test_pack_bound_guard_only_counts_the_sides_that_pack,
    test_topology_pack_ratio_is_diagnostic_only,
    test_pack_bound_guard_matches_the_native_step_layout_trace,
    test_cuda_plan_gate_rejects_pack_traffic_per_flop,
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
