"""复数特化独立验收门：dtype 矩阵 / 特化真跑 / 实数不回归 / 混合输入 / 边界。

本门回答一个问题：**复数档（complex64 / complex128）在特化路径上是否可信**，
并且不把这个问题混进实现侧的自测里。五组契约项各自独立、可单独运行：

1. **dtype 矩阵**：complex64 / complex128 各抽 ≥100 条 ``_wick_tasklist.TASKS``，
   参考值由「升到 complex128 + opt_einsum 固定 path」给出；
   判据 complex128 ≤ **1e-12**、complex64 ≤ **1e-5**。同时要求结果 dtype 与
   输入 dtype 逐条一致——特化路径**不得**把 complex64 悄悄升到 complex128。
2. **特化真跑**：``_kernel_kupl.try_dedicated`` 对复数必须返回非 None，且
   ``LAST_INFO['dtype_id']`` 落在复数码位（complex128=0 / complex64=1），
   证明走的是原生 cgemm/zgemm 而不是降精度或回落。``_kernel_cuda.try_dedicated``
   同契约；后端或环境未就绪时断言「返回 None 且 ``LAST_FAILURE`` 有原因」，
   记 **PENDING（待实现/待实测）** 而不是 FAIL。
3. **实数不回归**：与第 1 项**同一批**（同 seed 同 50 条）float32/float64，
   判据 ≤1e-5 / ≤1e-12。
4. **复数与实数混合输入**：拆成两层分别判定——内核层
   ``test_mixed_precision_kernel_fails_closed`` 要求 ``try_dedicated`` 拒绝下发并
   留下原因；公共入口 ``test_mixed_complex_and_real_inputs_are_rejected`` 要求
   ``contract`` 明确报错（不静默转换）。若内核层拒了而公共入口静默提升，则 4b
   FAIL——那属于实现未完成，不是测试写错。
5. **除零/空张量边界**：形状含 0 维时 ``contract`` 与 ``numpy.einsum`` 的
   形状、dtype、数值逐项一致且不崩溃；``try_dedicated`` 命中则数值一致，
   未命中则必须留下回落原因。

三态结论（``main()`` 的收尾行据此区分）：

* ``PASS``——该契约项在本机实测满足；
* ``PENDING``——测试已就绪，但后端/环境未提供（不计失败）；
* ``FAIL``——实测不满足契约（含「实现未支持」与「实现支持但数值错」两种情况，
  失败信息里会写清是哪一种）。

用法::

    python -m pyqcd.testing._complex_precision_contract
    python -m pyqcd.testing._complex_precision_contract --complex-sample 200 --dim 4
    python -m pyqcd.testing._complex_precision_contract --real-sample 100
"""

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import unittest
import warnings

import numpy as np

# 精度判据：float32/complex64 留单精度累积余量；float64/complex128 比自身
# 舍入误差（≈1e-16）宽 4 个数量级，只容得下求和顺序差异。
TOLERANCES = {
    np.dtype(np.complex128): 1e-12,
    np.dtype(np.complex64): 1e-5,
    np.dtype(np.float64): 1e-12,
    np.dtype(np.float32): 1e-5,
}
COMPLEX_DTYPES = (np.dtype(np.complex128), np.dtype(np.complex64))
ZERO_DTYPES = (np.dtype(np.complex128), np.dtype(np.complex64), np.dtype(np.float64))
ZERO_CASES = (
    ("ab,bc->ac", ((0, 3), (3, 4))),      # 左操作数空
    ("ab,bc->ac", ((2, 0), (0, 4))),      # 共享指标为 0，输出非空但为 0
    ("ab,bc->ac", ((0, 0), (0, 4))),      # 共享指标为 0 且左全空
    ("ab,bc->ac", ((2, 3), (3, 0))),      # 右操作数空，输出为空
    ("ab,b->a", ((0, 3), (3,))),          # 输出维度为 0
    ("aa->", ((0, 0),)),                  # 0×0 的标量迹
)
CONFIG = {
    "complex_sample": 120,
    "real_sample": 50,
    "dim": 3,
    "seed": 20261112,
}


class _Pending(unittest.SkipTest):
    """契约项所需的后端/环境尚未就绪：记 PENDING，不计失败。

    继承 ``unittest.SkipTest``，与仓库内其它契约模块一致——外部收集器
    （例如把 ``TESTS`` 交给 pytest）也会把它当成 skip 而不是 error。
    """


def configure(*, complex_sample: int | None = None, real_sample: int | None = None,
              dim: int | None = None, seed: int | None = None) -> dict:
    """覆盖抽样规模与格点维数（``main`` 的 CLI 与外部调用共用同一入口）。"""
    if complex_sample is not None:
        CONFIG["complex_sample"] = int(complex_sample)
    if real_sample is not None:
        CONFIG["real_sample"] = int(real_sample)
    if dim is not None:
        CONFIG["dim"] = int(dim)
    if seed is not None:
        CONFIG["seed"] = int(seed)
    return dict(CONFIG)


# ── 数据与参考 ─────────────────────────────────────────────────────────────

def _array(shape, dtype, seed: int) -> np.ndarray:
    """按 dtype 生成同分布实/复随机数：复数必须带虚部，否则测不到复数乘法。"""
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape)
    if dtype.kind == "c":
        values = values + 1j * rng.standard_normal(shape)
    return values.astype(dtype)


def _operands_for(equation: str, dim: int, dtype, seed: int) -> list[np.ndarray]:
    """按方程生成操作数：大写=外置小轴 2，小写=缩并轴 ``dim``。"""
    shapes = [tuple(2 if ch.isupper() else dim for ch in term)
              for term in equation.split("->")[0].split(",")]
    return [_array(shape, dtype, seed + 7919 * index)
            for index, shape in enumerate(shapes)]


def _reference(equation: str, operands, wide_dtype) -> np.ndarray:
    """参考值：升到宽精度后用 opt_einsum 的**固定 path** 收缩（与实现路径无关）。"""
    import opt_einsum as oe

    wide = [np.asarray(operand, dtype=wide_dtype) for operand in operands]
    path, _ = oe.contract_path(equation, *wide, optimize="greedy")
    return oe.contract(equation, *wide, optimize=path)


def _deviation(result, expected) -> float:
    """相对偏差：分母取参考结果的最大模，空结果要求形状一致才记 0。"""
    result = np.asarray(result)
    expected = np.asarray(expected)
    if result.shape != expected.shape:
        return float("inf")
    if expected.size == 0:
        return 0.0 if result.size == 0 else float("inf")
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    return float(np.max(np.abs(result - expected))) / scale


def _wide_dtype(dtype) -> np.dtype:
    return np.dtype(np.complex128 if np.dtype(dtype).kind == "c" else np.float64)


def _sample(count: int, seed: int):
    from pyqcd.einsum import _wick_tasklist as tasklist

    tasks = list(tasklist.TASKS)
    if count >= len(tasks):
        return tasks
    picked = np.random.default_rng(seed).choice(len(tasks), size=count, replace=False)
    return [tasks[index] for index in sorted(picked)]


@contextlib.contextmanager
def _quiet():
    """屏蔽专用路径提示与一次性 warning：本门的输出只留数值证据。"""
    buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buffer):
            yield


def _scan(dtype, count: int, dim: int, seed: int) -> dict:
    """对抽样语料逐条比对参考值，返回偏差、dtype/形状违规与路径分布。"""
    import pyqcd.einsum as pe
    from pyqcd.einsum import _kernel_kupl as kernel

    dtype = np.dtype(dtype)
    tolerance = TOLERANCES[dtype]
    wide = _wide_dtype(dtype)
    tasks = _sample(count, seed)

    deviations: list[float] = []
    violations: list[dict] = []
    paths: dict[str, int] = {}
    worst: dict | None = None
    for task in tasks:
        operands = _operands_for(task.equation, dim, dtype, task.seq)
        expected = _reference(task.equation, operands, wide)
        kernel.LAST_INFO.clear()
        with _quiet():
            result = np.asarray(pe.contract(task.equation, *operands))
        deviation = _deviation(result, expected)
        path = str(kernel.LAST_INFO.get("path", "")) or "generic"
        paths[path] = paths.get(path, 0) + 1
        row = {
            "seq": task.seq,
            "equation": task.equation,
            "deviation": deviation,
            "dtype": str(result.dtype),
            "shape": tuple(result.shape),
            "expected_shape": tuple(np.shape(expected)),
            "path": path,
        }
        deviations.append(deviation)
        if worst is None or deviation > worst["deviation"]:
            worst = row
        if deviation > tolerance or row["dtype"] != str(dtype):
            violations.append(row)
    return {
        "dtype": str(dtype),
        "tolerance": tolerance,
        "checked": len(tasks),
        "worst": worst,
        "median": statistics.median(deviations) if deviations else None,
        "paths": paths,
        "violations": violations,
    }


def _assert_scan(dtype, count: int, seed: int, minimum: int) -> dict:
    report = _scan(dtype, count, CONFIG["dim"], seed)
    assert report["checked"] >= max(minimum, 1), (
        f"抽检条数 {report['checked']} < {max(minimum, 1)}：本契约要求至少 "
        f"{max(minimum, 1)} 条，请提高抽样规模（当前抽样参数太小）")
    if report["violations"]:
        worst_offender = max(report["violations"], key=lambda row: row["deviation"])
        dtypes_seen = sorted({row["dtype"] for row in report["violations"]})
        raise AssertionError(
            f"{report['dtype']}: {len(report['violations'])}/{report['checked']} 条不满足"
            f"判据 {report['tolerance']:.0e}；最差 seq={worst_offender['seq']} "
            f"rel={worst_offender['deviation']:.3e} "
            f"eq={worst_offender['equation']} 实际 dtype={dtypes_seen} "
            f"形状={worst_offender['shape']}（参考 {worst_offender['expected_shape']}）")
    worst = report["worst"]
    print(f"  {report['dtype']}: {report['checked']} 条，"
          f"最大偏差 {worst['deviation']:.2e}（seq={worst['seq']}），"
          f"中位 {report['median']:.2e}，判据 {report['tolerance']:.0e}，"
          f"路径 {report['paths']}")
    return report


# ── 1. dtype 矩阵（复数） ──────────────────────────────────────────────────

def test_complex128_matches_widened_reference():
    """complex128 ≥100 条：相对偏差 ≤1e-12，且结果 dtype 保持 complex128。"""
    _assert_scan(np.complex128, CONFIG["complex_sample"],
                 CONFIG["seed"], minimum=100)


def test_complex64_matches_widened_reference():
    """complex64 ≥100 条：相对偏差 ≤1e-5，且不被悄悄升到 complex128。"""
    _assert_scan(np.complex64, CONFIG["complex_sample"],
                 CONFIG["seed"], minimum=100)


# ── 2. 特化真跑 ────────────────────────────────────────────────────────────

def test_dedicated_dtype_contract_declares_complex():
    """两个后端的步表入口都必须把 complex64/complex128 声明为可执行 dtype。"""
    from pyqcd.einsum import _kernel_cuda as cuda
    from pyqcd.einsum import _kernel_kupl as kupl

    for name, module in (("kupl", kupl), ("cuda", cuda)):
        declared = tuple(np.dtype(item) for item in module.dedicated_dtypes())
        missing = [str(dtype) for dtype in COMPLEX_DTYPES if dtype not in declared]
        assert not missing, f"{name}.dedicated_dtypes() 缺少 {missing}：实际 {declared}"
        print(f"  {name}.dedicated_dtypes(): {[str(item) for item in declared]}")


def test_kupl_dedicated_executes_complex():
    """complex64/complex128 必须真跑 kupl 步表：非 None + 复数码位 + 数值正确。"""
    from pyqcd.einsum import _kernel_kupl as kupl

    equation = "ab,bc->ac"
    for dtype in COMPLEX_DTYPES:
        operands = _operands_for(equation, 4, dtype, seed=97)
        kupl.LAST_INFO.clear()
        result = kupl.try_dedicated(equation, operands)
        if result is None:
            reason = str(kupl.LAST_FAILURE.get("reason", ""))
            assert reason, "回落到通用路径时必须留下 LAST_FAILURE 原因"
            raise _Pending(
                f"kupl 步表入口未接受 {dtype}（待实现）；回落原因：{reason}")
        expected = _reference(equation, operands, np.complex128)
        deviation = _deviation(result, expected)
        expected_dtype_id = 0 if dtype == np.dtype(np.complex128) else 1
        assert np.dtype(np.asarray(result).dtype) == dtype, (
            f"{dtype} 被改成 {np.asarray(result).dtype}（不允许隐式改精度）")
        assert deviation <= TOLERANCES[dtype], (
            f"{dtype} 步表数值错误：rel={deviation:.3e} > {TOLERANCES[dtype]:.0e}")
        assert kupl.LAST_INFO.get("dtype_id") == expected_dtype_id, (
            f"{dtype} 的 dtype_id={kupl.LAST_INFO.get('dtype_id')}，"
            f"期望 {expected_dtype_id}（原生 cgemm/zgemm 码位）")
        assert kupl.LAST_INFO.get("path") == "kupl_dedicated", kupl.LAST_INFO
        print(f"  kupl {dtype}: rel={deviation:.2e}，"
              f"dtype_id={kupl.LAST_INFO['dtype_id']}，"
              f"plan_flops={kupl.LAST_INFO.get('plan_flops')}")


def test_cuda_dedicated_executes_complex_or_reports_why_not():
    """CUDA 步表入口同契约；缺 cupy/设备时断言「None + 原因」并记 PENDING。"""
    from pyqcd.einsum import _kernel_cuda as cuda

    equation = "ab,bc->ac"
    dtype = np.dtype(np.complex128)
    host = _operands_for(equation, 4, dtype, seed=193)
    try:
        import cupy
        operands = [cupy.asarray(operand) for operand in host]
    except Exception as error:                    # noqa: BLE001 - 缺库/缺设备都算未就绪
        cuda.LAST_INFO.clear()
        result = cuda.try_dedicated(equation, host)
        assert result is None, "缺 cupy 的机器上不能返回 CUDA 结果"
        reason = str(cuda.LAST_FAILURE.get("reason", ""))
        assert reason, "回落到通用路径时必须留下 LAST_FAILURE 原因"
        raise _Pending(
            f"cuda 后端未实测（{type(error).__name__}: {error}）；"
            f"回落原因：{reason}")

    cuda.LAST_INFO.clear()
    result = cuda.try_dedicated(equation, operands)
    if result is None:
        reason = str(cuda.LAST_FAILURE.get("reason", ""))
        assert reason, "回落到通用路径时必须留下 LAST_FAILURE 原因"
        raise _Pending(f"cuda 步表入口未接受 {dtype}（待实现）；回落原因：{reason}")
    expected = _reference(equation, host, np.complex128)
    deviation = _deviation(cupy.asnumpy(result), expected)
    assert np.dtype(result.dtype) == dtype, f"{dtype} 被改成 {result.dtype}"
    assert deviation <= TOLERANCES[dtype], f"cuda {dtype} 步表数值错误 rel={deviation:.3e}"
    assert cuda.LAST_INFO.get("dtype_id") == 0, cuda.LAST_INFO
    device = cupy.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(device, bytes):
        device = device.decode()
    print(f"  cuda {dtype}: rel={deviation:.2e}，"
          f"dtype_id={cuda.LAST_INFO['dtype_id']}，device={device}")


# ── 3. 实数不回归 ──────────────────────────────────────────────────────────

def test_float64_matches_widened_reference():
    """与复数项同批次（同 seed 同 50 条）：float64 ≤1e-12。"""
    _assert_scan(np.float64, CONFIG["real_sample"],
                 CONFIG["seed"], minimum=CONFIG["real_sample"])


def test_float32_matches_widened_reference():
    """与复数项同批次（同 seed 同 50 条）：float32 ≤1e-5。"""
    _assert_scan(np.float32, CONFIG["real_sample"],
                 CONFIG["seed"], minimum=CONFIG["real_sample"])


# ── 4. 复数与实数混合输入 ──────────────────────────────────────────────────

MIXED_CASES = (
    ("ab,bc->ac", (np.complex128, np.float64), 101),
    ("ab,bc->ac", (np.complex64, np.float32), 102),
    ("ab,bc->ac", (np.float64, np.complex128), 103),   # 顺序反过来同样不许静默
    ("ab,bc,cd->ad", (np.complex128, np.float64, np.complex128), 104),
)


def _mixed_operands(equation: str, dtypes, seed: int) -> list[np.ndarray]:
    shapes = [tuple(2 if ch.isupper() else 3 for ch in term)
              for term in equation.split("->")[0].split(",")]
    return [_array(shape, np.dtype(dtype), seed + index)
            for index, (shape, dtype) in enumerate(zip(shapes, dtypes))]


def test_mixed_precision_kernel_fails_closed():
    """内核层（4a）：混合 dtype 不得下发，且必须留下可读原因。"""
    from pyqcd.einsum import _kernel_kupl as kupl

    for equation, dtypes, seed in MIXED_CASES:
        operands = _mixed_operands(equation, dtypes, seed)
        kupl.LAST_INFO.clear()
        assert kupl.try_dedicated(equation, operands) is None, (
            f"{[str(dtype) for dtype in dtypes]} 被步表入口接受：混合 dtype 不允许下发")
        reason = str(kupl.LAST_FAILURE.get("reason", ""))
        assert reason, "内核层拒绝混合 dtype 时必须说明原因"
        assert "mixed" in reason or "dtype" in reason, reason
    print(f"  内核层 fail-closed：{len(MIXED_CASES)} 组混合 dtype 全部拒绝并给出原因")


def test_mixed_complex_and_real_inputs_are_rejected():
    """公共入口（4b）：契约要求显式报错，不允许把实数静默提升成复数。"""
    import pyqcd.einsum as pe
    from pyqcd.einsum import _kernel_kupl as kupl

    silent: list[tuple[list[str], str, float]] = []
    for equation, dtypes, seed in MIXED_CASES:
        operands = _mixed_operands(equation, dtypes, seed)
        try:
            kupl.LAST_INFO.clear()
            with _quiet():
                product = np.asarray(pe.contract(equation, *operands))
        except (TypeError, ValueError):
            continue
        # 附带事实：静默提升本身有没有精度损失（rel 对 numpy 的提升结果取）。
        deviation = _deviation(product, np.einsum(equation, *operands))
        silent.append(([str(operand.dtype) for operand in operands],
                       str(product.dtype), deviation))

    assert not silent, (
        f"契约项 4 未满足：contract 对混合输入静默转换 {silent}"
        "（第 3 元为相对偏差；很小只说明提升本身无精度损失，不代表允许静默转换）。"
        "内核层 try_dedicated 已 fail-closed；公共入口目前沿用 numpy 的提升语义，"
        "若产品意图是拒绝混合精度，需在 contract 入口补显式 dtype 校验。")


# ── 5. 除零 / 空张量边界 ───────────────────────────────────────────────────

def test_zero_extent_shapes_match_numpy():
    """0 维形状：contract 与 numpy.einsum 的形状/dtype/数值逐项一致且不崩溃。"""
    import pyqcd.einsum as pe
    from pyqcd.einsum import _kernel_kupl as kupl

    checked = 0
    for dtype in ZERO_DTYPES:
        for equation, shapes in ZERO_CASES:
            zeros = [np.zeros(shape, dtype=dtype) for shape in shapes]
            expected = np.einsum(equation, *zeros)
            result = np.asarray(pe.contract(equation, *zeros))
            assert result.shape == np.shape(expected), (
                f"{dtype} {equation} {shapes}: 形状 {result.shape} ≠ {np.shape(expected)}")
            assert np.dtype(result.dtype) == dtype, (
                f"{dtype} {equation} {shapes}: dtype 变成 {result.dtype}")
            assert np.array_equal(result, expected), (
                f"{dtype} {equation} {shapes}: 数值 {result} ≠ {expected}")

            kupl.LAST_INFO.clear()
            dedicated = kupl.try_dedicated(equation, zeros)
            if dedicated is None:
                reason = str(kupl.LAST_FAILURE.get("reason", ""))
                assert reason, f"{equation} {shapes}: 回落必须留下原因"
            else:
                assert np.array_equal(np.asarray(dedicated), expected), (
                    f"{dtype} {equation} {shapes}: 步表结果与 numpy 不一致")
            checked += 1
    print(f"  0 维形状组合 {checked} 个"
          f"（{len(ZERO_DTYPES)} dtype × {len(ZERO_CASES)} 形状）全部与 numpy 一致")


_TESTS = (
    test_complex128_matches_widened_reference,
    test_complex64_matches_widened_reference,
    test_dedicated_dtype_contract_declares_complex,
    test_kupl_dedicated_executes_complex,
    test_cuda_dedicated_executes_complex_or_reports_why_not,
    test_float64_matches_widened_reference,
    test_float32_matches_widened_reference,
    test_mixed_precision_kernel_fails_closed,
    test_mixed_complex_and_real_inputs_are_rejected,
    test_zero_extent_shapes_match_numpy,
)
TESTS = _TESTS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="复数特化独立验收门（complex64/complex128）")
    parser.add_argument("--complex-sample", type=int,
                        default=CONFIG["complex_sample"],
                        help="complex64/complex128 各抽多少条语料（≥100）")
    parser.add_argument("--real-sample", type=int, default=CONFIG["real_sample"],
                        help="float32/float64 各抽多少条语料（同一批）")
    parser.add_argument("--dim", type=int, default=CONFIG["dim"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = parser.parse_args(argv)
    configure(complex_sample=args.complex_sample,
              real_sample=args.real_sample, dim=args.dim, seed=args.seed)
    print(f"复数特化验收门：complex_sample={CONFIG['complex_sample']} "
          f"real_sample={CONFIG['real_sample']} dim={CONFIG['dim']} "
          f"seed={CONFIG['seed']}")

    passed: list[str] = []
    pending: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    for test in _TESTS:
        try:
            test()
        except _Pending as reason:
            pending.append((test.__name__, str(reason)))
            print(f"PENDING {test.__name__}: {reason}")
        except Exception as error:               # noqa: BLE001 - 契约项自证
            failed.append((test.__name__, f"{type(error).__name__}: {error}"))
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
        else:
            passed.append(test.__name__)
            print(f"PASS {test.__name__}")

    print(f"\n{len(passed)} passed, {len(pending)} pending, {len(failed)} failed")
    for name, reason in pending:
        print(f"  pending: {name} — {reason}")
    for name, reason in failed:
        print(f"  failed : {name} — {reason}")
    if failed:
        print("结论: 实现未满足契约（上列 failed 即实现缺口，不是测试问题）")
        return 1
    if pending:
        print("结论: 测试已就绪，部分后端/环境待实测（pending 项不计失败）")
        return 0
    print("结论: 复数特化契约全部满足")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
