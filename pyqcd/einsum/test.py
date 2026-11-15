"""``qcd-einsum`` 的单方程快捷验证入口（**日后默认的测试规范端**）。

为什么单独给一个入口
--------------------
``pyqcd.testing`` 里的套件回答"整库是否健康"，而日常排查需要的是"**这一条方程、
这组形状、这个 dtype、这块设备，算得对不对、多快**"。过去这类验证靠临时脚本，
口径（warmup、容差、参照实现）每次都不一样，结论无法互相比对——本模块把口径
钉死成一处，任何一次验证都可复现、可对照。

调用样式
--------
::

    # 你给的标准形态
    python -m pyqcd.einsum.test --device gpu --dtype complex128 \\
        --equation 'ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak' \\
        --shape '[400,4,1,1],[4,4,9,1],[4,400,400,400],[4,4,9,9],[4,4],[4,4],[4,4],[1,9,400],[1,9],[1,9,400]->[400,400]' \\
        --atol 1e-12 --backend cupy --verbose true --compare opt_einsum

    # 不写 --shape 时按语料里的 roles 反推生产尺寸（seq 自动匹配）
    python -m pyqcd.einsum.test --equation 'cadb,menf,ac,em,bd,fn->' --scene distill

    # 看全部选项与缺省值
    python -m pyqcd.einsum.test --help

口径约定（改这里就是改规范，不要在临时脚本里另立一套）
------------------------------------------------------
* **容差**：``--atol`` 是**相对**容差（按 ``max|expected|`` 归一），不是逐元素
  ``np.allclose`` 的绝对阈值——后者对接近零的元素会误判。缺省 ``1e-12``。
* **计时**：先丢弃一次（plan 缓存 + 原生库绑定 + JIT），再取 ``--rounds`` 次中位。
  单次测量会被首调开销完全支配（实测 seq=2583 首调 45 ms vs 热态 0.56 ms），
  所以 **warmup 不可关闭时不给 0**：``--warmup 0`` 会被拒绝。
* **参照**：``--compare`` 决定与谁比。``numpy`` 用 ``np.einsum(optimize="greedy")``；
  ``opt_einsum`` 用 ``oe.contract(optimize=...)``；``backend`` 用同一批张量上的
  **另一个后端**（CPU↔GPU 互证）；``none`` 只跑不比对。
* **形状**：``--shape`` 与 ``--equation`` 的操作数个数必须一致，且同一标签在各
  操作数里长度必须相同；不符直接报错，**不做广播猜测**。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np

__all__ = ["main", "run", "parse_shape_spec", "DTYPES", "COMPARE_MODES", "DEVICES"]

class UsageError(SystemExit):
    """参数或环境错误。

    继承 ``SystemExit`` 以便顶层直接终止，但把退出码固定成 **2**，与 ``--help`` 里
    写的"0 通过 / 1 数值或形状不符 / 2 参数或环境错误"一致。裸
    ``SystemExit(str)`` 的退出码是 1，会与"数值失败"混淆——脚本据此判断是
    "结果不对"还是"命令写错了"。
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = 2


#: 四类 dtype 平权（与 ``_kernel_kupl.TOPOLOGY_DTYPES`` 同集合）。
DTYPES = ("complex128", "complex64", "float64", "float32")
#: ``--device``：决定数组落在哪里。``cpu``→numpy，``gpu``/``cuda``→cupy。
DEVICES = ("cpu", "gpu", "cuda")
#: ``--backend``：原样传给 ``contract(backend=...)``。``auto`` 让库自己判。
BACKENDS = ("auto", "numpy", "cupy", "torch", "kupl", "cuda")
#: ``--compare``：参照实现。
COMPARE_MODES = ("numpy", "opt_einsum", "backend", "none")

DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "complex128"
DEFAULT_BACKEND = "auto"
DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 0.0
DEFAULT_ROUNDS = 5
DEFAULT_WARMUP = 1
DEFAULT_COMPARE = "numpy"
DEFAULT_OPTIMIZE = "auto"
DEFAULT_SCENE = "distill"
DEFAULT_VERBOSE = False


# ── 形状解析 ────────────────────────────────────────────────────────────────
def parse_shape_spec(spec: str) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    """把 ``'[..],[..]->[..]'`` 解析成 (输入形状列表, 输出形状)。

    只接受你给的这种纯数字形态；任何解析失败都**直接报错**，不猜。
    也允许省略 ``->…``（此时输出形状为 ``()``，表示整式归约）。
    """
    text = spec.strip().replace(" ", "")
    if not text:
        raise ValueError("--shape 不能为空")
    if "->" in text:
        left, right = text.split("->", 1)
    else:
        left, right = text, ""
    inputs = _parse_bracket_list(left, "--shape 的输入部分")
    if right:
        outputs = _parse_bracket_list(right, "--shape 的输出部分")
        if len(outputs) != 1:
            raise ValueError(f"--shape 输出必须恰好一个方括号，得到 {len(outputs)}")
        out_shape = outputs[0]
    else:
        out_shape = ()
    return inputs, out_shape


def _parse_bracket_list(text: str, where: str) -> list[tuple[int, ...]]:
    """解析 ``[1,2],[3,4]`` 形式的方括号列表。"""
    shapes: list[tuple[int, ...]] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "[":
            if depth != 0:
                raise ValueError(f"{where}: 出现嵌套的 '[', {text!r}")
            depth = 1
            current = []
        elif char == "]":
            if depth != 1:
                raise ValueError(f"{where}: 未配对的 ']', {text!r}")
            depth = 0
            body = "".join(current).strip()
            if body == "":
                shapes.append(())
            else:
                try:
                    shapes.append(tuple(int(item) for item in body.split(",")))
                except ValueError as exc:
                    raise ValueError(
                        f"{where}: 非整数的轴长 {body!r}") from exc
        elif char == "," and depth == 0:
            continue
        else:
            if depth != 1:
                raise ValueError(f"{where}: 方括号外的多余字符 {char!r}")
            current.append(char)
    if depth != 0:
        raise ValueError(f"{where}: 未闭合的 '[', {text!r}")
    if not shapes:
        raise ValueError(f"{where}: 没有解析到任何形状, {text!r}")
    return shapes


def _validate_against_equation(equation: str, shapes: Sequence[tuple[int, ...]]) -> None:
    """操作数个数、秩、同标签等长——任一不符直接报错。"""
    if "->" not in equation:
        raise UsageError(
            "error: --equation 必须带显式 '->'（隐式输出会与 --shape 的输出段冲突）")
    lhs, rhs = equation.split("->", 1)
    terms = lhs.split(",")
    if len(terms) != len(shapes):
        raise UsageError(
            f"error: --equation 有 {len(terms)} 个操作数，--shape 给了 {len(shapes)} 个")
    sizes: dict[str, int] = {}
    for term, shape in zip(terms, shapes):
        if len(term) != len(shape):
            raise UsageError(
                f"error: 操作数 {term!r} 的秩是 {len(term)}，形状 {shape} 的秩是 "
                f"{len(shape)}")
        for label, extent in zip(term, shape):
            if extent <= 0:
                raise UsageError(f"error: 操作数 {term!r} 的轴长必须为正，得到 {extent}")
            known = sizes.setdefault(label, extent)
            if known != extent:
                raise UsageError(
                    f"error: 标签 {label!r} 在不同操作数里长度不一致"
                    f"（{known} vs {extent}）——本工具不做广播猜测")
    for label in rhs:
        if label not in sizes:
            raise UsageError(f"error: 输出标签 {label!r} 未在任何输入里出现")


# ── 设备与张量 ──────────────────────────────────────────────────────────────
def _xp_for_device(device: str):
    """返回该设备对应的数组模块（numpy 或 cupy）。"""
    if device == "cpu":
        return np
    try:
        import cupy
    except Exception as exc:                    # noqa: BLE001
        raise UsageError(
            f"error: --device {device!r} 需要 cupy，但导入失败：{type(exc).__name__}: "
            f"{exc}\n  提示: --device cpu 可在无 GPU 时运行") from exc
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:                    # noqa: BLE001
        raise UsageError(
            f"error: cupy 无法枚举 CUDA 设备：{type(exc).__name__}: {exc}") from exc
    if count == 0:
        raise UsageError(
            "error: --device gpu 但本进程看不到任何 CUDA 设备"
            "（检查 CUDA_VISIBLE_DEVICES / 驱动）")
    return cupy


def _make_operands(xp, shapes, dtype: np.dtype, seed: int):
    """按固定 seed 生成操作数（同一 seed 下各路径看到同一批数据）。"""
    rng = np.random.default_rng(seed)
    host = []
    for shape in shapes:
        if dtype.kind == "c":
            real = rng.standard_normal(shape)
            imag = rng.standard_normal(shape)
            host.append((real + 1j * imag).astype(dtype))
        else:
            host.append(rng.standard_normal(shape).astype(dtype))
    if xp is np:
        return host, host
    return [xp.asarray(item) for item in host], host


# ── 计时 ────────────────────────────────────────────────────────────────────
def _timed(call, warmup: int, rounds: int) -> float:
    """丢弃 ``warmup`` 次后取 ``rounds`` 次中位（秒）。"""
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _relative_error(actual, expected) -> float:
    """按 ``max|expected|`` 归一的相对误差（与仓内其他基准同口径）。"""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float("inf")
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    if scale == 0.0:
        scale = 1.0
    return float(np.max(np.abs(actual - expected))) / scale


# ── 运行 ────────────────────────────────────────────────────────────────────
def run(*, equation: str, shapes: Sequence[tuple[int, ...]], out_shape: tuple[int, ...],
        device: str = DEFAULT_DEVICE, dtype: str = DEFAULT_DTYPE,
        backend: str = DEFAULT_BACKEND, atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL, compare: str = DEFAULT_COMPARE,
        optimize: Any = DEFAULT_OPTIMIZE, rounds: int = DEFAULT_ROUNDS,
        warmup: int = DEFAULT_WARMUP, seed: int = 0, verbose: bool = DEFAULT_VERBOSE,
        expected_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    """跑一条方程的完整检查，返回结构化结果（供 CLI 与外部脚本消费）。"""
    import pyqcd.einsum as pe

    if dtype not in DTYPES:
        raise UsageError(f"error: --dtype 必须是 {DTYPES} 之一，得到 {dtype!r}")
    if compare not in COMPARE_MODES:
        raise UsageError(f"error: --compare 必须是 {COMPARE_MODES} 之一，得到 {compare!r}")
    if warmup < 1:
        raise UsageError(
            "error: --warmup 不能为 0：首调包含 plan 缓存/原生库绑定/JIT，"
            "不丢弃会让耗时完全失真")
    if rounds < 1:
        raise UsageError("error: --rounds 必须 ≥ 1")
    if expected_shape is not None and tuple(expected_shape) != tuple(out_shape):
        raise UsageError(
            f"error: 方程输出与 --shape 的输出段不一致：{tuple(out_shape)} vs "
            f"{tuple(expected_shape)}")

    _validate_against_equation(equation, shapes)
    xp = _xp_for_device(device)
    np_dtype = np.dtype(dtype)
    # 构造前先拍一次快照，用来算"本次运行新增了多少内存/显存"。
    memory_before = _memory_snapshot(xp)
    operands, host = _make_operands(xp, shapes, np_dtype, seed)
    memory_loaded = _memory_snapshot(xp)

    report: dict[str, Any] = {
        "equation": equation,
        "shapes": [list(item) for item in shapes],
        "out_shape": list(out_shape),
        "device": device,
        "dtype": dtype,
        "backend": backend,
        "compare": compare,
        "atol": atol,
        "rounds": rounds,
        "warmup": warmup,
        "optimize": str(optimize),
        "n_operands": len(shapes),
        "input_elements": int(sum(int(np.prod(item, dtype=np.int64)) for item in shapes)),
        "output_elements": int(np.prod(out_shape, dtype=np.int64)) if out_shape else 1,
    }

    # 主路径
    def call():
        return pe.contract(equation, *operands, backend=backend, optimize=optimize)

    try:
        got = call()
    except Exception as exc:                    # noqa: BLE001
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    seconds = _timed(call, warmup, rounds)
    got_host = got if xp is np else _to_host(got)
    report["seconds"] = seconds
    report["actual_shape"] = list(np.shape(got_host))
    report["kernel_info"] = _kernel_info()
    report["install"] = _install_info()
    report["memory"] = {
        "inputs_delta": _memory_delta(memory_before, memory_loaded),
        "peak": _memory_delta(memory_before, _memory_snapshot(xp)),
    }

    # 参照
    if compare != "none":
        reference, ref_seconds, ref_note = _reference(
            compare, equation, host, operands, out_shape, np_dtype, xp,
            optimize, rounds, warmup, device)
        report["reference"] = ref_note
        report["reference_seconds"] = ref_seconds
        if reference is None:
            report["ok"] = False
            report["error"] = f"参照实现 {compare} 不可用：{ref_note}"
            return report
        report["rel"] = _relative_error(got_host, reference)
        report["max_abs_diff"] = float(np.max(
            np.abs(np.asarray(got_host) - np.asarray(reference)))) if np.size(got_host) else 0.0
        report["ok"] = bool(
            report["actual_shape"] == list(out_shape)
            and report["rel"] <= atol + rtol * report["rel"])
        if report["rel"] != report["rel"]:      # NaN
            report["ok"] = False
    else:
        report["ok"] = bool(report["actual_shape"] == list(out_shape))

    if verbose:
        _print_detail(report)
    return report


def _to_host(value):
    """把 cupy/torch 张量转回 numpy（只用于比对与打印）。"""
    module = type(value).__module__
    if module.startswith("cupy"):
        return value.get()
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _kernel_info() -> dict[str, Any]:
    """读 KUPL/CUDA 入口最近一次调用的路线信息（不触发加载）。"""
    info: dict[str, Any] = {}
    from pyqcd.einsum import _kernel_cuda, _kernel_kupl

    for name, module in (("kupl", _kernel_kupl), ("cuda", _kernel_cuda)):
        last = getattr(module, "LAST_INFO", None)
        if last:
            info[name] = {key: last[key] for key in
                          ("path", "specialized", "steps", "dtype", "copied")
                          if key in last}
    return info


def _reference(compare, equation, host, operands, out_shape, dtype, xp,
               optimize, rounds, warmup, device):
    """按 ``--compare`` 取参照结果，返回 ``(数组或 None, 秒数或 None, 说明)``。"""
    if compare == "numpy":
        def call():
            return np.einsum(equation, *host, optimize="greedy")
        return call(), _timed(call, warmup, rounds), "np.einsum(optimize='greedy')"
    if compare == "opt_einsum":
        try:
            import opt_einsum as oe
        except Exception as exc:                # noqa: BLE001
            return None, None, f"opt_einsum 导入失败：{type(exc).__name__}"
        path = "greedy" if optimize in (None, "auto") else optimize
        if not isinstance(path, str):
            path = "greedy"

        def call():
            return oe.contract(equation, *host, optimize=path)
        try:
            seconds = _timed(call, warmup, rounds)
            return call(), seconds, f"opt_einsum.contract(optimize={path!r})"
        except Exception as exc:                # noqa: BLE001
            return None, None, f"opt_einsum 执行失败：{type(exc).__name__}: {exc}"
    if compare == "backend":
        other = "numpy" if xp is not np else ("cupy" if device == "cpu" else "numpy")
        try:
            other_xp = np if other == "numpy" else _xp_for_device("gpu")
        except SystemExit as exc:
            return None, None, f"对方后端不可用：{exc}"
        other_ops = (host if other == "numpy"
                     else [other_xp.asarray(item) for item in host])

        def call():
            return pe_contract(equation, other_ops, other)
        try:
            seconds = _timed(call, warmup, rounds)
            result = _to_host(call())
            return result, seconds, f"pyqcd.contract(backend={other!r}, {other})"
        except Exception as exc:                # noqa: BLE001
            return None, None, f"对方后端执行失败：{type(exc).__name__}: {exc}"
    return None, None, f"未知 compare 模式 {compare!r}"


def pe_contract(equation, operands, backend):
    """``compare='backend'`` 用：在指定后端上跑同一批张量。"""
    import pyqcd.einsum as pe
    return pe.contract(equation, *operands, backend=backend)


def _install_info() -> dict[str, Any]:
    """报告**实际被加载的** pyqcd 版本与包路径。

    为什么需要它：``python -m pyqcd.einsum.test`` 用哪一个 pyqcd 由 ``sys.path``
    决定，而 conda 环境里可能同时存在 ``site-packages/pyqcd``（旧版本普通安装）
    与仓库源码。用户实际踩过：base 环境装的是 pyqcd 0.1.3，在仓库目录外运行
    ``-m pyqcd.einsum.test`` 直接 ``No module named``。版本与路径打到 verbose 里，
    这类问题一眼可见。
    """
    info: dict[str, Any] = {}
    try:
        import pyqcd
        info["version"] = getattr(pyqcd, "__version__", "?")
        info["package"] = pyqcd.__file__
    except Exception as exc:                    # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _memory_snapshot(xp) -> dict[str, Any]:
    """采集进程内存与显存占用。

    只在**计时循环之外**调用（操作数构造后、收尾时各一次），因此不进入任何
    measured region，不污染 ``--rounds`` 的中位数。三项都是 O(1) 查询或一次
    ``/proc`` 读取，开销在微秒量级：

    * ``rss_mib``：进程常驻集（``/proc/self/status`` 的 VmRSS，不受 allocator
      缓存假象影响）；
    * ``gpu_used_mib``：cupy 显存池**已分配**总量（``memory_pool().used_bytes()``）；
    * ``gpu_total_mib``：该设备总显存。
    """
    info: dict[str, Any] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    info["rss_mib"] = round(int(line.split()[1]) / 1024.0, 1)
                    break
    except OSError:
        pass
    if xp is not np:
        try:
            import cupy
            pool = cupy.get_default_memory_pool()
            info["gpu_used_mib"] = round(pool.used_bytes() / 2 ** 20, 1)
            info["gpu_reserved_mib"] = round(pool.total_bytes() / 2 ** 20, 1)
            free, total = cupy.cuda.runtime.memGetInfo()
            info["gpu_total_mib"] = round(total / 2 ** 20, 1)
            info["gpu_free_mib"] = round(free / 2 ** 20, 1)
        except Exception as exc:                # noqa: BLE001
            info["gpu_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _memory_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """前后快照相减，得到本次运行的内存占用增量，并保留绝对值。

    为什么两者都要：小规模张量（如 256×256 = 1 MiB）的 RSS 增量可能是 **0**——
    Python 进程的 arena 早已持有足够内存，分配器不需要向 OS 再要一页，``VmRSS``
    就不会涨。只看增量会误以为"没占内存"。绝对值 ``*_after`` 让读者仍能看到规模。
    """
    delta: dict[str, Any] = {}
    for key in ("rss_mib", "gpu_used_mib"):
        if before.get(key) is not None and after.get(key) is not None:
            delta[key] = round(after[key] - before[key], 1)
            delta[f"{key}_after"] = after[key]
    for key in ("gpu_total_mib", "gpu_free_mib"):
        if after.get(key) is not None:
            delta[key] = after[key]
    return delta


def _print_detail(report: dict[str, Any]) -> None:
    """``--verbose true`` 时的人类可读明细。"""
    out = sys.stderr
    print(f"equation : {report['equation']}", file=out)
    print(f"device   : {report['device']}  dtype={report['dtype']}  "
          f"backend={report['backend']}", file=out)
    print(f"operands : {report['n_operands']} 个，共 "
          f"{report['input_elements']:,} 元素；输出 {report['out_shape']} "
          f"({report['output_elements']:,} 元素)", file=out)
    if "seconds" in report:
        print(f"time     : {report['seconds'] * 1000:.3f} ms "
              f"（warmup={report['warmup']}，rounds={report['rounds']}，中位）", file=out)
    if report.get("reference"):
        print(f"reference: {report['reference']}", file=out)
    ref_seconds = report.get("reference_seconds")
    own_seconds = report.get("seconds")
    if ref_seconds is not None:
        ratio = ""
        if own_seconds:
            ratio = f"  （pyqcd/参照 = {own_seconds / ref_seconds:.2f}×）"
        print(f"ref time : {ref_seconds * 1000:.3f} ms{ratio}", file=out)
    if "rel" in report:
        print(f"rel      : {report['rel']:.3e}  (atol={report['atol']:g})", file=out)
    memory = report.get("memory") or {}
    inputs_delta = memory.get("inputs_delta") or {}
    peak = memory.get("peak") or {}
    parts = []
    if inputs_delta.get("rss_mib") is not None:
        parts.append(f"输入张量 +{inputs_delta['rss_mib']:.0f} MiB RSS")
    if inputs_delta.get("gpu_used_mib") is not None:
        parts.append(f"+{inputs_delta['gpu_used_mib']:.0f} MiB 显存")
    if parts:
        print(f"memory   : {'，'.join(parts)}", file=out)
    parts = []
    if peak.get("rss_mib") is not None:
        after = peak.get("rss_mib_after")
        suffix = f"（进程共 {after:.0f} MiB）" if after is not None else ""
        parts.append(f"RSS +{peak['rss_mib']:.0f} MiB{suffix}")
    if peak.get("gpu_used_mib") is not None:
        parts.append(f"显存 +{peak['gpu_used_mib']:.0f} MiB")
    if peak.get("gpu_reserved_mib") is not None:
        parts.append(f"（池保留 {peak['gpu_reserved_mib']:.0f} MiB）")
    if peak.get("gpu_total_mib") is not None:
        parts.append(f"总显存 {peak['gpu_total_mib']:.0f} MiB")
    if parts:
        print(f"peak     : {'，'.join(parts)}", file=out)
    install = report.get("install") or {}
    if install.get("version"):
        print(f"pyqcd    : {install['version']}  {install.get('package', '?')}",
              file=out)
    if report.get("kernel_info"):
        print(f"kernel   : {report['kernel_info']}", file=out)


# ── 语料反推（不写 --shape 时） ──────────────────────────────────────────────
def shapes_from_corpus(equation: str, scene: str) -> tuple[list[tuple[int, ...]], bool]:
    """按语料 ``roles`` + 生产 dims 反推形状；返回 (形状, 是否真的是生产口径)。"""
    from pyqcd.einsum import _bench_scene
    from pyqcd.einsum import _wick_tasklist as tasklist
    from pyqcd.einsum.bench_specialized import _label_sizes
    from pyqcd.testing._topology_kernel_gen import parse

    table = _bench_scene.dims(scene)
    for seq in range(1, len(tasklist.by_seq()) + 1):
        task = tasklist.lookup(seq)
        if task is None or task.equation != equation:
            continue
        terms, _out = parse(equation)
        sizes = _label_sizes(terms, task.roles, table)
        shapes = [tuple(sizes[ch] for ch in term) for term in terms]
        return shapes, True
    return [], False


# ── CLI ─────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。缺省值全部写在这里，``--help`` 即规范。"""
    parser = argparse.ArgumentParser(
        prog="python -m pyqcd.einsum.test",
        description=(
            "qcd-einsum 单方程快捷验证（日后默认的测试规范端）。\n"
            "给方程 + 形状 + dtype + 设备，输出：是否在容差内、耗时、走的哪条原生路线。"),
        epilog=(
            "示例:\n"
            "  python -m pyqcd.einsum.test --device gpu --dtype complex128 \\\n"
            "      --equation 'ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak' \\\n"
            "      --shape '[400,4,1,1],[4,4,9,1],[4,400,400,400],[4,4,9,9],[4,4],[4,4],"
            "[4,4],[1,9,400],[1,9],[1,9,400]->[400,400]' \\\n"
            "      --atol 1e-12 --backend cupy --verbose true --compare opt_einsum\n\n"
            "  # 不写 --shape 时按语料 roles 反推生产尺寸\n"
            "  python -m pyqcd.einsum.test --equation 'cadb,menf,ac,em,bd,fn->' "
            "--scene distill\n\n"
            "口径: --atol 是**相对**容差（按 max|expected| 归一）；耗时先丢 --warmup "
            "次再取 --rounds 次中位，warmup 不允许 0。\n"
            "输出: 默认一行 PASS 里带 pyqcd 耗时、参照耗时、比值与内存增量；"
            "--verbose true 另给逐项明细（含 ref time / memory / peak / pyqcd 来源）。\n"
            "内存采样只在计时循环之外（构造前/构造后/收尾），不影响计时。\n"
            "退出码: 0 通过 / 1 数值或形状不符 / 2 参数或环境错误。\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--equation", required=True,
                        help="Einstein 方程，必须带显式 '->'")
    parser.add_argument("--shape", default=None,
                        help=("形状，形如 '[a..],[b..],...->[c..]'。省略时按 --scene "
                              "从语料 roles 反推生产尺寸（语料里没有该方程则报错）"))
    parser.add_argument("--scene", default=DEFAULT_SCENE,
                        help=(f"省略 --shape 时的尺寸档（distill / distill_curr / "
                              f"scene / full），缺省 {DEFAULT_SCENE}"))
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=DEVICES,
                        help=f"数组落在哪：cpu→numpy，gpu/cuda→cupy。缺省 {DEFAULT_DEVICE}")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=DTYPES,
                        help=f"四类平权，缺省 {DEFAULT_DTYPE}")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=BACKENDS,
                        help=(f"传给 contract(backend=...)。auto 让库自判；"
                              f"numpy/cupy/torch 指定数组库；kupl/cuda 指定原生入口。"
                              f"缺省 {DEFAULT_BACKEND}"))
    parser.add_argument("--compare", default=DEFAULT_COMPARE, choices=COMPARE_MODES,
                        help=("参照实现：numpy=np.einsum(greedy)、"
                              "opt_einsum=oe.contract、backend=另一后端互证、"
                              f"none=只跑不比。缺省 {DEFAULT_COMPARE}"))
    parser.add_argument("--optimize", default=DEFAULT_OPTIMIZE,
                        help=(f"收缩序优化器（auto/optimal/dp/greedy/branch-*，"
                              f"或显式 path）。缺省 {DEFAULT_OPTIMIZE}"))
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL,
                        help=f"**相对**容差（按 max|expected| 归一）。缺省 {DEFAULT_ATOL:g}")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                        help=f"附加的相对余量。缺省 {DEFAULT_RTOL:g}")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"计时轮数（取中位）。缺省 {DEFAULT_ROUNDS}")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"丢弃的预热次数（不可为 0）。缺省 {DEFAULT_WARMUP}")
    parser.add_argument("--seed", type=int, default=0,
                        help="操作数随机种子（各路径看到同一批数据）。缺省 0")
    parser.add_argument("--verbose", default="false",
                        help="true 时打印逐项明细到 stderr。缺省 false")
    parser.add_argument("--json", default=None,
                        help="把完整结果写到该 JSON 路径")
    return parser


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.shape is None:
        shapes, matched = shapes_from_corpus(args.equation, args.scene)
        if not matched:
            raise UsageError(
                "error: 省略 --shape 时方程必须在语料里（用于按 roles 反推生产尺寸），"
                f"但 {args.equation!r} 不在 {args.scene} 语料中。请显式给 --shape。")
        _, rhs = args.equation.split("->", 1)
        report_note = f"形状由语料 roles 反推（scene={args.scene}）"
        out_shape = None
    else:
        shapes, declared_out = parse_shape_spec(args.shape)
        report_note = "形状来自 --shape"
        out_shape = declared_out

    result = run(
        equation=args.equation, shapes=shapes,
        out_shape=out_shape if out_shape is not None else (),
        device=args.device, dtype=args.dtype, backend=args.backend,
        atol=args.atol, rtol=args.rtol, compare=args.compare,
        optimize=args.optimize, rounds=args.rounds, warmup=args.warmup,
        seed=args.seed, verbose=_as_bool(args.verbose),
    )
    result["shape_source"] = report_note

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, ensure_ascii=False)
        print(f"wrote {args.json}")

    if not result.get("ok"):
        print(f"FAIL {args.equation}")
        if result.get("error"):
            print(f"  error: {result['error']}")
        if "rel" in result:
            print(f"  rel={result['rel']:.3e} > atol={args.atol:g}")
            print(f"  actual shape={result.get('actual_shape')} "
                  f"expected={result['out_shape']}")
        return 1
    detail = ""
    if "rel" in result:
        detail = f" rel={result['rel']:.2e} vs {result['reference']}"
    if "seconds" in result:
        detail += f"  {result['seconds'] * 1000:.3f}ms"
        ref_seconds = result.get("reference_seconds")
        if ref_seconds:
            detail += f" (ref {ref_seconds * 1000:.3f}ms, {result['seconds'] / ref_seconds:.2f}x)"
        peak = (result.get("memory") or {}).get("peak") or {}
        if peak.get("gpu_used_mib") is not None:
            detail += f"  vram+{peak['gpu_used_mib']:.0f}MiB"
        elif peak.get("rss_mib") is not None:
            detail += f"  rss+{peak['rss_mib']:.0f}MiB"
    print(f"PASS {args.equation}{detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
