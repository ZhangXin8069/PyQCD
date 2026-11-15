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
        raise SystemExit(
            "error: --equation 必须带显式 '->'（隐式输出会与 --shape 的输出段冲突）")
    lhs, rhs = equation.split("->", 1)
    terms = lhs.split(",")
    if len(terms) != len(shapes):
        raise SystemExit(
            f"error: --equation 有 {len(terms)} 个操作数，--shape 给了 {len(shapes)} 个")
    sizes: dict[str, int] = {}
    for term, shape in zip(terms, shapes):
        if len(term) != len(shape):
            raise SystemExit(
                f"error: 操作数 {term!r} 的秩是 {len(term)}，形状 {shape} 的秩是 "
                f"{len(shape)}")
        for label, extent in zip(term, shape):
            if extent <= 0:
                raise SystemExit(f"error: 操作数 {term!r} 的轴长必须为正，得到 {extent}")
            known = sizes.setdefault(label, extent)
            if known != extent:
                raise SystemExit(
                    f"error: 标签 {label!r} 在不同操作数里长度不一致"
                    f"（{known} vs {extent}）——本工具不做广播猜测")
    for label in rhs:
        if label not in sizes:
            raise SystemExit(f"error: 输出标签 {label!r} 未在任何输入里出现")


# ── 设备与张量 ──────────────────────────────────────────────────────────────
def _xp_for_device(device: str):
    """返回该设备对应的数组模块（numpy 或 cupy）。"""
    if device == "cpu":
        return np
    try:
        import cupy
    except Exception as exc:                    # noqa: BLE001
        raise SystemExit(
            f"error: --device {device!r} 需要 cupy，但导入失败：{type(exc).__name__}: "
            f"{exc}\n  提示: --device cpu 可在无 GPU 时运行") from exc
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:                    # noqa: BLE001
        raise SystemExit(
            f"error: cupy 无法枚举 CUDA 设备：{type(exc).__name__}: {exc}") from exc
    if count == 0:
        raise SystemExit(
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
        raise SystemExit(f"error: --dtype 必须是 {DTYPES} 之一，得到 {dtype!r}")
    if compare not in COMPARE_MODES:
        raise SystemExit(f"error: --compare 必须是 {COMPARE_MODES} 之一，得到 {compare!r}")
    if warmup < 1:
        raise SystemExit(
            "error: --warmup 不能为 0：首调包含 plan 缓存/原生库绑定/JIT，"
            "不丢弃会让耗时完全失真")
    if rounds < 1:
        raise SystemExit("error: --rounds 必须 ≥ 1")
    if expected_shape is not None and tuple(expected_shape) != tuple(out_shape):
        raise SystemExit(
            f"error: 方程输出与 --shape 的输出段不一致：{tuple(out_shape)} vs "
            f"{tuple(expected_shape)}")

    _validate_against_equation(equation, shapes)
    xp = _xp_for_device(device)
    np_dtype = np.dtype(dtype)
    operands, host = _make_operands(xp, shapes, np_dtype, seed)

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
    if "rel" in report:
        print(f"rel      : {report['rel']:.3e}  (atol={report['atol']:g})", file=out)
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
            raise SystemExit(
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
    print(f"PASS {args.equation}{detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
