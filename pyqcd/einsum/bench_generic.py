"""通用内核-基准脚本（generic-kernel benchmark）：15 种通用路径 vs opt_einsum。

「通用路径」指**不命中** wick_12cases 特化语料的形状——走的是通用分步收缩
（计划 + BLAS/逐元素），也是 `docs/*/performance.md` 里那两张表的口径。

用法::

    python -m pyqcd.einsum.bench_generic              # cpu + double
    python -m pyqcd.einsum.bench_generic cpu single   # 单精度更快
    python -m pyqcd.einsum.bench_generic gpu          # GPU（需 cupy）
    python -m pyqcd.einsum.bench_generic cpu double --quick
    python -m pyqcd.einsum.bench_generic --list       # 只列 15 条路径

计时口径（与特化脚本完全一致，便于横向比较）：
  1. 先对 opt_einsum 校验数值，错得再快也没意义；
  2. 交错计时（每轮交换先后顺序），机器漂移在比值里相消；
  3. 先取中位数再相除，单次抖动无法主导结论；
  4. 每条样本约 0.3 s，避免 µs 级假数据。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from functools import partial

import numpy as np

# 15 条通用路径。形状**全部由生产参数推出**（见 :mod:`pyqcd.einsum._bench_scene`），
# 不再用 4096²/12000² 这类与 QCD 无关的合成尺寸。
#
# 边长取 ``Nx * Nev``：一个传播子 ``(Lt, Nt, Ns, Ns, Nev, Nev)`` 按「空间 × 本征模」
# 展平后的规模，也就是生产里最常见的矩阵尺寸。
#
# (标签, 方程, 是否是复数, 形状键)
CASES: tuple[tuple[str, str, bool, tuple[str, ...]], ...] = (
    ("matmul 2D", "ij,jk->ik", False, ("matrix", "matrix")),
    ("matmul 2D (complex)", "ij,jk->ik", True, ("matrix", "matrix")),
    ("element-wise", "ij,ij->ij", False, ("matrix", "matrix")),
    ("outer product", "i,j->ij", False, ("vector", "vector")),
    ("full reduction", "ij,ij->", False, ("matrix", "matrix")),
    ("partial reduction", "ij->i", False, ("matrix",)),
    ("chain 3 ops", "ij,jk,kl->il", False, ("matrix", "matrix", "matrix")),
    ("chain 4 ops", "ij,jk,kl,lm->im", False,
     ("matrix", "matrix", "matrix", "matrix")),
    ("hyper-edge", "ij,jk,ik->i", False, ("matrix", "matrix", "matrix")),
    ("batched GEMM", "bij,bjk->bik", False, ("batch", "batch")),
    ("ellipsis", "...ij,...jk->...ik", False, ("batch", "batch")),
    ("batch x matrix", "abc,cd->abd", False, ("batch_matrix",)),
    ("many small batches", "bij,bjk->bik", False, ("many_batch", "many_batch")),
    ("trace", "ii->", False, ("matrix",)),
    ("diagonal", "ii->i", False, ("matrix",)),
)


def case_shapes(scene: str = "scene") -> list[tuple[str, str, bool, tuple]]:
    """把形状键解析成具体尺寸；--list 与 run 共用同一套解析。"""
    from ._bench_scene import generic_shapes

    table = generic_shapes(scene)
    resolved = []
    for label, equation, is_complex, keys in CASES:
        shapes = []
        for key in keys:
            value = table[key]
            # 一个键可能给出**多个**操作数（如 batch_matrix 是 (A, B) 一对）。
            if isinstance(value[0], tuple):
                shapes.extend(value)
            else:
                shapes.append(value)
        resolved.append((label, equation, is_complex, tuple(shapes)))
    return resolved


def _builders(device):
    """返回 (real, complex) 两个构造器；GPU 时搬到设备上。"""
    rng = np.random.default_rng(0)

    if device == "gpu":
        import cupy as cp

        def to_device(array):
            return cp.asarray(array)

        def sync():
            cp.cuda.Device(0).synchronize()

        def release():
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        def to_device(array):
            return array

        def sync():
            return None

        def release():
            return None

        name = "cpu"

    state = {"real": np.float64, "complex": np.complex128}

    def real(*shape):
        return to_device(rng.standard_normal(shape).astype(state["real"]))

    def complex_(*shape):
        real_part = rng.standard_normal(shape)
        imag_part = rng.standard_normal(shape)
        return to_device((real_part + 1j * imag_part).astype(state["complex"]))

    def configure(precision):
        single = precision == "single"
        state["real"] = np.float32 if single else np.float64
        state["complex"] = np.complex64 if single else np.complex128

    return real, complex_, sync, release, name, configure


def _one_call(function, sync, repeat=1):
    """调用 repeat 次取最快；repeat>1 摊薄固定开销。"""
    sync()
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        sync()
        best = min(best, time.perf_counter() - started)
    return best


def run(device="cpu", precision="double", rounds=3, quick=False,
        cases=None, scene: str = "scene") -> int:
    # opt_einsum 只是对照基准，不是运行时依赖：缺了要给出可操作提示，
    # 而不是抛 ModuleNotFoundError（--list 不需要它，仍然可用）。
    try:
        import opt_einsum as oe
    except ImportError:
        raise SystemExit(
            "benchmark needs opt_einsum as the reference implementation:\n"
            "  pip install opt_einsum\n"
            "(--list works without it.)")

    import pyqcd
    import pyqcd.einsum as pe

    real, complex_, sync, release, device_name, configure = _builders(device)
    configure(precision)

    from ._bench_scene import describe

    rtol = 1e-5 if precision == "single" else 1e-10
    chosen = cases or case_shapes(scene)

    arrays = {"real": real, "complex": complex_}
    as_numpy = (lambda value: value.get()) if device == "gpu" else np.asarray

    print(f"pyqcd {pyqcd.__version__} {pyqcd.__file__}")
    print(f"通用内核-基准脚本 | {device_name} | {precision} | {len(chosen)} 条路径 "
          f"| rounds={rounds}")
    print(describe(scene))
    print(f"{'case':26s} {'pyqcd(s)':>9s} {'opt(s)':>9s} {'ratio':>7s}  ok")
    print("-" * 62)

    ratios: list[float] = []
    for label, equation, is_complex, shapes in chosen:
        builder = arrays["complex" if is_complex else "real"]
        operands = [builder(*shape) for shape in shapes]

        reference = oe.contract(equation, *operands, optimize="dp")
        got = pe.contract(equation, *operands)
        ours_value = as_numpy(got)
        their_value = as_numpy(reference)
        magnitude = max(float(np.max(np.abs(their_value), initial=0.0)), 1e-30)
        deviation = float(np.max(np.abs(ours_value - their_value), initial=0.0))
        correct = deviation <= rtol * magnitude
        del reference, got, ours_value, their_value
        release()

        ours_call = partial(pe.contract, equation, *operands)
        theirs_call = partial(oe.contract, equation, *operands, optimize="dp")
        inner = max(1, min(9, int(0.3 / max(_one_call(ours_call, sync), 1e-6))))

        ours_times, theirs_times = [], []
        for round_index in range(rounds):
            if round_index % 2 == 0:
                ours_times.append(_one_call(ours_call, sync, inner))
                theirs_times.append(_one_call(theirs_call, sync, inner))
            else:
                theirs_times.append(_one_call(theirs_call, sync, inner))
                ours_times.append(_one_call(ours_call, sync, inner))
            release()

        median_ours = statistics.median(ours_times)
        median_theirs = statistics.median(theirs_times)
        ratio = median_ours / median_theirs if median_theirs else float("inf")
        ratios.append(ratio)
        print(f"{label:26s} {median_ours:9.4f} {median_theirs:9.4f} "
              f"{ratio:6.2f}x  {'y' if correct else 'N'}")
        del operands
        release()

    if ratios:
        print("-" * 62)
        print(f"中位比值 {statistics.median(ratios):.2f}x  "
              f"（<1 表示 pyqcd 更快；{len(ratios)} 条路径）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="通用内核-基准脚本：15 种通用路径 vs opt_einsum")
    parser.add_argument("device", nargs="?", default="cpu",
                        choices=("cpu", "gpu"))
    parser.add_argument("precision", nargs="?", default="double",
                        choices=("single", "double"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="只跑前 4 条，便于冒烟")
    parser.add_argument("--scene", default="scene",
                        choices=("full", "scene", "smoke"),
                        help="生产场景档位（默认 scene；full 需约 31 GB 内存）")
    parser.add_argument("--list", action="store_true", help="只列路径")
    args = parser.parse_args(argv)

    if args.list:
        from ._bench_scene import describe

        print(describe(args.scene))
        print()
        for index, (label, equation, _, shapes) in enumerate(
                case_shapes(args.scene), 1):
            print(f"{index:2d}. {label:22s} {equation:24s} {shapes}")
        return 0

    cases = case_shapes(args.scene)[:4] if args.quick else None
    return run(device=args.device, precision=args.precision,
               rounds=args.rounds, quick=args.quick, cases=cases,
               scene=args.scene)


if __name__ == "__main__":
    raise SystemExit(main())
