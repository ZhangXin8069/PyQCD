"""特化内核-基准脚本（specialized-kernel benchmark）：15 种特化路径 vs opt_einsum。

「特化路径」指**命中** wick_12cases 语料（8638 条）的形状——按拓扑签名分派到
专用执行器（KUPL/CUDA topology kernel），不走通用分步收缩。

与 ``bench_generic`` 的计时口径完全一致（交错、中位数、每条样本约 0.3 s），
因此两个脚本的数字可以直接横向比较。

用法::

    python -m pyqcd.testing.einsum.bench_specialized            # cpu + double
    python -m pyqcd.testing.einsum.bench_specialized cpu single
    python -m pyqcd.testing.einsum.bench_specialized gpu
    python -m pyqcd.testing.einsum.bench_specialized --list     # 只列 15 条路径

15 条路径按元数与结构从 8638 条语料中挑选，覆盖 6/10/13/14/16 算子四档。
生产尺度取 nvec=100（``ACCEPTANCE_PROFILES`` 的验收档 T1）。
"""

from __future__ import annotations

import argparse
import statistics
import time
from functools import partial

import numpy as np

# 15 条特化路径：直接从 8638 条语料里取代表式，按用例小节归类。
# (标签, 方程, 是否是复数, 单指标尺寸)
CASES: tuple[tuple[str, str, bool, int], ...] = (
    # 6 算子的三种基本几何：2pt、带动量投影的 2pt、直接 2pt
    ("2pt vacuum (6 ops)", "cadb,menf,ac,em,bd,fn->", True, 40),
    ("2pt vacuum (6 ops, real)", "cadb,menf,ac,em,bd,fn->", False, 56),
    ("2pt momentum (6 ops)", "cadb,menf,ac,em,Mbd,Nfn->MN", True, 40),
    # 7 算子：3pt
    ("3pt vacuum (7 ops)", "ambn,codp,egfh,ce,mo,bdf,nph->ag", True, 24),
    ("3pt momentum (7 ops)",
     "ambn,codp,egfh,ce,mo,Mbdf,Nnph->MNag", True, 20),
    # 9 算子：带两条插入流
    ("4pt two-insert (9 ops)", "cadb,menf,gohp,ac,em,og,bd,fn,ph->", True, 24),
    # 10 算子：带张量插入
    ("4pt tensor-insert (10 ops)",
     "ambn,cgdh,eifj,okpl,ce,mo,gi,bdf,np,hjl->ak", True, 32),
    ("4pt tensor-insert (10 ops, real)",
     "ambn,cgdh,eifj,okpl,ce,mo,gi,bdf,np,hjl->ak", False, 40),
    # 12 算子：三条插入流
    ("5pt three-insert (12 ops)",
     "cadb,menf,gohp,kilj,ac,em,og,ik,bd,fn,ph,jl->", True, 20),
    # 13 算子：混合几何（两个代表用例小节）
    ("3pt full geom (13 ops)",
     "cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek", True, 28),
    ("4pt full geom (13 ops)",
     "ambn,cgdh,ekfl,oqpr,isjt,ce,mo,gi,kq,bdf,np,hj,lrt->as", True, 28),
    # 14 算子：六条传播子的完整几何
    ("5pt full geom (14 ops)",
     "aibj,ckdl,eqfr,msnt,oupv,gwhx,ce,og,ik,su,bdf,nph,jlr,tvx->aqmw",
     True, 20),
    ("5pt full geom (14 ops, real)",
     "aibj,ckdl,eqfr,msnt,oupv,gwhx,ce,og,ik,su,bdf,nph,jlr,tvx->aqmw",
     False, 28),
    # 16 算子：最深的一档
    ("6pt full geom (16 ops)",
     "cadb,egfh,mknl,ospt,iujv,qwrx,ac,mo,gi,kq,su,bd,fnp,hj,lr,tvx->ew",
     True, 16),
    ("6pt momentum (16 ops)",
     "cadb,egfh,mknl,oqpr,isjt,wuxv,ac,mo,gi,kq,uw,Nbd,Nfnp,Nhj,Mlrt,Mvx->MNes",
     True, 16),
)


def _builders(device):
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


def _operands(equation: str, size: int, builder):
    """按方程里的标签生成操作数；同一标签在所有操作数里同长。

    大写标签（``M``/``N``/``G`` 等）是 2 维流分量指标，小写是 ``size`` 维。
    """
    return [builder(*[2 if label.isupper() else size for label in term])
            for term in equation.split("->")[0].split(",")]


def _one_call(function, sync, repeat=1):
    sync()
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        sync()
        best = min(best, time.perf_counter() - started)
    return best


def _hit_kernel(equation: str, operands, sync, release) -> str:
    """返回实际执行的标签：`kernel(...)` 或 `plan-cache`，用于校验真的走了特化。"""
    import contextlib
    import io

    import pyqcd.einsum as pe

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        pe.contract(equation, *operands)
    text = buffer.getvalue()
    if "kernel:" in text:
        return text.split("kernel:", 1)[1].split("]", 1)[0].strip()
    if "plan-cache" in text:
        return "plan-cache"
    return ""


def run(device="cpu", precision="double", rounds=3, quick=False,
        cases=None) -> int:
    import opt_einsum as oe

    import pyqcd
    import pyqcd.einsum as pe

    real, complex_, sync, release, device_name, configure = _builders(device)
    configure(precision)

    rtol = 1e-5 if precision == "single" else 1e-10
    chosen = cases or CASES
    arrays = {"real": real, "complex": complex_}
    as_numpy = (lambda value: value.get()) if device == "gpu" else np.asarray

    print(f"pyqcd {pyqcd.__version__} {pyqcd.__file__}")
    print(f"特化内核-基准脚本 | {device_name} | {precision} | {len(chosen)} 条路径 "
          f"| rounds={rounds}")
    print(f"{'case':28s} {'pyqcd(s)':>9s} {'opt(s)':>9s} {'ratio':>7s}  path")
    print("-" * 74)

    ratios: list[float] = []
    for label, equation, is_complex, size in chosen:
        builder = arrays["complex" if is_complex else "real"]
        operands = _operands(equation, size, builder)

        reference = oe.contract(equation, *operands, optimize="dp")
        got = pe.contract(equation, *operands)
        ours_value = as_numpy(got)
        their_value = as_numpy(reference)
        magnitude = max(float(np.max(np.abs(their_value), initial=0.0)), 1e-30)
        deviation = float(np.max(np.abs(ours_value - their_value), initial=0.0))
        correct = deviation <= rtol * magnitude
        del reference, got, ours_value, their_value
        release()

        executed = _hit_kernel(equation, operands, sync, release)

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
        status = executed if correct else "WRONG"
        print(f"{label:28s} {median_ours:9.4f} {median_theirs:9.4f} "
              f"{ratio:6.2f}x  {status}")
        del operands
        release()

    if ratios:
        print("-" * 74)
        print(f"中位比值 {statistics.median(ratios):.2f}x  "
              f"（<1 表示 pyqcd 更快；{len(ratios)} 条路径）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="特化内核-基准脚本：15 种 wick_12cases 特化路径 vs opt_einsum")
    parser.add_argument("device", nargs="?", default="cpu",
                        choices=("cpu", "gpu"))
    parser.add_argument("precision", nargs="?", default="double",
                        choices=("single", "double"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="只跑前 4 条，便于冒烟")
    parser.add_argument("--list", action="store_true", help="只列路径")
    args = parser.parse_args(argv)

    if args.list:
        for index, (label, equation, _, size) in enumerate(CASES, 1):
            print(f"{index:2d}. {label:28s} {equation}")
        return 0

    cases = CASES[:4] if args.quick else None
    return run(device=args.device, precision=args.precision,
               rounds=args.rounds, quick=args.quick, cases=cases)


if __name__ == "__main__":
    raise SystemExit(main())
