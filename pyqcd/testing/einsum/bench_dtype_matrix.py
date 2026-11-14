"""四类 dtype × 三条执行路径的耗时矩阵（KUPL / CUDA）。

回答的问题只有一个：**同一条语料方程、同一次规划下，某一类 dtype 走特化路径是
赚还是赔**。因此每条路径都在同一循环里交替计时（A/B/A/B…），取中位数，并用
``_plan_statistics`` 的 FLOPs 换算有效 GFLOP/s——只看毫秒无法区分"算术慢"与
"带宽慢"。

三条路径：

``generic``    ``contract(..., specialize=False)``，Python 分步 + BLAS；
``dedicated``  Python 展开的固定 GEMM 步表（``_kernel_*.try_dedicated``）；
``topology``   原生按拓扑查表（KUPL 为 ``topology_kernels.inc`` 步表，CUDA 为
               ``topology_steps.h``）。未生成步表或附录方程未收录时会回落，表中
               记 ``MISS``——那是"没有这张表"，不是 dtype 不支持。

用法::

    python -m pyqcd.testing.einsum.bench_dtype_matrix --backend kupl
    python -m pyqcd.testing.einsum.bench_dtype_matrix --backend cuda --dims 2,4,8
"""
from __future__ import annotations

import argparse
import gc
import statistics
import time
from typing import Any, Callable, Sequence

import numpy as np

from pyqcd.einsum import _kernel_cuda, _kernel_kupl, contract
from pyqcd.einsum import _wick_tasklist as tasklist

DTYPES = (np.complex128, np.complex64, np.float64, np.float32)
# 生产场景按角色反推每个标签的轴长（Nx/Lt/Nev/Ns/Mom），小 dim 的超立方体档
# 只用于路径与正确性证据 —— 性能结论只认 ``--scene`` 档（性能数据门）。

TOLERANCE = {"complex128": 1e-12, "float64": 1e-12,
             "complex64": 1e-4, "float32": 1e-4}
_SHAPE_CACHE: dict[tuple[Any, ...], list[tuple[int, ...]]] = {}


def _scene_shapes(equation: str, roles: Any, scene: str,
                  scale: int | None) -> list[tuple[int, ...]]:
    """生产形状：直接从角色表推出每个标签的轴长，纯函数、不经过 builder。

    ``operands_for`` 会按角色写标签表，跨 dtype 反复调用时状态不干净（实测
    第二次会给出 1e20 量级的假轴长）；这里只读地构造一次形状。
    """
    from pyqcd.einsum._bench_scene import dims as scene_dims
    from pyqcd.einsum.bench_specialized import _label_sizes

    terms = equation.split("->")[0].split(",")
    sizes = _label_sizes(terms, roles, scene_dims(scene, scale=scale))
    return [tuple(int(sizes[char]) for char in term) for term in terms]


def _scene_operands(shapes: Sequence[tuple[int, ...]], dtype: Any):
    # 每个 dtype 都用**新建**的纯 Python tuple 副本：缓存里的元组若被上游复用
    # 会读到失效内容（实测出现 1e20 量级的假轴长），复制一次就彻底隔离。
    import os
    if os.environ.get("PYQCD_BENCH_DEBUG"):
        print(f"[dbg] {dtype} shapes[0:2]={shapes[:2]!r} id={id(shapes)}", flush=True)
    return [_random(tuple(int(item) for item in shape), dtype) for shape in shapes]


def _random(shape: tuple[int, ...], dtype: Any):
    """确定性、低内存的操作数：``arange`` 取模后 reshape，再按需加虚部。

    生产档单个操作数可达上千万个元素，``standard_normal`` 还要额外分配一份
    float64 与一份复数中间体；这里只分配一份实数再原地转 dtype。
    """
    total = int(np.prod([int(item) for item in shape])) if shape else 1
    if total > 10 ** 12:                  # 形状解析出错的显式指认，别静默 OOM
        raise ValueError(f"scene shape {shape!r} has {total:.3e} elements")
    base = np.arange(total, dtype=np.float64) % 13.0 - 6.0
    data = base.reshape(shape)
    if np.dtype(dtype).kind == "c":
        data = data + 1j * ((base[::-1] % 5.0) - 2.0).reshape(shape)
    return np.ascontiguousarray(data.astype(dtype))


def _operands(equation: str, dim: int, dtype: Any, seed: int):
    terms = equation.split("->")[0].split(",")
    rng = np.random.default_rng(seed)
    values = []
    for term in terms:
        data = rng.standard_normal((dim,) * len(term))
        if np.dtype(dtype).kind == "c":
            data = data + 1j * rng.standard_normal(data.shape)
        values.append(np.ascontiguousarray(data.astype(dtype)))
    return values


def _sync(backend: str) -> None:
    if backend != "cuda":
        return
    import cupy

    cupy.cuda.Device(0).synchronize()


def _timed(backend: str, call: Callable[[], Any]) -> float:
    _sync(backend)
    start = time.perf_counter()
    call()
    _sync(backend)
    return time.perf_counter() - start


def _median(backend: str, call: Callable[[], Any], rounds: int) -> float:
    _timed(backend, call)
    return statistics.median(_timed(backend, call) for _ in range(rounds))


def _flops(equation: str, operands: Sequence[np.ndarray]) -> float:
    cached = _kernel_kupl._dedicated_cached(
        equation, tuple(operand.shape for operand in operands),
        np.dtype(operands[0].dtype), "auto", "kupl")
    return float(cached[3]) if cached else 0.0


def _paths(backend: str) -> tuple[tuple[str, Callable[..., Any]], ...]:
    if backend == "cuda":
        return (_kernel_cuda.try_dedicated, _kernel_cuda.try_topology_apply)
    return (_kernel_kupl.try_dedicated, _kernel_kupl.try_topology_apply)


def run(backend: str, seqs: Sequence[int], dims: Sequence[int], rounds: int,
        scene: str | None = None, scale: int | None = None,
        mem_budget_gib: float = 60.0) -> int:
    module = _kernel_cuda if backend == "cuda" else _kernel_kupl
    to_device = (lambda value: value) if backend != "cuda" else _cupy_asarray()
    to_host = (lambda value: np.asarray(value)) if backend != "cuda" else _cupy_asnumpy()

    try:
        dedicated, topology = _paths(backend)
    except AttributeError:                                # pragma: no cover - 诊断用
        print(f"{backend}: kernel module has no dedicated/topology entry")
        return 1

    if scene is not None:
        from pyqcd.einsum._bench_scene import describe

        print(describe(scene, scale=scale))
    header = (f"{'dtype':11s} {'seq':>5s} {'dim':>6s} {'generic':>9s} {'dedicated':>10s} "
              f"{'ratio':>6s} {'topology':>10s} {'ratio':>6s} {'GFLOP/s':>8s}")
    print(f"=== {backend} ===")
    print(header)
    losses: list[tuple[str, str, int, int, float, float]] = []
    shapes_by_seq: dict[int, list[tuple[int, ...]]] = {}
    if scene is not None:
        # 每个方程只在**进入 dtype 循环前**解析一次形状：形状推导依赖角色表，
        # 重复调用会拿到被污染的结果（实测第二次出现 1e20 量级的假轴长）。
        for seq in seqs:
            task = tasklist.lookup(seq)
            shapes_by_seq[seq] = _scene_shapes(task.equation, task.roles,
                                               scene, scale)
    for dtype in DTYPES:
        name = np.dtype(dtype).name
        for seq in seqs:
            task = tasklist.lookup(seq)
            grid = (None,) if scene is not None else tuple(dims)
            for dim in grid:
                if scene is not None:
                    host_ops = _scene_operands(shapes_by_seq[seq], dtype)
                    shape_note = "scene"
                else:
                    host_ops = _operands(task.equation, dim, dtype, seed=seq + dim)
                    shape_note = str(dim)
                footprint = sum(item.nbytes for item in host_ops) * 4 / 2 ** 30
                if footprint > mem_budget_gib:
                    print(f"{name:11s} {seq:5d} {shape_note:>6s} SKIP 估算峰值 "
                          f"{footprint:.1f} GiB > 预算 {mem_budget_gib:.0f} GiB")
                    continue
                device_ops = [to_device(item) for item in host_ops]
                # 无精度损失的判据：特化路径必须复现**同 dtype 通用路径**的结果。
                # 跨 dtype 比较会把"精度本来就不同"误判成内核损失。
                expected = contract(task.equation, *host_ops, specialize=False,
                                    optimize="greedy")
                reference_dtype = np.dtype(expected.dtype)
                try:
                    generic = _median(backend, lambda: contract(
                        task.equation, *device_ops, specialize=False), rounds)
                except Exception as error:      # noqa: BLE001 - 基准要跑完回整表
                    print(f"{name:11s} {seq:5d} {shape_note:>6s} ERROR generic: "
                          f"{type(error).__name__}: {str(error)[:60]}")
                    continue
                flops = _flops(task.equation, host_ops)
                row = [f"{name:11s} {seq:5d} {shape_note:>6s} {generic * 1e3:8.2f}m"]
                for label, entry in (("dedicated", dedicated), ("topology", topology)):
                    result = entry(task.equation, device_ops)
                    if result is None:
                        row.append(f"{label:>10s} {'MISS':>6s}")
                        continue
                    actual = to_host(result)
                    if np.dtype(actual.dtype) != reference_dtype or \
                            actual.shape != expected.shape:
                        row.append(f"{label:>10s} {'DTYPE':>6s}")
                        continue
                    scale = float(np.max(np.abs(expected))) or 1.0
                    error = float(np.max(np.abs(actual - expected))) / scale
                    if error > TOLERANCE[name]:
                        row.append(f"{label:>10s} {'WRONG':>6s}")
                        continue
                    cost = _median(backend, lambda entry=entry: entry(
                        task.equation, device_ops), rounds)
                    ratio = cost / generic
                    row.append(f"{cost * 1e3:9.2f}m {ratio:6.2f}x")
                    if ratio > 1.0:
                        losses.append((backend, name, seq, dim, ratio, error))
                throughput = flops / (generic * 1e9) if generic > 0 else 0.0
                row.append(f"{throughput:8.2f}")
                print("   ".join(row), flush=True)
                # 生产档一张 peram 就上 GiB：不显式释放会跨 dtype 累加，最后
                # 触发 numpy 的"array is too big"而不是干净地报 OOM。
                del host_ops, device_ops, expected
                gc.collect()

    print()
    if losses:
        print("特化慢于通用的单元格（ratio > 1，越小越亏）:")
        for backend_name, name, seq, dim, ratio, error in sorted(
                losses, key=lambda item: -item[4])[:10]:
            print(f"  {backend_name}/{name} seq={seq} dim={dim}: {ratio:.2f}x "
                  f"(rel={error:.1e})")
    else:
        print("特化在所有单元格都不慢于通用路径")
    return 0


def _cupy_asarray():
    import cupy

    return cupy.asarray


def _cupy_asnumpy():
    import cupy

    return cupy.asnumpy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="kupl", choices=("kupl", "cuda"))
    parser.add_argument("--seqs", default="2,100,500")
    parser.add_argument("--dims", default="2,4,8")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--scene", default=None,
                        help="生产场景档（scene/full）；给定后忽略 --dims")
    parser.add_argument("--scale", type=int, default=None,
                        help="按 scale/32 缩放生产维度（仅路径证据，不作性能结论）")
    parser.add_argument("--mem-budget-gib", type=float, default=60.0)
    args = parser.parse_args(argv)
    return run(args.backend, tuple(int(v) for v in args.seqs.split(",")),
               tuple(int(v) for v in args.dims.split(",")), args.rounds,
               scene=args.scene, scale=args.scale,
               mem_budget_gib=args.mem_budget_gib)


if __name__ == "__main__":
    raise SystemExit(main())
