"""特化内核-基准脚本（specialized-kernel benchmark）：15 种特化路径 vs opt_einsum。

「特化路径」指**命中** wick_12cases 语料（8638 条）的形状——按拓扑签名分派到
专用执行器（KUPL/CUDA topology kernel），不走通用分步收缩。

与 ``bench_generic`` 的计时口径完全一致（交错、中位数、每条样本约 0.3 s），
因此两个脚本的数字可以直接横向比较。

用法::

    python -m pyqcd.einsum.bench_specialized            # cpu + double
    python -m pyqcd.einsum.bench_specialized cpu single
    python -m pyqcd.einsum.bench_specialized gpu
    python -m pyqcd.einsum.bench_specialized --list     # 只列 15 条路径
    python -m pyqcd.einsum.bench_specialized --sample 16 --seed 0

15 条路径按元数与结构从 8638 条语料中挑选，覆盖 6/7/9/10/12/13/14/16 算子。
``--sample N`` 改为从语料里按元数分层随机抽 ``N`` 条（排除这 15 条），
给定 ``--seed`` 可复现——固定 15 条只说明"挑出来的能跑"，抽样才说明
"语料整体都能跑"。

**维度按操作数的物理角色取自生产代码**（``refer/sush/_distillation/contraction*.py``），
不是统一的合成尺寸：

===============  =================================  ============================
角色              形状                               出处
===============  =================================  ============================
``peram_u``       ``(Lt, Nt, Ns, Ns, Nev, Nev)``     ``contraction.py:123``
                  = ``(64, 32, 4, 4, 100, 100)``
``VVV``           ``(Lt, Mom_len, Nev, Nev, Nev)``   ``contraction.NN.2pt.py:68``
                  = ``(64, 8, 100, 100, 100)``
``VDV``           ``(Lt, Mom_len, Nev, Nev)``       同上（注释行）
                  = ``(64, 8, 100, 100)``
``gamma_*``       ``(Ns, Ns)`` = ``(4, 4)``          ``Nc = 3`` → ``Ns = 4``
===============  =================================  ============================

生产参数：``lattice_size = [32, 32, 32, 64]``（``Nx = 32``、``Lt = 64``）、
``Nev_src = 100``、``Nc = 3``、``Mom_len = 8``。这是「每个指标一个真实维度」，
所以同一个标签在不同操作数里必定同长，与真实收缩一致。
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence
from functools import partial

import numpy as np

# 15 条特化路径：直接从 8638 条语料里取代表式，按用例小节归类。
# (标签, 方程, 是否是复数, 单指标尺寸)
# 场景定义统一由 :mod:`pyqcd.einsum._bench_scene` 提供（生产参数、等比缩小的
# 默认场景、冒烟档都在那里），这里不再重复一遍——重复必然漂移。
from ._bench_scene import dims as scene_dims, describe as describe_scene


def _label_sizes(terms, roles, dims) -> dict[str, int]:
    """由**角色**反推每个标签的物理尺寸。

    规则来自参考实现的结构，不是猜的：

    * ``gamma_*`` / ``C * gamma_*`` 是 ``(Ns, Ns)``（旋量 4x4），它们出现的标签
      一律 ``Ns``；这包括 peram 里与 gamma 共享的那些轴。
    * ``VVV`` = ``(Lt, Mom, Nev, Nev, Nev)``、``VDV`` = ``(Lt, Mom, Nev, Nev)``，
      它们贡献 ``Lt`` / ``Mom`` / ``Nev``。
    * ``peram_u`` 独有的标签是酉空间维 ``Nev``。

    实测校验（seq 1/3/129/855）：peram 里与 gamma 共享的轴恰好是旋量轴，
    其余是 Nev 轴，与 ``(Lt, Nt, Ns, Ns, Nev, Nev)`` 的组成一致。
    """
    sizes: dict[str, int] = {}

    def bind(label: str, extent: int, where: str) -> None:
        known = sizes.setdefault(label, extent)
        if known != extent:
            raise ValueError(
                f"label {label!r} is {known} in one role but {extent} in {where}; "
                f"the corpus roles are inconsistent")

    # 先处理小张量：它们把标签钉死在 Ns / Mom / Lt / Nev 上。
    for term, role in zip(terms, roles):
        if role.startswith("peram"):
            continue
        if "gamma" in role:
            for label in term:
                bind(label, dims["Ns"], role)
        elif role.startswith("VVV"):
            # (Lt, Mom, Nev, Nev, Nev)
            for label, extent in zip(term, (dims["Lt"], dims["Mom"],
                                            dims["Nev"], dims["Nev"], dims["Nev"])):
                bind(label, extent, role)
        elif role.startswith("VDV"):
            # (Lt, Mom, Nev, Nev)
            for label, extent in zip(term, (dims["Lt"], dims["Mom"],
                                            dims["Nev"], dims["Nev"])):
                bind(label, extent, role)
        else:
            raise ValueError(f"unknown role {role!r}")

    # peram 的剩余标签：一律 Nev（酉空间维）。
    for term, role in zip(terms, roles):
        if not role.startswith("peram"):
            continue
        for label in term:
            bind(label, sizes.get(label, dims["Nev"]), role)
    return sizes


def operands_for(equation: str, dims: dict[str, int], builder,
                 roles: Sequence[str] | None = None):
    """按真实角色生成操作数：标签尺寸由角色反推，同一标签全局同长。"""
    terms = equation.split("->")[0].split(",")
    if roles is None:
        raise ValueError("roles are required: the corpus supplies them per operand")
    if len(roles) != len(terms):
        raise ValueError(
            f"{len(roles)} roles for {len(terms)} operands in {equation!r}")
    sizes = _label_sizes(terms, roles, dims)
    return [builder(*[sizes[label] for label in term]) for term in terms]


# 15 条特化路径：直接引用语料里的 seq，方程与角色都从 _wick_tasklist 取，
# 不在这里重抄（重抄就会与语料漂移）。复数/实数成对出现，便于看 dtype 影响。
CASES: tuple[tuple[str, int, bool], ...] = (
    ("2pt 6 ops  (gamma + VDV)", 1, True),
    ("2pt 6 ops  (real)", 1, False),
    ("3pt 7 ops  (gamma + VVV)", 3, True),
    ("4pt 9 ops  (three gamma + VDV)", 849, True),
    ("4pt 10 ops (VVV/VDV mix)", 855, True),
    ("4pt 10 ops (real)", 855, False),
    ("5pt 12 ops (four gamma + VDV)", 1599, True),
    ("3pt 13 ops (full geometry)", 9, True),
    ("4pt 13 ops (full geometry)", 1623, True),
    ("4pt 13 ops (VDV/VVV mix)", 1743, True),
    ("5pt 14 ops (six peram)", 129, True),
    ("5pt 14 ops (six peram, real)", 129, False),
    ("6pt 16 ops (deepest)", 1863, True),
    ("6pt 16 ops (deepest, real)", 1863, False),
    ("6pt 16 ops (momentum)", 3461, True),
)


def case_listing(cases=None) -> list[tuple[str, str, int]]:
    """(标签, 方程, 元数)；--list 用，不导入 opt_einsum。"""
    import pyqcd.einsum._wick_tasklist as tasklist

    by_seq = tasklist.by_seq()
    return [(label, by_seq[seq].equation, by_seq[seq].arity)
            for label, seq, _ in (CASES if cases is None else cases)]


def sample_cases(count: int = 16, seed: int = 0) -> list[tuple[str, int, bool]]:
    """从 8638 条语料里**随机抽** ``count`` 条，排除固定那 15 条。

    抽样必须可复现：给定 ``seed`` 时结果稳定，便于跨机器/跨版本对照。
    覆盖尽量分散——先按元数分层，再在层内随机，避免抽到的全是同一档。
    """
    import pyqcd.einsum._wick_tasklist as tasklist

    if count <= 0:
        raise ValueError(f"sample count must be positive, got {count}")
    used = {seq for _, seq, _ in CASES}
    pool = [task for task in tasklist.TASKS if task.seq not in used]
    if not pool:
        raise ValueError("no unused corpus path left to sample")
    # 按元数分层：层内顺序由 rng 置换决定，同 seed 下稳定。
    by_arity: dict[int, list[int]] = {}
    for task in pool:
        by_arity.setdefault(task.arity, []).append(task.seq)
    arities = sorted(by_arity)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    # 轮流从各元数层抽，直到凑够 count 条。
    cursors = {arity: 0 for arity in arities}
    shuffled = {arity: rng.permutation(by_arity[arity]).tolist()
                for arity in arities}
    while len(picked) < count:
        progressed = False
        for arity in arities:
            if len(picked) >= count:
                break
            layer = shuffled[arity]
            index = cursors[arity]
            if index < len(layer):
                picked.append(layer[index])
                cursors[arity] = index + 1
                progressed = True
        if not progressed:
            break
    by_seq = tasklist.by_seq()
    cases = []
    for seq in picked:
        task = by_seq[seq]
        # 复数/实数交替，兼顾 dtype 影响。
        cases.append((f"sampled seq={seq} ({task.arity} ops)",
                      seq, len(cases) % 2 == 0))
    return cases


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
        cases=None, scene: str = "scene", scale: int | None = None) -> int:
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

    rtol = 1e-5 if precision == "single" else 1e-10
    chosen = cases or CASES
    arrays = {"real": real, "complex": complex_}
    as_numpy = (lambda value: value.get()) if device == "gpu" else np.asarray
    dims = scene_dims(scene, scale=scale)
    import pyqcd.einsum._wick_tasklist as tasklist

    by_seq = tasklist.by_seq()
    print(f"pyqcd {pyqcd.__version__} {pyqcd.__file__}")
    print(f"特化内核-基准脚本 | {device_name} | {precision} | {len(chosen)} 条路径 "
          f"| rounds={rounds}")
    print(describe_scene(scene, scale=scale))
    print(f"{'case':30s} {'pyqcd(s)':>9s} {'opt(s)':>9s} {'ratio':>7s}  path")
    print("-" * 78)

    ratios: list[float] = []
    for label, seq, is_complex in chosen:
        task = by_seq[seq]
        equation = task.equation
        builder = arrays["complex" if is_complex else "real"]
        operands = operands_for(equation, dims, builder, task.roles)

        try:
            reference = oe.contract(equation, *operands, optimize="dp")
        except MemoryError as exc:
            # opt_einsum(dp) 只按 flops 选路，高元数路径可能要求上百 GiB 的
            # 单步中间体（16 算子实测要过 95 GiB）。那是参考实现的病态，
            # 不是内核错误：标 REF-FAIL、不给倍率，继续下一条。
            print(f"{label:30s} {'-':>9s} {'-':>9s} {'-':>7s}  "
                  f"REF-FAIL  {exc}")
            del operands
            release()
            continue
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
        if not (median_ours < 1e-3 and median_theirs < 1e-3):
            ratios.append(ratio)
        status = executed if correct else "WRONG"
        # 双方都 < 1 ms 时，计时噪声会主导比值：标出来，不作为性能结论。
        too_short = median_ours < 1e-3 and median_theirs < 1e-3
        if too_short:
            status += " too_short"
        print(f"{label:30s} {median_ours:9.4f} {median_theirs:9.4f} "
              f"{ratio:6.2f}x  {status}")
        del operands
        release()

    if ratios:
        print("-" * 78)
        print(f"中位比值 {statistics.median(ratios):.2f}x  "
              f"（<1 表示 pyqcd 更快；{len(ratios)} 条参与统计，"
              f"too_short 已排除）")
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
    parser.add_argument("--scene", default="scene",
                        choices=("full", "scene", "smoke"),
                        help="生产场景档位（默认 scene；full 需约 31 GB 内存）")
    parser.add_argument("--scale", type=int, default=None,
                        help="按 scale/32 同比缩放生产维度（覆盖默认场景）")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="从语料随机抽 N 条新路径（排除固定 15 条）")
    parser.add_argument("--seed", type=int, default=0,
                        help="抽样种子（配合 --sample 复现同一批路径）")
    parser.add_argument("--list", action="store_true", help="只列路径")
    args = parser.parse_args(argv)

    if args.sample < 0:
        parser.error("--sample must be positive")
    chosen = sample_cases(args.sample, args.seed) if args.sample else list(CASES)
    if args.list:
        # --list 不导入 opt_einsum，纯安装环境也能用。
        print(describe_scene(args.scene, scale=args.scale))
        print()
        for index, (label, equation, arity) in enumerate(case_listing(chosen), 1):
            print(f"{index:2d}. {label:30s} arity={arity:2d}  {equation}")
        return 0

    cases = chosen[:4] if args.quick else chosen
    return run(device=args.device, precision=args.precision,
               rounds=args.rounds, quick=args.quick, cases=cases,
               scene=args.scene, scale=args.scale)


if __name__ == "__main__":
    raise SystemExit(main())
