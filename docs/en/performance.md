# Performance

`pyqcd.einsum` is benchmarked against `opt_einsum` on the same machine, same dtype,
same thread settings, with both implementations **interleaved inside one loop** so
machine drift cancels out. Every number below is measured, not modelled, and each
table comes with the script that produced it.

## Measurement protocol

| Item | Setting |
|---|---|
| Oracle | `opt_einsum` 3.4.0, `dp` planner |
| dtype | `complex128` where noted, otherwise `float64` (both sides) |
| Timing | one warm-up call, then **3 interleaved timed calls**; median of the per-round ratios |
| Sizing | production table: one call takes **0.4–2.7 s**; small-shape table: 1–300 ms |
| Correctness | every case is asserted against `numpy.einsum` before it is timed |
| Machine | 4 physical cores (8 threads) + RTX 4060 Laptop, 8 GB |

## Case coverage

Fifteen cases, one per structural category the library routes differently — not a
hand-picked highlight reel:

| Family | Cases |
|---|---|
| two-operand contraction | matmul 2D (real), matmul 2D (complex) |
| no summed index | element-wise, outer product |
| reduction only | full reduction (`ij,ij->`), partial reduction (`ij->i`) |
| multi-operand chain | chain 3 ops, chain 4 ops |
| closed triangle | hyper-edge (`ij,jk,ik->i`) |
| batch structure | batched GEMM, ellipsis batch, batch × matrix, many small batches |
| production kernels | nucleon 2pt, γ sandwich |

Both tables use **the same fifteen cases at the same shapes**, so a row is directly
comparable across devices. One call takes 0.003–2.4 s; three rounds per case, and the
implementation that runs first alternates every round.

### CPU (`numpy` arrays, KUPL backend)

| Case | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate |
|---|---|---|---|
| nucleon 2pt, nvec=112 (kernel) | 2.38 s | 44.24 s | **18.6×** |
| batched GEMM, 16×1024² | 0.25 s | 2.91 s | **11.9×** |
| ellipsis, 256×512² | 0.65 s | 5.92 s | **9.18×** |
| γ sandwich, 8192×1024 (kernel) | 0.25 s | 1.93 s | **7.83×** |
| many small batches, 4096×64² | 0.087 s | 0.201 s | **2.32×** |
| outer product, 8000 | 0.051 s | 0.083 s | **1.62×** |
| element-wise, 12000² | 0.20 s | 0.25 s | **1.25×** |
| chain 3 ops, 3×3072 | 0.54 s | 0.59 s | **1.07×** |
| matmul 2D (complex), 2048² | 0.31 s | 0.32 s | 1.04× |
| full reduction, 12000² | 0.039 s | 0.040 s | 1.02× |
| partial reduction, 16000² | 0.090 s | 0.092 s | 1.02× |
| batch × matrix, 8×1024×512 | 0.044 s | 0.045 s | 1.01× |
| chain 4 ops, 4×2048 | 0.26 s | 0.26 s | 1.00× |
| matmul 2D (real), 4096² | 0.63 s | 0.63 s | 0.99× |
| hyper-edge, 2048² | 0.088 s | 0.082 s | 0.93× |

### GPU (`cupy` arrays, CUDA backend)

| Case | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate |
|---|---|---|---|
| γ sandwich, 8192×1024 (fused kernel) | 0.51 s | 2.05 s | **4.04×** |
| nucleon 2pt, nvec=112 (kernel) | 0.91 s | 2.43 s | **2.68×** |
| full reduction, 12000² | 0.017 s | 0.036 s | **2.13×** |
| matmul 2D (complex), 2048² | 0.48 s | 0.55 s | **1.15×** |
| batch × matrix, 8×1024×512 | 0.071 s | 0.080 s | **1.13×** |
| ellipsis, 256×512² | 0.52 s | 0.58 s | **1.11×** |
| chain 3 ops, 3×3072 | 0.57 s | 0.63 s | **1.10×** |
| chain 4 ops, 4×2048 | 0.37 s | 0.41 s | **1.10×** |
| batched GEMM, 16×1024² | 0.35 s | 0.38 s | **1.09×** |
| matmul 2D (real), 4096² | 0.67 s | 0.71 s | **1.05×** |
| element-wise, 12000² | 0.015 s | 0.015 s | 1.02× |
| hyper-edge, 2048² | 0.157 s | 0.160 s | 1.02× |
| outer product, 8000 | 0.003 s | 0.003 s | 1.00× |
| partial reduction, 16000² | 0.009 s | 0.009 s | 1.00× |
| many small batches, 4096×64² | 0.034 s | 0.033 s | 0.96× |

GPU rows are GPU against GPU. Rows below ~0.05 s are bandwidth-bound on the GPU, so
their ratios are barely above noise; the same caveat applies to the 0.09 s CPU rows.

## Why some rows are 10–19× and others are 1.0×

The large ratios are **not** general speedups — they are three specific effects, each
measurable:

1. **The reference materialises a huge intermediate.** For the nucleon 2pt network
   `opt_einsum`'s `dp` path needs 7 steps and peaks at an 89.9 M-element intermediate
   (1.44 GB in `complex128`), while the inputs total only 11.8 M elements. Our kernel
   contracts the network in one pass over the `n³` block structure. That is where
   44 s → 2.4 s (18.6×) comes from — the arithmetic is the same, the traffic is not.

2. **The reference does not turn a batch axis into a batched BLAS call.** On
   `bij,bjk->bik` with 16×1024², `opt_einsum` executes its single path step in 2.99 s,
   while `numpy.matmul` on the same arrays takes 0.28 s and `numpy.einsum(optimize=…)`
   0.27 s. Our 0.25 s is simply the floor. So the 11.9× is "the reference is 11× above
   the hardware floor", not "we are 11× faster than BLAS".

3. **The reference pays per-batch pack/copy on the sandwich shape.** `ab,...bce,cd->...dae`
   needs 2 steps and a 33.6 M-element intermediate (537 MB) in `opt_einsum`; the fused
   CUDA kernel writes the result once, which is the 4× on GPU and 7.8× on CPU.

The honest summary: on the generic shapes both libraries sit at parity (0.93–1.15×),
and the two-digit multiples appear exactly where an intermediate gets avoided or a
batch axis becomes one batched GEMM.

## What runs where

| Input / shape | Executes on | Notes |
|---|---|---|
| element-wise, transpose, diagonal, trace, outer | the backend's own `einsum` | no summed label — the planner has nothing to add |
| `numpy` + batch index (a label shared by ≥2 operands **and** kept in the output) | native KUPL plan executor | one batched GEMM instead of numpy's per-batch loop |
| `numpy` + no batch index | numpy step ladder (tensordot) | the native executor is 6–10% slower here |
| `cupy` + γ sandwich | native CUDA executor | has the only fused CUDA kernel |
| `cupy` + anything else generic | `cupy.einsum` (cuBLAS) | the native batched GEMM is ~5× slower at 16×1024² |
| nucleon 2pt / 3pt main shapes | dedicated KUPL/CUDA kernels | `[kernel: …]` in the hit line |

The hit line tells you which path actually ran: `[kernel: kupl_nucleon_2pt]` means a
dedicated kernel, `[plan-cache]` means only the plan was reused. If a registered
kernel shape does not get its kernel, `pyqcd.einsum` warns immediately with the
reason instead of silently degrading.

## Where the gains come from

1. **Shape specialisation** — nucleon 2pt/3pt main get purpose-built kernels, and
   the CUDA executor has a fused γ-sandwich kernel (volume-gated: `z·y·x ≤ 1`).
2. **Not paying for planning where it cannot help** — element-wise/permutation
   equations and generic `cupy` equations are handed straight to the backend.
3. **Batched execution** — when a batch index exists, one batched GEMM replaces
   numpy's per-batch loop (that is the 7–12× column above).
4. **Plan cache + bound operands** — any equation with three or more operands caches
   its contraction list, so repeated calls skip planning entirely. Planning is 31–84%
   of call time on small/medium graphs (8-operand chain: 0.76 ms of 0.86 ms), and
   parsing is 0.002 ms, so this is where repeated calls win.

## Repeated calls

Planning is skipped on the second and later calls with the same equation, shapes,
dtypes and backend (three or more operands; a two-operand path is one step and the
cache key would cost more than the plan). Median per call, 8×8 operands:

| Case | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate |
|---|---|---|---|
| 8-operand chain `ab,bc,cd,de,ef,fg,gh,hi->ai` | 95 µs | 291 µs | **3.07×** |
| 3-operand chain `ij,jk,kl->il` | 51 µs | 72 µs | **1.41×** |
| mixed rank `ef,b,e->be` | 52 µs | 60 µs | **1.14×** |

## Memory

Peak RSS for one contraction, measured with `resource.getrusage` in separate
processes (`float64`, `d = 1200`):

| Shape | `opt_einsum` | `pyqcd.einsum` |
|---|---|---|
| 2 operands, 1200² | 18 MB | 16 MB |
| 3 operands, 1200² | 29 MB | 27 MB |
| 4 operands, 1200² | 29 MB | 27 MB |
| batched 8×512² | 16 MB | **52 MB** |

The GEMM ladders are now at or below `opt_einsum`. The batched case is the
exception: it runs through the native executor, whose packing workspace costs about
3× the reference. Size a long batched job accordingly.

## Correctness

Speed only means something on correct numbers, so the engine is verified at three
levels, and **every example in this documentation ends with an `assert`** — the
build executes all of them and fails if one stops verifying itself.

### Verify it yourself

Independent references, same inputs, all three must agree:

```python
import numpy as np
import torch
import pyqcd.einsum as pe

rng = np.random.default_rng(0)


def complex_noise(*shape):
    """按给定形状生成复数随机数组：实部与虚部各自独立采样。"""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return (real_part + 1j * imag_part).astype(np.complex128)


def check(equation, *operands):
    """用两套独立参考实现校验同一个结果。

    - numpy 用 optimize="optimal" 自行规划路径；
    - torch 用它对同一批输入的原生实现；
    三者逐元素一致（rtol 1e-10）才算通过。
    """
    ours = np.asarray(pe.contract(equation, *operands))
    numpy_ref = np.einsum(equation, *operands, optimize="optimal")
    torch_ref = torch.einsum(equation, *[torch.as_tensor(o) for o in operands]).numpy()
    assert np.allclose(ours, numpy_ref, rtol=1e-10, atol=1e-12), "differs from numpy"
    assert np.allclose(ours, torch_ref, rtol=1e-10, atol=1e-12), "differs from torch"
    print(f"{equation:32s} ok  max|d| = {np.max(np.abs(ours - numpy_ref)):.2e}")


# 逐类覆盖：从最简单的矩阵乘一路到生产形状的核子 2pt
check("ij,jk->ik", complex_noise(8, 8), complex_noise(8, 8))
# 三算子链：规划器要决定先乘哪一对
check("ij,jk,kl->il", complex_noise(8, 8), complex_noise(8, 8), complex_noise(8, 8))
# 省略号批轴：同一方程作用在一批矩阵上
check("...ij,...jk->...ik", complex_noise(2, 4, 5), complex_noise(2, 5, 6))
# 重复标签：取对角元（输入输出维度不同）
check("ii->i", complex_noise(6, 6))
# 全缩并成标量：没有输出标签
check("ij,ij->", complex_noise(5, 7), complex_noise(5, 7))

# 生产形状：8 算子的核子 2pt。Nvec=8 只是为了让文档门禁秒级跑完，
# 形状关系与真实计算完全一致（perambulator、gamma、VVV、投影算符）。
n = 8
check(
    "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",
    complex_noise(4, 4, n, n),        # perambulator（sink）
    complex_noise(4, 4, n, n),        # perambulator
    complex_noise(4, 4, n, n),        # perambulator
    complex_noise(4, 4),              # gamma_0
    complex_noise(4, 4),              # gamma_4
    complex_noise(4, n, n, n),        # sink VVV
    complex_noise(4, n, n, n),        # source VVV
    complex_noise(4, 4),              # 投影算符
)
```

### What the project checks on every change

| Check | Scope |
|---|---|
| `python -m pyqcd.testing` | 124 contracts: numerical edge cases, threading, metadata |
| differential suite (source checkout) | 980 cases against a frozen `opt_einsum` |
| `python -m pyqcd.testing.docs_examples` | every documentation block, each self-asserting |
| `verify_native --backend kupl\|cuda` | native library vs the Python planner |

### Invariants that must never break

* **the result never changes when the fast path changes** — a missing library, a
  rejected shape or a kernel error falls back (one warning at most) and the numbers
  are identical;
* **`contract_path` agrees with `opt_einsum`** on path and `opt_cost`;
* **precision is preserved** — `complex128` in, `complex128` out;
* **dtype/shape errors are raised, not guessed**, with the same wording as `opt_einsum`.

## Batch of independent contractions (`einsums`)

`pyqcd.parallel.einsum.einsums` runs a **batch of mutually independent** contractions
on several MPI ranks. Unlike the tables above, the comparison partner here is *the
same library running serially* — the question is only whether splitting a queue of
tasks across processes pays off.

Protocol: same machine, `complex64`/`float32` operands, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`
(single-thread baseline, so parallel and serial use the same per-core budget),
3 rounds with serial and parallel timed back to back, median of the per-round ratios.
Every batch is verified against serial `contract` before the ratio is printed.

**Two calibers, both reported.** `einsums` returns the worker-measured pure compute
time (`return_info=True` → `info["timings"]`), so a self-launched pool can be charged
two ways:

* **init included** (`×wall`) — what the caller actually waits for: `mpirun` start-up,
  task/result staging, plus compute. This is the number to quote for end-to-end cost.
* **init excluded** (`×compute`) — worker-reported contraction time only. This is the
  ceiling the task split can reach once the pool already exists (typical when you are
  inside `mpirun`, or when the same pool serves a much larger batch).

### Already inside `mpirun -np 4` (collective, no staging)

| Scenario | Tasks | Serial | Parallel | `×wall` | `×compute` |
|---|---|---|---|---|---|
| general: many small (128² matmul) | 400 | 42.5 ms | 25.7 ms | 1.65× | 3.81× |
| general: one-to-one (1200² matmul, #tasks = #ranks) | 3 | 101.0 ms | 41.8 ms | 2.42× | 2.97× |
| general: skewed (3 heavy + 9 light) | 12 | 103.8 ms | 37.3 ms | 2.78× | 3.21× |
| kernel: nucleon 2pt (E1) | 8 | 184.4 ms | 46.1 ms | **4.00×** | **4.07×** |
| sush: NN 2pt (S1) | 8 | 197.4 ms | 50.6 ms | **3.90×** | 3.96× |
| sush: 3pt main (P1) | 8 | 134.7 ms | 39.7 ms | **3.39×** | 3.51× |
| sush: NN 2pt, Nvec=100 (large task) | 8 | 6066.8 ms | 1536.7 ms | **3.95×** | 3.95× |
| kernel: γ sandwich (E5, ~1 ms/task) | 8 | 0.5 ms | 0.7 ms | 0.73× | 1.59× |

Here the two calibers nearly coincide (staging is zero, only the plan broadcast is
missing from `×compute`), which is exactly why collective is the mode to use.

### Self-launched pool (single process, no pre-existing `mpirun`)

| Scenario | Tasks | Serial | Pool wall | Pool compute | `×wall` | `×compute` |
|---|---|---|---|---|---|---|
| general: many small | 400 | 38.7 ms | 283.3 ms | 14.1 ms | 0.14× | 2.74× |
| general: one-to-one | 3 | 77.8 ms | 206.9 ms | 28.3 ms | 0.38× | 2.74× |
| general: skewed | 12 | 80.0 ms | 222.3 ms | 30.4 ms | 0.36× | 2.64× |
| kernel: nucleon 2pt (E1) | 8 | 142.4 ms | 337.2 ms | 162.2 ms | 0.42× | 0.88× |
| sush: NN 2pt (S1) | 8 | 140.0 ms | 218.7 ms | 57.2 ms | 0.64× | **2.45×** |
| sush: 3pt main (P1) | 8 | 101.6 ms | 327.6 ms | 146.9 ms | 0.31× | 0.69× |
| sush: NN 2pt, Nvec=100 (large task) | 8 | 4997.0 ms | 1879.7 ms | 1586.8 ms | **2.66×** | **3.15×** |
| kernel: γ sandwich (E5, ~1 ms/task) | 8 | 0.6 ms | 156.7 ms | 6.4 ms | 0.00× | 0.10× |

Reading the pool table:

* **The pool costs ≈0.15–0.3 s per call** (`wall − compute`): `mpirun` start-up
  (~150 ms) plus writing every operand and reading it back (~1.5 GB/s here). It pays
  off on end-to-end time only once the batch itself takes seconds (Nvec=100 row:
  **2.66×**).
* **`×compute` shows what the split can do.** With init excluded, small batches reach
  2.6–2.7× — i.e. the balancer is doing its job and the loss is pure fixed cost.
* **Kernel equations score below 1× on `×compute`** because each fresh worker pays
  the one-off kernel/plan warm-up on its first call (measured: E1 first call 140 ms vs
  17 ms warm). In a pool that runs one batch and exits, that cold start is real cost;
  in a long-lived pool it amortises away.

Reproduce both calibers with the bundled harnesses:

```bash
python -m pyqcd.testing.parallel_einsum bench --equation S1 --nvec 100 --procs 4
mpirun -np 4 python -m pyqcd.testing.parallel_einsum bench-collective --equation S1
```

The self-launched harness prints `speedup … (init included)` and
`speedup … (worker compute only, init excluded)`, plus the measured pool overhead.

### Reproduce

```python
# docs-skip: 会拉起 mpirun worker 进程
import statistics
import time
from functools import partial

import numpy as np
import pyqcd.einsum as pe
from pyqcd.parallel import einsums

ROUNDS = 3        # 串行与并行各测几轮，最后取每轮比值的中位数
PROCS = 4         # 自建进程池的进程数；已在 mpirun 内运行时由 MPI rank 数决定

rng = np.random.default_rng(0)


def random_array(*shape):
    """单精度实数随机数组：受限机器上用 float32 就够暴露并行收益。"""
    return rng.standard_normal(shape).astype(np.float32)


def random_complex_array(*shape):
    """单精度复数随机数组。"""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return (real_part + 1j * imag_part).astype(np.complex64)


# 真实格点 QCD 方程：sush 蒸馏的核子两关联（NN 2pt），8 个操作数
SUSH_NN_2PT = "eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN"


def sush_nn_2pt(nvec, count=8):
    """造 count 个 NN 2pt 任务：同一方程、同一形状、不同数据。

    形状照 refer/sush/_distillation/contraction.NN.2pt.py：
    三个 perambulator + 两个 gamma 矩阵 + 两个 VVV 顶点 + 一个投影算符。
    每个任务用独立随机数，模拟"同一方程、不同组态"的真实批量。
    """
    tasks = {}
    for index in range(count):
        operands = [
            random_complex_array(4, 4, nvec, nvec),        # perambulator
            random_complex_array(4, 4, nvec, nvec),        # perambulator
            random_complex_array(4, 4, nvec, nvec),        # perambulator
            random_complex_array(4, 4),                    # gamma_7
            random_complex_array(4, 4),                    # gamma_7
            random_complex_array(1, nvec, nvec, nvec),     # sink VVV
            random_complex_array(1, nvec, nvec, nvec),     # source VVV
            random_complex_array(4, 2),                    # 投影算符（前两列）
        ]
        tasks[f"S1-{index}"] = (SUSH_NN_2PT, operands)
    return tasks


def general_matmul(size, count, heavy=3):
    """通用矩阵乘批次。

    ``heavy`` 控制前几个任务使用大尺寸，用来制造负载偏斜：
    中间那组 12 个任务里只有 3 个是重的，看装箱器会不会把重的堆在同一个进程上。
    """
    tasks = {}
    for index in range(count):
        side = size if index < heavy else max(64, size // 6)
        tasks[f"m{index}"] = (
            "ij,jk->ik",
            [random_array(side, side), random_array(side, side)],
        )
    return tasks


CASES = {
    "general: many small": general_matmul(128, 400),    # 大批量、小单量（本功能的目标场景）
    "general: one-to-one": general_matmul(1200, 3),     # 任务数 = 进程数，一对一
    "general: skewed": general_matmul(1200, 12),        # 3 个重任务 + 9 个轻任务
    "sush: NN 2pt": sush_nn_2pt(40),                    # 真实蒸馏方程
    "sush: NN 2pt large": sush_nn_2pt(100),             # 同上，单任务更大
}


def timed(function, rounds=ROUNDS):
    """跑 rounds 轮，返回耗时中位数与最后一次的返回值。"""
    durations = []
    result = None
    for _ in range(rounds):
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), result


def run_serial(tasks):
    """逐个串行调用 contract —— 这就是并行要打败的基准。"""
    return [pe.contract(equation, *operands)
            for equation, operands in tasks.values()]


def run_parallel(tasks):
    """一次调用交给 einsums；返回与输入同键的字典。"""
    return einsums(tasks, max_procs=PROCS, reminder=False, verbose=False)


print(f"{'case':24s} {'tasks':>5s} {'serial(s)':>9s} "
      f"{'parallel(s)':>11s} {'speed-up':>8s}")
for label, tasks in CASES.items():
    serial_seconds, _ = timed(partial(run_serial, tasks))
    parallel_seconds, results = timed(partial(run_parallel, tasks))

    # 先验正确性：并行结果必须等于串行结果，否则倍率没有意义
    for key, (equation, operands) in tasks.items():
        reference = pe.contract(equation, *operands)
        scale = max(float(np.max(np.abs(reference))), 1e-30)
        deviation = float(np.max(np.abs(results[key] - reference))) / scale
        assert deviation <= 1e-5, f"{label} {key}: rel={deviation:.1e}"

    print(f"{label:24s} {len(tasks):5d} {serial_seconds:9.3f} "
          f"{parallel_seconds:11.3f} {serial_seconds / parallel_seconds:7.2f}x")
```

To measure the collective mode instead, save the loop above as `bench.py` and run
`mpirun -np 4 python bench.py`: every rank calls `einsums` and the work is divided over
those ranks with no data staging (the table above uses exactly that path).

## Reproduce

One script, both tables: `python bench.py cpu` then `python bench.py gpu`.

### The benchmark

```python
import statistics
import sys
import time
from functools import partial

import numpy as np
import opt_einsum as oe
import pyqcd
import pyqcd.einsum as pe

# docs-skip: 15 个用例的全尺寸基准（单次 0.4–2.7 s），文档门禁不跑
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"     # "cpu" 或 "gpu"
ROUNDS = 3                                               # 交错计时轮数

if DEVICE == "gpu":
    import cupy as cp

    def to_device(array):
        """把 numpy 数组搬到显存。"""
        return cp.asarray(array)

    def synchronise():
        """GPU 计时前必须同步，否则只量到 kernel 下发的时间。"""
        cp.cuda.Device(0).synchronize()

    def release_memory():
        """每个用例后清空显存池，避免上一次的残留把下一个用例挤爆。"""
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
else:
    def to_device(array):
        return array

    def synchronise():
        return None

    def release_memory():
        return None

    device_name = "cpu"

# 先打印版本与路径：site-packages 里的旧版会把每个数字都变成"PyQCD 很慢"
print("pyqcd", pyqcd.__version__, pyqcd.__file__, "|", device_name)

rng = np.random.default_rng(0)


def random_array(*shape):
    """实数随机数组（float64）。"""
    return to_device(rng.standard_normal(shape))


def random_complex_array(*shape):
    """复数随机数组（complex128）：实部与虚部独立采样。"""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return to_device((real_part + 1j * imag_part).astype(np.complex128))


def one_call(function, repeat=1):
    """调用 repeat 次并返回最快的一次；repeat>1 用来摊掉固定开销。"""
    synchronise()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        synchronise()
        best = min(best, time.perf_counter() - start)
    return best


# 15 个结构类别，形状与上面两张表完全一致。
# 每行 = (标签, 方程, 造数据的函数, 各操作数的形状)
CASES = [
    ("matmul 2D (real)", "ij,jk->ik", random_array,
     [(4096, 4096), (4096, 4096)]),
    ("matmul 2D (complex)", "ij,jk->ik", random_complex_array,
     [(2048, 2048), (2048, 2048)]),
    ("element-wise", "ij,ij->ij", random_array,
     [(12000, 12000), (12000, 12000)]),
    ("outer product", "i,j->ij", random_array,
     [(8000,), (8000,)]),
    ("full reduction", "ij,ij->", random_array,
     [(12000, 12000), (12000, 12000)]),
    ("partial reduction", "ij->i", random_array,
     [(16000, 16000)]),
    ("chain 3 ops", "ij,jk,kl->il", random_array,
     [(3072, 3072), (3072, 3072), (3072, 3072)]),
    ("chain 4 ops", "ij,jk,kl,lm->im", random_array,
     [(2048, 2048), (2048, 2048), (2048, 2048), (2048, 2048)]),
    ("hyper-edge", "ij,jk,ik->i", random_array,
     [(2048, 2048), (2048, 2048), (2048, 2048)]),
    ("batched GEMM", "bij,bjk->bik", random_array,
     [(16, 1024, 1024), (16, 1024, 1024)]),
    ("ellipsis", "...ij,...jk->...ik", random_array,
     [(256, 512, 512), (256, 512, 512)]),
    ("batch x matrix", "abc,cd->abd", random_array,
     [(8, 1024, 512), (512, 1024)]),
    ("many small batches", "bij,bjk->bik", random_array,
     [(4096, 64, 64), (4096, 64, 64)]),
    ("nucleon 2pt (kernel)", "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",
     random_complex_array,
     [(4, 4, 112, 112), (4, 4, 112, 112), (4, 4, 112, 112), (4, 4), (4, 4),
      (4, 112, 112, 112), (4, 112, 112, 112), (4, 4)]),
    ("gamma sandwich (kernel)", "ab,...bce,cd->...dae", random_complex_array,
     [(4, 4), (8192, 4, 4, 1024), (4, 4)]),
]

print(f"{'case':26s} {'pyqcd(s)':>9s} {'opt(s)':>9s} {'ratio':>6s}")
for label, equation, builder, shapes in CASES:
    operands = [builder(*shape) for shape in shapes]

    # 1) 先验正确性：与 opt_einsum 逐元素对照，快而不对的数字没有意义
    reference = oe.contract(equation, *operands, optimize="dp")
    got = pe.contract(equation, *operands)
    ours_value = cp.asnumpy(got) if DEVICE == "gpu" else np.asarray(got)
    their_value = cp.asnumpy(reference) if DEVICE == "gpu" else np.asarray(reference)
    scale = max(float(np.max(np.abs(their_value))), 1e-30)
    assert np.max(np.abs(ours_value - their_value), initial=0.0) <= 1e-10 * scale, label
    del reference, got, ours_value, their_value
    release_memory()

    # 2) 先探一次耗时，据此决定内层重复次数，让每个样本至少约 0.3 s
    ours_call = partial(pe.contract, equation, *operands)
    theirs_call = partial(oe.contract, equation, *operands, optimize="dp")
    probe = one_call(ours_call)
    inner = max(1, min(9, int(0.3 / max(probe, 1e-6))))

    # 3) 交错计时：每轮交换先后顺序，机器漂移在比值里互相抵消
    ours_times, theirs_times = [], []
    for round_index in range(ROUNDS):
        if round_index % 2 == 0:
            ours_times.append(one_call(ours_call, inner))
            theirs_times.append(one_call(theirs_call, inner))
        else:
            theirs_times.append(one_call(theirs_call, inner))
            ours_times.append(one_call(ours_call, inner))
        release_memory()

    # 4) 取中位数再相除，避免被单次抖动带偏
    median_ours = statistics.median(ours_times)
    median_theirs = statistics.median(theirs_times)
    print(f"{label:26s} {median_ours:9.3f} {median_theirs:9.3f} "
          f"{median_theirs / median_ours:6.2f}")
    del operands
    release_memory()
```

### Before you trust a number

```bash
python -m pyqcd --gpu-probe
```

A stale `pyqcd` in `site-packages` turns every number into "PyQCD is slow"; the script
prints `pyqcd.__version__` and `pyqcd.__file__` for exactly that reason.
