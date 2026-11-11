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
    """Complex random array of the given shape (real and imaginary parts sampled separately)."""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return (real_part + 1j * imag_part).astype(np.complex128)


def check(equation, *operands):
    """Verify one result against two independent reference implementations.

    - numpy plans its own path with optimize="optimal";
    - torch evaluates the same operands with its native implementation;
    the three must agree element-wise (rtol 1e-10) for the case to pass.
    """
    ours = np.asarray(pe.contract(equation, *operands))
    numpy_ref = np.einsum(equation, *operands, optimize="optimal")
    torch_ref = torch.einsum(equation, *[torch.as_tensor(o) for o in operands]).numpy()
    assert np.allclose(ours, numpy_ref, rtol=1e-10, atol=1e-12), "differs from numpy"
    assert np.allclose(ours, torch_ref, rtol=1e-10, atol=1e-12), "differs from torch"
    print(f"{equation:32s} ok  max|d| = {np.max(np.abs(ours - numpy_ref)):.2e}")


# One case per structural category, from a plain matmul up to a production shape.
check("ij,jk->ik", complex_noise(8, 8), complex_noise(8, 8))

# Three-operand chain: the planner has to choose which pair to contract first.
check("ij,jk,kl->il", complex_noise(8, 8), complex_noise(8, 8), complex_noise(8, 8))

# Ellipsis batch axis: the same equation applies to a batch of matrices.
check("...ij,...jk->...ik", complex_noise(2, 4, 5), complex_noise(2, 5, 6))

# Repeated label: take the diagonal (input and output have different rank).
check("ii->i", complex_noise(6, 6))

# Full contraction to a scalar: the output has no labels.
check("ij,ij->", complex_noise(5, 7), complex_noise(5, 7))

# Production shape: the 8-operand nucleon 2pt. Nvec=8 only keeps the documentation
# gate fast; the shape relations are exactly those of the real calculation
# (perambulators, gammas, VVV vertices, projector).
n = 8
check(
    "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",
    complex_noise(4, 4, n, n),        # perambulator (sink)
    complex_noise(4, 4, n, n),        # perambulator
    complex_noise(4, 4, n, n),        # perambulator
    complex_noise(4, 4),              # gamma_0
    complex_noise(4, 4),              # gamma_4
    complex_noise(4, n, n, n),        # sink VVV
    complex_noise(4, n, n, n),        # source VVV
    complex_noise(4, 4),              # projector
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

> **Memory warning — collective mode replicates the input on every rank.**
> The planner runs inside `einsums`, so each rank must build the *whole* batch:
> input memory becomes `one copy × ranks`. Rank 0 additionally holds **all**
> results, and `broadcast=True` gives every rank its own full copy of them.
> Measured with `resource.ru_maxrss` on float32 matrix products (batch input
> 134 MB, 64 tasks of 512²): serial peaks at **234 MB**, while collective on
> 4 ranks totals ≈ **1.15 GB** (`4 × input + output`). On GPU this means one full
> batch per visible card — keep the rank count equal to the device count or VRAM
> multiplies. If the batch does not fit, use the self-launched pool: it ships tasks
> through disk and each worker reads only its own bin, at ~0.15–0.3 s fixed cost
> per call. `einsums` prints this warning on every collective call, independently
> of `reminder=`.

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
# docs-skip: launches MPI worker processes
import statistics
import time
from functools import partial

import numpy as np
import pyqcd.einsum as pe
from pyqcd.parallel import einsums

ROUNDS = 3        # serial and parallel are each timed this many times; take the median ratio
PROCS = 4         # size of the self-launched pool; inside mpirun the rank count decides

rng = np.random.default_rng(0)


def random_array(*shape):
    """Single-precision real random array: float32 is enough to expose the parallel gain."""
    return rng.standard_normal(shape).astype(np.float32)


def random_complex_array(*shape):
    """Single-precision complex random array."""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return (real_part + 1j * imag_part).astype(np.complex64)


# A real lattice-QCD equation: the sush distillation nucleon two-point (NN 2pt), 8 operands.
SUSH_NN_2PT = "eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN"


def sush_nn_2pt(nvec, count=8):
    """Build ``count`` NN 2pt tasks: same equation, same shapes, different data.

    Shapes follow refer/sush/_distillation/contraction.NN.2pt.py: three perambulators,
    two gamma matrices, two VVV vertices and one projector. Each task uses its own
    random numbers, mimicking the real case of "one equation, many configurations".
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
            random_complex_array(4, 2),                    # projector (first two columns)
        ]
        tasks[f"S1-{index}"] = (SUSH_NN_2PT, operands)
    return tasks


def general_matmul(size, count, heavy=3):
    """Batch of plain matmuls.

    ``heavy`` sets how many of the first tasks use the large size, which creates the
    load skew: in the 12-task case only 3 tasks are heavy, so we can see whether the
    bin packer puts them on the same process or spreads them out.
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
    "general: many small": general_matmul(128, 400),    # many tasks, tiny each (the target case)
    "general: one-to-one": general_matmul(1200, 3),     # tasks == processes, one each
    "general: skewed": general_matmul(1200, 12),        # 3 heavy tasks + 9 light tasks
    "sush: NN 2pt": sush_nn_2pt(40),                    # real distillation equation
    "sush: NN 2pt large": sush_nn_2pt(100),             # same, with a larger single task
}


def timed(function, rounds=ROUNDS):
    """Run ``rounds`` times and return the median duration plus the last value."""
    durations = []
    result = None
    for _ in range(rounds):
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), result


def run_serial(tasks):
    """Call contract one task at a time — this is the baseline parallel has to beat."""
    return [pe.contract(equation, *operands)
            for equation, operands in tasks.values()]


def run_parallel(tasks):
    """Hand the whole batch to einsums; it returns a dict with the same keys."""
    return einsums(tasks, max_procs=PROCS, reminder=False, verbose=False)


print(f"{'case':24s} {'tasks':>5s} {'serial(s)':>9s} "
      f"{'parallel(s)':>11s} {'speed-up':>8s}")
for label, tasks in CASES.items():
    serial_seconds, _ = timed(partial(run_serial, tasks))
    parallel_seconds, results = timed(partial(run_parallel, tasks))

    # Correctness first: the parallel result must equal the serial one, otherwise the
    # speed-up means nothing.
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

# docs-skip: full-size 15-case benchmark (0.4-2.7 s per call); the docs gate skips it
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"     # "cpu" or "gpu"
ROUNDS = 3                                               # interleaved timing rounds

if DEVICE == "gpu":
    import cupy as cp

    def to_device(array):
        """Move a numpy array to the GPU."""
        return cp.asarray(array)

    def synchronise():
        """Synchronise before timing on the GPU; otherwise you only measure kernel launch."""
        cp.cuda.Device(0).synchronize()

    def release_memory():
        """Empty the memory pools after each case so leftovers do not crowd out the next one."""
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

# Print version and path first: a stale copy in site-packages turns every number into
# "PyQCD is slow".
print("pyqcd", pyqcd.__version__, pyqcd.__file__, "|", device_name)

rng = np.random.default_rng(0)


def random_array(*shape):
    """Real random array (float64)."""
    return to_device(rng.standard_normal(shape))


def random_complex_array(*shape):
    """Complex random array (complex128): real and imaginary parts sampled separately."""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return to_device((real_part + 1j * imag_part).astype(np.complex128))


def one_call(function, repeat=1):
    """Call ``repeat`` times and keep the fastest; repeat>1 amortises fixed overhead."""
    synchronise()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        synchronise()
        best = min(best, time.perf_counter() - start)
    return best


# Fifteen structural categories, at exactly the shapes used in the two tables above.
# Each row is (label, equation, data builder, shapes of the operands).
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

    # 1) Correctness first: compare element-wise against opt_einsum. A fast wrong
    #    number is worthless.
    reference = oe.contract(equation, *operands, optimize="dp")
    got = pe.contract(equation, *operands)
    ours_value = cp.asnumpy(got) if DEVICE == "gpu" else np.asarray(got)
    their_value = cp.asnumpy(reference) if DEVICE == "gpu" else np.asarray(reference)
    scale = max(float(np.max(np.abs(their_value))), 1e-30)
    assert np.max(np.abs(ours_value - their_value), initial=0.0) <= 1e-10 * scale, label
    del reference, got, ours_value, their_value
    release_memory()

    # 2) Probe once to choose the inner repeat count, so each sample is about 0.3 s.
    ours_call = partial(pe.contract, equation, *operands)
    theirs_call = partial(oe.contract, equation, *operands, optimize="dp")
    probe = one_call(ours_call)
    inner = max(1, min(9, int(0.3 / max(probe, 1e-6))))

    # 3) Interleaved timing: swap the order every round so machine drift cancels out
    #    in the ratio.
    ours_times, theirs_times = [], []
    for round_index in range(ROUNDS):
        if round_index % 2 == 0:
            ours_times.append(one_call(ours_call, inner))
            theirs_times.append(one_call(theirs_call, inner))
        else:
            theirs_times.append(one_call(theirs_call, inner))
            ours_times.append(one_call(ours_call, inner))
        release_memory()

    # 4) Take medians first, then divide, so a single jittery sample cannot dominate.
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
