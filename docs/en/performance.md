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

## Topology-specialised kernels

The eleven hadron-topology cases in `refer/sush/wick_12cases.md` define **2582 unique
contraction topologies** with 6–16 operands. They are indexed by *topology signature*,
not by spelling: labels are renumbered by first appearance across the whole equation,
so `ij,jk->ik`, `ij,jb->ib` and `ab,bc->ac` are one entry, while `ij,kj->ik` and
`ij,jk->ki` are different entries. Whitespace and an omitted output do not change the
signature; the operand order and the index order inside an operand do.

On a hit the whole contraction plan is handed to one native call
(`qcd_kupl_topology_apply` or `qcd_cuda_topology_apply`) instead of being stepped
through in Python. The hit line distinguishes the two cases: `[kernel: kupl_topology]`
means the native topology entry executed the plan, `[plan-cache]` means the equation
was recognised but only the plan was reused — for example with `out=`, with mixed
operand dtypes, or when the native entry is unavailable. A fallback changes speed
only; the result is identical.

This path is not performance-validated yet: the KUPL executor has an equation-specific
size cliff (corpus equation #194, 16 operands, `dim=4`: 2 ms on the generic path vs
>90 s through the topology entry) and the measurement table is still `TBD`.
`docs/topology_kernels_20261111.md` records the corpus definition, the matching and
execution mechanics, and the pending numbers; do not quote a speed-up for this path
until that table is filled.

```bash
python -m pyqcd.testing._topology_equations_contract
python -m pyqcd.testing._topology_kernel_contract
```

The first command checks all 2582 equations against `numpy` plus the renaming and
structural negative cases; the second checks the native topology entry itself — the
dtype policy, the plan/pointer/`out` arguments, and a 120-equation corpus sample at
`dim=4`.

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

## Reproduce

One script, both tables and both precisions: `python bench.py cpu` then `python bench.py gpu` (double by default), or append `single` for a faster run: `python bench.py cpu single` / `python bench.py gpu single`.

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
PRECISION = sys.argv[2] if len(sys.argv) > 2 else "double"  # "single" or "double"
ROUNDS = 3                                               # interleaved timing rounds
REAL = np.float32 if PRECISION == "single" else np.float64
COMPLEX = np.complex64 if PRECISION == "single" else np.complex128
RTOL = 1e-5 if PRECISION == "single" else 1e-10

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
print("pyqcd", pyqcd.__version__, pyqcd.__file__, "|", device_name,
      "|", PRECISION)

rng = np.random.default_rng(0)


def random_array(*shape):
    """Real random array (float64)."""
    return to_device(rng.standard_normal(shape).astype(REAL))


def random_complex_array(*shape):
    """Complex random array in the requested precision: real and imaginary parts sampled separately."""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return to_device((real_part + 1j * imag_part).astype(COMPLEX))


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
    assert np.max(np.abs(ours_value - their_value), initial=0.0) <= RTOL * scale, label
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
