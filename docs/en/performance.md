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

## Production-scale results

Sized so that a single call is in the 0.4–2.7 s range — short runs measure cache
warm-up and timer resolution rather than the contraction, and they hide the BLAS
blocking regime that production shapes live in.

### CPU (KUPL backend, `numpy` arrays)

| Case | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate |
|---|---|---|---|
| batched matmul, 32×1024² | 0.61 s | 6.98 s | **11.6×** |
| ellipsis, 256×512² | 0.91 s | 6.48 s | **7.1×** |
| Hadamard, 16000² | 0.44 s | 0.55 s | **1.26×** |
| matmul, 4600² | 1.05 s | 1.06 s | 1.01× |
| 3-operand chain, 3×3570 | 1.00 s | 0.98 s | 0.98× |
| complex matmul, 3100² | 1.36 s | 1.34 s | 0.96× |

### GPU (CUDA backend, `cupy` arrays)

GPU numbers are **GPU against GPU**: both libraries run on the same device, so the
ratio measures implementations, not CPU-vs-GPU throughput.

| Case | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate |
|---|---|---|---|
| γ sandwich, 8192×1024 (fused kernel) | 0.76 s | 1.54 s | **2.0×** |
| batched GEMM, 16×1024² | 0.75 s | 0.74 s | 0.98× |
| nucleon 2pt, nvec=112 (kernel) | 2.67 s | 2.65 s | 0.99× |

Two honest notes. The γ-sandwich row is *not* the 7–16× measured on smaller shapes:
at 8192×1024 the fused kernel is bound by its 2.1 GB workspace traffic, not by
arithmetic. And the nucleon-2pt kernel reaches parity at nvec=112 — the shape is
already cuBLAS-bound, so there is nothing left for the kernel to win.

## Small-shape results

Useful for orientation, but do not extrapolate: at this scale the numbers are
dominated by fixed per-call overhead.

### CPU

| Case | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate |
|---|---|---|---|
| batched matmul, 32×256² | 9.3 ms | 97.1 ms | **10.6×** |
| ellipsis, 16×128×128 | 1.1 ms | 6.3 ms | **5.8×** |
| 3-operand chain, 3×500 | 3.1 ms | 3.2 ms | 1.03× |
| complex matmul, 800² | 20.4 ms | 21.0 ms | 1.03× |
| Hadamard, 4000² | 30.1 ms | 30.2 ms | 1.00× |
| outer product, 2000×2000 | 4.1 ms | 4.1 ms | 1.00× |
| 4-operand chain, 300 | 1.15 ms | 1.11 ms | 1.00× |
| matmul, 1000² | 10.7 ms | 10.2 ms | 0.97× |

### GPU

| Case | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate |
|---|---|---|---|
| γ sandwich, 512×512 | 3.3 ms | 53.1 ms | **16.2×** |
| nucleon 2pt, nvec=64 (kernel) | 146 ms | 263 ms | **1.81×** |
| batched GEMM, 32×512² | 178 ms | 167 ms | 0.94× |

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
4. **Plan cache + bound operands** — repeated calls with the same equation/shape
   skip planning entirely.

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


def check(equation, *operands):
    """Compare pyqcd against two independent references on the same inputs."""
    ours = np.asarray(pe.contract(equation, *operands))
    numpy_ref = np.einsum(equation, *operands, optimize="optimal")
    torch_ref = torch.einsum(equation, *[torch.as_tensor(o) for o in operands]).numpy()
    assert np.allclose(ours, numpy_ref, rtol=1e-10, atol=1e-12), "differs from numpy"
    assert np.allclose(ours, torch_ref, rtol=1e-10, atol=1e-12), "differs from torch"
    print(f"{equation:32s} ok  max|Δ| = {np.max(np.abs(ours - numpy_ref)):.2e}")


d = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(np.complex128)

check("ij,jk->ik", d(8, 8), d(8, 8))                     # plain matmul
check("ij,jk,kl->il", d(8, 8), d(8, 8), d(8, 8))         # 3-operand chain
check("...ij,...jk->...ik", d(2, 4, 5), d(2, 5, 6))      # ellipsis
check("ii->i", d(6, 6))                                  # repeated label (diagonal)
check("ij,ij->", d(5, 7), d(5, 7))                       # full contraction to scalar
n = 8
check("gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",            # production nucleon 2pt shape
      d(4, 4, n, n), d(4, 4, n, n), d(4, 4, n, n), d(4, 4), d(4, 4),
      d(4, n, n, n), d(4, n, n, n), d(4, 4))
```

### What the project checks on every change

| Check | Scope |
|---|---|
| `python -m pyqcd.testing` | 121 contracts: numerical edge cases, threading, metadata |
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

### CPU, production scale

These two scripts run for minutes; the documentation gate skips them (`# docs-skip`)
but still executes every other block.

```python
import statistics
import time

import numpy as np
import opt_einsum as oe
import pyqcd

import pyqcd.einsum as pe

print("measuring pyqcd", pyqcd.__version__, pyqcd.__file__)   # docs-skip: 生产尺度基准，单次 0.4–2.7 s，门禁不跑


def one(call):
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def compare(ours, theirs, rounds=3):
    """One warm-up call, then `rounds` interleaved timed calls (seconds)."""
    ours()
    theirs()
    left, right, ratios = [], [], []
    for _ in range(rounds):
        a = one(ours)
        b = one(theirs)
        left.append(a)
        right.append(b)
        ratios.append(b / a)
    return statistics.median(left), statistics.median(right), statistics.median(ratios)


rng = np.random.default_rng(0)
real = lambda *s: rng.standard_normal(s)
cplx = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(np.complex128)

# Sized so that one call takes roughly 0.5-1.5 s: shorter runs measure timer and
# cache noise, not the contraction, and they hide the BLAS register-blocking regime.
CASES = [
    ("matmul 4600^2", "ij,jk->ik", [real(4600, 4600), real(4600, 4600)]),
    ("chain 3x3570", "ij,jk,kl->il", [real(3570, 3570)] * 3),
    ("batched 32x1024^2", "bij,bjk->bik", [real(32, 1024, 1024), real(32, 1024, 1024)]),
    ("ellipsis 256x512^2", "...ij,...jk->...ik",
     [real(256, 512, 512), real(256, 512, 512)]),
    ("complex mm 3100^2", "ij,jk->ik", [cplx(3100, 3100), cplx(3100, 3100)]),
    ("hadamard 16000^2", "ij,ij->ij", [real(16000, 16000), real(16000, 16000)]),
]

print(f"{'case':22s} {'pyqcd(s)':>9s} {'opt(s)':>8s} {'ratio':>6s}")
for label, equation, operands in CASES:
    reference = np.einsum(equation, *operands, optimize="optimal")
    assert np.allclose(np.asarray(pe.contract(equation, *operands)), reference), label
    ours, theirs, ratio = compare(lambda: pe.contract(equation, *operands),
                                  lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:22s} {ours:9.2f} {theirs:8.2f} {ratio:6.2f}")
```

### GPU, production scale

```python
import statistics
import time
import warnings

import numpy as np
import pyqcd

import cupy as cp
import opt_einsum as oe
import pyqcd.einsum as pe

print("measuring pyqcd", pyqcd.__version__, pyqcd.__file__, "|",  # docs-skip: GPU 基准
      cp.cuda.runtime.getDeviceProperties(0)["name"].decode())


def one(call):
    call()
    cp.cuda.Device(0).synchronize()
    start = time.perf_counter()
    call()
    cp.cuda.Device(0).synchronize()
    return time.perf_counter() - start


def compare(ours, theirs, rounds=3):
    left, right, ratios = [], [], []
    for _ in range(rounds):
        a = one(ours)
        b = one(theirs)
        left.append(a)
        right.append(b)
        ratios.append(b / a)
    return statistics.median(left), statistics.median(right), statistics.median(ratios)


rng = np.random.default_rng(0)


def g(*shape):
    return cp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


n = 112                                                  # production-shaped nucleon 2pt
nucleon = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
nucleon_ops = [g(4, 4, n, n), g(4, 4, n, n), g(4, 4, n, n), g(4, 4), g(4, 4),
               g(4, n, n, n), g(4, n, n, n), g(4, 4)]
sandwich_ops = [g(4, 4), g(8192, 4, 4, 1024), g(4, 4)]
batched_ops = [g(16, 1024, 1024), g(16, 1024, 1024)]

CASES = [("nucleon 2pt nvec=112", nucleon, nucleon_ops),
         ("gamma sandwich 8192x1024", "ab,...bce,cd->...dae", sandwich_ops),
         ("batched GEMM 16x1024^2", "bij,bjk->bik", batched_ops)]

print(f"{'case':26s} {'pyqcd(s)':>9s} {'opt(s)':>8s} {'ratio':>6s}")
for label, equation, operands in CASES:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = cp.asnumpy(oe.contract(equation, *operands, optimize="dp"))
        assert np.allclose(cp.asnumpy(pe.contract(equation, *operands)), reference,
                           rtol=1e-10, atol=1e-12), label
        ours, theirs, ratio = compare(
            lambda: pe.contract(equation, *operands),
            lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:26s} {ours:9.2f} {theirs:8.2f} {ratio:6.2f}")
```

Before running anything, check which build you are measuring — a stale `pyqcd` in
`site-packages` silently turns every number into "PyQCD is slow":

```bash
python -m pyqcd --gpu-probe
```
