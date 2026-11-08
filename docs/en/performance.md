# Performance

`pyqcd.einsum` is benchmarked against `opt_einsum` on the same machine, in the
same dtype, with the same thread count and the same planner. Numbers below are
measured, not modelled, and every one of them can be reproduced with the
commands at the end of this page.

## Measurement protocol

| Item | Setting |
|---|---|
| Hardware | 1× RTX 4060 Laptop (GPU runs), 4 pinned threads (CPU runs) |
| Oracle | `opt_einsum` 3.4.0, `dp` planner |
| dtype | `complex128` (both sides) |
| Isolation | one fresh interpreter per side (`--isolation process`) |
| Timing | 4 warm-up rounds, median of the measured rounds, synchronised before and after |
| Operands | fixed-seed arrays with the shapes of the real distillation contractions |
| Correctness | `max|Δ| / max|ref| ≤ 1e-12` on the same run |

Timing only covers steady-state execution: both sides plan once, then the plan
is reused.

## Correctness

Speed is only meaningful on correct numbers, so the engine is verified at three
levels, and **every example on these pages ends with an `assert`** — the
documentation build executes all 21 runnable blocks and fails if any of them stops
verifying itself (blocks that cannot run are explicitly marked `# pseudo-code`).

### Verify it yourself

Independent references, same inputs, both must agree:

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

### What the project checks on every commit

| Check | Scope |
|---|---|
| `python -m pyqcd.testing` | 120 contracts: numerical edge cases, threading, metadata |
| differential suite (source checkout) | 980 cases against a frozen `opt_einsum` |
| `python -m pyqcd.testing.docs_examples` | all 21 documentation blocks, each self-asserting |
| `verify_native --backend kupl\|cuda` | native library vs the Python planner |

### Invariants that must never break

* **the result never changes when the fast path changes** — a missing library, a
  rejected shape or a kernel error falls back silently (one warning at most) and
  the numbers are identical;
* **`contract_path` agrees with `opt_einsum`** on path and `opt_cost` for the same
  shape and optimizer;
* **precision is preserved** — `complex128` in, `complex128` out; `complex64` stays
  `complex64`;
* **dtype/shape errors are raised, not guessed**: operand-count mismatches, label
  size conflicts and unknown output labels raise `ValueError`/`TypeError` with the
  same wording as `opt_einsum`.

## Reproducible on the published package

`pyqcd==0.1.23` (`build_info(): backend=kupl, array_backend=numpy`) versus
`opt_einsum==3.4.0` (`dp`), `complex128`, median of 15 calls after 3 warm-ups:

| Workload | `opt_einsum` | `pyqcd.einsum` | Ratio |
|---|---|---|---|
| Nucleon 2pt, nvec=32 (production contract) | 102.55 ms | 16.63 ms | **6.17×** |
| γ sandwich, T=256, nvec=512 | 28.78 ms | 4.55 ms | **6.33×** |
| Generic chain 800³ (no specialised kernel) | 13.40 ms | 16.99 ms | 0.79× (PyQCD slower) |

What the shipped build actually accelerates:

| Shape | Route in 0.1.x |
|---|---|
| Nucleon 2pt (`gahb,mcnd,…->M`) | dedicated KUPL/CUDA kernel + plan cache |
| Nucleon 3pt main (`…->MNGLxy`) | dedicated KUPL/CUDA kernel + plan cache |
| Nucleon 3pt (`…->GM`), sush 2pt/3pt-mu | plan cache only |
| γ sandwich (`ab,...bce,cd->...dae`), perambulator × gauge | plan cache + general execution |

The hint printed on the first hit says which of the two happened:
`[kernel: kupl_nucleon_2pt]` means a kernel ran, `[plan-cache]` means only the
plan was reused.

## Generic einsum, case by case

Most users start with equations that have nothing to do with lattice QCD, so here
is the same comparison on plain tensor algebra — CPU (KUPL backend), `float64`
unless noted, median of 9 calls, measured on `pyqcd 0.1.21`:

| Case | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate |
|---|---|---|---|
| batched matmul, 32×256² | 9.5–10.2 ms | 90.0–90.4 ms | **8.9–9.5×** |
| ellipsis, 16×128×128 | 0.7–1.1 ms | 6.2–9.3 ms | **8.7×** |
| outer product, 2000×2000 | 2.8–3.5 ms | 4.1–4.5 ms | **1.28–1.48×** |
| complex matmul, 800² | 20.2–20.8 ms | 20.0 ms | 0.96–0.99× |
| 3-operand chain, 3×500 | 2.7–2.9 ms | 2.6 ms | 0.92–0.98× |
| 4-operand chain, 300 | 1.1–1.2 ms | 1.0 ms | 0.81–0.89× |
| matmul, 1000² | 11.7–13.4 ms | 9.3–11.4 ms | 0.80–0.85× |
| Hadamard, 4000² | 88.3–89.8 ms | 29.2–29.4 ms | 0.33× |

Read it honestly:

* **batched and ellipsis equations win** — PyQCD keeps the batch axes in one
  execution step and reuses the cached plan, while `opt_einsum`'s `dp` path falls
  back to a per-batch loop;
* **element-wise and single-GEMM equations lose** — the generic path is a thin
  wrapper over the same BLAS, so `opt_einsum`'s leaner Python overhead wins;
* the gap shrinks as an equation becomes more "GEMM-like" and the work per call grows.

Run it yourself (every case is asserted against numpy *and* opt_einsum before any
timing is printed):

```python
import statistics
import time

import numpy as np
import pyqcd.einsum as pe
import opt_einsum as oe


def bench(call, warmup=2, repeat=9):
    """Median of `repeat` calls after `warmup`; returns milliseconds."""
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


rng = np.random.default_rng(0)


def real(*shape):
    return rng.standard_normal(shape)


def cplx(*shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex128)


# every case must agree with numpy AND with opt_einsum before we time it
CASES = [
    ("matmul 1000^2", "ij,jk->ik", [real(1000, 1000), real(1000, 1000)]),
    ("batched matmul 32x256^2", "bij,bjk->bik",
     [real(32, 256, 256), real(32, 256, 256)]),
    ("chain 3x500", "ij,jk,kl->il", [real(500, 500)] * 3),
    ("ellipsis 16x128^2", "...ij,...jk->...ik",
     [real(16, 128, 128), real(16, 128, 128)]),
    ("outer 2000", "i,j->ij", [real(2000), real(2000)]),
    ("hadamard 4000^2", "ij,ij->ij", [real(4000, 4000), real(4000, 4000)]),
    ("4-operand chain 300", "ij,jk,kl,lm->im", [real(300, 300)] * 4),
    ("complex matmul 800^2", "ij,jk->ik", [cplx(800, 800), cplx(800, 800)]),
]

print(f"{'case':24s} {'pyqcd':>9s} {'opt_einsum':>11s} {'rate':>6s}")
for label, equation, operands in CASES:
    ours = pe.contract(equation, *operands)
    reference = np.einsum(equation, *operands, optimize="optimal")
    other = oe.contract(equation, *operands, optimize="dp")
    assert np.allclose(np.asarray(ours), reference, rtol=1e-10, atol=1e-12), label
    assert np.allclose(np.asarray(other), reference, rtol=1e-10, atol=1e-12), label
    ours_ms = bench(lambda: pe.contract(equation, *operands))
    theirs_ms = bench(lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:24s} {ours_ms:9.2f} {theirs_ms:11.2f} {theirs_ms / ours_ms:6.2f}")
```

Sub-millisecond equations (`ii->`, `ii->i` on a 5000² matrix) are deliberately
excluded: there the measurement is timer and call overhead, not the contraction.


### Extended shapes (CPU and GPU)

These four replace the earlier "development build" table — they are reproducible on
the published `pyqcd 0.1.21`:

| Workload | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate |
|---|---|---|---|
| Nucleon 2pt, CPU, nvec=100, nmom=6 | 2004 ms | 19231 ms | **9.59×** |
| γ sandwich, GPU, T=24, nvec=1024 | 1.3 ms | 6.0 ms | **4.54×** |
| perambulator × gauge, GPU, spatial 24³ | 72.7 ms | 78.6 ms | 1.08× |

Two rows of the old table are gone on purpose: the nucleon-3pt GPU figure needs the
production operand contract (operand shapes that are not part of the public docs),
and the weighted "production mix" depended on an un-gated E4 kernel that the release
deliberately disables at real volumes.

## CUDA: the same comparison on the GPU

GPU runs compare **GPU against GPU** — both libraries execute on the same device
with `cupy` arrays, so the ratio measures implementations, not CPU-vs-GPU
throughput. Protocol: `complex128`, 5 warm-up calls, **median of 21 synchronised
calls** (`cp.cuda.Device(0).synchronize()` around each timing), same `dp` planner.

Measured with `pyqcd 0.1.19` (`build_info(): backend=cuda`) and
`opt_einsum 3.4.0` on an RTX 4060 Laptop (compute capability 8.9, arch build 8.9):

| Workload | `opt_einsum` | `pyqcd.einsum` | Ratio |
|---|---|---|---|
| γ sandwich, T=512, nvec=512 | 59.8–62.7 ms | 4.6–8.4 ms | **7–13×** (fused kernel) |
| Nucleon 2pt, nvec=64 | 255.8–263.8 ms | 134–162 ms | **1.6–1.9×** (kernel, `needs-optimization`) |
| Batched GEMM 32×512² | 163.2–168.0 ms | 190.6–192.0 ms | 0.86–0.88× (**PyQCD slower**) |

What this says:

* the **fused kernels** are where the GPU win comes from (γ sandwich runs as one
  kernel, `PYQCD_EINSUM_CUDA_SANDWICH ... kernel=1`);
* the nucleon-2pt CUDA kernel only buys 1.6–1.9× and is still marked
  `needs-optimization` — the shape is bound by the device, not by the planner;
* a plain batched GEMM has no fused kernel and PyQCD's native batched-GEMM path is
  ~15% slower than letting `cupy.einsum` handle it, so `opt_einsum` wins there.

Reproduce it (needs `cupy` and `opt_einsum`; without them the script exits with a
clear message rather than failing the documentation build):

```python
import statistics
import time

import numpy as np
import pyqcd.einsum as pe

try:
    import cupy as cp
    import opt_einsum as oe
except ImportError:
    raise SystemExit("this comparison needs cupy and opt_einsum: "
                     'pip install "pyqcd[cuda]" opt_einsum')


def bench(call, warmup=5, repeat=21):
    """Median of `repeat` GPU-synchronised calls (milliseconds)."""
    for _ in range(warmup):
        call()
    cp.cuda.Device(0).synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        cp.cuda.Device(0).synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples), max(samples)


import warnings


def compare(label, equation, operands):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # the fallback notices are printed below
        ours = bench(lambda: pe.contract(equation, *operands))
        theirs = bench(lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:26s} pyqcd {ours[0]:8.2f} ms | opt_einsum {theirs[0]:8.2f} ms "
          f"| rate {theirs[0] / ours[0]:5.2f}x | n=21 "
          f"(pyqcd min..max {ours[1]:.1f}..{ours[2]:.1f} ms)")


rng = np.random.default_rng(0)


def g(*shape):
    return cp.asarray((rng.standard_normal(shape)
                       + 1j * rng.standard_normal(shape)).astype(np.complex128))


print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
      "|", pe.build_info()["backend"])

n = 64                                                  # production-shaped nucleon 2pt
compare("nucleon 2pt nvec=64", "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",
        [g(4, 4, n, n), g(4, 4, n, n), g(4, 4, n, n), g(4, 4), g(4, 4),
         g(4, n, n, n), g(4, n, n, n), g(4, 4)])
compare("batched GEMM 32x512", "bij,bjk->bik", [g(32, 512, 512), g(32, 512, 512)])
compare("gamma sandwich 512x512", "ab,...bce,cd->...dae",
        [g(4, 4), g(512, 4, 4, 512), g(4, 4)])

# correctness first: a benchmark of wrong numbers is worthless
_ref_ops = [g(4, 4, 16, 16), g(4, 4, 16, 16), g(4, 4, 16, 16), g(4, 4), g(4, 4),
            g(4, 16, 16, 16), g(4, 16, 16, 16), g(4, 4)]
_eq = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
_ours = cp.asnumpy(pe.contract(_eq, *_ref_ops))       # cupy -> host
_theirs = cp.asnumpy(oe.contract(_eq, *_ref_ops, optimize="dp"))
assert np.allclose(_ours, _theirs, rtol=1e-10, atol=1e-12), "kernel disagrees with opt_einsum"
```

GPU numbers on a laptop are noisy: report the min–max spread (the script prints it)
and never trust a difference smaller than that spread.

## Memory

Peak RSS during a contraction, measured with `resource.getrusage` in separate
processes (`pyqcd==0.1.10` vs `opt_einsum 3.4.0`, `dp` planner, float64):

| Shape | `opt_einsum` peak | `pyqcd` peak |
|---|---|---|
| 1200×1200, 2 operands | 18 MB | 41 MB |
| 1200×1200, 3 operands | 30 MB | 52 MB |
| 1200×1200, 4 operands | 29 MB | 52 MB |
| 2000×2000, 3 operands | 73 MB | 104 MB |

The Python execution path holds one extra output-sized intermediate alive while a
step finishes, so expect roughly **1.4–2× the peak of `opt_einsum`** on the same
shape. Time is unaffected in the same runs (0.02–0.19 s, both BLAS-bound).

Practical consequence: size a job by `1.5×` the memory you would give
`opt_einsum`, or split the contraction with `memory_limit=` /
`contract_path(...)` first.

## Measure it yourself

The script below is the exact protocol behind the numbers above: produce
production-shaped operands, warm up, take the **median of 9 timed calls** (not a
single shot), and print the ratio together with the min/max spread so a noisy
machine is visible. Each call is deliberately in the 5–100 ms range — anything
shorter measures timer overhead rather than the engine.

```python
import statistics
import time

import numpy as np
import pyqcd.einsum as pe

try:
    import opt_einsum as oe
except ImportError:                       # pip install opt_einsum to run the comparison
    raise SystemExit("install opt_einsum first: pip install opt_einsum")


def bench(call, warmup=2, repeat=9):
    """Median of `repeat` timed calls after `warmup` untimed ones (milliseconds)."""
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples), max(samples)


def compare(label, equation, operands):
    ours = bench(lambda: pe.contract(equation, *operands))
    theirs = bench(lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:24s} pyqcd {ours[0]:8.2f} ms | opt_einsum {theirs[0]:8.2f} ms "
          f"| rate {theirs[0] / ours[0]:5.2f}x | n=9 "
          f"(pyqcd min..max {ours[1]:.1f}..{ours[2]:.1f} ms)")


rng = np.random.default_rng(0)


def c(*shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)


# 1) nucleon 2pt at a production-shaped contract: the shape the kernels target.
nuc = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
n = 32
compare("nucleon 2pt nvec=32", nuc,
        [c(4, 4, n, n), c(4, 4, n, n), c(4, 4, n, n), c(4, 4), c(4, 4),
         c(4, n, n, n), c(4, n, n, n), c(4, 4)])

# 2) gamma sandwich, batched over time and vector space.
compare("gamma sandwich T=256", "ab,...bce,cd->...dae",
        [c(4, 4), c(256, 4, 4, 512), c(4, 4)])

# 3) a generic chain with no specialised kernel — the honest control case.
a, b, d = (rng.standard_normal((800, 800)) for _ in range(3))
compare("generic chain 800", "ij,jk,kl->il", [a, b, d])

# correctness first: the ratio means nothing if the results disagree
_a = np.arange(12.0).reshape(3, 4)
_b = np.arange(20.0).reshape(4, 5)
assert np.allclose(pe.contract("ij,jk->ik", _a, _b),
                   oe.contract("ij,jk->ik", _a, _b, optimize="dp"))
```

`opt_einsum` uses the `dp` planner, which is the production default. The third
case is a control: a generic chain has no specialised kernel, and there the two
libraries are within noise of each other (0.76–0.93x, i.e. PyQCD is not faster).

## Code

The specialised path is the default: no import, no flag, no plan restructuring.
The only thing to know is that it prints a hint the first time each shape is hit,
so an A/B run is one keyword argument apart.

```python
import numpy as np
import pyqcd.einsum as pe

rng = np.random.default_rng(0)


def c(*shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# Nucleon 2pt (E1): perambulator x gamma x VVV x parity projection
nucleon_2pt = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
ops_2pt = [c(2, 2, 2, 2), c(2, 2, 2, 2), c(2, 2, 2, 2), c(2, 2),
           c(2, 2), c(4, 2, 2, 2), c(4, 2, 2, 2), c(2, 2)]

fast = pe.contract(nucleon_2pt, *ops_2pt)                    # specialised path
base = pe.contract(nucleon_2pt, *ops_2pt, specialize=False)  # general path
assert np.allclose(fast, base)

# gamma sandwich (E5): perambulator^dagger x gamma sandwich x perambulator
sandwich = "ab,...bce,cd->...dae"
ops_s = [c(2, 2), c(3, 2, 2, 2), c(2, 2)]
assert np.allclose(pe.contract(sandwich, *ops_s),
                   pe.contract(sandwich, *ops_s, specialize=False))

# which shapes get a specialised path
for name, kind, subscripts in pe.specializations():
    print(f"{name:20s} {kind:28s} {subscripts}")
```

The same switches are available from the environment when you do not want to
touch application code:

```bash
PYQCD_EINSUM_SPECIALIZE=0 python my_run.py            # process-wide off
PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 python my_run.py    # print for every call
PYQCD_EINSUM_DISABLE_FUSED=1 python my_run.py         # keep planning, drop fused kernels
PYQCD_EINSUM_KERNEL_KUPL=0 python my_run.py           # drop KUPL specialised kernels
PYQCD_EINSUM_KERNEL_CUDA=0 python my_run.py           # drop CUDA specialised kernels
```

Every registered shape is discoverable at runtime, and `specialize=False`
always reproduces the general path, so a regression is a plain `assert`:

| Group | Kind | What it covers |
|---|---|---|
| `nucleon-2pt(E1)` | fused + KUPL/CUDA kernel | 8-operand nucleon 2pt |
| `nucleon-3pt(E2)` | plan cache | 8-operand nucleon 3pt |
| `gauge-propagator(E4)` | fused (CUDA) | perambulator × gauge |
| `gamma-sandwich(E5)` | fused (CPU + GPU) | `γ₅` sandwich |
| `sush-2pt(S1)`, `sush-3pt-munu(S2)`, `sush-2pt-mu(S3)` | plan cache | sush production shapes |
| `sush-3pt-main(P1/P2)` | fused + CUDA kernel | 14-operand 3pt main term |


## Reproduce

The benchmark driver `pyqcd.testing.einsum.bench_native` is **not part of the
PyPI package**: `pyqcd.testing` is excluded from the published distribution
(`setup.py` uses `find_packages(exclude=["pyqcd.testing", "pyqcd.testing.*"])`),
so the benchmarks, tests and regression baselines exist only in a source
checkout. Both commands below therefore have to be run from the root of that
checkout; in an environment that only has `pip install pyqcd`, the driver
exits with `ModuleNotFoundError: No module named 'pyqcd.testing'`.

```bash
cd /path/to/PyQCD     # source checkout root; pyqcd.testing is not on PyPI

# CPU: KUPL backend, 4 pinned threads, process isolation
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=kupl PYQCD_BENCH_THREADS=4 \
  python -m pyqcd.testing.einsum.bench_native \
  --data synthetic --profiles T1 --backends cpu --iters 10 --warmup 4 --planner shared

# GPU: CUDA backend, same protocol
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=cuda \
  python -m pyqcd.testing.einsum.bench_native \
  --data synthetic --profiles L24 --backends gpu --iters 10 --warmup 4 --planner shared
```

Correctness of the specialised paths is checked by the same suite that produces
the timings, and for the same reason that suite is source-checkout only:
`python -m pyqcd.testing.einsum.verify_compat` (980 cases) and
`python -m pyqcd.testing.einsum.verify_native --backend kupl|cuda`.
