# PyQCD

**English** | [中文](https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD/-/blob/main/README.zh-CN.md)

PyQCD is a lattice-QCD toolkit in Python with optional native acceleration.
It provides building blocks for propagator and operator contractions, gradient
flow, renormalisation and gluon TMD-PDF extraction, plus a self-contained
contraction engine with an **`opt_einsum`-compatible API**.

## Install

Requires **Python 3.8+** (3.8 is the oldest interpreter with maintained
`numpy`/`scipy` wheels at the declared floors).

```bash
pip install pyqcd
```

The native kernels are **optional**: without a C++ compiler (or without CUDA)
the package installs anyway and the contraction engine runs its pure-Python
planner — same results, only slower.

Set `PYQCD_EINSUM_REQUIRE_NATIVE=1` to make a missing toolchain a hard error.

Check what you actually got (no imports needed):

```bash
python -m pyqcd            # version, native backend, optional deps, subpackages
python -m pyqcd --json     # machine-readable
```

> **CUDA builds.** The build auto-detects your GPU (`nvidia-smi compute_cap`) and
> always embeds PTX, so current cards work out of the box; set
> `PYQCD_CUDA_ARCHITECTURES=<cap>` (e.g. `8.9`) to get native SASS, or when building on a
> GPU-less login node for a known target.

> **Threads are tuned per call.** Two profiles are switched automatically (`kernel` for the
> specialised shapes, `generic` for BLAS chains) on the already-loaded runtimes; override with
> `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` (or `PYQCD_EINSUM_OMP_THREADS` /
> `PYQCD_EINSUM_BLAS_THREADS`), disable with `PYQCD_EINSUM_THREADS=off`, inspect with
> `pyqcd.einsum.thread_info()`.

## Documentation

* English — https://py-qcd.readthedocs.io/en/latest/usage.html
* 中文 — https://py-qcd.readthedocs.io/zh-cn/latest/usage.html

## Repositories

| Platform | Role | URL |
|---|---|---|
| GitLab (gitlab.itp.ac.cn) | 国内开发 — primary development | https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD |
| Gitee | 国内发布 — China release mirror | https://gitee.com/zhangxin8069/PyQCD |
| GitHub | 国外发布 — global release mirror | https://github.com/zhangxin8069/PyQCD |

## Quick start

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)

pe.contract("ij,jk->ik", a, b)                       # like a @ b
path, info = pe.contract_path("ij,jk->ik", a, b)     # (path, PathInfo)
int(info.opt_cost)

expr = pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))
expr(a, b)                                           # reusable plan
```

The public surface mirrors `opt_einsum`: `contract`, `contract_path`,
`contract_expression`, `shared_intermediates`, `PathInfo`, `paths`, `backends`,
`helpers`, `parser`, `blas`.

## What is inside

| Subpackage | Content |
|---|---|
| `pyqcd.lattice` | lattice constants, DeGrand–Rossi `gamma`, SU(2) Clebsch–Gordan |
| `pyqcd.gauge` | Wilson/Polyakov loops, Clover field strength, topological charge |
| `pyqcd.smear` | HYP and Stout smearing |
| `pyqcd.operator` | Clover / dual field strength, flowed gluon OPE operators |
| `pyqcd.vertex` | VdV/VVV vertices, momentum smearing, eigenvector compression |
| `pyqcd.contraction` | Wick contraction, baryon operators, distillation contractions |
| `pyqcd.renorm` | gradient flow, Z_R, hybrid scheme, NLO matching, TMD extraction |
| `pyqcd.analysis` | resampling, fits, ratio / effective-mass / energy helpers |
| `pyqcd.pipeline` | centralised configuration and multi-step pipeline |
| `pyqcd.parallel` | resource detection and MPI meta-task planning |
| `pyqcd.einsum` | contraction engine (this package's headline feature) |

## Contraction engine

* Pure-Python planning (`greedy` / `optimal` / `branch` / `dp`) is always
  available.
* Two optional native backends: **KUPL** (CPU) and **CUDA** (GPU). Both export
  the same C ABI and share one planning core.
* Fallback is a hard invariant: a missing library, a failed native call or a
  rejected shape silently degrades to the Python path — results never change.
* A small set of lattice-QCD equation shapes gets a specialised execution path;
  it is enabled by default and can be turned off per call
  (`contract(..., specialize=False)`) or per process
  (`PYQCD_EINSUM_SPECIALIZE=0`).

Introspection:

```python
import pyqcd.einsum as pe

pe.build_info()        # backend, mpi, mpi_label, native_library
pe.mpi_enabled()       # whether the KUPL library has real MPI support
pe.specializations()   # registered equation shapes
```

MPI sharding exists only in the KUPL backend and only shards a single output
axis; the CUDA backend has no MPI implementation.

## Performance

Fifteen cases — one per structural category the library routes differently, not a
highlight reel — measured against `opt_einsum` 3.4.0 (`dp`), alternating which
implementation runs first every round, three rounds per case. The same cases and the
same shapes appear in both columns, so the two device columns are directly comparable.

| Case (category) | CPU | GPU |
|---|---|---|
| nucleon 2pt, nvec=112 (kernel) | **18.6x** | **2.68x** |
| batched GEMM, 16×1024² | **11.9x** | **1.09x** |
| ellipsis, 256×512² | **9.18x** | **1.11x** |
| γ sandwich, 8192×1024 (kernel) | **7.83x** | **4.04x** |
| many small batches, 4096×64² | **2.32x** | 0.96x |
| outer product, 8000 | **1.62x** | 1.00x |
| element-wise, 12000² | **1.25x** | 1.02x |
| chain 3 ops, 3×3072 | **1.07x** | **1.10x** |
| matmul 2D complex, 2048² | 1.04x | **1.15x** |
| full reduction, 12000² | 1.02x | **2.13x** |
| partial reduction, 16000² | 1.02x | 1.00x |
| batch × matrix, 8×1024×512 | 1.01x | **1.13x** |
| chain 4 ops, 4×2048 | 1.00x | **1.10x** |
| matmul 2D real, 4096² | 0.99x | **1.05x** |
| hyper-edge, 2048² | 0.93x | 1.02x |

Read this honestly: the generic shapes sit at parity (0.93–1.15x). The double-digit
CPU multiples come from two effects, both measured — `opt_einsum` materialises an
89.9 M-element intermediate for the nucleon 2pt network (1.44 GB vs 11.8 M input
elements), and it does not turn a batch axis into one batched BLAS call (2.99 s vs
0.28 s for `numpy.matmul` on the same arrays). Where it does use BLAS, we are level
with it, not 10× ahead.

```python
import numpy as np
import pyqcd.einsum as pe

expr = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"   # nucleon 2pt
fast = pe.contract(expr, *operands)                    # specialised path (default)
base = pe.contract(expr, *operands, specialize=False)  # general path
assert np.allclose(fast, base)                         # same numbers, different speed
```

Per-call `specialize=False` and the process-wide `PYQCD_EINSUM_SPECIALIZE=0`
switch the optimisation off; both paths must agree numerically.

- **Memory.** The Python execution path peaks at roughly 1.4–2× the RSS of
  `opt_einsum` on the same shape (measured 18→41 MB at 1200², 73→104 MB at
  2000²), because one extra output-sized intermediate stays alive per step.
  Size long jobs accordingly.

The gain comes from recognising a handful of lattice-QCD equation shapes and
routing them to purpose-built kernels (fusion, buffer reuse) instead of a
generic BLAS chain. It is not universal: on the GPU nucleon 2pt/3pt *against
opt_einsum* both sides sit at the hardware limit (~1x), small tensors lose to
per-call overhead, and shapes without a specialised kernel run the generic
path. Full protocol and limitations:
https://py-qcd.readthedocs.io/en/latest/performance.html

### PyTorch

PyTorch tensors are supported through `pip install "pyqcd[torch]"`: `contract()`
dispatches on the operand type, so `torch.Tensor` in → `torch.Tensor` out on the same
device (executed by `torch.einsum`, not by the native libraries). Convert to numpy or
cupy at the boundary if you want the native kernels. Full matrix and examples:
https://py-qcd.readthedocs.io/en/latest/usage.html#using-pytorch-tensors

## Tests

These suites live in `pyqcd/testing/`, which is **not** part of the PyPI
distribution; run them from a source checkout:

```bash
python -m pyqcd.testing                       # integration suite
bash pyqcd/testing/simple/run-all.sh          # per-function API parity vs opt_einsum
bash pyqcd/testing/einsum/simple/run-all.sh   # einsum sub-system suite
```

## License

MIT
