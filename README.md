# PyQCD

**English** | [中文](README.zh-CN.md)

PyQCD is a lattice-QCD toolkit in Python with optional native acceleration.
It provides building blocks for propagator and operator contractions, gradient
flow, renormalisation and gluon TMD-PDF extraction, plus a self-contained
contraction engine with an **`opt_einsum`-compatible API**.

## Install

```bash
pip install pyqcd
```

The native kernels are **optional**: without a C++ compiler (or without CUDA)
the package installs anyway and the contraction engine runs its pure-Python
planner — same results, only slower.

Set `PYQCD_EINSUM_REQUIRE_NATIVE=1` to make a missing toolchain a hard error.

## Documentation

* English — https://py-qcd.readthedocs.io/en/latest/usage.html
* 中文 — https://py-qcd.readthedocs.io/zh-cn/latest/usage.html

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

Measured against `opt_einsum` 3.4.0 on the same machine, same `complex128`
dtype, same thread count and same `dp` planner:

| Workload | `opt_einsum` | `pyqcd.einsum` | Ratio |
|---|---|---|---|
| Nucleon 3pt (GPU, specialised) | 285.40 ms | 30.74 ms | **9.29x** |
| γ sandwich, L24 (CPU) | 1188.9 ms | 142.7 ms | **8.33x** |
| γ sandwich, L24 (GPU) | 0.373987 s | 0.110959 s | **3.371x** |
| Nucleon 2pt (CPU, N=100) | 120.43 ms | 22.65 ms | **5.32x** |
| perambulator × gauge, L24 (GPU) | 1.112793 s | 1.095407 s | 1.016x (parity) |

Weighted by the shape mix of a real production run the aggregate is roughly
**2.82x** (4757 s → 1688 s, model estimate).

The gain comes from recognising a handful of lattice-QCD equation shapes and
routing them to purpose-built kernels (fusion, buffer reuse) instead of a
generic BLAS chain. It is not universal: on the GPU nucleon 2pt/3pt *against
opt_einsum* both sides sit at the hardware limit (~1x), small tensors lose to
per-call overhead, and shapes without a specialised kernel run the generic
path. Full protocol and limitations:
https://py-qcd.readthedocs.io/en/latest/performance.html

## Tests

Run from a source checkout:

```bash
python -m pyqcd.testing                       # integration suite
bash pyqcd/testing/simple/run-all.sh          # per-function API parity vs opt_einsum
bash pyqcd/testing/einsum/simple/run-all.sh   # einsum sub-system suite
```

## License

MIT
