# PyQCD

**English** | [中文](https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD/-/blob/main/README.zh-CN.md)

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

Check what you actually got (no imports needed):

```bash
python -m pyqcd            # version, native backend, optional deps, subpackages
python -m pyqcd --json     # machine-readable
```

> **CUDA builds.** The build auto-detects your GPU (`nvidia-smi compute_cap`) and
> always embeds PTX, so current cards work out of the box; set
> `PYQCD_CUDA_ARCHITECTURES=<cap>` (e.g. `8.9`) to get native SASS, or when building on a
> GPU-less login node for a known target.

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

Numbers you can reproduce on the **published package**, measured with
`pyqcd==0.1.14` (`build_info(): backend=kupl, array_backend=numpy`),
`opt_einsum==3.4.0` with the `dp` planner, `complex128`, median of 15 calls after
3 warm-up calls:

| Workload | `opt_einsum` | `pyqcd.einsum` | Ratio |
|---|---|---|---|
| Nucleon 2pt, nvec=16 (production contract) | 12.95 ms | 2.15 ms | **6.02x** |
| Nucleon 2pt, nvec=24 | 41.45 ms | 7.11 ms | **5.83x** |
| γ sandwich, T=8, nvec=64 | 0.169 ms | 0.058 ms | **2.91x** |

Those three run through the shipped kernels/plan cache with no extra setup.
Dedicated kernels in this release cover the nucleon 2pt and 3pt shapes only;
other lattice shapes (γ sandwich, perambulator × gauge, the sush variants) get
the plan cache and the general execution layer.

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
