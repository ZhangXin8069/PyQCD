# PyQCD

**English** | [中文](https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD/-/blob/main/README.zh-CN.md)

PyQCD is a lattice-QCD toolkit in Python with optional native acceleration.
It provides building blocks for propagator and operator contractions, gradient
flow, renormalisation and gluon TMD-PDF extraction, plus a self-contained
contraction engine with an **`opt_einsum`-compatible API**.

## Install

Requires **Python 3.9+**. Python 3.8 is no longer supported because the
current PEP 639 build metadata requires `setuptools>=77`.

```bash
pip install pyqcd
```

The **KUPL native kernel is required by default**: `pip install` compiles it and
aborts with an actionable message if no C++ toolchain is available, instead of
silently handing you a package that falls back to the pure-Python planner.
Install `g++`/`c++` plus `bash` and `binutils`, or opt into the pure-Python
planner explicitly with `PYQCD_SKIP_NATIVE_BUILD=1 pip install pyqcd`
(`PYQCD_EINSUM_REQUIRE_NATIVE=0` does the same). CUDA stays opportunistic: a
missing `nvcc` only costs you the GPU kernels.

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

## Release 0.4.1 — frozen qcd-einsum stable release

`0.4.1` is the **key stable release** for `pyqcd.einsum` (baseline tag `stab31`).

* New: `python -m pyqcd.einsum.verify_specialized "<eq>"` reports which
  specialised path an equation takes — corpus hit, topology digest, declared
  kernels and provenance (`seq` / T-id / roles) — **without contracting**:
  no operands are built, no global state is touched. Exit codes: `0` hit,
  `1` outside the corpus, `2` malformed equation. Library entry point:
  `pyqcd.einsum.query_specialization(eq)`.
* Frozen: until the next key qcd-einsum stable release, changes that affect
  qcd-einsum behaviour are **disallowed by default** unless the user explicitly
  forces them. See `pyqcd/einsum/AGENTS.md` §0 for the three exceptions.
* Everything from `0.4.0` (resource-gated topology specialisation over 8,638
  equations) is unchanged; numerics, dispatch and native ABI are untouched.

## Release 0.4.0

`0.4.0` is the production-specialisation release for `pyqcd.einsum`.

* The topology-specialised registry covers **8,638 equations** with 6–16
  operands on both KUPL and CUDA. Every plan passes a fail-open resource gate
  before native allocation.
* On the 43-path full-scene audit, CPU is not slower than PyQCD's internal
  generic path on **42/43** cases (median selected/generic `0.861x`), and CUDA
  on **42/43** cases (median `0.276x`). The two retained exceptions are
  `seq=856` on CPU (+0.7%) and `seq=6911` on CUDA (+36.2%).
* A representative CPU long-chain case, `seq=6585`, drops from **22.445 s** to
  **10.310 s** (`0.459x`).
* CUDA is not slower than `opt_einsum 3.4.0` (`dp`) on **42/43** audited paths.
  CPU is not slower on **36/43**: this release does **not** claim universal CPU
  superiority over `opt_einsum`; seven CPU plans remain slower.

The specialised route is a production optimisation, not a promise that every
shape is faster. Native resource limits, pack-heavy low-FLOP plans, unavailable
kernels and registered native failures deliberately fall back to the generic
path. Results are unchanged.

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
| `pyqcd.parallel` | read-only CPU, memory and GPU resource detection |
| `pyqcd.einsum` | contraction engine (this package's headline feature) |

## Contraction engine

* Pure-Python planning (`greedy` / `optimal` / `branch` / `dp`) is always
  available.
* Two optional native backends: **KUPL** (CPU) and **CUDA** (GPU). Both export
  the same C ABI and share one planning core.
* Fallback is a hard invariant: a missing library, a failed native call or a
  rejected shape silently degrades to the Python path — results never change.
* The complete 8,638-equation lattice-QCD topology corpus gets resource-gated
  specialised dispatch; it is enabled by default and can be turned off per call
  (`contract(..., specialize=False)`) or per process
  (`PYQCD_EINSUM_SPECIALIZE=0`).

Introspection:

```python
import pyqcd.einsum as pe

pe.build_info()        # backend, native_available, array_backend, native_library
pe.specializations()   # registered equation shapes (8,638 in this release)
```

## Performance

These are full-scene benchmarks from an A100 node, not smoke tests. Inputs use
`Nx=Ny=Nz=32`, `Lt=64`, `Nev=100`, `Ns=4`, `Mom=8`, `complex128`, and each
specialised result is checked against the generic path before timing. Ratios are
`selected/generic`; values below `1` are faster.

| Evidence set | CPU | GPU |
|---|---|---|
| 43 stratified production paths: median ratio | **0.861x** | **0.276x** |
| 43 paths: not slower than PyQCD generic | **42/43** | **42/43** |
| 43 paths: at least 10% faster than PyQCD generic | **26/43** | **40/43** |
| 43 paths vs `opt_einsum(dp)`: not slower | **36/43** | **42/43** |

The six core paths (ratio `<1` means faster):

| `seq` | CPU vs generic | GPU vs generic | CPU vs `opt_einsum(dp)` | GPU vs `opt_einsum(dp)` |
|---:|---:|---:|---:|---:|
| 2584 | 0.856x | 0.265x | 0.598x | 0.152x |
| 866 | 0.936x | 0.248x | 0.891x | 0.213x |
| 855 | 0.982x | 0.177x | 0.842x | 0.130x |
| 129 | 0.791x | fallback, 1.017x | 1.297x | 1.179x |
| 1863 | 0.974x | 0.157x | 0.837x | 0.025x |
| 6585 | **0.459x** | fallback | not completed | 0.606x, fallback |

All **8,638** plans were additionally audited without executing giant tensors:
the CPU gate admits 3,833 plans and blocks 4,805; the CUDA gate admits 2,864 and
blocks 5,774. This prevents plans with multi-terabyte static arenas, oversized
single GEMMs, or pack-heavy low-FLOP work from reaching the specialised route.
Full-scene maximum relative errors were `6.95e-15` on CPU and `6.68e-15` on GPU.

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

The gain comes from topology recognition, resource-aware plan selection, fused
or dedicated execution where profitable, lifetime-based workspace reuse and
fallback when the resource envelope is unsafe. It is not universal:
`opt_einsum(dp)` still wins on seven audited CPU paths, two internal full-scene
paths are retained as known counterexamples, and the gates intentionally send
unsafe plans back to the generic path. Full protocol, evidence and reproduction:
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
