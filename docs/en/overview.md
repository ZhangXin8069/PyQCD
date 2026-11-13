# Overview

`pyqcd` is a lattice-QCD toolkit published on PyPI. The long-term goal of the
project is gluon TMD-PDF extraction with gradient-flow renormalisation; the
package itself is the reusable tooling around that chain.

```bash
pip install pyqcd
```

Everything on this site is available from that single installation; no source
checkout is required. Native acceleration is optional at runtime, but the
default installation builds KUPL and therefore needs a C++17 toolchain.

```python
import pyqcd
import pyqcd.einsum as pe

print(pyqcd.__version__)    # e.g. 0.4.2
print(pe.build_info())      # backend / native_available / native_library

info = pe.build_info()
assert pyqcd.__version__.count(".") == 2, pyqcd.__version__
assert info["backend"] in {"cuda", "kupl", "python"}, info["backend"]
```

`pyqcd/__init__.py` intentionally stays almost empty (only `__version__`), so
everything is reached through subpackages.

## What the package provides

| Subpackage | Highlights |
|---|---|
| `pyqcd.lattice` | `Nc`, `Nd`, `Ns`, `fm2GeV`, DeGrand–Rossi `gamma(i)`, `Mom_times_sigma`, `SU2combine` / `SU2decompose`, `cg_coefficient` |
| `pyqcd.gauge` | `wilson_loop`, `polyakov_loop`, `clover_field_strength`, `topological_charge` (+ density/average variants) |
| `pyqcd.smear` | `stout_smear`, `hyp_smear`, `proj_su3` |
| `pyqcd.operator` | `flowed_gluon_ope`, `compute_dual_field_strength`, `FieldStrengthCache`, gradient-flow OPE schema constants |
| `pyqcd.vertex` | `Mom_VdV_sink_t`, `Mom_VVV_sink_t`, `VdV_sink_t_link`, `apply_momentum_smearing`, eigenvector compression (`compress_matrix_V1`…`V4`), `check_orthonormal` |
| `pyqcd.contraction` | `dynamic_contraction`, `parity_and_boundary`, `conjugate_operator`, `classify_structure`, `identify_equivalent_diagrams`, `contract_donghx_2pt_pair` |
| `pyqcd.renorm` | `gradient_flow_renormalized_tmd`, `flow_action_density`, `flow_derivative`, `flow_time_gev_m2`, `flow_scale_mask`, NLO matching helpers (`A_s`, `C`, `C_gluon_ratio`, `Si`, `M_mu_lambda_nu_rho`) |
| `pyqcd.tools` | backend switching (`set_backend`, `get_backend`, `get_backend_name`, `get_precision`), `cached_contract`, `ArraySlicer`, `creat_mom_list`, `dump_env` |
| `pyqcd.analysis` | `Bootstrap`, `FitParams`, `AnaParams`, ratio / effective-mass / energy / FH fitting and plotting helpers |
| `pyqcd.pipeline` | centralised configuration and multi-step pipeline; `run_pipeline` executes the numbered steps |
| `pyqcd.parallel` | `detect_resources`, `cpu_threads`, `gpu_info` (read-only resource probing; no MPI scheduling) |
| `pyqcd.einsum` | contraction engine: an `opt_einsum`-compatible API plus optional native backends |

## The contraction engine

`pyqcd.einsum` re-implements the public `opt_einsum` API so it can be dropped in
wherever `opt_einsum` is used:

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)

pe.contract("ij,jk->ik", a, b)                        # same as a @ b
pe.contract_path("ij,jk->ik", a, b)                   # (path, PathInfo)
pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))   # reusable expression

assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)
_path, _info = pe.contract_path("ij,jk->ik", a, b)
assert int(_info.opt_cost) > 0
assert np.allclose(pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))(a, b), a @ b)
```

### Layering

| Layer | Modules |
|---|---|
| Public API | `contract`, `contract_path`, `contract_expression`, `PathInfo`, `paths`, `backends`, `shared_intermediates` |
| Planning | `_plan_py.py` (pure-Python greedy/optimal/branch/dp), `helpers.py`, `parser.py`, `blas.py` |
| Execution | `_execute.py` (BLAS dispatch + buffer pooling) |
| Native | `_native.py` (ctypes) and `cuda/` (Cython extension) |

### Native backends (optional)

| Backend | Usable for | Built during `pip install` when |
|---|---|---|
| KUPL | CPU | a C++17 compiler is available; this is the default baseline and a missing toolchain fails the install |
| CUDA | GPU | `nvcc` and Cython are available |

Use `PYQCD_SKIP_NATIVE_BUILD=1` (or `PYQCD_EINSUM_REQUIRE_NATIVE=0`) to request
a pure-Python planner explicitly. CUDA is always opportunistic and never causes
the installation to fail.

### Fallback invariants

- A failed or unavailable native execution at run time switches to the generic
  path: same results, slower.
- `_native.py` only does `ctypes`; importing it never loads a library.
- Any native failure emits a single warning and falls back to Python instead of
  raising.
- A topology-specialised plan must pass CPU/GPU arena, single-GEMM and
  pack-heavy resource gates before native allocation.

### Introspection helpers

```python
import pyqcd.einsum as pe

pe.build_info()        # backend / native_available / array_backend / native_library
pe.specializations()   # equation-shape specialisations currently registered

info = pe.build_info()
assert isinstance(info, dict) and "backend" in info
assert len(pe.specializations()) >= 1
```

## Production snapshot (first shipped in the 0.4 series)

Full-scene A100 evidence uses `Nx=Ny=Nz=32`, `Lt=64`, `Nev=100`, `Ns=4`,
`Mom=8` and `complex128`. Ratios are `selected/generic`, so values below `1`
are faster.

| Evidence | CPU | GPU |
|---|---|---|
| 43 stratified paths, median | **0.861×** | **0.276×** |
| 43 paths, not slower than PyQCD generic | **42/43** | **42/43** |
| 43 paths, not slower than `opt_einsum` 3.4.0 (`dp`) | **36/43** | **42/43** |
| Core long-chain `seq=6585` | **0.459×** | fallback |

The full registry contains 8,638 topology paths. A plan-level audit admitted
3,833 CPU and 2,864 GPU plans; unsafe or unprofitable plans fall back before
native allocation. The release therefore claims broad speedups over PyQCD's
generic route, not universal CPU superiority over `opt_einsum`. See
:doc:`performance` for the exceptions, protocol and reproduction.
