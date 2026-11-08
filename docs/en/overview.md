# Overview

`pyqcd` is a lattice-QCD toolkit published on PyPI. The long-term goal of the
project is gluon TMD-PDF extraction with gradient-flow renormalisation; the
package itself is the reusable tooling around that chain.

```bash
pip install pyqcd
```

Everything on this site is available from that single installation — no source
checkout and no manual build step are required. Native acceleration is
optional, and the package installs either way.

```python
import pyqcd
import pyqcd.einsum as pe

print(pyqcd.__version__)    # e.g. 0.1.5
print(pe.build_info())      # backend / mpi / mpi_label / native_library
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
| `pyqcd.parallel` | `detect_resources`, `plan_parallel`, `run_meta_task`, `run_parallel_pipeline`, `format_plan` |
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
| KUPL | CPU | a C++17 compiler is available |
| CUDA | GPU | `nvcc` and Cython are available |

When no toolchain is present the package still installs and the engine runs its
pure-Python planner: same results, only slower.

### Fallback invariants

- A missing native library switches to the pure-Python planner: same results, slower.
- `_native.py` only does `ctypes`; importing it never loads a library.
- Any native failure emits a single warning and falls back to Python instead of
  raising.

### Introspection helpers

```python
import pyqcd.einsum as pe

pe.build_info()        # backend / mpi / kupl_mpi_build / mpi_label / native_library
pe.mpi_enabled()       # whether the KUPL binary has real MPI support
pe.specializations()   # equation-shape specialisations currently registered
```

## Performance snapshot

Measured on a laptop RTX 4060 with `complex128`, against `opt_einsum` 3.4.0:

| Workload | Ratio vs `opt_einsum` |
|---|---|
| Nucleon 3pt (GPU, specialised) | 9.29× (30.74 ms vs 285.40 ms) |
| γ sandwich L24 (CPU / GPU) | 8.33× / 3.371× |
| Nucleon 2pt (CPU, N=100) | 5.32× (22.65 ms vs 120.43 ms) |
| perambulator × gauge L24 (GPU) | 1.016× (parity) |

See :doc:`performance` for the protocol, the aggregate production estimate and
the cases where there is no gain.
