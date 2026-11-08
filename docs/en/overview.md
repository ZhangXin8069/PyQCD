# Overview

PyQCD is a lattice-QCD toolkit written in Python. The long-term goal of the
project is gluon TMD-PDF extraction with gradient-flow renormalisation; the
package itself is a set of reusable building blocks for that chain and for
general lattice-QCD work.

`pyqcd/__init__.py` intentionally stays almost empty (only `__version__`), so
everything is reached through subpackages.

## What the subpackages provide

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
| `pyqcd.einsum` | contraction engine: an `opt_einsum`-compatible API plus optional native KUPL/CUDA backends |
| `pyqcd.testing` | integration test entry point and all runnable examples/regression suites |

## The contraction engine

`pyqcd.einsum` re-implements the public `opt_einsum` API so it can be dropped in
wherever `opt_einsum` is used:

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)

pe.contract("ij,jk->ik", a, b)          # same as a @ b
pe.contract_path("ij,jk->ik", a, b)     # (path, PathInfo)
pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))   # reusable expression
```

### Layering

| Layer | Modules |
|---|---|
| Public API | `contract`, `contract_path`, `contract_expression`, `PathInfo`, `paths`, `backends`, `shared_intermediates` |
| Planning | `_plan_py.py` (pure-Python greedy/optimal/branch/dp), `helpers.py`, `parser.py`, `blas.py` |
| Execution | `_execute.py` (BLAS dispatch + buffer pooling) |
| Native | `pyqcd/einsum/_native.py` (ctypes) and `pyqcd/einsum/cuda/` (Cython); C ABI in `cpp/` |

### Native backends

| Backend | Sources | Artifact |
|---|---|---|
| KUPL (CPU) | `cpp/kupl/einsum/` | `libpyqcdeinsum_kupl.so` |
| CUDA (GPU) | `cpp/cuda/einsum/` | `libpyqcdeinsum_cuda.so` + Cython extension |

Both export the same eight `Qcd` entry points; the planning core is compiled
into both, so paths and costs are identical by construction.

### Fallback invariants

- A missing `.so` switches to the pure-Python planner: same results, slower.
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
