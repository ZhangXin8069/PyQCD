# Overview

`pyqcd` is a lattice-QCD toolkit published on PyPI. The long-term goal of the
project is gluon TMD-PDF extraction with gradient-flow renormalisation; the
package itself is the reusable tooling around that chain.

```bash
pip install -U pyqcd
```

Everything on this site is available from that single installation; no source
checkout is required. Native acceleration is optional at runtime and only changes
speed, never results.

```python
import pyqcd
import pyqcd.einsum as pe

print(pyqcd.__version__)    # e.g. 0.5.15
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

### Configuration table

The engine can keep a per-equation configuration table with exactly one owner:
`--tune` writes it, ordinary calls only read it.

* File: `pyqcd.einsum.profile.csv`, under `PYQCD_EINSUM_PROFILE_DIR` (default
  `$XDG_CACHE_HOME/pyqcd.einsum`, i.e. `~/.cache/pyqcd.einsum`). The bare name
  `profile.csv` is **not** the table. The directory is per **user**, not per working
  directory, so a table tuned once serves every project.
* Reference table: `pyqcd.einsum.record.csv`, in the same directory, same columns as
  the profile table plus a `peak_memory_mb` column. It records **every** group of the
  sweep and is never consulted for dispatch (nothing on the hot path reads it; only
  reports and tooling do) — copying a record row into the profile table by hand is how
  you opt into a cheaper configuration.
* First call without a matching row: by default the engine runs a full `--tune` for
  that key and then executes; `PYQCD_EINSUM_AUTOTUNE=0` turns that off.
* Performance column: `wall_ms`, the wall clock of the **whole** `contract()` call
  — there is no separate per-step timer.
* `--tune` takes a value: empty / `true` = full sweep, `pass` = the fast pass that
  skips compare and equivalence rewriting.

See :doc:`performance` for the command, what a sweep prints, and how to read it.

### Native backends (optional)

| Backend | Usable for | Built during `pip install` when |
|---|---|---|
| KUPL | CPU | a C++17 compiler is available |
| CUDA | GPU | `nvcc` and Cython are available, and an NVIDIA toolchain is present |

Native acceleration is optional: when a library is missing the engine falls back
to the generic route, and only the speed changes.

### Fallback invariants

- A failed or unavailable native execution at run time switches to the generic
  path: same results, slower.
- `_native.py` only does `ctypes`; importing it never loads a library.
- Any native failure emits a single warning and falls back to Python instead of
  raising.
- A topology-specialised plan must pass the arena, single-GEMM and pack-heavy
  resource gates before native allocation.

### Introspection helpers

```python
import pyqcd.einsum as pe

pe.build_info()        # backend / native_available / array_backend / native_library
pe.specializations()   # equation-shape specialisations currently registered

info = pe.build_info()
assert isinstance(info, dict) and "backend" in info
assert len(pe.specializations()) >= 1
```

## Speed and self-checks

This page quotes no performance numbers on purpose: a ratio only means something on
the shape, dtype and device it was measured on. Calibrate on yours with
`python -m pyqcd.einsum.test --tune` (:doc:`performance`), and check whether an
equation has a specialised route at all with
`python -m pyqcd.einsum.verify_specialized "<equation>"`
({doc}`appendix_specialized`).
