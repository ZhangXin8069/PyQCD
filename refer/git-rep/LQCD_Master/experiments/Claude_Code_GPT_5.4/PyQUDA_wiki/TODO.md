# TODO & Roadmap

This page tracks missing features, planned improvements, and known limitations in PyQUDA.

---

## Table of Contents

- [Fermion Actions & Dirac Operators](#fermion-actions--dirac-operators)
- [Unwrapped QUDA Functions](#unwrapped-quda-functions)
- [I/O Formats](#io-formats)
- [Documentation](#documentation)
- [Testing & CI/CD](#testing--cicd)
- [Code Quality & Compatibility](#code-quality--compatibility)

---

## Fermion Actions & Dirac Operators

### Missing Dirac Operators

QUDA supports these Dslash types but PyQUDA has no wrapper:

| Dirac Operator | QUDA Dslash Type | Priority | Difficulty |
|---|---|---|---|
| Domain-Wall Fermion (DWF) | `QUDA_DOMAIN_WALL_DSLASH`, `QUDA_DOMAIN_WALL_4D_DSLASH` | High | Medium |
| Möbius DWF | `QUDA_MOBIUS_DWF_DSLASH`, `QUDA_MOBIUS_DWF_EOFA_DSLASH` | High | Medium |
| Twisted Mass | `QUDA_TWISTED_MASS_DSLASH` | Medium | Medium |
| Twisted Clover | `QUDA_TWISTED_CLOVER_DSLASH` | Medium | Medium |

Reference: QUDA enum definitions in `pyquda_core/quda/include/enum_quda.h`.

### Missing HMC Actions

| Action | Status | Notes |
|---|---|---|
| Pure Wilson (no Clover) Action | Missing | `WilsonDirac` exists but no `WilsonAction` for HMC |
| Naive Staggered Action | Missing | `StaggeredDirac` exists but no `StaggeredAction` for HMC |
| DWF / Möbius Action | Missing | Requires DWF Dirac operators first |
| Twisted Mass Action | Missing | Requires Twisted Mass Dirac first |

### Known Limitations

- Clover Wilson action does not support anisotropic lattices (`pyquda_core/pyquda/action/clover_wilson.py`)
- HISQ action does not support anisotropic lattices (`pyquda_core/pyquda/action/hisq.py`)
- Clover action multishift precision setting: TODO in code — "Check if it's better or worse" (`pyquda_core/pyquda/action/clover_wilson.py:46`)

---

## Unwrapped QUDA Functions

These functions have Python bindings in `pyquda_core/pyquda/quda.pyi` but lack high-level Python API wrappers:

### High Priority

| Function | Description | Potential Location |
|---|---|---|
| `contractQuda` | Fermion contraction (color/spin trace) | `LatticeGauge` or standalone |
| `eigensolveQuda` | Eigensolver (only in `pyquda_utils/eigensolve.py`, not in core Dirac API) | `FermionDirac` method |
| `newDeflationQuda` / `destroyDeflationQuda` | Deflated solver | `FermionDirac` method |
| `performTwoLinkGaussianSmearNStep` | Two-link Gaussian smearing | `LatticeGauge` method |

### Medium Priority

| Function | Description | Notes |
|---|---|---|
| `blasGEMMQuda` | GPU BLAS GEMM | Useful for custom contractions |
| `blasLUInvQuda` | GPU BLAS LU inverse | Useful for matrix inversion on lattice |
| `flushChronoQuda` | Flush chronological forecasting | For advanced solver tuning |
| `computeTwoLinkQuda` | Two-link computation | HISQ internal, but could be useful |
| `dumpMultigridQuda` | Dump multigrid state | For debugging |

### Low Priority

| Function | Description | Notes |
|---|---|---|
| `freeGaugeQuda` / `freeGaugeSmearedQuda` / `freeCloverQuda` | Explicit resource cleanup | Currently managed implicitly |
| `initQudaDevice` / `initQudaMemory` | Separate device/memory init | `initQuda` already handles both |
| `computeKSLinkQuda` | KS link computation | Used internally by HISQ at Cython level |

---

## I/O Formats

### Missing Write Support

| Format | Read | Write | Gap |
|---|---|---|---|
| **Chroma QIO** | Gauge, Propagator, Staggered Propagator | — | No write functions at all |
| **ILDG** | Gauge, Gauge (binary) | — | No write functions |
| **MILC** | Gauge, QIO Propagator, QIO Staggered Propagator | Gauge only | No propagator write |
| **NERSC** | Gauge | Gauge | No propagator support |
| **OpenQCD** | Gauge | Gauge | No propagator support |
| **XQCD** | Propagator, Staggered Propagator | Propagator, Staggered Propagator | No gauge support |

### Complete Formats (No Action Needed)

- **KYU**: Read/write for both gauge and propagator
- **NPY**: Read/write for both gauge and propagator
- **io_general**: Generic read/write

### Desired Features

- [ ] LIME record writing (prerequisite for Chroma/ILDG write support)
- [ ] ILDG gauge write with metadata
- [ ] Chroma QIO propagator write (for interoperability with Chroma)

---

## Documentation

| Item | Priority | Notes |
|---|---|---|
| Sphinx / ReadTheDocs integration | Medium | Auto-generate API docs from docstrings |
| Tutorial notebooks | Medium | Expand `examples/` with more annotated notebooks |
| HMC parameter tuning guide | Low | Document how to choose `multigrid`, `tol`, `maxiter` |
| FAQ / Troubleshooting page | Low | Common errors (CUDA OOM, MPI hang, tuning cache) |
| Architecture overview diagram | Low | Package dependency graph, data flow |

---

## Testing & CI/CD

### Missing Test Coverage

| Module / Feature | File | Notes |
|---|---|---|
| Naive Staggered Dirac | `pyquda_core/pyquda/dirac/staggered.py` | Only HISQ tested |
| Eigensolver | `pyquda_utils/eigensolve.py` | No test |
| FFT utilities | `pyquda_utils/fft.py` | No test |
| Source construction | `pyquda_utils/source.py` | No test |
| Phase factors | `pyquda_utils/phase.py`, `phase_v2.py` | No test |
| Remez algorithm | `pyquda_utils/alg_remez.py` | Has a TODO: "what if equal?" |
| MILC RHMC parameters | `pyquda_utils/milc_rhmc_param.py` | No test |
| HMC parameter utilities | `pyquda_utils/hmc_param.py` | No test |
| GPT interface | `pyquda_utils/gpt.py` | No test |
| Quasi-axial gauge fixing | `pyquda_utils/quasi_axial_gauge_fixing.py` | No test |
| Geo block size | `pyquda_utils/geo_block_size.py` | No test |
| SU(N>3) gauge utilities | `pyquda_utils/gauge_nd_sun.py` | No test |
| Wuppertal smearing | `pyquda/field.py` | API exists, no dedicated test |
| I/O write functions | Various | `test_io.py` exists but coverage unclear |

### CI/CD Improvements

- [ ] GitHub Actions workflow for automated testing (requires GPU runner or CPU-only test mode)
- [ ] Migrate test scripts to `pytest` framework for better reporting
- [ ] Add test configuration for multi-GPU (MPI) tests
- [ ] Code coverage reporting

---

## Code Quality & Compatibility

### Deprecations to Remove

These deprecated functions in `pyquda_utils/deprecated.py` should eventually be removed:

| Function | Replacement |
|---|---|
| `cb2()` | `evenodd()` |
| `smear()` | `GaugeField.stoutSmear()` |
| `smear4()` | `GaugeField.stoutSmear()` |
| `invert12()` | `core.invert()` |
| `getDslash()` | `core.getDirac()` |
| `getStaggeredDslash()` | `core.getStaggeredDirac()` |

Also in `pyquda_utils/source.py`: old signatures accepting `latt_size: List[int]` instead of `latt_info: LatticeInfo`.

### Backend Compatibility

- [ ] DPNP (Intel GPU) support is experimental — blocked by upstream issues:
  - `dpnp` array conversion workaround (`pyquda_core/pyquda_comm/array.py:137–148`)
  - Tracking: https://github.com/IntelPython/dpnp/issues/2568
- [ ] Verify all field operations work correctly with PyTorch backend
- [ ] `Pointer` class does not support `ndarray.ndim > 3` (`pyquda_core/pyquda_comm/src/pointer.pyx:137`)

### Code Modernization

- [ ] Consider dropping Python 3.8 support (use `typing` features from 3.9+: `list[int]`, `X | Y`)
- [ ] Add comprehensive type annotations to `pyquda_utils` functions
- [ ] Consider using `dataclasses` for parameter containers (e.g., `HMCParam`)
