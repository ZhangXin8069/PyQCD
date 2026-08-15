# PyQUDA

Python wrapper for [QUDA](https://github.com/lattice/quda) written in Cython. **Current version: 0.10.52**

This project aims to benefit from the optimized linear algebra library [CuPy](https://cupy.dev/) in Python based on CUDA. CuPy and QUDA will allow us to perform most lattice QCD research operations with high performance. [PyTorch](https://pytorch.org/) and [DPNP](https://intelpython.github.io/dpnp/) (for Intel GPU) are alternative options.

This project is based on the latest QUDA [`develop`](https://github.com/lattice/quda/tree/develop) branch. PyQUDA should be compatible with any commit of QUDA after [PR #1489](https://github.com/lattice/quda/pull/1489), but leave some features disabled.

## Package Structure

PyQUDA is organized into four packages:

| Package | PyPI Name | Description |
| --- | --- | --- |
| **pyquda_core** | `pyquda` | Core QUDA wrapper: Cython bindings, lattice field classes, Dirac operators, HMC framework, gauge operations |
| **pyquda_utils** | `pyquda-utils` | High-level utilities: fermion inversion helpers, source generation, phase, FFT, gamma algebra, HMC parameters, I/O wrappers |
| **pyquda_io** | (included in pyquda-utils) | Low-level file I/O: Chroma, MILC, KYU, XQCD, ILDG, NERSC, OpenQCD, NPY format readers/writers |
| **pyquda_plugins** | (included in pyquda-utils) | Plugin system: auto-generate Cython wrappers from C header files |

## Requirements

- Python >= 3.8
- GPU backend: [CuPy](https://cupy.dev/) >= 12, [PyTorch](https://pytorch.org/) >= 2, or [DPNP](https://intelpython.github.io/dpnp/) (experimental)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/) for multi-GPU support
- [QUDA](https://github.com/lattice/quda) compiled as a shared library

## Features

- Multi-GPU supported (MPI)
- Multiple GPU backends: CuPy, PyTorch, NumPy (CPU-only), DPNP
- Quark propagator: Wilson, Clover (with multigrid), HISQ, Staggered, Laplace
- HMC: pure gauge, 1/2-flavor Clover, N-flavor HISQ
- Gauge smearing: APE, stout, HYP (3D/4D)
- Fermion smearing: Gaussian (Wuppertal)
- Gradient flow: Wilson/Symanzik (with adjoint flow for fermions)
- Gauge fixing: over-relaxation, FFT
- Rich file I/O: Chroma QIO, MILC, KYU, XQCD, ILDG, NERSC, OpenQCD, NPY, HDF5
- Plugin system for wrapping external C libraries
