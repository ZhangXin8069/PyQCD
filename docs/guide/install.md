# Install

PyQCD runs from a source checkout. The native contraction backends are optional:
without them everything still works through the pure-Python path.

## Requirements

| Item | Needed for |
|---|---|
| Python ≥ 3.10 | always |
| C++17 compiler | native KUPL backend |
| `nvcc` + Cython | native CUDA backend |
| `mpicxx` | MPI-enabled KUPL build |
| OpenBLAS | numerical kernels (loaded at runtime) |

## Install

```bash
bash install.sh kupl     # CPU (KUPL) backend
bash install.sh cuda     # CUDA backend + Cython extension
bash install.sh          # auto: cuda when nvcc and CUDA sources are present
bash install.sh --help
```

`install.sh` builds the selected native library, then runs
`python setup.py build_ext --inplace` and finally checks that the library
exists. `PYQCD_SKIP_NATIVE_BUILD=1` reuses an existing library;
`PYQCD_EINSUM_SKIP_CYTHON=1` skips the Cython extension.

Load the environment in every shell:

```bash
source ./env.sh
```

`env.sh` exports the `PYQCD_EINSUM_*` paths, adds the repository root to
`PYTHONPATH`, adds both native library directories to `LD_LIBRARY_PATH`, and
sets `OMP_DYNAMIC=FALSE`.

## CUDA architecture

`make-nv.sh` resolves the architecture from `PYQCD_CUDA_ARCHITECTURES`, then
`CMAKE_CUDA_ARCHITECTURES`, then `PYQCD_SM_ARCH`, and finally falls back to its
built-in default. Always set it explicitly for a new GPU generation:

```bash
PYQCD_CUDA_ARCHITECTURES=8.9 bash install.sh cuda
```

## MPI

MPI sharding exists only in the KUPL (CPU) backend and only shards a single
output axis; the CUDA backend has no MPI implementation.

`PYQCD_EINSUM_ENABLE_MPI` selects the mode:

| Value | Behaviour |
|---|---|
| `auto` (default) | MPI is enabled only if `mpicxx` exists **and** the effective `CXX` can preprocess `#include <mpi.h>` |
| `1` | force MPI (`CXX` defaults to `mpicxx`) |
| `0` | force non-MPI |

```bash
PYQCD_EINSUM_ENABLE_MPI=1 bash install.sh kupl
```

Check what you actually built:

```python
import pyqcd.einsum as pe

print(pe.build_info())
# {'backend': 'kupl', 'mpi': False, 'kupl_mpi_build': False,
#  'mpi_label': 'non-MPI build', 'native_library': '.../libpyqcdeinsum_kupl.so'}
```

`mpi_enabled()` returns `True` only when the KUPL binary contains the real MPI
implementation. The CUDA backend is always reported as non-MPI.

## Environment variables

| Variable | Effect |
|---|---|
| `PYQCD_EINSUM_BACKEND` | `kupl` / `cuda` / `auto` |
| `PYQCD_EINSUM_BUILD_DIR` | override the native build directory |
| `PYQCD_SKIP_NATIVE_BUILD=1` | reuse an existing native library |
| `PYQCD_EINSUM_SKIP_CYTHON=1` | skip the Cython extension |
| `PYQCD_EINSUM_ENABLE_MPI` | `auto` / `1` / `0` |
| `PYQCD_CUDA_ARCHITECTURES` | CUDA architecture list (highest priority) |
| `PYQCD_OPENBLAS_LIBRARY`, `PYQCD_OPENBLAS_INCLUDE_DIR`, `PYQCD_OPENBLAS_FLAGS` | explicit OpenBLAS selection |
| `NVCC`, `CXX` | override the compilers |
| `PYQCD_EINSUM_ORACLE=installed` | use the installed `opt_einsum` as the differential oracle |

## Tests

```bash
source ./env.sh
python -m pyqcd.testing                                  # integration suite

python -m pyqcd.testing.einsum.verify_native --backend kupl
python -m pyqcd.testing.einsum.verify_native --backend cuda
python -m pyqcd.testing.einsum.verify_compat             # differential vs opt_einsum
mpirun -np 2 python -m pyqcd.testing.einsum.verify_mpi

bash pyqcd/testing/einsum/simple/run-all.sh              # one-shot sub-system suite
bash pyqcd/testing/simple/run-all.sh                     # per-function API parity suite
```

`verify_compat` needs an `opt_einsum` oracle: by default it uses the
`refer/git-rep/opt_einsum` submodule; set `PYQCD_EINSUM_ORACLE=installed` to use
the installed copy (it also needs `pytest`, because the oracle imports
`opt_einsum.testing`).
