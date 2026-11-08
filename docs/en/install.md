# Install

The package is distributed on PyPI as `pyqcd` (source distribution). Pip builds
it locally, which also lets it compile the optional native kernels on your
machine.

```bash
pip install pyqcd
```

## Requirements

| Item | Needed for |
|---|---|
| Python ≥ 3.10 | always |
| `numpy` | always (installed automatically) |
| C++17 compiler | native KUPL (CPU) kernels |
| `nvcc` + Cython | native CUDA (GPU) kernels |
| `mpicxx` | MPI-enabled KUPL build (source checkout only) |

## Optional extras

```bash
pip install "pyqcd[cuda]"     # adds cupy
pip install "pyqcd[mpi]"      # adds mpi4py
```

The extras only pull the Python dependencies; the native libraries are compiled
during the same `pip install` when a toolchain is available.

## What the installer does

1. Chooses a backend (override with `PYQCD_EINSUM_BACKEND=kupl|cuda`):
   * `cuda` when `nvcc` **and** Cython are available and the CUDA sources exist,
   * otherwise `kupl` when a C++ compiler is available,
   * otherwise nothing — the engine falls back to the pure-Python planner.
2. Builds the native library and stages it into the package.
3. Installs the package; a missing or failing toolchain only emits a warning.

Set `PYQCD_EINSUM_REQUIRE_NATIVE=1` to turn that warning into a hard error, and
`PYQCD_SKIP_NATIVE_BUILD=1` to skip native compilation entirely.

## Verify the installation

```python
import pyqcd
import pyqcd.einsum as pe

print(pyqcd.__version__)
print(pe.build_info())
# {'backend': 'kupl', 'mpi': False, 'kupl_mpi_build': False,
#  'mpi_label': 'non-MPI build', 'native_library': '.../libpyqcdeinsum_kupl.so'}
```

`backend` is `cuda`, `kupl` or `python` (the last one means the pure-Python
planner is in use). `mpi_enabled()` is `True` only when the KUPL library was
built with real MPI support; the CUDA backend never has it.

Quick functional check:

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)
```

## Runtime environment variables

| Variable | Effect |
|---|---|
| `PYQCD_EINSUM_BACKEND` | `kupl` / `cuda` / `auto` (also honoured at build time) |
| `PYQCD_EINSUM_SPECIALIZE=0` | disable equation-shape specialisations |
| `PYQCD_EINSUM_DISABLE_FUSED=1` | disable the native fused kernels |
| `PYQCD_EINSUM_KERNEL_KUPL=0` / `PYQCD_EINSUM_KERNEL_CUDA=0` | disable the per-backend specialised kernels |
| `PYQCD_EINSUM_ORACLE=installed` | use the installed `opt_einsum` as the differential oracle |

## Building from a source checkout (development)

```bash
bash install.sh kupl     # CPU (KUPL) backend
bash install.sh cuda     # CUDA backend + Cython extension
bash install.sh          # auto
source ./env.sh          # PYTHONPATH / LD_LIBRARY_PATH / backend selection
```

CUDA architecture is resolved from `PYQCD_CUDA_ARCHITECTURES`, then
`CMAKE_CUDA_ARCHITECTURES`, then `PYQCD_SM_ARCH`; always set it for a new GPU
generation:

```bash
PYQCD_CUDA_ARCHITECTURES=8.9 bash install.sh cuda
```

MPI sharding exists only in the KUPL backend and only shards a single output
axis. `PYQCD_EINSUM_ENABLE_MPI` accepts `auto` (default: enabled only when
`mpicxx` exists and the effective `CXX` can preprocess `#include <mpi.h>`), `1`
and `0`.

## Tests (source checkout)

```bash
python -m pyqcd.testing                                  # integration suite

python -m pyqcd.testing.einsum.verify_native --backend kupl
python -m pyqcd.testing.einsum.verify_native --backend cuda
python -m pyqcd.testing.einsum.verify_compat             # differential vs opt_einsum
mpirun -np 2 python -m pyqcd.testing.einsum.verify_mpi

bash pyqcd/testing/einsum/simple/run-all.sh              # einsum sub-system suite
bash pyqcd/testing/simple/run-all.sh                     # per-function API parity suite
```
