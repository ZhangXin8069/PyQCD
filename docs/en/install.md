# Install

The package is distributed on PyPI as `pyqcd` (source distribution). Pip builds
it locally, which is also when the native kernels are compiled on your machine.

```bash
pip install pyqcd
```

Current release: `0.1.4`. To pin it: `pip install "pyqcd==0.1.4"`.

## Requirements

| Item | Needed for |
|---|---|
| Python ≥ 3.10 | always |
| `numpy`, `h5py`, `scipy` | always (installed automatically) |
| `bash` + C++17 compiler | native KUPL (CPU) kernels — the baseline build |
| `nvcc` | optional native CUDA (GPU) kernels |
| Cython | optional CUDA extension; supplied by the pip build environment, no manual install |
| `mpicxx` | MPI-enabled KUPL build (source checkout only) |
| `torch` | optional: the torch backend (`pip install "pyqcd[torch]"`) |
| `matplotlib` | optional: plotting helpers (`pip install "pyqcd[plot]"`) |

Cython is not something you install yourself: `pyproject.toml` already lists
`cython>=3.0` in `[build-system] requires`, so the isolated build environment
that pip creates for `pip install pyqcd` always has it. A manual
`pip install Cython` only matters when the build runs with
`--no-build-isolation`.

## Optional extras

```bash
pip install "pyqcd[cuda]"     # adds cupy
pip install "pyqcd[mpi]"      # adds mpi4py
pip install "pyqcd[torch]"    # adds torch (optional backend)
pip install "pyqcd[plot]"     # adds matplotlib (Wick/analysis plots)
```

The extras only pull the Python dependencies; the native libraries are compiled
during the same `pip install` when a toolchain is available.

## What the installer does

1. **KUPL is always built.** The CPU native library (`cpp/kupl`) is the
   baseline: it is compiled with the host C++17 compiler and staged into
   `pyqcd/einsum/kupl/`.
2. **CUDA is attempted on top of that, never instead of it.** In the default
   `auto` mode the CUDA library and its Cython extension are built only when
   `nvcc`, the CUDA sources under `cpp/cuda/einsum/src` and Cython are all
   present. Cython comes from the pip build environment, so in practice the
   CUDA decision hangs on `nvcc`. KUPL is compiled first, so a broken CUDA
   toolchain can never cost you the CPU backend.
3. **A failing native build is a warning, not a failed install.** A CUDA failure
   emits a `RuntimeWarning`, the installation continues, and `pyqcd.einsum`
   keeps using KUPL. A KUPL failure also only warns and degrades the engine to
   the pure-Python planner.
4. `pyqcd` is installed with whichever libraries were staged, and
   `build_info()['backend']` reports at run time what is actually usable.

The switches below are read while `pip install` builds the package:

| Variable | Effect |
|---|---|
| `PYQCD_EINSUM_BACKEND=kupl` | build KUPL only |
| `PYQCD_EINSUM_BACKEND=cuda` | build KUPL **and** CUDA — the value means "both", KUPL stays the mandatory baseline |
| `PYQCD_EINSUM_BACKEND=auto` (default) | KUPL always, plus CUDA when `nvcc`, the CUDA sources and Cython are present |
| `PYQCD_EINSUM_REQUIRE_NATIVE=1` | turn a skipped or failed **KUPL** build into a hard error; it never applies to CUDA |
| `PYQCD_EINSUM_SKIP_CYTHON=1` | pretend Cython is missing: `auto` drops to KUPL only, `cuda` still builds the CUDA library but skips its Cython extension |
| `PYQCD_SKIP_NATIVE_BUILD=1` | skip native compilation entirely and reuse an existing library if there is one; without one, only the pure-Python planner is available |

## What the installed package contains

`pip install pyqcd` installs the whole tool package — `pyqcd` itself plus
`lattice`, `gauge`, `smear`, `operator`, `vertex`, `contraction`, `renorm`,
`tools`, `analysis`, `pipeline` and `parallel` — together with `pyqcd.einsum`
and whichever native libraries were built (KUPL always attempted, CUDA only if
its build succeeded).

`pyqcd.testing` is **not** in the package. `setup.py` publishes with
`find_packages(exclude=["pyqcd.testing", "pyqcd.testing.*"])`, so the tests, the
benchmark drivers and the regression baselines stay in the source repository.
In an installed environment `python -m pyqcd.testing` raises
`ModuleNotFoundError: No module named 'pyqcd.testing'`; run those commands from
a source checkout instead (see the last two sections).

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
planner is in use). CUDA is preferred and falls back to KUPL whenever the CUDA
extension cannot be imported, so a machine whose CUDA build was skipped still
reports `kupl`. `kupl_mpi_build` is `True` only when the KUPL library was built
with real MPI support, and `mpi` is `True` only when that build is also the
active backend; the CUDA backend never has MPI. `pe.mpi_enabled()` exposes the
KUPL build flag alone, independent of which backend is currently active.

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
| `PYQCD_EINSUM_NATIVE` | `kupl` / `cuda` / `auto` (default); `auto` prefers CUDA, falls back to KUPL, then to the pure-Python planner |
| `PYQCD_EINSUM_SPECIALIZE=0` | disable equation-shape specialisations |
| `PYQCD_EINSUM_DISABLE_FUSED=1` | disable the native fused kernels |
| `PYQCD_EINSUM_KERNEL_KUPL=0` / `PYQCD_EINSUM_KERNEL_CUDA=0` | disable the per-backend specialised kernels |
| `PYQCD_EINSUM_ORACLE=installed` | use the installed `opt_einsum` as the differential oracle |

`PYQCD_EINSUM_BACKEND` (see above) selects the same backend at build time; it
is the build-time knob, while `PYQCD_EINSUM_NATIVE` is the one read at run time.

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

These commands import `pyqcd.testing`, which is not shipped on PyPI, so they
only work from the root of a source checkout:

```bash
python -m pyqcd.testing                                  # integration suite

python -m pyqcd.testing.einsum.verify_native --backend kupl
python -m pyqcd.testing.einsum.verify_native --backend cuda
python -m pyqcd.testing.einsum.verify_compat             # differential vs opt_einsum
mpirun -np 2 python -m pyqcd.testing.einsum.verify_mpi

bash pyqcd/testing/einsum/simple/run-all.sh              # einsum sub-system suite
bash pyqcd/testing/simple/run-all.sh                     # per-function API parity suite
```

## Source repositories

The package is developed on GitLab and mirrored to Gitee (国内发布) and GitHub
(国外发布); all three carry the same history:

* 国内开发 — https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD
* 国内发布 — https://gitee.com/zhangxin8069/PyQCD
* 国外发布 — https://github.com/zhangxin8069/PyQCD
