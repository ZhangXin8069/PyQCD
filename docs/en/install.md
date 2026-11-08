# Install

The package is distributed on PyPI as `pyqcd` (source distribution). Pip builds
it locally, which is also when the native kernels are compiled on your machine.

```bash
pip install pyqcd
```

Current release: `0.1.14`. To pin it: `pip install "pyqcd==0.1.13"`.

There is deliberately no platform wheel on PyPI: the native library links against
the C++ runtime of the build host, so a portable manylinux wheel would need
`auditwheel` repair. Expect the first install to compile for about a minute on a
laptop; pip caches the built wheel, so installing into another environment of the
same interpreter is fast.

## Requirements

| Item | Needed for |
|---|---|
| Python ≥ 3.10 | always |
| `numpy`, `h5py`, `scipy` | always (installed automatically) |
| `bash` + C++17 compiler | native KUPL (CPU) kernels — the baseline build |
| `nvcc` | optional native CUDA (GPU) kernels |
| Cython | optional CUDA extension; supplied by the pip build environment, no manual install |

`requires-python` is `>=3.10`. The suites behind this documentation were executed
on Python 3.12 (full test suite) and 3.14 (build + import + `python -m pyqcd`);
3.10, 3.11 and 3.13 are expected to work but are not covered by an automated run
on the release machine.
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
pip install "pyqcd[cuda]"     # adds cupy-cuda12x (CUDA 12.x)
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

The fastest check is the module entry point — it needs no imports and reports
what is actually installed:

```bash
python -m pyqcd            # version, native backend, optional deps, subpackages
python -m pyqcd --version  # version only
python -m pyqcd --json     # machine-readable
```

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

### CUDA architecture: auto-detected, and PTX is always embedded

The build resolves the architecture in this order:

| Priority | Source |
|---|---|
| 1 | `PYQCD_CUDA_ARCHITECTURES` (comma/semicolon/space separated list) |
| 2 | `CMAKE_CUDA_ARCHITECTURES` |
| 3 | `PYQCD_SM_ARCH` (legacy name) |
| 4 | auto-detect: `nvidia-smi --query-gpu=compute_cap`, then a GPU-model fallback table |
| 5 | default `6.0 7.0` (build machines without a GPU, e.g. CI or a cluster login node) |

Every architecture is compiled as `code=[sm_XX,compute_XX]` — **cubin plus PTX**. The
PTX means a newer GPU that has no matching SASS still runs, because the driver
JIT-compiles the PTX; without it the kernel entry returns `QCD_STATUS_INTERNAL`
and the engine silently falls back to the CPU path.

Measured on an RTX 4060 Laptop (compute capability 8.9):

| build | kernel status | result |
|---|---|---|
| old default, SASS only (`sm_70`) | `3` (`QCD_STATUS_INTERNAL`) | falls back to CPU |
| auto-detected `8.9`, cubin+PTX | `0` (`QCD_STATUS_OK`) | GPU, rel 2.8e-14 |
| explicit `7.0`, cubin+PTX | `0` (`QCD_STATUS_OK`) | GPU via PTX JIT, rel 2.8e-14 |

So the out-of-the-box install works on current GPUs. You only need to set the
variable when you want the fastest path (native SASS for your card) or when the
build machine has no GPU and you know the target:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader      # e.g. 8.9
PYQCD_CUDA_ARCHITECTURES=8.9 pip install "pyqcd[cuda]"
```

Diagnose an existing install with:

```bash
# 1) which architectures were compiled (visible during the build)
pip install -v "pyqcd[cuda]" 2>&1 | grep -E "^  arch:"

# 2) does the installed archive carry a CUDA fatbin at all (cubin + PTX)?
readelf -S <site-packages>/pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so | grep nv_fatbin

# 3) the decisive check — does the GPU kernel really run?
PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 python -c "
import numpy as np, cupy as cp, pyqcd.einsum as pe
e = 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'
r = np.random.default_rng(0)
o = [cp.asarray((r.standard_normal(s)+1j*r.standard_normal(s)).astype(np.complex128))
     for s in [(4,4,16,16)]*3 + [(4,4)]*2 + [(4,16,16,16)]*2 + [(4,4)]]
pe.contract(e, *o)
"
```

The hint says `[kernel: cuda_nucleon_2pt]` when the GPU kernel ran and
`[plan-cache]` when it did not. A `strings | grep compute_` check only works for
SASS-only archives — with the fatbin layout the architecture names are not plain
ASCII, so use the runtime hint instead.

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

## Backends and array types

| Input arrays | What runs |
|---|---|
| `numpy` | the KUPL native library (CPU) when it is installed, otherwise the pure-Python planner |
| `cupy` | the CUDA kernels for the specialised shapes; the generic executor currently may report `QCD_STATUS_INTERNAL` and fall back to Python — a single `RuntimeWarning` names the reason, and results stay correct |
| `torch` | the torch backend (`set_backend("torch")`), i.e. `torch.einsum` under PyQCD's dispatch |

`pyqcd.einsum.build_info()` reports `backend` (which native library is usable) and
`array_backend` (what your arrays are); the two are not the same question. A
`backend='cuda'` build still runs numpy input on the CPU library.
