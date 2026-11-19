# Install

```bash
pip install -U pyqcd
```

**Strongly recommended: always install with `pip install -U pyqcd` so you get the
latest release, and read the output and any notices printed during the install.**
The package is distributed on PyPI as `pyqcd`; a single install gives you the whole
toolkit and everything this site documents.

Current release: `0.5.9`. To pin it: `pip install "pyqcd==0.5.9"`.

## Requirements

| Item | Needed for |
|---|---|
| Python ≥ 3.9 | always |
| `numpy`, `h5py`, `scipy` | always (installed automatically) |
| `torch` | optional: the torch backend (`pip install "pyqcd[torch]"`) |
| `cupy-cuda12x` | optional: GPU arrays (`pip install "pyqcd[cuda]"`) |
| `matplotlib` | optional: plotting helpers (`pip install "pyqcd[plot]"`) |

`requires-python` is `>=3.9`. Python 3.8 is no longer supported.

## Optional extras

```bash
pip install "pyqcd[cuda]"     # adds cupy-cuda12x (CUDA 12.x)
pip install "pyqcd[torch]"    # adds torch (optional backend)
pip install "pyqcd[plot]"     # adds matplotlib (Wick/analysis plots)
```

## Verify the installation

The fastest check is the module entry point — it needs no imports and reports what
is actually installed:

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
# {'backend': 'kupl', 'native_available': True,
#  'array_backend': 'numpy', 'native_library': '.../libpyqcdeinsum_kupl.so'}

info = pe.build_info()
assert pyqcd.__version__.count(".") == 2
assert info["backend"] in {"cuda", "kupl", "python"}
assert isinstance(info["native_library"], str)
```

`backend` is `cuda`, `kupl` or `python` (the last one means the pure-Python planner
is in use); `array_backend` is what your arrays are — the two are different
questions, and a `backend='cuda'` install still runs numpy input on the CPU library.

Quick functional check:

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)
```

## What the installed package contains

`pip install pyqcd` installs the whole tool package — `pyqcd` itself plus
`lattice`, `gauge`, `smear`, `operator`, `vertex`, `contraction`, `renorm`,
`tools`, `analysis`, `pipeline` and `parallel` — together with `pyqcd.einsum`.

`pyqcd.testing` is **not** in the package: the test suites, benchmark drivers and
regression baselines stay in the source repository, so `python -m pyqcd.testing`
raises `ModuleNotFoundError` in an installed environment. Everything this site
shows is reachable from the installed package alone.

## Runtime environment variables

| Variable | Effect |
|---|---|
| `PYQCD_EINSUM_NATIVE` | `kupl` / `cuda` / `auto` (default); `auto` prefers CUDA, falls back to KUPL, then to the pure-Python planner |
| `PYQCD_EINSUM_SPECIALIZE=0` | disable equation-shape specialisations |
| `PYQCD_EINSUM_DISABLE_FUSED=1` | disable the native fused kernels |
| `PYQCD_EINSUM_KERNEL_KUPL=0` / `PYQCD_EINSUM_KERNEL_CUDA=0` | disable the per-backend specialised kernels |
| `PYQCD_EINSUM_KERNEL_FORCE` | force one route: `generic` / `dedicated` / `topology` / `backend` (single value, default `generic`) |
| `PYQCD_EINSUM_NOTICE=0` | silence the first-hit notices |
| `PYQCD_EINSUM_LANG=zh\|en` | notice language |

## Backends and array types

| Input arrays | What runs |
|---|---|
| `numpy` | the native CPU library when it is installed, otherwise the pure-Python planner |
| `cupy` | the CUDA kernels for the specialised shapes |
| `torch` | the torch backend (`set_backend("torch")`), i.e. `torch.einsum` under PyQCD's dispatch |

## Threads

The engine configures OpenMP and BLAS once at import, because leaving both at
"all logical cores" is the single most common cause of bad performance here (both
runtimes oversubscribe the same cores).

| Variable | Effect | Default |
|---|---|---|
| `OMP_NUM_THREADS` | OpenMP worker threads — highest priority | auto |
| `OPENBLAS_NUM_THREADS` | BLAS threads (matmul) — highest priority | auto |
| `PYQCD_EINSUM_OMP_THREADS` | same, second priority | auto |
| `PYQCD_EINSUM_BLAS_THREADS` | same, second priority | auto |
| `PYQCD_EINSUM_THREADS=off` | disable the auto configuration entirely | on |

Inspect or change it at runtime:

```python
import pyqcd.einsum as pe

pe.thread_info()                 # {'omp': 4, 'blas': 2, 'source': 'auto (...)', ...}
pe.configure_threads(4, 2)       # pin OpenMP=4, BLAS=2 explicitly
pe.configure_threads(blas=4)     # only change BLAS

info = pe.thread_info()
assert info["omp"] >= 1 and info["blas"] >= 1
assert "profile" in info
```

`python -m pyqcd` prints the effective values too.

## Introspection

```python
import pyqcd.einsum as pe

pe.build_info()        # backend / native_available / array_backend / native_library
pe.thread_info()       # effective omp / blas threads and where they came from
pe.specializations()   # registered equation-shape specialisations

info = pe.build_info()
assert {"backend", "native_available", "array_backend"} <= set(info)
assert len(pe.specializations()) >= 1
```

## Source repositories

The package is developed on GitLab and mirrored to Gitee (国内发布) and GitHub
(国外发布); all three carry the same history:

* 国内开发 — https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD
* 国内发布 — https://gitee.com/zhangxin8069/PyQCD
* 国外发布 — https://github.com/zhangxin8069/PyQCD
