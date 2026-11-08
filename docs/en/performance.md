# Performance

`pyqcd.einsum` is benchmarked against `opt_einsum` on the same machine, in the
same dtype, with the same thread count and the same planner. Numbers below are
measured, not modelled, and every one of them can be reproduced with the
commands at the end of this page.

## Measurement protocol

| Item | Setting |
|---|---|
| Hardware | 1× RTX 4060 Laptop (GPU runs), 4 pinned threads (CPU runs) |
| Oracle | `opt_einsum` 3.4.0, `dp` planner |
| dtype | `complex128` (both sides) |
| Isolation | one fresh interpreter per side (`--isolation process`) |
| Timing | 4 warm-up rounds, median of the measured rounds, synchronised before and after |
| Operands | fixed-seed arrays with the shapes of the real distillation contractions |
| Correctness | `max|Δ| / max|ref| ≤ 1e-12` on the same run |

Timing only covers steady-state execution: both sides plan once, then the plan
is reused.

## Reproducible on the published package

`pyqcd==0.1.10` (`build_info(): backend=kupl, array_backend=numpy`) versus
`opt_einsum==3.4.0` (`dp`), `complex128`, median of 15 calls after 3 warm-ups:

| Workload | `opt_einsum` | `pyqcd.einsum` | Ratio |
|---|---|---|---|
| Nucleon 2pt, nvec=16 (production contract) | 12.95 ms | 2.15 ms | **6.02×** |
| Nucleon 2pt, nvec=24 | 41.45 ms | 7.11 ms | **5.83×** |
| γ sandwich, T=8, nvec=64 | 0.169 ms | 0.058 ms | **2.91×** |

What the shipped build actually accelerates:

| Shape | Route in 0.1.x |
|---|---|
| Nucleon 2pt (`gahb,mcnd,…->M`) | dedicated KUPL/CUDA kernel + plan cache |
| Nucleon 3pt main (`…->MNGLxy`) | dedicated KUPL/CUDA kernel + plan cache |
| Nucleon 3pt (`…->GM`), sush 2pt/3pt-mu | plan cache only |
| γ sandwich (`ab,...bce,cd->...dae`), perambulator × gauge | plan cache + general execution |

The hint printed on the first hit says which of the two happened:
`[kernel: kupl_nucleon_2pt]` means a kernel ran, `[plan-cache]` means only the
plan was reused.

## Historical development-build measurements

The figures below were taken on development builds that carried extra fused
kernels (γ sandwich, perambulator × gauge). **Those kernels are not in the 0.1.x
release**, so the numbers are *not* reproducible from PyPI — they are kept only as
a record of what the shapes can cost:

| Workload | opt_einsum | development build | Ratio |
|---|---|---|---|
| Nucleon 2pt (CPU, N=100) | 120.43 ms | 22.65 ms | 5.32× |
| Nucleon 3pt (GPU) | 285.40 ms | 30.74 ms | 9.29× |
| γ sandwich, L24 (GPU) | 0.373987 s | 0.110959 s | 3.371× |
| γ sandwich, L24 (CPU) | 1188.9 ms | 142.7 ms | 8.33× |
| perambulator × gauge, L24 (GPU) | 1.112793 s | 1.095407 s | 1.016× (parity) |

A weighted production-mix model built on those numbers gave ≈2.82×
(4757 s → 1688 s). It inherits the same caveat: it assumes the extra kernels.

## Code

The specialised path is the default: no import, no flag, no plan restructuring.
The only thing to know is that it prints a hint the first time each shape is hit,
so an A/B run is one keyword argument apart.

```python
import numpy as np
import pyqcd.einsum as pe

rng = np.random.default_rng(0)


def c(*shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# Nucleon 2pt (E1): perambulator x gamma x VVV x parity projection
nucleon_2pt = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
ops_2pt = [c(2, 2, 2, 2), c(2, 2, 2, 2), c(2, 2, 2, 2), c(2, 2),
           c(2, 2), c(4, 2, 2, 2), c(4, 2, 2, 2), c(2, 2)]

fast = pe.contract(nucleon_2pt, *ops_2pt)                    # specialised path
base = pe.contract(nucleon_2pt, *ops_2pt, specialize=False)  # general path
assert np.allclose(fast, base)

# gamma sandwich (E5): perambulator^dagger x gamma sandwich x perambulator
sandwich = "ab,...bce,cd->...dae"
ops_s = [c(2, 2), c(3, 2, 2, 2), c(2, 2)]
assert np.allclose(pe.contract(sandwich, *ops_s),
                   pe.contract(sandwich, *ops_s, specialize=False))

# which shapes get a specialised path
for name, kind, subscripts in pe.specializations():
    print(f"{name:20s} {kind:28s} {subscripts}")
```

The same switches are available from the environment when you do not want to
touch application code:

```bash
PYQCD_EINSUM_SPECIALIZE=0 python my_run.py            # process-wide off
PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 python my_run.py    # print for every call
PYQCD_EINSUM_DISABLE_FUSED=1 python my_run.py         # keep planning, drop fused kernels
PYQCD_EINSUM_KERNEL_KUPL=0 python my_run.py           # drop KUPL specialised kernels
PYQCD_EINSUM_KERNEL_CUDA=0 python my_run.py           # drop CUDA specialised kernels
```

Every registered shape is discoverable at runtime, and `specialize=False`
always reproduces the general path, so a regression is a plain `assert`:

| Group | Kind | What it covers |
|---|---|---|
| `nucleon-2pt(E1)` | fused + KUPL/CUDA kernel | 8-operand nucleon 2pt |
| `nucleon-3pt(E2)` | plan cache | 8-operand nucleon 3pt |
| `gauge-propagator(E4)` | fused (CUDA) | perambulator × gauge |
| `gamma-sandwich(E5)` | fused (CPU + GPU) | `γ₅` sandwich |
| `sush-2pt(S1)`, `sush-3pt-munu(S2)`, `sush-2pt-mu(S3)` | plan cache | sush production shapes |
| `sush-3pt-main(P1/P2)` | fused + CUDA kernel | 14-operand 3pt main term |


## Reproduce

The benchmark driver `pyqcd.testing.einsum.bench_native` is **not part of the
PyPI package**: `pyqcd.testing` is excluded from the published distribution
(`setup.py` uses `find_packages(exclude=["pyqcd.testing", "pyqcd.testing.*"])`),
so the benchmarks, tests and regression baselines exist only in a source
checkout. Both commands below therefore have to be run from the root of that
checkout; in an environment that only has `pip install pyqcd`, the driver
exits with `ModuleNotFoundError: No module named 'pyqcd.testing'`.

```bash
cd /path/to/PyQCD     # source checkout root; pyqcd.testing is not on PyPI

# CPU: KUPL backend, 4 pinned threads, process isolation
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=kupl PYQCD_BENCH_THREADS=4 \
  python -m pyqcd.testing.einsum.bench_native \
  --data synthetic --profiles T1 --backends cpu --iters 10 --warmup 4 --planner shared

# GPU: CUDA backend, same protocol
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=cuda \
  python -m pyqcd.testing.einsum.bench_native \
  --data synthetic --profiles L24 --backends gpu --iters 10 --warmup 4 --planner shared
```

Correctness of the specialised paths is checked by the same suite that produces
the timings, and for the same reason that suite is source-checkout only:
`python -m pyqcd.testing.einsum.verify_compat` (980 cases) and
`python -m pyqcd.testing.einsum.verify_native --backend kupl|cuda`.
