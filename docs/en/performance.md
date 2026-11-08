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

## What is actually faster

| Workload | Equation shape | opt_einsum | pyqcd.einsum | Ratio |
|---|---|---|---|---|
| Nucleon 2pt (CPU, N=100) | 8 operands, perambulator × γ × VVV × parity projection | 120.43 ms | 22.65 ms (specialised) / 25.26 ms (generic) | **5.32×** vs opt_einsum, 1.12× vs our generic path |
| Nucleon 3pt (GPU) | 8 operands, source perambulator × current × γ × VdV × projection | 285.40 ms | 30.74 ms (specialised) / 74.13 ms (generic) | **9.29×** vs opt_einsum, 2.41× vs our generic path |
| γ sandwich, L24 (GPU) | `γ₅ · peramb† · γ₅` over the full spatial volume | 0.373987 s | 0.110959 s | **3.371×** |
| γ sandwich, L24 (CPU) | same shape, 4 threads | 1188.9 ms | 142.7 ms | **8.33×** |
| perambulator × gauge, L24 (GPU) | 4 operands including the lattice `z,y,x` volume | 1.112793 s | 1.095407 s | 1.016× (parity) |
| perambulator × gauge, `z·y·x = 1` (GPU) | same shape without the volume | — | — | **3.6×** |

Two things are worth separating:

* the ratio **against our own generic path** shows what the lattice-QCD
  specialised kernels buy (1.12× for 2pt, 2.41× for 3pt, 8.33× for the CPU
  sandwich);
* the ratio **against `opt_einsum`** also includes everything the general
  execution layer already wins on these shapes.

## Aggregated over a production mix

Weighting the five contraction shapes by their share of a real production run
(2pt 51 %, 3pt 15.5 % + 3.1 %, perambulator × gauge 26.4 %, sandwich 4 %):

| Implementation | Total | Relative |
|---|---|---|
| `opt_einsum` | 4757 s | 1.00× |
| `pyqcd.einsum` | 1688 s | **2.82×** |

This is a model built from the per-shape measurements above, not a single
end-to-end run; treat it as an estimate of the aggregate effect.

## Where the gain comes from

1. **Equation-shape specialisation.** A small set of lattice-QCD shapes
   (nucleon 2pt/3pt, γ sandwich, perambulator × gauge) is recognised and routed
   to purpose-built kernels instead of a generic BLAS chain.
2. **Fusion.** Intermediate contraction results stay in registers/shared memory
   instead of being materialised as temporary tensors.
3. **Buffer reuse.** Operand packing and output buffers are pooled across calls;
   the fixed per-call overhead is not paid again for every intermediate.
4. **Native planning.** Planning is compiled into the native library, so paths
   and costs are identical to the Python planner (verified by the 980-case
   differential suite) while dispatch is cheaper.

### Reading the A/B switch correctly

`specialize=False` (and `PYQCD_EINSUM_SPECIALIZE=0`) switches off the
**whole** specialised branch, including its plan cache — it is not a pure
kernel-vs-no-kernel comparison. Measured on the production nucleon-2pt contract
(median of 30 calls, same machine):

| nvec | default | `PYQCD_EINSUM_SPECIALIZE=0` | default vs off |
|---|---|---|---|
| 8 | 0.85–0.97 ms | 1.21 ms | 1.3× faster |
| 16 | 2.11–2.22 ms | 16.1 ms | 7.5× faster |
| 24 | 7.16–7.24 ms | 44–49 ms | 6.3× faster |

The specialised kernels themselves only engage when the operands satisfy the
production contraction contract (3×(4,4,nvec,nvec) perambulators, 3×(4,4)
gamma matrices, 2×(nmom,nvec,nvec,nvec) VVV); otherwise the hint prints
`[plan-cache]` and only the plan cache is used.

## Honest limitations

| Situation | Measured behaviour |
|---|---|
| GPU nucleon 2pt/3pt vs `opt_einsum` | ≈1× — four independent implementations land in 373–444 ms; the shape is bound by the hardware, not by the planner |
| perambulator × gauge on the full L24 volume (GPU) | parity (1.016×); the fused kernel currently applies only when `z·y·x = 1` |
| Small tensors (Nvec=20, `spatial=1`) | 0.51×–1.32× — per-call overhead dominates |
| Complex dense `zgemm` throughput on this laptop GPU | ~71 GFLOP/s vs ~228 GFLOP/s on 4 CPU threads; GPU wins only when the work is big enough |
| Shapes without a specialised kernel | generic path only; the plan is still `opt_einsum`-compatible, but no extra speed |
| MPI | only the KUPL backend shards, and only one output axis; never used in these comparisons |

The GHz-class caveat matters: the GPU numbers above were taken on a laptop GPU,
so absolute times are not representative of a data-centre card. Both sides were
measured on the same device, so the **ratios** remain meaningful.

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
