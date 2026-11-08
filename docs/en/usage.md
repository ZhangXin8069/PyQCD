# Usage

Every example below runs on CPU with only `numpy`; the optional native
backends change speed, not results.

## Contractions

```python
import numpy as np
import pyqcd.einsum as pe

rng = np.random.default_rng(0)
a = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))).astype(np.complex128)
b = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(np.complex128)

result = pe.contract("ij,jk->ik", a, b)
path, info = pe.contract_path("ij,jk->ik", a, b)
print(int(info.opt_cost), path)

expr = pe.contract_expression("ij,jk->ik", (3, 4), (4, 5))
for _ in range(3):
    expr(a, b)           # reuses the compiled plan
```

Shared intermediates memoise repeated sub-contractions inside a context:

```python
with pe.shared_intermediates() as cache:
    pe.contract("ab,bc->ac", a, b, backend="numpy")
    pe.contract("ab,bc->ac", a, b, backend="numpy")
print(pe.count_cached_ops(cache))
```

### Backend and precision

```python
from pyqcd.tools import set_backend, get_backend, get_backend_name, get_precision

set_backend("numpy")          # also: "cupy", "torch"
print(get_backend_name(), get_precision())
```

### Turning specialisations off

```python
pe.contract("ij,jk->ik", a, b, specialize=False)   # per call
```

Environment switches: `PYQCD_EINSUM_SPECIALIZE=0`,
`PYQCD_EINSUM_DISABLE_FUSED=1`, `PYQCD_EINSUM_KERNEL_KUPL=0`,
`PYQCD_EINSUM_KERNEL_CUDA=0`.

## Lattice constants and Dirac algebra

```python
import pyqcd.lattice as L

print(L.Nc, L.Nd, L.Ns, L.fm2GeV)
print(L.gamma(5).shape)          # (4, 4) DeGrand–Rossi basis
```

## Gauge observables and smearing

```python
import numpy as np
import pyqcd.gauge as G
import pyqcd.smear as S

# a trivial 2^4 lattice of unit links: U[t, z, y, x, mu, colour, colour]
U = np.zeros((2, 2, 2, 2, 4, 3, 3), dtype=np.complex128)
for mu in range(4):
    U[..., mu, :, :] = np.eye(3)

print(G.wilson_loop(U, 1, 1, 0, 1))
print(G.polyakov_loop(U).shape)
print(S.stout_smear(U, nstep=1).shape)     # (2, 2, 2, 2, 4, 3, 3)
print(S.hyp_smear(U).shape)
```

## Gradient flow and gluon operators

```python
import numpy as np
import pyqcd.gauge as G
import pyqcd.operator as O
import pyqcd.renorm as R

rng = np.random.default_rng(0)
U = np.zeros((2, 2, 2, 2, 4, 3, 3), dtype=complex)      # random SU(3) links
for mu in range(4):
    q, _ = np.linalg.qr(rng.standard_normal((2, 2, 2, 3, 3))
                        + 1j * rng.standard_normal((2, 2, 2, 3, 3)))
    U[..., mu, :, :] = q

print(R.flow_time_gev_m2(1.0, a_fm=0.1))               # flow time in GeV^-2
F = {(mu, nu): G.clover_field_strength(U, mu, nu)
     for mu in range(4) for nu in range(4) if mu != nu}
print(O.compute_dual_field_strength(F, 0, 1).shape)    # (2, 2, 2, 2, 3, 3)
```

Flow-based renormalised TMD:

```python
result = R.gradient_flow_renormalized_tmd(
    U, tau=1.0, z_list=[0, 2, 4], b_list=[0], z_dir=2, b_dir=0)
```

## Distillation contractions

```python
import pyqcd.contraction as C

# pseudo-code: the registries must be filled with real perambulators,
# vertices and gamma matrices before this block runs.
result = C.dynamic_contraction(
    operator_groups, peram_registry=..., v_registry=..., gamma_registry=...,
    Cpt="2pt", verbose=True)
```

`parity_and_boundary` applies the parity projectors and the anti-periodic
boundary sign; `classify_structure` and `identify_equivalent_diagrams` help
group equivalent operator structures before contracting.

## Analysis helpers

```python
import numpy as np
import pyqcd.analysis as A

data = np.random.default_rng(0).standard_normal((20, 5))   # 20 configs x 5 slices
samples = A.Bootstrap(data, Nconf_axes=0)                  # dict of resampled arrays
print(sorted(samples))
```

## Parallel planning

```python
import pyqcd.parallel as P

resources = P.detect_resources()
plan = P.plan_parallel(m=12, a_mem_mb=500, resources=resources)
print(P.format_plan(plan))
```

The planner follows `N*a = n*b` (a = memory per worker, b = usable memory per
GPU, n = number of GPUs) and reports the worker count `N`, the batch `X = m/N`
and the per-GPU process count `Y = N/n`.

## See also

Relative speed against `opt_einsum`, the measurement protocol and the cases
where there is no gain are on :doc:`performance`.

### Differences from `opt_einsum` 3.4.0

Two deliberate deviations, both additive:

* `optimize=` also accepts a **callable** `(input_subsets, output_subset, size_dict) -> path`.
  `opt_einsum` 3.4.0 raises `TypeError: 'function' object is not iterable` for the same
  argument; PyQCD plans with it.
* `pyqcd.einsum.PathInfo` is exported at package level (it lives under `opt_einsum.paths`
  upstream).

Everything else — including the `constants=` handling of `contract_expression`, which
raises the same `ValueError` in both — behaves identically.
