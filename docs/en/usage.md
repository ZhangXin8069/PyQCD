# Usage

Every example below runs on CPU with only `numpy`; the optional native
backends change speed, not results.

## Contractions

```python
import numpy as np
import pyqcd.einsum as pe

rng = np.random.default_rng(0)

# Two complex matrices: contract takes exactly what numpy would, plus a smarter planner.
a = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))).astype(np.complex128)
b = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(np.complex128)

# 1) Just compute it: parse the contract, plan, execute — one call.
result = pe.contract("ij,jk->ik", a, b)

# 2) Ask how it intends to do it: contract_path returns the order and the cost
#    without touching the data.
path, info = pe.contract_path("ij,jk->ik", a, b)
print(int(info.opt_cost), path)

# 3) When the same equation runs repeatedly, compile the plan once and reuse it.
expr = pe.contract_expression("ij,jk->ik", (3, 4), (4, 5))
for _ in range(3):
    expr(a, b)           # reuses the compiled plan, no re-planning

# All three routes must produce the same matrix product.
assert np.allclose(result, a @ b)
assert int(info.opt_cost) > 0
assert np.allclose(expr(a, b), a @ b)
```

Shared intermediates memoise repeated sub-contractions inside a context:

```python
# Repeated sub-contractions inside the context are memoised: the second call hits the cache.
with pe.shared_intermediates() as cache:
    pe.contract("ab,bc->ac", a, b, backend="numpy")
    pe.contract("ab,bc->ac", a, b, backend="numpy")
print(pe.count_cached_ops(cache))

# The count is a Counter keyed by operand signature, not a single integer.
ops = pe.count_cached_ops(cache)
assert sum(ops.values()) >= 1, ops
```

### Backend and precision

```python
from pyqcd.tools import set_backend, get_backend, get_backend_name, get_precision

# Global backend: numpy (default, CPU) / cupy (NVIDIA GPU) / torch (CPU or CUDA).
set_backend("numpy")          # also accepts "cupy" and "torch"
print(get_backend_name(), get_precision())

assert get_backend_name() == "numpy"
assert get_precision() in {"complex64", "complex128"}
```

### Turning specialisations off

```python
# specialize=False affects this call only: skip the equation-specific path and use the
# generic planner instead.
pe.contract("ij,jk->ik", a, b, specialize=False)

# The result must be identical to the specialised path — specialisation only changes speed.
assert np.allclose(pe.contract("ij,jk->ik", a, b, specialize=False), a @ b)
```

Environment switches: `PYQCD_EINSUM_SPECIALIZE=0`,
`PYQCD_EINSUM_DISABLE_FUSED=1`, `PYQCD_EINSUM_KERNEL_KUPL=0`,
`PYQCD_EINSUM_KERNEL_CUDA=0`.

## Lattice constants and Dirac algebra

```python
import pyqcd.lattice as L

# Basic constants: colours, spacetime dimensions, spinor dimension, lattice spacing factor.
print(L.Nc, L.Nd, L.Ns, L.fm2GeV)

# Gamma matrices use the DeGrand-Rossi basis; gamma(5) is 4x4.
print(L.gamma(5).shape)

assert (L.Nc, L.Nd, L.Ns) == (3, 4, 4)
assert L.gamma(5).shape == (4, 4)
assert L.fm2GeV > 0
```

## Gauge observables and smearing

```python
import numpy as np
import pyqcd.gauge as G
import pyqcd.smear as S

# Build a trivial 2^4 gauge field: every link is the identity.
# Layout is U[t, z, y, x, mu, colour, colour].
U = np.zeros((2, 2, 2, 2, 4, 3, 3), dtype=np.complex128)
for mu in range(4):
    U[..., mu, :, :] = np.eye(3)

# Wilson loop, Polyakov loop, Stout smearing, HYP smearing.
print(G.wilson_loop(U, 1, 1, 0, 1))
print(G.polyakov_loop(U).shape)
print(S.stout_smear(U, nstep=1).shape)     # (2, 2, 2, 2, 4, 3, 3)
print(S.hyp_smear(U).shape)

# With unit links a 1x1 Wilson loop is exactly 1; smearing never changes the shape.
assert abs(complex(G.wilson_loop(U, 1, 1, 0, 1)) - 1.0) < 1e-12
assert G.polyakov_loop(U).shape == (2, 2, 2) or G.polyakov_loop(U).size > 0
assert S.stout_smear(U, nstep=1).shape == U.shape
assert S.hyp_smear(U).shape == U.shape
```

## Gradient flow and gluon operators

```python
import numpy as np
import pyqcd.gauge as G
import pyqcd.operator as O
import pyqcd.renorm as R

# One random SU(3) link per site and direction (take the orthogonal factor of a QR).
rng = np.random.default_rng(0)
U = np.zeros((2, 2, 2, 2, 4, 3, 3), dtype=complex)
for mu in range(4):
    q, _ = np.linalg.qr(rng.standard_normal((2, 2, 2, 3, 3))
                        + 1j * rng.standard_normal((2, 2, 2, 3, 3)))
    U[..., mu, :, :] = q

# Flow-time conversion: tau=1.0 at a=0.1 fm is how many GeV^-2?
print(R.flow_time_gev_m2(1.0, a_fm=0.1))

# Clover field-strength tensor F_{mu nu}, one tensor per (mu, nu) pair; then the
# dual tensor F~_{01}.
F = {(mu, nu): G.clover_field_strength(U, mu, nu)
     for mu in range(4) for nu in range(4) if mu != nu}
print(O.compute_dual_field_strength(F, 0, 1).shape)    # (2, 2, 2, 2, 3, 3)

assert R.flow_time_gev_m2(1.0, a_fm=0.1) > 0
assert set(F) and all(np.shape(v) == (2, 2, 2, 2, 3, 3) for v in F.values())
assert O.compute_dual_field_strength(F, 0, 1).shape == (2, 2, 2, 2, 3, 3)
```

Flow-based renormalised TMD:

```python
# Gluon TMD after gradient-flow renormalisation: given a flow time tau and two sets of
# separations (z, b), return the matrix element.
result = R.gradient_flow_renormalized_tmd(
    U, tau=1.0, z_list=[0, 2, 4], b_list=[0], z_dir=2, b_dir=0)

# Shape is (len(z_list), len(b_list)) and every entry must be finite.
assert np.shape(result) == (3, 1), np.shape(result)
assert np.all(np.isfinite(np.asarray(result))), "non-finite TMD result"
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

# Fake data: 20 configurations x 5 time slices; axis 0 is the configuration axis.
data = np.random.default_rng(0).standard_normal((20, 5))
samples = A.Bootstrap(data, Nconf_axes=0)                  # dict of resampled arrays
print(sorted(samples))

# The dict carries mean, error and per-sample values; the slice axis matches the input.
assert {"data_mean", "data_err", "data_sample"} <= set(samples)
assert np.shape(samples["data_mean"])[0] == 5
```

## Using PyTorch tensors

PyTorch support is optional (`pip install "pyqcd[torch]"`) and is **type driven**:
`pyqcd.einsum.contract` looks at the operands, not at a global switch.

| Operands | Executes on | Result type | Native KUPL/CUDA library |
|---|---|---|---|
| all `torch.Tensor` | `torch.einsum` on the tensors' device | `torch.Tensor` (same device) | **not used** |
| all `numpy.ndarray` | KUPL (CPU) or the BLAS planner | `numpy.ndarray` | yes |
| all `cupy.ndarray` | CUDA kernels / cuBLAS | `cupy.ndarray` | yes |
| mixed numpy + torch | falls back to the numpy path | `numpy.ndarray` | yes |

Consequences worth knowing:

* torch tensors never reach the KUPL/CUDA libraries, so the specialised kernels
  and the per-call thread profiles do not apply — you get `torch.einsum` quality.
  The first native attempt logs one `RuntimeWarning: native plan unavailable …
  Cannot interpret 'torch.complex128' as a data type`; that is expected.
* to get the native speed with torch data, convert at the boundary:
  `t.detach().cpu().contiguous().numpy()` for KUPL, `cupy.asarray(t)` for CUDA.
* `set_backend("torch")` changes what `pyqcd.tools.get_backend()` returns for the
  physics modules; it does **not** redirect `contract()` on numpy input.
* autograd works (the complex case needs a real scalar, e.g. `y.real.sum()`).

```python
import numpy as np
import torch
import pyqcd.einsum as pe
import pyqcd.tools as T

# 1) torch tensors are dispatched to torch.einsum — no extra flag needed
a = torch.randn(4, 5, dtype=torch.complex128)
b = torch.randn(5, 6, dtype=torch.complex128)
out = pe.contract("ij,jk->ik", a, b)          # Tensor, same device as the inputs
print(type(out).__name__, out.dtype, bool(torch.allclose(out, a @ b)))

# 2) a CUDA tensor keeps running on the GPU
if torch.cuda.is_available():
    ga, gb = a.to("cuda"), b.to("cuda")
    print("device:", pe.contract("ij,jk->ik", ga, gb).device)

# 3) autograd flows through the contraction (use a real dtype for the scalar)
x = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
y = pe.contract("ij,jk->ik", x, torch.randn(5, 6, dtype=torch.float64))
y.sum().backward()
print("grad ok:", tuple(x.grad.shape))

# 4) make torch the default backend for pyqcd.tools / the physics modules
T.set_backend("torch", device="cuda" if torch.cuda.is_available() else "cpu")
print(T.get_backend_name(), T.get_precision(), T.get_torch_device())
T.set_precision("complex64")                  # switch the global complex precision
T.set_precision("complex128")
T.set_backend("numpy")                        # back to the default

assert torch.allclose(out, a @ b)

# 5) and the correctness contract: same numbers as numpy
assert np.allclose(np.asarray(out), np.asarray(a) @ np.asarray(b))
```

`set_precision("complex64")` halves the memory traffic of the torch path as well;
the gamma-matrix helpers (`pyqcd.lattice.gamma`) follow the same global precision.

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
