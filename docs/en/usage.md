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

assert np.allclose(result, a @ b)
assert int(info.opt_cost) > 0
assert np.allclose(expr(a, b), a @ b)
```

Shared intermediates memoise repeated sub-contractions inside a context:

```python
with pe.shared_intermediates() as cache:
    pe.contract("ab,bc->ac", a, b, backend="numpy")
    pe.contract("ab,bc->ac", a, b, backend="numpy")
print(pe.count_cached_ops(cache))

ops = pe.count_cached_ops(cache)          # Counter, not a plain int
assert sum(ops.values()) >= 1, ops
```

### Backend and precision

```python
from pyqcd.tools import set_backend, get_backend, get_backend_name, get_precision

set_backend("numpy")          # also: "cupy", "torch"
print(get_backend_name(), get_precision())

assert get_backend_name() == "numpy"
assert get_precision() in {"complex64", "complex128"}
```

### Turning specialisations off

```python
pe.contract("ij,jk->ik", a, b, specialize=False)   # per call

assert np.allclose(pe.contract("ij,jk->ik", a, b, specialize=False), a @ b)
```

Environment switches: `PYQCD_EINSUM_SPECIALIZE=0`,
`PYQCD_EINSUM_DISABLE_FUSED=1`, `PYQCD_EINSUM_KERNEL_KUPL=0`,
`PYQCD_EINSUM_KERNEL_CUDA=0`.

## Lattice constants and Dirac algebra

```python
import pyqcd.lattice as L

print(L.Nc, L.Nd, L.Ns, L.fm2GeV)
print(L.gamma(5).shape)          # (4, 4) DeGrand–Rossi basis

assert (L.Nc, L.Nd, L.Ns) == (3, 4, 4)
assert L.gamma(5).shape == (4, 4)
assert L.fm2GeV > 0
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

assert abs(complex(G.wilson_loop(U, 1, 1, 0, 1)) - 1.0) < 1e-12     # unit links
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

assert R.flow_time_gev_m2(1.0, a_fm=0.1) > 0
assert set(F) and all(np.shape(v) == (2, 2, 2, 2, 3, 3) for v in F.values())
assert O.compute_dual_field_strength(F, 0, 1).shape == (2, 2, 2, 2, 3, 3)
```

Flow-based renormalised TMD:

```python
result = R.gradient_flow_renormalized_tmd(
    U, tau=1.0, z_list=[0, 2, 4], b_list=[0], z_dir=2, b_dir=0)

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

data = np.random.default_rng(0).standard_normal((20, 5))   # 20 configs x 5 slices
samples = A.Bootstrap(data, Nconf_axes=0)                  # dict of resampled arrays
print(sorted(samples))

assert {"data_mean", "data_err", "data_sample"} <= set(samples)
assert np.shape(samples["data_mean"])[0] == 5
```

## Parallel planning

```python
import pyqcd.parallel as P

resources = P.detect_resources()
plan = P.plan_parallel(m=12, a_mem_mb=500, resources=resources)
print(P.format_plan(plan))

assert isinstance(P.format_plan(plan), str)
assert {"N", "Y", "X", "m"} <= set(plan)      # plan_parallel returns a dict
assert plan["N"] >= 1 and plan["m"] == 12
```

The planner starts from `N*a = n*b` (a = memory per worker, b = usable memory per
GPU, n = number of GPUs) and reports the worker count `N`, the batch
`X = ceil(m/N)` and the per-GPU process count `Y = N/n`. The equality is the
*uncapped* target: `Y` is then reduced so that there is at least one task per
process (`Y <= m/n`), no more processes than CPU threads, and no more than RAM
allows, so with few tasks `N*a < n*b` — the reduction is reported in `notes`
(e.g. `Y capped from 13 to 3`) and `gpu_ok` still tells you whether one worker
fits in the per-GPU budget.

### Batch of independent contractions (`einsums`)

`pyqcd.parallel.einsum.einsums` runs a **batch of mutually independent**
contractions in parallel, one MPI process per work bin. Each task is an ordinary
`contract(...)` call, so its arguments and semantics are unchanged.

```python
import numpy as np
import pyqcd.einsum as pe
from pyqcd.parallel import einsums, plan_einsums

rng = np.random.default_rng(0)
tasks = {
    f"t{i}": {"subscripts": "ij,jk->ik",
              "operands": [rng.standard_normal((64, 64)),
                           rng.standard_normal((64, 64))]}
    for i in range(4)
}
plan = plan_einsums(tasks, max_procs=4)
result = einsums(tasks, max_procs=1, reminder=False, verbose=False)   # serial path

assert plan["workers"] == min(4, plan["limits"]["limit"])
assert sorted(i for bin_ in plan["bins"] for i in bin_) == [0, 1, 2, 3]
assert result.keys() == tasks.keys()
for key, spec in tasks.items():
    assert np.allclose(result[key], pe.contract(spec["subscripts"], *spec["operands"]))
assert plan["skew"] >= 1.0 and len(plan["bins"]) == plan["workers"]
```

```python
# docs-skip: launches mpirun worker processes
from pyqcd.parallel import einsums

result = einsums(tasks, max_procs=4)          # self-launched MPI pool
```

* **Container shape decides the result shape**: a `dict` of specs returns a
  `dict` with the same keys, a `list`/`tuple` returns a `list` in the same order.
* **A task spec** is `{"subscripts": str, "operands": [...], "kwargs": {...}}`
  (also accepted as `equation`/`eq`; `kwargs` may be omitted) or the equivalent
  `(subscripts, operands[, kwargs])` tuple. Per-task `kwargs` override the
  call-level `optimize`/`backend`.
* **Work balancing**: work per task is estimated from `contract_path`'s optimized
  FLOP count (with input/output element count as a floor for sum-free equations),
  then packed with LPT (longest-processing-time-first). The slowest process stays
  within `4/3 - 1/(3K)` of the optimal makespan, so bins can hold different
  numbers of tasks without creating a straggler.
* **Process count** defaults to `min(number of tasks, CPU threads, RAM budget)`;
  pass `max_procs=` to cap it. If you already launched with `mpirun -np N`, every
  rank calls `einsums` and the work is divided over those `N` ranks (no data
  staging; the payload returns on rank 0 unless `broadcast=True`). Without MPI it
  starts its own `mpirun -np K` pool and exchanges tasks/results through a
  temporary directory.
* **Precision follows the operands**: `complex64`/`float32` inputs halve both
  memory and arithmetic cost — the natural first lever on a memory-constrained
  machine. `einsums` never casts anything itself.

Measured on this 4-physical-core / 8-thread, 16 GB machine — real sush
distillation equation S1 (NN 2pt, 8 operands), `complex64` operands,
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1` (single-thread baseline, the fair
comparison point), results verified against serial `contract`:

| mode | workload | serial | `einsums` | speed-up |
|---|---|---|---|---|
| `mpirun -np 2` (collective) | S1, Nvec=40, 8 tasks | 168 ms | 88 ms | **1.86×** |
| `mpirun -np 4` (collective) | S1, Nvec=40, 8 tasks | 183 ms | 59 ms | **3.09×** |
| `mpirun -np 4` (collective), default threads | S1, Nvec=40, 8 tasks | 138 ms | 53 ms | 2.28× |
| self-launched pool, `max_procs=4` | S1, Nvec=60, 8 tasks | 736 ms | 622 ms | 1.18× |
| self-launched pool, `max_procs=4` | S1, Nvec=100, 8 tasks | 5326 ms | 2160 ms | **2.47×** |
| self-launched pool, `max_procs=4` | S1, Nvec=40, 8 tasks | 217 ms | 391 ms | 0.56× |

Two conclusions from the measurements, both reproducible with
`python -m pyqcd.testing.parallel_einsum bench` (self-launched pool) and
`mpirun -np 4 python -m pyqcd.testing.parallel_einsum bench-collective`
(collective):

* **Collective (already inside `mpirun`) is the efficient mode**: a 3.1× speed-up
  on 4 ranks with no data staging.
* **The self-launched pool pays a fixed cost**: MPICH `mpirun` start-up is ~150 ms
  and every operand is written to and read back from disk (~1.5 GB/s here). It only
  wins once the serial run takes roughly a second or more (1.18× at 0.74 s serial,
  2.47× at 5.3 s), and loses on small batches. Always compare with `bench` on your
  own machine instead of assuming a speed-up.

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
