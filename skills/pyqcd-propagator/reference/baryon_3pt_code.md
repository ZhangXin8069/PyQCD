# Baryon 3pt — Sequential Source Workflow Reference

This file describes the **sequential source method** for baryon 3pt correlators.
It is a **complete, runnable template** matching the generate_einsum output.
Einsum strings for each specific operator combination come from generate_einsum;
this file shows the surrounding framework (setup, gamma definitions, inversions,
final contraction, save).

---

## Data layout

```
LatticePropagator.data.shape = [parity, t, z, y, x//2, spin_snk, spin_src, color_snk, color_src]
```

Spin/color order is always `[snk, src, col_snk, col_src]`.

---

## Workflow

### 1. Imports

```python
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source
```

### 2. Parameters (hard-coded)

```python
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = "/path/to/configs/beta6.20_mu-..._cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]
t_sep = 8

xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

Gamma_cur_bit = 1   # gamma(1) for vector current
Gamma_T_bit = 1     # projector gamma(1)

out_dir = "output"
```

### 3. Read gauge and initialize

```python
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)
```

### 4. Dirac operators

```python
dirac_l = core.getDirac(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getDirac(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
```

### 5. Forward propagators

```python
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)
```

After forward solves, identify which propagator serves as the spectator
in the 3pt sequential-source final contraction:

```python
# For Lambda -> p:   spectator = strange propagator
# For Xi -> Lambda:  spectator = strange propagator (the s that doesn't go through current)
# General rule: the propagator NOT passed through the sequential solve is the spectator
prop_current = prop_s
```

### 6. Gamma matrices and epsilon (GPU)

```python
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)            # gamma5
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray(gamma.gamma(Gamma_T_bit), dtype=cp.complex128)  # projector
Gamma_cur = cp.asarray(gamma.gamma(Gamma_cur_bit), dtype=cp.complex128)  # current

# Levi-Civita epsilon_{abc}
eps = cp.asarray(
    [[[0, 0, 0], [0, 0, 1], [0, -1, 0]],
     [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
     [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]],
    dtype=cp.complex128,
)
```

### 7. Momentum phases

For zero momentum, use a ones-phase field matching the local lattice shape:

```python
ones_phase = cp.ones(prop_l.data.shape[:5], dtype=cp.complex128)
```

For non-zero momentum, use `phase_v2.MomentumPhase(latt_info).getPhase(...)`.

### 8. Sink block B(x)

Einsum strings are obtained by calling `generate_einsum(type="baryon_3pt")`
**at generation time**. The returned strings are Python string literals
that get pasted directly into the `contract()` calls — no placeholders,
no direct Python import of codegen modules in the runtime script — always use the generate_einsum tool.

For $\Lambda \to p$ with $J = \bar u \gamma_1 s$ and $T = \gamma_1$:

```python
# The einsum strings below are the ACTUAL result of:
#   result = baryon_3pt_sink('Lambda', 'proton', 's', 'u', 'gamma1')
#   topo0['einsum'] = "wtzyx, ijk, lmn, AB, GH, ID, wtzyxAGil, wtzyxBHjm -> wtzyxIDnk"
#   topo1['einsum'] = "wtzyx, ijk, lmn, AB, GH, ID, wtzyxDGkl, wtzyxBHjm -> wtzyxIAni"
B = core.LatticePropagator(latt_info)
B.data = (
    + 1 * contract(
        "wtzyx, ijk, lmn, AB, GH, ID, wtzyxAGil, wtzyxBHjm -> wtzyxIDnk",
        ones_phase, eps, eps, Cg5, Cg5, Tmat, prop_l.data, prop_l.data,
    )
    - 1 * contract(
        "wtzyx, ijk, lmn, AB, GH, ID, wtzyxDGkl, wtzyxBHjm -> wtzyxIAni",
        ones_phase, eps, eps, Cg5, Cg5, Tmat, prop_l.data, prop_l.data,
    )
)
```

> **Critical note on propagator assignment in the sink block for Λ→p**:
> BOTH sink-block propagators must be LIGHT (`prop_l`), because they correspond to
> the u and d spectator quarks that go directly from source to sink WITHOUT
> passing through the current. The strange quark (prop_s) goes THROUGH the current
> (s → ūγ₁s → u) and enters ONLY in the final contraction.
> - The direct ε-structure in both Λ and p uses (u Cg5 d) — both light.
> - The strange quark from Λ is changed by the current into the third p quark.
> - Therefore the sink block = ε_{abc}(u_a Cg5 d_b) needs two light propagators.
> - `prop_current` = the current-side propagator (prop_s for ūγ₁s current).

### 9. Dagger and sequential source

```python
B.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, B.data.conj(), G5)
src_seq = source.sequential12(B, t_sep)
```

### 10. Sequential solve and second dagger

```python
with dirac_l.useGauge(gauge_stout):
    G_l_seq = core.invertPropagator(dirac_l, src_seq)

G_l_seq_dag = core.LatticePropagator(latt_info)
G_l_seq_dag.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, G_l_seq.data.conj(), G5)
```

### 11. Final contraction and MPI gather

```python
three_pt_site = contract(
    "wtzyxijba, jk, wtzyxkiab -> wtzyx",
    G_l_seq_dag.data, Gamma_cur, prop_current.data,
)

# Sum over spatial sites (q=0: no phase)
C3_t_local = contract("wtzyx -> t", three_pt_site)
C3_t = core.gatherLattice(array.arrayAsNumpy(C3_t_local, backend="cupy"), [0, -1, -1, -1])
```

> **gatherLattice returns None on non-root ranks.** Always guard downstream operations.

### 12. Save (rank 0 only)

```python
if core.getMPIRank() == 0:
    # Drop contact terms (optional)
    C3_window = np.asarray(C3_t[1:t_sep], dtype=np.complex128)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"c3_cfg{n_cfg:05d}_tsep{t_sep}.npy")
    np.save(out_path, C3_window)
```

For HDF5 with metadata, follow the [`pyqcd-infra` I/O reference](../../pyqcd-infra/references/io.md);
for atomic pipeline completion and resume semantics, follow the
[`pyqcd-pipeline` runbook](../../pyqcd-pipeline/references/runbook.md).

---

## Einsum generation

Always generate the sink block einsum strings using `generate_einsum(type="baryon_3pt")` from the Executor. The tool returns a complete code block ("code" key). Do NOT import codegen modules directly.

The einsum strings in this reference are for the specific $\Lambda\to p$ case.

---

## Common mistakes

| Mistake | Fix |
|---------|------|
| `cp.einsum` with mixed backends | Use `opt_einsum.contract` |
| `getPhase()` without `.data` | Always append `.data` |
| gamma matrix on CPU | `cp.asarray(..., dtype=cp.complex128)` |
| `gatherLattice` on CuPy array | `array.arrayAsNumpy()` first |
| Epsilon/propagator label mismatch | Generate via generate_einsum tool |
| `gatherLattice` result used on non-root rank | Guard with `if core.getMPIRank() == 0:` |

---

## PyQUDA API quick reference

| API | Returns |
|-----|---------|
| `core.invertPropagator(dirac, src)` | LatticePropagator |
| `source.source12(info, "point", pos)` | LatticeFermion (point source) |
| `source.sequential12(field, t_sink)` | LatticeFermion (sequential source) |
| `core.gatherLattice(arr, gather_dims)` | NumPy array on rank 0, None elsewhere |
| `array.arrayAsNumpy(arr, backend)` | NumPy array from GPU |
