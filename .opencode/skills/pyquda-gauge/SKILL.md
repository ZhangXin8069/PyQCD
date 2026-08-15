---
name: pyquda-gauge
description: >
  Generates code to compute pure-gauge observables (Wilson loops,
  Polyakov loops, topological charge, etc.) directly from gauge links.
  Does NOT involve fermion propagators, Dirac solvers, or inversions.
  This skill covers BOTH the physics reasoning (what to compute and why)
  and the code generation (how to compute it).
  Trigger on: "compute Wilson loop", "calculate Wilson loop",
  "measure Polyakov loop", "compute topological charge",
  "pure gauge observable", "gauge-only computation",
  "static potential extraction", "Wilson flow",
  "link smearing", or when the task explicitly involves gauge links only
  without fermion propagators.
---

# Gauge Observables Tool

## Purpose

Translate a pure-gauge observable specification into executable code
that reads gauge configurations and produces per-configuration results.
This skill generates a **data production script only** — no analysis code.

See `reference/wilson_loop.md` for full worked examples.

## Prerequisites

- Access to gauge configurations (LIME-format Chroma QIO files, etc.)
- A gauge-link processing library: PyQUDA (GPU), Python+NumPy (small / CPU-only), MILC, etc.
- Python 3.8+, numpy, h5py for output
- (Optional) MPI environment for GPU multi-rank runs

## Code style

- **Flat, self-contained script**: single file, parameters hardcoded at top.
  Read top-to-bottom, no deep class hierarchies.
- **MPI**: if using PyQUDA with multiple GPUs, launch via `mpirun -np N python script.py`
  or `srun -n N` on SLURM. For single-GPU or CPU-only runs, MPI is not required.
- **Heavy I/O, light computation**: Gauge loading dominates runtime.
  Measurement is typically fast. No progress bars.

---

## Physics reasoning (summary)

### Wilson loop

$$
W_{\mu\nu}(R,T) = \frac{1}{N_c} \text{Re}\,\text{Tr}\,
\mathcal{P} \left[
\prod_{i=0}^{R-1} U_\mu(x + i\hat{\mu})
\prod_{j=0}^{T-1} U_\nu(x + R\hat{\mu} + j\hat{\nu})
\prod_{i=0}^{R-1} U_\mu^\dagger(x + T\hat{\nu} + (R-1-i)\hat{\mu})
\prod_{j=0}^{T-1} U_\nu^\dagger(x + (T-1-j)\hat{\nu})
\right]
$$

Path: forward $\mu$ (R) → forward $\nu$ (T) → backward $\mu$ (R) → backward $\nu$ (T).

### Static potential

$$V(R) = -\lim_{T\to\infty} \frac{1}{T} \log \langle W(R,T) \rangle, \qquad
V(R) = V_0 + \frac{\alpha}{R} + \sigma R$$

Confinement ⇒ $\sigma > 0$.

### Key differences from hadronic correlators

| | Wilson loops | Hadronic correlators |
|---|---|---|
| I/O | Gauge links only | Need quark propagators (inversions) |
| Cost | Fast, all-to-all | One inversion per source |
| Signal | Noisy at large area | SNR decays differently |
| Smearing | Stout/APE/HYP on links | Gaussian smearing on sources |

### Plane choices & smearing

| Plane | Use |
|-------|-----|
| XT | Standard potential |
| YT/ZT | Rotational symmetry check |
| XY | Spatial string tension, glueballs |

| Smearing | Parameters | Notes |
|----------|-----------|-------|
| Stout | $\rho=0.08$–$0.12$, 1–3 iter | Analytic, differentiable |
| APE | $\alpha=0.5$–$0.75$, 1–5 iter | Simple |
| HYP | $(\alpha_1,\alpha_2,\alpha_3)$ | Aggressive UV suppression |

---

## Conventions (PyQUDA alignment)

Where the implementation uses PyQUDA, these conventions apply.
For other libraries (e.g. MILC, pure NumPy), adjust accordingly.

### Lattice field layout

Gauge: `[4, 2, Lt, Lz, Ly, Lx//2, Nc, Nc]`
- 0 = direction (4)
- 1 = parity (even/odd)
- 2-4 = local lattice (tzyx)
- 5 = Lx//2 (even-odd preconditioning)
- 6-7 = color (Nc×Nc)

`LatticeLink` (from `gauge.loop()`): `[2, Lt, Lz, Ly, Lx//2, Nc, Nc]`

### Direction constants (PyQUDA)

| Constant  | Meaning |
|----------|-------|---------|
| X | +x |
| Y | +y |
| Z | +z |
| T | +t |
| -X | -x |
| -Y | -y |
| -Z | -z |
| -T | -t |

---

## Workflow: code generation (PyQUDA example)

### Step 1: Initialize

```python
from pyquda_utils import core
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]           # 4 MPI ranks in t — optional, can use [1,1,1,1]
core.init(grid_size, latt_size, resource_path="./tunecache")
```

### Step 2: Load gauge

```python
from pyquda_utils import io
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
```

### Step 3: (Optional) Link smearing

```python
gauge.stoutSmear(n_step=1, rho=0.1, n_dim=4)
```

### Step 4: Compute Wilson loops

Use `gauge.loop()` with direction lists:

```python
from pyquda_utils.core import X, Y, Z, T

path_XT = [X]*R + [T]*Tlen + [-X]*R + [-T]*Tlen
path_YT = [Y]*R + [T]*Tlen + [-Y]*R + [-T]*Tlen
path_ZT = [Z]*R + [T]*Tlen + [-Z]*R + [-T]*Tlen

res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])
```

**⚠️ PyQUDA caveat**: `gauge.loop()` requires **exactly 4 outer groups**.
Pad with weight 0 if fewer planes are needed. For other libraries this
constraint may not apply.

### Step 5: Extract per-site ReTr

```python
import numpy as np
Nc = gauge.latt_info.Nc
U = res[i].getHost().reshape(-1, Nc, Nc)    # GPU→CPU, flatten sites
tr = np.trace(U, axis1=-2, axis2=-1)         # complex trace per site
tr_real = tr.real                             # ReTr per site
```

### Step 6: Global MPI reduction (PyQUDA)

Use `core.gatherLattice` — **NOT** `mpi4py.MPI.Allreduce` directly:

```python
grid = core.getGridSize()
local_shape = (2, latt_size[3]//grid[3], latt_size[2], latt_size[1], latt_size[0]//grid[0]//2)
re_tr_field = tr_real.reshape(local_shape)
global_sum = core.gatherLattice(re_tr_field, [-1, -1, -1, -1])

if core.getMPIRank() == 0:
    wl_value = float(global_sum.sum()) / (total_sites * Nc)
```

For non-MPI runs (single GPU / CPU NumPy), simply average over all sites.

### Step 7: Save to HDF5

```python
import h5py
if core.getMPIRank() == 0 or not MPI:
    with h5py.File(out_path, "w") as f:
        g = f.create_group("/wilson_loops")
        g.create_dataset("W_R{Tlen}_T{Tlen}".format(Tlen=Tlen), data=np.float64(wl_value))
        g.attrs["R"] = np.int32(R)
        g.attrs["T"] = np.int32(Tlen)
```

---

## Other gauge observables

- **Polyakov loops**: `gauge.polyakov()` if available (PyQUDA). Same extraction: `getHost()` → trace → reduce.
- **Topological charge**: Clover-leaf $F_{\mu\nu}$ from plaquettes, then
  $Q = \frac{1}{32\pi^2} \sum_x \epsilon_{\mu\nu\rho\sigma} \text{Tr}[F_{\mu\nu}F_{\rho\sigma}]$.
- **Wilson flow**: Iterative smoothing. Compute $E(t)$ and $Q(t)$ per flow time.

## Common issues

| Problem | Fix |
|---------|-----|
| `IndexError` in `gauge.loop()` (PyQUDA) | Always pass exactly 4 groups (dummy with weight 0 if needed) |
| GPU OOM | More GPUs, or use CPU backend for small lattices |
| `LatticeLink` treated as scalar (PyQUDA) | Always `getHost()` → `reshape()` → `trace()` |
| `gatherLattice` shape mismatch (PyQUDA) | Input must match `(2, Lt, Lz, Ly, Lx//2)` locally |
| Noisy loops | Apply smearing before loop, start with 1-step stout $\rho=0.1$ |
