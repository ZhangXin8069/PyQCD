# Wilson Loops — Reference

## 1. Physics

### 1.1 Definition

On the lattice, the **Wilson loop** $W_{\mu\nu}(R,T)$ is a gauge-invariant
observable constructed from gauge links along a closed rectangular path:

$$
W_{\mu\nu}(R,T) = \frac{1}{N_c} \text{Re}\,\text{Tr}\,
\mathcal{P} \bigg[
\prod_{i=0}^{R-1} U_\mu(x + i\hat{\mu})
\prod_{j=0}^{T-1} U_\nu(x + R\hat{\mu} + j\hat{\nu})
\prod_{i=0}^{R-1} U_\mu^\dagger(x + T\hat{\nu} + (R-1-i)\hat{\mu})
\prod_{j=0}^{T-1} U_\nu^\dagger(x + (T-1-j)\hat{\nu})
\bigg]
$$

Path (starting at $x$):
1. $R$ steps forward along $\mu$
2. $T$ steps forward along $\nu$
3. $R$ steps backward along $\mu$
4. $T$ steps backward along $\nu$

returns to $x$. $\frac{1}{N_c}\text{Re}\,\text{Tr}$ is real by construction.
Closure makes $\text{Tr}\,U_C$ gauge invariant, but it does not make the trace real:
for SU(3), a single oriented loop generally has a complex trace, while reversing the
path gives $\text{Tr}(U_C^{-1})=\text{Tr}(U_C)^*$. Taking `Re Tr` is therefore an
explicit observable projection, not a consequence of path closure.

### 1.2 Static potential and area law

The vacuum expectation value of a rectangular Wilson loop behaves as:

$$
\langle W(R,T) \rangle \sim e^{-V(R)\,T},\qquad T \gg R
$$

This defines the **static quark-antiquark potential**:

$$
V(R) = -\lim_{T\to\infty} \frac{1}{T} \log \langle W(R,T) \rangle
$$

**Area law for confinement**: if the theory confines, the Wilson loop
satisfies an area law:

$$
\langle W(R,T) \rangle \sim e^{-\sigma \, R\,T},\qquad R,T \to \infty
$$

⇒ $V(R) \sim \sigma R$ (linear rising potential), where $\sigma$ is the
**string tension**. In QCD, $\sigma \approx (420\text{ MeV})^2 \approx 0.18\text{ GeV}^2$.

**Coulomb + linear fit** parametrizes the potential at all $R$:

$$
V(R) = V_0 + \frac{\alpha}{R} + \sigma R
$$

- $\alpha$: Coulomb coefficient (from one-gluon exchange at short distances)
- $\sigma$: string tension (confinement slope at large $R$)
- $V_0$: additive constant (renormalization dependent)

### 1.3 Effective mass / ratio method

In practice, instead of fitting $V(R)$ from $\log\langle W(R,T)\rangle$, define:

$$
V_{\text{eff}}(R,T) = \log\frac{\langle W(R,T) \rangle}{\langle W(R,T+1) \rangle}
$$

which approaches $V(R)$ as $T\to\infty$. A plateau in $T$ indicates ground-state
dominance. This is analogous to effective mass plots for hadronic correlators.

### 1.4 Creutz ratio

The Creutz ratio cancels self-energy and Coulomb contributions, isolating the
string tension:

$$
\chi(R,T) = -\log\frac{\langle W(R,T) \rangle \langle W(R-1,T-1) \rangle}
{\langle W(R-1,T) \rangle \langle W(R,T-1) \rangle}
$$

For the area law $\langle W \rangle \sim e^{-\sigma RT}$:

$$
\chi(R,T) \to \sigma \quad \text{as } R,T \to \infty
$$

### 1.5 Polyakov loop

The Polyakov loop wraps around the temporal direction:

$$
P(\vec{x}) = \frac{1}{N_c} \text{Tr} \prod_{t=0}^{L_t-1} U_4(\vec{x}, t)
$$

Its expectation value is the order parameter for the **finite-temperature
deconfinement transition**:
- $\langle P \rangle = 0$ ↔ confined phase (center symmetry unbroken)
- $\langle P \rangle \neq 0$ ↔ deconfined phase (center symmetry broken)

Polyakov loop correlators give the **singlet / averaged** static potential
at finite temperature.

---

## 2. Link smearing — quick reference

| Method | Parameters | Use case |
|--------|-----------|----------|
| **Stout** | $\rho=0.08$–$0.12$, 1–3 iter | Analytic, differentiable. Default choice. |
| **APE** | $\alpha=0.5$–$0.75$, 1–5 iter | Simple, non-analytic |
| **HYP** | $(\alpha_1,\alpha_2,\alpha_3)=(1.0,0.5,0.5)$ | Aggressive UV suppression |

In PyQUDA: `gauge.stoutSmear(n_step=2, rho=0.1, n_dim=4)`.

---

## 3. Code

### 3.1 `gauge.loop()` — what it returns

```python
res = gauge.loop(groups, weights)
```

- `groups`: list of **exactly 4** outer groups. Each group is a list of paths
  (typically one path per group). If fewer than 4 paths are needed, pad the
  remaining groups with a dummy and set its weight to 0.
- `weights`: list of 4 floats, one per group.
- **Returns**: a `LatticeGauge` with shape `[4, 2, Lt, Lz, Ly, Lx//2, Nc, Nc]`.
  Each `res[i]` is a `LatticeLink` (per-site SU(Nc) matrix field), NOT a scalar.

To extract a per-site scalar:
```python
U = res[i].getHost()          # GPU → CPU
U_flat = U.reshape(-1, Nc, Nc)
re_tr = np.trace(U_flat, axis1=-2, axis2=-1).real   # shape: (n_sites,)
```

### 3.2 Minimal working example: one (R,T), one plane

Compute a single Wilson loop `W(R=2, T=3)` in the XT plane:

```python
# Launch: mpirun -np 4 python wilson_loop.py ~/.cache 10000
import os, sys
import numpy as np
import h5py
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── Parameters ────────────────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]     # 4 MPI ranks in t
Nc = 3
R = 2
Tlen = 3
cfg_path_template = "/path/to/configs"
out_dir = "./output"

# ── Init & load gauge ─────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# ── Build path: XT plane, R=2, T=3 ────────────────────────
# forward x ×2 → forward t ×3 → backward x ×2 → backward t ×3
path = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen

# gauge.loop() requires exactly 4 groups; use weight 0 for dummies
res = gauge.loop([[path], [path], [path], [path]], [1, 0, 0, 0])

# ── Extract per-site ReTr ─────────────────────────────────
U = res[0].getHost().reshape(-1, Nc, Nc)
re_tr = np.trace(U, axis1=-2, axis2=-1).real   # shape: (total_sites,)

# ── MPI gather to rank 0 ──────────────────────────────────
grid = core.getGridSize()
local_shape = (2, latt_size[3] // grid[3], latt_size[2],
               latt_size[1], (latt_size[0] // grid[0]) // 2)
field = re_tr.reshape(local_shape)
global_sum = core.gatherLattice(field, [-1, -1, -1, -1])

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

if core.getMPIRank() == 0:
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    print(f"W(R={R}, T={Tlen}) = {W_val:.6f}")

    # ── Save ──
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(os.path.join(out_dir, f"wl_cfg{n_cfg:04d}.h5"), "w") as f:
        g = f.create_group("/wilson_loops")
        g.create_dataset(f"W_R{R}_T{Tlen}", data=np.float64(W_val))
        g.attrs.update(lattice_dimensions=np.array(latt_size, dtype=np.int32),
                       R=np.int32(R), T=np.int32(Tlen), Nc=np.int32(Nc))
```

### 3.3 Multi-plane averaging (XT+YT+ZT)

To average over 3 spatial-temporal planes for better statistics:

```python
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])

# Average ReTr over 3 planes
re_tr_sum = None
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_sum = re_tr if re_tr_sum is None else re_tr_sum + re_tr

re_tr_avg = re_tr_sum / 3.0

# Then MPI gather as above, save as float(global_sum.sum()) / (total_sites * Nc)
```

### 3.4 Loop over R and T

See the previous (too complicated) version. In short: wrap the steps above
in a double loop over R and T, building the path dynamically with
`[X]*R + [T]*Tlen + [-X]*R + [-T]*Tlen`.

### 3.5 gauge.loop() 4-group packing

| # active planes | Groups | Weights |
|-----------------|--------|---------|
| 1 | `[[p1], [p1], [p1], [p1]]` | `[1, 0, 0, 0]` |
| 2 | `[[p1], [p2], [p1], [p2]]` | `[1, 1, 0, 0]` |
| 3 | `[[p1], [p2], [p3], [p1]]` | `[1, 1, 1, 0]` |
| 4 | `[[p1], [p2], [p3], [p4]]` | `[1, 1, 1, 1]` |

### 3.6 Path construction reference

```python
# XT plane, R=3, T=4
[X, X, X, T, T, T, T, -X, -X, -X, -T, -T, -T, -T]

# XY plane, R=2, T=5
[X, X, Y, Y, Y, Y, Y, -X, -X, -Y, -Y, -Y, -Y, -Y]

# ZY plane, R=3, T=2
[Z, Z, Z, Y, Y, -Z, -Z, -Z, -Y, -Y]
```

---

## 4. Common mistakes

| Mistake | Fix |
|---------|-----|
| Assuming `gauge.loop()` returns scalars | It returns `LatticeGauge` (per-site Nc×Nc matrices). Must `.getHost()` → `.reshape()` → `np.trace()`. |
| Missing MPI gather | Each rank has 1/grid_size of the lattice. Use `gatherLattice(field, [-1,-1,-1,-1])` to sum to rank 0. |
| `float(global_sum)` instead of `float(global_sum.sum())` | `gatherLattice` returns an array, `.sum()` collapses it to scalar. |
| Forgetting `getHost()` before reshape | PyQUDA GPU fields need explicit CPU transfer. |
| 1/Nc normalization | `W = global_sum / (total_sites * Nc)`. The `/ Nc` normalizes the trace. |

---

## 6. Quick reference

| Equation | Purpose |
|----------|---------|
| $W_{\mu\nu}(R,T) = \frac{1}{N_c}\text{ReTr}[...]$ | Definition |
| $\langle W(R,T) \rangle \sim e^{-\sigma RT}$ | Area law for confinement |
| $V(R) = V_0 + \alpha/R + \sigma R$ | Static potential fit |
| $V_{\text{eff}}(R,T) = \log\frac{W(R,T)}{W(R,T+1)}$ | Effective potential |
| $\chi(R,T) = -\log\frac{W(R,T)W(R-1,T-1)}{W(R-1,T)W(R,T-1)}$ | Creutz ratio |
| $P(\vec{x}) = \frac{1}{N_c}\text{Tr}\prod_t U_4(\vec{x},t)$ | Polyakov loop |
