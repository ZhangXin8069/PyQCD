## Example: Rho meson mass (ρ⁺ channel)

> **Notation note**: In the code, the light quark propagator is called `prop_l` and the strange propagator `prop_s`. The LaTeX $S_l$ and $S_s$ in the formulas below correspond to `prop_l` and `prop_s` respectively.

**Goal**: Extract $m_\rho$. We need to calculate the two-point correlation function of $\rho^+$.

**Step 1 — Operator**:
  $$\mathcal{O}_{\rho^+_i} = \bar{d} \gamma_i u \quad (i = 1, 2, 3 \text{ for the three polarizations})$$
We have the corresponding Dirac conjugate operator (creation operator):
  $$\mathcal{O}_{\rho^+_i}^\dagger = \bar{u} \gamma_4\gamma_i^\dagger\gamma_4 d \quad (i = 1, 2, 3 \text{ for the three polarizations})$$

**Step 2 — Correlator**: Average over polarizations for better statistics:
  $$C_\rho(\vec{p}; t,0) = \frac{1}{3} \sum_i \langle \mathcal{O}_{\rho^+_i}(\vec{p},t) \mathcal{O}^\dagger_{\rho^+_i}(\vec{p},0) \rangle$$

**Step 3a — Quark fields**: Expand the operator in terms of quark fields and Fourier transform:
  $$C_\rho(\vec{p}; t,0) = \frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \langle \bar{d}(\vec{x},t) \gamma_i u(\vec{x},t) \bar{u}(\vec{y},0) \gamma_4 \gamma_i^\dagger \gamma_4 d(\vec{y},0) \rangle$$

**Step 3b — Wick contraction**:
Same topology as the pion, just replace $\gamma_5 \to \gamma_i$:

  $$C_\rho(\vec{p}; t,0) = -\frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_d(\vec{y},0; \vec{x},t) \gamma_i S_u(\vec{x},t; \vec{y},0) \gamma_4 \gamma_i^\dagger \gamma_4 ]$$

**Step 3c — Simplification**:
  1. Apply the $\gamma_5$-hermiticity and the flavor symmetry:
  $$C_\rho(\vec{p}; t,0) = -\frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ \gamma_5 S_l^\dagger(\vec{x},t; \vec{y},0) \gamma_5 \gamma_i S_l(\vec{x},t; \vec{y},0) \gamma_4 \gamma_i^\dagger \gamma_4 ]$$
  2. Apply the cyclic property to simplify the gamma matrix structure. Unlike the pion case, $\gamma_5 \gamma_i$ does not reduce to the identity — you must explicitly carry the spin-color matrix through the trace:
  $$C_\rho(\vec{p}; t,0) = \frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_l^\dagger(\vec{x},t; \vec{y},0) (\gamma_5 \gamma_i) S_l(\vec{x},t; \vec{y},0) (\gamma_i \gamma_5) ]$$

**Step 4 — Propagators needed**:
As in the pion case, the full source sum is intractable, so we estimate the correlator with point or wall source propagators.

For point source propagator
$$C_\rho(\vec{p}; t,0) \approx \frac{1}{3}\sum_i\sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{point}(\vec{x}_0,0)}^\dagger(\vec{x},t) (\gamma_5 \gamma_i) S_{l,\text{point}(\vec{x}_0,0)}(\vec{x},t) (\gamma_i \gamma_5) ]$$

For wall source propagator
$$C_\rho(\vec{p}; t,0) \approx \frac{1}{3}\sum_i\sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{wall}(-\vec{p}_2,0)}^\dagger(\vec{x},t) (\gamma_5 \gamma_i) S_{l,\text{wall}(\vec{p}_1,0)}(\vec{x},t) (\gamma_i \gamma_5) ]$$

**Step 5 — Current PyQCD distillation runtime**:

PyQCD does not expose an `Executor.generate_einsum` API, and the point/wall `prop_l`
formula above must not be passed to the distillation engine without an explicit layout
adapter.  The supported contraction path uses perambulators plus `VDV` vertices.  This
self-contained smoke example exercises the current public API and averages the three
spatial polarizations:

```python
import numpy as np

from pyqcd.contraction import (
    GammaRegistry,
    PeramRegistry,
    VRegistry,
    conjugate_operator,
    dynamic_contraction,
    seq_peram,
)
from pyqcd.lattice import gamma
from pyqcd.tools import set_backend

set_backend("numpy")
rng = np.random.default_rng(20260831)
nev = 2


def rand(shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


# One fixed (tsink,tsrc) pair; production code loads the matching perambulator.
peram = rand((4, 4, nev, nev))
perams = PeramRegistry()
perams.register("light", ("tsink", "tsrc"), peram)
perams.register("light", ("tsrc", "tsink"), seq_peram(peram))

# Each VDV has one momentum slot M.  Production data must retain its momentum metadata.
vertices = VRegistry()
vertices.register("VDV_0", "tsink", rand((1, nev, nev)))
vertices.register("VDV_0", "tsrc", rand((1, nev, nev)).conj())

gammas = GammaRegistry()
for i in (1, 2, 3):
    gammas.register(f"gamma_{i}", gamma(i))

components = []
for i in (1, 2, 3):
    sink = ["|", "d^d", f"gamma_{i}", "u", "|"]
    source = conjugate_operator(sink)
    assert source[0] == -1.0  # spatial-gamma Dirac conjugation sign

    contraction = dynamic_contraction(
        [(sink, source)],
        peram_registry=perams,
        v_registry=vertices,
        gamma_registry=gammas,
        Cpt="2pt",
        Vindex=["M", "M"],
        Oindex="M",
        ignore_dis=False,
        Projection=False,
        verbose=False,
    )
    component = np.asarray(contraction.calculate_all())
    assert len(contraction) == 1 and component.shape == (1,)
    components.append(component)

rho_2pt = np.mean(np.stack(components, axis=0), axis=0)
assert rho_2pt.shape == (1,) and np.isfinite(rho_2pt).all()
```

The random tensors only verify API wiring.  A physical result requires matched
perambulator/`VDV` provenance, source/sink times, momentum/Fourier conventions, ensemble
averaging, and the finite-time spectral analysis defined by `pyqcd-physics-spectrum`.
