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

**Step 5 — Einsum** (⚠️ call generate_einsum, do NOT write contract by hand):

Call `generate_einsum(type="meson_2pt", quark="u", antiquark="d", gamma="g1")`
from the Executor. The tool returns a complete code block ("code" key).

For rho, average over gamma1, gamma2, gamma3:
```python
base_code = ...  # from generate_einsum(type="meson_2pt", ..., gamma="g1")
# Repeat for g2, g3 and average
twopt = 0
gamma_g = gamma.gamma
for idx in [1, 2, 3]:
    # Replace gamma_g(1) with gamma_g(idx) in the code
    code = spec['args'].copy()
    # (the args already have the gamma expression; replace g1 -> g{idx})
    twopt += eval(code)
twopt /= 3
```

For wall source propagator, replace `prop_l` with the wall propagator variable in the code.
