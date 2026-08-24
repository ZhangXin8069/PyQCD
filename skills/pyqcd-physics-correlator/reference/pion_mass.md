## Example: Pion meson mass (π⁺ channel)

> **Notation note**: In the code, the light quark propagator is called `prop_l` and the strange propagator `prop_s`. The LaTeX $S_l$ and $S_s$ in the formulas below correspond to `prop_l` and `prop_s` respectively.

**Goal**: Extract $m_\pi$. We need to calculate the two-point correlation function of $\pi^+$.

**Step 1 — Operator**:
  $$\mathcal{O}_{\pi^+} = \bar{d} \gamma_5 u$$
We have the corresponding Dirac conjugate operator (creation operator)
  $$\mathcal{O}_{\pi^+}^\dagger = \bar{u} \gamma_4\gamma_5^\dagger\gamma_4 d$$

Using the property of $$\gamma$$ matrix: $\gamma_5^\dagger=\gamma_5$, $\gamma_5\gamma_4=-\gamma_4\gamma_5$, $\gamma_4\gamma_4=1$, we obtain:
  $$\mathcal{O}_{\pi^+}^\dagger = -\bar{u} \gamma_5 d$$

**Step 2 — Correlator**:
  $$C_\pi(\vec{p}; t,0) = \langle \mathcal{O}_{\pi^+}(\vec{p},t) \mathcal{O}^\dagger_{\pi^+}(\vec{p},0) \rangle$$

**Step 3a - Quark fields**: Expand the operator in terms of quark fields and Fourier transform:
  $$C_\pi(\vec{p}; t,0) = \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \langle \bar{d}(\vec{x},t) \gamma_5 u(\vec{x},t) \bar{u}(\vec{y},0) \gamma_4 \gamma_5^\dagger \gamma_4 d(\vec{y},0) \rangle$$

Using the property of $$\gamma$$ matrix, we obtain:
  $$C_\pi(\vec{p}; t,0) = -\sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \langle \bar{d}(\vec{x},t) \gamma_5 u(\vec{x},t) \bar{u}(\vec{y},0) \gamma_5  d(\vec{y},0) \rangle$$

**Step 3b — Wick contraction**:
One connected diagram (no disconnected pieces for charged pion, and the negative sign arises from the anticommutation of fermion fields):

  $$C_\pi(\vec{p}; t,0) = -\sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_d(\vec{y},0; \vec{x},t) \gamma_5 S_u(\vec{x},t; \vec{y},0) \gamma_4 \gamma_5^\dagger \gamma_4 ]$$

Using the property of $$\gamma$$ matrix, we obtain:
  $$C_\pi(\vec{p}; t,0) = \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_d(\vec{y},0; \vec{x},t) \gamma_5 S_u(\vec{x},t; \vec{y},0)\gamma_5  ]$$


**Step 3c - Simplification**:
  1. Apply the $\gamma_5$-hermiticity and the flavor symmetry:
  $$C_\pi(\vec{p}; t,0) = \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ \gamma_5 S_l^\dagger(\vec{x},t; \vec{y},0) \gamma_5 \gamma_5 S_l(\vec{x},t; \vec{y},0) \gamma_5 ]$$
  2. Apply the cyclic property to simplify the gamma matrix structure:
  $$C_\pi(\vec{p}; t,0) = \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_l^\dagger(\vec{x},t; \vec{y},0) S_l(\vec{x},t; \vec{y},0) ]$$

**Step 4 — Propagators needed**:
It is impossible to calculate propagator from all source points, and we can use point or wall source propagators to estimate the correlator.

For point source propagator
$$C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{point}(\vec{x}_0,0)}^\dagger(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)}(\vec{x},t) ]$$

For wall source propagator
$$C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{wall}(-\vec{p}_2,0)}^\dagger(\vec{x},t) S_{l,\text{wall}(\vec{p}_1,0)}(\vec{x},t) ]$$

**Step 5 — Einsum** (⚠️ call generate_einsum, do NOT write contract by hand):

Call `generate_einsum(type="meson_2pt", quark="u", antiquark="d", gamma="g5")`
from the Executor. The tool returns a complete code block ("code" key).

For wall source, just replace the propagator variable name in the returned code.
