## Example: Proton baryon mass (positive parity channel)

> **Notation note**: In the code, the light quark propagator is called `prop_l` and the strange propagator `prop_s`. The LaTeX $S_l$ and $S_s$ in the formulas below correspond to `prop_l` and `prop_s` respectively.


**Goal**: Extract $m_p$. We need to calculate the two-point correlation function of $p$.

**Step 1 — Operator**:
  $$\mathcal{O}_{p} = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) u^c$$
We have the corresponding Dirac conjugate operator (creation operator)
  $$\bar{\mathcal{O}}_{p} = \mathcal{O}_{p}^\dagger\gamma_4 = \epsilon^{abc} \bar{u}^c(\bar{d}^{b} \gamma_4\gamma_5^\dagger\gamma_4 C \bar{u}^{Ta})$$

**Step 2 — Correlator**: A positive-parity projector is needed to isolate the ground-state nucleon:
  $$C_p(\vec{p}; t,0) = \mathrm{Tr}[P^+ \langle \mathcal{O}_{p}(\vec{p},t) \bar{\mathcal{O}}_{p}(\vec{p},0) \rangle ],\quad P^+ = \frac{1 + \gamma_4}{2}$$

**Step 3a — Quark fields**: Expand in quark fields and Fourier transform, writing out all spin indices explicitly:
  $$C_p(\vec{p}; t,0) = P^+_{\gamma'\gamma} \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} u^a_\alpha(\vec{x},t) (C\gamma_5)_{\alpha\beta} d^b_\beta(\vec{x},t) u^c_\gamma (\vec{x},t) \epsilon^{a'b'c'} \bar{u}^{c'}_{\gamma'}(\vec{y},0) \bar{d}^{b'}_{\beta'}(\vec{y},0) (\gamma_4 \gamma_5^\dagger\gamma_4 C)_{\beta'\alpha'} \bar{u}^{a'}_{\alpha'}(\vec{y},0)$$

**Step 3b — Wick contraction**:
Two contraction paths contribute, corresponding to the two ways of pairing the sink $u$ quarks with the source $\bar{u}$ quarks:

  $$C_p(\vec{p}; t,0) = P^+_{\gamma'\gamma} \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} \\
  [ S_{u\alpha\alpha'}^{aa'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\gamma'}^{cc'} (\vec{x},t;\vec{y},0) (\gamma_4 \gamma_5^\dagger \gamma_4 C)_{\beta'\alpha'} \\
  -S_{u\alpha\gamma'}^{ac'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\alpha'}^{ca'} (\vec{x},t;\vec{y},0) (\gamma_4 \gamma_5^\dagger \gamma_4 C)_{\beta'\alpha'} ]$$

**Step 3c — Simplification**:
  1. Relabel dummy color indices ($a' \leftrightarrow c'$) on the second term, the epsilon antisymmetry cancels the minus sign:
  $$C_p(\vec{p}; t,0) = P^+_{\gamma'\gamma} \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} \\
  [ S_{u\alpha\alpha'}^{aa'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\gamma'}^{cc'} (\vec{x},t;\vec{y},0) (\gamma_4 \gamma_5^\dagger \gamma_4 C)_{\beta'\alpha'} \\
  + S_{u\alpha\gamma'}^{aa'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\alpha'}^{cc'} (\vec{x},t;\vec{y},0) (\gamma_4 \gamma_5^\dagger \gamma_4 C)_{\beta'\alpha'} ]$$
  2. Apply the flavor symmetry ($\text{prop}_u = S_d = S_l$) and transpose the structure $(\gamma_4\gamma_5^\dagger\gamma_4C)^T=-C\gamma_4\gamma_5\gamma_4=C\gamma_5$, finally simplify the gamma structure:
  $$C_p(\vec{p}; t,0) = \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (C\gamma_5)_{\alpha'\beta'} P^+_{\gamma'\gamma} \\
  [ S_{l\alpha\alpha'}^{aa'}(\vec{x},t;\vec{y},0) S_{l\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{l\gamma\gamma'}^{cc'} (\vec{x},t;\vec{y},0) \\
  + S_{l\alpha\gamma'}^{aa'}(\vec{x},t;\vec{y},0) S_{l\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{l\gamma\alpha'}^{cc'} (\vec{x},t;\vec{y},0) ]$$

**Step 4 — Propagators needed**:
Baryons do not naturally admit wall sources, because a single wall cannot be shared among three quark lines with a well-defined momentum partition. In practice we use point (or smeared-point) source propagators, where all three quark lines originate from the same source point $\vec{x}_0$ at $t = 0$. Only a single light-quark propagator is required — the three quark lines reuse it:

$$C_p(\vec{p}; t,0) \approx -\sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (C\gamma_5)_{\alpha'\beta'} P^+_{\gamma'\gamma} \\
  [ S_{l,\text{point}(\vec{x}_0,0)\,\alpha\alpha'}^{aa'}(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)\,\beta\beta'}^{bb'}(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)\,\gamma\gamma'}^{cc'}(\vec{x},t) \\
  + S_{l,\text{point}(\vec{x}_0,0)\,\alpha\gamma'}^{aa'}(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)\,\beta\beta'}^{bb'}(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)\,\gamma\alpha'}^{cc'}(\vec{x},t) ]$$

**Step 5 — Einsum** (⚠️ call generate_einsum, do NOT write contract by hand):
Call `generate_einsum(type="baryon_2pt")` from the Executor.
The tool returns a complete code block ("code" key). Do NOT write these contractions by hand.
