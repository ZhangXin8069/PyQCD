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

**Step 5 — Explicit einsum**:

PyQCD 没有 code-generation contraction API。下面的显式公式限定于
`prop_l.data` 的轴序 `(w,t,z,y,x,spin_sink,spin_src,color_sink,color_src)`；
`.conj()` 与标签 `jiba` 一起完成源端共轭/转置。生产代码必须先用真实 layout 做
小数组形状和零位移 oracle，再接入批量链路。

Point source:

```python
from opt_einsum import contract

twopt = contract(
    "wtzyx,wtzyxjiba,wtzyxjiba->t",
    phase, prop_l.data.conj(), prop_l.data,
)
```

Wall source 使用两条分别带 \(\vec p_1\) 与 \(-\vec p_2\) 源相位的传播子：

```python
twopt = contract(
    "wtzyx,wtzyxjiba,wtzyxjiba->t",
    phase, prop_l_np2.data.conj(), prop_l_p1.data,
)
```

`phase` 是 sink 动量投影，wall 的两个源相位和动量分裂必须按
[`source-estimators`](../references/source-estimators.md) 另行固定。
