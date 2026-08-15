## Physics Objective

Compute the **nonlocal two-point correlation function** of the $\eta_s$ meson ($\bar{s}\gamma_5 s$, pseudoscalar $J^{PC}=0^{-+}$). The $s$-quark field in the interpolating operator is spatially shifted by $z$ lattice units along the $+z$-direction and connected by a straight Wilson line built from the original (unsmeared) gauge links. The maximum separation is $z_{\max}=10$. This is a building block for quasi-distribution amplitude (quasi-DA) studies of the $\eta_s$.

## Operator and Correlator

- **Nonlocal sink operator**: $\mathcal{O}_{\eta_s}(t; z) = \sum_{\vec{x}} \bar{s}(\vec{x},t)\, \gamma_5\, W(\vec{x}, \vec{x}+z\hat{z}; t)\, s(\vec{x}+z\hat{z}, t)$
- **Local source operator**: $\mathcal{O}^\dagger_{\eta_s}(0) = -\sum_{\vec{y}} \bar{s}(\vec{y},0)\, \gamma_5\, s(\vec{y},0)$ (the minus sign follows from $\gamma_4\gamma_5^\dagger\gamma_4 = -\gamma_5$ in the DeGrand-Rossi basis)
- **Connected correlator** (zero momentum, point source at $[0,0,0,0]$):
  $$C(z; t) = \sum_{\vec{x}} \operatorname{Tr}_{\text{spin}\otimes\text{color}}\!\Big[ S_s^\dagger(\vec{x},t; \vec{0},0)\; W(\vec{x}, \vec{x}+z\hat{z}; t)\; S_s(\vec{x}+z\hat{z}, t; \vec{0},0) \Big]$$
  For $z=0$, $W$ is the identity and this reduces to the standard local $\eta_s$ two-point function, providing a built-in consistency check.

## Key Implementation Details

1. **Propagator**: A single forward strange-quark propagator $S_s(\vec{x},t; \vec{0},0)$ from a point source at $[0,0,0,0]$. The Dirac operator (Wilson-clover, $c_{sw}=1.160920226$, $m_s=-0.2356$) is inverted on stout-smeared gauge links (`n_steps=1, rho=0.125, ndim=4`) using a two-level multigrid solver with tolerance $10^{-12}$.

2. **Wilson line**: Built as the ordered product of original (unsmeared) $U_z$ gauge links along the $+z$ direction at each time slice: $W(\vec{x}, \vec{x}+z\hat{z}; t) = \prod_{k=0}^{z-1} U_z(\vec{x} + k\hat{z}, t)$. For $z=0$, $W = \mathbb{1}_{3\times 3}$.

3. **Shifted propagator**: `prop_s_shifted(x,t;z) = W(x, x+z*e_z; t) * prop_s(x+z*e_z, t)`. Periodic boundary conditions handle spatial wrap-around on the $24^3$ volume.

4. **Only the connected diagram** is computed. Disconnected contributions (quark loops) are omitted, which is standard for an exploratory quasi-DA calculation.

## Output

11 plain-text files (`etas_nonlocal_2pt_z00.txt` through `etas_nonlocal_2pt_z10.txt`), one per $z$ value. Each file contains 72 lines (one per time slice $t = 0,\dots,71$) with three whitespace-separated columns: `t  Re[C(z,t)]  Im[C(z,t)]`. No header, no comments, no extra text. Saved in the run directory.

## Requirement Verification

| Requirement | How it is satisfied |
|---|---|
| Nonlocal shift on quark (s) propagator, not antiquark | The $s$ field is shifted by $z$; $\bar{s}$ stays at $\vec{x}$ |
| Original gauge field for Wilson line | Wilson line built from unsmeared links |
| Stout-smeared inversions (1, 0.125, 4) | Dirac operator uses stout-smeared gauge with `n_steps=1, rho=0.125, ndim=4` |
| Point source at [0,0,0,0] | Single point source at the origin |
| Maximum separation 10 | $z = 0,1,\dots,10$ inclusive |
| Plain-text output, no header | Each file: 72 lines, `t Re Im`, no header |