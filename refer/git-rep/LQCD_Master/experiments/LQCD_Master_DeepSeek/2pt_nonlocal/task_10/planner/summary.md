## Physics objective

Compute the nonlocal two-point correlation function of the vector $D_s^*$ meson ($\bar{s} c$, $J^P = 1^-$) with a spatial Wilson-line shift in the $+z$-direction, $n_z = 0, 1, \ldots, 10$.  The nonlocal shift is applied only to the charm quark propagator at the sink; the strange antiquark propagator remains unshifted.  The Wilson line is built from the **original unsmeared** gauge links, while both Dirac inversions use **stout-smeared** links ($\rho = 0.125$, one step, four-dimensional smearing).

## Key corrections relative to the previous plan

1. **Correlator gamma structure** — The previous plan omitted the $\gamma_5$ factors that arise from $\gamma_5$-hermiticity of the backward strange propagator.  The correct trace (derived step-by-step in the DeGrand–Rossi basis) is
   $$C(n_z, t) = -\frac{1}{3} \sum_{i\in\{x,y,z\}} \sum_{\mathbf{x}} \operatorname{Tr}\!\big[ S_s^\dagger(\mathbf{x},t;\mathbf{0}) \,(\gamma_5 \gamma_i)\, W(\mathbf{x},\mathbf{x}+n_z\hat{z};t)\, S_c(\mathbf{x}+n_z\hat{z},t;\mathbf{0}) \,(\gamma_i \gamma_5) \big].$$
   The $\gamma_5$ factors do **not** commute with $\gamma_i$ and must be carried explicitly.  For $n_z = 0$ ($W = 1$) the expression reduces to the standard local $D_s^*$ two-point function with the correct spin structure.

2. **generate_einsum bypassed** — The nonlocal Wilson-line shift is a custom post-inversion construction that falls outside the scope of `generate_einsum`.  The revised plan specifies an explicit five-step measurement procedure: (a) load original gauge links, (b) build the on-axis Wilson line, (c) shift the charm propagator, (d) contract with the strange propagator and the correct gamma matrices, (e) average over polarisations.

3. **Two gauge-field copies** — The plan now explicitly requires two copies of the gauge field: the original unsmeared links for Wilson-line construction and the stout-smeared links for Dirac inversions.  Both must be available concurrently or reloaded.

4. **Source operator clarified** — The source is the creation operator $\mathcal{O}_{\text{src}}(0) = \bar{c}(0) \gamma_i s(0)$; the two-point correlator is $\langle \mathcal{O}_{\text{sink}}(t) \mathcal{O}_{\text{src}}(0) \rangle$ with **no additional Dirac adjoint**.  This avoids the sign ambiguity present in the previous plan.

5. **Multigrid coarse-grid note** — The second-level block sizes $[4,4,4,6]$ on the $4\times4\times4\times24$ first-level coarse grid produce a $1\times1\times1\times4$ coarsest grid ($1\times1\times1\times1$ per MPI rank).  Most multigrid backends handle this via a direct solve, but a fallback suggestion ($[2,2,2,4]$ blocks → $2\times2\times2\times6$ coarsest grid) is recorded.

## Strategy

- **Hadron operators**:  
  - Sink (nonlocal): $\mathcal{O}_{\text{sink},i}(\mathbf{x},t; \hat{z}, n_z) = \bar{s}(\mathbf{x},t)\, \gamma_i\, W(\mathbf{x}, \mathbf{x}+n_z\hat{z}; t)\, c(\mathbf{x}+n_z\hat{z}, t)$  
  - Source (local): $\mathcal{O}_{\text{src},i}(\mathbf{0},0) = \bar{c}(\mathbf{0},0)\, \gamma_i\, s(\mathbf{0},0)$

- **Propagators**: single point source at $[0,0,0,0]$, zero momentum, stout-smeared inversions.  Post-inversion, the charm propagator is shifted for each $n_z$ using the original gauge links.

- **Output**: plain `.txt` file, no header, columns `n_z  t  Re[C]  Im[C]` for $n_z \in [0,10]$, $t \in [0,71]$.

## Fidelity to original requirements

| Requirement | How it is met |
|---|---|
| Nonlocal $D_s^*$ two-point function | Nonlocal sink operator with $z$-direction Wilson line |
| Max separation 10 in $z$ | $n_z$ loop from 0 to 10 |
| Point source at $[0,0,0,0]$ | Both propagators use a single point source at the origin |
| Shift on quark (charm) only | Only $S_c$ receives the Wilson-line shift; $S_s$ is unchanged |
| Original gauge field for shift | Wilson line $W$ built from unsmeared gauge links |
| Inversions with stout-smeared links | `stout_link_smear: enabled, rho=0.125, n_steps=1, ndim=4` |
| Vector operators $\gamma_x,\gamma_y,\gamma_z$ averaged | Three polarisations explicitly averaged |
| Plain text output, no header | `output.format: txt`, `output.metadata: false` |
| Correct gamma structure | Full DeGrand–Rossi derivation with $\gamma_5$ factors retained |