# D⁺ Meson Two-Point Correlator

## Physics Objective

Extract the ground-state mass of the D⁺ meson (pseudoscalar, $J^P = 0^-$) via its two-point Euclidean correlation function. The D⁺ has valence quark content $\bar{d}c$ (anti-down plus charm).

## Strategy

### Operator and Correlator

The standard local interpolating operator is $\mathcal{O}_{D^+} = \bar{d} \gamma_5 c$. The zero-momentum two-point function is:

$$C_{D^+}(t) = \langle \mathcal{O}_{D^+}(t) \, \mathcal{O}_{D^+}^\dagger(0) \rangle$$

where $\mathcal{O}_{D^+}^\dagger = -\bar{c} \gamma_5 d$. After Wick contraction, a single connected diagram (no disconnected contributions for this flavor-off-diagonal meson) yields:

$$C_{D^+}(t) \propto \sum_{\vec{x}} \text{Tr}\left[ S_c^\dagger(\vec{x}, t; \vec{0}, 0) \, S_l(\vec{x}, t; \vec{0}, 0) \right]$$

up to gamma-matrix factors that the `generate_einsum` tool handles automatically. The contraction call is `generate_einsum(type="meson_2pt", quark="c", antiquark="d", gamma="g5")`.

### Propagator Requirements

- **One light-quark propagator** (`prop_l`, mass = −0.277): from point source at $(0,0,0,0)$ to all lattice sites, using multigrid solver.
- **One charm-quark propagator** (`prop_c`, mass = 0.4159): from the same point source, using standard CG solver (heavy quark converges rapidly).

Both propagators are computed on gauge configurations that have undergone **one step of stout smearing** with $\rho = 0.125$ in all 4 dimensions, as specified by the user.

### Source Details

- **Type**: point source at spatial origin $(0,0,0)$ and time slice $t = 0$.
- **Momentum**: zero, so no momentum phase factor is injected at the source.
- **Sink**: point sink (no smearing), matched to the point source.

## Technical Details

| Parameter | Value |
|---|---|
| Lattice | $24^3 \times 72$, $a \approx 0.1052$ fm |
| Clover coefficient | 1.160920226 |
| Stout smearing | $n_\text{step}=1$, $\rho=0.125$, $n_\text{dim}=4$ |
| Light quark mass | −0.277 |
| Charm quark mass | 0.4159 |
| Light solver | Multigrid, tol $10^{-8}$, maxiter 20000 |
| Charm solver | CG, tol $10^{-8}$, maxiter 5000 |
| Configurations | cfg 10000 (single configuration) |
| MPI layout | $1 \times 1 \times 1 \times 4$ |

## Requirement Satisfaction

1. **Two-point function of D⁺**: satisfied via the meson_2pt correlator with the $\bar{d}\gamma_5 c$ operator.
2. **Point source at [0,0,0,0]**: both propagators use a point source at the origin.
3. **Stout-smeared links (1, 0.125, 4)**: applied to gauge links before both inversions.
4. **Plain text output**: correlator data saved as raw values to a `.txt` file with no metadata or headers.

## Reasonable Completions

- **Solver tolerances and maxiter**: chosen as conservative standard values ($10^{-8}$ tolerance, 20000/5000 maxiter for light/charm).
- **Zero momentum**: assumed since the user did not request momentum projection.
- **Single configuration**: the ensemble lists cfg 10000; a single-configuration run is standard for initial checks.
- **Multigrid for light quark**: using the two-level blocking parameters `[[6,6,6,3], [4,4,4,6]]` from the ensemble, as heavy multigrid is essential for light-quark inversions on this lattice volume.