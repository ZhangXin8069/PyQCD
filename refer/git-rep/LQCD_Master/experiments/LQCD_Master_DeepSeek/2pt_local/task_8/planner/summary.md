## Physics Objective

Compute the two-point correlation function of the J/ψ meson (vector charmonium, $c\bar{c}$, $J^{PC}=1^{--}$) on the C24P29 ensemble ($24^3\times 72$, $a\approx 0.105$ fm, $m_c = 0.4159$). The correlator uses a single charm-quark propagator from a point source at $[0,0,0,0]$, stout-smeared gauge links in the Dirac operator, and vector interpolating operators $\bar{c}\gamma_i c$ ($i=x,y,z$) averaged over the three polarizations.

## Contraction Structure (corrected)

For a vector meson in the DeGrand-Rossi basis, the correct contraction after Wick contraction, $\gamma_5$-hermiticity, and flavor simplification is:

$$C(t) = \frac{1}{3}\sum_{i=x,y,z} \mathrm{Tr}\big[S_c^\dagger(\vec{x},t;\vec{0},0)\,(\gamma_5\gamma_i)\,S_c(\vec{x},t;\vec{0},0)\,(\gamma_i\gamma_5)\big]$$

This matches the rho-meson reference derivation (`rho_mass.md`). The $\gamma_5$ factors originate from the conjugate operator $\mathcal{O}^\dagger = \bar{c}\gamma_4\gamma_i^\dagger\gamma_4 c$ and the application of $\gamma_5$-hermiticity to convert the backward propagator. The `generate_einsum` tool receives `gamma="gi"` and internally constructs the full $\gamma_5\gamma_i$ / $\gamma_i\gamma_5$ sandwich — the plan's `gamma_structures: [gamma_x, gamma_y, gamma_z]` is the correct input specification; the tool, not the plan text, owns the final einsum string.

## Strategy

- **Category**: standard meson spectroscopy (meson_2pt)
- **Diagram**: connected only — the disconnected $\bar{c}c\to\bar{c}c$ loop is OZI-suppressed for charmonium and omitted as standard practice
- **Momentum**: $\vec{p}=\vec{0}$ (rest frame, appropriate for mass extraction)
- **Source**: single point source at $[0,0,0,0]$ with no quark-field smearing
- **Gauge smearing**: stout smearing $(n_{\text{steps}}=1,\ \rho=0.125,\ n_{\text{dim}}=4)$ applied to gauge links before the Dirac inversion

## Propagator Requirements

One charm-quark propagator `prop_charm`. Both the quark and antiquark lines in the connected meson_2pt contraction reuse this same propagator:

`generate_einsum(type="meson_2pt", quark="c", antiquark="c", gamma="gi")`

The `propagator_quark` and `propagator_antiquark` fields in the correlator block explicitly wire `prop_charm` to both contraction legs, removing any ambiguity for the code generator.

## Solver & Convergence

The multigrid solver uses the ensemble's pre-tuned blocking `[[6,6,6,3],[4,4,4,6]]` on stout-smeared gauge links. Stout smearing substantially alters the gauge-field UV structure, so near-null-space vectors tuned on unsmeared configurations may be ineffective. **Mitigation**: after each inversion, the true residual is checked against the requested tolerance ($10^{-12}$). If multigrid fails to converge within 20000 iterations, a standard CG solver with maxiter 40000 is used as fallback. This prevents silently propagating unconverged propagators into the correlator.

## Measurement

For each spatial gamma matrix ($\gamma_1$, $\gamma_2$, $\gamma_3$), the meson_2pt contraction is evaluated at all 72 time slices via `generate_einsum`. The three polarization correlators are averaged pointwise to produce a single array $C(t)$ of length 72.

## Output

Plain text file `jpsi_2pt_result.txt` in the run directory containing 72 lines — one polarization-averaged correlator value per time slice, no header or metadata. Suitable for subsequent effective-mass or fitting analysis.