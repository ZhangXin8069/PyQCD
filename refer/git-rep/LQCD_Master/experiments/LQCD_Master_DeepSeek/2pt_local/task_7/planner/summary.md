## Physics Objective

Compute the zero-momentum two-point correlation function of the $D_s^+$ meson on a single gauge configuration (cfg #10000) from the C24P29 ensemble ($24^3\times 72$, $a\approx 0.1052$ fm).

## Hadron and Operator

The $D_s^+$ is a flavour non-singlet pseudoscalar meson with quark content $c\bar{s}$. The interpolating operator is

$$\mathcal{O}_{D_s^+}(x) = \bar{s}(x)\,\gamma_5\,c(x)$$

with creation operator $\mathcal{O}^\dagger = -\bar{c}\,\gamma_5\,s$.

## Correlator and Wick Contraction

At zero momentum the two-point function is

$$C(t) = \sum_{\vec{x}} \big\langle \mathcal{O}_{D_s^+}(\vec{x},t)\,\mathcal{O}_{D_s^+}^\dagger(\vec{0},0) \big\rangle.$$

Wick contraction gives

$$C(t) = \sum_{\vec{x}} \operatorname{Tr}\big[ S_s(0;x)\,\gamma_5\,S_c(x;0)\,\gamma_5 \big],$$

where $S_c$ is the charm quark propagator and $S_s$ is the strange quark propagator (representing the anti-strange antiquark).  Applying $\gamma_5$-hermiticity $S_s(0;x) = \gamma_5 S_s^\dagger(x;0)\gamma_5$ and cyclicity of the trace eliminates the gamma matrices:

$$C(t) = \sum_{\vec{x}} \operatorname{Tr}\big[ S_s^\dagger(x;0)\,S_c(x;0) \big] = \sum_{\vec{x}} \operatorname{Tr}\big[ \texttt{prop\_s\_dag} \cdot \texttt{prop\_c} \big].$$

**Critical correction from peer review:** The dagger belongs on the strange (antiquark) propagator (`prop_s`), not the charm quark propagator (`prop_c`).  The previous plan had the dagger on the wrong propagator (`Tr[prop_c_dag @ prop_s]`), which would compute the complex conjugate of the correct trace and produce wrong numerical values.  The corrected contraction is `Tr[prop_s_dag @ prop_c]`.  This is obtained by calling `generate_einsum(type="meson_2pt", quark="c", antiquark="s", gamma="g5")`, which correctly places the Hermitian conjugate on the antiquark propagator.

## Technical Strategy

- **Source**: single point source at spacetime position $[0,0,0,0]$.
- **Gauge links**: stout-smeared with $n_{\text{stout}}=1$, $\rho=0.125$, smeared in all 4 dimensions before inversion.
- **Propagators**: two independent forward solves — strange quark ($\kappa_s = -0.2356$, role: antiquark) and charm quark ($\kappa_c = 0.4159$, role: quark) — using the clover-improved Wilson Dirac operator with $c_{\text{SW}} = 1.160920226$.
- **Solver**: conjugate gradient to relative residual $10^{-12}$, maximum 2000 iterations.
- **Output**: 72 real numbers (correlator values for $t=0,\dots,71$) written as a plain-text file with no header, one value per line.

## Reasonable Completions

- Zero momentum is assumed (standard for spectroscopy ground-state mass extraction).
- CG solver tolerance $10^{-12}$ and max-iter 2000 are conservative defaults for clover fermions.
- No disconnected diagrams are needed because $D_s^+$ is flavour non-singlet.
- Only one configuration is processed (the user specified `cfg_num: ['10000']`).
- The output file is placed in the current run directory.
- Propagator roles (`quark`/`antiquark`) are now explicitly annotated in both the propagator definitions (`role: quark` / `role: antiquark`) and the correlator block (`quark: prop_c, antiquark: prop_s`) to prevent ambiguity in downstream code generation.