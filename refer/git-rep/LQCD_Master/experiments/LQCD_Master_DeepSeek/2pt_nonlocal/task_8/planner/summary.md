# J/ψ Nonlocal Two-Point Function — Revised Plan

## Physics Objective

Compute the nonlocal two-point correlation function of the J/ψ vector meson ($c\bar{c}$) with spatial separation $z_{\text{len}} \in [0, 10]$ in the $+z$ direction.  This is the building block for quasi-distribution amplitudes (quasi-DAs).  The nonlocal operator shifts only the charm-quark leg, leaving the anti-charm leg unshifted.

## Key Revisions from Peer Review

### 1. Corrected contraction formula (critical)

The original plan used $\operatorname{Tr}[S_c^\dagger \gamma_i S_{\text{shift}} \gamma_i]$, which omits essential $\gamma_5$ factors.  Following the rho-meson reference derivation, the correct contraction after $\gamma_5$-hermiticity is:

$$C_i(z_{\text{len}}, t) = \sum_{\vec{x}} \operatorname{Tr}\big[ S_c^\dagger(\vec{x},t)\, (\gamma_5 \gamma_i)\, S_{\text{shift}}(\vec{x},t)\, (\gamma_i \gamma_5) \big]$$

The $\gamma_5$ factors arise from applying $S(0;x) = \gamma_5 S^\dagger(x;0) \gamma_5$ to the Wick-contracted expression.  After tracking the adjoint-operator minus sign and the fermion-loop minus sign (which cancel), the full contraction reduces to the structure above.  Omitting the $\gamma_5$ factors produces a numerically different correlator that does **not** match the standard local J/ψ correlator in the $z_{\text{len}}=0$ limit.

### 2. Use `generate_einsum` instead of hardcoding
The Executor **must** call `generate_einsum(type="meson_2pt", quark="c", antiquark="c", gamma="g{i}")` for each polarization $i \in \{1,2,3\}$ and substitute `S_shifted` for the quark-line propagator argument and the unshifted `S_c` for the antiquark-line argument.  This resolves the conflict between the hardcoded formula in the original plan and the `generate_einsum` toolchain which is the single source of truth for contraction structures.

### 3. Preserve original gauge links for Wilson line
Before stout smearing for the propagator inversion, the original unsmeared gauge configuration must be explicitly copied (e.g., `gauge_orig = gauge.copy()` or reload from disk).  The Wilson line $W(x, x+z\hat{z})$ is built from `gauge_orig`, **not** from the smeared links.  Without this mechanism, the Executor would silently use smeared links for the Wilson line.

### 4. Periodic boundary conditions for Wilson line
When $x_z + z_{\text{len}} \ge L_z$, the shifted point wraps around.  Gauge-link indices must be taken modulo $L_z$, and the boundary-crossing link $U_z(x_z = L_z-1)$ must be used.

### 5. $z_{\text{len}}=0$ consistency check
At zero separation $W = \mathbb{1}$ and the correlator must recover the standard local J/ψ vector correlator with the same sign convention as the rho meson (positive at large $t$).

## Technical Details (unchanged)
- **Ensemble**: C24P29, $24^3 \times 72$, $a \approx 0.1052$ fm, config 10000
- **Source**: Point source at $[0,0,0,0]$, single charm inversion
- **Solver**: CG with stout-smeared links (1 step, $\rho=0.125$, 4D), tol $10^{-12}$
- **Polarization**: Average $\gamma_x$, $\gamma_y$, $\gamma_z$
- **Output**: `nonlocal_2pt_jpsi.txt`, columns `z_len t Re[C] Im[C]`, no header