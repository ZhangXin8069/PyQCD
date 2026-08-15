## Physics Objective

Compute the three-point correlation function for the semileptonic decay $D^0 \to \pi^-$ via the flavour-changing vector current $\bar{d}\gamma_1 c$ (a $c \to d$ transition), using the sequential-source method on the C24P29 ensemble ($24^3 \times 72$, $a \approx 0.1052$ fm, clover fermions).

## Physical Setup

| Component | Details |
|-----------|--------|
| **Source hadron** | $D^0$ meson, flavour $\bar{u}c$, interpolator $\gamma_5$, at $t=0$ |
| **Sink hadron** | $\pi^-$ meson, flavour $\bar{u}d$, interpolator $\gamma_5$, at $t=8$ |
| **Transition current** | $J_\mu = \bar{d}\,\gamma_1\,c$ (vector, $x$-direction) |
| **Momenta** | All zero: $\vec{p}_i = \vec{p}_f = \vec{q} = \vec{0}$ |
| **Source type** | Point source at $[0,0,0,0]$ |

## Strategy — Sequential Source Method

1. **Forward light propagator** (`prop_l_fwd`): $S_l(x;0,0,0,0)$, the $u$-quark line from source to sink.
2. **Forward charm propagator** (`prop_c_fwd`): $S_c(x;0,0,0,0)$, the $c$-quark line from source to the current.
3. **Sink block construction** at $t_{\text{seq}}=8$: $B(x) = S_l(x,t_{\text{seq}};0)\,\gamma_5$.
4. **Sequential source**: $\eta_{\text{seq}}(x) = \gamma_5\,B^\dagger(x)\,\gamma_5$ (two-dagger convention).
5. **Sequential propagator** (`prop_l_seq`): solve $D_l\,G_{\text{seq}} = \eta_{\text{seq}}$ for the $d$-quark line from sink to current.
6. **Contraction**: $C_3(\tau) = \sum_{\vec{z}} \operatorname{Tr}\big[G_{\text{seq}}(\vec{z},\tau)\;\gamma_1\;S_c(\vec{z},\tau;0,0,0,0)\big]$ for $\tau = 0,\dots,8$.

## Gauge Smearing

All three propagator inversions use stout-smeared gauge links with parameters $\rho = 0.125$, $n_{\text{steps}} = 1$, $n_{\text{dim}} = 4$.

## Solver Configuration

Multigrid solver for clover-improved Wilson fermions with $c_{\text{SW}} = 1.160920226$:
- Light quark mass: $-0.277$, tolerance $10^{-10}$, max iterations $5000$
- Charm quark mass: $0.4159$, tolerance $10^{-10}$, max iterations $5000$
- Multigrid blocking: level 0 $[6,6,6,3]$, level 1 $[4,4,4,6]$

## Output

Single plain-text file `d0_to_pi_3pt_result.txt` with 9 complex values (one per $\tau$), each line formatted as `real imag`, no headers or metadata. Only the three-point function is computed; no two-point functions are evaluated.

## Reasonable Completions

- **Solver parameters**: tolerance $10^{-10}$ and maxiter $5000$ are conservative defaults for multigrid clover inversions; these can be tightened if the correlator shows residual noise.
- **Tau range**: full range $\tau = 0,\dots,t_{\text{seq}}$ is computed; the executor may skip $\tau=0$ and $\tau=t_{\text{seq}}$ if contact terms are problematic.
- **Single configuration**: only cfg 10000 is specified; this is treated as production (no dry-run or debug flags).
- **Einsum generation**: the plan instructs the executor to use `generate_einsum(type="meson_3pt", ...)` for the contraction code rather than hand-writing einsum strings.