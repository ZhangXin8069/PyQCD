## Physics objective

Compute the two-point correlation function of the **Λₛ baryon** (flavor `udc`, spin-1/2, positive parity) in the rest frame to extract its ground-state mass. The correlator is projected onto the positive-parity channel with \(P^+ = (1+\gamma_4)/2\).

## Interpolating operator

The Λₛ uses a scalar ud diquark with a charm spectator:

\[
\mathcal{O}_{\Lambda_c}(\vec{x},t) = \epsilon^{abc}\, \big(u^{Ta}(\vec{x},t)\, C\gamma_5\, d^{b}(\vec{x},t)\big)\, c^{c}(\vec{x},t)
\]

The diquark uses the `Cg5 = C @ γ₅` combination as specified. The conjugate operator is:

\[
\bar{\mathcal{O}}_{\Lambda_c}(0) = \epsilon^{a'b'c'}\, \bar{c}^{c'}(0)\,\big(\bar{d}^{b'}(0)\,\gamma_4\gamma_5^\dagger\gamma_4\,C\,\bar{u}^{Ta'}(0)\big)
\]

## Wick contraction — single topology

**Critical correction from review**: The Λₛ has three **distinct** quark flavors (u, d, c). Unlike the proton (`uud`, two identical light quarks → two contraction topologies), the Λₛ admits **exactly one** Wick contraction:

| Contraction | Sink | Source | Propagator |
|-------------|------|--------|------------|
| u ↔ ū | \(u^a_\alpha(x)\) | \(\bar{u}^{a'}_{\alpha'}(0)\) | \(S_l\) |
| d ↔ d̄ | \(d^b_\beta(x)\) | \(\bar{d}^{b'}_{\beta'}(0)\) | \(S_l\) |
| c ↔ c̄ | \(c^c_\gamma(x)\) | \(\bar{c}^{c'}_{\gamma'}(0)\) | \(S_c\) |

There is **no exchange topology**: the charm quark at the sink cannot contract with a light antiquark at the source — that would be flavor-violating.

## Correlator expression

With a point source at the origin and zero sink momentum:

\[
C_{\Lambda_c}(t) = \sum_{\vec{x}} \epsilon^{abc}\epsilon^{a'b'c'}\, (C\gamma_5)_{\alpha\beta}\, (C\gamma_5)_{\alpha'\beta'}\, P^+_{\gamma'\gamma}
\, S_{l,\alpha\alpha'}^{aa'}(x,0)\, S_{l,\beta\beta'}^{bb'}(x,0)\, S_{c,\gamma\gamma'}^{cc'}(x,0)
\]

where \(x = (\vec{x}, t)\) and \(P^+ = (1+\gamma_4)/2\). The source gamma structure \((\gamma_4\gamma_5^\dagger\gamma_4 C)^T\) simplifies to \(C\gamma_5\), matching the sink diquark projector exactly.

## Propagator requirements

| Propagator | Quark | Mass | Source | Solver |
|------------|-------|------|--------|--------|
| `prop_l` | light (u, d) | −0.277 | point at (0,0,0,0) | CGNR + multigrid |
| `prop_c` | charm (c) | 0.4159 | point at (0,0,0,0) | CGNR (MG optional) |

Isospin symmetry (\(S_u = S_d = S_l\)) means `prop_l` serves both the u and d lines. Only one light and one charm inversion are needed.

## Numerical strategy

- **Solver**: CGNR (CG on the normal equations \(M^\dagger M\)) is required because the Wilson-clover Dirac operator \(M\) is not Hermitian. Most LQCD libraries apply CG to \(M^\dagger M\) internally; the plan assumes this convention. If the executor interprets "CG" literally as CG on \(M\) directly, the solver must be switched to CGNR.
- **Light quark**: CGNR with two-level multigrid preconditioning ([6,6,6,3] → [4,4,4,6]), tolerance \(10^{-12}\), max 20000 iterations.
- **Charm quark**: same CGNR setup, but the multigrid parameters are optimized for the light-quark near-null space. For the heavy charm (mass 0.4159), multigrid may provide little speedup or even stall. If convergence is problematic, fall back to plain CGNR without multigrid.
- **Gauge smearing**: 1 step of stout smearing (\(\rho=0.125\), 4-dim) applied before both inversions.
- **Statistics**: single point source per configuration on cfg 10000.
- **Output**: bare correlator \(C(t)\) for \(t = 0,\dots,T-1\) written as whitespace-separated floats with no headers.

## Key correction from peer review

The original plan inherited the proton (`uud`) template with two contraction topologies. The exchange topology — where the charm quark contracts with a light antiquark — is **flavor-violating** and physically wrong for Λₛ (`udc`). The corrected correlator has only the direct term \(S_l \otimes S_l \otimes S_c\). The `generate_einsum` tool must be called with `flavor='udc'` to produce the single-term contraction.