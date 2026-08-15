## Physics Objective

Compute the nonlocal two-point correlation function of a positively charged pion (π⁺) with a spatial nonlocal shift along the z-direction. This is the lattice realization of the pion quasi-distribution amplitude (quasi-DA), where the two quark fields in the interpolating operator are separated by a spacelike Wilson line. The nonlocal sink operator is

$$\mathcal{O}_{\pi^+}(\vec{x}, t; z) = \bar{d}(\vec{x}, t)\, \gamma_5\, W(\vec{x}, t; \vec{x}+z\hat{z}, t)\, u(\vec{x}+z\hat{z}, t)$$

where $W$ is a straight Wilson line along $+z$. The source operator at $t=0$ is the local pion creation operator $\mathcal{O}_{\pi^+}^\dagger = -\bar{u}\gamma_5 d$ at the origin. After Wick contraction with $\gamma_5$-hermiticity and $u$–$d$ flavor symmetry, the correlator reduces to

$$C(z; t) = \operatorname{Re} \sum_{\vec{x}} \operatorname{Tr}_{\mathrm{spin,color}}\left[ S_l^\dagger(\vec{x}, t; 0, 0)\; W(\vec{x}, \vec{x}'+z\hat{z})\; S_l(\vec{x}'+z\hat{z}, t; 0, 0) \right]$$

where $x'_z = (x_z + z) \bmod 24$ (periodic wrap). Only one light-quark point-source propagator is required.

## Peer Review Revisions

Three critical issues identified by peer review are resolved in this revision:

### 1. Periodic boundary-condition handling for the nonlocal shift

The spatial sum over $\vec{x}$ includes sites where $x_z + z \ge L_z = 24$. The revised plan explicitly specifies: (a) the shifted coordinate wraps as $(x_z + z) \bmod 24$, and (b) the Wilson line crosses the periodic boundary by including the gauge link $U_z^{\rm orig}$ at $z = 23$, which connects site 23 to site 0 under periodic boundary conditions. The Wilson line product $\prod_{k=0}^{z-1} U_z^{\rm orig}(x + k\hat{z})$ handles intermediate wrapping by taking all indices modulo 24.

### 2. Dual gauge-field requirement

The user requires the Wilson line to use **original (unsmeared)** gauge links while the Dirac inversion uses **stout-smeared** links. The revised plan mandates that the code load the gauge field once, preserve the original copy in memory, apply stout smearing to a separate copy for the CG inversion, and pass the original links to the Wilson line construction. This is enforced in the solver notes and measurement correlator notes.

### 3. Output data-type and format specification

The contracted trace $\operatorname{Tr}[S^\dagger W S]$ is complex-valued on a single gauge configuration. The revised plan specifies: (a) **only the real part is saved**, justified because the imaginary part is identically zero after the full spatial sum by hermiticity of the pion two-point correlator and carries only roundoff noise; (b) the output is a plain text file with exactly 11 lines (one per $z = 0,\ldots,10$), each containing 72 space-separated floating-point values for $t = 0,\ldots,71$; (c) no header, metadata, or imaginary-part columns are written.

### 4. Spin-trace clarification

The trace $\operatorname{Tr}_{\mathrm{spin,color}}$ is explicitly declared to run over both Dirac ($4\times4$) and color ($3\times3$) indices, implemented via an explicit einsum contraction. The $\gamma_5$ insertions at source and sink have been fully absorbed by the Wick contraction and $\gamma_5$-hermiticity, leaving a trivial $S^\dagger W S$ structure with no residual gamma matrices between the propagators.

## Technical Details (unchanged)

- **Ensemble**: C24P29 — $24^3 \times 72$, $a_s = 0.1052$ fm, clover Wilson fermions, $m_l = -0.277$, $c_{sw} = 1.160920226$.
- **Source**: Point source at $[0,0,0,0]$, zero momentum.
- **Inversion**: CG + multigrid on stout-smeared links $(n_{\mathrm{steps}}=1, \rho=0.125, n_{\mathrm{dim}}=4)$, tolerance $10^{-12}$.
- **z-range**: $0 \le z \le 10$ (11 separations). At $z=0$, $W = \mathbb{1}$ and the standard local pion 2pt is recovered.
- **Single configuration**: cfg 10000 (debugging/validation run).

## Caveat

The operator uses pure $\gamma_5$ without a $\gamma_z$ structure, which is not the standard quasi-DA operator (canonical form uses $\gamma_z\gamma_5$). The resulting correlator may exhibit stronger power-divergent mixing at large $z$. This is a physics-choice risk noted but not corrected, as the user explicitly specified this operator.