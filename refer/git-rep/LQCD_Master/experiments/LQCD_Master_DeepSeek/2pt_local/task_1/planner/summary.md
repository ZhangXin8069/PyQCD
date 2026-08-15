## Physics Objective

Compute the zero-momentum pion two-point correlation function

$$C(t) = \sum_{\vec{x}} \langle \mathcal{O}_\pi(\vec{x}, t) \mathcal{O}_\pi^\dagger(\vec{0}, 0) \rangle$$

using the interpolating operator $\mathcal{O}_\pi = \bar{u} \gamma_5 d$ on a single gauge configuration (cfg 10000) from the C24P29 ensemble ($24^3\times 72$, $a\approx 0.1052$ fm, $m_\pi \approx 290$ MeV).

## Strategy

1. **Source** — Point source at the spacetime origin $[0,0,0,0]$, with all 12 spin×color components (spin 0–3, color 0–2) set to unity. This is the standard point-source convention in QUDA/PyQUDA and yields the usual normalisation for the meson two-point function. The trace $\operatorname{Tr}[S^\dagger S]$ at each sink time slice implicitly sums over the 12 source components, so no additional averaging is required.

2. **Gauge field** — Stout-smeared links ($n_{\text{steps}}=1$, $\rho=0.125$, 4-d smearing) suppress UV fluctuations and improve the chiral properties of the Wilson-clover Dirac operator.

3. **Propagator inversion** — One light-quark propagator (`prop_l`) is computed with a CG solver preconditioned by a two-level multigrid. The multigrid parameters are now fully specified: block sizes $[6,6,6,3]$ (fine) and $[4,4,4,6]$ (coarse), MR smoother with 4 pre- and 4 post-smoothing steps, coarse-grid CG with tolerance $10^{-1}$ and at most 200 iterations, and 24 near-null-space vectors per level. The outer CG uses $c_{\text{SW}} = 1.160920226$, bare quark mass $m_l = -0.277$, tolerance $10^{-12}$, and at most $10^4$ iterations.

4. **Wick contraction** — Isospin symmetry ($m_u = m_d = m_l$) lets both quark lines reuse the same `prop_l`. After applying $\gamma_5$-hermiticity and the cyclic property of the trace the $\gamma_5$ matrices cancel, giving

$$C(t) = \sum_{\vec{x}} \operatorname{Tr}\left[ S_l^\dagger(\vec{x}, t; \vec{0}, 0) \, S_l(\vec{x}, t; \vec{0}, 0) \right].$$

5. **Output** — The correlator $C(t)$ for $t = 0, \ldots, 71$ is written as a plain text file (`pion_2pt_result.txt`), one value per line, no headers.

## Key corrections from the previous version

- **Point source convention**: The plan now explicitly states that all 12 spin×color components are excited at the source point. This removes the ambiguity flagged in the risk assessment and ensures the executor uses the intended normalisation.
- **Multigrid solver details**: The smoother type (MR), pre/post-smoothing iteration counts (4/4), coarse-solver algorithm (CG), coarse tolerance ($10^{-1}$), coarse max-iterations (200), and number of near-null-space vectors (24) are specified. These are standard, conservative values for light-quark inversions on a $24^3\times 72$ lattice and close the gap noted in the critique.

## Technical notes

- **Operator**: $\bar{u}\gamma_5 d$ (pseudoscalar, $J^{PC}=0^{-+}$). Under isospin this is the $\pi^-$ annihilation operator; the two-point function is identical to the standard $\pi^+$ channel since both involve one u and one d propagator.
- **Momentum**: Zero momentum ($\vec{p}=\vec{0}$), suitable for ground-state mass extraction.
- **Sink**: Point sink, matching the point source → PP (point-point) correlator.
- **Euclidean time**: Anti-periodic temporal BC for fermions → meson correlator satisfies $C(t)=C(T-t)$ (cosh-like, $T=72$).