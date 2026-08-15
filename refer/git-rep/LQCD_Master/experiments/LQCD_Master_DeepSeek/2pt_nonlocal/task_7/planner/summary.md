## Physical Essence

This plan computes the **nonlocal two-point correlation function** of the $D_s^+$ meson ($c\bar{s}$, $J^P=0^-$) with spatial separation $z = 0, 1, \dots, 10$ along the $+z$ direction, using a single configuration (cfg 10000) on the C24P29 ensemble ($24^3\times72$, $a\approx 0.1052$ fm). The nonlocal sink operator $\bar{s}(x)\gamma_5 W(x,x+z) c(x+z)$ contains a straight Wilson line built from **original (unsmeared)** gauge links, making this a quasi-distribution amplitude (quasi-DA) calculation in the LaMET framework. The source operator is the local $D_s$ interpolator $\bar{c}(0)\gamma_5 s(0)$. Momentum projection is zero throughout.

## Key Revision: Solver Strategy for Heavy Charm Quark

The most important change from the previous plan is the **inverter choice for the charm propagator**. The multigrid solver—with block sizes $[6,6,6,3]/[4,4,4,6]$ tuned for near-massless nullspace vectors—is **ineffective for the heavy charm quark** ($m_c = 0.4159$). At this mass the Dirac operator lacks the near-nullspace structure that the coarse-grid correction relies on; using multigrid may slow or even destabilize convergence. The revised plan therefore uses:

- **Charm propagator** (`prop_c`): **BiCGStab** Krylov solver (standard for non-Hermitian Wilson-clover systems with heavy quarks), tolerance $10^{-8}$, max 10000 iterations.
- **Strange propagator** (`prop_s`): **Multigrid** solver retained, since $m_s \approx -0.2356$ is light enough that the existing nullspace vectors still provide effective coarse-grid acceleration.

Both inversions are performed on stout-smeared gauge links (1 step, $\rho = 0.125$, 4-dim) with Wilson-clover Dirac operator ($c_{sw} = 1.160920226$).

## Other Minor Fix

The contraction notation in Step 7 has been cleaned up: the unused dummy index $b$ in $\Sigma_{a,b}$ was removed, leaving the correct sum $\Sigma_a$ over the single color index. The contraction $C(z,t) = \sum_{\vec{x}} \mathrm{Tr}[S_s^\dagger(\vec{x},t;0) \cdot S_c^{\text{shifted}}(\vec{x},t;z)]$ (trace over both spin and color) is unchanged.

## Preserved Elements

- Point source at $[0,0,0,0]$ for both propagators.
- Nonlocal shift applied only to the charm (quark) propagator via Wilson line; strange (antiquark) propagator remains local.
- Wilson lines constructed from original gauge links at each time slice, with periodic spatial wrapping.
- Output: plain text file `ds_nonlocal_2pt.txt` with columns `z t Re[C] Im[C]`, no header.
- Single configuration only (cfg 10000); the plan notes that production physics would require many more configurations.