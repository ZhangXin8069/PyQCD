## Revised plan summary

The revised plan keeps the original task intact but makes the physics definition and execution details explicit enough to be checkable.

The observable is now stated unambiguously as a **connected isovector light-light nonlocal vector two-point correlator in the rho channel**, not a flavor-diagonal rho correlator. The sink operator is
\[
\bar d(x,t)\,\gamma_i\,W_z(x,x+z\hat z;t)\,u(x+z\hat z,t),
\]
with a **local source** \(\bar u(0)\gamma_i d(0)\), and the calculation uses one degenerate light propagator because of exact \(u/d\) degeneracy.

The key correction is that the **mixed-link setup** is promoted from an implementation note to part of the physics definition:
- the light propagator is inverted using the **stout-smeared clover operator** with parameters \((n_{\rm steps},\rho,n_{\rm dim})=(1,0.125,4)\),
- the nonlocal Wilson line is built from the **original unsmeared gauge links**.

The quark-leg-only displacement is also defined precisely. For each separation \(z=0,1,\dots,10\), the shifted propagator is constructed as
\[
S_l^{\rm shift}(x,t;0)=W_z(x,x+z\hat z;t)\,S_l(x+z\hat z,t;0),
\]
where the Wilson line is a straight product of **same-timeslice spatial +z links** with periodic spatial wrapping. The antiquark leg remains local at \(x\).

Because the displacement is in the \(z\) direction, the plan no longer assumes the three vector polarizations are automatically equivalent. Instead, it requires measuring **\(\gamma_x\), \(\gamma_y\), and \(\gamma_z\) separately first**, then forming the requested average only after this validation.

The output format is now fully fixed: the final file is a plain txt file in the run directory with **no header** and exactly four columns
\[
z\quad t\quad \mathrm{Re}\,C_{\rm avg}(z,t)\quad \mathrm{Im}\,C_{\rm avg}(z,t).
\]
Rows are written deterministically for all \(z=0\ldots 10\) and \(t=0\ldots 71\).

Finally, the plan explicitly states the scientific scope. Using only **cfg 10000** and a single point source at **[0,0,0,0]** is treated as a **validation/smoke-test run**, suitable for checking the operator construction, the Wilson-line orientation, and the \(z=0\) local limit, but not for a statistically meaningful physics result. It also records the limitation that the measured object is a **bare nonlocal correlator** with Wilson-line self-energy contamination and no renormalization prescription.