This task is a standard hadronic matrix-element calculation, specifically a mesonic three-point function for the flavor-changing transition D0(anti-u c) -> K*-(anti-u s) induced by the vector current J_x = sbar gamma_x c. It is therefore classified as a standard meson_3pt problem, not a pure spectroscopy or gauge-only observable.

Core physics content:
- Source state: D0 with quark content anti-u c, represented by the pseudoscalar interpolating operator cbar gamma_5 u.
- Sink state: K*- with quark content anti-u s, represented by the vector interpolating operator sbar gamma_x u.
- Current insertion: J_x = sbar gamma_x c.
- Kinematics: source momentum and sink momentum both set to zero, so the transfer momentum is also zero.
- Correlator type: three-point only; the user explicitly requested not to compute 2pt functions.

Numerical strategy chosen:
- Use one point source at [0,0,0,0] as requested.
- Compute forward light propagator for the spectator anti-u/u line from the source.
- Compute forward charm propagator from the same source for the initial D0 charm quark.
- Build a mesonic sequential source at sink time tseq = 8 using the sink operator gamma_x and zero sink momentum, then invert with the strange mass to obtain the strange sequential propagator.
- Form the final 3pt correlator by contracting the strange sequential propagator with the gamma_x current and the forward charm propagator. This is the standard mesonic sequential-sink setup for a heavy-to-light transition with fixed sink operator and sink time.

Reasonable completions I made:
- I interpreted “Invert all propagators using stout-smeared gauge links with parameters (1, 0.125, 4)” in the usual PyQUDA order as n_steps = 1, rho = 0.125, ndim = 4, and encoded it explicitly.
- I used the ensemble clover coefficient and the provided quark masses directly.
- Since solver stopping conditions were not specified, I chose conservative production-style defaults: tolerance 1e-12 and maxiter 10000.
- I kept point sink treatment to match the user’s request and to avoid introducing unsaid smearing.
- Because the template only allows a single propagator field reference per measurement item, the detailed dependence on the forward light and charm propagators is described in the task and solver notes; operationally the 3pt contraction requires all three propagators.
- The output format is marked as “other” with metadata disabled to enforce the user’s requirement of a plain txt file with no header or extra text in the run directory.

This plan is executable and engineering-ready for code generation: it distinguishes the task correctly as a standard meson 3pt calculation, specifies the operator chain and propagator content, fixes the source/sink/current structure, and respects all user constraints including no 2pt output and plain text final data.