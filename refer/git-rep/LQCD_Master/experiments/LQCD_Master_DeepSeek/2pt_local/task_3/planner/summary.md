## Physics Objective

Compute the **connected-only** s̄s pseudoscalar two-point correlator at zero momentum on the C24P29 ensemble (24³×72, β=6.20, a≈0.105 fm). The interpolating operator is O = s̄ γ₅ s. The result is the connected Wick contraction C_conn(t) = Σ_x Tr[S_s†(x,t) S_s(x,t)].

**Critical caveat**: The η_s meson is flavor-diagonal — its physical correlator requires both the connected diagram (computed here) and an equal-order disconnected fermion-loop diagram (NOT computed). The output of this plan is therefore NOT the physical η_s correlator and cannot be used directly for η_s mass extraction. The observable and hadron are renamed to `eta_s_connected_corr` / `connected_sbar_s_ps` to prevent misinterpretation.

## Strategy

- **Gaussian-smeared point source**: A Wuppertal-smeared point source at [0,0,0,0] (n_step=60, width≈2.0) replaces the bare point source from the original plan. This substantially improves ground-state overlap and yields a longer effective-mass plateau.
- **Stout-smeared gauge links**: Applied on a copy of the gauge field (1 step, ρ=0.125, 4-dim) before the Dirac operator inversion. The original unsmeared links are preserved.
- **Zero momentum**: Both source and sink momenta are [0,0,0]; the sink-side spatial sum performs the zero-momentum projection.
- **Multigrid solver**: Uses ensemble blocking parameters [[6,6,6,3],[4,4,4,6]] with a note to verify the axis ordering convention ([T,Z,Y,X] vs [X,Y,Z,T]) against the PyQUDA API at runtime.

## Technical Details

| Item | Value |
|------|-------|
| Hadron label | connected_sbar_s_pseudoscalar |
| Observable label | eta_s_connected_corr |
| Operator | s̄ γ₅ s (connected only) |
| Source | Gaussian-smeared point, [0,0,0,0] |
| Smearing | Wuppertal, n_step=60, width=2.0 |
| Propagator | Strange quark (mass = −0.2356) |
| Gauge smearing | Stout, 1 step, ρ=0.125, 4-dim (on copy) |
| Solver | Multigrid, tol=1×10⁻¹², maxiter=10000 |
| Clover coefficient | 1.160920226 |
| Output | Plain text file, no header/metadata |