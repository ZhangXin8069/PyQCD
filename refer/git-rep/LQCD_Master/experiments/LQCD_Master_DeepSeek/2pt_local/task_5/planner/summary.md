## Physics Objective

Compute the two-point correlation function of the rho+ meson ($\rho^+ = \bar{d} u$) at zero momentum for mass extraction. This is a standard meson spectroscopy task on the C24P29 ensemble (single configuration, cfg 10000).

## Strategy

### Operator and Correlator

The interpolating operator for the vector meson is $\mathcal{O}_{\rho^+_i} = \bar{d} \gamma_i u$ with $i = 1,2,3$ for the three polarization states. The two-point correlation function averaged over polarizations is:

$$C_\rho(\vec{0}; t, 0) = \frac{1}{3} \sum_{i=1}^3 \langle \mathcal{O}_{\rho^+_i}(\vec{0}, t) \mathcal{O}^\dagger_{\rho^+_i}(\vec{0}, 0) \rangle$$

Wick contraction yields one connected diagram (no disconnected contributions for the charged rho). Applying $\gamma_5$-hermiticity ($S(x,y) = \gamma_5 S^\dagger(y,x) \gamma_5$) and flavor symmetry ($S_u = S_d = S_l$) simplifies the trace to:

$$C_\rho(\vec{0}; t, 0) = \frac{1}{3} \sum_{i=1}^3 \sum_{\vec{x}} \text{Tr}\left[ S_l^\dagger(\vec{x}, t; \vec{0}, 0) (\gamma_5 \gamma_i) S_l(\vec{x}, t; \vec{0}, 0) (\gamma_i \gamma_5) \right]$$

Only one light-quark propagator inversion is required — the same `prop_l` appears in both positions of the trace. The overall sign is positive after the fermion anticommutation sign and the gamma-algebra simplification.

### Source and Gauge Smearing

- **Point source** at fixed position $[0,0,0,0]$ (spatial origin, time slice 0), exactly as the user requested.
- **Stout link smearing** is applied to the gauge field before the Dirac inversion with parameters `n_steps=1, rho=0.125, ndim=4`. This suppresses UV noise in the propagator without modifying the source.
- **Point sink** — no additional smearing at the sink.

### Solver

Multigrid solver with light quark mass `-0.277`, clover coefficient `1.160920226`, tolerance `1e-12`, max iterations 10000, and two-level blocking `[6,6,6,3]` / `[4,4,4,6]` from the C24P29 ensemble.

### Design Choices

- **Zero momentum** ($\vec{p} = \vec{0}$): standard for ground-state mass extraction; the user did not specify a momentum.
- **Point sink**: consistent with the point source; no sink smearing.
- **Three-polarization average**: $\gamma_1, \gamma_2, \gamma_3$ are traced separately, then the traces are averaged to improve statistics.
- **Single configuration**: cfg 10000 provides one gauge sample. No statistical error estimates are produced, which is consistent with the user's request for a raw correlator output.
- **Output**: the full 72-time-slice correlator (one real number per time slice, T=0..71) is written as a space-separated plain-text file with no header, saved in the run directory.

## Technical Details

| Item | Value |
|------|-------|
| Hadron | $\rho^+$ (vector meson, $J^{PC}=1^{--}$) |
| Flavor | $\bar{d} u$ (both light quarks) |
| Gamma matrices | $\gamma_1, \gamma_2, \gamma_3$ (DeGrand-Rossi basis) |
| Momentum | $\vec{p} = (0,0,0)$ |
| Source position | $(x,y,z,t) = (0,0,0,0)$ |
| Propagator inversions | 1 (light quark, point source) |
| Gauge smearing | Stout: 1 step, $\rho=0.125$, 4-dim |
| Diagram topology | Connected only (no disconnected for $\rho^+$) |
| Output | Plain text, one value per time slice (T=0..71) |