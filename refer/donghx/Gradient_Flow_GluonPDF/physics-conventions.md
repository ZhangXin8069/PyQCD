# Physics and convention contract

Read this reference for physics interpretation, sign checks, and status claims.

## Scope of the observable

The workflow computes connected nucleon matrix elements of spatially nonlocal
quark bilinears.  The channel labels are:

- `unpolarized`: vector-type quark-PDF operator;
- `helicity`: axial-vector-type quark-PDF operator;
- `transversity`: tensor/chiral-odd quark-PDF operator.

Use the implementation's saved DeGrand--Rossi gamma matrices, contractions,
and channel factors as the executable convention.  Run the convention tests
instead of reconstructing gamma signs from memory.  Gamma-algebra tests alone
cannot establish the historical transversity relative sign; that status stays
`pending_external_physics_confirmation` until checked against the external
operator definition.

The flavors `u` and `d` in this pipeline are connected contributions.
`u_minus_d` is their configuration-aligned difference.  Do not silently call
individual connected `u` or `d` results full physical flavor PDFs when
disconnected diagrams have not been included.  Any statement that disconnected
terms cancel in `u-d` must state the isospin assumptions.

The output is bare: it still depends on lattice spacing, Wilson-line length,
momentum, operator convention, and regulator.  It is not yet renormalized,
matched to a light-cone PDF, Fourier reconstructed, extrapolated to the
continuum, or corrected for finite-volume effects.

## Ratio and excited states

For source-sink separation `T` and insertion time `tau`, the leading structure
is

\[
R(T,\tau)=M+A e^{-\Delta E\tau}
 +B e^{-\Delta E(T-\tau)}+C e^{-\Delta E T}+\cdots .
\]

This form explains both endpoint cuts and why a Sumratio slope can be less
sensitive than a direct plateau.  A multi-state Ratio fit is not identifiable
merely because an optimizer converges: obtain `Delta E` from a compatible C2
spectrum analysis or use transparent priors and retain a prior-constrained
status.

## Six momentum directions and even/odd

For nonzero momentum magnitude the diagnostic directions are
`+z,+y,+x,-z,-y,-x`.  The production estimator forms each direction's C3/C2
ratio inside each bootstrap replica, then performs the VDV even/odd symmetry
combination and averages directions.  For `pabs=0`, only the three positive
axes are distinct.

The analyzed components are `even_real` and `odd_imag`.  Before comparing
direction signs, multiply the three negative-momentum odd directions by the
expected parity factor `-1`.  Only a statistically resolved sign disagreement
after this alignment is evidence for a convention problem.  A direction that
fails its fit gate makes the sign conclusion inconclusive; it is not evidence
of agreement or disagreement.

At algebraic zeros, such as the odd signal at zero displacement, do not turn
floating-point values near `1e-17` into a high-significance physical tension.
The current comparison products record values below `1e-14` in both methods
and their paired error as `numerical_zero_agreement`.

Representative small/intermediate-displacement checks found no global
six-direction sign convention reversal.  A few large-displacement,
near-zero-signal direction outliers exist and must remain local diagnostics;
they do not justify a global sign flip.
