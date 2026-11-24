# Symmetric expanded two-state C3/C2

## Model and independent switches

For a declared C3/C2 estimator, set \(T=t_{\rm sep}\), \(\tau=t_{\rm ins}\).
The default two-state parameterization is

\[
R_z(T,\tau)=c_{0,z}+c_{1,z}
\left[e^{-\Delta E_z(T-\tau)}+e^{-\Delta E_z\tau}\right]
+c_{2,z}e^{-\Delta E_zT}.
\]

Here \(c_{0,z}=M_z\) is the bare ground-state matrix element. The two
transition terms share one coefficient for each retained component. Real and
imaginary parts have independent \(c_0,c_1,c_2\), with a common positive gap.
Save these independent configuration choices:

```yaml
model: symmetric_expanded_two_state_c3c2
joint_c2: false
share_gap_across_z: false
share_gap_real_imag: true
```

- `joint_c2=false`: estimate the gap from the ratio likelihood alone. C2 is
  still the measured denominator of C3/C2; no C2 spectrum likelihood, imported
  gap estimate, C2 prior, or fitted spectral denominator is added.
- `joint_c2=true`: jointly fit C2 and ratio with one consistent gap. Include
  their cross covariance from the shared configuration bootstrap. Save the C2
  window, C2 nuisance parameters, rank, total parameter count, and joint dof.
  A sequential C2 fit with a propagated fixed gap is a different mode, labelled
  `paired_c2_constraint`, and must not be called a joint likelihood.
- `share_gap_across_z=false`: one fitted gap per z, common to its retained
  components. If also `joint_c2=true`, perform separate per-z C2+ratio fits;
  each includes C2 once and the fits remain correlated. Do not add repeated
  copies of the same C2 into one global likelihood with independent gaps.
- `share_gap_across_z=true`: one gap across the registered z set at each
  momentum, fitted in one stacked likelihood preserving cross-z covariance.
  When joint C2 is enabled, include C2 only once per momentum. Do not share
  across momenta or ensembles unless separately requested.
- `share_gap_real_imag=true`: default; the propagation energies do not depend
  on choosing real or imaginary components. At z=0 fit the real part only;
  at P=0 the symmetry-projected imaginary part is excluded at every z.

With ratio-only per-z gaps there are 4 parameters for real-only data and 7 for
joint real/imaginary data. With n retained components and m z values, all-z
sharing uses 3*n*m+1 parameters (adjust for real-only z). Gap units must be
explicit: using integer lattice times fits \(a\Delta E\).

## Explicit joint rational alternative

For the unexpanded C2+C3/C2 model, shared overlap ratio, parameter counts,
physical gap bounds and gap-comparison products, read
[joint-c2-ratio.md](joint-c2-ratio.md). Its model id is
`joint_c2_symmetric_rational_two_state_c3c2`. This is separate from enabling
`joint_c2=true` for the expanded model above; the ratio ansatz must be recorded
explicitly. Both retain per-z gaps and real/imaginary sharing by default.

## Relation to a spectral ratio

The symmetric expression imposes equality of the source and sink transition
amplitudes in the selected projection. Register this assumption; it does not
follow for every nonlocal operator or source/sink setup. An asymmetric or
unexpanded spectral model must be selected explicitly, not substituted.

The two-state spectral ratio before expanding the C2 denominator is

\[
R_z=\frac{M_z+A_z e^{-\Delta E\tau}
+B_z e^{-\Delta E(T-\tau)}+D_z e^{-\Delta E T}}
{1+r e^{-\Delta E T}}.
\]

Expanding the denominator gives the displayed symmetric model when
\(A_z=B_z=c_{1,z}\) and \(c_{2,z}=D_z-rM_z\), omitting higher products such
as \(e^{-\Delta E(T+\tau)}\) and \(e^{-\Delta E(2T-\tau)}\).
Thus the default is an expanded two-state ansatz, not the exact rational
truncation. It must not be reinterpreted as C3/C3(z=0).

For a joint C2 likelihood one may parameterize

\[
C_2(t)=A_0e^{-E_0t}(1+r e^{-\Delta E t}),
\]

and use the same gap in the declared ratio ansatz. The expanded ratio carries
its stated truncation; do not insert a second spectral denominator into it.

## Fit windows, covariance and gap identification

Use the user-declared physical open T interval and cuts. For an AIC scan,
enumerate all contiguous subwindows inside the interval, including the full
available-T window for every cut. Pool cuts and windows under the main skill's
common-domain omission-penalized AIC and fallback contract. Each window must
contain at least two distinct T values for this expanded two-state model;
single-T candidates remain registered but unavailable. Require full linear
and nonlinear Jacobian rank and positive dof for acceptance. Fixed-window
studies remain explicit and use no implicit AIC mixing. A one-candidate family
has weight 1. Never choose a cut per point solely for its observed error except
under the declared no-quality-candidate fallback.

Record covariance method and any SVD cut. Retain real/imaginary cross
covariance and, for shared-z or joint C2 fits, their additional cross blocks.
Count only retained eigenmodes in dof, then subtract all free parameters.

Fit every paired bootstrap replica. A positive gap can be optimized in its
logarithm with declared broad bounds, a coarse profile search and refinement
of all found basins. Bounds are numerical restrictions, not measurements or
priors. Record boundary fractions and test open profiles. For bound scans and
physical interpretation follow the gap-analysis sections of
[joint-c2-ratio.md](joint-c2-ratio.md); a hard bound is not evidence for a state. At a minimum save
central and bootstrap residual quality, full Jacobian rank, gap profile,
finite fraction and identified fraction. Separate residual acceptance from
gap identifiability and final acceptance. A tiny c1 or a flat profile can make
the gap unidentifiable despite an excellent Q.

Unless overridden, this two-state mode uses chi2/dof<1000, Q>1e-6, finite
bootstrap fraction>=0.90, with identifiable central covariance/design and
positive dof. For an unqualified accepted gap also require a closed central
Delta-chi2=1 profile, no central bound hit, and >=0.90 identifiable nonboundary
replicas. Preserve finite rejected fits as diagnostics. Never replace them
silently by constrained-gap values.

## Products and plots

Save all c0,c1,c2 and gaps, complete parameter replicas, masks, coordinates,
covariance/SVD metadata, grouping choices, indices and hashes. Report SD from
all registered replicas without division by sqrt(Nboot). If replicas fail,
keep their slots NaN; a finite-subset diagnostic SD must be named separately.
Expose `M_quality`, `M_boot_quality`, `quality_accepted`, `M_selected`,
`M_boot_selected`, `selected_available`, fallback flags/status/fractions and
candidate statistical errors. In a single-candidate fixed-window test,
fallback is that finite candidate with diagnostic status, not another model.

Signal plots show data and full finite-T fit curves for every fitted z,
distinguish excluded insertion times, and label the asymptotic c0. State
whether prediction bands are quantiles or SD. Bare plots distinguish accepted
and diagnostic values, retain a full-error-range companion when axes are
zoomed, and show real and imaginary parts. Keep all-z and per-z spectra
conceptually separate: physical energies do not change with operator length;
independent fitted gaps can reflect excited-state truncation or weak data.
