# Joint C2 and symmetric rational two-state C3/C2

## Selection and scope

Use this reference for a requested joint C2+C3/C2 likelihood, physical gap
bounds or bound scans, and analysis of effective gaps versus z or channel.
The rational model is an explicit alternative to the expanded model in
[two-state-ratio.md](two-state-ratio.md). Keep these choices independent:

```yaml
model: joint_c2_symmetric_rational_two_state_c3c2
joint_c2: true
share_gap_across_z: false
share_gap_real_imag: true
```

The existing expanded model with `joint_c2=false` remains the default two-state
route. An existing expanded configuration with `joint_c2=true` keeps its
expanded ratio and receives a C2 likelihood with a shared gap. Do not replace
its model silently. A sequential C2 fit or a C2-derived prior is labelled
`paired_c2_constraint` or `c2_gap_prior`, respectively; neither is the joint
likelihood below. This reference concerns the connected-quark ratio route;
check the other product's spectral assumptions before adapting it to gluons.
It is not the C3(z)/C3(0) self-ratio model.

## Spectral model and units

For one momentum, with integer lattice times, let
\(\delta=a\Delta E\) and \(\epsilon_0=aE_0\). Fit

\[
C_2(t)=A_0e^{-\epsilon_0t}\left(1+r e^{-\delta t}\right),
\]

\[
R_{z,\alpha}(T,\tau)=
\frac{M_{z,\alpha}
+B_{z,\alpha}\left[e^{-\delta\tau}+e^{-\delta(T-\tau)}\right]
+D_{z,\alpha}e^{-\delta T}}
{1+r e^{-\delta T}},
\qquad \alpha\in\{\operatorname{Re},\operatorname{Im}\}.
\]

Here \(r=A_1/A_0\) is the C2 excited-to-ground overlap ratio; the numerator
coefficient \(D\) is not the expanded ratio's exponential coefficient.
Expanding the denominator gives \(D-rM\) at order \(e^{-\delta T}\).
Record the equality of source/sink transition amplitudes as a projection and
source/sink assumption. If unsupported, choose an asymmetric model explicitly.
Finite temporal backward/thermal terms are omitted in this template; record
why the fitted time window permits that approximation or extend the model.

Use \(A_0>0\), \(\epsilon_0>0\), \(\delta>0\) after the declared C2 phase
convention. Restrict \(r\geq0\) only when identical source/sink and the selected
C2 projection justify positive spectral weights. Otherwise register the
appropriate amplitude form and require a finite nonzero ratio denominator.
Save every nuisance bound and its boundary hits; instability is not cured by
an unreported bound on r or E0.

For real-only per-z data the seven parameters are

```text
[A0_C2, E0_lattice, r_overlap_ratio, M,
 B_symmetric_transition, D_expT_numerator, deltaE_lattice]
```

For n retained real-valued ratio components there are \(k=4+3n\) parameters:
four shared C2/spectrum parameters and three amplitudes per component. Thus
joint Re/Im has ten, not seven, parameters. A shared-z fit has
\(k=4+3\sum_z n_z\), provided all C2 spectrum parameters are shared. Count
additional parameters if the source, momentum, or nuisance model differs.
For a registered fixed-gap or fixed-parameter study subtract the fixed
parameters from k and record their values; a Gaussian prior does not remove
a parameter and requires a separately labelled likelihood/score policy.
The expanded joint model happens to have the same count but a different
functional form; its ratio exponential coefficient remains independent of r.

At P=0 omit symmetry-forbidden imaginary data at every z. At z=0 keep only
the symmetry-allowed real component for the current connected-quark route.
For nonzero P and z, retain Re/Im cross covariance and a common gap.

With gaps in GeV and a in fm, convert using

\[
\delta=\frac{a_{\rm fm}\Delta E_{\rm GeV}}{\hbar c},
\qquad
\Delta E_{\rm GeV}=\delta\frac{\hbar c}{a_{\rm fm}},
\qquad \hbar c=0.1973269804\;\mathrm{GeV\,fm}.
\]

Apply the same conversion to bounds and profiles. If times are physical,
state the corresponding exponential convention instead of multiplying by a
again.

## Joint observations, covariance and grouping

Construct one real vector per replica,

\[
\mathbf y_b=(C_{2,b}(t_1),\ldots,C_{2,b}(t_q),
\operatorname{Re}R_{z,b},\operatorname{Im}R_{z,b},\ldots),
\]

containing only valid retained coordinates. Verify configuration manifests,
bootstrap indices and row ordering, source/sink setup, momentum and direction
projection. C2 must be the compatible denominator for the declared ratio
estimator. With ratio-before-direction-average products, verify that the
projected C2 model is compatible with the averaged ratio; an average of
rational functions need not equal a ratio using an averaged denominator.
Do not substitute an unrelated C2 or silently rebuild a different estimator.

Estimate the full covariance from the same paired replicas, including the
C2-ratio block. Their statistical dependence is allowed; a block-diagonal
likelihood is a separately labelled approximation. For a first-N pilot subset
all C2, ratio and index rows identically and state the center/covariance policy.
Do not divide bootstrap covariance by Nboot.

Use the main skill's registered correlation shrinkage and SVD prescription.
Save raw covariance/correlation spectra, sample rank, shrinkage, effective
rank, retained-mode mask, cutoff rule and whitening matrix. Declare whether
the cutoff acts on covariance or correlation and distinguish hard mode removal
from eigenvalue flooring. If an elbow is requested, determine it from the
registered central spectrum, save the selection rule and chosen threshold,
and hold the whitening transform fixed for that candidate's replicas.
Regularization does not create independent sample information or repair a
rank-deficient model. Compare the raw sample rank with the required parameter
rank as well as checking the conditioned likelihood.

Minimize

\[
\chi^2=\left\|W(\mathbf y-\mathbf f)\right\|^2,
\qquad \nu=\operatorname{rank}_{\rm retained}-k.
\]

Require positive dof and full whitened model Jacobian rank with declared
scaling/tolerance. Preserve Q as a conditioned-likelihood diagnostic; strong
conditioning and covariance estimation limit its nominal interpretation.
A marginal-C2 plus conditional-ratio Schur decomposition can diagnose which
block drives chi2. Evaluate both at the same joint optimum and record the
conditioning convention; do not assign independent dof or use their sum as
two independent likelihoods. For truncated modes, only claim an additive
block decomposition if it is consistent with that retained likelihood.

For per-z fits C2 appears once in each separate C2+R_z likelihood. These fits
remain correlated across z. A global shared-gap fit stacks all retained z and
includes C2 once per compatible spectrum group, never once for each z or
component. Share neither gaps across momenta/ensembles nor across channels
without a separate declared study.

## Windows and AIC

Register C2 and ratio windows separately in lattice and physical units. The
ratio follows the main skill's physical open T interval and complete contiguous
window/cut registry, including the full available-T candidate for every cut.
The C2 window is fixed across that family unless a C2-window scan is requested;
do not copy integer time bounds blindly between lattice spacings. Preserve
requested and available lists, masks and unavailable candidates.

At fixed nonlinear parameters require full ratio-amplitude design rank. Three
columns \(1\), \(e^{-\delta\tau}+e^{-\delta(T-\tau)}\), and
\(e^{-\delta T}\) can be degenerate despite several data rows. Single-T
windows cannot separate M and D; register them as unavailable for this model.
C2 observations cannot restore that missing ratio-amplitude rank.

Use [interval-aic.md](interval-aic.md): joint chi2, all k parameters and one
common maximal C2+ratio domain. A fixed C2 window adds no omitted observations;
if C2 is scanned, omissions from both blocks count. Pool cuts and ratio
windows within the declared family. Compare ratio-only and joint fits as
separate paired analyses, rather than pooling their unequal data domains.
Average candidate predictions using their central/replica weights; inserting
AIC-averaged nonlinear parameters into one curve is not equivalent.

## Gap bounds, searches and acceptance

Declare the bound policy before inspecting errors. Positive numerical bounds,
a fixed gap, a Gaussian prior and a joint C2 likelihood are distinct constraints.
Bounds are search restrictions, not a measured confidence interval. Do not infer
the physical first excited state from a bound-limited optimum.

A reusable recent-test preset is

```yaml
gap_bounds_GeV: [0.2, 2.0]
gap_bound_role: hard_test_interval
gap_prior: null
chi2_dof_max: 1000
q_min: 1.0e-6
finite_replica_fraction_min: 0.90
identified_nonboundary_replica_fraction_min: 0.90
central_gap_profile_delta_chi2: 1.0
```

Apply strict residual gates \(\chi^2/\nu<1000\), \(Q>10^{-6}\) unless
explicitly overridden. This preset does not set universal gap bounds; use
user-requested limits and save their lattice conversion. For an unrestricted
study, register broad positive numerical bounds and inspect their influence.
If no physical interval was requested, do not silently impose 0.2--2.0 GeV.

Use a coarse gap/profile search and multiple starts, refine all found basins,
and compare minima. For a gap profile fix the gap and reoptimize every other
free parameter, including C2 nuisances. Save the grid, chi2 and nuisance
solutions. A closed central \(\Delta\chi^2=1\) interval must cross on both
sides of the minimum within the registered bounds. It is a local likelihood
diagnostic; with multiple basins save all intervals, not just one Hessian error.
Do not claim closure if the crossing is supplied only by a hard bound.

Save convergence, raw/effective/model rank, scaled Jacobian condition, profile
closure, central and replica boundary flags, finite and identified fractions,
and separate lower/upper hit counts. Record bound-hit tolerances and whether
they are in GeV, lattice units or log-gap. For the existing two-state project
contract, an unqualified accepted gap requires residual/rank gates, no central
bound hit, a closed central profile, and at least 90% locally identified,
nonboundary replicas. State whether each replica also has a full profile or
only local Jacobian/boundary checks; do not claim full replica profiles if not
computed. Count fractions against the full registered replica count.

Keep residual acceptance, data identifiability and physical interpretation as
separate statuses. A tiny transition amplitude or a flat profile can leave
the gap unmeasured even when M is precise and chi2 is very small. Preserve
finite rejected candidates and report M stability separately from gap
identification. Follow the main skill's M-error fallback; do not choose another
candidate solely for a smaller gap error or promote a fallback gap to quality.
For a shared-component/group fit use one common candidate and weight set.

For bound sensitivity, predeclare a lower/upper scan and hold data, replicas,
model, T windows, cuts, covariance rule and grouping fixed. Save paired changes
in M and gap, their correlations/pulls, boundary and fallback fractions, and
AIC weights. An error reduction induced by a tighter bound is not evidence of
reduced excited-state contamination. No post-fit tail clipping, sample deletion,
or substitution of bounds for failed replicas is allowed.

## Physical interpretation and comparisons

A true energy spectrum at fixed ensemble, momentum and symmetry sector is
independent of operator length z and of taking Re/Im. Different currents and
z change transition matrix elements, and a truncated fit can therefore return
different effective gaps. C2 and C3 may emphasize different states; even a
joint seven-parameter fit need not identify one physical excited level.
Inspect C2-only and ratio-only comparisons and residuals before interpreting
a shared fitted gap as a spectrum measurement.

For unpolarized gamma4 at z=0, check the local vector-charge constraint. For
an exactly conserved integrated charge within one charge sector, transitions
vanish and diagonal charges agree: B=0 and D=rM make the ratio constant. It
then provides no additional gap information beyond C2. A local lattice current
need not satisfy that identity exactly; inspect the appropriate normalization
and discretization effects instead of enforcing it silently.

Use measured m_N, m_pi and L to estimate allowed noninteracting multiparticle
levels as diagnostics. Account for total momentum, parity, lattice symmetry
and current quantum numbers; interactions shift the energies. For example,
at rest a positive-parity N-pi state needs odd relative orbital angular
momentum, whereas different multiparticle/current channels have other
thresholds. arXiv:2508.09006 emphasizes current-enhanced states, including
vector-channel N-rho/N-pi-pi contributions: the dominant contaminant need not
be the lowest level. Its local-current results are guidance, not a universal
hard gap prior for a nonlocal PDF operator or transversity.

Compare ensembles at physical z, L=N_s*a, masses and physical time windows;
site counts alone do not order physical volumes. At the current spacings,
L24x72 has L=2.520 fm and L32x96 has L=2.480 fm. Their effective-gap differences
are not a controlled finite-volume scan. Verify metadata rather than treating
these example numbers as immutable constants.

Within one ensemble use paired replica differences across z or channels,
including correlations and any zero-variance case; adjacent-z error bars alone
do not determine significance. Across independent ensembles do not pair draws
by row number; use independent uncertainty propagation or explicitly independent
resampling. Interpolation to common physical z, when needed, must act on the
replicas and preserve correlations. Report local pulls without claiming that
many correlated z points are independent tests.

## Required saved products and figures

Save named parameter vectors and complete replicas with axes, units, masks,
confs, bootstrap indices, original row ids, hashes and grouping. Store each
candidate's parameters, weights, joint chi2/dof/Q, profiles, ranks, boundary
flags and covariance diagnostics so nonlinear summaries can be audited.
At minimum expose:

```text
parameters[z,param], parameters_boot[boot,z,param], parameter_names, units
deltaE_lattice, deltaE_GeV, deltaE_boot_lattice, deltaE_boot_GeV
M_quality, M_boot_quality, M_selected, M_boot_selected
deltaE_quality, deltaE_boot_quality, deltaE_selected, deltaE_boot_selected
residual_accepted, gap_identified, quality_accepted, selected_available
fallback_used, fallback_status, fallback_fraction, candidate_statistical_error
central_lower_bound_hit, central_upper_bound_hit, replica_boundary_fraction
profile_closed, finite_replica_fraction, identified_nonboundary_fraction
```

Adapt axes for components or shared-z groups and document repeated shared
values as the same parameter. Parameter names must distinguish r from B and
D. AIC-averaged gaps and nuisance vectors are model-mixture diagnostics, not a
single spectral solution. Keep unavailable slots NaN and statuses explicit.
Report bootstrap SD(ddof=1) without division by sqrt(Nboot). If any required
replica is unavailable, report the count and label a finite-subset SD; do not
silently call it a full-replica uncertainty. Keep constraint sensitivity and
window/model spread separate from statistical SD.

Produce C2 signal/effective-energy diagnostics and C3/C2 data with finite-T
fit bands, plus bare M and gap versus z. Label lattice and physical z, gap
units/bounds, cuts, actual windows, Nboot and quality/fallback markers. Include
a full-error-range gap figure when zooming and a fixed full-T-window companion
with gap and chi2/dof (Q and retained dof in a table). Per-candidate diagnostics
must remain accessible when AIC summaries look smooth. Predictions outside
the fitted domain must be labelled as extrapolations. Preserve pending external
operator-sign/convention status, including transversity, in the delivery.

## Local implementation examples

The following are audited starting points from the 2026-11-16 first-100 P=0
pilots, relative to the Diagram_GluonPDF workspace:

- `Fit_SKILL/Joint_C2_Rational_Ratio_P0_L32x96_20261116/`
- `Fit_SKILL/Joint_C2_Transversity_P0_L32x96_20261116/`
- `Fit_SKILL/Joint_C2_Channels_P0_L24x72_20261116/`

Inspect their config, engine, report and check scripts and verify current
inputs before reuse. They are P=0 real-only pilots, not a general CLI. A shared-z
or complex production needs the corresponding covariance/parameter extension;
do not claim those examples already implement it. Keep result locations and
chosen test settings in the run manifest, not as implicit universal defaults.
