---
name: fit-bare
description: Fit and audit unrenormalized bare matrix elements from a declared ratio product. Supports connected-quark ratio_v4 products, joint C2 and C3/C2 two-state fits with energy-gap constraints and diagnostics, and separately routed flowed-gluon disconnected C3/C2 products. It does not produce renormalized or matched PDFs.
---

# Fit bare matrix elements

This skill defines a reusable fit contract. The input estimator, operator
convention, projection, fit model, and ensemble metadata must be recorded
before fitting. For connected quarks the result is an unrenormalized bare matrix element;
for flowed gluons it is a finite-flow bare matrix element. Neither is a
renormalized or matched physical PDF.

## Route and input audit

Read the relevant local-product and operator-convention reference before
consuming data. The connected-quark route uses corrected `ratio_v4` products;
the flowed-gluon route uses its own C3/C2 schema and must not be silently
treated as a quark product. Reject products with wrong estimator id, axes,
mask, configuration list, or bootstrap-index alignment.

Verify and save:

- configuration manifest and its hash;
- the actual `Nboot`, resample size, method, seed, and bootstrap indices;
- all channel/flavor/momentum/direction products and input hashes;
- complex projection, parity projection, operator formula, and overall sign;
- axes and masks, with invalid padding kept as NaN.

Do not hard-code `Nboot=1500`. Use the product's declared bootstrap count. A user may explicitly request a pilot using the first N existing replicas.
Subset every observable and bootstrap-index row identically; save parent Nboot,
original row ids, configuration list, and subset indices. Declare whether the
center is the original-data estimate or the subset bootstrap mean, and whether
covariance uses that subset. This is a subset diagnostic, not a new bootstrap.
Otherwise changing the bootstrap count requires regenerating the raw product;
never call second-level resampling of fitted values a configuration bootstrap.

## Physical separation interval and candidate windows

Before fitting register a physical open interval `tmin_fm < T*a_fm < tmax_fm`
(the upper bound may be infinity), lattice spacing, cuts, and model. The integer
bounds are `floor(tmin_fm/a_fm)+1` and `ceil(tmax_fm/a_fm)-1`; verify the strict
inequalities numerically. Intersect with actual valid source separations and
save both requested and available T lists. Never invent missing T values.

Enumerate all contiguous subwindows of that ordered available T list, varying
both endpoints, for every registered symmetric cut. The candidate registry
must contain the full window using every available T, for each cut, even if
that candidate later fails rank or quality gates. Register single-T windows as
unavailable when the model cannot identify them. Record gaps in source T and
whether contiguity refers to the available list or lattice integers. Do not
silently retain the old fixed-Tmin-only scan. Fixed-window studies remain an
explicit option. Read [references/interval-aic.md](references/interval-aic.md)
for the common-domain omission count and cross-cut rules.

## Select one fit model before fitting

Choose one model globally for the requested production:

- direct ratio: `R(T,tau)=M`;
- forward summed-ratio difference:
  `S_c(T)=sum_{tau=c}^{T-c} R(T,tau)` and
  `D_c(T)=S_c(T+1)-S_c(T)=M`;
- undifferenced summed-ratio slope;
- symmetric expanded two-state C3/C2, documented in
  [references/two-state-ratio.md](references/two-state-ratio.md). Its defaults
  are `joint_c2=false`, `share_gap_across_z=false`, and
  `share_gap_real_imag=true`. Fit the gap from the ratio alone unless the
  user requests a C2 constraint. Do not silently switch to a rational spectral
  denominator or independent source/sink amplitudes;
- joint C2 plus symmetric rational two-state C3/C2, documented in
  [references/joint-c2-ratio.md](references/joint-c2-ratio.md). Select
  `model: joint_c2_symmetric_rational_two_state_c3c2` and `joint_c2=true`.
  C2 and ratio share the gap and C2 overlap ratio in the spectral denominator.
  The default gap groups remain per-z, with shared real/imaginary gaps.

When two-state fitting is requested, read the relevant model reference before
selecting the likelihood, gap groups, parameter count, or windows. The expanded
model can also use a joint C2 likelihood; changing `joint_c2` does not change
its declared ratio ansatz. The rational model is a separate explicit choice.
The two gap switches are independent; save both with every product. A per-z gap is an effective
truncation diagnostic, not a claim that the physical spectrum depends on z.

Do not select the model point by point. Ratio and Sumratio consume the same
underlying data but have different excited-state and contact-term response;
agreement within errors is a validation target, not an assumption enforced by
the fit. If a comparison is requested, use the same configurations, indices,
projection, direction-average stage, physical `Tmin`, cuts, and admissible
separations. Save paired bootstrap differences, correlation, pull, and method
spread. If the pull is inconsistent, report unresolved method/window
dependence rather than averaging it away.

## Energy-gap constraints and analysis

For either two-state model register positive gap bounds and their units before
fitting; keep hard bounds, Gaussian priors, fixed gaps, and a joint C2
likelihood as separate modes. Read the gap-analysis sections of
[references/joint-c2-ratio.md](references/joint-c2-ratio.md) when imposing or
scanning constraints, comparing channels/ensembles, or studying gaps versus z.
The recent `0.2 < DeltaE < 2.0 GeV` pilot is an explicit reusable test preset,
not a universal physical interval or an automatic new default. Convert physical
bounds using the actual ensemble spacing when fitting integer times.

Save candidate and replica gaps, complete parameter vectors, residual gates,
rank, profile closure, bound hits and tolerances, and identification fractions
separately. A low chi2 or small M error does not identify the gap. Preserve
quality and fallback status independently for M and DeltaE; rectangular
fallback coverage does not validate an effective energy gap. Plot DeltaE versus
z in fm, its uncertainty, imposed bounds, full-T diagnostics, and chi2/dof.
Compare paired gaps within an ensemble and independent errors across ensembles.
AIC-averaged gaps are diagnostics; average candidate predictions for curves.

## Direction and projection contract

Declare whether directions are averaged before constructing `S`/`D`, before
fitting direct `R`, or after independent fits. The default is to average the
ratio values within each shared bootstrap replica before fitting. Preserve the
cross-direction covariance when doing a joint fit. These orders are not
interchangeable because ratios contain denominators.

For gluon products, record the explicit tensor combinations, for example

```text
T = ti + tj
S = 2*ij
T-S = ti + tj - 2*ij
T+S = ti + tj + 2*ij
```

Record whether unpolarized uses even-z, helicity uses odd-z, and whether the
available product actually contains both orientations. A plus-z-only product
must not be labelled an even-z result.

## Correlated fits and windows

Use the declared covariance estimator (default: Schaefer--Strimmer
correlation-to-identity) and registered SVD cut. Apply fixed central quality
gates: finite inputs, identifiable covariance/design rank, positive degrees of
freedom, `Q >= q_min`, `chi2/dof <= limit`, and finite bootstrap fraction above
the configured minimum. Use globally registered symmetric cuts. For an AIC
scan enumerate the interval's complete candidate family before inspecting fit
results. For an explicitly fixed-window test preserve its declared window.

Pool different cuts and T windows in one family for the same observable and
parameter target, using the omission-penalized score independently per replica:

\[
\mathrm{AIC}_{i,b}=\chi^2_{i,b}+2k_i+2N_{\mathrm{omit},i},\qquad
w_{i,b}=\frac{e^{-\Delta\mathrm{AIC}_{i,b}/2}}
{\sum_{j\in\mathcal A_b}e^{-\Delta\mathrm{AIC}_{j,b}/2}}.
\]

Count omitted real observations relative to one predeclared common maximal
domain across all cuts and windows; never reset the baseline per cut. Preserve
covariance ranks, likelihood convention, weights and cut totals. This is a
registered omission-penalized selection prescription; covariance conditioning
and masks can affect scores. If cuts change the observable itself (e.g.
summed-ratio differences), use the separate normalization policy in the
reference, rather than claiming the raw likelihoods are identical.

## Complete replica selection and fallback

Fit every bootstrap replica for every retained `z`, candidate window, and cut.
For each fixed `(model, channel, momentum, z, replica)`:

1. AIC-average all candidates passing that replica's quality gates.
2. If none passes, choose the finite rejected candidate with the **smallest
   precomputed candidate statistical error**. This error is the candidate's
   `std(M_boot, ddof=1)` over the complete finite bootstrap ensemble, not an
   error manufactured from the one replica being selected. Break ties by
   smaller raw `chi2`, then registered candidate order.
3. If no candidate has a finite statistical error, choose the finite candidate
   with smallest raw `chi2`; if no positive-dof candidate exists, use the
   registered single-point/zero-dof diagnostic only when the model defines one.
4. If no finite candidate exists, retain NaN and a terminal
   `no_finite_candidate` status; do not fabricate a value.

The fallback value is accepted for **complete rectangular coverage**, but is
not quality-accepted. Save separate fields:

```text
M_quality, M_boot_quality, quality_accepted
M_selected, M_boot_selected, selected_available
fallback_used, fallback_status, fallback_candidate_index
fallback_fraction, candidate_statistical_error
```

The reported statistical error for selected values is `std(M_boot_selected,
ddof=1)` with no division by `sqrt(Nboot)`. Always report quality-accepted and
fallback counts separately.

## Completion checks

Before reporting, verify the physical open interval, all candidate windows including the full-T window, actual bootstrap count and
shared indices, masks and axes, input/code hashes, AIC normalization, finite
coverage, candidate fallback statuses, cut spread, and method-comparison
pulls. Plots must distinguish quality-accepted and fallback points. Keep any
operator-convention or parity-projection limitation in the analysis status.
For joint fits also verify C2/ratio pairing and cross covariance, C2 appearing
only once in each likelihood, the selected spectral denominator, parameter
count, physical gap-bound conversion, profile/boundary status, and saved
DeltaE diagnostics. Preserve any pre-existing external convention limitation.
