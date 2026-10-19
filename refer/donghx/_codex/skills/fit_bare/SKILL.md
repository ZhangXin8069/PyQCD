---
name: fit-bare
description: Fit and audit unrenormalized bare matrix elements from a declared ratio product. Supports connected-quark ratio_v4 products and separately routed flowed-gluon disconnected C3/C2 products. It does not produce renormalized or matched PDFs.
---

# Fit bare matrix elements

This skill defines a reusable fit contract. The input estimator, operator
convention, projection, fit model, and ensemble metadata must be recorded
before fitting. A result is a finite-flow bare matrix element, not a physical
PDF.

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

Do not hard-code `Nboot=1500`. Use the product's declared bootstrap count. A
user may override it only by regenerating the raw bootstrap product; record the
override and never call a second-level resampling of already fitted values a
configuration bootstrap when raw configuration data are available.

## Physical minimum separation

Configure a physical threshold per analysis, for example
`tmin_fm_threshold = 0.6`. For lattice spacing `a_fm`, compute

```text
Tmin = floor(tmin_fm_threshold / a_fm) + 1
```

and require `Tmin*a_fm > tmin_fm_threshold`. The threshold, spacing, computed
`Tmin`, and `Tmin` in lattice units travel with every output. Candidate windows
start at this fixed `Tmin`; only their upper endpoint may vary. If the input
does not contain `Tmin` and the next separation required by the model, stop
with a registered unavailable status.

## Select one fit model before fitting

Choose one model globally for the requested production:

- direct ratio: `R(T,tau)=M`;
- forward summed-ratio difference:
  `S_c(T)=sum_{tau=c}^{T-c} R(T,tau)` and
  `D_c(T)=S_c(T+1)-S_c(T)=M`;
- undifferenced summed-ratio slope, or a separately registered constrained
  excited-state model.

Do not select the model point by point. Ratio and Sumratio consume the same
underlying data but have different excited-state and contact-term response;
agreement within errors is a validation target, not an assumption enforced by
the fit. If a comparison is requested, use the same configurations, indices,
projection, direction-average stage, physical `Tmin`, cuts, and admissible
separations. Save paired bootstrap differences, correlation, pull, and method
spread. If the pull is inconsistent, report unresolved method/window
dependence rather than averaging it away.

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
the configured minimum. Scan the globally registered symmetric cuts and all
legal windows beginning at `Tmin`.

Within one fixed cut, use the registered omission-penalized AIC and recompute
weights independently in every bootstrap replica:

```text
AIC_i,b = chi2_i,b + 2*k + 2*N_omit,i,  k=number of fit parameters
```

Do not compare likelihood normalizations across cuts when changing the
observable. AIC-average windows within each cut, then mix usable cuts by the
declared policy and save the cut spread separately.

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

Before reporting, verify fixed physical `Tmin`, actual bootstrap count and
shared indices, masks and axes, input/code hashes, AIC normalization, finite
coverage, candidate fallback statuses, cut spread, and method-comparison
pulls. Plots must distinguish quality-accepted and fallback points. Keep any
operator-convention or parity-projection limitation in the analysis status.
