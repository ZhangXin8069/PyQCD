# Correlator Analysis Pipeline

This reference covers statistical and spectroscopic analysis after contractions are complete. Use the current `src/lqcddb` and `api-quick-reference.md` for exact signatures.

## Data contract

Record configurations, source time, sink time, insertion time, momentum, operator, spin/projection, link direction, and real/imaginary axes. State whether the input consists of raw configurations, source-time averages, Jackknife samples, or Bootstrap replicates.

Source-time averaging must follow the correlator's wraparound quantum number. The boundary factor of an antiperiodic quark field is determined by the number of fermion lines crossing the temporal boundary; not every meson and baryon receives the same minus sign. For a two-point function, use `loop_tsrc` to retain a length-one source axis and shift one time axis; for multi-current functions, explicitly shift every insertion axis and do not apply a wraparound phase twice.

## Resampling

The Jackknife sample formed by deleting configuration $k$ is defined as
$$
x^{(k)}=\frac{\sum_i x_i-x_k}{N-1},\qquad
\sigma^2_\mathrm{JK}=\frac{N-1}{N}\sum_k
\left(x^{(k)}-\bar x_\mathrm{JK}\right)^2.
$$

Report the configuration axis, sample count, central estimator, and covariance axes. The current `Jackknife` preserves the original configuration-axis position, while `Bootstrap` places the replicate axis at axis zero. Complex data should use the Hermitian covariance
$$
\operatorname{Cov}_{ij}=\left\langle
(x_i-\bar x_i)(x_j-\bar x_j)^*\right\rangle
$$
or stack real and imaginary parts before constructing a real covariance. A product without conjugation is only a pseudo-covariance and cannot replace the full covariance.

For Bootstrap, record the random seed, number of replicates, samples per draw `M`, and whether `M=Nconf`. The current implementation scales the variance by `M/Nconf` when `M\ne Nconf`; use the same convention when interpreting errors.

## Ratios, effective masses, and GEVP

Derive a three-point ratio from the normalization of the selected states, specifying the initial/final two-point functions, `t_sep`, momenta, and square-root branch. Check the valid domain and `valid_mask` returned by `ratio_3pt` before selecting a plateau region.

The `log`/`GEVP` modes of `meff` use adjacent-time ratios, while the `cosh` mode uses a three-point `arccosh`; invalid points should remain NaN. GEVP requires a Hermitian positive-definite reference matrix; record `t0`, eigenvalue ordering, state tracking, and residuals.

## Plotting and fitting

Use `Axes.errorbar` for statistical points by default, showing `yerr`, units, labels, and the fit window. Use clearly different colors, markers, or line styles for different momenta, operators, and states. Fit with `lsqfit.nonlinear_fit`: scan mean-data windows with `chi2/dof\approx1`, then fit every configuration or Jackknife sample with the same window and compute parameter errors.
