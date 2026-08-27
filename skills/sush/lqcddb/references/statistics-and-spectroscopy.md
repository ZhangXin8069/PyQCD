# Statistics and spectroscopy

## Resampling contract

Write down the estimator before calling code:

- original configuration count;
- number and size of resamples;
- leave-one-out versus delete-d or m-out-of-n scheme;
- central estimator;
- error prefactor;
- covariance convention;
- random seed and reproducibility policy.

For delete-one Jackknife, use `(sum(data) - data[k]) / (Nconf - 1)`. The current implementation has this standard sign, while the displayed formula in the current README reverses it.

The current Bootstrap implementation returns complex samples even for real input and uses a nonstandard default resample size. Do not describe `float64` input to `meff` as a physics requirement; it is a current implementation restriction. Convert only after demonstrating that the discarded imaginary part is statistically consistent with zero or after applying the intended symmetry projection.

For complex correlators, decide whether covariance means a Hermitian covariance using a conjugated residual or a covariance of separately stacked real and imaginary parts. The current direct product without conjugation is not the general Hermitian complex covariance.

## Effective masses and fits

- Select the log, cosh, or other estimator from the temporal boundary condition and spectral form.
- Restrict to time points for which all required neighboring correlator values exist and the estimator is real and finite.
- Treat zero-filled invalid endpoints as invalid/masked, not physical zero masses.
- Validate fit windows, parameter ordering, covariance regularization, and degrees of freedom.

## Three-point ratios

- Derive the ratio and square-root factors from the chosen initial/final-state normalization.
- Test equal initial/final states, constant synthetic two-point functions, and a known insertion signal.
- Check axis alignment explicitly; different `tau_axes` and `t_sink_axes` positions are advertised but currently can yield a wrong broadcasted shape.
- For complex ratios, justify any projection to the real part.

## GEVP

Require `C(t0)` to be Hermitian positive definite after a documented conditioning strategy. Preserve complex Hermitian off-diagonal elements. The current `solve_gevp()` symmetrizes and then applies `.real`, which changes valid complex correlator matrices.

Compare against `scipy.linalg.eigh(C(t), C(t0))` on a small complex Hermitian positive-definite example. Track states using eigenvector overlaps across time rather than relying only on independently sorted eigenvalues when level crossings or near-degeneracies are possible.
