# Alternative bare-matrix-element models

Use these models only when requested or as registered cross-checks. Preserve
the same configurations, bootstrap indices, complex projection, masks,
covariance convention, and output status separation used by the default fit.

## Undifferenced Sumratio slope

Fit

\[
S_{L,R}(T)=B_{L,R}+M N_T,
\qquad N_T=T-L-R+1.
\]

This two-parameter correlated linear fit retains the Sumratio intercept rather
than removing it by differencing. The established implementation is
`Fit_SKILL/Bare_Matrix_Fit`. Use preregistered windows and keep statistical,
window, and method-spread uncertainties separate. It remains a valuable check
because its covariance and excited-state response differ from the default
difference plateau.

## Direct Ratio plateau

Fit all retained insertion-time points to

\[
R(T,\tau)=M.
\]

Use the same `T` window and symmetric endpoint cuts as the comparison model.
The implementation is `Fit_SKILL/Bare_Matrix_Fit` command `fit-ratio`. Compare
it to Sumratio using paired replicas:

\[
p=\frac{M_R-M_S}{\operatorname{std}_b(M_{R,b}-M_{S,b})}.
\]

Do not add the two independent errors in quadrature for this pull: the methods
consume the same configurations and are correlated.

## Historical AICc Sumratio and Ratio branch

`Fit_SKILL/AIC_Fit` retains Sumratio and Ratio AICc window scans. Treat these as
sensitivity products rather than the default. In particular, do not use a
Gaussian log-determinant likelihood to compare different-length data vectors
without a common-observation or explicit omission model. Prefer the default
fit-range AIC omission penalty for nested ranges.

Single accepted-candidate results can supply a statistical error but cannot
estimate a window systematic. Preserve this status explicitly.

## Excited-state Ratio model

When an independently validated C2 analysis supplies a gap, a diagnostic
spectral form may be used, for example

\[
R(T,\tau)=M+A e^{-\Delta E\tau}
 +B e^{-\Delta E(T-\tau)}+C e^{-\Delta E T}.
\]

Do not freely fit a separate `DeltaE` for every `(channel,flavor,p,z)` with the
current sparse separations. Share or constrain gaps using the C2 spectrum, and
record whether the result is data-identifiable or prior-constrained. Ratio and
Sumratio gaps refer to the same spectral energy differences only when derived
from one consistent spectral decomposition; amplitudes and truncation terms
are not the same.

## Comparison policy

- Define candidate models globally before viewing pointwise outcomes.
- Do not select the method having the smallest error, largest Q, or preferred
  sign on a point-by-point basis.
- Save paired bootstrap differences, correlation coefficients, pulls, and
  window/cut stability.
- If methods disagree beyond the registered tolerance, report unresolved
  excited-state or window dependence rather than averaging the discrepancy
  away.
- Keep transversity convention status pending until externally confirmed.

