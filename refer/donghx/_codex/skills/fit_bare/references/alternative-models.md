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

## Two-state Ratio model

Read [two-state-ratio.md](two-state-ratio.md). The user-selectable symmetric
expanded C3/C2 model defaults to a ratio-only gap independently fitted at each
z, shared between its real and imaginary components. Joint C2 fitting and
sharing the gap across all z are independent opt-in switches. Save likelihood,
gap grouping, uncertainty and identifiability diagnostics. Sparse data can
leave a free gap unidentifiable; preserve that status instead of silently
imposing C2, priors, or z sharing.

## Joint C2 and rational two-state C3/C2

Read [joint-c2-ratio.md](joint-c2-ratio.md). This explicitly selected model
shares the two-state gap and overlap ratio between measured C2 and the ratio
spectral denominator. Use the full paired cross covariance. It supports
independent per-z fits or one shared-z likelihood, registered gap bounds,
profile and boundary diagnostics, and paired ratio-only comparisons. Its seven
parameters for a real-only per-z fit differ from the expanded ratio's four.

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

