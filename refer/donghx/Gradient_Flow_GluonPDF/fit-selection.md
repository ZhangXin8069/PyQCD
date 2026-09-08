# Sumratio, Ratio, cuts, and result selection

Read this reference before fitting, changing windows, comparing methods, or
choosing a quoted bare matrix element.

## Current fit definitions

The Sumratio fit is

\[
S_{L,R}(T)=\sum_{\tau=L}^{T-R}R(T,\tau)
 =B_{L,R}+M N_T,\qquad N_T=T-L-R+1.
\]

Fit the undifferenced sum.  The central estimator uses `ratio_original`;
bootstrap replicas determine covariance and statistical uncertainty.

The independent Ratio cross-check is a correlated common constant over all
retained `(T,tau)` points:

\[
R(T,\tau)=M.
\]

Do not use a free one-gap Ratio model as the default without an independently
constrained C2 gap.  If a reliable C2 spectrum becomes available, a shared-gap
excited-state fit is a third diagnostic rather than a post-hoc replacement.

Both methods use Schaefer--Strimmer correlation shrinkage, an explicit SVD
inverse, and the same bootstrap indices.  Preserve raw covariance rank,
condition number, eigenvalues, shrinkage, Q, correlated chi2/dof, and bootstrap
success diagnostics.

## Preregistered current windows

| ensemble | primary | scanned Tmax | cuts |
|---|---|---|---|
| L24x72 | `T5-9_L2_R2` | 8,9,10 | 1 or 2 |
| L32x64 | `T6-9_L2_R2` | 8,9,10 | 1 or 2 |
| L32x96 | `T7-10_L2_R2` | 9,10,11 | 1 or 2 |

Cut 2 improves direct-Ratio acceptance relative to cut 1, while Sumratio is
less sensitive, which supports endpoint contamination and makes `L=R=2` the
current default.  Convert cuts to physical distances when lattice spacings are
available; roughly 0.2--0.3 fm is a useful starting scale, not a universal hard
threshold.

Use symmetric cuts for an elastic matrix element with identical source/sink
smearing.  Allow `L != R` only if direction/time residuals or a spectral model
demonstrate a genuine source/sink asymmetry.

The present grid varies `Tmax` but not `Tmin`.  Before a publication-level
choice, add preregistered `Tmin+1` and, when enough separations remain,
`Tmin+2` candidates.  This is particularly important for L24x72.

## Fit gates and selection

A candidate must have valid masked inputs, enough covariance/model rank,
positive fit degrees of freedom, finite output, `Q >= 0.05`, correlated
`chi2/dof <= 3`, and bootstrap success at least 0.90.

Apply this policy:

1. Define cuts and windows globally per ensemble before inspecting individual
   `(channel,flavor,p,z)` estimates.
2. Require the preregistered primary window to pass; do not fall back to an
   alternate that happens to pass.
3. Require at least one accepted alternate.  Save the equal-weight RMS shift
   of accepted alternates relative to primary as a separate window systematic.
4. Use Sumratio as the primary method and Ratio plateau as the paired
   cross-check.
5. Compute
   \[
   p=(M_R-M_S)/\operatorname{sd}_b(M_{R,b}-M_{S,b}).
   \]
6. If both methods pass and `|p| <= 2`, quote the Sumratio central value and
   retain statistical, window, and full method-spread
   `|M_R-M_S|` uncertainties separately; a quadrature total may be an
   additional summary.
7. If `|p| > 2`, do not average methods or select the better-looking one.
   Mark unresolved excited-state/window dependence and test later `Tmin` or a
   C2-constrained multi-state model.

The smallest uncertainty, largest Q, and optimizer convergence are not
definitions of an optimal result.  “Optimal” here means the simplest
preregistered estimator that passes quality, neighboring-window stability, and
cross-method consistency over a physically motivated range.

## Dated production summary: 2026-09-01

Recompute these numbers if any fit or comparison artifact changes.

Across 27 observables, 2624 points have both methods valid.  Of these, 2327
(88.7%) agree within paired 2 sigma, 252 lie between 2 and 3 sigma, and 45 lie
above 3 sigma.  The median absolute paired pull is 0.742.  Agreement is weaker
for `even_real`, `pabs=3,4`, and unpolarized.  The points are correlated across
`z`, flavor, and channel, so do not convert the counts to a binomial global
p-value.

| ensemble | jointly valid | within 2 sigma | joint fraction of all points |
|---|---:|---:|---:|
| L24x72 | 404 | 339 (83.9%) | 28.1% |
| L32x64 | 1175 | 1031 (87.7%) | 62.2% |
| L32x96 | 1045 | 957 (91.6%) | 55.3% |

The low L24x72 joint fraction makes its present primary window provisional.
Do not repair missing results by per-point fallback.  Extend the preregistered
`Tmin` scan or add a constrained spectral analysis.
