# Interval and cross-cut AIC contract

Register the physical open interval, available ordered T list, all symmetric
cuts, all contiguous subwindows, model parameter count, and quality gates
before examining fit outcomes. Include each full-T window. Preserve unavailable
candidates and reasons. Missing source T is not generated; declare contiguity
on the available list. Symmetric cut c retains c <= tau <= T-c.

For direct ratios, define one common maximal real observation vector from all
available T at the loosest cut, with the same projection and retained components.
A candidate selects a mask of that vector. Thus

\[
N_{\mathrm{omit},i}=N_{\mathrm{common}}-N_{\mathrm{data},i}.
\]

Count Re and Im as separate real observations; do not count padding or exact
normalization identities. The omission count refers to coordinates before SVD,
not retained eigenmodes. Keep the full Re/Im covariance. Declare the covariance
estimator and conditioning identically across the family, and save raw/effective
rank, eigenvalues, shrinkage and SVD. Large conditioning changes merit a
sensitivity check; this score is a prescribed generalized AIC diagnostic,
not a proof that differently regularized data spaces have identical evidence.
Do not add a covariance log determinant silently.

Cuts that redefine the observable itself, e.g. S_c or D_c, are not masks of the
same vector. To request pooled scores there, predeclare a common parent-ratio
coordinate domain and its omission rule and label the result a generalized
selection prescription. If no common rule is declared, retain separate-cut
likelihoods and report the obstruction; do not invent a common normalization.

Compute the score and normalized weights separately for the central estimate
and every bootstrap replica, pooling all passing cut/window candidates. Save
central residual gates, replica residual gates, gap identification, parameter
rank, boundary status, finite fraction, weights and cut totals independently.
Use the model's registered identification gates for final eligibility.
Evaluate model-averaged curves by averaging candidate predictions with those
same weights, not by inserting averaged nonlinear parameters into a curve.
Averaged nuisance parameters/gaps are diagnostics, not one spectral solution.

If no candidate passes, apply the main skill's precomputed full-replica SD
fallback. For a joint Re/Im target register the scalar fallback score, e.g.
sqrt(SD(M_Re)^2+SD(M_Im)^2), and use one common candidate for both components.
Never compare errors computed from just the current replica. If a fit cannot
exist, retain unavailable status rather than inventing a single-T two-state fit.

A pilot may use the first N existing paired bootstrap rows. Save parent count,
row ids and exact subset indices. Label all error estimates as pilot errors;
use SD(ddof=1), never SD/sqrt(N), and never trim long tails after fitting.

## Joint C2 plus ratio candidates

For [joint C2+ratio](joint-c2-ratio.md), the maximal real-observation domain
includes C2 once and all registered ratio coordinates. With a fixed C2 window,
C2 contributes to joint chi2 and k in every candidate but adds no omitted
observations; only missing ratio coordinates contribute to N_omit. If C2
windows are explicitly scanned, omissions from both blocks use that same
maximal domain. Count all shared and nuisance parameters exactly once. Keep
ratio-only and joint-likelihood families separate when comparing methods;
their different observation vectors do not define one automatic pooled AIC
family. Register the full available-T ratio window even if C2 cannot rescue
its ratio-amplitude rank.
