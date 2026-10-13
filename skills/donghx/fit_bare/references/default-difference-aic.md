# Default forward-Sumratio-difference AIC model

## Observable and threshold

For a symmetric endpoint cut `c`, construct

\[
S_c(T)=\sum_{\tau=c}^{T-c}R(T,\tau),\qquad
D_c(T)=S_c(T+1)-S_c(T).
\]

Fit `D_c(T)=M` with a correlated one-parameter plateau. Label the difference
by its lower endpoint. This matters: `D(Tmin)` consumes `S(Tmin)` and
`S(Tmin+1)`, never `S(Tmin-1)`.

Use the strict inequality

\[
T_{\min}=\lfloor0.6/a\rfloor+1.
\]

The registered ensembles are:

| ensemble | a [fm] | Tmin | Tmin a [fm] |
|---|---:|---:|---:|
| L24x72 | 0.1050 | 6 | 0.6300 |
| L32x64 | 0.0897 | 7 | 0.6279 |
| L32x96 | 0.0775 | 8 | 0.6200 |

Every accepted fit range must contain the first legal difference. Scan
`c=1,2,3` and all available `Tmax` values. Require at least two difference
points for positive-dof acceptance. A single difference may exist only for the
explicit zero-dof diagnostic fallback.

## Covariance and candidate fit

Use the paired bootstrap replicas to estimate a Schaefer--Strimmer correlation
shrinkage covariance. Preserve the unshrunk sample rank, retained/effective
rank, eigenvalues, shrinkage coefficient, condition number, SVD threshold,
`chi2`, dof, `chi2/dof`, Q, and finite-replica fraction.

Candidate eligibility is fixed from the central fit:

- all selected summed-ratio and difference inputs are valid and finite;
- covariance and constant design are identifiable;
- dof is positive;
- `Q >= 0.01`;
- correlated `chi2/dof <= 3`;
- bootstrap success fraction is at least `0.90`.

Do not dynamically change the candidate grid by `z`, momentum, flavor, or
channel after examining results.

## Fit-range AIC

For candidates at fixed cut, define

\[
\mathrm{AIC}_{i,b}=\chi^2_{i,b}+2+2N_{\rm omit,i},
\qquad
w_{i,b}=\frac{e^{-\Delta\mathrm{AIC}_{i,b}/2}}
{\sum_j e^{-\Delta\mathrm{AIC}_{j,b}/2}}.
\]

The omission penalty makes nested ranges comparable without letting a shorter
range win merely by discarding points. Do not reuse the old Gaussian
log-determinant score across candidates with different data vectors; that can
produce nearly unit weights and underestimate range sensitivity.

Different cuts change `S_c(T)` and its covariance. Their likelihood
normalizations are not a common model comparison. First average ranges within
each cut, then take an equal-weight mean over usable cuts. Record usable cut
count and the cut-to-cut spread. If only one cut is usable, the statistical
result exists but cut dependence is not estimable.

## Replica selection and error

For complete diagnostic products, perform the following separately in every
replica:

- AIC-average candidates passing the replica Q and correlated-chi-square gates;
- otherwise select the finite positive-dof candidate having minimum raw
  `chi2_b`;
- if no positive-dof candidate exists, select the registered one-difference
  diagnostic.

The final uncertainty is

\[
\sigma_M=\operatorname{std}_{b=1}^{1500}(M_b,\mathrm{ddof}=1).
\]

This spread includes the replica-dependent window choice and fallback. It is
not divided by `sqrt(Nboot)`. Save bootstrap mean minus the original-data
estimate as a bias diagnostic.

## Interpretation

Use accepted fields for quality-controlled physics analysis. Filled fields are
useful when a downstream transform requires a rectangular finite array, but
they remain a sensitivity product. A fallback fraction near one means the
reported distribution is almost entirely driven by the fallback rule.

L24x72 `|P|=6` currently has only `T=6,7` valid after the ratio mask. It
therefore supplies only `D(6)` and necessarily uses the zero-dof fallback.
Increasing its fit success requires valid higher-`T` data, not a looser fit
threshold.
