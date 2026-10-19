# τ/a² > 1.0 flow-time extrapolation

This is a seven-point linear extrapolation using exactly

\[
\tau/a^2=1.4,\;1.8,\;2.2,\;2.6,\;3.0,\;3.4,\;3.8,
\qquad M(\tau)=M_0+c_\tau\tau .
\]

The fit uses the shared 406-configuration, 1500-bootstrap ensemble
(`seed=1115`) and the same correlated GLS covariance/SVD prescription as the
earlier flow-time scans.  Errors are the standard deviation of the fitted
bootstrap intercepts (`ddof=1`).  This is deliberately labeled a
**large-flow diagnostic**, not a controlled small-flow extrapolation.

For `z/a=6`, all seven points satisfy the necessary scale-separation check
`sqrt(8 tau) < z`.  For `z/a=2` they do not: the largest flow radius is
`sqrt(30.4)=5.51a`, so the figure is unrestricted and should not be used as a
scale-separated result.

## Accepted-only results

`P/F` denotes passing/failing `Q >= 0.01`, `chi2/dof <= 3`, and finite replica
fraction >= 0.90.  A missing entry means at least one of the seven input flow
times did not have an accepted bare fit.

| z/a | operator | Pz=3 | Pz=4 | Pz=5 |
|---:|---|---:|---:|---:|
| 2 | unpolarized TH-SH | unavailable | unavailable | unavailable |
| 2 | helicity TH | -0.0350(0.0088) F | -0.0953(0.010) F | -0.106(0.015) F |
| 2 | helicity TH+SH | -0.0432(0.012) F | -0.131(0.015) F | -0.143(0.025) F |
| 6 | unpolarized TH-SH | unavailable | unavailable | +0.183(0.16) F |
| 6 | helicity TH | -0.0221(0.013) F | unavailable | -0.185(0.024) P |
| 6 | helicity TH+SH | -0.0279(0.019) F | -0.0294(0.020) F | -0.343(0.048) P |

The very small Q values for most accepted-only fits show that the seven large
flow points are not described by one common straight line when the data quality
gates are applied.

The separate `filled_fallback_diagnostic` branch supplies complete finite
diagnostic input where the accepted branch is unavailable, but it is not an
accepted fit and is not used to upgrade any result.

## Files

- `results_v1/flow_time_tau_gt_1p0_v1.npz`: numerical product, including
  intercept/slope bootstrap arrays, covariance diagnostics, geometry flags,
  and both estimator branches.
- `results_v1/flow_time_tau_gt_1p0_points.csv`: row-wise table.
- `results_v1/plots/tau_gt_1p0_fit_z6_accepted.png`: accepted-only 3x3 plot
  for the scale-compatible `z/a=6` example.
- `results_v1/plots/tau_gt_1p0_fit_z2_accepted.png`: unrestricted `z/a=2`
  plot; included only to expose the oversmearing/large-flow behavior.
- `analyze.py`, `plot.py`: reproducible analysis and plotting scripts.

The plots use error bars without joining the data points.  Solid fit lines pass
the quality gates; dashed lines are computed but fail those gates.
