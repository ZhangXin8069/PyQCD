---
name: analyse-hardon
description: Build and use hadron correlator analysis programs bound to lqcddb, covering Jackknife/Bootstrap, source-time averaging, effective masses, three-point ratios, GEVP, error-bar plots, and lsqfit fits.
metadata:
  short-description: Hadron correlator analysis and fitting
---

# Hadron Correlator Analysis

This skill focuses on analysis programs and does not generate Wick/dynamic contraction code. It is bound to `/public/home/sush/distillation/lqcddb_update/src/lqcddb/analyse/analyse.py` and refers to `/public/home/sush/distillation/analyse/pion_form_factor/analyse.py` and `/public/home/sush/distillation/analyse/0v2b/analyse.E32P29.NJNp-.3.py`. Before writing analysis code, read the `lqcddb` README/physics documentation and the current analysis source to confirm data axes, boundary conditions, normalization, and units.

## Data flow

Make the following flow explicit:

$$
\text{raw correlator for each configuration}\to\text{source-time averaging}\to\text{Jackknife/Bootstrap}\to\text{effective mass/GEVP/ratio}\to\text{per-configuration fits}\to\text{plots and physical quantities}.
$$

Common entry points are `loop_tsrc`, `Jackknife`, `Bootstrap`, `meff`, `ratio_3pt`, `solve_gevp`, `Mom2GeV`, `mean_over_array_of_list`, and `sum_over_array_of_list`. Before calling them, write down the actual `(Nconf, ..., t_src, t_sink[, t_insert])` shape; do not infer axes from filenames.

Use `loop_tsrc(data, indx=[src_axis, sink_axis], Boundary_Conditions=..., Ctype='2pt'|'3pt', t_sep=...)` for source-time averaging. Apply the antiperiodic phase exactly once on the wraparound path of the complete correlator. `Jackknife(..., only_sample=True)` is suitable for later per-configuration processing; use `cov_axes` or explicitly construct a Hermitian covariance when fitting requires a covariance matrix. `meff` returns `data_sample/data_mean/data_err/valid_mask`; invalid points remain NaN.

## Fitting conventions

Use `lsqfit.nonlinear_fit` and `gvar` by default. First scan valid windows on the mean data, then refit the selected window for each configuration (or Jackknife sample), saving samples, means, and errors for every parameter; do not fit only the full-ensemble mean and treat its error as a configuration error. Select windows primarily by `chi2/dof` near 1, while also checking `dof>0`, `Q`, parameter priors, and physical stability. Record and explain failed configuration fits; never silently replace them with zeros.

For two-point functions, use a one-state/two-state exponential or a backward-propagating hyperbolic-cosine model; the model `Nt`, boundary condition, and normalization must match the data. For three-point functions, first use `ratio_3pt` to specify the initial/final two-point functions and `t_sep`, then fit an insertion-time plateau or a physical model. Before GEVP, ensure that `C(t0)` is Hermitian positive definite; after calling `solve_gevp`, record eigenvalue/eigenvector ordering and the reference time to avoid mismatching states.

## Plotting conventions

Unless the user explicitly requests another graphic, plot statistical data with `matplotlib.axes.Axes.errorbar`, showing `yerr`, labels, units, and the fit window. Use different colors, markers, or line styles for different momenta, operators, states, projections, or fit schemes, and state the physical meaning in the legend; do not overlay nearly identical colors. Fit curves and plateau bands may use `plot`/`fill_between`, but retain error bars on the original statistical points. Annotate `chi2/dof`, the window, and unit conversions such as `fm2GeV/alttc`.

## Minimal program structure

```python
from lqcddb import set_backend, get_backend, loop_tsrc, Jackknife, meff
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt

set_backend('numpy')
backend = get_backend()
# Load data here and document the configuration shape and every axis.
# data_loop = loop_tsrc(...)
# jk = Jackknife(data_loop, Nconf_axes=0, only_sample=True)
# Fit the mean first, then fit each configuration with the same window.
# ax.errorbar(t, mean, yerr=err, marker='o', linestyle='none', label='...')
```

Do not hide the complete data-loading, fitting, and plotting workflow inside an opaque general framework; keep clear modules corresponding to the target physical quantities. Do not add meaningless tests or checks. See [references/workflow.md](references/workflow.md) for detailed function signatures, fit windows, and plotting templates.

When delivering results, report the data path and configuration list, every array axis, normalization/boundary conventions, the selected window and `chi2/dof`, the number of per-configuration failures, plot files, and paths that were not run. If the result depends on existing contraction files, link their physical definition and output axes.
