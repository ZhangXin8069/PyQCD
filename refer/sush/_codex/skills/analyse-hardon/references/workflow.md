# Hadron Analysis Workflow Reference

## lqcddb functions

- `loop_tsrc`: converts source/sink times to relative time and averages over source time; the 2pt wraparound condition is determined by `tsink < tsrc`, and the 3pt condition by `tsrc+t_sep`.
- `Jackknife`: takes a configuration axis and uses `data_sample=(sum-data)/(Nconf-1)`; with `only_sample=True`, it returns samples only.
- `Bootstrap`: `M` is the number of configurations per resample, `N` is the number of repeated samples, and `seed` fixes reproducibility.
- `meff`: `log`/`GEVP` use adjacent-time ratios, while `cosh` uses a three-point `arccosh`; the result includes `valid_mask`.
- `ratio_3pt`: first check the configuration, source/sink/insertion axes, and `t_sep` of all three inputs, then confirm the momenta and normalization of the initial and final two-point functions.
- `solve_gevp`: takes the correlator time axis and `t0`; first hermitize the input and check that `C(t0)` is positive definite, then record state ordering.

## Minimal `lsqfit` pattern

```python
def c2_model(t, p, Nt):
    return p['Z']**2 * (gv.exp(-p['E']*t) + gv.exp(-p['E']*(Nt-t)))

def fit_window(samples, tmin, tmax, prior, Nt):
    y = samples[:, tmin:tmax]
    mean = backend.mean(y, axis=0)
    cov = backend.einsum('ci,cj->ij', y-mean, backend.conj(y-mean)) * (y.shape[0]-1)/y.shape[0]
    t = backend.arange(tmin, tmax, dtype=float)
    fit = lsqfit.nonlinear_fit(
        data=(t, gv.gvar(mean, cov)),
        fcn=lambda x, p: c2_model(x, p, Nt), prior=prior)
    return fit
```

When scanning windows, skip `dof<=0` and fit failures, select candidates by `abs(chi2/dof-1)`, and then check window stability. After selecting a window, call the same `lsqfit.nonlinear_fit` for every configuration or Jackknife sample; collect `fit.p[name].mean` and compute errors with the Jackknife or Bootstrap rule. For complex data, first establish from the physical projection that the real or imaginary part may be taken; otherwise fit the complex correlator.

## Error-bar plots

```python
for i, label in enumerate(labels):
    ax.errorbar(t + 0.04*i, mean[i], yerr=err[i],
                marker=markers[i], color=colors[i], linestyle='none',
                capsize=2, label=label)
ax.set_xlabel(r'$t/a_t$'); ax.set_ylabel(r'$C(t)$')
ax.legend(); fig.tight_layout()
```

Every variable group must have a distinguishable marker, color, or horizontal offset. Fit lines for the same physical quantity should use the corresponding color and show the window and `chi2/dof`.
