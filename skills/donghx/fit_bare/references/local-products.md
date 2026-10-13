# Local code, products, and audit contract

## Authoritative input

Workspace:

`/public/group/lqcd/donghx/Diagram_GluonPDF`

Consume only corrected `ratio_v4` products with estimator id
`post_ratio_six_direction_vdv_average_v1`. They have already performed the
ratio per direction and per bootstrap replica before six-direction and VDV
averaging. Expected ratio axes are
`(boot,pabs,delta_z,tsep,tau)`. Consume `ratio_valid_mask`; common-axis padding
is NaN, not a zero measurement.

The current z-cut inputs are under:

`Fit_SKILL/artifacts/zcut_sensitivity_v1/ratios/<ensemble>/ratios`

Reject older ratio-v2 and bare-fit-v1 products because their C2 momentum axis
was wrong. Require exact even/odd alignment of configuration lists,
`boot_indices`, `tseps`, momenta, seed, and bootstrap method.

## Default implementation

Code:

`Fit_SKILL/Sumratio_Difference_AIC06`

Configuration:

`Fit_SKILL/Sumratio_Difference_AIC06/config.json`

Current complete-fallback v2 products:

`Fit_SKILL/artifacts/sumratio_difference_aic06_fallback_v2/<ensemble>`

Commands:

```bash
cd /public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Sumratio_Difference_AIC06

python run.py show-config --ensemble L32x96
python run.py audit-inputs --ensemble L32x96
python run.py fit --ensemble L32x96 --channel unpolarized --flavor u
python run.py fit-all --ensemble L32x96
python run.py plot-sixpanel --ensemble L32x96 --channel unpolarized
python audit_products.py
python -m unittest discover -s tests -v
python -m py_compile run.py audit_products.py diffaic/*.py
```

Do not overwrite a prior branch unless the user requests replacement. New
threshold, gate, candidate-grid, or fallback studies should use a new
versioned output root and sensitivity status.

## Default v2 output fields

Accepted products:

- `M`: `(component,pabs,delta_z)`;
- `M_boot`: `(boot,component,pabs,delta_z)`;
- `statistical_error`, `fit_accepted`/`fit_valid`.

Complete diagnostic products:

- `M_filled`, `M_boot_filled`, `filled_statistical_error`;
- `estimate_available`;
- `central_fallback_used`, `central_fallback_candidate_index`,
  `central_fallback_status`;
- `replica_fallback_used`, `replica_fallback_fraction`,
  `replica_minchi_candidate_index`.

Candidate/cut diagnostics include candidate values, Q, chi-square, dof,
covariance rank/condition/shrinkage, eligibility and failure reasons, AIC
weight summaries, usable cut count, and exact difference times.

Every output must also save configuration lists, bootstrap indices, seed and
count, input paths and SHA256, config/code hashes, command, axes, analysis
status, and transversity convention status.

## Other implementations

- `Fit_SKILL/Bare_Matrix_Fit`: undifferenced Sumratio slope, direct Ratio
  plateau, paired comparison, direction diagnostics, and six-panel plots.
- `Fit_SKILL/AIC_Fit`: historical Sumratio and Ratio AICc scans. Preserve these
  as sensitivity models; they are not the default estimator.

## Audit checklist

- Verify all expected channel/flavor products and completion JSON files.
- Verify NPZ hashes against completion records.
- Verify all accepted candidates contain fixed `Tmin` and no difference reads
  a lower separation.
- Verify `M_boot_filled` has 1500 finite values at every retained point.
- Verify invalid accepted outputs remain NaN.
- Verify shared configurations and bootstrap indices across all observables.
- Verify AIC weights normalize within each comparable fixed-cut family.
- Count central and replica fallbacks, including zero-dof cases.
- Plot accepted, single-cut, and fallback points with distinct markers.

As audited on 2026-09-03, v2 contains 27 NPZ products, 27 completion records,
and nine PNG/PDF six-panel pairs. All 7,830,000 filled bootstrap values are
finite. There are 4,787 accepted central points and 433 central fallbacks; 288
fallbacks are the L24x72 `|P|=6` zero-dof single-difference diagnostic. Re-run
the audit rather than quoting these counts after any input or code change.
