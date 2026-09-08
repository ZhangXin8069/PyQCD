# Local pipeline and product contract

Read this reference before changing or running the local C3/ratio/bare-fit
workflow.

## Authoritative roots

Workspace:
`/public/group/lqcd/donghx/Diagram_GluonPDF`

Code:

- C3 projection and ratio bootstrap:
  `Fit_SKILL/C3_Ratio_Bootstrap`
- bare-matrix-element fits:
  `Fit_SKILL/Bare_Matrix_Fit`

Current artifacts:

- projected C3: `Fit_SKILL/artifacts/c3_ratio_v3/<ensemble>/`
- ratio: `Fit_SKILL/artifacts/c3_ratio_v4/<ensemble>/ratios/`
- fits: `Fit_SKILL/artifacts/bare_matrix_fit_v3/<ensemble>/`

Never use `c3_ratio_v2` or `bare_matrix_fit_v1` for physics.  The old C2
expression `c2[ipol, :, tsrc, tsink]` triggered NumPy advanced-index axis
movement and the later mean removed momentum.  The corrected two-stage form is
`c2[ipol][:, tsrc, tsink]`, with a hard `(Nmom,)` result-shape check.

## Ensemble coverage

| ensemble | Nconf | tsep | delta-z | pabs |
|---|---:|---|---|---|
| L24x72 | 199 | 2--12 | 0--15 | 0,3,4,5,6 |
| L32x64 | 425 | 5--10 | 0--20 | 0,3,4,5,6 |
| L32x96 | 194 | 5--12 | 0--20 | 0,3,4,5,6 |

Use the configured explicit configuration lists.  Historical exclusions for
L24x72 and L32x96 have incomplete recorded reasons; do not restore excluded
configurations silently.

## Estimator and storage

For each bootstrap replica and direction, compute

```text
bootstrap_cfg_mean(mean_tsrc(C3_dir)) /
bootstrap_cfg_mean(mean_tsrc(C2_dir))
```

before VDV symmetry and direction averaging.  Do not replace this with
`mean_tsrc(C3/C2)` or a ratio of separately direction-averaged correlators.
The required estimator id is
`post_ratio_six_direction_vdv_average_v1`.

The format uses the compact tau scheme:

- projected C3 stores `tau=0,...,tsep` after source averaging;
- ratio axes are `(boot,pabs,delta_z,tsep,tau)`;
- a common tau axis is accompanied by validity masks;
- nonphysical and denominator-rejected entries are NaN;
- bootstrap indices have shape `(Nboot,Nconf)` and are stored once.

The current production contract is iid configuration bootstrap,
`Nboot=1500`, `seed=1115`, and resample size `Nconf`.  All observables in an
ensemble must share exactly the same index array.

## Commands

From `Fit_SKILL/C3_Ratio_Bootstrap`:

```bash
python3 run.py show-config --ensemble L32x64
python3 run.py validate-conventions --ensemble L32x64
python3 run.py audit-inputs --ensemble L32x64
python3 run.py bootstrap --ensemble L32x64 \
  --channels unpolarized,helicity \
  --flavors u,d,u_minus_d \
  --nboot 1500 --seed 1115 --c2-mode complex
python3 -m unittest discover -s tests -v
bash -n slurm/*.sh slurm/*.slurm
```

From `Fit_SKILL/Bare_Matrix_Fit`:

```bash
python3 run.py audit-inputs --ensemble L32x64 \
  --channel helicity --flavor u_minus_d
python3 run.py fit --ensemble L32x64 \
  --channel helicity --flavor u_minus_d
python3 run.py fit-ratio --ensemble L32x64 \
  --channel helicity --flavor u_minus_d
python3 run.py compare --ensemble L32x64 \
  --channel helicity --flavor u_minus_d
python3 -m unittest discover -s tests -v
python3 -m py_compile run.py barefit/*.py
bash -n slurm/*.sh slurm/*.slurm
```

Use `--allow-quality-failures`, `--allow-no-valid-results`, or
`--allow-pending-transversity` only to preserve diagnostics.  These flags do
not change validation status.

## Provenance and completion checks

Before declaring production complete, verify:

- exact ensemble/configuration list and `Nconf`;
- axis labels and shapes;
- shared bootstrap indices, seed, and `Nboot`;
- validity masks and NaN placement;
- input and output SHA256 against completion JSON;
- estimator id, code/config hashes, command, and product status;
- no non-warning content in error logs;
- all expected channel/flavor/even/odd files exist.

For comparison products require `comparison_pipeline_version >= 1.1.0`,
`axes_paired_difference_boot=(boot,component,pabs,delta_z)`, and 1500 finite
paired replicas where both fits are valid.  Earlier comparisons indexed the
component-first fit arrays incorrectly and their quoted consistency rate is
invalid.
