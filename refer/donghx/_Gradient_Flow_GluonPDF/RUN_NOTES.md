# Execution record, 2026-11-20

- Initial tests and full task0: job `4029853`, completed on CPU Slurm (12s).
  Five contract tests passed: complex division/telescoping, physical T/all-window registry,
  correlated complex constant/full-bootstrap fallback, independent z8 anchor/quality-chain propagation,
  exact-zero variance rejection.
- Production initial array `4029855`: tasks1,2,9,10,17,18 completed on cpu074.
  Other tasks failed at launch on cpu073 without creating requested stdout/stderr files.
  This is a node-specific launch failure observation, not an established filesystem diagnosis.
- Retry array `4029877`: only failed tasks3–8,11–16,19–21, restricted to cpu074.
  No successful fit was re-fitted or overwritten; source-lock guards run on each task.
- First report attempt `4029856` also failed at launch on cpu073; retry `4029878` uses cpu074.
- When cpu074 became fully occupied, the still-pending retry tasks5–8,11–16,19–21
  and report retry were moved to i72c512g without resubmitting or duplicating fits.
  All 22 fits completed; report array `4029882` verifies/plots tasks1–21 after task0.
  Collector `4029895` is dependency-linked to the report array and runs on cpu058.
- All numerical work, original bootstrap construction checks, fitting, verification and plots
  run on CPU allocations with CUDA disabled and one BLAS thread.
- No upstream source data, fits or external Markdown files were changed.
- Final verification passed for all22 flow products, all200 paired bootstrap rows,
  all11613 candidate records and reconstruction/quality propagation. There are1855
  absolute matrix-element points:390 central quality-accepted,1465 diagnostic fallback,
  all1855 with complete200-row coverage. Final declared product SHA checks passed.
- Visually inspected the first-flow matrix image and the three-ensemble reference-flow
  contact sheet; quality and diagnostic points use filled and open markers respectively.

Reproduction (after checking available healthy CPU nodes):

```bash
sbatch --parsable --partition=a192c1t --nodelist=cpu074 --array=0-21%6 run_fit.sh
sbatch --parsable --nodelist=cpu074 --array=0-21%6 run_report.sh
sbatch --parsable run_collect.sh
```

Use afterok dependencies or wait for complete receipts before the later stages.
The versioned directory already contains the present results; do not rerun into it
when changing windows, cuts, models, data or bootstrap scope. Register a new successor instead.
