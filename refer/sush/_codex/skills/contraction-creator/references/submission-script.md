# Contraction Submission-Script Rules

This skill uses only `/public/home/sush/distillation/0v2b/contraction.NJNp-.3pt.GEVP.cupy.py` as an external contraction-code reference and its paired `/public/home/sush/distillation/0v2b/contraction.NJNp-.3pt.GEVP.cupy.sh` as the submission-script reference. Do not read or cite other contraction programs in `0v2b`; the contraction code itself is generated from this skill's assets and M0–M8 rules.

## Paired outputs

Whenever a contraction Python file is generated, generate a same-named `.sh` file. Copy the submission script from `assets/contraction.NJNp-.3pt.GEVP.cupy.sh`, preserving line order, blank lines, checks, environment setup, directory handling, GPU checks, `mpirun` options, success/failure handling, and cleanup logic. Keep the same stem as the Python file, for example:

```text
contraction.<target>.py
contraction.<target>.sh
```

The script's `exe` must point to the generated Python file; `dir_file` and `output_file` must use the same target name and configuration variables.

## Only allowed modifications

Except for replacing values needed to adapt the generated filename during copying, do not add, delete, move, or reorder script lines:

1. Partition: the enabled `#SBATCH --partition=...` value. Keep other candidate partition lines commented out.
2. Compute resources: `--array`, `--nodes`, `--ntasks`, `--ntasks-per-node`, `--cpus-per-task`, `--gres`, `--gpus`, `--time`, and the `mpirun -n` directly tied to resources.
3. Configuration variables: `conf_start`, `gap`, `gap_2`, configuration-loop ranges, and numeric values in the existing `conf` expression.
4. Path variables: `run_dir`, `exe`, `dir_file`, and `output_file`.

Keep `job-name`, `--error`, `--output`, environment variables, environment-loading commands, `UCX_TLS`, directory checks, GPU checks, Python launch arguments, output redirection, return-value handling, cleanup commands, and all other text identical to the reference script. Unless the user explicitly requests it, do not modify these items or add parameters, loops, checks, dependencies, or log formats.

## Pre-generation checks

- The Python file and submission script have the same stem, and `exe`, `dir_file`, and `output_file` all agree.
- `mpirun -n` matches the allowed `--ntasks` setting; do not change `--bind-to core --map-by core`.
- The configuration formula still uses the existing `SLURM_ARRAY_TASK_ID` and `conf_start/gap/gap_2` relation, changing only the values requested by the user.
- Only whitelisted fields differ; use a line-by-line diff to confirm that the remaining text matches the asset.
- Deliver both files, do not submit the job, and do not change the partition, resources, or configuration range on the user's behalf.
