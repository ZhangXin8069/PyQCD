# Paired submission script

Use the shell paired with a user-specified reference when available. Bundled Python assets share `assets/contraction.NJNp-.3pt.GEVP.cupy.sh`. The external historical pair is `/public/home/sush/distillation/0v2b/contraction.NJNp-.3pt.GEVP.cupy.py` and its `.sh`; it differs from the bundled snapshot and is not its automatic replacement.

Deliver `<target>.py` and `<target>.sh` unless Python-only is requested. The bundled `.na800.py` asset need not determine the output basename: the 3pt shell targets `contraction.NJNp-.3pt.GEVP.cupy.py`, whose contents can be identical to that asset. A different requested basename changes only the existing shell path values.

## Required settings and allowed patches

Resolve settings from the input contract, earlier answers, or an explicitly inherited profile. Preserve other text, blank lines, environment setup, checks, redirection, return handling and cleanup.

1. Partition/controller and resource directives: `--array`, `--nodes`, `--ntasks`, `--ntasks-per-node`, `--cpus-per-task`, `--gres`, `--gpus`, `--time`; matching `mpirun -n`.
2. Configuration selection: `conf_start`, `gap`, `gap_2`, existing loop range/formula. An unrepresentable irregular list needs an explained small extension, never a substitution of configurations.
3. Paths: `run_dir`, `exe`, `dir_file`, `output_file`.
4. Environment loading or other text only for an explicitly supplied different setting or demonstrated compatibility repair; record the exception. An API-source selector alone does not request adding `PYTHONPATH` or a CLI to the generated file.

Keep job name, scheduler log directives, `UCX_TLS`, launch arguments, `--bind-to core --map-by core`, and cleanup unchanged without such an exception. Do not infer a production partition from a test partition or asset filename.

## Checks

- `prod(grid_size) == --ntasks == mpirun -n`; match tasks/node, nodes and GPUs. The maintained bundled profile uses two ranks; the historical external shell used eight.
- `exe` targets the delivered Python basename; `dir_file` and `output_file` match its name/configuration. Verify the intended `run_dir` even if files are prepared elsewhere.
- Expand the array/loop formula. Bundled `--array=0-49`, `conf_start=1000`, `gap=40`, `gap_2=40`, `i=0` means configurations 1000 through 2960 inclusive, step 40.
- Identical paths/resources/configurations require byte equality. Otherwise enumerate and verify replacements using `reference_copy.py`, then review the whitelist diff.
- Run `bash -n`; creating the pair does not request submission.
