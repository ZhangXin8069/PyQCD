# Required input contract

Collect this contract **before generation**, not through a new CLI inside the generated program. Preserve the reference's `sys.argv[1]` configuration interface. Natural-language input is sufficient; do not require JSON or a manifest.

## Required for every generation request

| Field | Sufficient information | Why it matters / existing code |
| --- | --- | --- |
| Reference and scope | Named bundled profile or specific reference path; creation/adaptation or explicit copying | Determines the preserved text/structure; “same as reference” needs an unambiguous file |
| Physical process | Correlator order, initial state, final sink, all operator-basis components/coefficients; source tokens or enough conventions to derive the conjugate initial operator | `sink_operator`, `source_operator`, `Cpt`; a hadron name may not fix isospin coupling or basis |
| Ensemble and configurations | Ensemble ID (or explicit geometry/ensemble directory), plus configuration IDs or inclusive start/end/stride, or a complete Slurm array formula | `lattice_size`, `conf_id`, `conf_start/gap/gap_2/--array`; an ensemble label is not a configuration number |
| Eigenvectors / distillation | `Nev_src`, distinct sink/current cutoff (`Nev_max` in 3pt), stored dimensions and filename/format; exact vs stochastic/blended construction | Shapes, truncation, files and omega; “eigen=100” does not determine a 400-dimensional current port |
| MPI / backend | MPI rank count, process grid, CuPy/NumPy, device allocation; derive the grid only from an explicitly inherited decomposition | `grid_size`, `mpinit`, transfers, `mpirun -n`; GPU count and MPI count are separate |
| Momenta | Ordered source/sink/current momentum sets or explicit reference setting, Fourier convention, momentum-sharing relationships | Momentum lists, phases, `Vindex` and saved axes; different order is a different input |
| Times | Source-time list/stride; 2pt sink/separation range; 3pt separation(s) and current-time range with contact-point policy | Source/sink/current loops, `tsep`, `loop_tsrc` |
| Conventions and output | Boundary, source sum vs mean vs retained source axis, spin projection and observable axes, flavor-degeneracy and diagram-selection assumptions | Gamma/Projector, `ignore_dis`, `use_equivalence`, complex output and normalization |
| Paths | Eigenvector/peram paths and flavor mapping, required gauge/links, result root/filename, generated Python/shell destination | Existing read/save expressions and shell paths; an empty `physics.md` path is not usable |

For the paired `.sh`, also resolve partition/controller, nodes, tasks per node, CPUs per task, GPU allocation, applicable wall-time setting, environment-loading command, `run_dir`, and configuration-to-array mapping. Explicitly inheriting a shell profile supplies these fields. A scheduler default can be inherited explicitly; do not invent a time limit or choose a production partition. Workspace **validation job** defaults are not production-output defaults.

## Conditional fields

- **3pt and higher:** every current's flavor direction, ordered gamma product, coefficients/components, placement/time ordering, and connected/disconnected selection. In the bundled 3pt case, `gamma(mu) @ gamma(5)` must not become `gamma(5) @ gamma(mu)` based only on an axial-current label.
- **Multiple hadrons/basis:** isospin/Clebsch–Gordan coefficients, component order, per-hadron momenta and source conjugation. A filename alone is insufficient.
- **Stochastic/blended eigenvectors:** exact part, `N_eigen`, `N_sum`, `N_extract`, noise count, stored layout and omega application. These specify the estimator. Files may establish stored metadata; the requested truncation still needs a clear input.
- **Nonlocal/derivative currents:** link directions/lengths, gauge path, smearing and gauge boundary conventions.
- **Actual GEVP projection:** supplied basis/projection data and axes; if solving is requested, reference time, method and retained subspace. Do not infer an eigensolver request from `GEVP` in a filename.
- **Multiple currents:** all insertion axes/orderings and source-relative shifts. A 3pt template and single `loop_tsrc` call do not automatically implement higher-point functions.

## Inheritance and questions

Inputs may come from (1) current/earlier user answers, (2) an explicitly inherited named reference/profile, or (3) a unique lookup from those inputs, such as E32P29 geometry in the selected `physics.md`. Record values and provenance in the task log. Do not assume an unspecified value equals an available example. Do not repeat questions already answered.

“全部与 `njn-3pt` 内置参考一致，只把组态改为 1040” inherits its physical/numerical settings and compatible shell profile. It changes the configuration mapping, not the Python interface. Ask for an output destination if none is established. “计算 E32P29 的三点函数，100 个 eigen，2 个 MPI” is incomplete: ask together for process/current, configurations, distinct current cutoff, momenta/times, paths and other unresolved fields.

Example question: “还缺少具体组态编号、初末态与流的味/Gamma 定义、流端 eigen 数以及动量/时间范围。请补充；若采用内置 `njn-3pt`，也可说明哪些项目全部沿用该参考。输出目录也请指定。” Adapt this to the actual missing fields. While answers are pending, inspect read-only but do not create final `.py`/`.sh` files.

## Consistency checks

- Positive integer cutoffs/ranks and compatible geometry/grid. Existing templates require temporal-only `grid_size=[1,1,1,nrank]`, `Nt % nrank == 0`, and `prod(grid_size) == nrank == --ntasks == mpirun -n`. Spatial decomposition needs a justified extension.
- Match every eigenvector port to stored dimensions. Do not change `Nev_max`, stochastic metadata or filenames simply to fit a requested `Nev_src`.
- Expand the array formula to exactly the requested configurations, including endpoints/stride. Never round an irregular list into a range.
- Check each operator combination's flavor balance; distinguish explicit source tokens from an initial operator needing conjugation.
- Check ordered momenta, time windows, axes and normalization. The current `loop_tsrc` sums source times without dividing by their count.
- Resolve reference/source/path/resource conflicts before generation. Matching a few headline parameters does not establish exact mode.
