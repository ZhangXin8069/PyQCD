---
name: lqcddb
description: Use and explain the lattice-QCD distillation, Wick, vertex, dynamic-contraction, MPI, I/O, and statistical-analysis APIs in lqcddb_update; suitable for API selection, parameter/array-axis explanations, and physics-consistency review, but not for generating complete contraction programs.
metadata:
  short-description: lqcddb API use and physics guidance
---

# lqcddb API Skill

This skill works only from the current source at `/public/home/sush/distillation/lqcddb_update`. First read that directory's `README.md` and `physics.md`, then use the actual definitions in `src/lqcddb/` and `__init__.pyi` as the authority; use the README, old reports, and installed copies only as background. Do not replace these API descriptions with names or signatures from an older version.

## Establish the physics-to-array correspondence first

For every task, make the following explicit:

$$
\text{hadron/current definition}\to\text{Euclidean convention}\to\text{Wick diagram}\to\text{registry tensor axes}\to\text{output axes}\to\text{estimator}.
$$

Preserve the gamma basis, quark flavors, operator order, coefficients, momentum signs, temporal boundary, and complex correlator specified by the user. Do not change conventions because a result looks reasonable. The hadron terms in `physics.md` are the physical expressions; preserve every coefficient and gamma order when writing Wick tokens.

## API routing

When you need function signatures, parameters, or minimal examples, read [references/api-quick-reference.md](references/api-quick-reference.md). Read only the relevant sections for the task: basic tools (backend, momentum, slicing, contraction cache), constants (gamma, sigma, and physical constants), contractions (conjugation, Wick, dynamic registries), eigenvectors/I/O (eigenvectors, vertices, file I/O), analysis (resampling, source averaging, effective masses, ratios, GEVP), and MPI (time mapping and data movement).

## Most important call rules

Start with named imports and set the backend explicitly:

```python
from lqcddb import set_backend, get_backend
set_backend('numpy')                 # Use cupy only when requested or required by measurement.
backend = get_backend()
```

Use `conjugate_operator(tokens)` for operator conjugation; it reverses multi-hadron products, conjugates numerical coefficients, and handles meson/baryon gamma chains. Use `wick_contraction(sink, source, curr, Cpt=...)` for Wick contraction, with `curr=[]` for 2pt; wildcards return multiple flavor-conserving substitutions. Preserve a user-provided `source_operator` verbatim after syntax and flavor-conservation checks.

The minimal dynamic-contraction order is:

1. For each perambulator time direction, call `PeramRegistry.register(flavor, (t_quark, t_antiquark), array)`; the array must have at least four dimensions `(Ns,Ns,Nev,Nev)`.
2. For each Wick output `VVV_0/VDV_0` and time end, call `VRegistry.register(name, time_label, array)`; the first axis is usually momentum.
3. For each gamma name, call `GammaRegistry.register(name, matrix)`; for projections, register `Projector=[proj_sink, proj_src]`.
4. Obtain the plan with `run_wick_analysis(operator_groups, Cpt=..., ignore_dis=False, use_equivalence=False)`; confirm registrations and output indices before calling `calculate_contraction(entry, ...)` or creating `dynamic_contraction(...).calculate(i)`.

`ignore_dis=False` is the default physics starting point; a shared insertion time does not prove that a diagram is disconnected. Use `use_equivalence=False` unless gamma-transpose signs, VVV antisymmetric permutations, topology, time direction, and output axes have all been proved identical. `validate_plan` checks registry lookup only; it does not replace an independent small-tensor `opt_einsum`/NumPy contraction.

## Array and time conventions

Write an axis table for every registered tensor before writing einsum expressions; lowercase symbols usually denote spin/internal indices, while uppercase symbols are external indices introduced by `Pindex/Vindex/Gindex`, and the final axis order is the right-hand side of `result_indx` after `->`. `get_mpi_tlist` returns the global-time to `(rank,local_index)` mapping; do not infer it with `tsrc//size`. Verify `get_mpi_data` Scatter/TScatter/Gather/Bcast operations against the actual mapping and a wraparound-time marker.

Perambulators are antiperiodic by default. The sign of a complete fermion line crossing the temporal boundary is determined by the number of lines that cross it and must be applied exactly once. `loop_tsrc`'s `Boundary_Conditions` applies to the complete correlator; 3pt also requires `Ctype='3pt'` and `t_sep`.

## Scientific review and delivery

Every review or explanation must report source location/signature, physical conventions, operator coefficients and flavor conservation, each tensor's shape/axes, independent invariants or a small reference contraction, MPI/GPU/data paths that were not run, and changed files. Unless explicitly requested, do not modify lqcddb source, add meaningless tests, or generate JSON manifests. Use `$contraction-creator` for complete contraction programs and `$analyse-hardon` for data plots and fits.
