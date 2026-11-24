---
name: lqcddb
description: Use and review lattice-QCD distillation, Wick, dynamic-contraction, MPI, I/O, and analysis APIs from a selected lqcddb source tree (currently lqcddb_update). Explain physics and tensor axes; route complete contraction-program generation to contraction-creator.
metadata:
  short-description: lqcddb API use and physics guidance
---

# lqcddb API Skill

## Source selection interface

Resolve `source_root` once per task: an explicit user-selected library directory takes priority, then the environment variable `LQCDDB_SOURCE_ROOT`, then the default `/public/home/sush/distillation/lqcddb_update`. Record the resolved path. A later switch can point this interface at `/public/home/sush/distillation/lqcddb_new`; do not migrate files, rename imports, or mix the two trees now.

Read the selected root's `README.md` and `physics.md` and the relevant definitions/exports in `src/lqcddb/` and `__init__.pyi`. Actual implementation and exports are authoritative. If the selected layout is missing or different, report it and locate/resolve it rather than silently falling back to another version. Existing API references and `current-source-status.md` describe `lqcddb_update`; recheck relevant claims after selecting another root. The directory name does not establish the import name or API compatibility. For executable checks verify the imported module's `__file__` belongs to the selected tree.

## Boundary with contraction generation

For complete contraction programs use `$contraction-creator` for the required-input gate, reference selection, byte-preserving copy, edit limits and output structure. This skill supplies API/physics facts; it must not replace that reference with `examples/*_flat.py`, rewrite imports/backend, insert manifests or add runtime validation. The legacy flat examples are API illustrations only. Keep review results outside a copied production file; report actual defects/incompatibilities and use the smallest justified repair. Reference fidelity does not certify physical correctness.

## Establish the physics-to-array correspondence first

For every task, make the following explicit:

$$
\text{hadron/current definition}\to\text{Euclidean convention}\to\text{Wick diagram}\to\text{registry tensor axes}\to\text{output axes}\to\text{estimator}.
$$

Preserve the gamma basis, quark flavors, operator order, coefficients, momentum signs, temporal boundary, and complex correlator specified by the user. Do not change conventions because a result looks reasonable. The hadron terms in `physics.md` are the physical expressions; preserve every coefficient and gamma order when writing Wick tokens.

## API routing

When you need function signatures, parameters, or minimal examples, read [references/api-quick-reference.md](references/api-quick-reference.md). Read only the relevant sections for the task: basic tools (backend, momentum, slicing, contraction cache), constants (gamma, sigma, and physical constants), contractions (conjugation, Wick, dynamic registries), eigenvectors/I/O (eigenvectors, vertices, file I/O), analysis (resampling, source averaging, effective masses, ratios, GEVP), and MPI (time mapping and data movement).

## Most important call rules

For standalone API explanations or small probes, use named imports and set the backend explicitly. When adapting a production reference, preserve its existing import style/backend unless the requested target or a verified compatibility issue requires a change:

```python
from lqcddb import set_backend, get_backend
set_backend('numpy')                 # Use cupy only when requested or required by measurement.
backend = get_backend()
```

Use `conjugate_operator(tokens)` for operator conjugation; it reverses multi-hadron products, conjugates numerical coefficients, and handles meson/baryon gamma chains. Use `wick_contraction(sink, source, curr, Cpt=...)` for Wick contraction, with `curr=[]` for 2pt; wildcards return multiple flavor-conserving substitutions. Preserve a user-provided or explicitly inherited reference `source_operator` verbatim after syntax and flavor-conservation checks.

The required dynamic-contraction API operations are below. Registration order is not prescribed; preserve the reference's order and ensure required arrays are registered before execution. Plan-only analysis can run before data loading:

1. For each perambulator time direction, call `PeramRegistry.register(flavor, (t_quark, t_antiquark), array)`; the array must have at least four dimensions `(Ns,Ns,Nev,Nev)`.
2. For each Wick output `VVV_0/VDV_0` and time end, call `VRegistry.register(name, time_label, array)`; the first axis is usually momentum.
3. For each gamma name, call `GammaRegistry.register(name, matrix)`; for projections, register `Projector=[proj_sink, proj_src]`.
4. Obtain the plan with `run_wick_analysis(operator_groups, Cpt=..., ignore_dis=False, use_equivalence=False)`; confirm registrations and output indices before calling `calculate_contraction(entry, ...)` or creating `dynamic_contraction(...).calculate(i)`.

For a new observable, `ignore_dis=False` is the physics starting point; a shared insertion time does not prove that a diagram is disconnected. Use `use_equivalence=False` unless gamma-transpose signs, VVV antisymmetric permutations, topology, time direction, and output axes have been proved identical. For reference reproduction preserve its explicit flags and report unsupported assumptions rather than silently resetting them. `validate_plan` checks registry lookup only; it does not replace an independent small-tensor `opt_einsum`/NumPy contraction.

## Array and time conventions

Write an axis table for registered tensors before writing einsum expressions; lowercase symbols usually denote spin/internal indices, while uppercase symbols are external indices introduced by `Pindex/Vindex/Gindex`, and the final axis order is the right-hand side of `result_indx` after `->`. Verify time ownership against `get_mpi_tlist` in the selected source. The current cyclic mapping uses `t % size` and `(t % Nt) // size`; retain reference expressions when their domain and decomposition make them equivalent. Verify Scatter/TScatter/Gather/Bcast against the actual mapping and a wraparound-time marker; time-list aliases do not imply identical data operations.

Perambulators are antiperiodic by default. The sign of a complete fermion line crossing the temporal boundary is determined by the number of lines that cross it and must be applied exactly once. `loop_tsrc`'s `Boundary_Conditions` applies to the complete correlator; 3pt also requires `Ctype='3pt'` and `t_sep`. In the current source it sums source times without dividing by their count; do not silently turn the estimator into a mean.

## Scientific review and delivery

Every review or explanation must report source location/signature, physical conventions, operator coefficients and flavor conservation, each tensor's shape/axes, independent invariants or a small reference contraction, MPI/GPU/data paths that were not run, and changed files. Unless explicitly requested, do not modify lqcddb source, add meaningless tests, or generate JSON manifests. Use `$contraction-creator` for complete contraction programs and `$analyse-hardon` for data plots and fits.
