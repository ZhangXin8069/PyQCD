---
name: contraction-creator
description: Derive required variables with dynamic_contraction first, then create contraction programs using the fixed M0–M8 modules from the two NA800 CuPy/MPI reference files, recording stage and total runtimes.
metadata:
  short-description: Minimal dynamic-contraction code generation
---

# Contraction Program Builder

This skill is paired with `$lqcddb`. Use `$lqcddb` for API facts, operator physics, and registry semantics; this skill only turns the target into a contraction program. The reference files are the sole minimal implementation units:

- 2pt: `assets/contraction.NN.I0.5.2pt.cupy.py`;
- 3pt/GEVP: `assets/contraction.NJNp-.3pt.GEVP.cupy.na800.py`.

Read the selected file in full before generating code, and interpret it as fixed top-level modules `M0` through `M8` according to [references/module-layout.md](references/module-layout.md). See [references/submission-script.md](references/submission-script.md) for the paired submission-script rules. The modules are stable script blocks, not a new runtime framework. Preserve the order of imports, MPI/CuPy initialization, parameters, phases, eigenvector/VDV/VVV construction, perambulator distribution, registry registration, time/momentum loops, `dynamic_contraction` calls, `TGather`/`loop_tsrc`, and saving. Do not introduce `main`, `run`, loaders/factories, schedulers, physically meaningless helper functions, tests, or extra abstraction layers; do not merge the two templates.

## Build workflow

1. From the user target and `physics.md`, determine the sink, physical initial state, currents, flavors, gamma matrices, momenta, `Cpt`, time separation, and output axes. Pass a user-provided source operator through unchanged; otherwise use `conjugate_operator(initial_operator)` and preserve its coefficient.
2. Complete the `M3 dynamic_manifest` first: use the `run_wick_analysis` core called by `dynamic_contraction` to find every peram, V, gamma, Projector, time label, and output axis, then decide what other modules must load. While registries are not ready, call only this analysis core; execute the actual `dynamic_contraction(...).calculate_all()` only in M7. Each 2pt group must be exactly `[sink, source]`, and each 3pt group exactly `[sink, source, current]`.
3. Fill `M0`–`M2` and `M4`–`M6` in order, obtaining only the arrays required by the M3 manifest; register them using the owner/local-index and backend/dtype rules already present in the template.
4. In `M7`, keep the common `dynamic_contraction(...).calculate_all()` shape, changing only operator groups, current-time/momentum slices, projections, and result axes. In `M8`, retain gathering, `loop_tsrc`, and saving, and print stage and total runtimes. Then copy the same-named `.sh` from the submission-script asset and modify only the partition, resources, configuration variables, and four path variables allowed by its whitelist.

## Physics constraints during generation

- Start with `ignore_dis=False` and `use_equivalence=False`; change them only with a graph-theoretic and tensor-identity proof.
- Check quark/antiquark conservation, coefficients, gamma order, current flavor, and source/sink routing flavor by flavor.
- Derive the antiperiodic wrap sign from the number of fermion lines crossing the boundary and apply it exactly once; preserve the complex correlator.
- Obtain the global-time to `(owner, local_index)` mapping from `get_mpi_tlist`; use local indices through the template's Scatter/Bcast/TGather pattern and never treat global time as a local array index.
- Before generation, compare an actual Wick plan with registry keys and einsum axes; use `validate_plan` only to report missing registrations from the existing API, and do not add test code to the script.
- GEVP projection matrices must be Hermitian positive definite; do not change the physical lattice extent for a performance suggestion.
- Every generated script must have the unified `M0` total timer and print vertex, peram, contraction-stage, and total runtimes on the root process.

## Deliverables

Deliver only the minimal script requested by the user and its same-named submission `.sh`. Preserve the fixed Python text/structure from `M0`–`M8`; put necessary physical correlators, operator coefficients, registry keys/shapes/time axes, and output axes in short comments in `M3` or the corresponding module. Modify the submission script only within the whitelist in [references/submission-script.md](references/submission-script.md). Preserve differences from the original template; report GPU, full-ensemble, or multi-process paths that were not run. Explain every structural change and compare values before and after it. Do not add test files, manifest frameworks, or meaningless checks.
