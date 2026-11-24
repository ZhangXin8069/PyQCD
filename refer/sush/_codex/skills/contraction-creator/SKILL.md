---
name: contraction-creator
description: Create lattice-QCD contraction Python and Slurm scripts from a selected reference with minimal input-driven edits and stage/loop timing. Require physics, configuration, eigenvector, MPI, and I/O inputs; ask when missing. Preserve reference structure and verify declared differences.
metadata:
  short-description: Reference-preserving contraction generation
---

# Contraction Program Builder

Generate from an actual reference file, not from a recollection of its algorithm or a rewritten module outline. Use `$lqcddb` for source/API facts and physical checks; this skill owns program structure, required inputs, and delivery. API examples in `lqcddb/examples/` are not competing production templates.

## Required inputs: stop and ask when incomplete

Before creating output files, read [references/input-contract.md](references/input-contract.md), collect the applicable required fields, and identify their sources in the user's request or earlier answers. Read-only investigation may continue while questions are pending. **If any applicable required field is missing, ambiguous, or contradictory, ask the user a concise, bundled question and wait; do not fill it with an example value or emit a purportedly runnable program.**

The user may explicitly inherit a named reference: “all parameters as in `njn-3pt`, except …” supplies its documented values. Merely mentioning “3pt”, “NJN”, or “100 eigenvectors” does not. Do not ask again for values already supplied. Values uniquely determined by an explicitly selected ensemble/reference need not be repeated; record their provenance. Internal constants and registry indices are derived, not user questionnaire items.

## Select and freeze one reference

1. A user-specified reference takes priority. Otherwise select the matching bundled profile from [references/reference-profiles.md](references/reference-profiles.md), and record its absolute path and SHA-256. Ask if the choice is ambiguous.
2. Read that Python file in full and its paired shell reference. The bundled files are `assets/contraction.NN.I0.5.2pt.cupy.py` (`nn-2pt`) and `assets/contraction.NJNp-.3pt.GEVP.cupy.na800.py` (`njn-3pt`); the shared shell baseline is `assets/contraction.NJNp-.3pt.GEVP.cupy.sh`.
3. Compare the complete resolved input contract with the selected file, including paths, momentum order, source/current times, operator coefficients, gamma order, estimator, backend, and execution options. Matching only ensemble, Nev, and MPI count is insufficient. A configuration ID passed through `sys.argv[1]` does not require editing the Python file.
4. Choose **exact copy** when no embedded value changes and the reference already supplies the requested progress output, or **minimal adaptation** when inputs differ or progress messages must be added. The maintained bundled assets include the default timing policy below. Do not replace the selected reference during generation. An external reference and a bundled snapshot may have different physics and parameters; never silently merge them.

## Copy, then patch

Read [references/allowed-edits.md](references/allowed-edits.md) and [references/module-layout.md](references/module-layout.md). Use [scripts/reference_copy.py](scripts/reference_copy.py) to copy bytes and apply explicitly enumerated replacements. It refuses an existing output, an ambiguous replacement, and undeclared differences at verification.

```bash
python3 <skill-dir>/scripts/reference_copy.py copy <reference.py> <output.py>
python3 <skill-dir>/scripts/reference_copy.py check <reference.py> <output.py>
# Example input-driven change; supply the same replacement to copy and check.
python3 <skill-dir>/scripts/reference_copy.py copy <reference.py> <output.py> --replace 'Nev_src = 100' 'Nev_src = 64'
```

- For exact copies retain every byte: imports, names, comments, whitespace, expressions, unused declarations, timing, and call order. Do not add generated headers, input validators, manifests, or tests to the production file.
- For numerical parameter/path changes edit their existing locations and dependent shapes/slices/launch values only. Retain block order, loop nesting, communication order, CPU/GPU residency, registry placement, contraction call form, and gather/save structure.
- For changed physics derive the actual Wick plan with the selected library's `run_wick_analysis` **outside the generated production file**, before adapting data loading/registrations. Preserve 2pt `[sink, source]` and 3pt `[sink, source, current]` group semantics. Record tensors/axes in the task report; do not insert `dynamic_plan` or `required_*` sets merely to document this analysis.
- Keep literal source tokens for an unchanged process. For a new process, pass an explicit source operator unchanged; otherwise conjugate the physical initial state, preserving the returned coefficient. Never assume the initial state is the final sink in a transition.
- Do not introduce `main`, loaders, factories, scheduling layers, new CLI/config systems, reformatted imports, or generalized templates. Localize and explain necessary extensions for new physics; a topology change cannot be described as identical structure.

## Physics and compatibility

Check flavor balance, coefficients, gamma order, projection dimensions, peram direction, eigenvector axes/weights, boundary signs, and complex output. Verify time ownership against the selected `get_mpi_tlist` implementation; retain a reference's proven `t % size` / `t // size` expressions instead of rewriting them for style. The current implementation accepts both `Scatter` and `TScatter` in `get_mpi_tlist`; this does not make the corresponding `get_mpi_data` operations interchangeable.

For a new observable start from `ignore_dis=False` and `use_equivalence=False` unless physical/topological justification supports otherwise. For an unchanged reference review existing flags without silently replacing them. If a defect/API incompatibility prevents a runnable result, explain it and make only the justified repair, recording its diff separately from parameter changes. Never claim both byte identity and a modified file. Literal archival copying is distinct from certifying physics/runtime correctness.

`Projector=[proj_sink, proj_src]` is a spin-space map; it need not be square or positive definite. A positive-definite reference correlator on the retained subspace is a separate requirement when actually solving a GEVP. A filename containing `GEVP` does not request an eigensolver.

## Verify and deliver

Follow [references/validation.md](references/validation.md) and [references/submission-script.md](references/submission-script.md). Deliver the Python and same-stem `.sh` unless the user requests Python only. Do not submit a production job merely because files were created.

Record the reference/hash, resolved parameters, exact/adapted mode, and each changed hunk with its input/repair/progress rationale in the task log. Exact mode requires byte equality; adaptation requires equality to the reference plus declared replacements and a structure review. Keep validation outside production files and state which GPU, MPI, data, and numerical paths were not run.

## Default progress and timing

Follow [references/progress-timing.md](references/progress-timing.md). Reuse existing timers/messages and add only missing milestones: VVV/VDV completion, peram readiness, every main-loop `tsrc`, meaningful progress in the first two loop levels when nested work is long, main-loop total time, gathering, save completion with the path, and whole-program elapsed time. Print stage starts before long work and use `flush=True` so redirected logs show the last reached stage.

This requested progress policy authorizes small logging/timer additions to an otherwise matching reference; no extra permission or required input is needed. Preserve calculations, array axes/residency and loop/collective order. A default progress addition is an explicitly recorded difference, not byte identity to the old file. An explicit request for a literal/archival byte-identical copy overrides these additions; report its missing progress nodes instead. Matching inputs still produce identical bytes to a maintained bundled asset that already contains these messages.

Default to root-only output. Label rank-local second-level timing as local; do not add a Barrier/Reduce inside loops with rank-dependent iteration counts or inside a root-only branch. Synchronize the relevant GPU stream before claiming completed GPU work, without synchronizing every contraction. Progress messages help locate slow/stalled stages; they do not prevent or recover from deadlocks.
