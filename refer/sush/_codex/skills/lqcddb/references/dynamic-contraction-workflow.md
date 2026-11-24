# Dynamic contraction API workflow

For complete program generation, `$contraction-creator` owns required inputs, reference selection and structure. The flat examples illustrate API usage only. This document explains preparation/review and must not cause additional runtime blocks to be inserted into an unchanged reference.

## Resolve physics and source

Use the `source_root` selected through `SKILL.md`. Resolve operators, currents, coefficients, flavor balance, gamma order, momentum signs, temporal boundary and output axes. Preserve explicit source tokens; for a new process without them, conjugate the physical initial operator and retain its coefficient. Do not replace a reference's literal source with a helper call merely for style.

## Derive tensors before adapting loading

For changed physics call `run_wick_analysis` outside the production file before modifying loading/registrations. Inspect actual plan entries for peram flavor/time directions, VDV/VVV names/time ends, gamma/projection labels, signs and output axes. Summarize these in the task report; do not require runtime `required_*` sets or M0–M8 blocks.

Do not equate the combined Wick flavor string (`p[2]`, e.g. `dd^d`) with a registry key. Check `PeramRegistry.resolve` in the selected source: it resolves the quark flavor first, then the u/d `light` fallback. Shared light data still requires the appropriate flavor-degeneracy assumption.

Raw Wick V numbering is also not the final lookup key: dynamic contraction renumbers VDV/VVV separately within each time end. Use the selected `validate_plan`/tensor-resolution code to check registrations instead of comparing raw V tuples directly. Lookup success alone does not establish dimensions, time direction or numerical correctness.

Use new-observable defaults `ignore_dis=False`, `use_equivalence=False` unless justified. An exact reference copy retains its actual flags; report unsupported assumptions instead of silently changing them. Instantiate `dynamic_contraction` for execution only after registries are ready. Check real signatures in the selected source, not merely examples.

## Review data flow without restructuring

Record tensor shape/axes, absolute time, owner/local index, conjugation, backend and dtype. Check collectives and mapping against the source; retain equivalent existing ownership expressions. Review gathering, all source-relative time axes, boundary factors and source sum/mean. Do not reorder data loading and registries to match a generic outline.

Preserve existing timers and follow `$contraction-creator`'s requested stage/loop/save progress policy when generating programs. Add missing progress inline, flush messages, and identify root-local timing. Synchronize relevant GPU work only at coarse boundaries; do not introduce MPI collectives into rank-dependent loops for logging. Report these additions as differences when needed, preserving an explicit byte-identical archival request. State unrun GPU/MPI/full-ensemble paths and distinguish text equality from numerical validation.
