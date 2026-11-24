# Progress at computation boundaries

Default generated programs should show that work is advancing and identify the last reached stage. Use ordinary inline `time.perf_counter()` and `print(..., flush=True)`, reusing existing timers and messages. Do not introduce a logging framework, monitoring thread, timeout, or change to the contraction algorithm.

| Node | Output |
| --- | --- |
| Eigenvectors / VVV / VDV | Start before loading/construction; completion and elapsed seconds when vertices, including required omega weighting, are ready |
| Perambulator loading/distribution | Start and completion with elapsed seconds |
| Main `tsrc` loop | Source ID and progress count on entry; completion and elapsed time for every source |
| Second level of a long nested loop | One completion/time message per meaningful outer batch, with source and batch/time identity; avoid output in every deepest spin/momentum/contraction call |
| End of the main loop | Elapsed time for the whole loop, measured separately from loading and saving |
| Gather | Start and completion with elapsed time |
| Save | Start before estimator/save work, then the actual filename and elapsed time only after the save returns successfully |
| End of the program | Whole-program elapsed time after successful save |

For the current bundled files, the first two main-loop levels are `tsrc -> i` (2pt sink basis) and `tsrc -> tcur0` (3pt insertion time). Keep those nests intact. The 3pt `tcur0` is a local array index: report its physical time via the existing `trank[tcur0]` mapping, with the local index labeled if printed. Combined VVV/VDV completion is appropriate when they are constructed together; do not split or reorder their loops just to report separate timings.

## MPI and GPU semantics

- Print from `rank == 0` by default to avoid duplicated logs. Include configuration/source and the relevant second-level item. Root-local messages must say `rank=0 local`; they do not mean all ranks finished that item. Retain existing all-rank stage/source synchronization for global completion messages.
- Never add MPI collectives for progress inside a rank-dependent `tcur`/sink-time loop or a root-only branch. Reuse existing safe stage boundaries. Do not add new barriers merely to align printed lines.
- GPU dispatch is asynchronous. Before a completed-work time is printed, synchronize the stream doing that work at the coarse stage/batch boundary. For a root-local batch this is only a local GPU synchronization, not an MPI barrier. Avoid synchronizing individual contractions; additional coarse synchronization can still reduce overlap and should remain limited to requested milestones.
- Preserve the reference's stream model. If it uses multiple streams, completion must cover the relevant streams/events; do not assume the default stream covers all work.
- Use monotonic elapsed time (`perf_counter`), and `flush=True` even when the submission script already uses `python -u`. A wall-clock timestamp may be included if useful, but a new timestamp/logging format is not required.
- If a single inner operation is very long, the last stage-start line identifies the active operation. Output is not a deadlock detector or recovery mechanism; do not introduce watchdog termination as part of this request.

## Preserve reference fidelity

The user requested these progress additions. When absent, patch only timer assignments, prints and necessary coarse local GPU synchronization, recording these hunks separately from parameter/physics edits. Preserve block ordering, variables, collective calls, allocations, registry data, contractions and save expressions. Never rename/restructure computational code for timing.

The maintained assets already contain these milestones and remain byte-copyable for matching requests. If the user explicitly requests a byte-identical copy of another reference, preserve it and report logging gaps rather than claiming a modified file is identical.

For the bundled save expression, `loop_tsrc(...)` is evaluated within `backend.save(...)`; the timer includes source summation and saving. Label it `source sum/save time`, not pure disk I/O. Print success only after this expression completes, and retain exceptions instead of printing success from a `finally` block.
