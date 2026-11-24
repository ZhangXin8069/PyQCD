# Allowed edits relative to a frozen reference

An allowed category is not permission to edit every occurrence. Each hunk must follow a resolved user input, its dependency, or a reported correctness/compatibility repair.

| Difference | Edit only |
| --- | --- |
| None | Nothing; copy bytes including comments/whitespace |
| Configuration ID/list | Existing shell formula/range or invocation argument; leave Python unchanged if `sys.argv[1]` suffices |
| Ensemble | Existing geometry and dataset/result path values |
| Eigenvector cutoff | Existing cutoff and dependent dimensions/slices; file patterns/omega only if the stored construction differs |
| MPI/resources | Existing grid and shell resource/launch values; preserve validated distribution/loops |
| Momentum/time | Existing lists/ranges/separation values and dependent axes; preserve ordering/boundaries |
| Physics | Necessary operator/current coefficients/tokens, gamma/projection, peram/V directions, dynamic indices and output axes |
| Destination/name | Existing save paths and shell path variables |
| Requested progress/timing policy | Missing milestone prints, coarse timers and necessary local GPU synchronization per `progress-timing.md`; do not change calculation or MPI collective order |
| Verified defect/API change | Smallest coherent repair, separately documented and validated |

Do not rename variables, reorganize imports, change CPU/GPU residency, deduplicate, replace literal operators with helper calls, introduce parsers, move existing blocks/timers, or insert manifests as a side effect. Add requested progress at existing boundaries, not by restructuring computations. Preserve comments and unused declarations unless a requested change or actual repair requires otherwise.

The copying helper accepts an old byte string only if it occurs exactly once. Supply enough context to identify the location and the same replacements to `check`. This proves equality to the reference plus declared patches; physical correctness and structural acceptance require separate review.

If new physics needs more loops or data directions, explain that extension rather than claiming identical structure. Do not force a 4pt estimator into a 3pt save expression. Ordinary generation does not imply permission for a general refactor.
