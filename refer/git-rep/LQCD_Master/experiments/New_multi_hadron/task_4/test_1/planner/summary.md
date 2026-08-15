This revised plan keeps the original physics target unchanged: a single `multi_hadron_2pt` measurement for the local 9-quark operator

`((u^T Cgamma5 d) d) * ((u^T Cgamma5 d) u) * ((u^T Cgamma5 d) s)`

with single-term baryon blocks only, no octet flavor recombination, and `P_plus = (1 + gamma4)/2` applied separately on each baryon block. The correlator remains explicitly non-factorized, so the measurement is one combined multi-hadron contraction rather than a product of neutron-, proton-, and uds-baryon two-point functions.

The key physics correction is the composite operator bookkeeping: the full 9-quark operator is now labeled with valence content `u4d4s1`, which matches the actual source/sink operator exactly. This fixes the most serious metadata error from the previous version, because multi-hadron contraction generation must see the same flavor content as the operator string.

The propagator setup is preserved where it was already sound. The plan still uses the minimal forward-propagator set for this operator basis: one light propagator reused for all `u` and `d` lines by isospin symmetry, and one strange propagator for the single `s` line. Both come from the common point source at `[0, 0, 0, 0]`, and both use stout-smeared links with `(n_steps = 1, rho = 0.125, ndim = 4)` in the inversions, as requested.

To avoid an execution-critical contradiction, the structured duplicate MPI block was removed from `measurement`. The `ensemble` section is kept consistent with the supplied fixed configuration, while the user’s requested launch layout is retained only as a short `extras` note (`launch_request_8_ranks_grid_1_1_2_4`) instead of a second conflicting structured source of truth. This preserves the request without leaving two incompatible MPI decompositions inside the YAML itself.

The output remains a plain-text correlator file in the run directory, with metadata enabled for traceability.