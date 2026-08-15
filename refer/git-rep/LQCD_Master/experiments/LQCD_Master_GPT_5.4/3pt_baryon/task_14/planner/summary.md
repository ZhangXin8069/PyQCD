This revision keeps the original objective—a single zero-momentum `Lambda_b -> Lambda_c+` three-point correlator with source at `[0,0,0,0]`, `tseq = 8`, stout-smeared inversions, and no 2pts—but fixes the main physics and execution gaps.

First, the current is now implemented unambiguously as the axial operator `gamma_x @ gamma_5` in the final three-point contraction. The baryon-3pt einsum generator is still used for the sink-block and sequential-source topology, but the runtime current matrix is explicitly specified as axial rather than leaving only `gamma_x`, which would target the wrong matrix element.

Second, the measurement chain is made explicit. The plan now states that the final observable is built from:
- one forward light propagator reused for both `u` and `d` spectator lines,
- one forward bottom propagator for the source-side current line,
- one charm sequential propagator generated from the `Lambda_c+` sink block with `Cg5` and the positive-parity projector `Tmat = (I + gamma_t)/2`.
The saved data are the insertion-time slices `t_ins = 1..7` for `tseq = 8`, written as plain text rows `(tseq, t_ins, Re, Im)` in the run directory with no header.

Third, the heavy-quark issue is handled honestly instead of being glossed over. Because the fixed ensemble information supplies only a clover-style bottom mass input and no validated RHQ/Fermilab/NRQCD tuning, the plan no longer presents this as a physically controlled `Lambda_b` decay calculation. It is explicitly downgraded to an exploratory bare correlator, with uncontrolled heavy-bottom discretization errors noted in the solver and task description.

So the revised plan remains executable and close to the original request, but its scope is now stated correctly: it computes one bare axial 3pt correlator at a single zero-recoil kinematic point and a single source-sink separation, not a complete or controlled form-factor determination.