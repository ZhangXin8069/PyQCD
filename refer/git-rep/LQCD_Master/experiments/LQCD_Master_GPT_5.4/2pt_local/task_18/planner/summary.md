## Revised plan summary

This revision keeps the original task structure and inputs, but closes the main execution and physics gaps.

### What will be computed
The plan now defines the observable explicitly as the **rest-frame Sigma_c+ baryon two-point correlator** for the local operator
\[
O_{\Sigma_c^+}=\epsilon^{abc}(u^{Ta} C\gamma_x d^b)c^c,
\]
with the requested **Cg1 = C@gamma_x** light diquark and the standard sink projector
\[
T_{\mathrm{mat}}=\frac{1+\gamma_t}{2}.
\]
The zero-momentum choice is kept, but it is now stated clearly as an **explicit rest-frame assumption** rather than something silently implied by the original task.

### Key fix: the correlator is now closed operationally
The previous plan left the measurement chain incomplete by referencing only the light propagator. The revised plan explicitly requires:
- **one light propagator** `prop_l`, reused for the degenerate `u` and `d` lines,
- **one charm propagator** `prop_c`,
- a baryon contraction that uses **two light lines plus one charm line**,
- inclusion of the **two Wick-contraction topologies** associated with the two light quarks,
- spin trace after applying the requested projector,
- zero-momentum sink sum over all spatial sites for each time slice.

So the observable is no longer described generically; it is tied directly to the propagators that must be computed.

### Operator/channel interpretation
The plan now states why this operator matches the requested channel: the **C gamma_x** diquark is treated as the requested **Sigma-type flavor-symmetric light-diquark structure**, appropriate for the intended **Sigma_c+** interpolating field. At the same time, the wording avoids overstating channel isolation.

### Projector wording corrected
The role of
\[
T_{\mathrm{mat}}=(I+\gamma_t)/2
\]
is now described more accurately. It is the **standard rest-frame forward positive-parity projector**, but it **does not by itself remove backward contamination or excited-state overlap**. This fixes the prior overclaim.

### Solver and action treatment
The requested stout-link setup is preserved exactly as
- `n_steps = 1`
- `rho = 0.125`
- `ndim = 4`
for both light and charm inversions.

The solver tolerance has been relaxed from an unsupported ultra-tight `1e-12` to a still conservative **`1e-10`**, and the plan now adds a required **residual verification step** instead of assuming solver quality.

For the charm quark, the revised plan no longer presents the ensemble charm setup as automatically validated for precision Sigma_c spectroscopy. Instead, it keeps the supplied charm mass and clover action as part of the fixed run configuration, while explicitly marking **charm tuning/discretization as a limitation** of any physics interpretation from this single run.

### Output is now fully specified
The txt output ambiguity is removed. The final file must contain:
- **one row per time slice**,
- **exactly three whitespace-separated columns**:
  `t Re Im`,
- **no header and no extra text**,
- written in the **run directory**.

### Scope clarification
Because the ensemble block contains only **one configuration** (`10000`) and only one source position, the plan now labels this correctly as a **code-test/debug measurement**, not a physics-quality spectroscopy determination.

Overall, the revised plan preserves the original task faithfully while making the Sigma_c+ correlator definition physically tighter and executable.