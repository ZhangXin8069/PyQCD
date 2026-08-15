**Revised plan summary**

This plan keeps the original physics target intact: a single `multi_hadron_2pt` measurement for the local six-quark operator
`O = ((u^T Cg5 d) d) ((u^T Cg5 d) u)`,
constructed as one genuine two-baryon correlator rather than a product of separate baryon two-point functions. The neutron-like block remains `(u^T Cg5 d)d`, and the proton-like block is corrected to the physically consistent flavor content `uud` for `(u^T Cg5 d)u`.

The key revision is that the combined source and sink operators are now specified explicitly in the measurement section in a form usable by a `multi_hadron_2pt` contraction generator. Each baryon block is listed in order, and `P_plus = (1 + gamma_4)/2` is required to act on each baryon block individually before the full six-quark contraction is assembled. This prevents the wrong implementation in which either the correlator factorizes into two baryon correlators or the parity projection is applied only at the composite level.

The locality and momentum projection are also tightened. The plan now states that the six quarks at the source share one common local source point, and the six quarks at the sink share one common local sink point before any spatial summation is performed. Zero total momentum is then enforced by a single overall sink spatial sum only, with independent baryon sums explicitly forbidden, so the observable remains the requested local six-quark correlator.

On the numerical side, the calculation still uses a single light point-source propagator from `[0,0,0,0]`, reused for all `u` and `d` lines via isospin symmetry, exactly as appropriate for this operator content. The inversion continues to use stout-smeared gauge links with the requested parameters `n_steps=1`, `rho=0.125`, `ndim=4`, together with the ensemble-consistent clover mass and coefficient already present in the original scheme.

Finally, the revised plan now makes executability stricter by requiring that the `multi_hadron_2pt` codegen path support a non-factorized local two-baryon six-quark contraction with per-block projectors; fallback to a simpler correlator template is disallowed. Output remains a plain-text correlator file in the run directory, with metadata enabled so that the source position, smearing choice, projector convention, and operator definition are recorded consistently.