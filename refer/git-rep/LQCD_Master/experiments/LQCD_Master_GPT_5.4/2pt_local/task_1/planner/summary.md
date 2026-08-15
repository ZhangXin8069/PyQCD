## Revised plan summary

This plan keeps the original objective and structure, but fixes the key ambiguity identified in review: the calculation is tied explicitly to the **exact requested bilinear**
\[
O(x)=\bar u(x)\gamma_5 d(x),
\]
not just to a generic “pion template.” The two-point function is therefore defined using this operator at the sink and its Hermitian conjugate at the source, with zero-momentum projection from the spatial sum.

On the numerical side, the plan now states clearly that the valence inversion uses the **ensemble Clover operator** with the ensemble’s light-quark mass and clover coefficient, evaluated on a **stout-smeared copy** of the gauge field with the user-requested parameters `(1, 0.125, 4)`. This makes the stout-smearing step part of the intended propagator definition, rather than an unexamined default.

The light-propagator reuse is also made explicit and conditional: one light propagator is used for both the `u` and `d` lines **only because** the ensemble is assumed to have degenerate light valence masses and identical boundary conditions for `u/d`. If that assumption were not valid, separate solves would be required.

For executability, the measurement specification is tightened to the exact `meson_2pt` code-generation mapping:
- `antiquark = u`
- `quark = d`
- `gamma_snk = gamma5`
- `gamma_src = gamma5`

Finally, the output is now concrete: rank 0 writes a plain-text file
`pion_2pt_ubar_g5_d_cfg10000.txt`
in the run directory, with exactly three columns
`t Re Im`
and no header or extra text. The plan also requires checking solver convergence before writing the file.