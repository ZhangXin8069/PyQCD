## Revised plan summary

The revised plan keeps the original task intact but makes the physics and execution definition precise enough for an LQCD production-style review.

The target observable is now stated unambiguously as the zero-momentum nonlocal pseudoscalar charmonium correlator
\(C(t,z)\), with:
- a **local source** \(\bar c(0)\gamma_5 c(0)\),
- a **nonlocal sink** \(\sum_{\vec x} \bar c(x)\gamma_5 W_z(x,x+z\hat e_z)c(x+z\hat e_z)\),
- the **shift applied only to the quark leg**, exactly as requested,
- the **Wilson line built from the original unsmeared gauge field**, also exactly as requested,
- and **stout-smeared links used only for the propagator inversion**.

The contraction is written explicitly, including the \(\gamma_5\)-hermiticity step:
\[
C(t,z)=\sum_{\vec x}\operatorname{Tr}\left[S_c^\dagger(x,t;0)\,W_z(x,x+z\hat e_z;U_{\rm orig})\,S_c(x+z\hat e_z,t;0)\right],
\]
so there is no ambiguity about the operator, the Wilson-line orientation, or where the nonlocal shift is applied.

The sink construction is also made executable: the full solved charm propagator field is shifted by \(+z\) in the spatial \(z\)-direction with periodic wrapping, then parallel transported back to the original sink point using the straight Wilson line on the **unsmeared** gauge field, and only then contracted with the unshifted conjugate leg.

Several weaknesses in the previous version were fixed:
- the measurement object is now explicitly **the spatially summed correlator \(C(t,z)\)**, not a site-resolved alternative;
- the nonlocal sink is no longer left as `type: other` without definition;
- boundary wrapping for both the shifted sink point and the Wilson line is specified;
- the output format is now unique and closed: one plain-text file with lines of the form `t z real imag`, no header;
- the mixed-link construction is documented as a deliberate task-defined observable, not an implicit choice;
- the run is explicitly labeled as a **validation/debug scope** because it uses only cfg 10000 and one source position.

The solver section was tightened as well. The charm mass and clover coefficient remain consistent with the fixed ensemble configuration, but the plan no longer assumes that the ensemble multigrid metadata is automatically valid for this heavy-quark stout-smeared clover solve. Instead, it requires use of a solver that is explicitly supported for this charm operator, with the listed multigrid setup used only if it is known to be valid.

Finally, task-specific validation checks were added:
1. \(z=0\) must reproduce the local \(\eta_c\) two-point correlator.
2. The zero-momentum correlator should be checked for the expected near-reality.
3. The code should verify that the nonlocal Wilson line really uses the unsmeared field by observing the effect of swapping to smeared links in a controlled check.

Overall, the revised plan is physically sharper, numerically safer, and operationally closed while preserving the original objective and requested conventions.