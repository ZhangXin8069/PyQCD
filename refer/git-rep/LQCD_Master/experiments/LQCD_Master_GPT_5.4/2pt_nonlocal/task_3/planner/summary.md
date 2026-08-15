## Revised plan summary

This revision keeps the original physics target but makes the correlator definition executable and unambiguous.

The task is the **connected nonlocal \(\eta_s\)** pseudoscalar two-point function at zero momentum on the provided **C24P29 / cfg 10000** ensemble, using a **point source at \([0,0,0,0]\)**. The source operator is local,
\[
\bar s(0)\gamma_5 s(0),
\]
and the sink operator is the nonlocal pseudoscalar
\[
\bar s(x)\gamma_5 W_z(x,x+n_z\hat z)s(x+n_z\hat z),\qquad n_z=0,\dots,10,
\]
with the **shift applied only to the quark leg**, exactly as requested.

At the contraction level, the correlator is stated explicitly as
\[
C(t,n_z)=\sum_{\vec x}\mathrm{Tr}\left[\gamma_5 S_s(x,0)\gamma_5 W_z(x,x+n_z\hat z)S_s(x+n_z\hat z,0)\right],
\]
which, using \(\gamma_5\)-hermiticity, is equivalently evaluated as
\[
C(t,n_z)=\sum_{\vec x}\mathrm{Tr}\left[S_s^\dagger(x;x_0)\,W_z(x,x+n_z\hat z)\,S_s(x+n_z\hat z;x_0)\right],
\]
with \(x_0=[0,0,0,0]\). This fixes the missing gamma structure issue from the previous plan.

The plan now also **states clearly that the observable is a mixed construction by user request**:
- the strange propagator is inverted using **stout-smeared links** with parameters **(n_steps=1, rho=0.125, ndim=4)**;
- the straight Wilson line for the nonlocal displacement is built from the **original unsmeared gauge field**.

This is not treated as an implicit standard choice. It is flagged as an intentional operator definition that can affect normalization/renormalization and should not be confused with an all-unsmeared or all-smeared nonlocal correlator.

The sink-side transport convention is fixed operationally:
- straight **+z** path,
- **path-ordered** product of original gauge links,
- fixed Euclidean time,
- **periodic z wrapping** for every sink site and every separation \(n_z\).

The previous vague "point sink" language has been replaced by a concrete nonlocal zero-momentum sink measurement object. Only **one forward strange propagator** is required in the plan, but this is now tied to a closed post-processing contraction formula rather than stated loosely.

The output convention is also made explicit. Since the nonlocal correlator can in general be complex before any further symmetry projection, the txt file stores **both real and imaginary parts** with no header:
- one row per separation \(n_z=0,\dots,10\),
- columns ordered as
  \([\mathrm{Re}\,C(t=0),\mathrm{Im}\,C(t=0),\mathrm{Re}\,C(t=1),\mathrm{Im}\,C(t=1),\dots,\mathrm{Re}\,C(t=71),\mathrm{Im}\,C(t=71)]\).

Finally, the revised plan adds task-specific validation requirements:
1. check that \(n_z=0\) reproduces the local connected \(\eta_s\) correlator built from the same stout-link inversion;
2. check boundary-crossing separations against direct Wilson-line construction on selected sink sites;
3. keep explicit record that this is the user-requested mixed-link observable.

So the revised scheme preserves the useful parts of the original plan, but now the operator, contraction, mixed-link status, path convention, and file layout are all defined tightly enough for a correct implementation.