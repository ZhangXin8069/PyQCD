This revised plan keeps the original objective intact but fixes the parts that were too generic for a valid \(\Lambda_c(udc)\) baryon correlator.

The target observable is now stated explicitly as the **local \(J^P=1/2^+\) \(\Lambda_c\) two-point function** at zero momentum, built from the operator
\[
\mathcal O_{\Lambda_c}=\epsilon^{abc}(u^{Ta}C\gamma_5 d^b)c^c,
\]
with the requested positive-parity projector
\[
T_{\mathrm{mat}}=\frac{1+\gamma_t}{2}.
\]
This makes clear that the plan is not using a generic baryon template, but the specific mixed-flavor \(udc\) channel with the scalar \(ud\) diquark.

The key executable correction is that the correlator is understood to require **two propagator species**:
- one light propagator `prop_l`, reused for both \(u\) and \(d\) by isospin symmetry,
- one charm propagator `prop_c` for the \(c\) line.

The plan also makes the contraction requirement explicit: the contraction generator must be parameterized for the **mixed-flavor baryon operator** \(\epsilon(u^T C\gamma_5 d)c\) at both source and sink, with the proper permutation/sign structure for the two light lines and one distinct charm line. This avoids the main failure mode of accidentally using a three-degenerate-light baryon contraction.

The stout-smearing instruction has been clarified: the tuple `(1, 0.125, 4)` is interpreted as
- `n_steps = 1`,
- `rho = 0.125`,
- `ndim = 4`,

and these stout-smeared links are to be used **inside the clover Dirac operator for all inversions**, not as an unrelated measurement-time preprocessing step.

The charm setup is kept consistent with the provided ensemble facts: the plan uses the supplied charm mass `0.4159` and clover coefficient `1.160920226`, but labels this run appropriately as a **smoke test/debug calculation** rather than a production spectroscopy result. That is important because the run uses only one configuration and one point source, which is enough to validate the pipeline but not enough for a credible physics extraction.

Finally, the output is now fully specified: the txt file should contain **one line per Euclidean time slice**, after spatial summation to zero momentum and application of the parity projector, with **two whitespace-separated columns**
`Re C(t)   Im C(t)`
and **no header or extra text**. This makes the saved result reproducible and directly analyzable downstream.