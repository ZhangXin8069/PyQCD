This revision keeps the original production strategy but fixes the main physics ambiguity: the target observable is now explicitly the connected proton three-point correlator for the axial current

\[
J_x^A = \bar u\,\gamma_x\gamma_5\,u,
\]

with proton source and sink operators

\[
\mathcal O_p = \epsilon^{abc}(u^{Ta} C\gamma_5 d^b)u^c,
\]

zero momentum, point source at `[0,0,0,0]`, positive-parity projector

\[
T = \frac{1+\gamma_t}{2},
\]

and fixed source-sink separation `tseq = 8`.

The required propagator content remains minimal and executable for the connected piece: one forward light propagator and one light sequential propagator, both inverted with the stout-smeared gauge field using `(n_steps, rho, ndim) = (1, 0.125, 4)`. Because the current is flavor-diagonal, the full QCD matrix element would also contain disconnected `u`-loop contributions; this plan now labels the deliverable correctly as the connected correlator only, rather than implying a complete proton axial-current matrix element.

The main technical correction is in the current insertion. Since the code-generation interface for `baryon_3pt` only exposes a single `current_gamma`, the plan now states unambiguously that the generated contraction must be post-edited so that the final current matrix is `gamma_x gamma_5`, not `gamma_x` alone. This preserves the sequential-source workflow while ensuring the saved correlator matches the requested axial channel.

The ensemble block has been kept fully consistent with the supplied fixed configuration. The plan does not introduce alternative masses or ensemble substitutions; it simply uses the verified light-quark action parameters needed for this proton connected 3pt calculation. The existing solver controls remain reasonable production choices: clover coefficient `1.160920226`, `xi_0 = 1.0`, multigrid from the ensemble metadata, `tol = 1e-10`, and `maxiter = 10000`.

The output definition is also tightened. The script is to save a plain text file in the run directory, with no header or extra text, written only by rank 0. The saved rows are fixed to the non-contact insertion window `t_ins = 1, ..., 7` for `tseq = 8`, with columns

`ts eq, t_ins, Re[C3], Im[C3]`.

No two-point correlator is computed or written.