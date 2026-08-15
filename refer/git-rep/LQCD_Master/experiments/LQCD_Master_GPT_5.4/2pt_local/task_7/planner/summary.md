## Revised plan summary

This revision keeps the task as a standard meson two-point calculation and fixes the main physics and execution gaps.

The requested "D_s meson" is now stated explicitly as an **interpretation**: the plan computes the **local pseudoscalar D_s channel** with interpolating operator
\[
O_{D_s}(x)=\bar s(x)\gamma_5 c(x).
\]
That is the conventional default for a D_s two-point function when no other channel is specified.

The correlator definition is now closed and executable. The measurement uses **both** required propagators:
- a strange propagator from the point source at `[0,0,0,0]`,
- a charm propagator from the same point source.

The two-point function is defined as the zero-momentum heavy-light pseudoscalar contraction
\[
C_{D_s}(t)=\sum_{\vec x}\operatorname{Tr}\big[S_s^\dagger(x,0)S_c(x,0)\big],
\]
which is equivalent to the local \(\gamma_5\)-source / \(\gamma_5\)-sink meson correlator after the usual \(\gamma_5\)-hermiticity step.

The stout-smearing prescription has been clarified at the operator level: the **full Wilson-clover Dirac operator** for both strange and charm is to be built from the **same one-step stout-smeared links** with parameters `(1, 0.125, 4)`, so the hopping and clover terms are consistent with the requested inversion setup.

The solver choice is tightened for charm: the plan no longer assumes the ensemble multigrid hierarchy is automatically valid for the heavy charm solve. It keeps multigrid as acceptable for the strange inversion, but specifies that charm should use a **standard non-multigrid clover solve unless separately validated**.

The scope is also clarified: because only configuration `10000` is listed, the result is described correctly as a **per-configuration correlator measurement**, not an ensemble-averaged physics result.

Finally, the output contract is made explicit:
- file path: `./ds_2pt_cfg10000.txt` in the run directory,
- exactly `Lt = 72` rows,
- rows ordered by increasing Euclidean time `t = 0, ..., 71`,
- columns: `t  Re[C(t)]  Im[C(t)]`,
- no header and no extra text,
- rank 0 writes only after full spatial and MPI reduction.

A minimal correctness layer is also added: require solver convergence for both inversions and verify that the zero-momentum correlator has a numerically negligible imaginary part.