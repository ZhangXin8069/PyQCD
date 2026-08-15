## Revised plan summary

This revision keeps the requested computational structure but fixes the main physics and execution issues.

The task is now stated explicitly as a **per-configuration connected pseudoscalar charmonium two-point measurement** on the provided C24P29 ensemble, using the local interpolating operator \(\bar c \gamma_5 c\). This avoids incorrectly labeling the result as the full flavor-singlet \(\eta_c\) correlator, since the full singlet channel would also require **disconnected charm-annihilation diagrams**, which are not part of the standard point-to-all connected workflow and were not requested in the original task.

The ensemble block has been kept **exactly consistent** with the provided fixed configuration. The charm propagator uses:
- source type: **point**
- source position: **[0, 0, 0, 0]**
- momentum: **[0, 0, 0]**
- stout-smeared inversion links with **(n_steps=1, rho=0.125, ndim=4)**
- charm mass **0.4159** and clover coefficient **1.160920226** from the supplied ensemble data.

The solver section has been tightened: it no longer presents the listed light-quark multigrid setup as if it were automatically valid for charm. Instead, the plan specifies a **stable charm-appropriate clover inversion** with explicit residual control, while preserving the fixed ensemble parameters.

The source/sink language has also been cleaned up. The source is a **point source**, while the sink is described as a **local operator sink** rather than conflating locality with source geometry. The measurement is the standard connected meson two-point correlator with a **zero-momentum spatial sum over sink positions**.

The output instructions are now operationally complete:
- exactly one plain text file is written in the run directory,
- filename: **`etac_connected_2pt_cfg10000.txt`**,
- content format: **`t Re Im`** rows only,
- **no header** and no extra text,
- **rank 0 only** performs the write in MPI.

So the revised plan is physically more honest, numerically safer, and more executable, while preserving the original objective and requested smearing/source/output choices.