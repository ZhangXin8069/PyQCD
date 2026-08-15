This revised plan keeps the original objective and structure, but fixes the key physics and execution issues.

The task is a standard hadron-spectroscopy calculation: the zero-momentum two-point function of the Lambda baryon built from the local interpolating operator
\(\epsilon_{abc}(u_a^T C\gamma_5 d_b)s_c\).
The source is a point source fixed at \([0,0,0,0]\), and the requested parity projection is implemented as the spin-traced correlator
\(C^{PP}_\Lambda(t)=\mathrm{Tr}[T_{\mathrm{mat}} C_\Lambda(t)]\) with
\(T_{\mathrm{mat}}=(I+\gamma_4)/2\), identifying the user’s \(\gamma_t\) with Euclidean \(\gamma_4\).

The main correction is in the measurement definition: a Lambda 2pt function is a single baryon contraction using **two light lines and one strange line simultaneously**, not two separate baryon measurements. The physics block now defines the Lambda operator with explicit **u, d, s slots**, while the implementation note separately states the isospin-symmetric shortcut
\(S_u=S_d=S_l\). This avoids conflating the physical operator with the propagator reuse.

The stout-smearing request is also made explicit: the tuple \((1,0.125,4)\) is resolved as
`n_steps=1, rho=0.125, ndim=4`, matching the usual stout API ordering. Because the provided ensemble metadata do not by themselves prove that the valence Dirac operator is meant to use stout-smeared links, the plan now states the necessary validity condition clearly: this should proceed only if stout-smeared inversion links are indeed the intended fermion operator for this task; otherwise the result would be a mixed-action correlator.

For output closure, the plan now specifies the exact artifact: write one **real projected correlator value per timeslice** for \(t=0,\dots,71\) into `./lambda_2pt_pp.txt` in the run directory, with **no header, labels, or metadata**.

Finally, the run is labeled honestly as a **single-configuration debug/smoke test** on cfg `10000`, not a production-statistics calculation.