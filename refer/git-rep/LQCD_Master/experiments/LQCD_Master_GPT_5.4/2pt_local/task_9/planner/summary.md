This revised plan keeps the original proton-spectroscopy objective but makes the deliverable and execution chain explicit.

It computes the **positive-parity proton two-point function** using the standard local proton interpolating operator
\(\epsilon^{abc}(u^{Ta} C\gamma_5 d^b)u^c\), a **point source at [0,0,0,0]**, and the requested projector
\(T_{\mathrm{mat}}=(I+\gamma_t)/2\) with **\(\gamma_t \equiv \gamma_4\)**.
The gauge links used in the light-quark inversions are **stout smeared** with the requested parameters interpreted as **(n_steps, rho, ndim) = (1, 0.125, 4)**.

To close the observable-to-output chain rigorously, the plan now states that the produced observable is the **zero-momentum projected correlator**
\(C(t)\), obtained by performing the standard proton baryon 2pt contraction, summing over all sink spatial sites at fixed Euclidean time, and then gathering the result across MPI ranks. Since the task did not explicitly specify momentum, **\(\vec p=0\)** is documented as an interpretation choice rather than left implicit.

The propagator content is tightened: only the **light clover propagator** is inverted for this task. The plan explicitly states the physics assumption that **u and d are treated as degenerate light quarks on this ensemble**, so the same light propagator is reused for all three uud lines. Strange and charm masses remain in the ensemble metadata for consistency, but are marked as **unused in this calculation**.

The solver target was relaxed from an unnecessarily aggressive residual to a more standard **1e-8** for a production proton 2pt, improving executability without changing the requested physics.

The output is now fully specified: write a plain text file named **`proton_2pt.txt`** in the **run directory**, with **no header**, and with one row per time slice containing:

` t   Re[C(t)]   Im[C(t)] `

Finally, the plan now includes minimal validation checks: confirm the projector convention, verify that all 72 time slices are produced after the spatial sum and MPI gather, check that the imaginary part is numerically negligible, and confirm the expected nontrivial forward/backward baryon time structure. It also explicitly notes the scientific limitation that a **point-source/point-sink local proton correlator** will generally have more excited-state contamination than a smeared-sink setup.