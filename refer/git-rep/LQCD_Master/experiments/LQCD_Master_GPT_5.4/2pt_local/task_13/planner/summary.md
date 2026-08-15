This revision keeps the original spectroscopy objective but closes the main physics and execution gaps.

The task is now explicitly a **Sigma- (dds) baryon two-point calculation** at **zero momentum** with a **point source at [0,0,0,0]**, using the requested **Cg1 = C@gamma_x** flavor-symmetric dd diquark and the **positive-parity projector** \(T=(I+\gamma_t)/2\). The hadron/operator section now states the operator in a channel-resolved way and clarifies that the projector is applied **after the full baryon spin-color contraction**, which is the correct place for parity projection.

The biggest fix is in the measurement definition: the correlator is no longer left as a generic baryon template consuming only one propagator. It now explicitly requires **both** ingredients needed for a dds baryon correlator:
- one **light propagator** reused for the two identical d-quark lines,
- one **strange propagator** for the spectator s line.

The measurement block also now specifies the actual contraction content: **quark content dds**, **diquark gamma Cg1**, **direct plus exchange terms for the two identical d quarks**, **zero-momentum sink sum**, and the **positive-parity projector**. This makes the plan physically matched to the requested Sigma- observable rather than a generic baryon placeholder.

The stout-smearing requirement has been made unambiguous: the plan now states that the **stout-smeared links are the links used inside the clover Dirac operator during inversion**, not merely an annotation for source construction. The user-given tuple \((1,0.125,4)\) is retained in the natural order **(n_steps=1, rho=0.125, ndim=4)**. Because solver compatibility with smeared-link inversions was a legitimate concern in the critique, the revised plan keeps the ensemble facts unchanged but softens unsupported claims: multigrid may be used **only if it supports the smeared-link operator**, otherwise the run should fall back to a standard solver with the same mass and clover parameters.

The output contract is also tightened. Since the file must contain raw text with no header, the saved observable is now defined explicitly as the **real part of the positive-parity, zero-momentum projected two-point correlator**, with **one numeric value per sink time slice**. A validation checklist is added to ensure:
- the Cg1 dd diquark is implemented with the required symmetry for identical d quarks,
- both identical-d contraction terms are present,
- the projector is applied in the correct place,
- the resulting correlator is numerically real within tolerance and nonvanishing.

Finally, the scientific scope is labeled honestly: with **one configuration and one source**, this should be treated as a **smoke test / code-validation run**, not a production-quality Sigma spectroscopy result.