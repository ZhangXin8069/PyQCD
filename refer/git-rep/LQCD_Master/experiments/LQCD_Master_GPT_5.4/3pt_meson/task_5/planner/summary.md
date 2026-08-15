## Revised plan summary

This revision corrects the flavor channel and closes the correlator definition so the computation now matches the requested **connected raw lattice 3pt function for** \(B^-\to\pi^-\) with current \(\bar d\gamma_x b\).

### What was fixed
- The meson interpolators are now aligned with the requested hadron content:
  - **Source:** \(B^- (\bar u b)\) represented by the pseudoscalar bilinear \(\bar u\gamma_5 b\)
  - **Sink:** \(\pi^- (\bar u d)\) represented by \(\bar u\gamma_5 d\)
- The contraction logic is now the correct one for this channel:
  - spectator antiquark: \(\bar u\)
  - transition line: \(b\to d\)
- The sequential propagator is no longer described as an ordinary point source at the sink. It is explicitly identified as a **sink-built sequential source** on time-slice \(t=8\), constructed from the \(\pi^-\) sink operator and the forward spectator light propagator.
- The insertion-time support is specified as **\(\tau=1,\dots,7\)**, excluding endpoint contact terms.
- The output format is made executable and unambiguous: one plain txt file per configuration, no header, with rows
  `tau  Re[C3]  Im[C3]`.

### What is preserved
- The task remains a **standard meson 3pt** plan.
- The provided ensemble block is kept consistent with the fixed configuration.
- The source is still a **point source at** `[0,0,0,0]`.
- Source and sink momenta remain **zero**.
- The sink-source separation remains **tseq = 8**.
- All inversions still use **stout-smeared links** with `(1, 0.125, 4)` interpreted as `(n_steps=1, rho=0.125, ndim=4)`.
- The user’s instruction to **not compute 2pt functions** is preserved.

### Physics scope of the revised product
This plan now clearly labels the result as a **bare, unrenormalized raw 3pt correlator**, not a full physical \(B\to\pi\) form-factor determination. That distinction is essential because:
- no 2pt functions are produced,
- no heavy-light current renormalization/matching is provided,
- the supplied bottom-quark setup is used exactly as requested but is flagged as a **formal/template or smoke-test configuration** unless validated heavy-quark tuning exists for this action and lattice spacing.

So the revised plan is physically cleaner, operationally tighter, and still faithful to the original run objective: produce the requested raw 3pt data only.