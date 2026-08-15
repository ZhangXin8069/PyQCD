This revised plan keeps the original task intact but closes the main executability gaps.

It computes the **Xi_c+ baryon two-point correlator** for the **usc** channel using the requested local operator
\[
\mathcal O_{\Xi_c^+}=\epsilon^{abc}(u^{Ta}C\gamma_5 s^b)c^c
\]
with a **point source at [0,0,0,0]**, **zero sink momentum**, and the **positive-parity projector**
\[
T_{\mathrm{mat}}=(I+\gamma_t)/2.
\]

The most important fix is that the measurement is now explicitly bound to **all three required propagators**:
- `prop_light` for the **u** quark,
- `prop_strange` for the **s** quark,
- `prop_charm` for the **c** quark.

The correlator block also now states the **flavor ordering `(u,s,c)`**, the **`us` diquark pair**, the **`Cgamma5` diquark structure**, the **sink spatial sum for zero-momentum projection**, and the **parity projection applied at the sink trace**. This removes the previous ambiguity where only a single propagator was named and the actual Xi_c+ contraction could have been misinterpreted.

The stout-smearing requirement is also tightened: the plan makes it explicit that the links are **stout smeared only for the Dirac inversions** with
- `n_steps = 1`
- `rho = 0.125`
- `ndim = 4`
and that **no additional source or sink smearing** is introduced.

To stay consistent with the supplied ensemble information, the propagator solves use the provided valence masses and clover coefficient. At the same time, the plan now avoids overstating the scientific scope: because this is **one configuration (`10000`) with one point source**, and because **charm tuning for Xi_c spectroscopy is not independently documented here**, the run is labeled as a **debug/executability correlator generation task**, not a physics-ready spectroscopy analysis.

Finally, the output is now fully specified: for each configuration, save **one plain-text file in the run directory**, with **72 rows** and **two numeric columns per row** containing the **real and imaginary parts** of the projected correlator, with **no header and no extra text**.