## Revised plan summary

This revision fixes the core flavor-flow and operator-definition problems while preserving the requested setup: a **3-point function only** for the formal transition **B- (\(\bar u b\)) → K*- (\(\bar u s\))** with a **local vector current** \(J_x = \bar s\gamma_x b\), a **point source at** `[0,0,0,0]`, **zero momentum**, **fixed** `tseq = 8`, and **stout-smeared inversion links** with `(n_steps=1, rho=0.125, ndim=4)`.

### What was corrected
- The **source operator** is now consistent with the stated B- flavor content: `ubar gamma_5 b`.
- The **sink operator** is now consistent with the stated K*- flavor content: `ubar gamma_x s`.
- The hadron naming and flavor orientation are kept consistent throughout.
- The plan now states the **actual connected meson 3pt contraction** for this process and uses it to justify the sequential-source choice.

### Contraction logic used
With source operator \(\bar u\gamma_5 b\), sink operator \(\bar u\gamma_x s\), and current \(\bar s\gamma_x b\), the connected correlator has the structure
\[
C_3(\tau) \propto \mathrm{Tr}\left[S_l(0,x)\,\gamma_x\,S_s(x,z)\,\gamma_x\,S_b(z,0)\,\gamma_5\right],
\]
with source at \(t=0\), current insertion time \(\tau\), and sink time \(t_f=8\).

This makes the flavor flow explicit:
- the **light propagator** carries the spectator line,
- the **bottom propagator** runs from source to current,
- the **strange propagator** runs from current to sink.

Using \(\gamma_5\)-hermiticity only on the spectator light line, the plan constructs a **sink-fixed strange sequential propagator** for the chosen sink operator and polarization. This closes the contraction chain correctly for the requested process.

### Heavy-quark status
The ensemble block includes a `bottom: 1.5` mass, and the original plan used it directly. The revised plan keeps that value to remain consistent with the fixed configuration, but it now states clearly that this is **not a validated physical b-quark formulation** at this lattice spacing. Therefore the computation is explicitly labeled as a **formal bare-correlator smoke test**, not a controlled B-physics determination.

### Current normalization
The revised plan also fixes an interpretational gap: it explicitly states that the output is the **bare lattice 3pt correlator only**, with **no renormalization or improvement factor** applied to the heavy-light vector current. That makes the deliverable scientifically honest and executable.

### Output definition
The output format is now fully specified:
- plain text file in the run directory,
- **no header**,
- one row per insertion time,
- columns: `tau ReC ImC`,
- only the physically relevant insertion range **1 ≤ tau ≤ 7** for `tseq=8`.

### Scope and limitations
The task still obeys the user instruction **not to compute 2pt functions**. The plan therefore does not attempt overlap or excited-state validation and explicitly treats this as a **minimal smoke-test measurement** on a single configuration with a single source and single source-sink separation.