## Revised plan summary

This revision keeps the original computational objective and user-fixed setup intact: compute the **zero-momentum meson three-point correlator** for
\(B_c^- (\bar c b) \to J/\psi(\bar c c)\) with source operator \(\bar c\gamma_5 b\), sink operator \(\bar c\gamma_x c\), current \(\bar c\gamma_x b\), a **point source at [0,0,0,0]**, **tseq = 8**, **stout-smeared inversion links** with \((1,0.125,4)\), **no 2pt calculation**, and **plain txt output with no header**.

The main correction is interpretational and methodological: this is now stated explicitly as a **bare Euclidean 3pt data-product run**, **not** a validated physical decay-form-factor calculation. Given the fixed ensemble facts, especially the bottom entry `mass = 1.5` with the same clover coefficient block used elsewhere, the plan does **not** pretend that the bottom action is a tuned RHQ/Fermilab/NRQCD setup. Instead, it flags the bottom sector as an **uncontrolled heavy-quark systematic** and limits the scientific claim accordingly.

### Physics and quark flow
The revised plan makes the flavor flow explicit and checks it against the meson-3pt code-generation convention:
- source meson: \(\bar c b\)
- sink meson: \(\bar c c\)
- current: \(\bar c\gamma_x b\)
- active quark line: **bottom annihilated at the current**
- created quark line: **charm produced at the current**
- spectator line: **anti-charm**, carried by the forward charm propagator

To avoid a silent convention error, the plan now requires that the code-generation stage use:
- `current_quark = b`
- `current_antiquark = c`
which is the correct mapping for the requested physical current in the meson_3pt generator convention.

### Propagator and solver strategy
The correlator uses the standard fixed-sink sequential-source structure:
1. **forward charm propagator** for the spectator anti-charm line,
2. **forward bottom propagator** for the source-side active \(b\) line,
3. **sequential charm propagator** for the fixed \(J/\psi\) sink with \(\gamma_x\), zero momentum, and `tseq = 8`.

The solver setup is tightened relative to the previous template in one important way: the plan no longer casually inherits the ensemble multigrid settings into heavy-quark inversions. Instead, it states that the charm and bottom solves should be treated as **direct heavy-clover solves on the stout-smeared links unless heavy-sector multigrid validation exists separately**. This addresses the critique that copied light-sector multigrid assumptions were not justified.

### Heavy-quark caveat
Because the ensemble/action facts are fixed and cannot be replaced here, the revised plan does the scientifically honest thing: it keeps the requested run executable while explicitly labeling it as a **debugging/exploratory heavy-heavy 3pt correlator** with **uncontrolled bottom discretization systematics**. That preserves the user’s requested task while removing the false implication that this alone is a credible decay-physics calculation.

### Excited-state and tseq caveat
The user-required point source, local operator setup, and `tseq = 8` are preserved. The plan now states clearly that:
- this setup is executable,
- it is likely to have nontrivial excited-state contamination for heavy-heavy mesons,
- `tseq = 8` corresponds to about **0.84 fm** on this lattice,
- without 2pt data or multiple separations, this run cannot internally validate a plateau.

So the plan respects the fixed request while documenting the limitation instead of ignoring it.

### Output and provenance
The output remains exactly what the user asked for: a **headerless txt file** in the run directory containing only the final 3pt data columns
`(tseq, t_ins, Re, Im)`.

To compensate for the no-header requirement, provenance is strengthened by encoding key identifiers into the filename:
- cfg number,
- source position,
- source/sink/current gamma structures,
- `tseq`.

The plan also requires the run log to record the ensemble/action identifiers and operator choices, since the txt file itself must remain minimal.

In short, the revised plan preserves the requested calculation, fixes the flavor-flow and solver-specification weaknesses, and most importantly reclassifies the result correctly as a **bare exploratory correlator**, not a validated \(B_c^- \to J/\psi\) decay observable.