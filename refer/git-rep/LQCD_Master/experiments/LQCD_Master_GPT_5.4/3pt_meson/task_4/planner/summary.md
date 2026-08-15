## Revised plan summary

This revision keeps the original task structure and user-fixed inputs, but makes the physics and execution chain explicit and correct for the requested channel.

### What is being computed
The plan computes the **connected meson three-point function** for
\(B^- (\bar u b) \to K^- (\bar u s)\) with
- source interpolator: \(\bar u\gamma_5 b\),
- sink interpolator: \(\bar u\gamma_5 s\),
- local current: \(J_x = \bar s\gamma_x b\),
- point source at \([0,0,0,0]\),
- zero source and sink momentum,
- fixed source-sink separation \(t_{\rm seq}=8\).

### Explicit contraction now stated
The connected correlator is written explicitly as
\[
C_3(\tau) = -\sum_{\vec x,\vec z}
\mathrm{Tr}\big[\gamma_5 S_s(x,z)\gamma_x S_b(z,0)\gamma_5 S_l(0,x)\big],
\]
with \(x=(\vec x,8)\) and \(z=(\vec z,\tau)\).
This fixes the flavor flow unambiguously:
- \(S_l\): light spectator line,
- \(S_b\): forward heavy line from source to current,
- \(S_s\): strange line from current to sink.

### Sequential-source choice is now justified
The revised plan uses a **fixed-sink strange sequential propagator**, because the sink operator is \(\bar u\gamma_5 s\). The sequential source is built from the sink \(\gamma_5\) structure and the spectator light propagator on the sink time slice \(t_f=8\). This is the correct meson 3pt flavor routing for this channel and replaces the previous unjustified template statement.

### Heavy-quark issue is handled honestly
The ensemble information had to remain unchanged, including the listed `bottom: 1.5`. Since no controlled heavy-quark action or RHQ/NRQCD tuning was provided, the revised plan **does not pretend this is a physical B-physics production run**. It explicitly labels the task as a **single-configuration debug/code-validation measurement** using the ensemble clover kernel with the given heavy mass parameter. This resolves the previous scientific overclaim while preserving the requested computation.

### Kinematics are clarified
With both source and sink at zero spatial momentum, the setup has \(\vec q=0\). For a pseudoscalar-to-pseudoscalar transition, the spatial vector-current matrix element is expected to vanish at zero three-momentum in the physical continuum interpretation. The revised plan therefore states clearly that this kinematic choice is **not suitable for extracting a physical \(B\to K\) form factor**, but can still be used as a contraction/pathway debug test, exactly as requested.

### Measurement block is tightened operationally
The run now specifies:
- one forward light propagator,
- one forward heavy propagator,
- one strange sequential propagator,
- contraction for each insertion time \(\tau=1,\dots,7\),
- explicit exclusion of \(\tau=0\) and \(\tau=8\) contact points.

### Validation without saved 2pt data
The plan still obeys the instruction **not to compute/save 2pt correlators**, but adds minimal internal checks:
- inversion residuals and iteration counts,
- gamma5-hermiticity spot checks,
- enforcement of the allowed insertion-time range,
- sanity check that the zero-momentum spatial-current correlator is near zero.

### Output is now reproducible while keeping the user's txt requirement
The payload remains a plain txt file in the run directory with **no header and no extra text**, but its structure is fixed to one row per insertion time:

`tau  Re[C3(tau)]  Im[C3(tau)]`

for \(\tau=1,\dots,7\). This preserves interpretability without violating the no-header requirement.