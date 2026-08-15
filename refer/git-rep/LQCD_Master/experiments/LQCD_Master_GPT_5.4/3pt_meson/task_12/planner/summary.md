## Revised plan summary

This revision keeps the original computational scope — **compute only the requested 3pt correlator** for
\[
D^0(\bar u c) \to \eta_u(\bar u u)
\]
with source/sink pseudoscalar operators, point source at \([0,0,0,0]\), zero momentum, stout-smeared inversions, and \(t_{\rm seq}=8\) — but fixes the main physics and implementation issues.

### What is now defined more carefully
- The output is explicitly the **raw connected meson 3pt correlator only**.
- The sink \(\eta_u\) is clarified as the **connected** \(\bar u u\) component only; the full flavor-diagonal channel would require disconnected diagrams, which are not part of this task.
- The **sequential propagator** is no longer misidentified as a point solve at \(t=8\). It is correctly specified as a **meson sequential-source inversion** built from the sink \(\gamma_5\) operator and the forward light spectator propagator on the full sink timeslice.
- The plan explicitly warns that with
  - source momentum \(\vec p_i=0\),
  - sink momentum \(\vec p_f=0\), and
  - a spatial vector current \(V_x = \bar u \gamma_x c\),
  the pseudoscalar-to-pseudoscalar matrix element is expected to be **kinematically zero in the continuum**. So this correlator is still computable, but any physical signal should be consistent with zero up to lattice artifacts and noise.

### Propagators used
1. **Forward light propagator** from the point source for the spectator anti-u line.
2. **Forward charm propagator** from the same point source for the line hit by the current.
3. **Light sequential propagator** for the sink-fixed meson 3pt setup at \(t_{\rm seq}=8\).

### What is intentionally not included
- No 2pt correlators.
- No normalization to extract a form factor or decay amplitude.
- No disconnected contribution for the full flavor-diagonal \(\bar u u\) sink channel.

### Heavy-quark limitation
The plan now states explicitly that the charm solve uses the ensemble clover setup as given, but that this does **not by itself guarantee a controlled charm action** at this lattice spacing. Any later interpretation should treat heavy-quark discretization as a real systematic limitation unless independent charm tuning/validation exists.

### Output
The saved file remains exactly in the requested form: a plain `.txt` file in the run directory, with rows
`(tseq, t_ins, Re, Im)`
for `t_ins = 1 ... 7`, and **no header or extra text**.