## Revised plan summary

This revision keeps the requested production scope narrow — a **single connected meson three-point function only** — but fixes the physics chain and closes the execution details.

### What is computed
We compute the bare connected correlator for
\[
D^0(\bar u c) \to \pi^-(\bar u d)
\]
with local pseudoscalar source and sink operators,
\[
\mathcal O_{D^0}=\bar u\gamma_5 c, \qquad \mathcal O_{\pi^-}=\bar u\gamma_5 d,
\]
and current
\[
J_x = \bar d\gamma_x c.
\]
The setup is fixed to
- point source at \([0,0,0,0]\),
- \(t_{\rm src}=0\),
- \(t_{\rm sink}=t_{\rm seq}=8\),
- zero source momentum and zero sink momentum,
- insertion times \(\tau=1,\dots,7\), excluding contact points.

### Correct contraction chain
The connected flavor flow is now stated explicitly:
- the **spectator anti-u** line is carried by the forward light propagator,
- the **active charm** leg is a forward charm propagator from source to current,
- the **sink d leg** is implemented by a **light sequential propagator through the pion sink**, built from the sink operator and the spectator light line at \(t=8\).

The correlator is written schematically as
\[
C_3^x(\tau) = -\sum_{\vec x,\vec z} \mathrm{Tr}\big[S_l(0,x)\gamma_5 S_l(x,z)\gamma_x S_c(z,0)\gamma_5\big],
\]
then reorganized into the standard sequential form using the sink-built light sequential object. This removes the ambiguity in the previous plan about whether the sequential inversion corresponded to the correct daughter-quark leg.

### Important physics check: the requested kinematics are symmetry-trivial
The revised plan explicitly flags that for a pseudoscalar-to-pseudoscalar transition,
\[
\langle \pi(p_f)|V_i|D(p_i)\rangle = f_+(q^2)(p_i+p_f)_i + f_-(q^2)(p_i-p_f)_i,
\]
so with
\[
\vec p_i=\vec p_f=0
\]
a spatial current component like \(V_x\) should vanish. Therefore this setup is **not a meaningful form-factor measurement**. It is retained only because the original task fixes these kinematics; the result should be labeled an **infrastructure/code-path test**, not physics extraction.

### Solver and action handling
The plan now distinguishes solver roles coherently:
- one forward light solve,
- one forward charm solve,
- one light sequential solve.

It also states that stout smearing with parameters **(1, 0.125, 4)** must be used **inside the valence Dirac operator being inverted**, not as an unspecified external smoothing step.

### Charm caveat
The ensemble-provided charm mass and clover coefficient are preserved to remain consistent with the fixed configuration, but the plan now clearly marks the charm setup as **unverified for controlled charm physics** on this ensemble. Without an independently validated charm action/tuning and current renormalization, this should not be advertised as a precision heavy-light result.

### Output is now deterministic
The file output is fully specified:
- filename: `./d0_to_piminus_vx_3pt_tseq8_cfg10000.txt`
- no header,
- one line per insertion time,
- each line contains
  `tau Re[C3_x(tau)] Im[C3_x(tau)]`
  in ascending `tau` order.

### Validation limitation retained and documented
The instruction not to compute 2pt functions is respected. The revised plan explicitly documents the consequence: no internal validation of energies, overlaps, or excited-state contamination at \(t_{\rm seq}=8\). Any scientific interpretation must rely on external, already validated 2pt information.