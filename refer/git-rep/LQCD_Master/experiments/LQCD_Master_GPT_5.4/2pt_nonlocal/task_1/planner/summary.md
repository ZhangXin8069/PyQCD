## Revised plan summary

This revision keeps the original objective but makes the physics definition, contraction, validation, and file contract explicit enough to execute reliably.

### Physics object now defined unambiguously
The measured quantity is the **full bare Euclidean correlator**
\[
C(z_{\rm sep},t)=\langle O_\pi(z_{\rm sep},t) O_\pi^\dagger(0)\rangle
\]
for **all** integer separations \(z_{\rm sep}=0,\dots,10\) and **all** Euclidean times \(t=0,\dots,71\). The source operator is local at \([0,0,0,0]\), while only the **sink operator** is nonlocal.

### Operator convention kept, but clarified
The sink operator is
\[
O_\pi(z,t)=\sum_{\vec x}\bar d(x)\gamma_5 W_z(x,x+z\hat z;U_{\rm orig})u(x+z\hat z),
\]
with the shift applied **only to the quark leg** as requested. The source remains local,
\[
O_\pi^\dagger(0)=\bar u(0)\gamma_5 d(0).
\]

### Wick contraction now shown correctly
The revised plan does not jump directly to \(\mathrm{Tr}[S^\dagger S_{\rm shift}]\). It first writes the connected charged-pion contraction with explicit \(\gamma_5\) factors and then reduces it using:
- isospin symmetry \(S_u=S_d=S_l\),
- \(\gamma_5\)-hermiticity of the stout-link fermion propagator,
- cyclicity of the trace,
- the fact that the Wilson line acts only in color space.

This yields the implemented estimator
\[
C(z,t)=-\sum_{\vec x}\mathrm{Tr}\big[S_l(x,t;0)^\dagger\,W_z(x,x+z\hat z)S_l(x+z\hat z,t;0)\big].
\]
The overall minus sign is retained consistently as part of the pseudoscalar convention.

### Hybrid-link nature is now documented explicitly
The plan now states clearly that this is a **custom bare hybrid correlator**:
- propagators are inverted on **stout-smeared** links with parameters \((1,0.125,4)\);
- the nonlocal Wilson line is built from the **original unsmeared** gauge field.

That matches the user request exactly, while warning that this is **not** the canonical same-action gauge-covariant bilinear.

### Solver and implementation path are tighter
The revised plan preserves the original propagator choice but corrects the implementation language:
- the standard **12 spin-color point-source solves** reconstruct the exact point-source propagator matrix;
- they are **not** treated as a statistical average;
- the plan explicitly requires access to the **full sink-site resolved propagator field** so the z-shift can be applied in postprocessing.

It also records that stout smearing is applied before fermion-operator construction and that the ensemble-consistent clover convention / standard even-odd preconditioned MG path should be used.

### Boundary handling is now explicit
The plan now fixes the missing convention for large sink shifts:
- spatial directions are periodic;
- both the shifted coordinate \(x+z\hat z\) and the Wilson-line product are built with **periodic wrapping in z**.

This is essential for separations near the boundary on \(L_z=24\).

### Validation step added
A mandatory internal check is now included:
- at \(z=0\), the nonlocal construction must reduce to the local pion correlator from the same propagator and contraction pipeline,
\[
C_{\rm local}(t)=-\sum_{\vec x}\mathrm{Tr}[S_l(x,t;0)^\dagger S_l(x,t;0)].
\]
This is the key sanity test for the custom shift logic.

### Run classification corrected
Because the ensemble block fixes only **cfg 10000**, the revised plan labels this explicitly as a **single-configuration validation/debug run**, not a production physics measurement.

### Output contract made unambiguous
The txt output remains headerless as required, but the format is now fixed precisely:
- one row per correlator value,
- **z-major then t-major** ordering,
- exactly four whitespace-separated columns:
  `z_sep  t  Re[C(z,t)]  Im[C(z,t)]`.

This preserves the user requirement of “no header or extra text” while removing ambiguity for downstream use.