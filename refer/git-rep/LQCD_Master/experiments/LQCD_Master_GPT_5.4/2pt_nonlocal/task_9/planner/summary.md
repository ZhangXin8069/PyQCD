## Revised plan summary

This revision keeps the original task structure and ensemble configuration, but fixes the physics definition and execution chain so the requested observable is unambiguous and implementable.

### What is being computed
The plan now explicitly defines the nonlocal D*+ correlator for
\(D^{*+}(\bar d c)\) as
\[
O_i(z,t)=\sum_{\vec x}\bar d(x,t)\,\gamma_i\,W(x,x+z\hat z)\,c(x+z\hat z,t),
\qquad
O_i^\dagger(0)=\bar c(0)\gamma_i d(0),
\]
with \(i=x,y,z\) and \(z=0,\dots,10\).

After Wick contraction, the implementation target is fixed as
\[
C_i(z,t)=\sum_{\vec x}\operatorname{Tr}\Big[(\gamma_5 S_d(x,0)^\dagger \gamma_5)\,\gamma_i\,W(x,x+z\hat z)\,S_c(x+z\hat z,0)\,\gamma_i\Big].
\]
This directly resolves the main critique: the Wilson-line orientation and the shifted propagator are now specified exactly, and the shift is applied only to the charm quark line as requested.

### Propagators and measurement chain
The measurement chain is now closed explicitly:
- one forward **light** propagator from the point source \([0,0,0,0]\), interpreted as the **d** line;
- one forward **charm** propagator from the same source;
- for each separation \(z=0..10\), a derived shifted charm object
  \(W(x,x+z\hat z)S_c(x+z\hat z,0)\);
- contraction with the local light anti-quark leg realized via \(\gamma_5\)-hermiticity;
- zero-momentum sink sum over spatial \(x\);
- separate evaluation for \(\gamma_x,\gamma_y,\gamma_z\).

### Mixed-link setup is now treated correctly
The previous plan presented the use of stout-smeared links for inversion and original unsmeared links for the Wilson line as if it were automatically benign. The revised plan keeps this setup because it is user-requested, but labels it clearly as a **mixed operator/action definition** whose renormalization and matching consequences are outside the scope of this run.

### Boundary handling is fixed
The revised plan adds an explicit periodic-wrapping rule in the spatial z direction. Both the displaced sink point and every link entering the Wilson line are evaluated with the z coordinate understood modulo \(L_z=24\). This is necessary for correctness at larger separations and near the boundary.

### Polarization handling is improved
A z-directed nonlocal displacement breaks cubic symmetry, so a blind average over \(\gamma_x,\gamma_y,\gamma_z\) is not physically protected. The new plan therefore measures and saves
- \(C_x\), \(C_y\), \(C_z\),
- transverse \(C_T=(C_x+C_y)/2\),
- longitudinal \(C_L=C_z\),
- and also \(C_{\rm avg}=(C_x+C_y+C_z)/3\)
for compatibility with the original request.

### Output contract is now deterministic
The txt output is now fully specified:
- no header, no extra text;
- one row per \((z,t)\) pair;
- ordering: ascending \(z\), then ascending \(t\);
- columns:
  `cfg_index z t Re[C_x] Im[C_x] Re[C_y] Im[C_y] Re[C_z] Im[C_z] Re[C_T] Im[C_T] Re[C_L] Im[C_L] Re[C_avg] Im[C_avg]`.

### Scientific status
Because the fixed ensemble input contains only configuration `10000`, the revised plan states clearly that this is a **single-configuration functional/test run**, not a statistically credible physics measurement. It also records that the charm parameters are taken from the provided configuration metadata and are **not independently validated here** for precision D* physics.