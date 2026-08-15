## Revised plan summary

This revision keeps the original task intact but closes the physics and execution details that were missing.

The target observable is now fixed unambiguously as a **nonlocal heavy-light vector meson two-point function** for \(D_s^* \sim \bar s c\), with:
- a **local source** at \([0,0,0,0]\),
- a **nonlocal sink** displaced along **+z** by \(z=0,\dots,10\),
- the displacement applied **only to the charm field**,
- a **straight Wilson line built from the original unsmeared gauge field**, and
- quark inversions done on a **stout-smeared copy** of the gauge field with parameters \((n_{\rm steps}=1, \rho=0.125, n_{\rm dim}=4)\).

The correlator is explicitly specified after Wick contraction, so the measurement is no longer under-defined. The plan now states that the observable requires **both** ingredients:
- the point-source strange propagator, and
- the sink-shifted charm propagator.

This resolves the main problem in the previous version, where only the shifted charm line was referenced in the measurement block.

The revised correlator convention is:
\[
C_i(z,t)=\sum_{\vec x} \mathrm{Tr}\left[S_s(0;x,t)\,\gamma_i\,W_{\rm orig}(x,x+z\hat z;t)\,S_c(x+z\hat z,t;0)\,\gamma_i\right],
\]
with \(i=x,y,z\), followed by the polarization average
\[
C_{\rm avg}(z,t)=\frac{1}{3}\left(C_x+C_y+C_z\right).
\]
The order of operations is fixed: build each polarization channel, do the zero-momentum spatial sum, then average the three polarizations.

The output format is also made reproducible. The file is plain text with **no header**, and each row is fixed as:
`z  t  ReC  ImC`
for all \(z=0,\dots,10\) and all \(t=0,\dots,71\).

Finally, the plan now labels this run correctly as a **single-configuration smoke test/debug measurement**, not a physics-quality ensemble result. It also adds minimal validation checks directly relevant to this task:
- \(z=0\) must reproduce the local vector \(D_s^*\) correlator,
- the three polarizations should agree within expected symmetry breaking and noise,
- the imaginary part should be numerically negligible.

This makes the plan physically tighter, executable, and aligned with standard LQCD correlator practice while preserving the original user requirements.