## Revised plan summary

This revision keeps the original objective intact: compute the nonlocal kaon two-point function for the \(\bar s u\) channel with a point source at \([0,0,0,0]\), apply the nonlocal displacement only to the \(u\)-quark line, use the original gauge field for the Wilson-line shift, use stout-smeared links only for propagator inversions, and write the result as plain text in the run directory.

### What was fixed
- The observable is now stated explicitly as a **bare mixed-link nonlocal pseudoscalar kaon correlator**. This matters because the requested setup uses two different gauge-link definitions: stout-smeared links in the Dirac operator and original links in the Wilson line.
- The **full contraction formula** is now written down, including the shifted light line, the local strange line, the Wilson-line orientation, and the \(\gamma_5\)-hermiticity form for the strange propagator.
- The implementation convention for the nonlocal displacement is now unambiguous: for each separation \(z=0,\dots,10\), the sink quark field is evaluated at \(x+z\hat z\), parallel transported back to \(x\) by a straight **forward +z** Wilson line built from the original gauge field, with **periodic wrapping** in the spatial z direction.
- The output contract is now unique and executable: one row per \((z,t)\), with exactly four columns
  \[
  z\;\; t\;\; \mathrm{Re}C\;\; \mathrm{Im}C
  \]
  and no header or extra text.
- A required validation check was added: **the \(z=0\) correlator must reproduce the corresponding local kaon 2pt** on the same inversion setup.
- The plan now explicitly labels the calculation as **single-configuration, single-source, debug-level** rather than implying physics-quality statistics.

### Physics definition used
The sink operator is
\[
O_K(z,t)=\sum_{\vec x}\bar s(x,t)\gamma_5 W_z(x,t;x+z\hat z,t)u(x+z\hat z,t),
\]
with source
\[
O_K^\dagger(0)=\bar u(0)\gamma_5 s(0).
\]
After Wick contraction, the correlator is built from:
- a **forward light propagator** from the point source,
- a **forward strange propagator** from the same point source,
- and a **derived shifted-light object**
  \[
  S_l^W(x+z\hat z,t;0)=W_z(x,t;x+z\hat z,t)S_l(x+z\hat z,t;0).
  \]
Only the light line is shifted; the strange line remains local.

### Numerical execution
- Load the original gauge configuration.
- Make a stout-smeared copy with parameters **(n_steps=1, rho=0.125, ndim=4)**.
- Invert the light and strange clover propagators on the stout-smeared gauge field.
- Build the Wilson line for each \(z\le 10\) using the original gauge field.
- Form the zero-momentum correlator by summing over all sink spatial sites for every sink time slice.
- Save plain text only.

### Important interpretation note
This output is a **bare mixed-discretization correlator**, exactly matching the requested setup. It is suitable for code-path validation or bare-data production, but not by itself a renormalized physical nonlocal matrix element.