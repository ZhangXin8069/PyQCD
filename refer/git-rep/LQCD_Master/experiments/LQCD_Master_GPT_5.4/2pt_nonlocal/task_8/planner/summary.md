## Revised plan summary

The revised scheme keeps the original objective and execution choices, but makes the correlator definition and numerical scope physically explicit.

### What is being computed
We compute a **zero-momentum bare connected hidden-charm nonlocal vector two-point correlator** in the J/psi channel on **cfg 10000** of the **C24P29** ensemble. The sink operator is nonlocal with a straight Wilson-line displacement in the **+z** direction for **dz = 0, ..., 10**, and the source operator is local at the point source **[0,0,0,0]**.

The task-imposed asymmetry is preserved exactly:
- the **quark propagator** is shifted nonlocally,
- the **antiquark propagator** remains local/unshifted,
- the nonlocal Wilson line is built from the **original unsmeared gauge field**,
- the Dirac inversion uses a **stout-smeared gauge copy** with **(n_steps=1, rho=0.125, ndim=4)**.

### Physics definition fixed
The previous version was too informal about the contraction. The revised plan now defines:
- the **local source bilinear**,
- the **nonlocal sink bilinear**,
- the **connected Wick contraction**,
- the **gamma5-hermiticity rewriting** used in code,
- the exact **shifted-propagator convention**
  \[
  S_c^{\rm shift}(x;dz)=W_z(x,x+dz\hat z)S_c(x+dz\hat z;0),
  \]
  which implements the requirement that only the quark line is displaced.

It also fixes the Wilson-line orientation and endpoint convention explicitly: forward **+z** transport with periodic spatial wrap when needed.

### Polarizations and normalization
The correlator is computed separately for **gamma_x, gamma_y, gamma_z** and only then averaged:
\[
C_{\rm avg}(dz,t)=\frac{C_x(dz,t)+C_y(dz,t)+C_z(dz,t)}{3}.
\]
The zero-momentum projection is the **plain spatial sum** over sink positions, with **no division by spatial volume**.

### Solver setup tightened
The plan now avoids silently inheriting an unsuitable heavy-quark solver strategy. It specifies:
- charm clover inversion on the stout-smeared gauge field,
- target residual **1e-12**,
- **BiCGStab/CGNR-style heavy-quark solve** as the default practical path,
- explicit residual verification,
- fallback to a **non-multigrid Krylov solver** if the listed ensemble MG hierarchy is unstable or unsupported for this stout-smeared heavy-clover setup.

### Scientific scope clarified
This is now labeled correctly as a **smoke test / bare correlator production**, not a statistically complete physics result, because only:
- **one configuration** and
- **one source position**
are used.

The revised plan also corrects the disconnected-diagram statement: charm-annihilation disconnected contributions are **neglected as an approximation** (typically OZI suppressed for J/psi), not claimed to be exactly absent.

### Interpretation caveats added
Two important caveats are now explicit:
1. Calling the observable “J/psi” assumes that the provided valence charm parameter is appropriately tuned; without an independent tuning check, the safer label is **bare connected hidden-charm vector correlator**.
2. The **dz dependence** of the bare nonlocal correlator is strongly affected by **Wilson-line self-energy / renormalization**, and the hybrid smeared-inversion / unsmeared-transport setup is a **task-imposed mixed construction**, not a standard fully matched operator definition.

### Output
The final output remains exactly as requested: a **plain txt file** in the run directory with **no header** and no extra text, containing rows
`dz  t  ReC  ImC`
for **dz = 0..10** and **t = 0..71**.