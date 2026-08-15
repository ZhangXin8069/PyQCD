## Revised plan summary

This revision keeps the original objective and ensemble setup, but fixes the key physics and execution issues in the three-point construction.

### What is being computed
The task is to produce a **bare connected meson 3pt correlator only** for
\[
D^0(\bar u c) \to K^-(\bar u s)
\]
with
- source interpolator: \(\bar u\gamma_5 c\),
- sink interpolator: \(\bar u\gamma_5 s\),
- local current insertion: \(J_x=\bar s\gamma_x c\),
- point source at \([0,0,0,0]\),
- zero **spatial** momentum at source and sink,
- fixed sink time separation \(t_{\rm seq}=8\).

No 2pt functions, no ratio construction, no form-factor extraction, and no current renormalization are included.

### Main physics fix
The previous plan incorrectly described the sink-side object as if it were a **point-source strange propagator started at \(t=8\)**. That is not the required object for a fixed-sink meson 3pt.

The corrected plan uses the proper sequential-source method:
1. solve the **forward light propagator** from the source point for the spectator \(\bar u\) line,
2. solve the **forward charm propagator** from the source point for the source-to-current leg,
3. build a **strange sequential source** from the **full sink time slice** at \(t=8\), including
   - the sink operator \(\gamma_5\),
   - the zero-momentum spatial sum over sink sites,
   - contraction with the forward spectator light propagator,
4. invert the strange Dirac operator on that sequential source.

This closes the actual Wick-contraction chain for the requested connected 3pt.

### Momentum-transfer language corrected
The plan now states the physically correct kinematics:
- source and sink are both projected to **zero spatial momentum**,
- therefore the **spatial** momentum transfer is zero,
- but the setup is **not** at zero four-momentum transfer, since \(m_D \neq m_K\).

This avoids the previous ambiguity where \(q=[0,0,0]\) could be misread as full zero momentum transfer.

### Output clarified
The output format is now fully specified so the production result is usable:
- one row per insertion time,
- insertion times restricted to \(\tau=1,\dots,7\),
- columns are
  `tau  ReC3  ImC3`,
- plain `.txt` in the run directory,
- **no header and no extra text**.

### Solver and action consistency
The revised plan keeps the original stout-smearing requirement and applies it consistently to **all** forward and sequential inversions:
- stout parameters interpreted as `n_steps=1`, `rho=0.125`, `ndim=4`.

It also records reproducibility assumptions more explicitly:
- same stout-smeared links define every inversion,
- even-odd preconditioning assumed,
- multigrid may be used for light solves if available,
- charm multigrid is not assumed.

The charm setup is flagged honestly as an **input assumption** of the ensemble/tuning rather than silently treated as automatically validated.

### Topology and interpretation
The revised plan now explicitly states that this flavor structure has **connected contribution only** for the requested correlator. It also states clearly that the result is a **bare 3pt correlator**, not a physical decay amplitude or renormalized form factor.

Overall, the new plan preserves the original structure and requested settings while making the correlator definition physically correct, numerically reproducible, and executable.