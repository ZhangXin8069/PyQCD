## Revised plan summary

This revision keeps the requested computation intact at the implementation level: a **Ds+ \u2192 Ds+ meson three-point correlator** with
- source and sink interpolators \(\bar s\gamma_5 c\),
- local current \(\bar c\gamma_x c\),
- zero source and sink momentum,
- point source at `[0,0,0,0]`,
- `tseq = 8`,
- stout-smeared inversion links `(n_steps=1, rho=0.125, ndim=4)`,
- and **no 2pt function**.

The key correction is **interpretation**. With a pseudoscalar initial and final state at **zero momentum**, the **spatial** vector-current matrix element is expected to vanish in the continuum decomposition. Therefore this job should **not** be described as a physical electromagnetic form-factor determination. It is instead a **raw connected charm-current 3pt correlator production / code-validation run**, where any nonzero result on one configuration is to be interpreted as noise and/or lattice artifact, not as a physical signal.

The plan also now states explicitly that this is **not the full electromagnetic current**. It includes only the **connected charm-current insertion**. It does **not** include the strange connected current contribution, and it does **not** include flavor-diagonal disconnected loops. In addition, the output is a **bare unrenormalized 3pt correlator**: there is no 2pt normalization and no \(Z_V\) factor in this task.

Numerically, the workflow remains the standard **meson sequential-source 3pt** chain:
1. solve the forward strange propagator,
2. solve the forward charm propagator,
3. build the zero-momentum Ds sink block at `tseq=8`,
4. solve the sequential charm propagator,
5. contract with the local \(\gamma_x\) current,
6. save only the insertion-time window `t=1..7` to a plain `.txt` file with columns
   `(t_seq, t_ins, Re, Im)` and no header.

The plan now also labels the statistics honestly: **one configuration (`cfg 10000`) and one source** make this a **smoke test**, not a credible physics measurement. Finally, it adds a heavy-quark caveat: the charm setup is used here only for raw correlator production, and no claim of controlled charm discretization systematics is made in this task.