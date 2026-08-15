## Revised plan summary

This revision keeps the user-requested correlator definition exactly intact: an anti-Bs0 source, a Ds+ sink, pseudoscalar `gamma5` interpolators at both ends, the spatial vector current `\bar c gamma_x b`, zero momentum, a point source at `[0,0,0,0]`, and `tseq = 8`, with no 2pt calculation.

The main corrections are structural and physical:

- The calculation is now explicitly labeled as a **raw executability / near-null test**, not a production semileptonic form-factor measurement. With both mesons at rest, a spatial vector current between pseudoscalar states is expected to be strongly suppressed at zero recoil.
- The **sequential charm propagator** is fixed correctly: it must be built from an **extended zero-momentum sink timeslice source at `t = 8`**, using the `Ds+` sink operator and the forward strange spectator propagator. It is **not** a literal point source at the sink.
- The measurement block is now **closed under the full contraction chain**. The final 3pt correlator depends on:
  - the forward bottom propagator,
  - the forward strange spectator propagator,
  - the charm sequential propagator,
  - the fixed sink projector and sink momentum,
  - and the `gamma_x` current insertion.
- The output payload is now explicit: save **one line per insertion time `tau = 1..7`**, with columns
  `tau  Re[C3(tau)]  Im[C3(tau)]`, no header, no extra text, in a `.txt` file in the run directory.

Two caveats are also made explicit rather than hidden:

1. Reusing the provided quark masses and clover coefficient together with stout-smeared inversions is treated here as a **user-mandated raw setup**, not as a re-tuned valence action.
2. The bottom quark is retained in the plan exactly as provided, but only as a **code-level heavy-quark test object**, not as a validated production heavy-quark treatment.

So the revised plan is executable, internally consistent, and faithful to the requested setup, while being honest that this specific kinematic choice is suitable mainly for a code check rather than a physically useful decay-amplitude extraction.