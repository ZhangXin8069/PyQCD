This revised plan keeps the original objective and structure, but makes the 3pt workflow physically explicit and executable. The calculation is the raw Euclidean `Lambda -> p` baryon three-point correlator at zero momentum with source operator `epsilon^{abc}(u^T C gamma_5 d)s`, sink operator `epsilon^{abc}(u^T C gamma_5 d)u`, local current `\bar u gamma_x s`, positive-parity projector `Tmat=(I+gamma_t)/2`, point source at `[0,0,0,0]`, and fixed `tseq=8`.

The propagation chain is now closed. The plan states that one must compute:
- one forward light propagator from the point source, reused for both `u` and `d` by isospin symmetry and for the two light lines in the proton sink block;
- one forward strange propagator from the same source, used on the current side;
- one light sequential propagator obtained by first building the proton sink block at zero momentum with `Tmat=(I+gamma_t)/2`, then forming the sequential source with the standard two-dagger convention `gamma_5 B^dagger gamma_5`, and finally inverting again in the light channel.

The scientific scope is also clarified: this run produces only the unnormalized Euclidean 3pt correlator for insertion times `t_ins = 1..7`. Because no 2pt functions or renormalization factors are included, the output is not by itself a form factor or physical decay amplitude.

To stay consistent with the fixed ensemble configuration provided for this rewrite, the ensemble block is preserved exactly, including the strange valence mass `-0.2356`. Since the gauge-file path label contains `ms-0.2400`, the plan now flags an explicit pre-production action check so that the clover action, mass convention, and stout-link inversion setup are verified rather than silently assumed.

The output contract is tightened for reproducibility: root rank only, plain `.txt` in the run directory, deterministic filename `c3pt_lambda_p_gamma_x_tseq8_cfg10000.txt`, four columns `tseq t_ins Re Im`, and no header or extra text.