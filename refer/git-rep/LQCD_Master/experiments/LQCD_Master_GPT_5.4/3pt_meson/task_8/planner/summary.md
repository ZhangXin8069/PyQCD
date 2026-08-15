## Revised plan summary

This revision keeps the original objective and setup, but makes the physics content explicit and executable.

### What is actually computed
The plan now states unambiguously that the calculation is a **bare meson 3pt correlator** for
\(D_s^+ \to \phi\) with
- source at rest,
- sink at rest,
- local current \(\bar s\gamma_x c\),
- source-sink separation `tseq = 8`.

Because both hadrons are at zero spatial momentum, the correlator is at **zero spatial momentum transfer** and corresponds to **\(q^2_{\max} = (m_{D_s}-m_\phi)^2\)**, **not** to \(q^2=0\).

### Wick-contraction content and sequential choice
The revised plan now documents the connected flavor flow explicitly:
- source operator: \(O_{D_s}=\bar s\gamma_5 c\), so at the source one uses \(O_{D_s}^\dagger\),
- sink operator: \(O_\phi=\bar s\gamma_x s\),
- current: \(J_x=\bar s\gamma_x c\).

For the **connected** topology,
- the source strange line is the spectator and runs directly to the sink anti-strange field,
- the source charm line propagates to the current,
- the current converts charm to strange,
- that strange line then propagates from the current to the sink.

This confirms that the chosen sequential inversion is a **strange sequential propagator built from the phi sink block**.

### Connected-only approximation made explicit
The sink \(\phi\) operator \(\bar s\gamma_x s\) is flavor singlet in the valence description, so a disconnected sink-loop contribution exists in principle. The revised plan does **not** include that piece. Instead, it now labels the computation clearly as **connected-only** and notes that this is an approximation appropriate only if the deliverable is the connected bare correlator, not the full flavor-singlet channel.

### Valence-action caveat clarified
The task requires all inversions to use **1-step stout-smearing with** `(1, 0.125, 4)`, and the revised plan preserves that. However, it now also records the important caution that the strange/charm bare masses and clover coefficient are being reused from the ensemble metadata rather than demonstrated to be retuned for that exact stout-smeared valence operator. Therefore the output is described as an **exploratory bare-valence correlator**, especially for charm.

### Insertion-time policy fixed
The previous version did not define the insertion window precisely enough. The revised plan now specifies:
- save only **`tau = 1, ..., 7`**,
- exclude contact terms at **`tau = 0`** and **`tau = tseq = 8`**,
- do not save times outside the source-sink interval.

This makes the output block directly executable and avoids ambiguity about wrap-around or endpoint contamination in the saved file.

### What is preserved from the original plan
The revised plan keeps the requested core setup unchanged:
- ensemble `C24P29`, cfg `10000`, `24^3 x 72`,
- point source at `[0,0,0,0]`,
- zero momentum source and sink,
- source interpolator `gamma_5`,
- sink interpolator `gamma_x`,
- current `sbar gamma_x c`,
- `tseq = 8`,
- stout-smeared inversion links `(1, 0.125, 4)`,
- no 2pt functions,
- plain txt output in the run directory with no header.

So the revised version is still close to the original plan, but it is now physically tighter, clearer about approximations, and more ready for implementation.