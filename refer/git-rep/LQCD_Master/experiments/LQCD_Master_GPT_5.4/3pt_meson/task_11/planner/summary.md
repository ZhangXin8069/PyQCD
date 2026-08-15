## Physics classification
This is a **matrix-element / weak-transition** LQCD task, not spectroscopy and not a pure-gauge observable. The target object is a **meson three-point correlator** for the semileptonic flavor-changing transition
\[
K^- (\bar u s) \to \pi^- (\bar u d)
\]
through the local vector current
\[
J_x = \bar d\,\gamma_x\, s.
\]

## Core physical content
- **Source hadron:** \(K^- = \bar u s\)
- **Sink hadron:** \(\pi^- = \bar u d\)
- **Interpolators:** pseudoscalar on both ends, \(\bar u\gamma_5 s\) and \(\bar u\gamma_5 d\)
- **Current:** local vector current \(\bar d\gamma_x s\)
- **Kinematics:** zero source momentum, zero sink momentum, hence zero momentum transfer in this setup
- **Correlator type:** **meson 3pt**
- **Wilson line / nonlocality:** none; this is a local current insertion
- **Renormalization:** not part of this task; compute the bare lattice 3pt only
- **2pt functions:** explicitly excluded by the user

## Correlator strategy
The correct numerical object is the Euclidean three-point function
\[
C_3(t_{\rm seq},\tau) = \langle O_{\pi^-}(t_{\rm seq})\, J_x(\tau)\, O_{K^-}^\dagger(0)\rangle
\]
with:
- source at \([0,0,0,0]\)
- \(t_{\rm seq}=8\)
- insertion times typically saved for \(\tau=1,\dots,7\)
- zero momentum at both source and sink

Because the sink hadron is fixed, the standard and economical implementation is the **sequential-source method**.

### Propagators required
A physically consistent completion of the task needs exactly these propagators:
1. **Forward light propagator** for the spectator anti-u line.
2. **Forward strange propagator** for the source strange quark line.
3. **Light sequential propagator** built from the fixed \(\pi^-\) sink block, because the current converts \(s \to d\) and the sink quark is light.

This is the standard meson-3pt structure for a flavor-changing transition with a fixed sink.

## Numerical choices fixed from the request
The plan uses exactly the user-specified choices:
- **Point source** at \([0,0,0,0]\)
- **Zero momentum**
- **Source-sink separation** \(t_{\rm seq}=8\)
- **Stout-smeared gauge links for all inversions** with parameters:
  - `n_steps = 1`
  - `rho = 0.125`
  - `ndim = 4`
- **No 2pt calculation**
- **Final output:** plain `.txt` in the run directory, no header

## Conservative completions I made
A few implementation details were not explicitly provided, so I completed them with standard production defaults:
- **Solver tolerance:** `1e-10`
- **Maximum iterations:** `10000`
- **Fermion action / inversion parameters:** clover masses and clover coefficient taken directly from the supplied ensemble block
- **Sequential save window:** insertion times `1..7`, excluding the source and sink contact points, which is the standard choice for a 3pt output table
- **Output schema:** rows as `tseq, t_ins, Re, Im`, plain text with no header

## Requirement satisfaction checklist
- Identified the task as a **standard meson 3pt matrix-element calculation**.
- Extracted source/sink hadrons, flavor flow, current structure, and zero-momentum kinematics.
- Chose a standard **sequential-source** numerical scheme suitable for PyQUDA code generation.
- Included the full provided **ensemble** block in the YAML.
- Avoided placeholders and used runnable defaults.
- Honored the instruction to **not compute 2pt** and to save only the final 3pt result to a **txt file in the run directory**.

## Toolchain note for downstream code generation
For contraction generation, the physics content corresponds to a `meson_3pt` setup with:
- `src_antiquark = u`
- `src_quark = s`
- `sink_antiquark = u`
- `snk_quark = d`
- `current_quark = s`
- `current_antiquark = d`
- `gamma_src = gamma5`
- `gamma_snk = gamma5`
- `gamma_cur = gamma_x`
- `tseq = 8`

That fixes the full propagator and contraction chain needed for the later PyQUDA script.