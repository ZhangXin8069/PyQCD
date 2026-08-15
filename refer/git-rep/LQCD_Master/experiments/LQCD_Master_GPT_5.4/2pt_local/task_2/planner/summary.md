## Revised plan summary

This revision keeps the original objective intact but fixes the physics and execution gaps.

The calculation is now defined unambiguously as the **connected zero-momentum pseudoscalar kaon two-point function** built from the local interpolating operator
\(\bar l \gamma_5 s\), where `light` denotes the degenerate up/down field. This avoids the previous inconsistency that mixed kaon channel labels and flavor assignments.

The measurement chain is now closed explicitly:
- one **light** point-source propagator from **[0,0,0,0]**,
- one **strange** point-source propagator from **[0,0,0,0]**,
- a **local-local gamma5–gamma5 meson contraction** using **both** propagators,
- a **spatial sum over sink positions** to project to **zero momentum**.

The stout-smearing requirement is also made operationally precise. The plan now states that the **Dirac operators themselves** for both light and strange inversions are defined on the **once-stout-smeared gauge field** with
- `n_steps = 1`
- `rho = 0.125`
- `ndim = 4`

rather than merely using smeared links in some auxiliary way. It also warns that multigrid/tuning must be consistent with that smeared operator.

To make the output executable, the plan now specifies a concrete per-configuration file:
- **`kaon_2pt_cfg10000.txt`**
- written in the **run directory**
- with rows formatted as **`t Re Im`**
- with **no header and no extra text**.

Because the provided fixed ensemble configuration includes only `cfg_num: ['10000']`, the revised plan states clearly that this is a **per-configuration measurement for cfg 10000**, rather than implying a statistically complete ensemble analysis.