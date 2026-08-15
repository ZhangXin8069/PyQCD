## Revised plan summary

This revision keeps the original objective and workflow intact but makes the rho-channel definition, inversion setup, and output contract more explicit and safer for production.

### Physics target
The plan now states explicitly that the calculation is for the **connected charged isovector rho channel**,
\(\rho^+ \sim \bar d\,\gamma_i\,u\), with \(i=x,y,z\). This is the standard connected vector-meson choice and avoids the disconnected-diagram ambiguity associated with neutral flavor combinations.

The correlator is the zero-momentum meson 2pt built from the three local vector interpolators \(\gamma_x,\gamma_y,\gamma_z\), averaged with weight \(1/3\).

### Propagator and inversion setup
The source remains exactly as requested:
- **point source** at **[0,0,0,0]**.

The inversion also remains as requested:
- use **stout-smeared links** with parameters
  - `n_steps = 1`
  - `rho = 0.125`
  - `ndim = 4`

The revised plan clarifies that this stout-smearing is being treated as part of the **valence inversion operator itself**, since that is what the task asked for. It also adds a necessary execution note: the listed clover/light-quark parameters should be confirmed as the intended **valence** parameters for C24P29 before running.

To avoid unnecessary cost inflation, the solver tolerance was relaxed from `1e-12` to a still-conservative **`1e-10`**, which is more appropriate for a single-configuration rho 2pt production step.

### Contraction definition
The plan now records more explicitly that the rho correlator is not a generic meson template but a **vector-channel contraction**, generated separately for:
- `gamma_x`
- `gamma_y`
- `gamma_z`

and only then averaged equally. This preserves the correct vector spin structure.

### Output
The output requirement is tightened operationally:
- one **headerless plain-text `.txt` file**
- stored in the **run directory**
- rows written as **`t Re Im`**

This keeps the file plain and header-free as requested, while fixing the earlier weakness of omitting the time index.

### Scope statement
The revised plan now clearly labels this as a **per-configuration data-production task only**. With one source and one configuration, it is suitable for pipeline generation/testing or per-config archive production, but **not** for a statistically meaningful rho-physics determination.