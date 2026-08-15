## Revised plan summary

The plan now fixes the measurement as a **specific, executable nonlocal D+ meson two-point correlator** rather than leaving the operator and output ambiguous.

### Physics object
It computes the connected zero-momentum correlator
\[
C(z,t)=\sum_{\vec x}\langle [\bar d(x,t)\gamma_5 W(x,x+z\hat z;U_{\rm orig}) c(x+z\hat z,t)] [\bar c(0)\gamma_5 d(0)]\rangle,
\]
for integer separations \(z=0,1,\dots,10\).

The nonlocality is placed **at the sink** so the requested displacement is applied **only to the charm-quark propagator**, not the anti-d leg. The Wilson line is a **straight ordered path along +z**, built from the **original unsmeared gauge field**. For \(z=0\), the Wilson line is the identity, giving the local D-meson correlator.

### Contraction and propagators
The measurement is now explicit at contraction level:
\[
C(z,t)=\sum_{\vec x} \mathrm{Tr}\left[S_l^\dagger(x,t;0)\, S_c^{\rm shift}(x,t;0,z)\right],
\]
with
\[
S_c^{\rm shift}(x,t;0,z)=W(x,x+z\hat z;U_{\rm orig})\,S_c(x+z\hat z,t;0).
\]

This makes clear that the implementation needs **both**:
- the forward light propagator from the point source, and
- the forward charm propagator from the same point source, from which the shifted charm line is constructed.

Only the connected diagram contributes because this is an open-flavor D+ channel.

### Gauge-link treatment
The mixed-link setup is now documented as an **intentional user-imposed definition**:
- **Dirac inversions** use a **stout-smeared** gauge field with parameters `(n_steps=1, rho=0.125, ndim=4)`.
- The **Wilson line** uses the **original unsmeared** gauge field.

The plan explicitly notes that this defines a **mixed-link nonlocal operator**, so the output should be treated as a raw correlator unless further matching/renormalization work is done.

### Execution details tightened
The revised plan fixes the previously ambiguous implementation details:
- source point is exactly `[0,0,0,0]`
- displacement is along **+z only**
- the Wilson line uses **forward path ordering** from `x` to `x+z zhat`
- periodic spatial wrapping is used in the z direction
- `z=0` is handled explicitly as `W=1`
- a sanity check is included: the `z=0` correlator must reproduce the local D correlator from the same propagators

### Output format
The txt output is now completely fixed and machine-parseable:
- filename: `dplus_nonlocal_2pt_cfg10000.txt`
- location: run directory
- no header, no comments, no metadata
- exactly **four numeric columns** per row:
  `z  t  ReC  ImC`
- row ordering is fixed as
  `(z=0,t=0..71), (z=1,t=0..71), ..., (z=10,t=0..71)`

### Scope
The revised plan also clarifies scientific scope: this is a **single-configuration, single-source raw correlator measurement**, appropriate as an implementation/data-production task, not a statistically validated D-meson physics result.