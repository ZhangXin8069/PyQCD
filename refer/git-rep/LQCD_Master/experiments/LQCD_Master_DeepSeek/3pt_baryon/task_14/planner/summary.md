## Physics Objective

Compute the three-point correlation function for the flavour-changing weak decay $\Lambda_b \to \Lambda_c^+$ with an axial-vector current insertion $\bar{c}\,\gamma_1\gamma_5\,b$ (the $\mu=1$ spatial component).  All gamma matrices follow the DeGrand-Rossi Euclidean basis: $\gamma_1$–$\gamma_3$ (spatial), $\gamma_4$ (temporal), $\gamma_5 = \gamma_1\gamma_2\gamma_3\gamma_4$.  The positive-parity projector $P^+ = (1+\gamma_4)/2$ isolates the ground-state baryon channel.

## Sequential-Source Strategy

The three-point correlator is evaluated via the sequential-source method:

$$C_3(\tau) = \sum_{\vec{z}} \; \mathrm{Tr}\Big[ G_c^{\text{seq}}(\vec{z},\tau)\; \gamma_1\gamma_5\; S_b(\vec{z},\tau) \Big]$$

where:
- $S_b$ is the **forward bottom-quark propagator** from the point source at $(\vec{0},0)$ to the current insertion at $(\vec{z},\tau)$.
- $G_c^{\text{seq}}$ is the **charm-quark sequential propagator**, obtained by solving $D_c\,G_c^{\text{seq}} = \eta^{\text{seq}}$ with a full-volume sequential source at $t_f=8$.

The sequential source $\eta^{\text{seq}}$ is constructed from the $\Lambda_c^+$ sink block $B$ via the **two-dagger convention**:

$$\eta^{\text{seq}} = \gamma_5\, B^\dagger\, \gamma_5$$

where $B$ is built from the forward light propagator `prop_l` (providing the $u,d$ diquark lines) evaluated over **all spatial points** at $t_f=8$, with the positive-parity projector $P^+ = (1+\gamma_4)/2$ applied at the sink.

### Propagator Roles — Strict Separation

| Propagator | Quark | Source | Role in 3pt |
|------------|-------|--------|------------|
| `prop_l` | light (u/d) | point [0,0,0,0] | **Sink-block construction only.** Provides u,d diquark lines at t=8 for building the sequential source. Does NOT appear in the final trace. |
| `prop_b` | bottom | point [0,0,0,0] | **Current contraction.** Contracts directly with $\gamma_1\gamma_5$ and $G_c^{\text{seq}}$ in the final trace. |
| `prop_c_seq` | charm | sequential, full-volume at t=8 | **Sequential propagator.** Connects the current insertion (at $\tau$) to the $\Lambda_c^+$ sink (at $t_f=8$). |

## Key Corrections from Peer Review

1. **Sequential source is full-volume, not a point.** The `source_position` for `prop_c_seq` uses `[null, null, null, 8]` — the `null` spatial entries explicitly signal that the source spans the entire spatial lattice at $t=8$, not a single spatial location.

2. **Gamma matrices use canonical DeGrand-Rossi names.** The current is $\bar{c}\,\gamma_1\gamma_5\,b$ (not $\gamma_x$) and the projector is $(1+\gamma_4)/2$ (not $\gamma_t$). This eliminates any ambiguity in code generation.

3. **Two-dagger convention is stated explicitly.** The sequential source is built as $\eta^{\text{seq}} = \gamma_5 B^\dagger \gamma_5$, matching the established $\Lambda\to p$ reference.

4. **Forward propagator roles are separated.** `sink_block_propagators: [prop_l]` and `current_contraction_propagator: prop_b` are distinct fields in the correlator specification, removing any ambiguity about which propagator appears where.

## Technical Details

- **Ensemble**: C24P29 ($24^3\times 72$, $a \approx 0.1052$ fm, $N_f=2+1$ clover-improved Wilson fermions)
- **Gauge smearing**: 1 step stout smearing ($\rho=0.125$, 4-dimensional) applied before all Dirac inversions
- **Light-quark solver**: Multigrid with coarse-grid levels $[6,6,6,3] \to [4,4,4,6]$
- **Charm / bottom solvers**: Standard CG, tolerance $10^{-12}$; bottom quark uses `maxiter=20000` due to its large mass ($am_b = 1.5$)
- **Kinematics**: Zero momentum at both source and sink ($\vec{p}_i = \vec{p}_f = \vec{0}$, $\vec{q} = \vec{0}$)
- **Output**: Plain-text file with 9 correlator values ($\tau = 0,\dots,8$), no header or metadata