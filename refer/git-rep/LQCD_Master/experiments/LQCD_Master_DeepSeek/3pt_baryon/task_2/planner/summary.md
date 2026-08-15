## Physics Objective

Compute the connected three-point correlation function for the proton with an isovector axial current insertion on the u-quark line:

$$C_3(\tau) = \text{Tr}\left[P^+ \langle \mathcal{O}_p(\vec{0}, t_{\text{seq}}) \, J_A^x(\vec{z}, \tau) \, \bar{\mathcal{O}}_p(\vec{0}, 0) \rangle\right]$$

where:
- $\mathcal{O}_p = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) u^c$ is the standard proton interpolating operator,
- $J_A^x = \bar{u} \gamma_1 \gamma_5 u$ is the axial current in the x-direction,
- $P^+ = (1 + \gamma_4)/2$ is the positive-parity projector,
- $t_{\text{seq}} = 8$ is the sink time, and $\tau \in [1, 7]$ is the current insertion time.

This correlator is a building block for extracting the proton axial charge $g_A$ from the forward matrix element $\langle p | \bar{u}\gamma_\mu\gamma_5 u | p \rangle$.

**Disconnected diagrams are neglected** (connected-only approximation). For $g_A$, disconnected u-quark-loop contributions are expected at the few-percent level and represent an unquantified systematic uncertainty.

## Critical Correction: Four Contraction Topologies (Not Two)

The original plan incorrectly followed the $\Lambda \to p$ template, which assumes *one* unique source quark (strange) through the current. For proton→proton with $u \to u$ current, there are **two identical u-quarks at the source** (diquark member $u^d$ and spectator $u^f$) and **two at the sink** ($u^a$ and $u^c$), yielding $2 \times 2 = 4$ distinct Wick contraction topologies:

| Topology | Sink u→current | Current→source u | Spectator routing |
|----------|---------------|-----------------|-------------------|
| T1 | $u^a$ | $u^d$ | $u^c \leftrightarrow u^f$ |
| T2 | $u^a$ | $u^f$ | $u^c \leftrightarrow u^d$ |
| T3 | $u^c$ | $u^d$ | $u^a \leftrightarrow u^f$ |
| T4 | $u^c$ | $u^f$ | $u^a \leftrightarrow u^d$ |

All four must be included; omitting T2 and T4 (or T3 and T4) loses half the physical signal.

## Strategy: Single Sequential Source with 4-Topology Sum

1. **Forward propagator** (`prop_l_fwd`): Solve $D_l S_l(x; 0) = \eta_{\text{point}}$ with a point source at $[0,0,0,0]$, zero momentum, on stout-smeared gauge links.

2. **Sequential source construction**: At $t_{\text{seq}}=8$, construct the sequential source block $B$ by contracting the sink and source proton operators through all propagator lines *except* the current-carrying u-quark line. The block $B^{rf}_{\rho\lambda}$ carries open indices for the sink current-quark (color $r$, spin $\rho$) and source current-quark (color $f$, spin $\lambda$). **Crucially**, both source u-quarks contribute:

   $$B = B^{u^d}_{\text{direct}} + B^{u^d}_{\text{exchange}} + B^{u^f}_{\text{direct}} + B^{u^f}_{\text{exchange}}$$

   where "direct" means sink $u^c$ couples to the current and "exchange" means sink $u^a$ couples to the current. All four terms are summed into a single sequential source vector:

   $$\eta^{\text{seq}} = \gamma_5 B^\dagger \gamma_5$$

3. **Sequential propagator** (`prop_l_seq`): Solve $D_l G^{\text{seq}} = \eta^{\text{seq}}$ on the same stout-smeared gauge links. By linearity of the CG solver, $G^{\text{seq}}$ is the sum of the sequential propagators from each of the four topologies.

4. **Three-point contraction**: For each $\tau = 1,\ldots,7$, contract:

   $$C_3(\tau) = \sum_{\vec{z}} \text{Tr}\left[G^{\text{seq}}(\vec{z},\tau) \, \gamma_1\gamma_5 \, S_l(\vec{z},\tau)\right]$$

   The trace over color and spin correctly separates the four contributions without cross-contamination because each term in $B$ routes through distinct color-index paths preserved by the Kronecker deltas in the contraction.

## Technical Details

- **Gauge links**: Stout-smeared with $(n_{\text{steps}}=1,\ \rho=0.125,\ n_{\text{dim}}=4)$ before constructing the Dirac operator.
- **Solver**: Multigrid-preconditioned CG with two levels `[[6,6,6,3], [4,4,4,6]]`, tolerance $10^{-12}$, max 10000 iterations.
- **Clover coefficient**: $c_{\text{SW}} = 1.160920226$.
- **Quark mass**: Light quark mass parameter $-0.277$ (Wilson/clover convention).
- **Momentum**: Zero momentum at source, sink, and current insertion.
- **generate_einsum**: The tool's `baryon_3pt` mode must be verified for the proton→proton $u\to u$ case. The $\Lambda\to p$ mode (which assumes one unique source quark through the current) is incorrect here.

## Validation Check

Before running production, the sequential source construction should be validated: compute $C_3(\tau)$ using both the summed-sequential-source method and an explicit 4-term contraction from forward propagators only (no sequential inversion) on a single configuration. The two must agree to machine precision.