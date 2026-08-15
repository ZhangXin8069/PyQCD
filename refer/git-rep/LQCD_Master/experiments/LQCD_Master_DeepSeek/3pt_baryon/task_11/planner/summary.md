## Physics Objective

Compute the three-point correlation function for the flavour-changing weak transition **Ξ⁻ → Λ** mediated by the vector current $\bar{u}\gamma_x s$, a `baryon_3pt` observable that gives access to the vector form factor of this decay.

## Key Corrections from Peer Review

1. **Strange-quark mass corrected to ms = −0.2400** (was −0.2356). The solver now uses the sea strange mass matching the C24P29 ensemble path (`beta6.20_mu-0.2770_ms-0.2400_L24x72`), ensuring unitarity of the three-point function. The ensemble metadata block is preserved as provided.

2. **Sequential-source geometry clarified**: The sequential source is a **wall at t_seq = 8** constructed by summing over all spatial $\vec{x}$ at the sink time slice (zero-momentum projection), not a point source at $[0,0,0,8]$. The propagator source type is set to `sequential_wall` with `source_time: 8`.

3. **Two contraction topologies explicitly enumerated**: The Ξ⁻ source has two strange quarks in distinct spin-color roles — $s^{b}$ inside the $C\gamma_5$ diquark and $s^{c}$ as the spectator. The sequential source $B = B_1 + B_2$ sums both topologies (spectator-$s$→sink with diquark-$s$→current, and vice versa) with correct spin-index routing through the parity projector $T = (1+\gamma_4)/2$. The two-dagger convention $\eta^{\rm seq} = \gamma_5 B^\dagger \gamma_5$ is explicitly noted.

4. **Output fully specified**: $C_3(\tau)$ is saved for $\tau = 0, 1, \ldots, t_{\rm seq} = 8$ as a whitespace-separated column (9 lines, one value per $\tau$) in a plain-text file with no header or metadata.

## Strategy

- **Source hadron** (Ξ⁻, *dss*):  $\mathcal{O}_{\Xi^-} = \epsilon^{abc}\,(d^{Ta} C\gamma_5 s^{b})\,s^{c}$
- **Sink hadron** (Λ, *uds*):   $\mathcal{O}_{\Lambda}  = \epsilon^{abc}\,(u^{Ta} C\gamma_5 d^{b})\,s^{c}$
- **Inserted current**:  $J = \bar{u}\,\gamma_x\,s$
- **Parity projector**:  $T = (1+\gamma_4)/2$  (positive-parity channel)
- **Kinematics**: zero momentum at source, sink, and transfer ($\vec{p}_i = \vec{p}_f = \vec{q} = 0$)

The sequential-source method:
1. Invert a **forward light** propagator $S_l(x;0)$ (for the *d* quark) and a **forward strange** propagator $S_s(x;0)$ (for both *s* quarks) from a point source at $[0,0,0,0]$.
2. Construct the **sequential wall source** $\eta^{\rm seq}$ at $t_{\rm seq}=8$ from $S_l$, the Λ sink operator, $T$, and both $s$-quark topologies via $\eta^{\rm seq} = \gamma_5 B^\dagger \gamma_5$ with $B = B_1 + B_2$.
3. Solve for the **light sequential propagator** $G_l^{\rm seq}$ (the *u*-quark line).
4. Contract at each $\tau$: $C_3(\tau) = \sum_{\vec{z}}\operatorname{Tr}\!\big[G_l^{\rm seq}(z,\tau)\;\gamma_x\;S_s(z,\tau)\big]$.

## Technical Details

- **Source**: point source at $[0,0,0,0]$ (single source, cfg 10000 — validation setup, not production).
- **Gauge smearing**: stout smearing $(n_{\rm steps}=1,\,\rho=0.125,\,n_{\rm dim}=4)$ before every inversion.
- **Solver**: multigrid (two-level, C24P29 parameters), clover $c_{\rm sw}=1.160920226$, tolerance $10^{-12}$, maxiter 5000.
- **Valence masses**: $m_l = -0.277$, $m_s = -0.2400$ (sea-matched).
- **Output**: 9-line plain-text file with $C_3(\tau=0\ldots8)$, one value per line, no header/metadata.