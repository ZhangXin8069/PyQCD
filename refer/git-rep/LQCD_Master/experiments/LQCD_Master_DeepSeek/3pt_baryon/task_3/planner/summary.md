## Physics Objective

Compute the three-point correlation function for the $\Lambda$ baryon with a flavor-diagonal vector current insertion $J = \bar{s}\gamma_x s$, as a first step toward extracting the strange-quark contribution to the $\Lambda$ vector form factor.

## Revisions from Critique

Three targeted changes were made in response to the peer review:

1. **Solver tolerance relaxed to $10^{-10}$** (was $10^{-12}$). The original tolerance is atypically tight for multigrid-accelerated CG on a $24^3\times 72$ lattice and risks non-convergence within 10000 iterations, especially for the light quark at $\kappa=-0.277$. $10^{-10}$ is a standard production value that safely ensures convergence while preserving sufficient precision for a 3pt correlator on a single configuration.

2. **Explicit documentation of the $\Lambda$ contraction topology.** The $\Lambda$ (uds) has exactly one $u$ and one $d$ quark, unlike the proton (uud) which has two $u$ quarks. Consequently the sink block $B$ contains only a single direct contraction term — there is no exchange term. The plan now explicitly warns that `generate_einsum(type='baryon_3pt', ...)` must use the single-term topology for the $\Lambda$; blindly reusing a proton-style template with an exchange term would produce an incorrect sink block with wrong sign and spin structure.

3. **Risk note on $t_{\text{seq}}=8$.** On a 72-slice lattice, $t_{\text{seq}}=8$ is short for a baryon and will carry significant excited-state contamination. The plan acknowledges this physics risk and notes that a proper form-factor extraction would require multiple sink times and a 2pt function for ratio-method analysis. The user's explicit exclusion of the 2pt is preserved.

## Physical Strategy (unchanged core)

- **Operator**: $\mathcal{O}_\Lambda = \epsilon^{abc}(u^{Ta}C\gamma_5 d^b)s^c$, zero momentum, positive-parity projector $T = (I+\gamma_4)/2$ at both source and sink.
- **Three-point function** (sequential-source form):
  $$C_3(\tau, t_{\text{seq}}) = \sum_{\vec{z}} \mathrm{Tr}\left[G^{\text{seq}}_s(\vec{z},\tau)\,\gamma_x\,S_s(\vec{z},\tau; \vec{0},0)\right]$$
  where $G^{\text{seq}}_s$ is the sequential strange propagator and $S_s$ is the forward strange propagator.
- **Sequential source method**: The sink block $B$ is built at $t_{\text{seq}}=8$ from two copies of the forward light propagator $S_l$ (for $u$ and $d$), with the $s$-quark spin-color index left open. Only one contraction term exists (no exchange). Sequential source $\eta^{\text{seq}} = \gamma_5 B^\dagger \gamma_5$, solved via $D_s G^{\text{seq}}_s = \eta^{\text{seq}}$ (one additional strange-quark inversion per configuration).

## Technical Details

| Item | Specification |
|------|---------------|
| **Gauge ensemble** | C24P29: $24^3\times 72$, $a = 0.1052\,\mathrm{fm}$, $\beta=6.20$ |
| **Configurations** | cfg 10000 only |
| **Source** | Point source at $[0,0,0,0]$, zero momentum |
| **Sink time** | $t_{\text{seq}} = 8$ |
| **Current** | $\bar{s}\gamma_1 s$ (vector, $\gamma_x$) at all $\tau \in [0,8]$ |
| **Gauge smearing** | Stout: $\rho=0.125$, $n_{\text{steps}}=1$, 4-dim |
| **Light mass** | $m_l = -0.277$ (clover $c_{\text{SW}}=1.160920226$) |
| **Strange mass** | $m_s = -0.2356$ (same clover coefficient) |
| **Solver** | Multigrid: blocks $[6,6,6,3]$, $[4,4,4,6]$; tol $10^{-10}$, maxiter 10000 |
| **MPI** | 4 ranks, process grid $[1,1,1,4]$ |
| **Propagators** | 1 forward light, 1 forward strange, 1 sequential strange |
| **Output** | Raw 3pt correlator values per $\tau$, plain text, no header |

### Key warnings for execution

- The `generate_einsum` call for the $\Lambda$ sink block must use the **single-term (no-exchange) baryon_3pt topology**. The proton (uud) template with an exchange term is incorrect for $\Lambda$ (uds).
- Contact terms at $\tau=0$ and $\tau=t_{\text{seq}}$ are saved verbatim; the analysis stage should separate them from the physical signal region.
- $t_{\text{seq}}=8$ is short; the 3pt result alone cannot disentangle ground and excited states without a companion 2pt and/or multiple sink times.