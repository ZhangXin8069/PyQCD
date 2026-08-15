## Physics Objective

Compute the **connected** three-point correlation function for the $\Lambda \to \Lambda$ transition with a flavor-diagonal axial current insertion $J_1^A = \bar{s} \gamma_1 \gamma_5 s$, on a single configuration (cfg 10000) of the C24P29 ensemble ($24^3 \times 72$, $a = 0.1052$ fm, $N_f = 2+1$ Wilson-clover).

**Critical limitation**: The current is flavor-diagonal ($s \to s$), so the full matrix element $\langle \Lambda | \bar{s}\gamma_1\gamma_5 s | \Lambda \rangle$ contains both connected and disconnected diagrams. This plan computes only the connected contribution; the gauge-invariant disconnected $s$-quark loop (requiring stochastic volume sources) is omitted. The result is physically incomplete and should be labeled accordingly.

## Strategy

### Interpolating Operators
- **Source** ($t=0$, position $[0,0,0,0]$): $\mathcal{O}_\Lambda = \epsilon^{abc} (u^{Ta} C\gamma_5 d^{b}) s^{c}$. The Dirac conjugate (creation) form is $\bar{\mathcal{O}}_\Lambda = \epsilon^{def} \bar{s}^f (\bar{d}^e \gamma_4\gamma_5\gamma_4 C \bar{u}^{Td})$. After index transposition in the Wick contraction, $(\gamma_4\gamma_5\gamma_4 C)^T$ simplifies to $(C\gamma_5)$, matching the sink-side spin structure.
- **Sink** ($t=8$, position $[0,0,0,8]$): same annihilation operator, projected onto positive parity with $T = (1+\gamma_4)/2$.
- **Current**: $J_1^A = \bar{s} \gamma_1 \gamma_5 s$ (axial vector, $x$-component).

### Propagator Requirements (3 inversions total)
1. **Forward light** (`prop_l`): point source at $[0,0,0,0]$, $\vec{p}=0$, inverted on stout-smeared links ($\rho=0.125$, $n=1$, $d=4$). Provides the $u$ and $d$ quark lines entering the sink block.
2. **Forward strange** (`prop_s`): same source. Provides the $s$-quark line from the source to the current insertion point.
3. **Sequential strange** (`prop_s_seq`): sequential source constructed at $t=0$ from the $\Lambda$ sink block at $t=8$ via the two-dagger convention ($\eta^{\mathrm{seq}} = \gamma_5 B^\dagger \gamma_5$). One extra inversion on stout-smeared links. Provides the $s$-quark line from the current insertion to the sink. **Risk**: the `generate_einsum` `baryon_3pt` mode must handle same-flavor sequential sources; cross-check against a manual derivation if only validated for flavor-changing cases (e.g., $\Lambda \to p$).

### Contraction
At each current insertion time $\tau \in [0, 8]$:
$$C_3(\tau) = \sum_{\vec{z}} \mathrm{Tr}\left[ G_s^{\mathrm{seq}}(\vec{z},\tau)\, \gamma_1\gamma_5\, S_s(\vec{z},\tau) \right]$$

### Technical Details
- **Gauge smearing**: Stout, 1 step, $\rho=0.125$, 4-dimensional
- **Solver**: Multigrid with blocking $[6,6,6,3] \to [4,4,4,6]$, tolerance $10^{-12}$, max 20000 iterations
- **Quark masses**: $m_l = -0.277$, $m_s = -0.2356$ (Wilson-clover $\kappa$ convention)
- **Clover coefficient**: $c_{\mathrm{sw}} = 1.160920226$
- **MPI**: 4 ranks, process grid $[1,1,1,4]$
- **Gamma basis**: DeGrand-Rossi ($\gamma_x = \gamma_1$, $\gamma_t = \gamma_4$, $\gamma_5 = \gamma_1\gamma_2\gamma_3\gamma_4$)

## Important Caveats
1. **Contact terms**: $\tau=0$ and $\tau=8$ contain unphysical contact contributions. Only $\tau \in [1,\ldots,7]$ should be used for physics analysis.
2. **Disconnected omission**: The result is not the full matrix element. If the complete $\langle \Lambda | \bar{s}\gamma_1\gamma_5 s | \Lambda \rangle$ is needed, add a disconnected measurement using stochastic $Z_2/Z_4$ volume sources on the strange quark with a loop estimator $\mathrm{Tr}[\gamma_1\gamma_5 S_s(z,z)]$.
3. **Single configuration**: No error estimate possible. Production would require $\mathcal{O}(100+)$ configurations.
4. **Same-flavor sequential**: Verify `generate_einsum` output against a manual spin-color test before production.

## Output
Raw $C_3(\tau)$ values for $\tau = 0,\ldots,8$ written to `C3_axial_tau.txt` in the run directory, one float per line, no header.