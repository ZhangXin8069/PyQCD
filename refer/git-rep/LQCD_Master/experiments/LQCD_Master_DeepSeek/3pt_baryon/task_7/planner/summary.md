## Revised Plan: Ξ_c → Ξ Three-Point Function via c→s Vector Current

### Physics Objective

Compute the three-point correlation function for the semileptonic decay $\Xi_c \to \Xi$ with a $c \to s$ vector current insertion $J_x = \bar{s}\gamma_x c$. This is a **baryon three-point function** using the sequential-source method (`task_mode: standard`).

### Operators and Conventions

| Role | Baryon | Quark Content | Operator |
|------|--------|---------------|----------|
| Source (t=0) | $\Xi_c^+$ | $d s c$ | $\epsilon^{abc}(d^{Ta} C\gamma_5 s^b) c^c$ |
| Sink (t=8) | $\Xi^-$ | $d s s$ | $\epsilon^{def}(d^{Td} C\gamma_5 s^e) s^f$ |
| Current | — | $\bar{s} c$ | $\bar{s}\gamma_x c$ ($\gamma_1$ in DeGrand-Rossi) |
| Projector | — | — | $T = (I + \gamma_4)/2$ (positive parity) |

### Key Revisions (from Peer Review)

1. **Sink specification**: All forward propagators (`prop_l`, `prop_s`, `prop_c`) now specify `sink: {type: full}` instead of `type: point`. Point-to-all propagators must be saved at all spacetime points — the sequential source at t=8 requires `prop_l` and `prop_s` at all spatial positions, and the final contraction sums `G_seq(z,τ)` against `S_c(z,τ)` over all $\vec{z}$ at every $\tau$.

2. **Charm solver**: Changed from `cg` to `cgne`. The Wilson/clover Dirac operator $D$ is $\gamma_5$-Hermitian ($D^\dagger = \gamma_5 D \gamma_5$), **not** Hermitian positive-definite. Standard CG requires a Hermitian positive-definite matrix and will diverge on $D$. CGNE solves $D^\dagger D x = D^\dagger b$, which is Hermitian positive-definite and convergent. Max iterations raised to 20000 for the heavy charm quark.

3. **Two-topology sign derivation**: The $\Xi$ sink has two identical $s$ quarks ($s^e$ in the diquark and $s^f$ as external). The $\bar{s}$ in the current can contract with either:
   - **Topology 1**: $\bar{s}(z) \to s^f(x)$
   - **Topology 2**: $\bar{s}(z) \to s^e(x)$
   
   Relabeling $e \leftrightarrow f$ in Topology 2: $\epsilon^{dfe} = -\epsilon^{def}$ (epsilon antisymmetry). The fermion exchange $s^e \leftrightarrow s^f$ in the Wick ordering yields a factor $-1$. Product: $(-1) \times (-1) = +1$. **Both topologies contribute with the same sign**, exactly as in the proton 2pt and $\Lambda\to p$ 3pt cases.

4. **Sequential source specification**: Replaced non-standard specifiers with a complete narrative. The sequential source is a full-spatial-volume source at t=8, constructed via:
   $$B = \text{contract}\big(\text{prop}_l(x,8), \text{prop}_s(x,8), \mathcal{O}_\Xi, T\big), \quad \eta_\text{seq} = \gamma_5 B^\dagger \gamma_5, \quad D_s \cdot G_\text{seq} = \eta_\text{seq}$$
   Zero momentum at sink eliminates Fourier phase factors; the spatial sum over $\vec{x}$ projects to $\vec{p}_f = \vec{0}$.

5. **Projector notation**: Unified to $(I + \gamma_4)/2$ throughout (removed non-standard `gamma_0` notation; in DeGrand-Rossi $\gamma_0 = I$, but the canonical form uses $\gamma_4$).

### Propagators Required

| ID | Flavor | Source | Solver | Purpose |
|----|--------|--------|--------|---------|
| `prop_l` | light ($d$) | point [0,0,0,0] | multigrid | Connects source $d$ to sink $d$ |
| `prop_s` | strange | point [0,0,0,0] | multigrid | Connects source $s$ to sink $s$; also used in sink block |
| `prop_c` | charm | point [0,0,0,0] | **CGNE** | Connects source $c$ to current insertion |
| `seq_s` | strange | sequential at t=8 | multigrid | Connects current insertion to sink $s$ |

All inversions use stout-smeared gauge links ($n_\text{steps}=1$, $\rho=0.125$, $n_\text{dim}=4$) and tolerance $10^{-12}$.

### Final Contraction

$$C_3(\tau) = \sum_{\vec{z}} \text{Tr}\left[ G_\text{seq}(\vec{z},\tau) \, \gamma_x \, S_c(\vec{z},\tau) \right]$$

No Fourier phase factor (zero momentum). The projector $T = (I+\gamma_4)/2$ is already applied in the sink block $B$.

### Known Limitations (Documented)

- **Point source, single configuration**: Statistical precision is minimal. Production runs require $\mathcal{O}(100+)$ configurations and multiple source time slices.
- **No quark-field smearing**: Baryon three-point functions with point source suffer from severe excited-state contamination. A ground-state plateau may not be reached at $t_\text{seq}=8$ with the given lattice spacing.
- **Single $t_\text{seq}$**: No ability to check sink-time independence of the extracted matrix element.