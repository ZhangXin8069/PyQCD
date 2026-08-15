## Physics Objective

Compute the three-point correlation function for the weak decay $\Lambda \to p$ with an axial-vector current insertion $J_\mu^A = \bar{u} \gamma_\mu \gamma_5 s$, specifically the $\mu = x$ component ($\gamma_x \gamma_5$). Both source and sink are at rest ($\vec{p} = \vec{0}$), giving zero momentum transfer.

## Strategy

**Operators:**
- Source ($\Lambda$, $t=0$): $\mathcal{O}_\Lambda = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) s^c$
- Sink ($p$, $t=8$): $\mathcal{O}_p = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) u^c$
- Current: $J = \bar{u} \gamma_x \gamma_5 s$
- Parity projector: $T = (I + \gamma_t)/2$ for positive-parity channel

**Sequential-source method:**
1. Solve two forward propagators from a point source at $(0,0,0,0)$: $S_l$ (light) and $S_s$ (strange).
2. At $t_{\text{seq}} = 8$, construct the sequential source $\eta^{\text{seq}}$ from the forward light propagators spanning the **full spatial volume** using the proton sink operator and the projector $T$, following the two-dagger convention ($\eta^{\text{seq}} = \gamma_5 B^\dagger \gamma_5$).
3. Solve the sequential light propagator $G_l^{\text{seq}}$ from $\eta^{\text{seq}}$.
4. Contract at each intermediate time $\tau$ using **both** the sequential light and forward strange propagators: $C_3(\tau) = \sum_{\vec{z}} \operatorname{Tr}[G_l^{\text{seq}}(\vec{z},\tau) \, \gamma_x\gamma_5 \, S_s(\vec{z},\tau)]$.

## Key Revisions from Original Plan

### Fix 1: Measurement section now lists both required propagators
The original plan listed only `prop_l_seq` under `measurement.correlators.propagator`. However, the 3pt contraction formula requires both the sequential light propagator *and* the forward strange propagator. The revised plan uses `propagators:` (plural) as a list containing both `prop_l_seq` and `prop_s_fwd`, so the code generator correctly includes both operands in the trace.

### Fix 2: Sequential source spatial extent clarified
The original plan specified `source_position: [0, 0, 0, 8]` for the sequential source, which could be misread as a single-site point source. In reality, the baryon 3pt sequential source is constructed from forward propagators at **all spatial points** on the $t=8$ time slice — it is a volume source, not a point. The revised plan adds `spatial_extent: full` to the source specification and expands the solver notes to explicitly state that the sequential source spans the entire spatial volume and that the spatial coordinates in `source_position` are placeholders.

## Technical Details (Unchanged)

- **Ensemble:** C24P29 ($24^3 \times 72$, $a \approx 0.1052$ fm, $m_\pi \approx 290$ MeV)
- **Configuration:** Single configuration (cfg 10000) — suitable for a test / validation run
- **Gauge smearing:** Stout-smeared links $(n=1, \rho=0.125, n_{\text{dim}}=4)$ applied to all three inversions
- **Solver:** Multigrid, $c_{\text{SW}} = 1.160920226$, tolerance $10^{-12}$, max 10000 iterations
- **Output:** Plain-text file with $C_3(\tau)$ values, no headers, saved in the run directory
- **Statistics note:** A single point source on one configuration is a minimal test setup; production-quality results would require multiple sources per configuration and more configurations. This is preserved as the user's explicit choice.