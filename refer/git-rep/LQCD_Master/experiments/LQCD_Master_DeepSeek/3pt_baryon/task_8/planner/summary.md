## Physics Objective

Compute the three-point correlation function for the $b \to c$ vector-current transition $\Lambda_b \to \Lambda_c^+$ via the inserted current $\bar{c}\gamma_x b$. This baryon 3pt function is a step toward the $\Lambda_b \to \Lambda_c$ form factors relevant for semileptonic decays.

## Wick-Contraction Topology: Single

**Critical correction from the previous plan:** Unlike the $\Lambda \to p$ case (where the proton sink contains two $u$ quarks and the current contains $\bar{u}$, producing two contraction topologies), the $\Lambda_b(udb) \to \Lambda_c(udc)$ channel with current $\bar{c}\gamma_x b$ has **exactly one** Wick-contraction topology:

| Field | Source ($\Lambda_b$) | Sink ($\Lambda_c$) | Current | Contraction |
|-------|----------------------|--------------------|----------|-------------|
| $b$   | yes (unique) | — | yes (unique) | $b$ (source) $\leftrightarrow$ $b$ (current) — unique |
| $c$   | — | yes (unique) | $\bar{c}$ (unique) | $c$ (sink) $\leftrightarrow$ $\bar{c}$ (current) — unique |
| $u$   | yes | $\bar{u}$ | — | $u$ (source) $\leftrightarrow$ $\bar{u}$ (sink) — unique |
| $d$   | yes | $\bar{d}$ | — | $d$ (source) $\leftrightarrow$ $\bar{d}$ (sink) — unique |

Every quark flavor appears exactly once, so the contraction is forced. The two-topology language from the $\Lambda \to p$ reference does **not** apply here.

## Strategy: Sequential Source Method

1. **Forward propagators** from a point source at $[0,0,0,0]$:
   - `prop_l_fwd`: light quark ($m_l = -0.277$, multigrid solver) — two copies for the $u,d$ diquark lines.
   - `prop_b_fwd`: bottom quark ($m_b = 1.5$, standard CG) — the heavy spectator quark, contracted only in the final step.

2. **Sequential propagator** `prop_c_seq`: charm quark ($m_c = 0.4159$, standard CG), built by:
   - Contracting the forward light propagators into the $\Lambda_c$ sink operator at $t_f=8$ with projector $T=(I+\gamma_4)/2$, forming the sink block $B$.
   - Constructing the sequential source via the two-dagger convention: $\eta^{\rm seq} = \gamma_5 B^\dagger \gamma_5$.
   - Solving $D_c\, G_c^{\rm seq} = \eta^{\rm seq}$ with source at $t=8$ (not $t=0$ — the `source_position` time coordinate `[0,0,0,8]` reflects the sink timeslice).

3. **Final contraction**: $C_3(\tau) = \sum_{\vec{z}} {\rm Tr}\big[G_c^{\rm seq}(\vec{z},\tau)\, \gamma_x\, S_b(\vec{z},\tau; 0,0)\big]$ for $\tau = 0,\dots,8$.

## Current Encoding

The current $\bar{c}\gamma_x b$ is encoded as `Current('b','c','gamma1')` following the convention from the `lqcd-physics-correlator` skill: `Current(quark, antiquark, gamma)` yields $\bar{\text{antiquark}}\,\gamma\,\text{quark}$. This must **not** be inverted to `Current('c','b','gamma1')`, which would produce the Hermitian conjugate $\bar{b}\gamma_x c$.

## Technical Details

| Item | Choice |
|------|--------|
| Source type | Point source at $[0,0,0,0]$ |
| Sink timeslice | $t_f = 8$ |
| Momentum | Zero (source, sink, and transfer; $q^2=0$) |
| Current | $\bar{c}\gamma_x b$, encoded `Current('b','c','gamma1')` |
| Projector | $T = (I + \gamma_4)/2$ (positive parity) |
| Gauge smearing | Stout, 1 step, $\rho=0.125$, 4-dim |
| Light solver | Multigrid (2-level), tol $10^{-12}$ |
| Bottom solver | Standard CG, tol $10^{-10}$, max 20000 iter |
| Charm solver | Standard CG, tol $10^{-10}$, max 15000 iter |
| Sequential source | Constructed at $t=8$, two-dagger convention |
| Wick topologies | Exactly 1 |
| Output | Plain text, one value per $\tau$, no header |

## Key Corrections from Previous Plan

1. **Single topology** — removed the incorrect claim of "both direct and exchange topologies as in Λ→p."
2. **Sequential source time** — `source_position` for `prop_c_seq` is $[0,0,0,8]$, not $[0,0,0,0]$, with an explicit warning that the sequential source lives at the sink timeslice.
3. **Current constructor** — explicitly mapped to `Current('b','c','gamma1')` with a note preventing flavor-argument inversion.