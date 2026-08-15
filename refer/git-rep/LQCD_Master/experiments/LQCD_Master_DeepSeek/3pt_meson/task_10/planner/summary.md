## Physics Objective

Compute the three-point correlation function for the semileptonic transition $\bar{B}_s^0 \to D_s^+$ mediated by the $b \to c$ vector current $\bar{c}\gamma_x b$. This is a prototype heavy-to-heavy flavour-changing matrix element relevant for $B_s$ semileptonic decays, evaluated on the C24P29 ensemble ($24^3\times 72$, $a\approx 0.105$ fm).

## Key Correction: Overall Sign

The original plan had $C_3 = -\mathrm{Tr}[G_{\text{seq}}\gamma_x S_b]$. A full re-derivation of the Wick contraction from the operator definitions through to the sequential-source trace gives **$C_3 = +\mathrm{Tr}[G_{\text{seq}}\gamma_x S_b]$** (i.e. the overall sign is **positive**, not negative). The derivation traces every minus sign:

1. **Source operator conjugate**: $\mathcal{O}_{\bar{B}_s^0} = \bar{s}\gamma_5 b$ → $\mathcal{O}_{\bar{B}_s^0}^\dagger = -\bar{b}\gamma_5 s$ → one minus sign.
2. **Wick contraction**: one closed fermion loop ($s \to \bar{s}$, $c \to \bar{c}$, $b \to \bar{b}$) contributes a factor $-1$.
3. **Product**: $(-1) \times (-1) = +1$.

The full trace after $\gamma_5$-hermiticity and cyclic permutation simplifies to $C_3 = +\sum_{\vec{x},\vec{z}} \mathrm{Tr}[S_s^\dagger(x;0)\,S_c(x;z)\,\gamma_x\,S_b(z;0)]$, which the sequential-source construction faithfully reproduces with a $+$ sign.

## Sequential Source Construction (Verified Correct)

The original plan's sequential source $\eta = \gamma_5 S_s \gamma_5$ is **correct** and was retained. Here is the explicit chain:

- **Sink block** (from the simplified trace $\mathrm{Tr}[S_s^\dagger S_c \gamma_x S_b]$): $B(x) = S_s^\dagger(x;0)$, defined on the $t=t_f=8$ time slice for all spatial points.
- **Two-dagger convention**: $\eta(x) = \gamma_5 B^\dagger(x) \gamma_5 = \gamma_5 S_s(x;0) \gamma_5$. By $\gamma_5$-hermiticity this is identically $\eta = S_s^\dagger(0;x)$.
- Solve $D_c G_{\text{seq}} = \eta$ (charm sequential propagator).
- Apply second dagger: $\tilde{G}_{\text{seq}} = \gamma_5 G_{\text{seq}}^\dagger \gamma_5$.
- Contract: $C_3(\tau) = +\sum_{\vec{z}} \mathrm{Tr}[\tilde{G}_{\text{seq}}(\vec{z},\tau) \gamma_x S_b(\vec{z},\tau;0)]$.

Both forms $\gamma_5 S_s \gamma_5$ and $S_s^\dagger$ are equivalent; the plan notes both for clarity.

## Propagator Requirements

1. **Forward strange** (`prop_s`): point source at $[0,0,0,0]$, multigrid-accelerated CG ($m_s=-0.2356$, tol $10^{-12}$). Used as spectator and to construct the sequential source.
2. **Forward bottom** (`prop_b`): point source at $[0,0,0,0]$, standard CG ($m_b=1.5$, tol $10^{-12}$). **Caveat**: bare $am_b \approx 1.5$ at $a \approx 0.105$ fm is far above the physical bottom scale; discretisation errors may be substantial. For a production run a Fermilab or RHQ action would be preferred.
3. **Sequential charm** (`prop_c_seq`): sequential source at $t=8$ built from `prop_s` via the two-dagger prescription, solved with CG ($m_c=0.4159$, tol $10^{-12}$).

## Limitations Noted But Retained As Specified

- **Single configuration** (cfg 10000): no statistical error estimate possible. An outlier gauge fluctuation or solver convergence issue would go undetected. Running on 4–10 configurations is recommended for any physics conclusion.
- **Bottom mass**: the partially-quenched $m_b=1.5$ is heavy; results should be treated as exploratory.
- **No 2pt functions**: as explicitly requested, only the 3pt is computed.

## Output

Plain two-column ASCII file (`tau`, `Re[C3(tau)]`) written to the run directory with no header or metadata.