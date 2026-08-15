## Physics Objective

Compute the zero-momentum two-point correlation function of the $\Xi_{cc}$ doubly-charmed baryon (quark content: $dcc$, $J^P = 1/2^+$) on the C24P29 ensemble ($24^3\times 72$, $a \approx 0.105$ fm, $m_\pi \approx 290$ MeV).

## Strategy

### Interpolating operator

The $\Xi_{cc}$ is created by the standard local operator

$$\mathcal{O}_{\Xi_{cc}} = \epsilon^{abc}\,(d^{Ta} C\gamma_5 c^b)\,c^c,$$

where the $dc$ pair at slots $a,b$ forms a scalar diquark with $C\gamma_5$ and the second charm quark at slot $c$ is the spectator. This is the standard operator for the ground-state $1/2^+$ channel.

### Quark-slot → propagator mapping (critical)

Unlike the proton ($uud$) where all three quarks are degenerate, the $\Xi_{cc}$ has two distinct masses:

| Slot | Quark | Propagator |
|------|-------|------------|
| $a$ | $d$ (light) | `prop_l` |
| $b$ | $c$ (charm, diquark partner) | `prop_c` |
| $c$ | $c$ (charm, spectator) | `prop_c` (reused) |

The `generate_einsum` call must explicitly receive `flavors=['light','charm','charm']` so that the correct propagator objects are contracted into the right slots. If the tool defaults to a three-degenerate-quark (proton) template, it will silently produce a wrong contraction.

### Wick contraction — exchange is b↔c, not a↔c

Because the two charm quarks are identical, the contraction yields two terms:

**Direct term:**
$$\epsilon^{abc}\epsilon^{a'b'c'}\,(C\gamma_5)_{\alpha\beta}\,(C\gamma_5)_{\alpha'\beta'}\,P^+_{\gamma'\gamma}\,S_{l\,\alpha\alpha'}^{aa'}\,S_{c\,\beta\beta'}^{bb'}\,S_{c\,\gamma\gamma'}^{cc'}$$

**Exchange term** (exchange of the two charm quarks at slots $b\leftrightarrow c$, then relabel $b'\leftrightarrow c'$ to absorb the fermion minus sign via $\epsilon$ antisymmetry):
$$\epsilon^{abc}\epsilon^{a'b'c'}\,(C\gamma_5)_{\alpha\beta}\,(C\gamma_5)_{\alpha'\gamma'}\,P^+_{\beta'\gamma}\,S_{l\,\alpha\alpha'}^{aa'}\,S_{c\,\beta\beta'}^{bb'}\,S_{c\,\gamma\gamma'}^{cc'}$$

> **Key difference from the proton template:** In the proton ($uud$), the exchange is between the two $u$ quarks at slots $a\leftrightarrow c$, and the propagator spin index $S_u^{aa'}_{\alpha\gamma'}$ differs between direct and exchange terms. In $\Xi_{cc}$, the exchange is between the two charm quarks at slots $b\leftrightarrow c$, so the light-quark propagator $S_l^{aa'}_{\alpha\alpha'}$ is identical in both terms — only the gamma-matrix contraction pattern changes ($P^+_{\gamma'\gamma}$ vs $P^+_{\beta'\gamma}$, and $(C\gamma_5)_{\alpha'\beta'}$ vs $(C\gamma_5)_{\alpha'\gamma'}$).

### Two-point function

Using a point source at $(0,0,0,0)$ and summing over the sink spatial volume:

$$C(t) = \text{Re}\left[\sum_{\vec{x}} \langle \mathcal{O}_{\Xi_{cc}}(\vec{x},t) \bar{\mathcal{O}}_{\Xi_{cc}}(\vec{0},0) \rangle\right]$$

Only the real part is taken — the zero-momentum baryon correlator is manifestly real, and any imaginary component is a lattice artifact or statistical noise.

## Technical Details

| Item | Choice | Rationale |
|------|--------|-----------|
| Source | Single point source at $(x,y,z,t)=(0,0,0,0)$ | Task specification |
| Gauge smearing | Stout, $n=1$, $\rho=0.125$, 4D | Improves signal; applied to all propagators |
| Light solver | Multigrid (tolerance $10^{-8}$, max 10000 iter) | Standard for light quarks |
| **Charm solver** | **CG** (tolerance $10^{-8}$, max 20000 iter) | **Must NOT use multigrid** — multigrid coarse-grid parameters are tuned for light quarks and will fail or converge incorrectly for heavy charm ($m=0.4159$) |
| Diquark gamma | $C\gamma_5$ (`Cg5`) | Scalar diquark for $1/2^+$ ground state |
| Parity projector | $P^+ = (1+\gamma_4)/2$ | Isolates positive-parity channel |
| Contraction tool | `generate_einsum(type='baryon_2pt', flavors=['light','charm','charm'], diquark_gamma='Cg5', diquark_slots=[0,1], projector='Pplus', projector_slot=2)` | Explicit non-degenerate flavor specification required |
| Output | `np.real(C(t))` written as plain text, one line per time slice | Downstream tools expect real-valued data |

## Concerns Addressed

1. **Non-degenerate flavor contraction:** The quark-slot → propagator mapping is explicitly specified. If `generate_einsum` lacks native support for mixed flavors, the contraction must be hand-coded or the tool extended.
2. **Charm solver type:** Explicitly set to CG, not multigrid. The light-quark multigrid solver's coarse-grid correction is optimized for near-critical ($m_l \approx -0.277$) quarks and will produce incorrect results for heavy charm.
3. **Exchange topology:** Corrected from the proton $a\leftrightarrow c$ pattern to the proper $b\leftrightarrow c$ exchange between the two identical charm quarks, with the correct gamma-structure in both terms.
4. **Real-part extraction:** `np.real()` is explicitly required before file output.