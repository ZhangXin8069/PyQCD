## Physics objective

Compute the **three-point correlation function** for the decay $\Lambda_c^+ \to \Lambda$ via the **vector current** $\bar{s}\gamma_x c$ (flavour-changing $c\to s$ transition), using the sequential-source method.

## Operators

| Role   | Hadron       | Interpolating operator |
|--------|--------------|------------------------|
| Source | $\Lambda_c^+$ | $\epsilon^{abc}\,(u^{Ta}C\gamma_5 d^b)\,c^c$ |
| Sink   | $\Lambda$     | $\epsilon^{abc}\,(u^{Ta}C\gamma_5 d^b)\,s^c$ |
| Current| —            | $\bar{s}\,\gamma_x\,c$ (vector, $\mu=x$) |

Positive-parity projector: $P^+ = (1+\gamma_4)/2$.

## Contraction topology

Only **one Wick contraction** exists: the current connects distinct quark flavours ($c\to s$), so there is no exchange term (unlike the $\Lambda\to p$ case where the $u$ quark in the current can attach to either of two sink $u$ quarks).

## Propagators

| ID | Flavour | Source | Solver |
|----|---------|--------|--------|
| `prop_l` | light ($m=-0.277$) | Point at $[0,0,0,0]$ | Multigrid |
| `prop_c` | charm ($m=0.4159$) | Point at $[0,0,0,0]$ | CG |
| `prop_s_seq` | strange ($m=-0.2356$) | **Wall** sequential at $t=8$ | **BiCGstab** |

### Key corrections from peer review

1. **Sequential source is a wall source**: The B-block is constructed by summing over **all spatial points** $\vec{x}$ at the sink time slice $t=8$.  Calling it a point source at a single spatial site would lose nearly all the signal.  The `source_position: [0,0,0,8]` spatial entries are placeholders; the source spans the entire spatial volume.

2. **Explicit spatial sum in final contraction**: At each current-insertion time $\tau$, the correlator is
   $$C_3(\tau) = \sum_{\vec{z}} \mathrm{Tr}\!\bigl[G_{\rm seq}^{(s)}(\vec{z},\tau)\;\gamma_x\;S_c(\vec{z},\tau)\bigr]$$
   where the sum runs over **all spatial lattice sites** $\vec{z}$.  This is $O(V)$ larger than a single-site trace.

3. **Output time range**: Only $\tau = 0, 1, \dots, t_{\rm seq}\,(=8)$ are physically meaningful (current insertion must not lie after the sink).  The output file contains exactly 9 values, not all 72 time slices.

4. **BiCGstab for strange quark**: The strange Wilson mass ($-0.2356$) may render $D_{\rm Wilson}$ non-positive-definite, so plain CG could fail or converge incorrectly.  BiCGstab handles non-Hermitian systems robustly and is the standard choice for this mass regime.

## Sequential-source construction

The B-block encodes the $\Lambda$ sink structure:
- Forward `prop_l` provides the $(ud)$ diquark propagators from source to sink
- Diquark spin structures: $C\gamma_5$ at the sink, $\gamma_5 C$ at the source
- Projector $P^+ = (1+\gamma_4)/2$ applied across the sink $s$-quark and source $c$-quark spin indices
- Sequential source: $\eta_{\rm seq} = \gamma_5 B^\dagger \gamma_5$ (two-dagger convention)
- Inversion: $D_s\,G_{\rm seq} = \eta_{\rm seq}$ using BiCGstab

The `generate_einsum` tool (baryon_3pt type) produces the einsum strings for both the B-block construction and the final trace contraction.

## Technical details

- **Ensemble**: C24P29, $24^3\times72$, $a=0.1052$ fm, anti-periodic temporal BC
- **Stout smearing**: $n_{\rm step}=1$, $\rho=0.125$, $n_{\rm dim}=4$ on all gauge links
- **Solver tolerances**: $10^{-12}$; max iterations 5000 (multigrid), 10000 (CG/BiCGstab)
- **Only the 3pt function** is computed; no 2pt functions
- **Single configuration** (cfg 10000)