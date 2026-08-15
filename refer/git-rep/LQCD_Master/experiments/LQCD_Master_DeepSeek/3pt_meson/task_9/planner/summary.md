## Physical Objective

Compute the connected three-point correlation function for the Ds+ meson electromagnetic transition mediated by the charm-quark vector current \(\bar{c}\gamma_x c\). The Ds+ meson (flavor content \bar{s}c\)) is used at both source and sink with a \(\gamma_5\) interpolating operator, giving the charm-quark contribution to the Ds+ electromagnetic form factor at zero momentum transfer.

## Strategy — Corrected Sequential-Source Method

### Operators
- Source (creation): \(\mathcal{O}_{Ds}^\dagger = -\bar{c}\gamma_5 s\)
- Sink (annihilation): \(\mathcal{O}_{Ds} = \bar{s}\gamma_5 c\)
- Current insertion: \(J_x = \bar{c}\gamma_x c\)

The overall minus sign in the source operator is critical and must propagate through to the final contraction.

### Wick contraction (connected topology only)
The strange quark is the spectator, propagating directly from source to sink. The charm quark connects source → current → sink. The contraction yields:
\[
C_3(t_f,\tau,0) = -\sum_{\vec{x},\vec{z}} \operatorname{Tr}\big[\gamma_5 S_s(x;0)\gamma_5 \cdot S_c(0;z)\gamma_x S_c(z;x)\big]
\]
where the sink block \(B(x) = \gamma_5 S_s(x;0)\gamma_5\) encodes **both** the sink \(\gamma_5\) (from \(\bar{s}\gamma_5 c\)) and the source \(\gamma_5\) (from \(-\bar{c}\gamma_5 s\)) acting on the spectator line.

### Sequential-source construction (corrected)
1. **Two forward propagators** solved from a spin-color Kronecker-delta point source at the origin \((0,0,0,0)\):
   - `prop_s`: strange quark (\(m_s = -0.2356\))
   - `prop_c`: charm quark (\(m_c = 0.4159\))
2. **Sink block** at \(t_f = 8\): \(B(x) = \gamma_5 \cdot S_s(x,t_f) \cdot \gamma_5\) — this is the key correction from the original plan, which omitted the source \(\gamma_5\).
3. **Sequential source** via the two-dagger convention: \(\eta_{\text{seq}}(y) = \gamma_5 B^\dagger(y) \gamma_5\), placed as a **wall source spanning all spatial points** at \(t=8\) for correct zero-momentum sink projection.
4. **Sequential propagator** `prop_c_seq` solved with the charm-quark Dirac operator using \(\eta_{\text{seq}}\) as the source.
5. **Contraction** derived via `generate_einsum(type="meson_3pt", source_op=gamma_5, sink_op=gamma_5, current=gamma_x, active_quark=charm, spectator_quark=strange)`. The tool handles all gamma insertions, signs (including the \(-1\) from \(\mathcal{O}^\dagger\)), and spin-color index contractions correctly.

## Key Corrections from Original Plan

| Issue | Original | Corrected |
|-------|----------|-----------|
| Sink block | \(\gamma_5 \cdot S_s\) (missing source \(\gamma_5\)) | \(\gamma_5 \cdot S_s \cdot \gamma_5\) (both source and sink \(\gamma_5\)) |
| Sequential source spatial extent | Listed as point at \([0,0,0,8]\) | Wall source on time slice \(t=8\) spanning all spatial points |
| Forward source spin structure | Unspecified | Explicitly spin-color Kronecker delta (no gamma insertion) |
| Contraction formula | Hard-coded; inconsistent with source/sink operators | Deferred to `generate_einsum` for correct derivation |

## Technical Details

| Item | Specification |
|------|---------------|
| Lattice | \(24^3 \times 72\), \(a \approx 0.1052\) fm, isotropic |
| Gauge config | C24P29 ensemble, cfg 10000 |
| Clover coefficient | \(c_{\text{sw}} = 1.160920226\) |
| Link smearing | 1-step stout, \(\rho=0.125\), 4-dim, applied before all inversions |
| Forward source | Point source at \((0,0,0,0)\), spin-color Kronecker delta, zero momentum |
| Sequential source | Wall source at \(t_{\text{seq}} = 8\), constructed from corrected sink block |
| Current insertion | \(\bar{c}\gamma_x c\) (spatial x-component), \(\tau = 0,1,\dots,8\) |
| Solver | CG on normal equations, tolerance \(10^{-10}\), max 3000 iterations |
| Output | Plain text file: one line per \(\tau\) with `tau  Re[C_3]  Im[C_3]` |

## Scope
- **Connected only**: The disconnected charm-loop diagram (where the current couples to a charm loop) is not computed.
- **No 2pt**: Explicitly excluded per user request; the raw 3pt correlator is delivered without forming the ratio \(C_3/C_2\).
- **All \(\tau\) values**: The correlator is computed for \(\tau = 0,\dots,8\) to allow inspection of the full time dependence, including contact terms.