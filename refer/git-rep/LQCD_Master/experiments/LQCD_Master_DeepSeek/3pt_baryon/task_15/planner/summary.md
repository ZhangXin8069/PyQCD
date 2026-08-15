## Physics Objective

Compute the three-point correlation function for the \(\Xi^- \to \Lambda\) weak transition mediated by the axial-vector current \(J = \bar{u} \gamma_1 \gamma_5 s\) (flavor change \(s \to u\), spatial direction \(x\)). This is a \(\Delta S = 1\) baryon three-point function relevant for hyperon semileptonic decay form factors.

## Physical Setup

- **Source hadron**: \(\Xi^-\) (dss), operator \(\mathcal{O}_{\Xi} = \epsilon^{abc} (d^{aT} C\gamma_5 s^b) s^c\), placed at \(t=0\) with point source at spatial origin.
- **Sink hadron**: \(\Lambda\) (uds), operator \(\mathcal{O}_{\Lambda} = \epsilon^{def} (u^{dT} C\gamma_5 d^e) s^f\), at \(t_{\text{seq}} = 8\).
- **Current**: \(J = \bar{u} \gamma_1 \gamma_5 s\) (axial-vector, \(s \to u\) flavor change), inserted at intermediate time \(\tau \in [1, 7]\).
- **Projector**: \(T = (1 + \gamma_4)/2\) selects the positive-parity channel at the sink.
- **Momentum**: Zero momentum at both source and sink (\(\vec{p}_i = \vec{p}_f = \vec{0}\), \(\vec{q} = \vec{0}\)).
- **Gauge smearing**: Stout smearing with \(n_{\text{steps}}=1\), \(\rho=0.125\), \(n_{\text{dim}}=4\) applied to all inversions.

## Critical Revision: Source-Side Antisymmetrization

**The original plan missed a key physical effect.** The \(\Xi^-\) source has **two identical strange quarks** (positions \(b, c\) in \(\epsilon^{abc}\)), unlike the Lambda→proton reference where antisymmetrization occurs at the sink. The Wick contraction for \(\Xi^- \to \Lambda\) produces **two terms with opposite Fermi sign**:

1. **Term A**: sink \(s^f\) contracts with source \(\bar{s}^b\), current \(s^r\) contracts with source \(\bar{s}^c\)
2. **Term B**: sink \(s^f\) contracts with source \(\bar{s}^c\), current \(s^r\) contracts with source \(\bar{s}^b\) (minus sign from \(b \leftrightarrow c\) exchange)

Both terms must be included in the sequential-source construction. The final contraction form remains \(C_3(\tau) = \sum_{\vec{z}} \operatorname{Tr}[G^{\text{seq}}_u(\vec{z},\tau) \, \gamma_1\gamma_5 \, S_s(\vec{z},\tau)]\), but the sequential propagator \(G^{\text{seq}}_u\) encodes both antisymmetrized contributions.

### BaryonOp Mapping

| Role | BaryonOp |
|------|----------|
| Source \(\Xi^-\) | `BaryonOp('xi_minus', {a:'d', b:'s', c:'s'}, 'Cg5')` |
| Sink \(\Lambda\) | `BaryonOp('lambda', {d:'u', e:'d', f:'s'}, 'Cg5')` |
| Current | `Current('s', 'u', 'g1g5')` |

The sequential propagator opens flavor position **d** (the \(u\) quark) at the sink. Flavor index \(e\) (\(d\) quark) contracts with forward `prop_l`; flavor index \(f\) (\(s\) quark) contracts with forward `prop_s`.

## Sequential Source Clarification

The sequential source is a **full-spatial-volume object** constructed at time slice \(t=8\), not a point source. The `source_position: [0,0,0,8]` in the propagator specification indicates only the time slice; the source spans all \((x,y,z)\). The two-dagger convention is used: \(\eta^{\text{seq}} = \gamma_5 B^\dagger \gamma_5\), then \(D_l G^{\text{seq}} = \eta^{\text{seq}}\).

## Computational Cost

Three inversions per configuration (two forward + one sequential), matching the original estimate. The sequential source construction and contraction adds negligible overhead relative to the inversions.

## Diagnostics & Verification

- Print \(\|\eta^{\text{seq}}\|_2\) after sequential source construction to verify non-vanishing.
- Compute a sample contraction \(\operatorname{Tr}[G^{\text{seq}}(z,\tau=4) \, \gamma_1\gamma_5 \, S_s(z,\tau=4)]\) as a sanity check.
- **Before running**: verify that `generate_einsum` supports the `dss(Ξ⁻)→uds(Λ)` topology with source-side antisymmetrization. If the tool was only validated on Lambda→proton (sink-side antisymmetrization), the einsum must be derived manually from the Wick-contraction steps documented in the plan extras.

## Output

Plain text file with \(C_3(\tau)\) values for \(\tau = 1,\dots,7\), one per line, saved to the run directory. No two-point functions are computed.