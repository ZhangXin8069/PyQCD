## Physics Objective

Compute the zero-momentum two-point correlation function for a charged kaon $K^+$ (pseudoscalar meson, flavor content $\bar{s}u$) on gauge configuration 10000 of the C24P29 ensemble. The correlator is used to extract the kaon ground-state mass.

## Correlation Function and Contraction

**Operator**: $\mathcal{O}_{K^+} = \bar{s}\,\gamma_5\,u$, the standard local pseudoscalar interpolator.

**Source**: Single point source at $[0,0,0,0]$ with zero injected momentum; zero-momentum projection at the sink.

After Wick contraction and $\gamma_5$-hermiticity, the correlator reduces to:

$$C_K(t) = \sum_{\vec{x}} \operatorname{Re}\operatorname{Tr}\big[ S_s^\dagger(\vec{x},t;\vec{0},0)\, S_l(\vec{x},t;\vec{0},0) \big]$$

where $S_l$ (`prop_l`, light $u$-quark) enters undaggered and $S_s$ (`prop_s`, strange $s$-antiquark) receives the dagger. The explicit real part guarantees a purely real-valued correlator, removing any machine-precision imaginary remnant that could appear on a single configuration.

**Contraction code generation**: The contraction is produced by the deterministic call `generate_einsum(type='meson_2pt', quark='u', antiquark='s', gamma='g5')`, which removes the positional ambiguity that existed in the original `propagator: [prop_l, prop_s]` list.

## Required Propagators

| ID       | Flavor  | Role        | Source | Position   | Gauge smearing      |
|----------|---------|-------------|--------|------------|---------------------|
| `prop_l` | light   | $u$-quark   | point  | [0,0,0,0]  | stout (1, 0.125, 4) |
| `prop_s` | strange | $s$-antiquark | point  | [0,0,0,0]  | stout (1, 0.125, 4) |

Both are inverted on stout-smeared gauge links ($\rho=0.125$, one iteration, 4-dim smearing) using the clover-improved Wilson Dirac operator ($c_{\text{SW}}=1.160920226$, $m_l=-0.277$, $m_s=-0.2356$) with a two-level multigrid solver (tolerance $10^{-12}$, max $10^4$ iterations).

## Changes from the Original Plan

1. **Measurement section**: The ambiguous `propagator: [prop_l, prop_s]` is replaced by explicit `quark_propagator: prop_l` and `antiquark_propagator: prop_s` fields, so code generators know unambiguously that `prop_s` receives the dagger.
2. **Contraction formula**: Changed from $\operatorname{Tr}[\ldots]$ to $\operatorname{Re}\operatorname{Tr}[\ldots]$ to explicitly discard finite-precision imaginary parts.
3. **Explicit `generate_einsum` call**: The exact invocation `generate_einsum(type='meson_2pt', quark='u', antiquark='s', gamma='g5')` is recorded in `extras` so the toolchain is invoked deterministically rather than relying on string parsing of flavor/operator fields.

## Output

A plain text file `corr_kaon_2pt.txt` in the run directory containing 72 real-valued numbers (one per time slice $T=0,\ldots,71$), no header or metadata.