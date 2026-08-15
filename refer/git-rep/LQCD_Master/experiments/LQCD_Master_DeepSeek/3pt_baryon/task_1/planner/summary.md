## Physics objective

Compute the three-point correlation function for a proton at zero momentum with an inserted flavour-diagonal vector current \(J = \bar{u}\,\gamma_x\,u\) (gamma index 1 in the DeGrand-Rossi basis).  The observable probes the proton vector charge (Dirac form factor \(F_1\) at \(Q^2=0\)) and serves as a baseline for future nucleon-structure calculations on ensemble C24P29 (24³×72, \(a\approx 0.1052\) fm, Wilson-clover, configuration 10000).

## Operators and conventions

- **Source and sink**: both use the standard proton interpolating operator  \(\mathcal{O}_p = \epsilon^{abc}\,(u^{Ta}\,C\gamma_5\,d^b)\,u^c\).
- **Projector**: \(T = (1 + \gamma_4)/2\) isolates the positive-parity ground-state channel.
- **Current**: \(\bar{u}\,\gamma_1\,u\) (vector, \(\gamma_x\)).
- **Sequential source**: the strict two-dagger convention is mandated — the sequential source is \(\eta^{\text{seq}} = \gamma_5 B^\dagger \gamma_5\), where \(B\) is the sink block constructed from the forward propagator at \(t_{\text{sink}}=8\).  Omitting this step yields an incorrect spin structure.

## Key revision: p→p antisymmetrisation

Unlike the \(\Lambda\to p\) reference (where the source contains exactly one strange quark coupled to the current), the proton has **two degenerate u quarks** in both source and sink.  This means:

1. The sink block \(B\) must be antisymmetrised over both u-quark attachment possibilities (direct + exchange).
2. The final 3pt contraction must include the exchange term where the current \(\bar{u}\gamma_1 u\) couples to the second source u quark.
3. The `generate_einsum` tool must be called with baryon-3pt parameters that explicitly specify the flavour-diagonal case with two indistinguishable light source quarks — not the \(\Lambda\to p\) pattern.

## Sequential-source workflow

1. **Forward light propagator** `prop_l_fwd`: CG solve from a point source at \([0,0,0,0]\) on stout-smeared gauge links (1 step, \(\rho=0.125\), 4‑dim), using a two-level multigrid preconditioner.
2. **Sink block** \(B\): built at \(t_{\text{sink}}=8\) from `prop_l_fwd` contracted with the proton sink operator and projector \((1+\gamma_4)/2\), antisymmetrised over both u-quark lines.
3. **Sequential source**: \(\eta^{\text{seq}} = \gamma_5 B^\dagger \gamma_5\).
4. **Sequential propagator**: \(D_l\,G^{\text{seq}} = \eta^{\text{seq}}\), same stout-smeared gauge field.  The object entering the final contraction is `G_l_seq_dag` (after the second dagger / conjugation / reordering).
5. **Three-point correlator**:  \(C_3(\tau) = \sum_{\vec{z}} \operatorname{Tr}\big[G^{\text{seq}}_{l,\text{dag}}(z,\tau)\;\gamma_1\;S_l(z,\tau)\big]\) for \(\tau = 0,1,\dots,8\).

## Critical consistency requirements

- The **same** `prop_l_fwd` variable must be used for constructing the sink block and inside the final 3pt trace, so all quark lines originate from the same point source at \([0,0,0,0]\).
- `G_l_seq_dag` is the daggered/conjugated sequential propagator, not the raw inversion result.  This matches the `Lambda_proton_formfactor.md` convention.
- Both the forward and sequential inversions use the identical stout-smeared gauge field.

## Computational set-up

- **Ensemble**: C24P29, configuration 10000.
- **MPI**: 4 ranks in the temporal direction (`process_grid: [1,1,1,4]`).
- **Solver**: CG, tolerance \(10^{-12}\), max 10 000 iterations.
- **Output**: raw \(C_3(\tau)\) values, one per line, written to `proton_3pt_vector_gammax_result.txt` with no header or metadata.
- Only the 3pt function is computed; no 2pt is produced.