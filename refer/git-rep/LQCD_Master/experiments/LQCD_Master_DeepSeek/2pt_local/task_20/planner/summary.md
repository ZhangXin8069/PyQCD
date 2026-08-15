## Physics Objective

Compute the two-point correlation function of the doubly-charmed Omega baryon Ω_{cc} (flavor content: scc, JP = 1/2⁺) on the C24P29 ensemble (24³×72, a ≈ 0.105 fm, β = 6.20). The ground-state mass can be extracted from the large-time exponential decay of the correlator.

## Key Revisions from Peer Review

### 1. Flavor-to-propagator mapping (critical fix)
The original plan listed `propagators: [prop_s, prop_c]` without specifying which quark line uses which propagator. For a non-degenerate scc baryon this is ambiguous — blindly substituting the first propagator into all three quark lines would produce a physically wrong contraction. The revised plan adds an explicit `quark_lines` mapping:
- Line `a` (strange in diquark) → `prop_s`
- Lines `b` and `c` (charm in diquark + spectator charm) → `prop_c`

This ensures the executor generates the correct einsum with two distinct propagator types.

### 2. Diquark structure justification
The sc-diquark form ε^{abc}(s^{Ta} Cγ₅ c^b) c^c is retained per user specification. While the cc-diquark form ε^{abc}(c^{Ta} Cγ₅ c^b) s^c would treat the two identical charm quarks symmetrically at the operator level and likely yield better ground-state overlap, the sc-diquark choice is physically valid: the sum of direct and exchange Wick contractions restores full antisymmetrization. A note documents the alternative for future optimization.

### 3. Charm solver hardening
Multigrid near-null-space structure is optimized for light/strange quarks and may perform poorly on the heavy charm quark (κ = 0.4159). The charm solver is revised:
- Tolerance relaxed from 1×10⁻¹² → 1×10⁻¹⁰ (avoids chasing convergence that multigrid cannot deliver)
- Max iterations increased from 2000 → 4000 (provides headroom if convergence is slow)

The strange-quark solver is unchanged (tol = 1×10⁻¹², maxiter = 2000).

### 4. Strange quark mass documented
The ensemble metadata gives ms = -0.2356, but the configuration filename encodes ms = -0.2400. The plan uses -0.2356 (from metadata) and flags the discrepancy for verification before production.

### 5. Stout smearing ρ retained
ρ = 0.125 is marginally above the recommended range (0.08–0.12) but with only 1 smearing step the risk of over-smearing is negligible. Kept per user specification with a note.

### 6. Point source limitation documented
Gaussian/Wuppertal source smearing is recommended for baryon spectroscopy but the user specifies a point source. The risk of degraded ground-state overlap and lower statistical precision is documented.

## Strategy Summary

- **Operator**: ε^{abc} (s^{Ta} Cγ₅ c^b) c^c with parity projector (1+γ₄)/2
- **Source**: Point source at [0,0,0,0], no quark-field smearing
- **Gauge links**: 1-step stout smearing (ρ=0.125, ndim=4)
- **Propagators**: prop_s (κ=-0.2356, tol=1e-12, maxiter=2000), prop_c (κ=0.4159, tol=1e-10, maxiter=4000)
- **Solver**: Clover-improved Wilson Dirac operator, cSW = 1.160920226, multigrid with blocking [[6,6,6,3],[4,4,4,6]]
- **Contraction**: Baryon 2pt with flavor-aware quark-line mapping (s → prop_s, c → prop_c)
- **Output**: 72 time-slice values of C(t) as plain text to `omega_cc_2pt_cfg10000.txt`
- **Single configuration**: cfg 10000 only