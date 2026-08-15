## Physics Objective

Compute the three-point correlation function for the semileptonic decay Ξ_cc++(ucc) → Ξ_c+(usc) mediated by the (c→s) vector current. This matrix element is needed for extracting the Ξ_cc → Ξ_c form factors and ultimately the CKM matrix element |V_cs|.

## Key Revisions

**1. Sequential source type corrected (critical).** The sequential source for a zero-momentum three-point function must be a **volume source** spanning the entire spatial slice at t=8 — not a point source at a single spatial location. The sink block B is summed over all spatial positions x at t=8 before constructing η_seq = γ₅ B† γ₅. Using a point source would produce an incorrect sequential propagator and wrong correlator.

**2. Wick contraction topology validated.** Two contraction topologies exist for the Ξ_cc→Ξ_c three-point function, corresponding to the current charm quark attaching to either the source spectator charm (color index c) or the source diquark charm (color index b). After relabeling b↔c in the second topology, the epsilon antisymmetry cancels the relative minus sign, and both topologies are identical because both charm quarks reuse the same prop_c. Only one topology needs implementation with a factor of 2. This mirrors the Λ→p case but with the simplification that both charm propagators are the same object.

**3. Parity projector placement clarified.** The projector T = (1+γ₄)/2 acts on the **spectator charm quark line** — connecting the source spectator charm (color index c in ε^{abc}, spin γ') to the sink spectator charm (color index f in ε^{def}, spin γ). It must NOT be applied to the strange quark (diquark partner) line. The projector is absorbed into the sequential-source B-block construction.

**4. prop_c dual-use documented.** The forward charm propagator prop_c serves two independent roles, read at different time slices with distinct spin-color index contractions: (a) spectator line connecting source→sink at t=8 for the B-block, and (b) annihilating line connecting source→current at t=τ for the final contraction. No additional charm inversion is needed.

**5. Solver tolerances relaxed.** Light and strange quark solver tolerances reduced from 1e-12 to 1e-10 with maxiter 10000, reflecting practical convergence limits on a 24³×72 stout-smeared clover ensemble at these masses. Convergence must be monitored and a warning raised if tolerance is not reached.

## Propagators Required (3 total)

| ID | Flavor | Type | Source | Mass | Tolerance | Maxiter |
|----|--------|------|--------|------|-----------|--------|
| prop_l | light | forward | point [0,0,0,0] | −0.277 | 1e-10 | 10000 |
| prop_c | charm | forward | point [0,0,0,0] | 0.4159 | 1e-10 | 5000 |
| prop_s_seq | strange | sequential | volume, t=8 | −0.2356 | 1e-10 | 10000 |

All propagators use stout-smeared gauge links (1 step, ρ=0.125, 4-dim).

## Contraction Summary

1. Forward propagators S_l(x,0) and S_c(x,0) computed from point source at origin.
2. Sink block B^{pe}_{ρβ} constructed at t=8 by summing over spatial x, encoding the Ξ_c+ sink operator, Cγ₅ diquark structures, T projector on the spectator charm line, and free strange-quark indices.
3. Sequential source: η_seq = γ₅ B† γ₅ (volume source at t=8).
4. Sequential strange propagator: D_s · G_s_seq = η_seq.
5. Final correlator: C₃(τ) = Σ_z Tr[G_s_seq(z,τ) · γ_x · S_c(z,τ)], τ = 1,…,7.

## Output

Seven numbers (C₃(τ) for τ=1…7) written as whitespace-separated values to a single .txt file in the run directory, no header, no metadata.