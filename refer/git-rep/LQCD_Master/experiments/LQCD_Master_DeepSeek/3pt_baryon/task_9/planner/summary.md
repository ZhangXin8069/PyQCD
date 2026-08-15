## Physics Objective

Compute the three-point correlation function for the **Λ_b → Λ** transition via the flavour-changing **(b→s) vector current** `J_x = s̄ γ_x b` on the C24P29 ensemble (24³×72, a≈0.1052 fm, N_f=2+1+1).

This 3pt function is the lattice building block for extracting the **Λ_b → Λ form factors** that parametrise semileptonic decays such as Λ_b → Λ ℓ⁻ ν̄_ℓ.

## Critical Correction: B-block Topology

A central revision from the previous plan: Λ_b (udb) and Λ (uds) each contain **exactly one u-quark and one d-quark**. Consequently the sequential-source B-block has **only one contraction topology** — there is no exchange term. The previous plan incorrectly carried over the two-term structure from the Λ→p case, where the proton sink has two u-quarks. The correct B-block is:

```
B^{rf}_{σ,ρ}(0) = ε^{abc} ε^{def} · (Cγ₅)_{αβ} · (Cγ₅)_{α'β'} · T_{σ,ρ}
                  · S_{l,αα'}^{ad}(x,0) · S_{l,ββ'}^{be}(x,0)
```

where `r` = s-quark colour (open), `σ` = s-quark spin (open), `T = (1+γ₄)/2`, and `x` is summed over spatial positions at t=8. The b-quark does **not** enter B — it is connected through the separate forward propagator `prop_b` in the final contraction.

## Strange-Quark Mass: Partially Quenched

The gauge configurations were generated with sea strange mass **ms = −0.2400** (as encoded in the filename `beta6.20_mu-0.2770_ms-0.2400_L24x72`). The valence strange quark uses the tuned value **ms = −0.2356** that reproduces physical strange-hadron masses. This is standard partially-quenched practice and is documented in the plan. Both the ensemble metadata and the sequential-propagator solver use ms = −0.2356.

## Strategy

| Element | Specification |
|---------|--------------|
| **Source** | Λ_b: `ε^{abc} (u^{Ta} Cγ₅ d^b) b^c`, point source at [0,0,0,0], t=0 |
| **Sink** | Λ: `ε^{def} (u^{Td} Cγ₅ d^e) s^f`, at t_seq = 8 |
| **Current** | `J_x = s̄ γ_x b`, vector, flavour-changing b→s |
| **Projector** | `(1+γ₄)/2` — positive-parity channel |
| **Momentum** | Zero at both source and sink |

## Propagators (3 inversions per configuration)

| Propagator | Flavour | Mass | Source Type | Stout |
|------------|---------|------|-------------|-------|
| `prop_l` | light (u/d) | −0.277 | Point [0,0,0,0] | (1, 0.125, 4) |
| `prop_b` | bottom | 1.5 | Point [0,0,0,0] | (1, 0.125, 4) |
| `prop_s_seq` | strange | −0.2356 | Sequential from Λ at t=8 | (1, 0.125, 4) |

A single forward light propagator `prop_l` is reused for both u and d spectator contractions (isospin symmetry S_u = S_d = S_l).

## Sequential Source Construction

1. Compute forward light propagator `prop_l` (one inversion).
2. At t=8, contract the Λ sink operator with two copies of `prop_l` and the projector T to form the B-block (single topology — see formula above).
3. Apply the two-dagger convention: `η^{seq} = γ₅ B^† γ₅` → sequential source placed at t=0.
4. Invert the strange-quark Dirac operator: `D_s · prop_s_seq = η^{seq}`.

## Final Contraction

```
C₃(τ) = Σ_z  Tr[ prop_s_seq(z,τ) · γ_x · prop_b(z,τ) ]
```

Both propagators share the same (z,τ) coordinates. No momentum phase is needed (all momenta zero). No disconnected diagrams arise (flavour-changing current).

## Sign Verification

Before accepting results, verify that `C₃(τ)` has the expected sign at large Euclidean time for a positive-parity Λ_b→Λ transition with the `(1+γ₄)/2` projector. A sign error in the generated contraction (e.g. from an incorrect adaptation of the Λ→p einsum template) would invert the extracted vector form factor.

## Known Systematics

- **Heavy quark**: `m_b = 1.5` in lattice units corresponds to ~2.8 GeV, substantially lighter than the physical bottom quark (~4.18 GeV). This is a Fermilab-type setup; form factors carry a heavy-quark systematic.
- **Statistics**: A single point source at one time slice provides minimal statistics. For production, Gaussian-smeared sources at multiple time slices are recommended.
- **Partially quenched strange**: The valence strange mass (−0.2356) differs from the sea mass (−0.2400); the associated systematic is typically subdominant.

## Output

A plain-text file (no header) in the run directory containing raw `C₃(τ)` values for τ = 0, …, 72.