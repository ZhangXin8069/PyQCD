## Physics Objective

Compute the three-point correlation function for the flavour-changing transition **B⁻ → K⁻** mediated by the vector current **s̄γₓb** (b → s).  This matrix element is relevant for semileptonic B → K form-factor extractions.

## Hadrons and Operators (corrected)

| Role   | Meson | Quark content | Interpolator | generate_einsum       |
|--------|-------|---------------|-------------|-----------------------|
| Source | B⁻    | b ū           | O†_{B⁻}=b̄γ₅u  | quark=b, antiquark=u, gamma=g5 |
| Sink   | K⁻    | s ū           | O_{K⁻}=ūγ₅s   | quark=s, antiquark=u, gamma=g5 |
| Current| —     | s̄b           | J = s̄γₓb     | quark=s, antiquark=b, gamma=gx |

The **u quark** is the common spectator.  The **b quark** at the source transitions to an **s quark** at the sink through the current.  With three distinct quark flavours the Wick contraction closes exactly — one connected diagram, no disconnected pieces.

## Key Corrections from the Previous Plan

1. **Hadron flavour fixed**: B⁻ = b ū (not b̄u), K⁻ = s ū (not s̄u).  The previous assignment gave fields that could not be paired under Wick’s theorem.
2. **Interpolating operators explicitly defined**:  O†_{B⁻}=b̄γ₅u (creates B⁻), O_{K⁻}=ūγ₅s (annihilates K⁻).  This provides the exact quark/antiquark roles needed by `generate_einsum`.
3. **Sequential-source construction corrected**: The spectator u quark propagates **backward** (sink → source).  The sequential source is built via the two-dagger convention: S_u(0;x)=γ₅S_u†(x;0)γ₅ is combined with the sink γ₅ structure to form the sink block B(x), then η_seq = γ₅·B†·γ₅.  The previous plan incorrectly assumed forward propagation.
4. **Zero-momentum warning**: At **p = 0** the spatial component ⟨K⁻|s̄γₓb|B⁻⟩ vanishes by rotational symmetry.  The three-point correlator will be statistically consistent with zero.  A non‑zero signal requires either non‑zero momentum or the temporal current γ₄.  This is flagged prominently.

## Strategy: Sequential Source Through the Sink

1. **Forward propagators** from a point source at `[0,0,0,0]`:
   - `prop_l` — light (u) spectator, multigrid CG.
   - `prop_b` — bottom quark (mass = 1.5), standard CG.
2. **Sequential source** at the sink time slice `t_seq = 8`:
   - Convert `prop_l` at t = 8 to backward via γ₅-hermiticity.
   - Multiply by the sink γ₅ structure to form the sink block.
   - Apply the two-dagger convention: η_seq = γ₅·B†·γ₅.
   - Invert with the strange-quark Dirac operator (mass = −0.2356).
3. **Three-point contraction** at each τ (1 ≤ τ ≤ 7):
   - C₃(τ) = Σ_z Tr[G_s_seq(z;τ) · γₓ · S_b(z;0)].

## Technical Specifications

| Item | Value |
|------|-------|
| Lattice | 24³ × 72, a ≈ 0.1052 fm, isotropic |
| Gauge config | C24P29, cfg 10000, Chroma QIO |
| Stout smearing | 1 step, ρ = 0.125, 4‑dim |
| Light solver | Multigrid CG, tol = 10⁻¹², max 20000 iter |
| Strange solver | CG, tol = 10⁻¹⁰, max 10000 iter |
| Bottom solver | CG, tol = 10⁻¹⁰, max 5000 iter |
| MPI | 4 ranks, grid [1,1,1,4] |
| Source | Point at origin, zero momentum |
| Sink | t_seq = 8, zero momentum, point |

## Known Caveats

- **Spatial vector at p = 0**: the correlator is expected to be zero; a physics result requires p ≠ 0 or γ₄.
- **Short tseq**: 8 lattice units (≈ 0.84 fm) is very short for a B meson (m_B ~ 5 GeV); excited-state contamination is likely severe.
- **generate_einsum**: the documented interface covers baryon 3pt and meson 2pt explicitly; meson 3pt‑with‑spectator may need a manual fallback if the tool does not support it.

## Output

A plain text file containing the real part of C₃(τ) for τ = 1,…,7, one value per line, no headers.