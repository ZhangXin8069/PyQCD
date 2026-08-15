## Physics Objective

Compute the three-point correlation function for the flavour-changing neutral-current transition **B⁻ → K*⁻** via the **b → s vector current** on a single gauge configuration (C24P29, cfg 10000, a ≈ 0.105 fm). The correlator encodes the hadronic matrix element ⟨K*⁻| s̄ γₓ b |B⁻⟩ and is the lattice input for extracting the B → K* form factor.

## Corrections from Peer Review

### 1. Sequential source specification (critical fix)
The original plan incorrectly labelled `prop_s_seq` as a point source at `[0,0,0,8]`. A sequential source is **not** a point source — it is constructed from the forward light propagator `prop_l_fwd` on the **full spatial volume** at the sink time slice. The corrected plan:
- Sets `source_type: sequential`.
- Sets `source_position: [0,0,0,0]` reflecting the time-translated location after the two-dagger construction.
- Documents the full construction chain: sink block B ← prop_l_fwd at t=8 ⊗ K* sink operator γₓ, summed over all spatial x → sequential source η = γ₅ B† γ₅ at t=0 → solve D_s G_seq = η.
- This provides **volume averaging** over the sink spatial position, not a single-point sink.

### 2. Operator convention clarified
`interpolating_operator` specifies the **annihilation** operator (e.g. O_B⁻ = b̄ γ₅ u). The executor automatically places the creation operator O† at the source time slice per the standard PyQUDA 3pt convention. This ensures the quark-field content at the source is correct (b̄ and u, not ū and b).

### 3. Contraction sign — delegated to generate_einsum
The manually-derived contraction formula with an explicit minus sign has been removed. The sign is determined at runtime by the `generate_einsum` tool (meson_3pt: pseudoscalar source γ₅, vector sink γₓ, vector current γₓ, three distinct flavours). No manual sign is specified in the plan.

### 4. Bottom quark mass caveat
`am_b = 1.5` on a clover action with a ≈ 0.105 fm yields O((am)²) ≈ 2.25 discretisation errors, far outside the reliable regime. The plan increases `maxiter` to 50000 and recommends a residual-norm check. For production use, a Fermilab heavy-quark action or NRQCD is advised. A warning is included in `extras`.

## Strategy Summary
- **Source** (t = 0): B⁻ creation operator O_B⁻† = −ū γ₅ b from point source at [0,0,0,0].
- **Sink** (t = 8): K*⁻ annihilation operator O_K*⁻ = s̄ γₓ u.
- **Current** (τ): s̄ γₓ b, x-component of the vector current.
- **Propagators**:
  1. `prop_l_fwd` — light spectator (anti-u), point source → all (x,t).
  2. `prop_b_fwd` — bottom quark, point source → all (z,τ).
  3. `prop_s_seq` — strange **sequential** propagator built from prop_l_fwd at t=8 via two-dagger convention, solved as one extra inversion.
- **Contraction**: C₃(τ) = Σ_z Tr[ G_seq(z,τ) γₓ S_b(z,τ) ] for τ = 1,…,7.
- All propagators use 1-step stout-smeared links (ρ = 0.125, 4-dim).
- Only the 3pt is computed; no 2pt functions. Output: plain text, one C₃ value per line.

## Remaining Risks
- **Bottom quark mass**: am_b = 1.5 is at the edge of clover-fermion usability. The solver may fail to converge even at 50000 iterations.
- **Single polarisation**: Only γₓ (γ₁) is used. The standard practice averages over γ₁, γ₂, γ₃ for better statistics and rotational-symmetry verification.