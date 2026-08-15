## Physics Objective

Compute the three-point correlation function for the flavour-changing transition **Ds⁺ → φ** mediated by the **c → s vector current** J_x = s̄ γ_x c. This prototype for semileptonic form-factor extractions involves a heavy-light initial meson (Ds⁺ = c s̄) decaying to a light vector meson (φ = s s̄).

## Operator and Correlator Structure

| Role | Hadron | Quark Content | Operator | Dagger at Source |
|------|--------|---------------|----------|:---:|
| Source (t=0) | Ds⁺ | c s̄ | O_Ds = s̄ γ₅ c → O_Ds† = c̄ γ₅ s | ✓ |
| Sink (t=8) | φ | s s̄ | O_φ = s̄ γ_x s | ✗ |
| Current (τ) | — | s̄ → c | J_x = s̄ γ_x c | — |

The three-point correlator is C₃(t_f=8, τ, 0) = Σ_{x,z} ⟨ O_φ(x,8) · J_x(z,τ) · O_Ds†(0,0) ⟩.

## Wick Contraction and Sequential Source

**Connected diagram only** (disconnected s-quark loop at the φ sink is OZI-suppressed and omitted; see caveat below). Three quark lines:

- **Spectator anti-s**: forward strange propagator S_s(x,8; 0,0) → `prop_s_fwd`
- **Active charm**: forward charm propagator S_c(z,τ; 0,0) → `prop_c_fwd`
- **Converted s quark**: handled via sequential source method → `prop_s_seq`

**Sequential source construction** (two-dagger convention, both source and sink operators enter):
1. Form the sink block B from the φ sink operator (γ_x), the spectator `prop_s_fwd`, **and the daggered Ds source operator (γ₅, daggered)**.
2. Compute η_seq = γ₅ B† γ₅.
3. Solve D_s · G_seq = η_seq to obtain the sequential strange propagator.
4. Contract at each insertion time τ: C₃(τ) = Σ_z Tr[ G_seq(z,τ) · γ_x · S_c(z,τ; 0) ].

The `generate_einsum` call receives both source (γ₅, daggered) and sink (γ_x) gamma structures to produce the correct spin contraction.

## Propagator Inventory (new: sequential propagator added)

| Propagator | Flavor | Source Type | Solver | Mass (κ) |
|-----------|--------|-------------|--------|----------|
| `prop_s_fwd` | strange | point [0,0,0,0] | multigrid CG | −0.2356 |
| `prop_c_fwd` | charm | point [0,0,0,0] | **standard CG** | 0.4159 |
| `prop_s_seq` | strange | sequential at t=8 | multigrid CG | −0.2356 |

The charm solver is explicitly set to standard CG — multigrid preconditioning is not designed for heavy quarks and may fail to converge. The strange forward and sequential solvers use the ensemble's 2-level multigrid parameters.

## Key Revisions from Previous Plan

1. **Sequential propagator entry added** (`prop_s_seq`): previously missing from the propagators list; now explicitly defined with quark flavour, source type, solver parameters, and reference to the operators that define its sequential source.
2. **Charm solver type made explicit**: `solver_type: cg` prevents accidental multigrid use on the heavy charm quark.
3. **Sequential source construction clarified**: the sink block now explicitly includes both the sink operator (γ_x) and the daggered source operator (γ₅). The previous plan only mentioned the sink operator.
4. **Source operator dagger marked**: `use_dagger_at_source: true` on `ds_plus` ensures the executor and `generate_einsum` form O_Ds† rather than O_Ds at the source.
5. **Disconnected diagram caveat added**: the neglected s-quark loop at the φ sink is noted as an approximation, with reference to OZI violation in φ–ω mixing.

## Numerical Details

- **Ensemble**: C24P29, 24³×72, a ≈ 0.1052 fm, clover coefficient 1.160920226
- **Configuration**: single config #10000
- **Gauge smearing**: stout (n_steps=1, ρ=0.125, ndim=4) on all inversions
- **Solvers**: multigrid CG for strange (2-level, blocks [6,6,6,3] / [4,4,4,6]); standard CG for charm; tolerance 1×10⁻¹⁰, max 20000 iterations
- **MPI**: 4 ranks in temporal direction, process grid [1,1,1,4]
- **Output**: plain text file with columns (τ, C₃(τ)) for τ = 0,…,8