## Physics Objective

Compute the three-point correlation function for the **B⁻ → D⁰** semileptonic transition mediated by the **b → c vector current** (c̄ γₓ b) on the C24P29 lattice ensemble (24³×72, a ≈ 0.105 fm, a⁻¹ ≈ 1.88 GeV).

**Critical physics note:** The spatial vector current γₓ at zero recoil (q = 0, pᵢ = p_f = 0) vanishes identically for a 0⁻ → 0⁻ pseudoscalar-to-pseudoscalar transition due to parity and angular momentum selection rules. The three-point function is expected to be consistent with zero modulo statistical noise. To obtain a non-vanishing signal, the user should consider switching to the temporal vector current (γ₄) or injecting non-zero momentum. The plan preserves the user's specification of γₓ but documents this limitation explicitly.

## Physical Setup

| Element | Specification |
|---------|--------------|
| Source hadron | B⁻ meson (anti-u b), interpolator ūγ₅b, creation operator O† = −b̄γ₅u at t = 0 |
| Sink hadron | D⁰ meson (anti-u c), interpolator ūγ₅c, annihilation operator at tₛₑq = 8 |
| Transition current | c̄ γₓ b (spatial vector, x-direction) |
| Momentum | Zero at source and sink (pᵢ = p_f = 0), zero momentum transfer (q = 0) |
| Source | Point source at [0,0,0,0], no Gaussian/Wuppertal smearing |
| Gauge links | Stout-smeared (n_steps = 1, ρ = 0.125, 4-dim) for all Dirac operator inversions |
| Sink time | tₛₑq = 8 |

## Key Revisions from Critique

### 1. Sequential Source Spatial Structure (Critical Fix)

The original plan specified `source_position: [0,0,0,0]` for the sequential propagator `prop_c_seq`, which could be misinterpreted as a **point** sequential source at a single spatial location. This is incorrect for zero-momentum sink projection.

**Fix:** The sequential source is now explicitly specified as a **spatial wall** at t = tₛₑq = 8 — constructed as a sum over **all** spatial points, not a single point. The `source_position` is set to `[all, all, all, 8]` with accompanying notes explaining the wall construction. The sequential source at each spatial point z at t = 8 is:

ηₛₑq(z) = γ₅ · [γ₅ · Sₗ(z, 8; 0, 0)]† · γ₅

where Sₗ is the spectator light-quark propagator. This spatial sum implements zero-momentum sink projection. A point-like sequential source would produce an unprojected correlator dominated by noise.

### 2. Meson 3pt Contraction Template Warning

The verified reference materials (pion, rho, proton, Lambda → proton) demonstrate meson_2pt, baryon_2pt, and baryon_3pt contractions, but **no verified meson_3pt sequential-source contraction template** exists. The meson 3pt sink block is structurally simpler than the baryon case (one spectator propagator × γ₅ rather than two forward propagators forming a diquark), but any mismatch in the generate_einsum toolchain would produce incorrect results.

**Mitigation:** The plan provides the explicit Wick contraction derivation and final contraction formula as a fallback:

C₃(τ) = Σ_z Tr_spin,color[ G_c_seq(z, τ) · γₓ · S_b(z, τ; 0, 0) ]

with the sequential source built via the two-dagger convention (η = γ₅ B† γ₅, where B(x) = γ₅ · Sₗ(x, 8; 0, 0)).

### 3. γₓ Selection Rule at Zero Recoil

The spatial vector current at zero momentum transfer vanishes for a 0⁻ → 0⁻ transition. This is documented as a physics warning in the task description, measurement notes, and extras. The user's specification of γₓ is preserved, but alternatives (γ₄ or non-zero momentum) are suggested.

## Computational Strategy: Sequential Source Method

### Propagators Required
1. **prop_l_fwd** — forward light-quark (up) propagator from point source [0,0,0,0]. Spectator quark connecting source to sink (the anti-up present in both B⁻ and D⁰). Solved with multigrid-accelerated CG (m = −0.277, c_sw = 1.160920226).
2. **prop_b_fwd** — forward bottom-quark propagator from point source. Active b quark propagating from source (t = 0) to current insertion (t = τ). Solved with standard CG (m = 1.5).
3. **prop_c_seq** — sequential charm-quark propagator. Built from a spatial-wall sequential source at t = 8 using the D⁰ sink operator and the light spectator propagator (two-dagger convention). Solved with standard CG (m = 0.4159).

### Contraction at Each Current Time
For each τ = 1, …, 7, compute the spatial sum over the current insertion point z:

C₃(τ) = Σ_z Tr[ G_c_seq(z, τ) · γₓ · S_b(z, τ; 0, 0) ]

The trace runs over spin and color indices. The real part is saved as the physical correlator value.

### Wick Contraction Structure
- The spectator ū line contracts ū(sink) with u(source) via Sₗ(x, 8; 0, 0).
- The b-quark line runs from source to current: S_b(z, τ; 0, 0).
- The charm line is reversed via sequential source: G_c_seq effectively propagates the c quark backward from the sink to the current insertion, where it meets the forward b-propagator through the γₓ vertex.

## Output

Single `.txt` file in the run directory containing 7 space-separated floating-point numbers: C₃(τ = 1) through C₃(τ = 7). No header, no metadata.

## Ensemble Parameters (Unchanged)

All ensemble parameters preserved exactly as provided: lattice 24³×72, a = 0.1052 fm, anti-periodic temporal BC, quark masses (light = −0.277, charm = 0.4159, bottom = 1.5), c_sw = 1.160920226, two-level multigrid [6,6,6,3]/[4,4,4,6], MPI 1×1×1×4.