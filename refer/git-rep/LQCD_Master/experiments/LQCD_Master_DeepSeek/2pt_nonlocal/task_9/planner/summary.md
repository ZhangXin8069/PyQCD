## Physics Objective

Compute the nonlocal two-point correlation function of the vector D*⁺ meson (d̄ c, Jᴾ = 1⁻) with a Wilson-line shift applied to the charm quark propagator in the z-direction. Separations range from 0 to 10 lattice units. This observable is the lattice QCD precursor to a quasi-distribution amplitude (quasi-DA) for the D* meson.

## Core Strategy

**Operator construction:**
- Sink (nonlocal): O_D*⁺(x,t; ẑΔ) = d̄(x,t) γᵢ W(x,t; x+ẑΔ,t) c(x+ẑΔ,t)
- Source (local): Ō_D*⁺(0,0) = c̄(0,0) γ₄ γᵢ γ₄ d(0,0)
- W is the straight Wilson line along z; γᵢ ∈ {γ₁, γ₂, γ₃} averaged over polarizations.

**Wick contraction (after γ₅-hermiticity):**

C(t, Δ) = (1/3) Σ_{i=1,2,3} Σ_x Tr[ S_l†(x,t;0,0) (γ₅γᵢ) W(x, x+ẑΔ; t) S_c(x+ẑΔ, t; 0,0) (γᵢγ₅) ]

The charm propagator is evaluated at the shifted position and multiplied by the Wilson line to maintain gauge invariance at point x.

**Pre-processing:** For each Δ, construct the shifted charm field S_c,shifted(x,t;Δ) = W(x, x+ẑΔ; t) · S_c(x+ẑΔ, t; 0,0) before the standard meson-2pt trace contraction with the light propagator. This avoids relying on a standard einsum tool to handle nonlocal shifts internally.

**Periodic boundary handling:** On the L_z = 24 lattice, separations up to Δ = 10 may cross the spatial boundary. When x_z + Δ ≥ 24, wrap the z-coordinate to (x_z + Δ) mod 24 and include the boundary-crossing U_z gauge link in the Wilson-line product. The gauge links obey periodic spatial boundary conditions, making this wrapping well-defined.

## Technical Details

| Item | Specification |
|------|--------------|
| **Source** | Point source at [0,0,0,0], momentum [0,0,0] |
| **Gauge for inversions** | Stout-smeared: n_steps=1, ρ=0.125, 4-dim |
| **Gauge for Wilson line** | Original (unsmeared) gauge links, with periodic wrapping |
| **Light quark** | mass = -0.277, multigrid solver, tol=1e-12 |
| **Charm quark** | mass = 0.4159, CG solver, tol=1e-12; fallback to CGNE or increased maxiter if convergence stalls |
| **Separations** | Δ = 0, 1, …, 10 (11 values) |
| **Polarizations** | γ₁, γ₂, γ₃ averaged |
| **Output** | Real part of C(t,Δ) as space-separated floats, one value per line; no header |
| **Time extent** | T = 72, output per time slice t × separation Δ |

## Requirement Satisfaction

- ✅ D*⁺ (d̄ c) vector meson
- ✅ Nonlocal shift applied to charm (quark) propagator, not anti-d
- ✅ Maximum separation 10 in z-direction, with periodic spatial wrapping
- ✅ Point source at [0,0,0,0]
- ✅ Original gauge links for Wilson line; stout-smeared links for inversions
- ✅ γ_x, γ_y, γ_z polarizations averaged
- ✅ Plain text output, real part only, no header