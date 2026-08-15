## Physics Objective

Compute the two-point correlation function of the **Omega_c⁰ baryon** (quark content **ssc**, J^P = 1/2⁺) to extract its ground-state mass. The correlator is projected onto positive parity using P⁺ = (1+γ₄)/2.

## Critical Correction: Diquark Gamma Structure

The original request specified a Cγ₁ (= C ⊗ γ_x) diquark for the two identical strange quarks. This is **physically invalid**: in the DeGrand-Rossi basis, (Cγ₁)ᵀ = +Cγ₁ (symmetric in spinor space), while the color tensor ε^{abc} is antisymmetric. The contraction Σ_{a,b} ε^{abc} s^{aT} (Cγ₁) s^b vanishes identically — the operator would produce pure noise. The only standard gamma combination yielding a non-zero diquark for identical quarks is **Cγ₅**, which satisfies (Cγ₅)ᵀ = −Cγ₅ (antisymmetric). The plan has been corrected to use Cg5.

## Spin Assignment

The Omega_c⁰ ground state has J^P = 1/2⁺ (PDG). The spin-3/2 label in the original plan was inconsistent with the chosen interpolating operator, which carries no Lorentz vector index. This has been corrected throughout.

## Strategy

- **Interpolating operator**: O_{Ω_c} = ε^{abc}(s^{aT} Cγ₅ s^b) c^c — the two strange quarks form a Cg5 diquark, the charm quark is the spectator.
- **Correlator**: C(t) = Tr[P⁺ ⟨O(0⃗,t) Ō(0⃗,0)⟩] at zero momentum.
- **Wick contractions**: Two topologies (direct and exchange) from the identical strange quarks, structurally identical to the proton case but with s,s,c flavors and Cg5 diquark. Handled by `generate_einsum(type="baryon_2pt", ...)` with `BaryonOp('omega_c0', {'a':'s','b':'s','c':'c'}, 'Cg5')`.

## Technical Details

| Item | Specification |
|---|---|
| **Source** | Point source at [0,0,0,0] |
| **Gauge smearing** | Stout: n_steps=1, ρ=0.125, ndim=4 (applied once, reused) |
| **Strange propagator** | Multigrid solver, mass=−0.2356, csw=1.160920226, tol=1e−12, maxiter=10000 |
| **Charm propagator** | **CG solver** (not multigrid), mass=0.4159, csw=1.160920226, tol=1e−12, maxiter=20000 with convergence check |
| **Diquark** | Cg5 = Cγ₅ (required by Pauli principle for identical quarks) |
| **Parity projector** | T = (I + γ₄)/2 |
| **Momentum** | p⃗ = 0⃗ |
| **Output** | Plain .txt: C(t) for t=0…71, one value per line, no header |

## Ensemble

C24P29: 24³×72 lattice, a ≈ 0.1052 fm, periodic spatial / anti-periodic temporal BCs, 4-way MPI in t-direction.

## Solver Choices

- **Strange**: Multigrid is the production-standard solver for near-physical strange quarks and efficiently handles critical slowing-down.
- **Charm**: Switched to CG. The heavy charm mass produces a well-conditioned Dirac operator where CG converges reliably; multigrid may stall or converge slowly at such masses. Maxiter is raised to 20000 with an explicit convergence-failure check.