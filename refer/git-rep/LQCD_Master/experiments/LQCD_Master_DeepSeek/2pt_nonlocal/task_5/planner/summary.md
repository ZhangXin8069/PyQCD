## Physics Objective

Compute the nonlocal two-point correlation function of the rho meson (ρ⁺, isovector vector meson, quark content d̄u) on ensemble C24P29 (24³×72, β=6.20, a≈0.105 fm). The nonlocal operator at the sink displaces the u-quark field by z ∈ [0,10] lattice units in the +z direction and connects it back to the local site x via a straight Wilson line constructed from the original (unsmeared) gauge links. The d̄-antiquark field remains at the local position. All three spatial polarizations (γ₁, γ₂, γ₃) are averaged. This observable is relevant for quasi-distribution and TMD studies of the rho meson.

## Key Revisions from Critique

1. **Nonlocal shift moved to correlator section**: The `nonlocal_shift` block has been removed from the propagator definition (`prop_l`). It now resides entirely under `measurement.correlators[rho_nonlocal_corr].nonlocal`, making clear that the shift is a contraction-time modification, not a property of the Dirac inversion. The propagator `prop_l` is a clean, standard point-source solve on stout-smeared links.

2. **Two propagator usages distinguished**: The correlator now defines two distinct roles for the same `prop_l` propagator — `antiquark_dbar` (unshifted, used directly) and `quark_u` (shifted, with Wilson line multiplication on the color index). A naive code generator will no longer ambiguously apply the shift to both lines or neither.

3. **Explicit Wilson line convention**: The Wilson line is defined as the forward product `W(x, x+ẑ) = U_z(x) U_z(x+ẑ) … U_z(x+(z−1)ẑ)` using original unsmeared gauge links, with gauge transformation property `Ω(x) W Ω†(x+ẑ)`. At z=0, W is the identity and the standard local rho correlator is recovered as a consistency check.

4. **Index-level contraction formula**: The contraction is specified with explicit color/spin indices: `C_i(z,t) = Σ_x Re Tr_{c,s}[ S_l†(x,t) (γ₅γ_i) W(x,x+ẑ) S_l(x+ẑ,t) (γ_iγ₅) ]`, clarifying that the Wilson line transports the color index from site x+ẑ back to x before the trace.

5. **Periodic boundary handling**: All z-direction index arithmetic uses modulo L_s = 24, ensuring Wilson lines that cross the periodic spatial boundary are correctly constructed using boundary-crossing U_z links.

6. **MPI gather specified**: With process grid [1,1,1,4] partitioning time across 4 ranks, the plan now explicitly requires an MPI reduction (sum) of the t-sliced correlator to rank 0 before writing the output file.

## Core Strategy

1. **Gauge configuration**: Load config 10000 from C24P29.
2. **Propagator inversion**: Compute one light-quark point-source propagator from [0,0,0,0] using stout-smeared gauge links (1 step, ρ=0.125, 4D) with a two-level multigrid solver (mass=-0.277, cSW=1.160920226, tolerance 1e-12).
3. **Nonlocal contraction**: For each z = 0,…,10 and each t-slice, construct the Wilson line from original gauge links, multiply onto the color index of the u-quark propagator at x+ẑ, and compute the color-spin trace with the d̄-quark propagator at x for all three gamma polarizations.
4. **Output**: Plain text file `rho_nonlocal_2pt.txt` with columns (z, t, Re[C(z,t)]), no header.

## Technical Details

- **Flavor symmetry**: Isospin symmetry gives S_u = S_d = S_l; no disconnected diagrams for the isovector ρ⁺. Only one propagator inversion is needed.
- **Solver**: Two-level multigrid, block sizes [[6,6,6,3], [4,4,4,6]], clover-improved Wilson Dirac operator.
- **Gauge fields**: Stout-smeared links for the Dirac operator; original unsmeared links for the Wilson line — this separation is physically motivated to keep the UV regularization of the propagator distinct from the Wilson line definition.
- **Zero momentum**: Source and sink momenta are [0,0,0]; the correlator is purely real after spatial summation and the real part is extracted for output.