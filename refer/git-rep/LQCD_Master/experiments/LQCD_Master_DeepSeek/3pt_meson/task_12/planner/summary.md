## Physics Objective

Compute the three-point correlation function for the flavour-changing neutral-current transition **D⁰ → η_u** mediated by the vector current **J_x = ū γ_x c**. The D⁰ meson (anti‑u, c) is created at the source (t=0) with a **γ₅** interpolator, and the η_u meson (anti‑u, u) is annihilated at the sink (t=tseq=8) with a **γ₅** interpolator. All momenta are zero.

## Strategy — Sequential-Source Method

### Propagator chain

1. **Forward light propagator** `prop_l` — point source at [0,0,0,0], mass −0.277. Serves as the spectator u‑quark line connecting the source D⁰ to the sink η_u.

2. **Forward charm propagator** `prop_c` — same point source, mass 0.4159. The c‑quark propagates from the source to the current insertion point.

3. **Sequential light propagator** `prop_l_seq` — constructed from a **full‑volume** sequential source at t=8: for every spatial site **x⃗** at the sink time slice, the source value is built from the forward light propagator S_l(**x⃗**,8; 0⃗,0), the sink γ₅, and the source γ₅ via the standard **two‑dagger convention** (η_seq = γ₅ B† γ₅). Solving D_l G_seq = η_seq yields the sequential propagator. This full‑volume construction ensures the correct zero‑momentum sink projection.

### Contraction

Both the sink γ₅ (from η_u) and the source γ₅ (from D⁰) are embedded in the sequential-source block. The 3pt correlator at each current insertion time τ ∈ [0, 8] is therefore:

**C₃(τ) = Σ_z Tr[ G_seq(z,τ) · γ_x · S_c(z,τ; 0,0) ]**

No additional gamma matrices appear in the final trace — they are already absorbed into G_seq.

### Diagram topology

Only the **connected** diagram is computed. The η_u meson is flavour‑diagonal (ūu) and would admit disconnected quark‑loop contributions that are physically relevant (they drive the η–η′ mass splitting). The result therefore represents the **connected part** of the D⁰ → η_u transition, not the full matrix element.

## Technical Details

| Item | Value |
|------|-------|
| Gauge ensemble | C24P29, 24³×72, β=6.20, a≈0.1052 fm |
| Quark masses | light=−0.277, charm=0.4159 |
| Clover coefficient | 1.160920226 |
| Gauge smearing | Stout, n_steps=1, ρ=0.125, ndim=4 (all inversions) |
| Light‑quark solver | Multigrid ([6,6,6,3]→[4,4,4,6]), tol=1e−12, maxiter=2000 |
| Charm‑quark solver | Same multigrid parameters; fallback to CG or tol=1e−10 if MG fails to converge |
| Source | Point at [0,0,0,0], p=0 |
| Sequential source | Full spatial volume at t=8, p=0 |
| Sink | η_u, γ₅ interpolator, zero momentum |
| Current | ū γ_x c (vector, x‑direction) |
| MPI | 4 ranks, process grid [1,1,1,4] |
| Output | Raw complex numbers, one per τ, written to `d0_to_eta_u_3pt_cfg10000.txt` |

## Revisions from Critique

1. **Sequential source spatial extent** — Corrected from an ambiguous point‑like specification to an explicit full‑volume construction at t=tseq. The `source_position` spatial coordinates are now documented as placeholders; the real construction sums over all spatial sites using the forward propagator.

2. **Source γ₅ embedding** — The `sequential_source` block now references both `sink_hadron: eta_u` and `source_hadron: D0`, ensuring that both γ₅ structures are embedded in the sequential‑source block via the two‑dagger convention. The final contraction formula is validated as C₃(τ) = Σ_z Tr[G_seq · γ_x · S_c] with no orphan gamma matrices.

3. **Disconnected diagram documentation** — Added a clear statement that only the connected diagram is computed and that disconnected contributions, while physically present for the flavour‑diagonal η_u, are omitted. This prevents misinterpretation of the result as the full matrix element.

4. **Charm multigrid convergence** — Added a solver note warning that the same multigrid parameters may degrade for the heavy charm quark and specifying fallback strategies (CG solver or relaxed tolerance to 1e−10).

## Requirement Satisfaction

- ✅ D⁰ source with γ₅, η_u sink with γ₅
- ✅ Current ū γ_x c
- ✅ Point source at [0,0,0,0], zero momentum, tseq=8
- ✅ Full‑volume sequential source for correct zero‑momentum sink projection
- ✅ Source and sink γ₅ both embedded in sequential‑source construction
- ✅ Stout‑smeared links (1, 0.125, 4) for all inversions
- ✅ No 2pt computation
- ✅ Plain text output, no headers