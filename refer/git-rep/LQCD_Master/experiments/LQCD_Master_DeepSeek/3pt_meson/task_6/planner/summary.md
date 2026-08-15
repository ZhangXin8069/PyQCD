## Physics Objective

Compute the three-point correlation function for the **D⁰ → K*⁻** transition mediated by the flavour-changing vector current **\bar{s} γₓ c**.  This is a building block for extracting the D → K* semileptonic form factors.

## Key Corrections from the Original Plan

### 1. Source operator fixed
- **Original**: stated the source operator as `\bar{u} γ₅ c` (annihilation operator) but used a contraction formula that implicitly required `\bar{c} γ₅ u`.
- **Corrected**: the operator that appears in the three-point correlator at the source is the **creation operator** `\bar{c} γ₅ u`, which is the hermitian conjugate of the standard pseudoscalar annihilation operator `\bar{u} γ₅ c`.  This is now stated explicitly, and all downstream expressions — sink block, sequential source, and final contraction — are consistent with this choice.

### 2. Sequential-source construction separated from final contraction
- The source γ₅ is **removed from the sequential-source construction** and placed only in the final contraction.  The sequential source `η_seq` is built solely from the sink block `B(x) = S_u(0; x, tseq) γ_x` (constructed from the forward light propagator and the sink gamma structure via γ₅-hermiticity).

### 3. Two-dagger convention made explicit
- **First dagger**: `η_seq = γ₅ B† γ₅` — forms the sequential source from the sink block.
- **Second dagger**: `G_seq_dag = γ₅ G_seq† γ₅` — prepares the sequential propagator for contraction.
- The final contraction uses `G_seq_dag`:  `C₃(τ) = Σ_y Tr[ G_seq_dag(y,τ) γ_x S_c(y,τ) γ₅ ]`.

### 4. Sequential source is NOT a uniform wall
- The phrase "spans the full spatial volume" has been replaced with a precise description: the sequential source is non-zero only on time slice `tseq = 8` and its spatial structure is determined point-by-point from the forward light propagator `S_l(x; 0)`.  It is emphatically **not** a spatially constant wall source.

## Strategy Summary

| Component | Specification |
|-----------|--------------|
| Source hadron | D⁰ (c \bar{u}, Jᴾ = 0⁻) — creation operator `\bar{c} γ₅ u` |
| Sink hadron | K*⁻ (s \bar{u}, Jᴾ = 1⁻) — annihilation operator `\bar{u} γₓ s` |
| Current | `\bar{s} γₓ c` (vector, spatial x) |
| Forward props | Light (am = −0.277) and charm (am = 0.4159), point source at origin |
| Sequential prop | Strange (am = −0.2356), tseq = 8, two-dagger convention |
| Final contraction | `Tr[ G_seq_dag · γₓ · S_c · γ₅ ]` per time slice τ |
| Gauge smearing | Stout, 1 step, ρ = 0.125, 4-dim |
| Output | Plain text, no header, run directory |

## Caveats Documented
- **Charm discretisation**: `am_c ≈ 0.42` at `a ≈ 0.105 fm` → large O((am_c)²) errors with Wilson-clover fermions.
- **Source-sink separation**: `tseq = 8` (~0.84 fm) may be insufficient to suppress excited-state contamination for the heavy D meson.
- **Single polarisation**: only γₓ is used; averaging over γₓ, γ_y, γ_z would require two additional sequential inversions but would improve statistics.