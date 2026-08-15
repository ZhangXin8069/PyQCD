## Physics Objective

Compute the zero-momentum two-point correlation function of the **Xi_c0 baryon** (flavor content: d, s, c) using a point source on configuration 10000 of the C24P29 ensemble (24³×72, β=6.20, a≈0.105 fm). The correlator enables ground-state mass extraction for the spin-1/2⁺ Ξ_c⁰.

## Interpolating Operator and Contraction

The Xi_c0 is interpolated with a **Cγ₅ (Cg5) diquark** operator:

O_Ξ = ε^{abc} (d^{Ta} Cγ₅ s^b) c^c

Since all three quark flavors (d, s, c) are distinct, there is **only one Wick contraction term** — no exchange diagram. The correlator is projected onto the positive-parity channel with P⁺ = (1 + γ₄)/2.

**Explicit propagator mapping:** prop_l → d (position a), prop_s → s (position b), prop_c → c (position c). This ordering is critical for correct contraction and must be supplied to the generate_einsum tool.

## Key Revision: Charm Quark Solver (CG instead of Multigrid)

The most critical revision addresses a **solver failure risk** identified in peer review. The 2-level multigrid parameters [6,6,6,3]→[4,4,4,6] are tuned for near-critical light quarks and rely on low-mode dominance for effective coarse-grid correction. The charm quark (mass 0.4159) lacks this low-mode dominance, so multigrid coarse-grid correction becomes ineffective and the solver would likely stagnate or fail to converge within 2000 iterations — breaking the entire measurement.

**Fix:** The charm propagator (`prop_c`) now uses a **standard CG solver** with tolerance 1×10⁻¹⁰ and maxiter 2000. CG converges reliably for heavy quarks on lattices of this size. The light and strange propagators continue to use the multigrid solver (tolerance 1×10⁻¹²), where MG remains effective.

## Propagator Summary

| Propagator | Flavor | Mass   | Solver | Tolerance | Maxiter |
|-----------|--------|--------|--------|-----------|--------|
| prop_l    | d      | −0.277  | MG     | 1×10⁻¹²   | 2000   |
| prop_s    | s      | −0.2356 | MG     | 1×10⁻¹²   | 2000   |
| prop_c    | c      | 0.4159  | CG     | 1×10⁻¹⁰   | 2000   |

All three use a point source at [0,0,0,0], stout-smeared gauge links (n_steps=1, rho=0.125, ndim=4), and clover coefficient c_sw = 1.160920226.

## Post-Contraction Sanity Checks

Before writing the output file, the code must verify:
1. **C(t=0) > 0** — a negative value indicates a sign error in the contraction or a failed inversion.
2. **|C(t)| decreases monotonically** for t < T/2 (modulo statistical fluctuations at late time slices) — non-monotonic behavior signals a propagator misassignment or solver failure.

If either check fails, the code aborts with an error message rather than silently writing corrupted data.

## Pre-Flight Tool Verification

The `generate_einsum(type='baryon_2pt')` tool must be verified to accept a distinct-flavor baryon with diquark='Cg5' and the flavor map {a:d, b:s, c:c} before the production run. If the tool does not natively support this combination, the einsum string and sign must be supplied manually from the explicit Wick contraction.

## Output

One plain-text file per configuration, saved in the run directory, containing one correlator value per line (72 time slices), with no header or metadata.