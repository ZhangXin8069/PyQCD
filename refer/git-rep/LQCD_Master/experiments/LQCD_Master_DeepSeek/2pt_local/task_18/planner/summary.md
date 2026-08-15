## Physics Objective

Compute the zero-momentum two-point correlation function of the **Σ_c^+ (udc)** charmed baryon with J^P = 1/2^+, using a single gauge configuration (cfg 10000) on the C24P29 ensemble (24³×72, a ≈ 0.1052 fm). The correlator serves as the primary input for extracting the Σ_c^+ ground-state mass via an effective-mass plateau analysis.

## Key Revisions from Peer Review

### 1. Sign Convention for the Cγ₁ Diquark (corrected)

The previous plan implicitly copied the proton's Cγ₅ derivation, which yields a positive overall sign.  For Cγ₁ the derivation differs:

- In the DeGrand-Rossi basis, **γ₁^T = −γ₁** (antisymmetric), unlike γ₅^T = +γ₅ (symmetric).
- The source diquark structure after γ₅-hermiticity is (γ₄γ₁γ₄C).  Since γ₄γ₁γ₄ = −γ₁, this becomes (−γ₁C).
- Transposing: (−γ₁C)^T = −C^T γ₁^T = −(−C)(−γ₁) = **−Cγ₁**.

Compared to the Cγ₅ case where (γ₄γ₅γ₄C)^T = +Cγ₅, the Cγ₁ case picks up an **extra minus sign**.  The corrected contraction formula is:

```
C(t) = − P⁺_{γ'γ} Σ_x ε^{abc} ε^{a'b'c'} (Cγ₁)_{αβ} (Cγ₁)_{α'β'}
       × S_l^{aa'}_{αα'}(x) S_l^{bb'}_{ββ'}(x) S_c^{cc'}_{γγ'}(x)
```

where P⁺ = (1+γ₄)/2 is the positive-parity projector.  The `generate_einsum` tool is expected to produce the correct sign, but the physics description now reflects the true sign structure.

### 2. Spin-State Content (corrected)

The Cγ₁ vector diquark operator ε^{abc}(u^T Cγ₁ d) c couples to **both** J^P = 1/2^+ (Σ_c^+) and J^P = 3/2^+ (Σ_c^*+).  The parity projector (1+γ₄)/2 isolates positive parity but does **not** separate spin-1/2 from spin-3/2.  The lighter Σ_c^+(1/2^+) is expected to dominate at large Euclidean time, but the plan now explicitly acknowledges the two-state contamination rather than incorrectly claiming a pure 1/2^+ channel.

### 3. Mixed-Flavor Propagator Assignment (execution risk flagged)

The baryon has two light quarks (u, d) and one charm quark (c).  The `generate_einsum(type="baryon_2pt")` call must receive explicit per-quark-line propagator mapping — `quark_lines = [prop_l, prop_l, prop_c]` — to ensure the charm line receives `prop_c` (mass 0.4159) and not `prop_l` (mass −0.277).  The reference proton example uses three degenerate propagators and does not exercise this code path.

## Strategy (unchanged core)

### Interpolating Operator

```
O_{Σ_c^+} = ε^{abc} (u^{Ta} C γ₁ d^b) c^c
```

The Σ_c^+ (I=1) requires a flavor-symmetric ud diquark.  By the Pauli principle — antisymmetric color (ε^{abc}) and symmetric flavor — the diquark spin wavefunction must be symmetric (spin-1), hence the vector diquark structure Cγ_i.  Per specification, only Cγ₁ is used with no spatial averaging.

### Wick Contraction

With three distinct quark flavors (u, d, c), there is only **one connected diagram** (no exchange term).  The contraction at rest (p=0) with point source at the origin yields the formula above with the overall minus sign.

### Propagator Requirements

| Propagator | Flavor | Source | Gauge Links | Solver |
|------------|--------|--------|-------------|--------|
| prop_l | light (u,d) | point [0,0,0,0] | stout(1, 0.125, 4) | CG + multigrid |
| prop_c | charm (c) | point [0,0,0,0] | stout(1, 0.125, 4) | standard CG |

The light propagator is reused for both u and d quark lines.  Charm uses standard CG (heavy quark, multigrid adds overhead without benefit).

## Technical Details

- **Ensemble**: C24P29, 24³×72, a ≈ 0.1052 fm, clover c_sw = 1.160920226
- **Configuration**: cfg 10000 only
- **Stout smearing**: 1 step, ρ = 0.125, 4-dimensional
- **Solver tolerance**: 1.0×10⁻¹², max 20000 iterations
- **Multigrid**: two-level blocking [[6,6,6,3], [4,4,4,6]] for light-quark solver
- **MPI**: 4 ranks in temporal direction [1,1,1,4]
- **Momentum**: zero (rest frame)
- **Output**: raw correlator values as plain text (no headers) to run directory