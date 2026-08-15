# Baryon 3pt Sequential Source — Notes

## 1. Three Propagator Segments

A full 3pt contraction involves three coordinates: **y**(source) → **z**(current) → **x**(sink)

wicklib to_einsum() outputs 10 operands, 4 of which are propagators:

| # | wicklib name       | Physics                | Symbol        |
|:-:|:-------------------|:-----------------------|:--------------|
| 7 | `propag_X_z_y`     | source y → current z   | **P_forward** |
| 8 | `propag_X_x_z`     | current z → sink x     | **P_after**   |
| 9 | `propag_X_x_y`     | source y → sink x      | **Spectator 1** |
|10 | `propag_X_x_y`     | source y → sink x      | **Spectator 2** |

## 2. Roles of Each Segment

- **P_forward** (operand 7): kept for the final contraction
- **Spectator 1 & 2** (operands 9,10): absorbed into the **sink block**
- **P_after** (operand 8): **not computed by wicklib directly** — produced by the sequential solve

## 3. Sequential Source Pipeline

```
sink block B(x) = ε·ε·Cγ₅·P₊ · spectator1 · spectator2   ← 2 spin + 2 color free indices left open
      ↓
B̃ = γ₅ · B† · γ₅                                         ← G5-dagger
      ↓
M·ψ = B̃   (Dirac inversion)                               ← propagate from sink x to all points
      ↓
ψ̃ = γ₅ · ψ† · γ₅                                         ← second G5-dagger
      ↓
C3 = Tr[ ψ̃ · Γ_cur · P_forward ]                          ← final contraction, all indices summed
```

## 4. Spin and Color Index Constraints

- Each **LatticePropagator** has 2 spin + 2 color indices (6 total, `wtzyx` is the batch spacetime dimension)
- The **sink block** contracts the spectators' 4+4 spin×color indices → leaves 2+2 free
- Those 2+2 free indices are the spin×color structure for the **P_after (z→x)** segment
- Sequential solve must preserve the correct spin/color index order

## 5. γ₅-Hermiticity Direction Reversal

Propagator direction cannot be swapped freely. Key relations:

```
S(z, y) = γ₅ · S(y, z)† · γ₅       ← γ₅-hermiticity for forward propagator
S(x, z) = γ₅ · S(z, x)† · γ₅       ← direction reversal formula
```

The sequential solve produces ψ(z) = ∫ S(z,x)·B̃(x) dx (direction x→z).
After G5-dagger (`ψ̃ = γ₅·ψ†·γ₅`), the direction flips to z→x, matching the free indices left by the sink block.

## 6. Structure of the Sequential Source

The full sequential source consists of six components:

```
sink block = Cg5(src) · ε(src) · Cg5(snk) · ε(snk) · P₊ · Spec1 · Spec2
             ↑src diquark ↑color ↑snk diquark ↑color ↑proj. ↑spec1  ↑spec2
```

- Diquark structure: `Cg5` (octet ½⁺) or `Cg1` (decuplet ³⁺²)
- Color structure: two ε tensors (source and sink)
- Projection operator: `P₊ = (I₄ + γ₄)/2`
- Two spectator propagators: y→x

**Important**: which spin/color index connects to which propagator leg depends on the specific Wick topology and **cannot be deduced from general physics rules** — it must be read from the wicklib to_einsum() einsum string.

### Symmetries

- **Cg5** = C·γ₅: two spin indices **anti-symmetric** (octet ½⁺ baryons)
- **Cg1** = C·γ₁: two spin indices **symmetric** (decuplet ³⁺² baryons)
- **ε**: three color indices **fully anti-symmetric**

These symmetries determine which topologies can be simplified when identical-flavor quarks are present (e.g., p→p with two identical u quarks and Cg5 anti-symmetry).

## 7. Two Implementation Strategies

### Strategy A: Per-term construction → Sum B → One solve

For each term, construct the sink block contribution from the known physical structure (Cg5, ε, P₊, spec1, spec2) plus P_forward, leaving P_after's 4 indices as output:

```
for each term:
    B_term = contract(structure + spec1 + spec2 + P_forward,  output=P_after_label)
    B += sign * B_term

B̃ = γ₅ · B† · γ₅
M·ψ = B̃
ψ̃ = γ₅ · ψ† · γ₅
C3 = Tr[ ψ̃ · Γ_cur · P_forward ]
```

- **Pro**: indices correctly assigned by wicklib to_einsum()
- **Con**: each term has different P_forward and Γ_cur labels, making the code verbose

### Strategy B: Unified index renaming → Sum B → One solve

Map each term's P_forward indices to the same standard labels. When P_forward in one term uses indices `ECdc`, rename `E→α, C→β, d→γ, c→δ`. This renaming propagates through the entire einsum (e.g., Gamma_cur's `DE` becomes `Dα`), keeping relative connections intact but standardizing across terms.

All terms then use identical index labels for the sink block:

- **Pro**: B summation, G5-dagger, sequential solve, final contraction all use the same template
- **Con**: requires a deterministic renaming rule

### Both strategies are equivalent for B summation

B is a LatticePropagator (6D array). Regardless of the label convention used in each term's `contract()`, the result correctly populates the appropriate (spin,color) slices of B. Different output labels (e.g., `GDfd` vs `GBfb`) write to different (spin,color) positions and do not conflict.

### Key clarification: Summation happens BEFORE the solve

```
Each term → B_term → accumulate B = Σ sign_i × B_term_i
                          ↓
                   one B̃ = γ₅·B†·γ₅
                          ↓
                   one M·ψ = B̃
                          ↓
                   one ψ̃ = γ₅·ψ†·γ₅
                          ↓
                   one final contraction
```

All terms share a single sequential solve (sum-then-solve, not solve-then-sum). This is the efficient approach.
