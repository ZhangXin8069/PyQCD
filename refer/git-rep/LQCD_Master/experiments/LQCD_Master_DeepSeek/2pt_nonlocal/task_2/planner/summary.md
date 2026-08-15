## Physics Objective

Compute the **nonlocal two-point correlation function** of a $K^+$ meson ($\bar s u$, pseudoscalar channel) on gauge configuration 10000. A Wilson-line shift of length $z$ is applied to the **light-quark (u) propagator** in the $z$-direction, for $z = 0, 1, \dots, 10$.

## Physical Setup (Corrected)

**Sink operator** (nonlocal, zero momentum):
$$\mathcal O_K(t;z) = \sum_{\vec x} \bar s(\vec x,t)\,\gamma_5\,W(\vec x,\vec x+z\hat z;t)\,u(\vec x+z\hat z,t)$$
where $W(\vec x,\vec x+z\hat z;t) = \prod_{k=0}^{z-1} U_z(\vec x+k\hat z,t)$ is a straight Wilson line built from **original (unsmeared)** gauge links.

**Source operator** (local, at the origin — corrected sign):
$$\mathcal O_K^\dagger(0) = -\bar u(\vec 0,0)\,\gamma_5\,s(\vec 0,0)$$

This minus sign is the proper Hermitian conjugate of $\bar s\gamma_5 u$ (cf. pion_mass.md in the skill reference). The previous plan omitted this sign, which propagated into the final contraction.

**Wick contraction** (correct derivation, $\gamma_5$-hermiticity applied):
$$C(z;t) = +\sum_{\vec x}\operatorname{Tr}\!\big[S_s^\dagger(\vec x,t;0)\;W(\vec x,\vec x+z\hat z;t)\;S_u(\vec x+z\hat z,t;0)\big]$$

The overall sign is **positive**, not negative. This follows from: (i) the minus sign in $\mathcal O_K^\dagger$, (ii) the fermion-anticommutation sign in the Wick contraction, and (iii) $\gamma_5$-hermiticity with cyclic trace simplification.

## Gauge-Field Handling (Corrected)

A critical missing step in the previous plan: **original gauge links must be explicitly copied and retained** before applying stout smearing.

1. **Load** the gauge configuration.
2. **Copy** the original gauge links into a separate storage.
3. **Apply stout smearing** (1 step, $\rho=0.125$, ndim=4) to the working gauge field for Dirac inversions.
4. **Invert** $D_u$ and $D_s$ using the smeared links → $S_u$, $S_s$.
5. **Build Wilson line** $W$ from the stored **original** links.

Without this explicit copy step, the stout smearing would overwrite the original links and the Wilson line would be constructed from smeared links, producing incorrect nonlocal correlators.

## Propagator Strategy

| Propagator | Flavour | Source | Gauge for Inversion | Sink |
|-----------|---------|--------|--------------------|-------|
| `prop_l` | light (u) | point, $[0,0,0,0]$ | stout-smeared | nonlocal-shifted: $\tilde S_u(\vec x,t) = W(\vec x,\vec x+z\hat z;t) \cdot S_u(\vec x+z\hat z,t;0)$ |
| `prop_s` | strange (s) | point, $[0,0,0,0]$ | stout-smeared | point (no shift) |

Both inversions use the two-level multigrid solver: level-1 `[6,6,6,3]`, level-2 `[4,4,4,6]`, tolerance $10^{-12}$, max 10000 iterations.

## Nonlocal Shift Implementation

For each $z \in [0,10]$ and each lattice site $(\vec x,t)$:
1. Build the straight Wilson line $W(\vec x,\vec x+z\hat z;t)$ from original $U_z$ links along $+\hat z$ for $k = 0,\dots,z-1$.
2. Handle periodic boundary: $(x_z + k) \bmod L_z$ where $L_z=24$.
3. Form $\tilde S_u(\vec x,t) = W \cdot S_u(\vec x+z\hat z,t;0)$ as a color-matrix × color-spinor-matrix product.
4. Contract: $C(z,t) = +\sum_{\vec x}\operatorname{Tr}[S_s^\dagger(\vec x,t;0) \cdot \tilde S_u(\vec x,t)]$.

## Output Format (Explicitly Specified)

- **File**: `kaon_nonlocal_2pt.txt`
- **Layout**: 11 rows ($z=0,\dots,10$), each with 72 entries ($t=0,\dots,71$)
- **Complex format**: `(re,im)` — parentheses, comma-separated real and imaginary parts, no spaces inside parentheses
- **Entry separator**: single space
- **No header, no metadata, no trailing whitespace**