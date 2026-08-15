## Physics objective

Compute the three-point correlation function for the **Bc⁻ → J/ψ** semileptonic
transition mediated by the **b → c vector current** \(\bar{c}\gamma_x b\).
This is the raw lattice observable needed to later extract the
\(B_c \to J/\psi\) form factors.

## Strategy

- **Source**: Bc⁻ meson at \(t=0\), flavor anti‑c + b, pseudoscalar
  interpolator \(\gamma_5\), point source at \([0,0,0,0]\), zero momentum.
- **Sink**: J/ψ meson at \(t_{\text{seq}}=8\), flavor anti‑c + c, vector
  interpolator \(\gamma_x\) (single polarisation), **zero‑momentum projection
  via coherent spatial sum** over all \(\mathbf{x}\) in the sequential‑source
  construction.
- **Current**: local vector current \(\bar{c}\gamma_x b\) inserted at all
  intermediate times \(\tau\in[0,t_{\text{seq}}]\).

The connected Wick contraction involves three quark lines: (i) the **b** quark
propagates from the source to the current insertion, (ii) the **c** quark
created at the current propagates to the sink (handled by the sequential
propagator), and (iii) the spectator **anti‑c** propagates from the source to
the sink (handled by the forward charm propagator, which also serves as the
base for the sequential source).  No two‑point functions are computed.

## Key revisions from the previous plan

1. **Zero‑momentum projection** (critical fix).  The original plan placed the
   sequential source at a single spatial point \([0,0,0,8]\), which corresponds
   to a point‑sink approximation and contaminates the result with all sink
   momenta.  The revised plan constructs the sequential source by **summing
   the forward charm propagator over all spatial positions** at \(t_{\text{seq}}=8\),
   yielding a coherent wall‑like source that correctly implements
   \(\mathbf{p}_f = \mathbf{0}\) projection.

2. **Source sign convention**.  The Bc⁻ creation operator is
   \(\mathcal{O}_{B_c}^\dagger = -\bar{b}\gamma_5 c\); the minus sign
   arises from \(\{\gamma_4,\gamma_5\}=0\).  This factor of \(-1\) must be
   carried through the contraction to obtain the correct overall correlator
   sign.  The plan flags this explicitly so the code‑generation step accounts
   for it.

3. **Bottom‑quark solver**.  The bottom quark mass \(am_b = 1.5\) is too
   heavy for reliable multigrid convergence.  The solver is changed from
   multigrid to a **direct CG / BiCGStab** solver with tolerance relaxed to
   \(10^{-8}\) and maxiter raised to 5000.  Validation on a test
   configuration before production is strongly advised.

4. **Sequential charm solver**.  Tolerance relaxed from \(10^{-12}\) to
   \(10^{-10}\) and maxiter increased to 3000, because the sequential source
   is intrinsically noisier than a point source and tight tolerances do not
   improve the physical signal.

5. **Forward charm solver**.  Maxiter increased from 1000 to 2000; a BiCGStab
   fallback is specified in case multigrid fails to converge.

## Technical summary

| Item | Choice |
|---|---|
| Gauge ensemble | C24P29, \(24^3\times 72\), \(a=0.1052\;\text{fm}\) |
| Configuration | `cfg 10000` |
| Source | point, \([0,0,0,0]\), \(\vec{p}=\vec{0}\) |
| Gauge smearing | 1‑step stout, \(\rho=0.125\), 4‑D |
| Charm mass (bare) | 0.4159 |
| Bottom mass (bare) | 1.5 |
| Clover coefficient | 1.160920226 |
| Charm forward solver | multigrid (BiCGStab fallback), tol \(10^{-12}\), maxiter 2000 |
| Bottom forward solver | **CG/BiCGStab**, tol \(10^{-8}\), maxiter 5000 |
| Charm sequential solver | CG/BiCGStab, tol \(10^{-10}\), maxiter 3000 |
| MPI grid | \(1\times 1\times 1\times 4\) |
| \(t_{\text{seq}}\) | 8 |
| Sink momentum projection | **coherent spatial sum** (wall‑like sequential source) |
| Output | plain text file, one number per \(\tau\), no header/metadata |

## Known risks

- **\(t_{\text{seq}}=8\)** (\(\approx 0.84\;\text{fm}\)) is short for heavy‑meson
  systems; expect significant excited‑state contamination in both Bc⁻ and J/ψ
  channels.  Ground‑state dominance should be checked a posteriori.
- **Bottom solver convergence** must be validated on a test configuration;
  if 5000 iterations are insufficient, further tolerance relaxation or a
  different solver algorithm may be needed.
- **generate_einsum** must be verified to support the distinct‑flavor meson‑3pt
  topology (\(\bar{c}b \to \bar{c}c\)).  If unsupported, the contraction must
  be implemented manually following the Wick‑contraction derivation.