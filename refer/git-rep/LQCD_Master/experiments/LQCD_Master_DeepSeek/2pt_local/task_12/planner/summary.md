## Physics Objective

Compute the zero-momentum two-point correlation function of the Σ⁺ baryon (flavor uus, J^P = 1/2⁺) on gauge configuration 10000 from the C24P29 ensemble (24³×72, a≈0.105 fm, N_f=2+1 clover-improved Wilson fermions). This is a **single-configuration debugging/validation run** — no statistical error estimate is obtained.

## Interpolating Operator and Diquark Structure

The Σ⁺ is constructed with a flavor-symmetric uu diquark using Cγ₁ (= Cγ_x):

$$\mathcal{O}_{\Sigma^+} = \epsilon^{abc} (u^{Ta} C\gamma_1 u^b) s^c$$

In the DeGrand-Rossi basis, Cγ₁ is **antisymmetric** under quark exchange [(Cγ₁)^T = −Cγ₁]. Paired with the symmetric uu flavor wavefunction and the antisymmetric color ε-tensor, this satisfies the Pauli principle and corresponds to a **spin-0 "good" diquark**. The positive-parity ground state is isolated with the projector $P^+ = (1+\gamma_4)/2$ applied at the sink. The spin-0 diquark structure suppresses coupling to the spin-3/2 Σ*(1385), though the P⁺ projector alone does not fully forbid it.

## Wick Contraction

Two topologies arise from the two ways of pairing u quarks between source and sink, analogous to the proton case. After color-index relabeling both contribute with the same sign:

$$C_{\Sigma^+}(\vec{0}; t,0) = \sum_{\vec{x}} \epsilon^{abc}\epsilon^{a'b'c'} (C\gamma_1)_{\alpha\beta}(C\gamma_1)_{\alpha'\beta'} P^+_{\gamma'\gamma} \Big[ S_{l\,\alpha\alpha'}^{aa'} S_{l\,\beta\beta'}^{bb'} S_{s\,\gamma\gamma'}^{cc'} + S_{l\,\alpha\gamma'}^{aa'} S_{l\,\beta\beta'}^{bb'} S_{s\,\gamma\alpha'}^{cc'} \Big](\vec{x},t)$$

## Propagators and Solver

| Propagator | Flavor | Source | Gauge links |
|---|---|---|---|
| `prop_l` | light (κ=−0.277) | Point at [0,0,0,0] | 1-step stout (ρ=0.125, 4D) |
| `prop_s` | strange (κ=−0.2356) | Point at [0,0,0,0] | 1-step stout (ρ=0.125, 4D) |

**Critical solver note**: The clover-improved Wilson Dirac operator is only γ₅-Hermitian, not Hermitian positive-definite. Standard CG **must** operate on the normal equations D†D (CGNR). The multigrid preconditioner (block sizes [6,6,6,3] → [4,4,4,6]) is applied to the normal equations. Tolerance 10⁻¹², max 20000 iterations, clover coefficient c_SW = 1.160920226.

## Limitations and Caveats

| Item | Status | Note |
|---|---|---|
| Configurations | Single (cfg 10000) | No error bars; production needs ≥50–100 cfgs |
| Source smearing | None (point) | Limits ground-state overlap; Gaussian/Wuppertal smearing recommended for production |
| Diquark operator | Cγ₁ only | Valid spin-0 diquark; no μ-sum may reduce statistics vs. standard Cγ_μ interpolator |
| Output | Two-column txt (t, C(t)) | No header; time-slice index enables postprocessing |

## Output Format

Plain text file with two space-separated columns per line: time-slice index `t` (0…71) and correlator value `C(t)`. No header line, no metadata in the file. Saved to the run directory.