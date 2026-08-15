## Physics Objective

Compute the two-point correlation function of the **Σ⁻ baryon** (quark content `dds`, isospin I=1, I₃=−1) at zero momentum to enable ground-state mass extraction from the large-time exponential decay.

## Interpolating Operator — Corrected

### Why Cγ₅, not Cγ₁

The Σ⁻ contains two identical `d` quarks in the diquark, so the **Pauli principle** governs the choice of gamma structure. The total wavefunction must be antisymmetric under exchange of the two `d` quarks:

| Degree of freedom | Symmetry |
|---|---|
| Color (ε^{abc}) | Antisymmetric (A) |
| Flavor (both d) | Symmetric (S) |
| Spatial (S-wave, ground state) | Symmetric (S) |
| **Spin-Dirac** | **Must be Antisymmetric (A)** |

The antisymmetry product: A × S × S × A = A ✓.

- **(Cγ₅)^T = −Cγ₅** → antisymmetric in spin indices → **correct, non-zero operator**
- **(Cγ₁)^T = +Cγ₁** → symmetric in spin indices → **operator vanishes identically** when summed with ε^{abc}

The original plan's assertion that "the spin structure must be symmetric" contradicted the Pauli principle and would yield numerical zero or pure noise.

### Operator

$$\mathcal{O}_{\Sigma^-}(\vec{x},t) = \epsilon^{abc}\, \bigl(d^{Ta}(\vec{x},t)\, C\gamma_5\, d^b(\vec{x},t)\bigr)\, s^c(\vec{x},t)$$

### Two-Point Correlator

$$C_{\Sigma^-}(t) = \text{Tr}\Bigl[P^+ \sum_{\vec{x}} \langle \mathcal{O}_{\Sigma^-}(\vec{x},t)\; \bar{\mathcal{O}}_{\Sigma^-}(\vec{0},0) \rangle\Bigr],\qquad P^+ = \frac{I + \gamma_4}{2}$$

### Wick Contraction

Two topologies (direct and exchange) contribute, both requiring:
- **Two light (d) propagators** — reused from `prop_l` (single inversion)
- **One strange (s) propagator** — from `prop_s`

With a point source at [0,0,0,0]:

$$C_{\Sigma^-}(t) \approx P^+_{\gamma'\gamma}\, \sum_{\vec{x}} \epsilon^{abc}\epsilon^{a'b'c'}\, (C\gamma_5)_{\alpha\beta}\, (C\gamma_5)_{\alpha'\beta'}
\Bigl[ S_{l\,\alpha\alpha'}^{aa'}(x;0)\, S_{l\,\beta\beta'}^{bb'}(x;0)\, S_{s\,\gamma\gamma'}^{cc'}(x;0) + \text{(exchange)}\Bigr]$$

where x = (\vec{x}, t), and the exchange term swaps the sink attachment of one light propagator. The overall sign and contraction are determined by `generate_einsum(type="baryon_2pt")` with the Cγ₅ diquark specification.

## Technical Implementation

| Item | Specification |
|------|---------------|
| Gauge ensemble | C24P29, 24³×72, β=6.20, a≈0.105 fm |
| Configuration | #10000 |
| Quark masses | light κ: −0.277, strange κ: −0.2356 |
| Clover coefficient | c_sw = 1.160920226 |
| Gauge smearing | **Stout**: 1 step, ρ=0.125, 4-dim (applied before both inversions) |
| Source | Point source at [0,0,0,0], single source time |
| Solver tolerance | 1×10⁻¹² |
| Solver max iterations | 20000 |
| Multigrid | 2-level: (6,6,6,3) → (4,4,4,6) |
| MPI decomposition | 1×1×1×4 process grid |
| Diquark gamma | **Cγ₅ only** (required by Pauli principle; no gamma averaging) |
| Parity projector | P⁺ = (I + γ₄)/2 |
| Required propagators | `prop_l` (light, for both d quarks) + `prop_s` (strange) |

## Known Limitations

- **Single point source without smearing**: Ground-state overlap for the Σ⁻ baryon will be poor, leading to large excited-state contamination and slow plateau formation. This is acceptable for a debugging/validation run but would need Gaussian/Wuppertal smearing for production physics.
- **Sign convention**: The conjugate operator with Cγ₅ involves γ₄(Cγ₅)†γ₄C, whose sign differs from the Cγ₁ case. The downstream `generate_einsum` toolchain must be verified to handle this correctly; a quick cross-check against the known proton case (which also uses Cγ₅) is advised.

## Output

Raw correlator values C(t) for t = 0…71 are saved as plain text (no header, no metadata) to a `.txt` file in the run directory.