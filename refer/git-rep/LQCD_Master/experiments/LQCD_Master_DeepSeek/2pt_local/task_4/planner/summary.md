## Physics Objective

Compute the two-point correlation function of the $\eta_c$ meson, the ground-state pseudoscalar charmonium state ($c\bar{c}$, $J^{PC}=0^{-+}$). The correlator is used to extract the $\eta_c$ mass via exponential fit at large Euclidean time.

## Strategy

- **Operator**: Local pseudoscalar interpolating operator $\mathcal{O}_{\eta_c} = \bar{c}\gamma_5 c$.
- **Correlator**: Zero-momentum two-point function
  $$C_{\eta_c}(t) = \sum_{\vec{x}} \mathrm{Tr}\left[ S_c^\dagger(\vec{x},t; \vec{0},0)\, S_c(\vec{x},t; \vec{0},0) \right]$$
  Only the connected diagram is evaluated. Disconnected contributions are OZI-suppressed for charmonium and safely neglected at this stage.
- **Source**: Single point source at $[0,0,0,0]$ (one measurement per configuration).
- **Gauge links**: Stout-smeared before inversion with parameters $(n_{\text{steps}}=1,\ \rho=0.125,\ n_{\text{dim}}=4)$ to suppress UV fluctuations and improve the signal.

## Technical Details

| Item | Value |
|------|-------|
| Ensemble | C24P29 ($24^3\times 72$, $a \approx 0.105$ fm, $m_\pi \approx 290$ MeV) |
| Configuration | `cfg_10000` |
| Quark action | Wilson-clover, $c_{\text{sw}} = 1.160920226$ |
| Charm mass parameter | $\kappa_c$ corresponding to $am_c = 0.4159$ |
| Solver | Multigrid, tolerance $10^{-12}$, max 10000 iterations |
| Dirac matrix | $\gamma_5$-hermitian, anti-periodic temporal BC |

## Output

Plain text file (no header/metadata) with one correlator value per time slice, written to the run directory.

## Reasonable Completions

- Assumed zero momentum projection ($\vec{p} = \vec{0}$) at source and sink, which is standard for ground-state mass extraction.
- Used only the connected diagram; the disconnected piece is OZI-suppressed for charmonium and its omission is standard in first-pass spectroscopy.
- No sink smearing applied, consistent with the use of a point source.
- Solver tolerance of $10^{-12}$ is conservative and appropriate for the relatively light computational cost of charm-quark inversions.