## Physics Objective
Compute the rectangular Wilson loop $W(R=3, T=2)$ on the C24P29 ensemble (configuration 10000) as a pure-gauge observable. The Wilson loop is defined as

$$W_{\mu\nu}(R,T) = \frac{1}{N_c} \text{Re}\,\text{Tr}\,\mathcal{P}\left[\prod U_\mu \times \prod U_\nu \times \prod U_\mu^\dagger \times \prod U_\nu^\dagger\right]$$

with $N_c=3$. The measurement is averaged over three spatial-temporal planes (XT, YT, ZT) to improve statistics, and no gauge-link smearing is applied.

## Strategy
- **Task mode**: `freeform` — this is a pure-gauge Wilson loop measurement with no fermion propagators, Dirac solvers, or Wick contractions.
- **Single configuration**: cfg=10000 from C24P29 ($24^3 \times 72$, $\beta=6.20$, $a \approx 0.1052\,\text{fm}$).
- **No smearing**: Use original unsmeared gauge links directly.
- **Plane averaging**: Compute W(R=3,T=2) independently for XT, YT, ZT planes and average the per-site ReTr values. This triples the effective statistics compared to a single-plane measurement.
- **MPI**: 4 ranks distributed in the T-direction (process_grid=[1,1,1,4]). Results are gathered to rank 0 via `core.gatherLattice` and summed.
- **Output**: A single floating-point number written to `wl_R3_T2.txt` with no header or extra text.

## Technical Details
- PyQUDA `gauge.loop()` requires exactly 4 outer groups; three active groups carry the XT/YT/ZT paths with weight 1 each, the fourth is a dummy with weight 0.
- Per-site extraction: `getHost() → reshape(-1, Nc, Nc) → np.trace().real`.
- Normalization: $W = \sum_{\text{sites}} \text{ReTr}_{\text{avg}} / (V \times N_c)$ where $V = 24 \times 24 \times 24 \times 72 = 995328$.

## Reasonable Completions
- The user specified R=3, T=2 explicitly; no extrapolation in R or T is performed.
- The user requested no smearing; this is honored exactly.
- The user requested 3-plane averaging (XT, YT, ZT); this is implemented as described.
- Only one configuration (10000) is measured; this is a single-measurement diagnostic run yielding one number per file.
- MPI grid [1,1,1,4] matches the ensemble's process_grid and is appropriate for a 72-slice temporal lattice.