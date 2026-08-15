## Physics Objective

Compute the rectangular Wilson loop $W(R=1, T=4)$ averaged over the three spatial-temporal planes (XT, YT, ZT) as a pure-gauge observable. The Wilson loop is the fundamental probe of the static quark-antiquark potential; at short distance $R=1$ it is dominated by the Coulomb (perturbative) contribution and serves as a baseline for confinement studies at larger $R$.

## Category Classification

This is a **pure-gauge measurement** — it involves only gauge links with no quark propagators, Dirac solvers, or fermion inversions. It is therefore classified as `task_mode: freeform`, with the full computational strategy described in `freeform_plan`.

## Technical Strategy

- **Gauge configuration**: Load the unsmeared gauge links from the provided LIME-format Chroma QIO file (cfg 10000) on a $24^3 \times 72$ lattice.
- **No smearing**: Per the user's explicit instruction, no stout/APE/HYP link smearing is applied.
- **Wilson loop path** ($R=1, T=4$): each plane path consists of 1 spatial step forward, 4 temporal steps forward, 1 spatial step backward, 4 temporal steps backward (10 total links).
- **Plane averaging**: XT, YT, and ZT planes are computed via `gauge.loop()` with 3 active groups (weight 1 each) and 1 dummy group (weight 0) to satisfy PyQUDA's 4-group API requirement.
- **MPI reduction**: `gatherLattice` sums per-site $\frac{1}{N_c}\mathrm{Re}\,\mathrm{Tr}$ over all MPI ranks (4 ranks in the t-direction).
- **Normalization**: $W = \sum \mathrm{ReTr} / (V \times N_c)$ where $V = 24^3 \times 72$.
- **Output**: A single plain-text file `W_R1_T4.txt` containing only the numerical value with no header.

## Reasonable Design Choices

- The user specified configuration 10000 only; the plan uses this single configuration.
- All three spatial-temporal planes (XT, YT, ZT) are averaged with equal weight to improve statistics over a single-plane measurement.
- The 4th `gauge.loop()` group is padded with a duplicate XT path at weight 0, which is the standard PyQUDA convention for handling fewer than 4 active planes.
- MPI grid [1,1,1,4] from the ensemble specification is used as-is.