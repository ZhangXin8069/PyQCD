# Physics Objective

Compute the rectangular Wilson loop $W(R=2, T=1)$ as a pure-gauge observable from unsmeared gauge links on a single configuration (cfg 10000) of the C24P29 ensemble ($24^3 \times 72$, $\beta=6.20$, $a=0.1052$ fm). No quark propagators, Dirac inversions, or fermion degrees of freedom are involved. Fermion-specific ensemble parameters (quark masses, $c_{\text{SW}}$, multigrid) are noted as irrelevant to this measurement.

# Observable Definition

The Wilson loop is the normalized real trace of the ordered product of gauge links around a closed rectangular path:

$$W_{\mu\nu}(R,T) = \frac{1}{N_c}\text{Re}\,\text{Tr}\,\mathcal{P}\big[U_\mu^R\,U_\nu^T\,U_\mu^{\dagger R}\,U_\nu^{\dagger T}\big]$$

For $R=2, T=1$ with $N_c=3$, each plane yields $W_{\mu\nu}(2,1) = \frac{1}{3}\text{Re}\,\text{Tr}[U_\mu U_\mu U_\nu U_\mu^\dagger U_\mu^\dagger U_\nu^\dagger]$.

# Strategy

To improve statistics from a single configuration, we average over the three independent spatial-temporal planes (XT, YT, ZT), gaining roughly $\sqrt{3}$ in effective statistics via rotational symmetry of the ensemble.

# Key Technical Decisions

1. **No smearing**: The user explicitly forbad stout/APE/HYP smearing. Unsmeared gauge links are used directly, preserving the bare Wilson loop value.

2. **Plane averaging with PyQUDA 4-group packing**: `gauge.loop()` requires exactly 4 outer groups. The three active plane paths occupy groups 0–2 with unit weights; group 3 is a dummy (duplicate XT path) with weight zero.

3. **MPI decomposition**: The grid `[1,1,1,4]` splits the temporal direction across 4 ranks. Each rank holds $L_t/4 = 18$ time slices. The local lattice field shape for `gatherLattice` is explicitly: `(2, 18, 24, 24, 12)`.

4. **Normalization**: The per-site trace is divided by $N_c=3$ and by the total number of lattice sites $24^3 \times 72 = 995328$. The three-plane average divides the summed contributions by 3.

5. **Output**: A single file `wl_R2T1_cfg10000.txt` containing one unadorned floating-point number.

# Revisions from Critique

- Added explicit `local_shape = (2, 18, 24, 24, 12)` computation to the freeform plan so the executor can correctly reshape the per-rank trace array before calling `core.gatherLattice`.
- Specified the output filename as `wl_R2T1_cfg10000.txt` to avoid ambiguity or overwrite.
- Noted in the freeform plan and extras that `quark_mass`, `clover`, and `multigrid` are fermion-only parameters not used in this pure-gauge measurement.