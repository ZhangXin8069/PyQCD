## Physics objective

Compute the rectangular Wilson loop \(W(R=2, T=3)\) on a single gauge configuration (cfg 10000) of the C24P29 ensemble (\(24^3\times72\), \(\beta=6.20\), \(a\approx 0.1052\) fm).  This is a pure-gauge observable extracted directly from the SU(3) link variables — no quark propagators, Dirac inversions, or fermion contractions are involved.

## Strategy

1. **Path definition** — For each of the three spatial-temporal planes (XT, YT, ZT) construct the closed rectangular path: 2 steps forward in the spatial direction, 3 steps forward in time, 2 steps backward spatially, and 3 steps backward temporally.

2. **Smearing** — The user explicitly requires NO link smearing.  Raw unsmeared gauge links are used throughout.

3. **Plane averaging** — Compute \(\frac{1}{N_c}\mathrm{Re}\,\mathrm{Tr}\) of the product of links around each path, average the three planes, then average over all lattice sites.  This gives one scalar value per configuration.

4. **MPI** — The lattice is decomposed in the t-direction across 4 MPI ranks.  Per-site contributions are gathered to rank 0 via `core.gatherLattice` before the final global sum.

5. **Output** — A plain `.txt` file containing a single floating-point number (no header, no metadata).

## Technical details

| Item | Value |
|------|-------|
| Observable | Rectangular Wilson loop W(R=2, T=3) |
| Planes | XT, YT, ZT (equal weight) |
| Smearing | None (raw links) |
| Library | PyQUDA (`gauge.loop()` with 4-group packing) |
| MPI grid | [1, 1, 1, 4] in (x,y,z,t) |
| Gauge file | Chroma QIO LIME format |
| Normalization | \(W = \sum \mathrm{ReTr} / (N_{\mathrm{sites}} \cdot N_c)\), \(N_{\mathrm{sites}} = 995328\), \(N_c = 3\) |
| Output file | `wilson_loop_R2_T3_cfg10000.txt` (one float, no header) |

## Reasonable completions made

- Because the user gave only a single configuration number (10000), the plan targets exactly that configuration.  Multi-configuration averaging would require a loop over `cfg_num` entries, which the executor can add if the list is extended.
- The 4-group packing `[[XT],[YT],[ZT],[XT]]` with weights `[1,1,1,0]` follows the PyQUDA `gauge.loop()` convention that requires exactly 4 outer groups.
- Result normalization uses \(N_c = 3\) and the total number of lattice sites (24×24×24×72 = 995328).
- No jackknife/bootstrap is applied — this is a per-configuration raw measurement.  Statistical analysis would be done downstream on the accumulated per-configuration results.
- The configuration path template uses `{n_cfg}` for formatting; the executor will substitute the actual configuration number.