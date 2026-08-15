## Physics objective

Compute the rectangular Wilson loop expectation value \(W(R=2,T=2)\) on a single gauge configuration (C24P29, cfg 10000) from the 24³×72 lattice at \(a \approx 0.1052\) fm.  This is a pure-gauge observable probing the static quark–antiquark potential at short distance; the area-law behaviour \(\langle W(R,T)\rangle \sim e^{-\sigma RT}\) encodes the confining string tension.

## Strategy

1. **Pure-gauge calculation**: Load gauge links directly — no Dirac inversions, no quark propagators, no sequential sources.
2. **No smearing**: The original unsmeared links are used.  While smearing would improve the signal-to-noise ratio, the task explicitly forbids it.
3. **Three-plane averaging**: Compute the Wilson loop in the XT, YT, and ZT planes with equal weight to exploit cubic symmetry and increase effective statistics by a factor of ~3 relative to a single plane.
4. **Path construction**: Forward X² → forward T² → backward X² → backward T² (and analogous for Y, Z), giving a closed rectangular 2×2 loop.
5. **MPI-parallel reduction**: The 4-way MPI decomposition in the t-direction is handled by `gatherLattice` to sum over all sublattices before averaging.

## Technical details

- **PyQUDA `gauge.loop()`** requires exactly 4 outer groups; three carry weight 1 (XT, YT, ZT) and the fourth is a dummy with weight 0.
- Per-site trace extraction uses `getHost()` → `reshape(-1, 3, 3)` → `np.trace(axis1=-2, axis2=-1).real`.
- Final value = `global_sum.sum() / (total_sites × N_c)` where `total_sites = 24×24×24×72 = 995328` and `N_c = 3`.
- Output is a single float written to `wl_R2_T2_cfg10000.txt` with no header or extra text.

## Satisfying requirements

| Requirement | How satisfied |
|---|---|
| W(R=2,T=2) | Paths use exactly 2 steps in the spatial direction and 2 steps in the temporal direction |
| Average over XT, YT, ZT | Three distinct paths computed in one `gauge.loop()` call, averaged with equal weight |
| Pure-gauge only | No propagators, no Dirac solvers, no quark fields |
| No smearing | `stoutSmear()` is never called; original gauge links used as-is |
| Save to txt | Plain-text file with the scalar value only |

## Reasonable completions

- **Single configuration**: cfg 10000 is processed; multi-configuration averaging is left for a separate production run.
- **No statistical analysis**: Jackknife/bootstrap not performed because only one configuration is available.
- **Output filename**: `wl_R2_T2_cfg10000.txt` in the run directory.
- **Dummy 4th group**: The XT path is repeated with weight 0 to satisfy PyQUDA's 4-group requirement.