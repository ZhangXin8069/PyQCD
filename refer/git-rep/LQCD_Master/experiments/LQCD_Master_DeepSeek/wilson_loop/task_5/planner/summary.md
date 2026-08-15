## Physics Objective

Compute the rectangular Wilson loop W(R=1, T=3) as a pure-gauge observable on the C24P29 ensemble (24³×72, β=6.20, a≈0.105 fm). At R=1 the loop probes the short-distance, Coulomb-like regime of the static quark-antiquark potential. The measurement uses a single configuration (cfg 10000) as a spot check, providing a baseline value without statistical interpretation (no error bar, no ensemble average).

## Strategy

- **Pure-gauge only**: Gauge links are loaded directly via `io.readChromaQIOGauge`. No quark propagators, Dirac inversions, or fermion fields are involved.
- **Unsmeared links**: No stout, APE, or HYP smearing is applied. The raw gauge links are used verbatim to preserve the unmodified short-distance signal.
- **Plane averaging**: The rectangular loop (R=1 spatial, T=3 temporal) is computed in all three spatial-temporal planes (XT, YT, ZT) and averaged. This exploits spatial rotational symmetry for a modest statistical gain even from a single configuration.
- **Path convention**: Forward R steps along the spatial direction, forward T steps along the temporal direction, then backward R and backward T to close the loop.

## Implementation

- **Library**: PyQUDA with CuPy GPU backend, 4 MPI ranks split along the t-direction (grid [1,1,1,4]).
- **API**: `gauge.loop()` with the 4-group packing convention (3 active planes + 1 weight-0 dummy group).
- **Extraction**: Each plane's result is transferred to host, reshaped, and the per-site real trace `Re Tr` is computed via `numpy.trace`. The three-plane average is then MPI-gathered to rank 0 using `core.gatherLattice`.
- **Normalization**: `W = global_sum / (total_sites × Nc)` with `Nc = 3`.
- **Output**: A single floating-point number written to `wilson_loop_R1_T3.txt` with no header or metadata.

## Notes

- The ensemble metadata lists `temporal: anti-periodic` boundary conditions, which is the fermion convention. Gauge fields in pure-gauge measurements obey periodic BC in all directions, but this metadata discrepancy does not affect the Wilson loop computation since the links are read as-is from the configuration file.
- The ensemble block also contains quark-mass, clover, and multigrid parameters that are irrelevant for this pure-gauge task; these are harmless dead metadata carried forward from the ensemble definition.