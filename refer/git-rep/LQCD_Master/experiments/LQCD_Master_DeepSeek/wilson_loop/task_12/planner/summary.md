## Physics objective

Extract the static quark-antiquark potential signal from all 16 rectangular Wilson loops W(R,T) with R, T ∈ {1,2,3,4} on the C24P29 ensemble (24³×72, β=6.20, a≈0.105 fm).  Each loop is averaged over the three spatial-temporal planes XT, YT, ZT.  The measurement uses a single unsmeared gauge configuration (cfg 10000) and requires no quark propagators or Dirac inversions.

## Revisions applied

Three issues from the peer review were addressed:

1. **Explicit `gauge.toDevice()` call inserted.**  The original plan loaded the gauge configuration via `io.readChromaQIOGauge()` but never transferred the field to the GPU.  In PyQUDA, `gauge.loop()` operates on device memory; omitting `toDevice()` causes a silent failure or empty results.  The revised freeform plan places `gauge.toDevice()` immediately after loading (step 2).

2. **Output format clarified and made consistent.**  The original `plan_yaml` metadata claimed a "two-column text file" while the freeform plan wrote three values per line (`R  T  W_val`).  The output is now consistently described as a three-column space-separated text file with columns R, T, and W_val, one line per (R,T) pair, no header.  This matches the original task instruction "(R T value per line)".

3. **Explicit local lattice shape for `gatherLattice` specified.**  The generic instruction "reshape to local lattice" was replaced with the concrete shape `(2, Lt//grid[3], Lz, Ly, (Lx//grid[0])//2) = (2, 18, 24, 24, 12)`, derived from the grid `[1,1,1,4]` and lattice `[24,24,24,72]`.  This eliminates any ambiguity for the executor about the correct reshape before the MPI gather.

## Strategy (unchanged)

- Pure-gauge only: load unsmeared SU(3) links, no Dirac inversions.
- No link smearing (stout/APE/HYP) — preserves bare short-distance physics.
- Plane averaging (XT+YT+ZT) exploits cubic symmetry for a ~×3 statistics gain.
- PyQUDA `gauge.loop()` with 4-group packing, `getHost()→reshape→trace`, MPI gather.
- Single configuration (code-validation run); multi-config extension requires only a loop over `cfg_num`.