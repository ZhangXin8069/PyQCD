## Physics Objective

Compute the rectangular Wilson loop $W(R=3, T=1)$ on the C24P29 ensemble
($24^3 \times 72$, $\beta = 6.20$, $a \approx 0.1052$ fm) using
configuration 10000. The Wilson loop is a pure-gauge observable defined as
the normalized real trace of the path-ordered product of SU(3) gauge links
around a closed rectangular $3 \times 1$ path. With $T = 1$, the loop probes
short-distance gluonic fluctuations; extracting a static potential would
require $T \to \infty$ extrapolation, which is beyond the scope of this
single-configuration, single-geometry measurement.

## Measurement Strategy

1. **Three-plane averaging**: Construct identical $3 \times 1$ rectangular
   paths in the XT, YT, and ZT planes. Each path consists of 8 hops: 3
   forward in the spatial direction, 1 forward in time, 3 backward in space,
   and 1 backward in time. Averaging over the three equivalent
   spatial-temporal planes improves statistics by $\sqrt{3}$ at negligible
   extra cost since all gauge links are already loaded.

2. **Unsmeared links**: The original gauge links are used directly without
   any stout, APE, or HYP smearing. This preserves the bare ultraviolet
   properties of the configuration and satisfies the explicit requirement
   to avoid link smoothing.

3. **PyQUDA `gauge.loop()`**: The built-in parallel Wilson-loop evaluator is
   used with the 4-group packing convention (three active plane groups plus
   one dummy with zero weight). This correctly handles MPI domain
   decomposition and per-site reduction.

4. **Single-number output**: After MPI gather and normalization by
   $(N_c \times N_{\text{sites}})$, the final scalar is written as a plain
   ASCII float to `wilson_loop_R3_T1.txt` with no header or metadata.

## Technical Details

| Item | Value |
|------|-------|
| Lattice | $24^3 \times 72$, periodic spatial / anti-periodic temporal BC |
| MPI grid | $[1, 1, 1, 4]$ — partitioned in t-direction |
| $N_c$ | 3 |
| Path length | 8 hops per plane |
| Normalization | $W = \frac{1}{N_c \cdot N_{\text{sites}}} \sum \mathrm{Re\,Tr}\,U_{\text{loop}}$ |
| I/O | Chroma QIO LIME gauge load; plain-text scalar write |

## Requirement Checklist

- ✅ Pure gauge only — no quark propagators, Dirac solvers, or fermion inversions
- ✅ No link smearing of any kind
- ✅ Average over XT, YT, and ZT planes
- ✅ $R = 3$, $T = 1$
- ✅ Bare scalar output to `.txt` file
- ✅ Ensemble parameters preserved exactly from the fixed configuration