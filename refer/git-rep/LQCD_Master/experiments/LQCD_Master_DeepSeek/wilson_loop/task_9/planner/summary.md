## Physics Objective

Compute the rectangular Wilson loop W(R=3, T=3) as a pure-gauge observable on the C24P29 ensemble (24³×72, a≈0.1052 fm). The Wilson loop probes the static quark-antiquark potential at separation R=3a ≈ 0.316 fm and is a fundamental probe of confinement via the area law.

## Strategy

- **Task mode**: `freeform` — pure-gauge Wilson loop computation, outside the standard hadrons→propagators→correlators chain.
- **Plane averaging**: Compute W(R=3,T=3) on three spatial-temporal planes (XT, YT, ZT) independently, then average the per-site real trace over all three planes. This exploits cubic symmetry for improved statistics.
- **No smearing**: Unsmeared gauge links used directly as requested; no stout/APE/HYP smearing is applied.
- **No fermions**: Pure gauge measurement — load gauge links, compute Wilson loop via `gauge.loop()`, extract per-site ReTr, MPI-gather, and normalize.
- **Single configuration**: Process configuration 10000.
- **Output**: A single float value written to a plain text file with no header or extra text.

## Key Fixes from Review

1. **Dummy group specification** (critical): PyQUDA's `gauge.loop()` requires exactly 4 outer groups. The plan now explicitly states: groups `[[path_XT],[path_YT],[path_ZT],[path_XT]]` with weights `[1,1,1,0]`, where the 4th group repeats `path_XT` as a dummy. Without this, an empty or invalid 4th group would crash at runtime.

2. **MPI gather reshape** (critical): Before calling `core.gatherLattice()`, the per-site ReTr array must be reshaped to `local_shape = (2, Lt//grid[3], Lz, Ly, (Lx//grid[0])//2)`. With grid `[1,1,1,4]` on 24³×72, this is `(2, 18, 24, 24, 12)`. The plan now computes this shape explicitly and explains that it matches PyQUDA's internal even-odd + local lattice layout.

3. **Required imports** (minor): The freeform plan now includes explicit import statements: `from pyquda_utils import core, io` and `from pyquda_utils.core import X, Y, Z, T`. Without these, a naive code generator would produce `NameError` for direction constants and module references.

## Technical Details

| Parameter | Value |
|-----------|-------|
| Lattice | 24³×72 |
| MPI grid | [1,1,1,4] |
| R, T | 3, 3 |
| Planes | XT, YT, ZT |
| Smearing | None |
| Nc normalization | 1/Nc |
| gauge.loop() groups | [[XT],[YT],[ZT],[XT]] |
| gauge.loop() weights | [1, 1, 1, 0] |
| local_shape for gather | (2, 18, 24, 24, 12) |
| Output format | Single float in .txt file |