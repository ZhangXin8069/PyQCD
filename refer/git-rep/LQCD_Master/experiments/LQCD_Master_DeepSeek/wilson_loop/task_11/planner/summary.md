## Physics Objective

Compute the rectangular Wilson loop W(R=4, T=1) as a pure-gauge lattice observable on the C24P29 ensemble (24³×72, β=6.20, a≈0.105 fm). The Wilson loop is a fundamental gauge-invariant probe of the static quark-antiquark potential; at short temporal extent T=1 it is dominated by the Coulomb self-energy and serves as a sensitive check of the lattice discretisation.

## Strategy

1. **Pure-gauge, no quarks**: The task involves only gauge links — no Dirac inversions, quark propagators, or Wick contractions. This is a `task_mode: freeform` job.

2. **Three-plane average**: Compute W(R=4,T=1) independently on the XT, YT, and ZT planes and take the unweighted mean. This exploits rotational symmetry to increase statistical precision by a factor of roughly √3 without extra inversions.

3. **No smearing**: Per the user's explicit instruction, the original unsmeared gauge links are used directly. No stout, APE, or HYP smearing is applied.

4. **Single configuration**: Only configuration number 10000 is processed.

## Technical Details

- **Lattice**: 24³×72, periodic spatial BC, anti-periodic temporal BC.
- **MPI layout**: 4 ranks split in the temporal direction (process grid [1,1,1,4]).
- **Path construction**: Each rectangular loop is built from 4 forward μ steps, 1 forward ν step, 4 backward μ steps, and 1 backward ν step, using PyQUDA direction constants.
- **PyQUDA gauge.loop()**: Requires exactly 4 outer groups; the three active planes are placed in groups 0–2 with weight 1, and group 3 is a dummy with weight 0.
- **Reduction**: Per-site real traces are averaged over the three planes, then MPI-gathered via `gatherLattice` to rank 0, where the global spatial average and 1/N_c normalisation are applied.
- **Output**: The single scalar W_avg is written to `wl_R4_T1.txt` as a bare floating-point number with no header or label.

## Requirement Satisfaction

| Requirement | How Satisfied |
|---|---|
| Compute W(R=4,T=1) | Explicit path construction with R=4, T=1 |
| Average over XT, YT, ZT | Three plane paths, equal-weight average |
| Pure-gauge only | No quark propagators, no Dirac solvers |
| No smearing | No stoutSmear/APE/HYP call in the workflow |
| Save as plain txt | Single float written to wl_R4_T1.txt, no header |

## Reasonable Completions

- **PyQUDA 4-group padding**: The fourth group is a copy of the XT path with weight 0, which is the standard workaround for the `gauge.loop()` API constraint.
- **No jackknife / bootstrap**: Since only one configuration is requested, no statistical resampling is needed; the result is a single per-configuration scalar.
- **No error analysis or effective mass extraction**: This is a pure measurement task; analysis (plateau fits, Creutz ratios, etc.) is left to the user.