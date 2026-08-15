## Task Classification
**Category:** Pure-gauge observable (`task_mode: freeform` — no standard hadrons/propagators/correlators chain).

## Physics Objective
Compute the 1×1 rectangular Wilson loop W(R=1,T=1) — the spatial-temporal plaquette — on a single gauge configuration (cfg 10000 from the C24P29 ensemble, a 24³×72 lattice at β≈6.20). The result is averaged over the three spatial-temporal planes (XT, YT, ZT) to maximise statistics.

## Strategy
1. **Load** the single Chroma QIO gauge configuration using PyQUDA with MPI grid [1,1,1,4].
2. **No smearing** — raw unsmeared gauge links are used throughout, as explicitly required.
3. **Build paths** for the three planes: [X,T,−X,−T], [Y,T,−Y,−T], [Z,T,−Z,−T] using PyQUDA direction constants.
4. **Compute** per-site SU(3) Wilson loops via `gauge.loop()` with the mandatory 4-group packing (three active groups + one dummy weighted zero).
5. **Extract** Re Tr at each site via `.getHost()` → `.reshape(−1,3,3)` → `np.trace(…).real`, average over the three planes, then MPI-reduce with `core.gatherLattice()` to rank 0.
6. **Normalise**: W = (sum of Re Tr) / (995328 × 3).
7. **Save** a single float value to `wilson_loop_R1_T1_cfg10000.txt` with no header, label, or trailing whitespace.

## Requirement Satisfaction
- **Pure-gauge only**: no quark propagators, Dirac solvers, or fermion inversions. ✓
- **No smearing**: stout, APE, and HYP are all explicitly excluded. ✓
- **Plane averaging**: XT, YT, ZT planes combined with equal weight. ✓
- **Plain text output**: single float, no metadata. ✓
- **Single configuration**: cfg 10000 as specified. ✓

## Revisions from Critique
No issues were identified in the critique; the plan requires no substantive changes. The ensemble block and all task specifications remain identical to the original submission.