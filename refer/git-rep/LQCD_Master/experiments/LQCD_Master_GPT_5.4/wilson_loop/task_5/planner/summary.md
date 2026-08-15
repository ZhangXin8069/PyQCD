## Physics classification
This task is a **pure-gauge observable** measurement, specifically a **Wilson loop** calculation. It is **not** a standard hadron spectroscopy or matrix-element workflow, so it should be treated as a **freeform** LQCD task rather than forcing it into the hadrons → propagators → correlators chain.

## Core physics objective
The requested observable is the rectangular Wilson loop
\(W(R=1,T=3)\), averaged over the three spatial-temporal planes:
- XT
- YT
- ZT

The measurement is to be done directly from the gauge links, with no quark propagators, no Dirac solves, and no fermion inversions.

## Numerical strategy
I completed the task with the standard and conservative pure-gauge PyQUDA workflow:
1. Initialize the lattice using the provided ensemble dimensions and MPI process grid.
2. Load the gauge configuration for cfg `10000` from the provided LIME path.
3. Construct the rectangular paths for the three planes:
   - XT: `[X, T, T, T, -X, -T, -T, -T]`
   - YT: `[Y, T, T, T, -Y, -T, -T, -T]`
   - ZT: `[Z, T, T, T, -Z, -T, -T, -T]`
4. Use `gauge.loop()` with the PyQUDA-required 4-group packing, assigning zero weight to the dummy fourth group.
5. Extract per-site loop matrices, compute `Re Tr`, average over volume and over the three planes, and normalize by `Nc=3`.
6. Save the final scalar result as a plain text file in the run directory with no header or extra text.

## Reasonable completions made
Because the user did not specify some implementation details, I made the following conservative choices:
- **No link smearing**: I used the original gauge links, since the task only asked for the basic Wilson loop measurement and did not request stout/APE/HYP smearing.
- **One output file per configuration**: since only cfg `10000` is listed, I specified a single txt output for that configuration.
- **Rank-0 writeout after MPI gather**: this is the standard safe approach for multi-rank jobs.
- **Plain scalar text output**: chosen to satisfy the explicit requirement of “without any header or extra text”.

## Requirement satisfaction
- Correctly classified as **pure-gauge / wilsonloop**.
- Used **task_mode: freeform** as required for a non-standard LQCD task.
- Included the full provided **ensemble** block directly in the YAML.
- Avoided misleading hadron/propagator sections by leaving them empty.
- Produced a runnable, conservative plan with explicit path construction, averaging strategy, MPI treatment, and output convention.

## Notes
The ensemble contains fermion-related metadata such as quark masses, clover coefficient, and multigrid settings. These were preserved in the `ensemble:` section for provenance, but they are not used in this pure-gauge measurement.