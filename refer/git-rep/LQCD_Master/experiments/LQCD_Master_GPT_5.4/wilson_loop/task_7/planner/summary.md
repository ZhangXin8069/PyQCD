## Physics classification
This task is a **pure-gauge measurement**: a rectangular **Wilson loop** evaluation, not a standard hadron correlator problem. Therefore the correct category is **freeform**, not spectroscopy or matrix-element production. There are **no hadrons, quark propagators, source-sink choices, Dirac solves, Wick contractions, or renormalization constants** involved.

## Observable and core objects
The requested observable is
- **Rectangular Wilson loop** \(W(R=2,T=3)\)
- averaged over the **XT, YT, and ZT** planes.

The physical object is the gauge-link product around a closed rectangle of spatial extent 2 and temporal extent 3. The measurement is the volume average of the normalized real trace of that closed loop, then the arithmetic average over the three planes.

## Numerical scheme
I set the plan to:
1. initialize PyQUDA on the provided ensemble geometry and MPI grid,
2. load configuration **10000** from the exact path template provided,
3. construct three loop paths for XT, YT, and ZT,
4. call `gauge.loop()` with the PyQUDA-required 4-group packing,
5. extract per-site \(\mathrm{ReTr}\) values,
6. average over the three planes,
7. reduce across MPI ranks,
8. divide by \(V N_c\) with \(N_c=3\),
9. write the final scalar to a plain `.txt` file in the run directory.

## Reasonable completions I made
Because the user did not specify a few implementation details, I completed them conservatively:
- **task_mode = freeform** because Wilson loops are non-standard gauge observables and should not be forced into hadron/propagator sections.
- **Unsmeared links** were chosen, since no stout/APE/HYP smearing was requested.
- **Output filename** chosen as `wilson_loop_R2_T3_avg.txt` in the run directory.
- **Resource path** chosen as `./tunecache` so the script can run in the current directory without placeholders.
- **Only rank 0 writes output**, which is the standard MPI-safe behavior.
- `output.metadata` set to `false` because the user explicitly requested **no header or extra text**.

## Requirement satisfaction
The plan explicitly satisfies all user constraints:
- computes **W(R=2,T=3)**,
- averages over **XT, YT, ZT**,
- uses **gauge links only**,
- includes **no quark propagators / no Dirac solvers / no inversions**,
- writes **plain text only** to a `.txt` file in the run directory,
- includes the required **ensemble block** exactly from the provided ensemble information.