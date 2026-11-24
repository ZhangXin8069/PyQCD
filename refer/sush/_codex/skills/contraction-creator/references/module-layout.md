# Read the actual template structure

The selected file is the structural authority. M0–M8 are existing comments in bundled assets, not a specification for reconstructing or normalizing a script. Do not add them to external references without such comments.

| Existing area | Preserve |
| --- | --- |
| Imports / M0 | Import style/order, backend, interface, existing total timer |
| M1 | MPI initialization, rank/size/communicator, geometry and `vertex_creator` placement |
| M2 | Momentum ordering, phases, cutoff/time declarations and `trank` at existing locations |
| M3 | Existing operator/flag declarations; no new runtime manifest |
| M4 | Array residency, eigenvector/VDV/VVV construction, omega and timer positions |
| M5–M6 | Actual peram, allocation and registry ordering in the selected file |
| M7 | Broadcasts, loop nesting, registrations, dynamic call and accumulation |
| M8 | Gather axis, `loop_tsrc`, source estimator, filenames and existing timing |

The assets are **not interchangeable block by block**. The 2pt correlator is allocated near M3 and peram distribution occurs before registries; the 3pt correlator is allocated near M5 and peram distribution occurs after registries near M7. The 2pt VVV is on the host and the bundled 3pt VVV on the backend. Preserve these differences.

Both assets distribute perams with `get_mpi_data(..., mdtype='TScatter', axis=1)`. The 2pt `trank` uses `gtype='Scatter'`; the current `get_mpi_tlist` implements it in the same cyclic branch as `TScatter`. Check the source when changing libraries; time-list aliases do not make data-distribution operations interchangeable.

Dynamic-first means deriving required tensors **during preparation**, before changing loading. It does not mean inserting `dynamic_plan`, `required_peram`, `required_v`, `required_gamma`, or a standardized call into the output. Inspect actual plan entries/registrations with `$lqcddb`; retain dynamic call text for unchanged inputs.

Use actual registry resolution when checking the plan. In the current source, `p[2]` is a combined Wick flavor string such as `dd^d`, not the registry key `d` or `light`. `PeramRegistry.resolve(p[2], p[4])` uses the quark flavor and permits the registered `light` fallback for u/d. Do not prescribe raw `(p[2], tuple(p[4]))` as registration keys or force duplicate registrations to satisfy that mistaken comparison.

Likewise, raw Wick V names can be globally numbered; dynamic contraction renumbers VDV and VVV independently at each time end via `_per_region_v_names`. For example, raw `VDV_1@tcur0` can resolve as `VDV_0@tcur0`. Use the selected API's `validate_plan`/tensor-resolution path, not raw `(v[1], v[3])` equality, to diagnose missing registrations. This lookup check does not validate numerical shapes or physics.

Parameter/path changes preserve statement order and nested loops/conditionals/collectives. For new physics, preserve unaffected blocks byte for byte and explain necessary topology changes. Never remove unused lines in an exact copy.

Apply the requested [progress/timing policy](progress-timing.md) inline at existing boundaries. The first two main-loop levels are `tsrc -> i` in 2pt and `tsrc -> tcur0` in 3pt. Added timer/print statements do not justify reorganizing these loops, moving data, or inserting MPI collectives.
