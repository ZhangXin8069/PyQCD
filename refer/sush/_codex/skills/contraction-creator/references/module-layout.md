# Dynamic-First Module Layout

The two NA800 examples are explained and generated in the fixed order below. A module is a stable top-level block in one script; it is not a new runtime framework, and array loading need not be hidden in helper functions. Replace only physical quantities, paths, and registry data while keeping block order, variable names, and timer positions unchanged.

## Fixed module order

| Module | Purpose | Main output variables |
| --- | --- | --- |
| `M0 Runtime` | Imports, backend, `st_total`, and command-line arguments | `backend`, `conf_id`, `st_total` |
| `M1 Lattice and MPI` | Lattice/process grid, `mpinit`, process identifiers, local sizes, and time mapping | `lattice_size`, `grid_size`, `Nx,Ny,Nz,Nt`, `rank,size`, `trank` |
| `M2 Momentum and phases` | Momentum lists, `vertex_creator`, and phase arrays | `fun_eigen`, `Mom_*`, `Mom_len`, `phase_exp_*` |
| `M3 Dynamic manifest` | Generate Wick/dynamic inputs from the physical target and list registry keys/output axes | `operator_groups`, `Cpt`, `Vindex/Gindex`, required flavor/time keys, V keys, and gamma names |
| `M4 Vertex data` | Load only the eigenvectors required by M3 and construct VDV/VVV; handle links/omega when needed | `VdV_*`, `sink_VVV`, `VVV_*` |
| `M5 Peram data` | Load and perform Scatter/TScatter to obtain the time directions required by M3 | `peram_*`, local/global time mapping |
| `M6 Registries` | Register peram, V, gamma, and Projector in the fixed order | `PR`, `VR`, `GR` |
| `M7 Dynamic loop` | Register current time/momentum slices, create `dynamic_contraction`, and accumulate results | `corr_*`, `st_cal` |
| `M8 Gather and save` | `TGather`/`Gather`, `loop_tsrc`, saving, and stage/total timing | output files, `st_total` runtime |

2pt and 3pt/GEVP may differ physically only in the registrations in `M3` and `M6`, the operator groups/projections in `M7`, and the output axes in `M8`. Keep the text of M0, M1, M2, M4, M5, and M8 as close to identical as possible; if actual 3pt time distribution differs, state the reason only in that module.

## Dynamic-first variable discovery

Do not copy a large loading block and guess what it needs. First write the sink, initial source operator, and every current from `physics.md`; use `conjugate_operator(initial_operator)` for a default source operator and preserve numerical coefficients. Then call the `run_wick_analysis` core used by `dynamic_contraction` (do not instantiate the calculator before registries are ready):

1. Obtain each diagram's `peram`, `V`, `gamma_pos`, and `result_indx` from `operator_groups` and `Cpt`.
2. Merge the peram keys `(flavor, (t_quark,t_antiquark))` required by all diagrams; time labels must be the labels actually used by dynamic code, such as `tsrc`, `tsink`, and `tcur0`.
3. Merge V names and time ends (`VVV_0/VDV_0` are numbered independently at each time end), together with gamma/Projector names.
4. Use the right-hand side of `result_indx ->` to determine the external axes of `corr_*`, then determine `Vindex/Gindex` and momentum slices from them.
5. M4/M5 must provide only variables in this manifest; do not load V, peram, or gamma data unrelated to the current physical target.

In the template, express the manifest with the fixed three sets `required_peram={(p[2],tuple(p[4]))}`, `required_v={(v[1],v[3])}`, and `required_gamma={g[1]}`, iterating over the actual Wick entries in `dynamic_plan`. These sets are the variable contract for later modules; if loading code disagrees with the manifest, fix the physical operator or registration direction before writing file paths.

Keep a short comment in M3 with the fixed format `required peram: ...; required V: ...; required gamma: ...; output axes: ...`. This is a readable variable contract, not an extra manifest file.

## Unified timing

M0 sets `st_total = time.perf_counter()`. M4, M5, and M7 use `st_eigen`, `st_load`, and `st_cal`, respectively; M8 prints vertex preparation, perambulator loading, contraction time for each source time, and total runtime on the root process. 3pt insertion-time distribution may differ between processes, so do not insert collective-communication timing inside the local `tcur0` loop. Timing records existing computation only and must not change MPI collectives or physical data.

## Fixed call shape for the dynamic module

```python
# M7 dynamic loop
for ... in requested_times:
    # Register only the time and momentum slices required by the M3 manifest.
    dc = dynamic_contraction(
        operator_groups,
        peram_registry=PR, v_registry=VR, gamma_registry=GR,
        Cpt=Cpt, Vindex=Vindex, Gindex=Gindex,
        use_equivalence=False, ignore_dis=False,
        Projection=Projection, optimize=optimize)
    corr[..., ...] = dc.calculate_all()
```

Keep the same call-argument order. Only `operator_groups`, `Cpt`, projections, external indices, and result slices change with the physical target; do not add target-independent branches or a new scheduling layer inside the loop.

## Module delivery rules

When generating a script, output modules `M0` through `M8`; each module declares and fills only its own output variables. Keep the selected original asset as the structural baseline and, when needed, retain module headings and one line explaining a modification. Do not create test files, extra frameworks, or a second equivalent template.
