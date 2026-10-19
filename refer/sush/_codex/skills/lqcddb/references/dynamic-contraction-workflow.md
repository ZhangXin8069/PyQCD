# Dynamic Contraction Workflow

Use this reference to create serial, CuPy, or MPI contraction programs; `dynamic_contraction` performs the actual tensor contraction. Use `../examples/nn_two_point_flat.py` for 2pt and `../examples/njn_three_point_flat.py` for 3pt and higher; begin a new program by copying the matching flat structure and then replacing the physical blocks.

## Input contract

Before starting, determine configurations, lattice and process grids, backend, source/sink operators, currents, momenta, `Cpt`, time separation, boundaries, the Nev cutoff, every Registry key, output axes, and save path. `physics.md` is authoritative for hadron valence-quark expressions, current conventions, and configuration tables; expand physical expressions into concrete tokens.

When no source operator is provided, call `conjugate_operator` on the declared physical initial state and preserve every coefficient; a transition process must not use the final sink as the initial state. Use a user-provided source operator unchanged after validation. First check flavor conservation, gamma order, Fourier sign, and temporal boundary.

## Dynamic first

First call the analysis core of `run_wick_analysis`, or inspect the plan that the `dynamic_contraction` constructor will use, to obtain peram flavor/time directions, VDV/VVV names and time ends, gamma/projection names, and `result_indx` output axes. Build `required_peram`, `required_v`, and `required_gamma` from these results; M4/M5/M6 prepare only these variables. Instantiate `dynamic_contraction` and call `calculate_all()` only after all registries are complete.

## Fixed top-level blocks

Keep this order: runtime and total timing; lattice/MPI; momentum phases; dynamic variable manifest; eigenvectors and VDV/VVV; peram reading and time distribution; registries; source/sink/insertion/momentum loops; `TGather`, `loop_tsrc`, and saving. Do not hide physical or ownership decisions in `main`, loader factories, or schedulers.

Record shape, logical axes, absolute time, owner process/local index, conjugation/transpose, backend, and dtype for every registration. Do not infer the local index from integer division of `get_mpi_tlist`; all processes must execute collectives in the same order. Use `ignore_dis=False` and `use_equivalence=False` by default unless the disconnectedness criterion and equivalent-diagram identities have been proved.

## Timing and delivery

M0 sets `st_total`; time vertices, peram, and each source-time contraction separately, and have M8 print stage and total runtimes on the root process. Synchronize the device before GPU timing; synchronize MPI timing only at stage boundaries reached by every rank. Baseline and optimized versions must preserve physical axes and numerical results; a Roofline estimate cannot replace a measurement. Clearly report GPU, MPI, and full-ensemble paths that were not run.
