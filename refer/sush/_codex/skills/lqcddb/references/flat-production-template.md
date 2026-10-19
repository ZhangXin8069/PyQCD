# Flat Production Template and Block Contract

When creating or refactoring a contraction program, use `examples/nn_two_point_flat.py` (2pt) or `examples/njn_three_point_flat.py` (3pt and higher) as the visible baseline. The program is a top-to-bottom Python file; a module is a stable top-level block, not a `main -> run -> loader` call tree or a collection of callbacks.

## Fixed blocks

1. Imports: use named imports from the current lqcddb API, NumPy, and timing/path modules.
2. Runtime parameters and paths: `Cpt`, lattice/process grids, backend, configuration, Nev cutoff, source/sink times, boundary, output axes, and source-averaging convention.
3. Backend/MPI initialization: set the backend first, construct `vertex_creator`, then call `mpinit` and record rank, size, grid, and device.
4. Momentum and phases: use `(p_z,p_y,p_x)`, route source and sink momenta separately, and call the current `phase_exp_VdV`/`phase_exp_VVV`.
5. Local VDV/VVV: use `get_mpi_tlist` to obtain owner times, load eigenvectors, construct vertices, and apply $\Omega$ exactly once.
6. Physical operators and registries: expand physical expressions into coefficient-bearing tokens, use `conjugate_operator` for a source operator when needed, and register gamma, projections, V, and peram.
7. Peram acquisition/distribution: the root reads data and uses `TScatter` along the time axis, making global time and local index explicit.
8. Dynamic contraction loop: loop over sources, owned sink/current times, momenta, and operators, update exact registry keys, create `dynamic_contraction`, and call `calculate_all`.
9. Gathering and estimators: after `TGather`, check global axes, use `loop_tsrc` and the declared wraparound factor, and have the root save results and record stage/total times.

Do not add `main`, loaders, registry factories, or schedulers that hide physics and ownership. Use helper functions only with user agreement or when inlining is mathematically unsafe, and explain the reason.
