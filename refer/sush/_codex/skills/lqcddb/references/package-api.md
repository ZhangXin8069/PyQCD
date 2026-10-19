# Current Package and API Cross-Reference

Use this reference to choose import paths, call public functions, and modify signatures. Treat `src/lqcddb` as the current version; installed copies, `build/lib`, README examples, and old stubs may lag behind.

## Package identity and loading

The distribution name is `lqcddb_update`, the import name is `lqcddb`, and the version comes from `pyproject.toml`. NumPy, SciPy, `opt_einsum`, and Matplotlib are base dependencies; CuPy, `mpi4py`, and SymPy are optional. The top-level `__getattr__` lazily loads concrete modules by name; set the backend before constructing `vector_creator`. Do not use wildcard imports because they initialize the complete public API and optional MPI/SymPy paths.

## Top-level public names

Basics and tensors: `get_backend`, `set_backend`, `levi_civita_tensor`, `creat_mom_list`, `ArraySlicer`, `cached_contract`, `clear_cache`, `get_cache_keys`.

SU(2), MPI, and gauge fields: `SU2combine`, `SU2decompose`, `get_mpi_data`, `get_mpi_tlist`, `mpinit`, `getMPIComm`, `getMPIRank`, `getMPISize`, `stout_smear_ndarray`.

Constants and matrices: `Nc`, `Ns`, `Nd`, `fm2GeV`, `gamma`, `tran_indx_to_gamma`, `PFF_Mom_to_gamma_new`, `sigma`, `Mom_times_sigma`, `Mom_cross_sigma`.

Wick and contractions: `conjugate_operator`, `wick_contraction`, `plot_figure_wick`, `identify_equivalent_diagrams`, `seq_peram`, `analyze_bandwidth`, `printGPUinfo`, `PeramRegistry`, `VRegistry`, `GammaRegistry`, `dynamic_contraction`, `run_wick_analysis`, `clear_plan_cache`, `calculate_contraction`.

Analysis: `Mom2GeV`, `loop_tsrc`, `sum_over_array_of_list`, `mean_over_array_of_list`, `Jackknife`, `Bootstrap`, `meff`, `ratio_3pt`, `solve_gevp`, `dis_connect`, `plot_analyse_marker`, `plot_analyse_color`. Eigenvectors/vertices: `vector_creator`, `vertex_creator`. I/O: `write_data_ascii`, `check_dir_path`, `safe_save`, `check_files_existence`.

Module-level entry points that are not exported at the top level include `contractadviser`'s `parse_subscript`, `estimate_cost`, `quick_check`, and cost data classes; grid/device queries in `mpi_init`; `readin_eigvecs`, `readin_peram`, `check_write`, and `savetxt` in `write_date`; and internal helpers for gamma and baroperator.
