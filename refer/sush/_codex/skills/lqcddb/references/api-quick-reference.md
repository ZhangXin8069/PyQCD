# lqcddb Current API Quick Reference

Use `src/lqcddb/__init__.py` and `__init__.pyi` as the version authority. The entries below are the most common entry points; other private functions are not user APIs.

## Basic tools and constants

```python
set_backend('numpy' | 'cupy')
backend = get_backend()
levi_civita_tensor(n=3)
creat_mom_list(Mom=[0, 0, 0], fix_Q2=False, only_g0=False)
ArraySlicer(array).slice(dims, indices)
ArraySlicer(array).assign(dims, indices, values, keep_dims=[])
cached_contract('ab,bc->ac', A, B, optimize='auto')
clear_cache(); get_cache_keys()
gamma(index); sigma(index); Mom_times_sigma(Mom, upto4dim=False)
PFF_Mom_to_gamma_new(Mom_list, allow_t=False)
```

The triplet convention of `creat_mom_list` comes from the current implementation; do not change `[pz,py,px]` to another order. `cached_contract` caches paths only and does not change einsum-axis semantics.

The full public name set also includes `SU2combine`, `SU2decompose`, `levi_civita_tensor`, `stout_smear_ndarray`, `analyze_bandwidth`, and `printGPUinfo`. The first two handle SU(2) Clebsch–Gordan combination/decomposition; `stout_smear_ndarray` requires link layout `(Nd,Nt,Nz,Ny,Nx,Nc,Nc)`; bandwidth functions provide model estimates only and cannot replace measurements.

## Operators, Wick, and dynamic contractions

```python
src = conjugate_operator(sink)
wick = wick_contraction(sink, src, curr_operators=[], Cpt='2pt')
plan = run_wick_analysis(
    [(sink, src)], Cpt='2pt', ignore_dis=False,
    use_equivalence=False, verbose=True)
out = calculate_contraction(
    plan[0], peram_registry=pr, v_registry=vr,
    gamma_registry=gr)
```

`wick_contraction` encloses each hadron in `|`; quarks are `u,d,s,c,b,t`, antiquarks include `u^d`, gamma matrices are string tokens, and numeric tokens are global coefficients. `curr_operators` is ignored for 2pt; with wildcards, the result is `list[dict]`. Key fields are `result_indx`, `result_name`, `result_sign`, `peram`, `V`, and `gamma_pos`.

Minimal registry usage:

```python
pr = PeramRegistry()
pr.register('light', ('tsink', 'tsrc'), tau)
vr = VRegistry()
vr.register('VDV_0', 'tsink', vdv)
gr = GammaRegistry()
gr.register('gamma_5', gamma(5))
```

`PeramRegistry.register` requires an array with at least four dimensions and time labels `(t_quark, t_antiquark)`; `light` can match `u/d`. The name passed to `VRegistry.register(name, time_label, data)` comes directly from Wick's `VVV_0/VDV_0` (numbered independently at each time end). `GammaRegistry.register(name, data)` registers any gamma combination; `transpose_sign` returns a sign only after a component-wise proof that `Gamma.T=±Gamma`.

`dynamic_contraction(...)` is a convenience class combining the plan, registries, and `calculate(i)`/`calculate_all()` above; `validate_plan` returns missing registrations. `Projection` uses `Projector=[proj_sink, proj_src]`; `Oindex` explicitly restricts uppercase output indices. Validate axes with a direct NumPy contraction before enabling equivalence diagrams or performance options.

Wick-diagram helpers are `identify_equivalent_diagrams(*dicts, gamma_transpose_signs=...)` and `plot_figure_wick(result_dict, diagram_index=0, Cpt=..., plot_text=True)`; the former merges diagrams only when network, time, topology, and port identities all match, and the latter returns `(Figure, Axes)`. `seq_peram(peram, ...)` performs a gamma5 temporal transformation.

## Eigenvectors, vertices, MPI, and I/O

`vector_creator` handles eigenvector inner products, orthogonalization, random noise, and compression; `vertex_creator` handles phases, VDV/VVV, gauge links, and weights. Use the source signatures of the corresponding classes for exact methods, and check input layout before calling them.

```python
mpinit(grid_size, latt_size=..., backend='numpy')
get_mpi_tlist(Nt, t, gtype='find')
get_mpi_data(data, mdtype='Scatter' | 'TScatter' | 'Gather' | 'TGather' | 'Bcast', ...)
getMPIRank(); getMPISize(); getMPIComm()
```

`get_mpi_tlist` returns owner/local indices for the actual communicator; do not infer them by integer division. File I/O entry points include `readin_eigvecs`, `readin_peram`, `safe_save`, `check_dir_path`, and `check_files_existence`.

MPI queries also provide `getRankFromCoord`, `getCoordFromRank`, `getDefaultGrid`, `getGridSize`, `getGridCoord`, `getGridMap`, `setGridMap`, `getCUDADevice`, `isCUDABackend`, and `isNumpyBackend`; rely on these states only after `mpinit` has completed.

## Analysis entry points

```python
loop_tsrc(data, indx=[t_src_axis, t_sink_axis],
          Boundary_Conditions='Periodic', Ctype='2pt', t_sep=0)
Jackknife(data, Nconf_axes=0, only_sample=False, cov_axes=None)
Bootstrap(data, Nconf_axes=0, only_sample=False, cov_axes=None,
          M=0, N=0, seed=None)
meff(data_sample, alttc, Nconf_axes=0, Nt_axes=1,
     meff_type='log' | 'cosh' | 'GEVP', Atype='JK' | 'BS')
ratio_3pt(data_3pt_sample, data_2ptI_sample,
          data_2ptF_sample=None, t_sep=..., ...)
solve_gevp(C, t0, track_states=True, hermitize=True, ...)
Mom2GeV(Nx, alttc, Mom, M0)
```

`Jackknife`'s `data_sample` omits one configuration and has error `sqrt(Nconf-1)*std`; `Bootstrap` samples with replacement. `meff` returns NaN and `valid_mask` for invalid points. `loop_tsrc` boundary conditions apply to the complete correlator, and 3pt must specify `Ctype` and `t_sep`. See `$analyse-hardon` for dedicated analysis, plotting, and `lsqfit` workflows.
