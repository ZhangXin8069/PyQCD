# Current Source Status

This is a version record for the current `src/lqcddb` in `lqcddb_update`, not a general conclusion about all distillation programs. Reproduce key findings after the source changes.

## Verified behavior

- `conjugate_operator` conjugates numerical coefficients, reverses multi-hadron factors, and handles baryon diquark transpose signs; it stops on ambiguous syntax.
- Strict equivalent-diagram checks cover tensor identities, time direction, diagram topology, ordered output axes, component-wise gamma-transpose signs, and VVV permutation parity.
- `contractadviser` uses `opt_einsum` path information, distinguishes real and complex arithmetic, and re-estimates slice memory; the result remains a model lower bound rather than a measured time.
- `Jackknife` supports any configuration axis; Jackknife and Bootstrap use Hermitian complex covariance, and Bootstrap supports sampling with replacement and $M/N_{conf}$ scaling.
- `ratio_3pt` aligns insertion/sink axes and returns NaN with `valid_mask=False` in invalid domains; `meff` accepts complex input; `solve_gevp` preserves complex Hermitian matrices and requires a positive-definite reference matrix.
- Named top-level imports resolve to the current implementation; wildcard imports initialize the complete public API and optional MPI/SymPy paths.

## Scientific limitations

- Although `stout_smear_ndarray` validates a seven-dimensional layout, an identity gauge field enters a non-finite zero-generator branch when `nstep>0`; do not use it for production data until the matrix exponential and path-staple/direction mapping are corrected.
- `run_wick_analysis` and `dynamic_contraction` default to `ignore_dis=True`; equal endpoint times are only a heuristic criterion. Use `ignore_dis=False` when the physics requires connected diagrams.
- Equivalent diagrams normalize `u,d` perams to `light` without proving light-flavor equality configuration by configuration; enable this only when registered arrays satisfy an exact isospin assumption.
- Direct `wick_contraction` may discard unknown tokens; validate `sigma_mu_nu`, multi-gamma chains, and excessive fermion pairs explicitly before dynamic planning.
- The list returned by wildcard Wick cannot be passed directly to `run_wick_analysis`; resolve the flavor sectors first.
- `validate_plan` does not check rank, broadcasting, dtype, or output axes; `calculate_contraction` treats a total coefficient with absolute value below $10^{-12}$ as zero.
- `loop_tsrc` shifts only one non-source time axis; two-current or four-point functions must explicitly shift every insertion axis.

## Numerical and engineering limitations

Eigenvector row layout should be explicitly reshaped as $E=\text{eigvecs.reshape}(N_{ev},-1)$ and checked with $\max|EE^\dagger-I|$. Do not rely only on the current `vector_creator.check`. Check zero norms, linear dependence, compression random seeds, and finiteness. The current `create_omega_accelerate(exact=0)` rejects its default value, and `random_type="Z_N"` is not a general normalized-root-of-unity construction.

For different batch axes, `cached_contract` uses a precheck for ellipses that is not right-aligned like NumPy; cache keys also do not automatically distinguish backend or dtype. Raw file readers have limited checks for endianness, file length, and metadata; `safe_save` may fall back to a site directory, so confirm the intent of the external write.
