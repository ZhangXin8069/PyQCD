# Source, API, and Modification Review

Use this reference for correctness reviews, documentation checks, releases, and explicit software changes.

## Evidence order

1. Write down the physical definition, conventions, input domain, and invariants.
2. Locate the current source, runtime exports, type stubs, README, physics documentation, and related records.
3. Compare signatures, axes, shapes, dtypes, defaults, return values, signs, conjugation, normalization, exceptions, and optional dependencies.
4. Construct a minimal independent numerical example or hand-calculable reference result.
5. Classify conclusions as confirmed defects, documentation/stub inconsistencies, scientific limitations, unsupported inputs, environment failures, or unverified risks.

Tests that merely reproduce README equations or duplicate the implementation are not independent validation.

## Keep the source tree clean

Use `rg --files` and `rg -n` to inventory files and definitions; inspect `src/lqcddb` first and do not treat `build/lib` as the current implementation. Use `ast.parse` for syntax checks that do not write files. For serial probes, set `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=src`, and put caches and outputs in `/tmp`. Use named imports during serial checks so wildcard imports do not trigger optional MPI.

If `pytest`, MPI, CUDA, production data, or scheduler resources are unavailable, report the limitation and continue with targeted serial checks. An explicit software change must preserve unrelated changes and modify the smallest consistent set among implementation, exports, stubs, documentation, and regression files; do not change physical conventions merely to pass a test.

## Scientific checks

Check the Clifford algebra, Hermiticity, `gamma_5`, charge conjugation, transpose signs, and sigma normalization of gamma matrices. At the Wick layer, check coefficient conjugation, reversal of hadron factors, flavor conservation, fermion-permutation signs, syntax, connected components, and free output axes. For distillation tensors, check peram time directions, spin/eigenvector axes, gamma5 Hermiticity, vertex conjugation, VVV antisymmetry, link directions, and Fourier phases. For dynamic plans, check registries, diagram counts, coefficients, output layout, and a direct einsum comparison. For statistical analysis, check estimators, sample axes, error normalization, complex covariance, seeds, and valid time domains. For GEVP, check complex Hermitian inputs, a positive-definite reference matrix, normalization, residuals, and state tracking. For gauge fields, check direction axes, forward/backward path staples, invariants, finiteness, unitarity, and determinant. For MPI/I/O, check ownership, collective order, global-axis reconstruction, round trips, endianness, fallback writes, and model/measurement differences.
