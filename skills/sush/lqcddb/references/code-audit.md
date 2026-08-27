# Source and README correctness audit

Use this process for code review, documentation comparison, release readiness, or API verification.

## Evidence order

1. Write the physical definition and required invariants.
2. Locate the current implementation, exports, type stub, README entry, and package metadata.
3. Compare signatures, shapes, axes, dtypes, defaults, return types, signs, conjugations, normalizations, and exceptions.
4. Build the smallest independent numerical counterexample or gold calculation.
5. Classify the result as a confirmed defect, documentation mismatch, scientific limitation, or unverified risk.

README agreement is not proof of correctness. A source test that repeats the same formula is not independent validation.

## Read-only procedure

- Inventory with `rg --files` and find definitions with `rg -n`.
- Parse all Python sources with `ast.parse` for a write-free syntax check.
- Run probes with `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src`, and temporary output under `/tmp`.
- Avoid wildcard imports and MPI entry points during serial inspection.
- Record the source-tree hash before and after when strict read-only verification matters.
- If pytest or MPI cannot run, report that explicitly and continue with safe targeted probes.

## Scientific checks

- Gamma matrices: Clifford algebra, hermiticity, gamma5, charge-conjugation identities, and sigma normalization.
- Wick contractions: flavor balance, fermion permutation sign, source-bar transformation, free-index order, and connected topology.
- Distillation tensors: perambulator source/sink orientation, spin/eigenvector axes, vertex conjugation, and momentum phase.
- Statistics: estimator definition, resample count, normalization, complex covariance convention, and valid time range.
- GEVP: Hermitian positive-definite reference matrix, complex information preservation, normalization, and state tracking.
- Gauge operations: direction-to-axis map, forward/backward transport, boundary convention, unitarity, and determinant.

## Severity

- Critical: silently changes a correlator, sign, topology, state, or gauge field.
- High: common documented input fails, invalid data passes validation, or optional functionality breaks basic import/use.
- Medium: restricted shape/path fails clearly, statistics are nonstandard without warning, or public documentation/stubs are materially stale.
- Low: diagnostics, naming, packaging hygiene, or error quality without changed scientific output.

Give exact file and line references. Do not patch package code during an audit.
