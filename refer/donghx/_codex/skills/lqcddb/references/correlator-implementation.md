# Perambulators, vertices, and correlator implementation

## Tensor contract

Before writing an einsum, make a table for every tensor containing:

- physical meaning;
- axis order and extent;
- source/sink time and direction;
- spin and distillation indices;
- flavor;
- conjugation/transposition state;
- momentum and displacement labels.

Translate the reviewed Wick term into that table, then into the einsum. Check that every repeated index is contracted exactly twice unless a trace/diagonal is intentional, and that output indices match the requested correlator layout.

## Perambulators and sequential objects

- Verify the stored perambulator convention from its actual reader or producer; comments and older scripts are not sufficient.
- Treat gamma5 hermiticity as an index-aware matrix identity. A `.conj()` can implement an adjoint only when the output index order performs the required transpose.
- Validate sequential-perambulator expressions against an explicit small combined spin-distillation matrix multiplication.
- Keep flavors distinct even when numerical arrays happen to be equal.

## Vertices and correlators

- Confirm whether a vertex is `V^dagger O V`, its conjugate, or a displaced/link version.
- Derive boundary signs from the number of fermion lines crossing the temporal boundary. Do not apply an antiperiodic baryon sign rule blindly to mesons.
- For three-point ratios, align the three-point insertion axis and the two-point sink axis explicitly before arithmetic. The current one-dimensional `ratio_3pt` path can broadcast incorrectly when those axes occupy different positions.
- For conserved/current insertions, document link direction, forward/backward orientation, midpoint phase, and normalization.

## Dynamic engine

The current dynamic module imports MPI helpers at module load, even though MPI is an optional package dependency. Use it only in a validated MPI environment and avoid wildcard package imports. Keep equivalence disabled unless the stricter proof in the Wick process has passed.

Verify any generated plan against direct `opt_einsum.contract` on small random tensors and, where possible, a hand contraction.
