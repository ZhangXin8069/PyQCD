# MPI, I/O, Cached Contractions, and Performance

## MPI ownership and collective communication

Enter the MPI API only after a valid MPI distribution exists. Freeze process decomposition, physical time labels, local array indices, root/non-root return values, and global output axes. All processes must execute collectives in the same order; `get_mpi_tlist(..., gtype="TScatter")` returns physical time, owner process, and local index, which must not be confused. If local leading shapes differ, `TGather` cannot gather directly; use a validated padding or variable-length scheme. Compare reconstruction with serial results using deterministic arrays labeled by process.

## I/O contract

Record paths, overwrite policy, shape/axes, dtype, endianness, file length, and round-trip consistency. `write_data_ascii` uses `numpy.savetxt`; raw `readin_eigvecs` and `readin_peram` assume site layout and have limited metadata/file-length checks; `safe_save` may fall back to a site result directory, so confirm that external write is intentional.

## Cached contractions

Compare `cached_contract` with `numpy.einsum` or `opt_einsum.contract` using the same expression and shapes, covering repeated indices, broadcasting, scalars, explicit output, ellipses, and complex dtypes. The current ellipsis precheck right-aligns different batch axes differently from NumPy; cache keys also omit backend and dtype, so clear the cache and repeat the comparison when switching either.

## Performance model

`contractadviser` provides path, FLOP, memory, and slicing suggestions; its time is only a Roofline-model lower bound and excludes path search, startup, transposes, allocators, cache reuse, MPI, and I/O. Hardware presets are theoretical; report measured bandwidth and throughput. Split only free/batch axes and restore the original output positions; do not change physical volume, momentum set, operator basis, or source count for performance.
