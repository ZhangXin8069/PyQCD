# MPI, I/O, contraction caching, and performance

## MPI

- Import MPI functionality only inside a working MPI allocation. Avoid `from lqcddb import *`.
- Validate rank decomposition, local time lists, root/non-root return semantics, and global axis order.
- Test one rank and multiple ranks separately: current gather helpers do not necessarily return the same shape at size one as at size greater than one.
- For `TGather`, require equal local leading sizes or use a variable-count gather design.
- Compare gathered output with a serial reference using rank-tagged deterministic arrays.

## I/O

- Check path creation, overwrite behavior, shape metadata, dtype, endian convention, and round-trip equality.
- Use an explicit directory in output paths. The current `write_data_ascii` attempts to create an empty directory for a basename-only path.
- The bundled `savetxt` depends on legacy `numpy.compat`; verify compatibility with the installed NumPy version.
- Ensure every opened file is closed and do not accept a site-specific fallback path as a portable recovery strategy.

## Cached contraction

Use `cached_contract` only after comparing its output with `numpy.einsum` or `opt_einsum.contract` for the exact subscript and shapes. Include tests for:

- repeated indices;
- singleton broadcasting;
- ellipsis broadcasting with unequal numbers of batch dimensions;
- scalar and explicit outputs;
- complex dtypes.

The current ellipsis shape validator aligns batch dimensions differently from NumPy in some valid cases.

## Performance adviser

Separate tensor-operation correctness from a Roofline estimate. Recompute FLOPs, transferred bytes, and output size independently for ellipsis expressions. Validate that efficiency factors have the documented units and physical range.

The current presets contain compute-efficiency values greater than one and ellipsis accounting can omit batch dimensions. Treat reported bandwidth and time as heuristic, not hardware-validated predictions.
