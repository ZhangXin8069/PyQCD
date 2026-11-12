# Appendix: comparison-test workflow and performance notes

> **Scope.** This appendix is source-checkout material. The PyPI package does
> not ship the MatX/Einsums bridges, the benchmark driver or the raw logs
> discussed here.

## Adding a new comparison engine

The MatX and Einsums integrations follow one workflow; reuse it for the next
engine.

1. Pin the upstream source as a Git submodule under `refer/git-rep/<name>` and
   record its source, purpose and fixed commit in `refer/git-rep/README.md`.
2. Add a C ABI bridge under `cpp/<device>/<name>/`: one header with status
   codes and flat `int64`/`int32` slot arrays, one implementation, a build
   script with an export version script, and a directory `AGENTS.md`.
3. Make the build fail closed: a missing upstream produces a stub archive that
   returns `UNAVAILABLE(4)` with an actionable reason; never a fake success.
4. Add a Python ctypes binding under `pyqcd/testing/<name>/` and derive the
   contract corpus from the official documentation. Mark every unverified
   statement explicitly.
5. Register the verification entry in `pyqcd/testing/__init__.py` and
   `pyqcd/testing/conftest.py`.
6. Add the engine to `pyqcd/testing/einsum/bench_four_engines.py` with a
   support matrix, a precision label and explicit `N/A` reasons.
7. Keep test code out of distributions: exclude the package from
   `find_packages` and prune its sources in `MANIFEST.in`.
8. Accept only after: stub build; real build when dependencies exist;
   correctness against an independent oracle; performance with equal threads
   and warm-up; `form-audit`; and a clean worktree.

## Four-engine performance summary

Measured on an i7-13620H host (8 threads) and an RTX 4060 Laptop (8 GB).
Median of 3 alternating rounds; correctness checked before timing; bridges use
pre-allocated outputs; CPU threads pinned to `OMP=8`, `BLAS=8`. `opt_einsum`
runs on NumPy/CuPy, so its numbers describe planner plus backend.

### CPU (NumPy arrays, milliseconds)

> **Historic data.** The two rows marked `(kernel)` — *nucleon 2pt* and
> *gamma sandwich* — refer to the dedicated C++/CUDA kernels that were
> **removed** when the library moved to the generic + topology-specialized
> design. They are kept as a record of the retired implementation and are
> **not reproducible** on the current tree.

| Case | qcd | opt | einsums |
|---|---:|---:|---:|
| matmul 2D (real) | 718.7 | 713.0 | 944.8 |
| matmul 2D (complex) | 386.1 | 339.4 | 453.5 |
| element-wise | 309.0 | 351.7 | 1547.6 |
| outer product | 63.8 | 102.4 | 294.8 |
| full reduction | 58.0 | 63.3 | N/A |
| partial reduction | 114.7 | 121.2 | N/A |
| chain 3 ops | 597.5 | 589.7 | N/A |
| chain 4 ops | 282.1 | 275.5 | N/A |
| hyper-edge | 91.2 | 99.2 | N/A |
| batched GEMM | 374.3 | 3802.7 | 401.0 |
| ellipsis | 658.1 | 6510.7 | N/A |
| batch × matrix | 49.1 | 49.5 | N/A |
| many small batches | 105.9 | 221.1 | 179.1 |
| nucleon 2pt (kernel) | 2487.4 | 43142.1 | N/A |
| gamma sandwich (kernel) | 245.8 | 5681.2 | N/A |

### GPU (CuPy arrays, milliseconds)

| Case | qcd | opt | matx |
|---|---:|---:|---:|
| matmul 2D (real) | 654.3 | 649.8 | 649.9 |
| matmul 2D (complex) | 326.8 | 325.0 | 327.0 |
| element-wise | 15.4 | 15.5 | 195.5 |
| outer product | 2.6 | 2.7 | 31.8 |
| full reduction | 9.5 | 20.1 | 282.2 |
| partial reduction | 8.8 | 8.9 | 17.9 |
| chain 3 ops | 575.0 | 584.7 | N/A |
| chain 4 ops | 252.4 | 251.4 | N/A |
| hyper-edge | 98.3 | 102.0 | N/A |
| batched GEMM | 169.7 | 176.8 | 163.4 |
| ellipsis | 334.0 | 324.3 | 324.2 |
| batch × matrix | 88.6 | 93.7 | N/A |
| many small batches | 25.0 | 25.1 | 23.7 |
| nucleon 2pt (kernel) | 875.7 | 2340.8 | N/A |
| gamma sandwich (kernel) | 796.4 | 1096.1 | N/A |

`matx` rows use 32-bit compute. `N/A` means the bridge cannot express the case:
MatX supports one or two operands with rank at most 4; Einsums supports exactly
two operands of equal rank, rank 1–4, output rank at least 1, and no ellipsis.

### Where the time goes (Einsums)

| Case | Engine (ms) | Allocation (ms) | Compute share |
|---|---:|---:|---:|
| GEMM 4096 | 578.9 | 134.2 | 81.2% |
| element-wise 12000 | 75.4 | 1103.7 | 6.4% |
| outer 8000 | 70.0 | 173.3 | 28.8% |
| batched 16×1024 | 173.5 | 136.4 | 56.0% |

For element-wise, only 6.4% of the bridge's 1.55 s is computation; the rest is
per-call tensor allocation and copies. The Einsums kernel itself takes about
75 ms, which is faster than both qcd and opt_einsum on the same case.

## Limitations

* `opt_einsum` is a planner, not a kernel: on CPU it measures NumPy/BLAS, on
  GPU it measures CuPy/cuBLAS.
* MatX `einsum` fixes the cuTensorNet compute type at 32F; MatX `matmul` is
  true FP64 (`rel = 1.9e-16` on a 256² case).
* The numbers cover one machine, one size per family, and three rounds of
  best-of-inner samples. Run-to-run variation on selected cases is 10–26%.
* Bridges and engines have different cost boundaries: qcd/opt allocate their
  output inside the call; the bridges take a pre-allocated output, and the
  Einsums bridge additionally allocates its input tensors.
* Multi-GPU/MPI, multi-stream, non-contiguous inputs, mixed precision,
  complex64/float32 and production-scale lattices are not covered.
* Differences below roughly 15% on CPU and 25% on GPU are within observed
  variation and should not be ranked.

## Outlook: comparing on more computing environments

* Repeat on a server CPU (NUMA, more cores) and a data-centre GPU (HBM).
* Add MPI/multi-stream and distributed batch sharding.
* Align precision: a MatX 64F path, or a run where every engine uses 32F.
* Sweep sizes to locate the crossover where cuTensorNet overhead disappears.
* Measure cold start, peak memory, tail latency and energy, not only median
  time.
* Sweep thread counts per engine and report both fixed and best-known settings.
* Re-run on a second machine before generalising any small ratio.

The full report, raw JSON and component probe live in the source repository:
`docs/report_matx_einsums_bridge_20261110.md`,
`logs/einsum-four-engine-20261110.txt` and
`cpp/cpu/cpu-einsums/tests/bench_components.cpp`.
