# Performance

> **Strongly recommended before real use: run `python -m pyqcd.einsum.test --tune`.**
> **Every performance claim must come from your own measurement — this page does not
> make things up.** Concrete speed-up ratios and benchmark tables have been removed:
> the older numbers were produced under a retired topology step-table convention, so
> they are not comparable with the current `ROLE_SHAPES` one.

## Why tune instead of trusting a page

On one backend, the fastest configuration of a given equation depends on four things:
**shape, dtype, device, and which kernels are actually available**. This library does
not guess: `--tune` really runs the sweep on *your* shape and writes the winning
configuration into `pyqcd.einsum.profile.csv`. Ordinary calls then only read that
table and execute whatever the record says.

```bash
python -m pyqcd.einsum.test --equation "ab,bc,cd->ad" \
  --shape "[100,100],[100,100],[100,100]->[100,100]" --tune
```

`--tune` takes a value; it is not an on/off switch:

| Form | Behaviour |
|---|---|
| `--tune` (no value) / `--tune true` | **Full mode**: probe whether the equation can be specialised, or specialised after an equivalent rewrite, then run the **ordered** sweep |
| `--tune pass` | **Fast pass**: take the `[if_equivalence=false, if_specialized=false]` line only, disable `--compare` and everything unrelated to the sweep, so it stays fast and never blocks |
| omitting `--tune` | no sweep; use the profile table or the built-in default |

The sweep is **not** a Cartesian product: it walks a branch-dependent **ordered** list
(29 groups for each specialisable branch, 28 for the plain branch). The first group is
`original` (`opt_einsum`, the only route that honours all seven `optimize` values), so the
bound is pinned to "how fast this equation can be" from the start. A group is cut short as
soon as it exceeds **that best time itself** (`min_wall_ms`, 1x); the `pruned` reason
reports both `wall_ms` and `min_wall_ms`. **Every group prints a `tune comb :` line the
moment it finishes** (index/total, the three axes, wall_ms, peak, status, reason verbatim),
so progress is visible before the sweep ends. The reported
`wall_ms` is the wall clock of the **whole** `contract()` call, so the number you read is
the number your program pays. Each group also reports its **peak memory / VRAM** in MB,
measured per group and reset before the next one so a group never inherits its
predecessor's footprint.

### The options that decide what is measured

| Option | Accepted values | Notes |
|---|---|---|
| `--backend` | `auto` / `numpy` / `cupy` / `torch` | array library only; forcing a native kernel is `--kernel-force`, not a backend |
| `--compare` | `opt_einsum` / `backend` / `none` | default **`none`**; the "numpy reference" is spelled `--backend numpy --compare backend` |
| `--optimize` | `greedy` / `optimal` / `dp` / `branch_1` / `branch_2` / `branch_all` / `none` | `auto` is the built-in default selector, **not** a writable value |
| `--warmup` | integer | default **1**; `0` remains allowed when cold-start cost is deliberately included. A positive value on an accelerated device is raised to 2 |
| `--rounds` | integer | default **3**; the fastest observed sample is the reported time, so scheduler jitter is less likely to hide the best run |
| `--kernel-force` | `generic` / `dedicated` / `topology` / `original` / `backend` | a single value, default `generic`; `original` hands the contraction to `opt_einsum.contract` (it honours `--optimize`, translating `branch_1`/`branch_2`/`branch_all`/`none` to `branch-1`/`branch-2`/`branch-all`/`False`); comma lists and `auto` are rejected |
| `--shape` | `[..],[..]->[..]` | omitted: sizes are inferred from the corpus roles of the equation |
| `--explain` | `auto` / `true` / `false` | with `--shape` omitted, a hit reached through an equivalent spelling prints `shape source: corpus(equivalence) via <equation>` |

`--shape` is not limited to a literal corpus match: an equation may reach a corpus entry
through an equivalent rewrite (operand reordering, output-axis permutation). In that case
`--explain true` names the corpus equation whose roles supplied the sizes, so the axis
order never looks unexplained.

### What the clock actually measures (since the synchronised-timing fix)

Two rules make the reported milliseconds comparable with a script of your own:

- **The device is synchronised around every sample.** `cupy` and CUDA `torch` dispatch
  asynchronously, so `perf_counter()` around `contract()` alone would stop at
  *enqueue* time. Each measured sample is bracketed by a device sync (a no-op on CPU),
  so the number is compute time on GPU too. Earlier releases did not do this, which made GPU
  timings look orders of magnitude too small — those historical numbers are void, see
  `docs/ERRATA_gpu_timing_20261119.md`（仓库内，不在本站）.
- **A positive `--warmup` is raised to at least 2 on an accelerated device.** The first
  call there also builds the CUDA context, cuBLAS handles and the workspace; discarding
  only one call can still leave residual startup work in the fastest sample. `0` stays
  `0` when explicitly requested. The report prints the
  **effective** value in `time :` and adds a `time note:` line when it was raised;
  `--json` carries `warmup_effective` / `warmup_raised`.

`--compare opt_einsum` runs the reference on the **same backend as the operands**
(`backend=` is passed explicitly and echoed in the reference line, e.g.
`opt_einsum.contract(backend='cupy', optimize='greedy')`). If `opt_einsum` has no
registered backend for those operands it reports that instead of silently measuring on
numpy — a cross-device "reference" is not a reference.

## Reading the output

The initialisation phase narrates itself before any measurement. Its fields are stable
machine-readable tokens, so this part of the log is the same in every language (the
per-step `<n>ms` timings are elided here; they vary run to run):

```text
tune init : mode=all  equation=ab,bc,cd->ad  operands=3  dtype=complex128  device=cpu  backend=auto
tune init : shapes ok  total_elements=192  est_bytes=0MiB
tune init : branch probe done  raw_ok=false  equiv_ok=false  branch=B3
tune init : kernel admission done  unavailable=topology
tune axes : kernel_force=original,dedicated,generic,topology,backend  as_equivalence=false,true  optimize=greedy,branch_1,branch_2,dp,branch_all,optimal,none
tune branch: branch=B3  raw_ok=false  equiv_ok=false  target_equation=-
tune plan : planned=28  kept=28  trimmed=0  skipped-kernels=-
```

* `tune axes` — the three swept axes. `optimize` only ever holds the seven leaves
  `greedy` / `optimal` / `dp` / `branch_1` / `branch_2` / `branch_all` / `none`;
  `kernel_force` only ever holds `original` / `dedicated` / `generic` / `topology` /
  `backend`.
* `tune branch` — whether the equation itself, or an equivalent spelling of it, is
  specialisable. That is what decides whether a `topology` group exists at all.
* `tune plan` — how many groups were planned and how many survive kernel admission; a
  kernel unavailable for this shape is removed here and reported once, so a missing
  group is never mistaken for a silent skip.

Every measured group then gets one `tune comb` line (its three axis values, `wall_ms`
and a status token), followed by `tune best` with the winner and `tune prune` with the
live upper bound that stopped the slow groups. The status vocabulary is machine
readable and **never translated**: `ok`, `pruned` (slower than the current bound and
cut), `timeout`, `error`, `mismatch` (outside the tolerance) and `skip` (that kernel
fundamentally has no route for this shape).

## The exact limit of `--tune-timeout`

The default is **15 seconds per `contract()` call**. Because one group performs
`1 + warmup + rounds` calls, its group budget is `15 x 5 = 75 seconds` with the default
`warmup=1` and `rounds=3`.

By default each group runs in a **fresh interpreter**. The parent process can kill the
child, so the group budget is a **hard** bound that also interrupts native C/CUDA code.
Setting `PYQCD_EINSUM_TUNE_INPROCESS=1` switches to the in-process route, where
`SIGALRM` is only a **best-effort** bound: it can interrupt long Python loops, but not a
native call that never returns to the interpreter. On GPU, either keep the default
subprocess isolation, choose a generous limit, or use `--tune pass` first.

## Correctness before speed

Speed only means something on correct numbers. Every example in this documentation ends
with an `assert`; the build executes all of them and fails if one stops verifying itself.

### Verify it yourself

Three independent implementations, same inputs, must agree element-wise:

```python
import numpy as np
import torch
import pyqcd.einsum as pe

rng = np.random.default_rng(0)


def complex_noise(*shape):
    """Complex random array of the given shape (real and imaginary parts sampled separately)."""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return (real_part + 1j * imag_part).astype(np.complex128)


def check(equation, *operands):
    """Verify one result against two independent reference implementations.

    - numpy plans its own path with optimize="optimal";
    - torch evaluates the same operands with its native implementation;
    the three must agree element-wise (rtol 1e-10) for the case to pass.
    """
    ours = np.asarray(pe.contract(equation, *operands))
    numpy_ref = np.einsum(equation, *operands, optimize="optimal")
    torch_ref = torch.einsum(equation, *[torch.as_tensor(o) for o in operands]).numpy()
    assert np.allclose(ours, numpy_ref, rtol=1e-10, atol=1e-12), "differs from numpy"
    assert np.allclose(ours, torch_ref, rtol=1e-10, atol=1e-12), "differs from torch"
    print(f"{equation:32s} ok  max|d| = {np.max(np.abs(ours - numpy_ref)):.2e}")
```

### Checks that run on every change

- `python -m pyqcd.testing` — the full contract suite (values, dispatch invariants,
  profile-table semantics, CLI behaviour); source checkout only;
- `python -m pyqcd.einsum.verify_specialized "<equation>"` — a pure query: it reports
  which shape family the equation hits, its corpus provenance and the declared kernels,
  **without building operands or calling `contract()`**;
- `python -m pyqcd.einsum.test --tune ...` — calibrate once on the real shape.

## On "how much faster"

This page gives **no** speed-up ratios. Not because there is no data, but because the
data has to share its conventions with *your* shape:

- the contraction order is shape sensitive: the optimum for mixed axes like
  `{4,5,40,50}` differs from the optimum on an equal-length probe;
- kernel availability depends on the device and the libraries (CUDA step tables, KUPL
  step tables and BLAS complex support all differ);
- so a speed-up quoted for one accelerator usually does not survive the trip to your
  machine.

Want numbers? Run one tune — that command prints the fastest observed wall time for
your machine and your shape.
