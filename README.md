# PyQCD

**English** | [中文](https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD/-/blob/main/README.zh-CN.md)

PyQCD is a lattice-QCD toolkit in Python with optional native acceleration. It
provides building blocks for propagator and operator contractions, gradient flow,
renormalisation and gluon TMD-PDF extraction, plus a self-contained contraction
engine with an **`opt_einsum`-compatible API**.

## Install

```bash
pip install -U pyqcd
```

**Read what the installer prints — every line.** Native kernels (KUPL on CPU,
CUDA on GPU) are compiled at install time, so the output tells you whether they
were built, which toolchain or driver was missing, and which optional dependencies
were skipped. Ignoring those lines is how you end up with a pure-Python package
while believing you have native acceleration. Requires Python 3.9+.

Check what you actually got (no imports needed):

```bash
python -m pyqcd           # version, native backend, optional deps, subpackages
python -m pyqcd --json    # machine-readable
```

Build steps, CUDA architectures, thread and environment knobs are maintained in
`pyqcd/einsum/AGENTS.md`; this README does not repeat them.

## Quick start

```python
import numpy as np, pyqcd.einsum as pe
a, b = np.ones((2, 3)), np.ones((3, 4))
c = pe.contract("ij,jk->ik", a, b)      # same as a @ b
```

The public surface mirrors `opt_einsum`: `contract`, `contract_path`,
`contract_expression`, `shared_intermediates`, `PathInfo`, `paths`, `backends`,
`helpers`, `parser`, `blas`.

## Performance: measure it, then tune

> **Strongly recommended before production use:**
> `python -m pyqcd.einsum.test --tune --equation "<your equation>" --shape "<your operand shapes>"`
>
> **Every performance figure must come from your own measurement. This README
> quotes no speed-ups, no ratios and no benchmark tables — any number copied from
> elsewhere is about someone else's machine, shapes and load.**

`--tune` is the **only writer** of the configuration table. On a fixed equation /
shape / dtype / device / backend it walks an *ordered* list of
`kernel_force × as_equivalence × optimize` combinations (22 groups for the two
specialisable branches, 21 for the generic one), prunes any group slower than the
current `min_wall_ms` bound, and writes the single winning row. Ordinary calls
only read that table; with no table they warn once, print the tune command for the
key, and continue unchanged.

For any single equation, `python -m pyqcd.einsum.test` is the only sanctioned
entry point — one-off scripts are not accepted as evidence:

```bash
python -m pyqcd.einsum.test --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]"
# PASS ij,jk->ik  <measured>ms  rss+<N>MiB

python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true
```

It really contracts, checks the tolerance and reports the route that ran. Exit
codes: `0` pass, `1` numeric or shape mismatch, `2` bad arguments. `--json <path>`
writes the full record, `--verbose true` prints the per-field detail including the
`route` / `kernel` evidence lines. Omit `--shape` and `--scene` infers shapes from
the corpus roles — those are path/correctness evidence, never production sizes.

## Specialised paths: ask before you assume

```bash
python -m pyqcd.einsum.verify_specialized "cadb,menf,ac,em,bd,fn->"
```

Strongly recommended before chasing a specialised route. It is a **pure query**:
no operands are built, `contract()` is not called, no global state changes. It
matches shape specialisations → the 8,638-equation corpus → the equivalence layer,
and reports either `HIT` (with digest, corpus provenance and declared kernels) or
`MISS` (plus an equivalent spelling, with the operand and output permutations,
when one exists). Exit codes: `0` hit, `1` miss, `2` malformed equation. `--json`
gives the machine-readable form with language-independent `reason_key` /
`reason_args`.

A `HIT` means "the corpus knows this topology", **not** "a native kernel will
run": resource gates, dtype policy, contiguity and library availability can all
send it back to the generic path. For what actually ran, read the `route` /
`kernel` lines of `python -m pyqcd.einsum.test`.

## Version summary

* **Unreleased** — the configuration table is written **only** by `--tune`;
  `contract()` is read-only. One row per key, one timing column (`wall_ms`, the
  wall clock of the whole `contract()` call). `--kernel-force` takes a single
  logical value (`generic` / `dedicated` / `topology` / `backend`). Tune sweeps an
  ordered per-branch list with `min_wall_ms` pruning and writes just the winner.
* **0.5.15** — when `opt_einsum` is absent, the `original` kernel is now skipped at admission with a single `skip  kernel_force=original  组数=7` line naming the fix, instead of leaving seven `状态=error` rows that look like seven independent failures (it is the first group since 0.5.14).
* **0.5.14** — `--tune` now prints a `tune comb :` progress line for **every** group the moment it finishes (index/total, the three axes, the dispatched backend/dtype/device, `wall_ms`, peak, status, reason verbatim) instead of holding them until the sweep ends. `original` (`opt_einsum`) is now the **first** group, so the `min_wall_ms` bound is pinned to the equation's real speed from the start; the pruning cutoff went back to **`min_wall_ms` itself** (1x, not 2x). The "a lower-memory configuration exists" hint was removed along with the record-table read on the hit path.
* **0.5.13** — `kernel_force="original"` now raises a `KernelUnavailableError` that names the fix (`pip install "pyqcd[einsum]"`) instead of a bare `ModuleNotFoundError` when `opt_einsum` is absent, and `opt_einsum>=3.3` is declared as the new `einsum` extra.
* **0.5.12** — a new `original` kernel hands the contraction to `opt_einsum.contract` (it honours `--optimize`, translating `branch_1`/`branch_2`/`branch_all`/`none` to `branch-1`/`branch-2`/`branch-all`/`False` — the underscore spellings are a `KeyError` there); it takes part in `--tune` and sits before `backend`. `--tune` now records each group's **peak memory / VRAM**, measured per group with the operands' own resident size subtracted (they are input, not overhead), and writes a full sweep snapshot to `pyqcd.einsum.record.csv` (reference only, never a hit target; a hit prints a hint when a cheaper-memory configuration exists). Pruning cuts at **2 x** the best time, defaults are **warmup 0 / rounds 1**, the profile table lives in `~/.cache/pyqcd.einsum`, and the first call without a row runs a sweep automatically (`PYQCD_EINSUM_AUTOTUNE=0` opts out). `optimize_topology_v2` is gone for good, and the operands are staged on tmpfs with a space pre-check (a full `/tmp` used to turn all 29 groups into `error`).
* **0.5.11** — the **CUDA** side now exposes the same `bypass_plan_gate()` as the CPU side, so `--device gpu --tune` no longer kills its `dedicated` combination on `plan arena exceeds the measured CUDA dedicated budget`: `_api` looked the bypass up with `getattr(module, ..., None)`, and a missing name meant "silently not bypassed". `--tune-timeout`'s help now spells out that the window is per GROUP (`limit x (1+warmup+rounds)` = 210 s by default), so an ~80 s return does not mean the bound failed; every historical GPU timing line in `docs/` carries an inline `[作废]` mark.
* **0.5.10** — `--tune` no longer skips a single combination: the plan-only "predicted" prunes are gone, so every group is really measured and the only outcomes are `ok` / `timeout` (hard bound: the parent kills the child) / `error` / `pruned` by measured `min_wall_ms`. `--atol` omitted now means **automatic**: 1e-12 for 64-bit dtypes and 64x the dtype epsilon for single precision (7.6e-6), because a 10-operand complex64 contraction accumulates ~16 eps and a 1e-12 threshold failed every run; an explicit number is used verbatim. GPU timings recorded before 0.5.9 are void — see `docs/ERRATA_gpu_timing_20261119.md`.
* **0.5.9** — `--compare opt_einsum` now really uses the operands' backend (`backend=` is passed to `oe.contract` and printed in the reference line); every measurement synchronises the device before and after each sample, so async cupy/torch calls report compute time instead of enqueue time (the same equation measured ~900 ms in a script and 0.32 ms in the tune table); accelerated devices raise warmup to >= 2 and say so; `optimize_topology_v2` is gone from the `contract` signature (its only consumer, the topology v2 executor, was removed in 0.5.4) and now raises `TypeError`; `pyqcd.einsum.last_kernel` / `last_profile` are exported.
* **0.5.8** — `--tune` and `--kernel-force <kernel>` no longer stop at the kernel's **budget** gates: arena / single-GEMM / pack traffic are *routing* heuristics for automatic dispatch, not capability limits, so they are now bypassed (`_kernel_kupl.bypass_plan_gate()`) on those two paths and the exceeded budget is still recorded in `LAST_INFO['budget_gate']`. Admission keeps only environmental checks (native library, dtype, torch backend, unencodable plan), so every tune combination really runs and gets a measured `wall_ms` — `dedicated` used to be dropped wholesale (7 groups) on `pack/flops = 3.950`. Automatic dispatch is unchanged.
* **0.5.7** — every `--tune` subprocess group now runs with `profile=False`: the configuration table is neither read nor written during a sweep, so a hit row can no longer collide with the axes the parent dispatches per group (the 0.5.6 `ProfileConflict` regression, 9 groups).
* **0.5.6** — `--tune` timeouts are now a **hard bound**: each combination runs in a fresh interpreter and the parent kills the child, so native C/CUDA code really is interrupted (SIGALRM could not); process start-up is reported separately and never enters `wall_ms`. Two plan-only prunes stop pathological groups before they run (`optimize=none` on production shapes plans ~7.6e17 flops; `optimal` needs 38 s to plan). `PYQCD_EINSUM_TUNE_INPROCESS=1` restores the old best-effort route.
* **0.5.5** — `--tune pass` (fast pass: the `[if_equivalence=false, if_specialized=false]` line only, `--compare` and every unrelated feature disabled), `tune init :` start/done progress lines for the initialisation phase, and an honest `--tune-timeout` contract (best-effort: interrupts Python loops, cannot interrupt native C/CUDA); the performance pages lost their stale ratios and now only say "run --tune, measure your own shape".
* **0.5.4** — topology step tables unified on `ROLE_SHAPES`: `production` (anisotropic axes) and `uniform` (equal-length `dim=4` probe) are deleted, `sizes` is mandatory; `optimize` is exactly seven underscore leaves; both tables are generated with `auto`; `kupl_topology_v2`, `--backend kupl|cuda` and `--compare numpy` removed; `--tune` no longer hangs on the reference contraction and rewrites before forcing a kernel.
* **0.5.3** — equivalence routing table, single-language notices, tune hardening.
* **0.5.2** — `--tune` and the equivalence memo introduced.
* **0.5.1** — the profile cache stops recommending dead ends.
* **0.5.0** — profile cache, narrowed `kernel_force` logical names, explain mode.
* **0.4.x** — native specialised dispatch over the 8,638-equation corpus, with a
  hard fallback invariant: results never change, only the route does.

## Documentation and repositories

* Docs — https://py-qcd.readthedocs.io/en/latest/usage.html
* GitLab (primary) — https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD
* Gitee — https://gitee.com/zhangxin8069/PyQCD
* GitHub — https://github.com/zhangxin8069/PyQCD

## License

MIT
