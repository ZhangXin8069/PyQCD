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
