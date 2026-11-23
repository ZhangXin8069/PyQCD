# Acknowledgements

PyQCD re-derives algorithms and workflows that were first published by others,
and it runs on open-source numerical and documentation software. We are grateful
to all the authors and maintainers listed below.

This page is intentionally the last page of the site: the reference code and the
reference Git repositories that informed PyQCD are acknowledged here, at the
end, instead of being scattered through the body of the documentation.

## Open-source software

* NumPy, SciPy and opt_einsum — array semantics, reference contractions and
  contraction planning.
* CuPy — GPU array backend used by the CUDA comparisons.
* MatX (NVIDIA) — CUDA tensor library. Its `cutensor::einsum` was the subject of
  the (now removed) `cpp/cuda/matx` bridge; its 32-bit compute type is recorded
  as an upstream limitation, not as a PyQCD choice.
* Einsums — C++20 tensor algebra library. Its `compute_graph::einsum` was the
  subject of the (now removed) `cpp/cpu/cpu-einsums` bridge.
* OpenBLAS and LAPACK — CPU BLAS/LAPACK implementations used by the measured
  engines.
* Sphinx, MyST-Parser and Read the Docs — documentation build and hosting.

## Reference code

The `refer/` tree of the source repository archives the contributor-organised
reference codes from which PyQCD's algorithms, conventions and renormalisation
logic were re-derived. PyQCD re-implements their logic; it does not import or
vendor their source into its runtime.

| Directory | Contributor | What it informed |
|---|---|---|
| `refer/donghx/` | Dong HX | proton two-point distillation and OPE operators |
| `refer/zhangxin/` | Zhang X | the complete gluon-PDF workflow and the data-analysis framework |
| `refer/huangcl/` | Huang CL | the multi-step Chroma pipeline (contract, ratio, energy) |
| `refer/sush/` | Su SH | the distillation contraction framework and the `lqcddb` package |
| `refer/zengch/` | Zeng C | renormalisation, the hybrid scheme and fitting logic (the origin of `pyqcd/renorm`) |
| `refer/papers/`, `refer/books/` | — | literature and textbooks used for the physics derivation and validation |

## Reference Git repositories

The external repositories below are pinned as Git submodules under
`refer/git-rep/<name>` and kept read-only at the upstream commit. They served as
software references, comparison targets and architecture examples; none of them
is vendored into the package.

| Project | Source | What it informed |
|---|---|---|
| PyQUDA | https://github.com/CLQCD/PyQUDA | Python/Cython wrapper around QUDA for GPU/MPI lattice QCD |
| LQCD_Skills | https://github.com/AI4LGT/LQCD_Skills | agent skills that take a physical observable to a numerical result |
| lamet-agent | https://github.com/Greyyy-HJC/lamet-agent | Python scaffolding for LaMET/LQCD analysis agents |
| EasyDistillation | https://github.com/IHEP-LQCD/EasyDistillation | multi-backend distillation: eigenvectors, perambulators, elemental contraction |
| LQCD_Master | https://github.com/sjtu-sai-agents/LQCD_Master | natural language to PyQUDA programs and Slurm scripts |
| PyQCU | https://gitlab.itp.ac.cn/zhangxin8069/PyQCU | torch, mpi4py and h5py architecture examples |
| opt_einsum | https://github.com/dgasmith/opt_einsum | contraction-path planning and cost diagnostics |
| Einsums | https://github.com/Einsums/Einsums | C++20 tensor algebra, packed GEMM and deferred execution |
| MatX | https://github.com/NVIDIA/MatX | CUDA C++17 tensor library and GPU `einsum` backends |

`PyQUDA` nests `pycparser` as its own submodule, initialised with
`git submodule update --init --recursive`. Fixed commits, licences and the
per-project purpose table are recorded in `refer/git-rep/README.md`.

## With our thanks

Our thanks go to everyone who published the code, workflows and notes archived
under `refer/`, and to the authors and maintainers of every open-source project
listed above. Making software and documentation openly available is what let
this toolkit be re-derived, checked and corrected against independent
implementations; any remaining error is ours alone.

These are excellent software toolkits for lattice QCD.
