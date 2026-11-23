# Acknowledgements

PyQCD re-derives algorithms and workflows that were first published by others,
and it runs on open-source numerical and documentation software. We are grateful
to all the authors and maintainers listed below.

This page is intentionally the first page of the site: the reference code and
the reference Git repositories that informed PyQCD are acknowledged here, up
front, instead of being scattered through the body of the documentation.

## Reference code

The `refer/` tree of the source repository archives the contributor-organised
reference codes from which PyQCD's algorithms, conventions and renormalisation
logic were re-derived. PyQCD re-implements their logic; it does not import or
vendor their source into its runtime.

### `refer/donghx/` — Dong Hongxin

Proton two-point distillation and OPE operators.

### `refer/zhangxin/` — Zhang Xin

The complete gluon-PDF workflow and the data-analysis framework.

### `refer/huangcl/` — Huang Chengle

The multi-step Chroma pipeline: contract, ratio, energy.

### `refer/sush/` — Su Shihao

The distillation contraction framework and the `lqcddb` package.

### `refer/zengch/` — Zeng Chunhua

Renormalisation, the hybrid scheme and fitting logic — the origin of
`pyqcd/renorm`.

### `refer/papers/` and `refer/books/`

Literature and textbooks used for the physics derivation and validation.

## Reference Git repositories

The external repositories below are pinned as Git submodules under
`refer/git-rep/<name>` and kept read-only at the upstream commit. They served as
software references, comparison targets and architecture examples; none of them
is vendored into the package.

### PyQUDA

Python/Cython wrapper around QUDA for GPU/MPI lattice QCD.
https://github.com/CLQCD/PyQUDA

### LQCD_Skills

Agent skills that take a physical observable to a numerical result.
https://github.com/AI4LGT/LQCD_Skills

### lamet-agent

Python scaffolding for LaMET/LQCD analysis agents.
https://github.com/Greyyy-HJC/lamet-agent

### EasyDistillation

Multi-backend distillation: eigenvectors, perambulators and elemental
contraction.
https://github.com/IHEP-LQCD/EasyDistillation

### LQCD_Master

Natural language to PyQUDA programs and Slurm scripts.
https://github.com/sjtu-sai-agents/LQCD_Master

### PyQCU

torch, mpi4py and h5py architecture examples.
https://gitlab.itp.ac.cn/zhangxin8069/PyQCU

### opt_einsum

Contraction-path planning and cost diagnostics.
https://github.com/dgasmith/opt_einsum

### Einsums

C++20 tensor algebra, packed GEMM and deferred execution.
https://github.com/Einsums/Einsums

### MatX

CUDA C++17 tensor library and GPU `einsum` backends. Its 32-bit compute type is
recorded as an upstream limitation, not as a PyQCD choice.
https://github.com/NVIDIA/MatX

Fixed commits, licences and the per-project purpose table are recorded in
`refer/git-rep/README.md`. `PyQUDA` nests `pycparser` as its own submodule,
initialised with `git submodule update --init --recursive`.

## Open-source software

These packages are not pinned as submodules; they are ordinary dependencies of
the build, the tests and this documentation.

### NumPy

Array semantics and the reference behaviour every comparison is written against.

### SciPy

Scientific computing routines that complement NumPy.

### CuPy

GPU array backend used by the CUDA comparisons.

### OpenBLAS

CPU BLAS implementation used by the measured engines.

### LAPACK

CPU LAPACK routines used by the measured engines.

### Sphinx

Documentation generator behind both Read the Docs sites.

### MyST-Parser

Markdown parsing for the Sphinx sources.

### Read the Docs

Documentation hosting for the English and Chinese sites.

## With our thanks

Our thanks go to everyone who published the code, workflows and notes archived
under `refer/`, and to the authors and maintainers of every open-source project
listed above. Making software and documentation openly available is what let
this toolkit be re-derived, checked and corrected against independent
implementations; any remaining error is ours alone.

These are excellent software toolkits for lattice QCD.
